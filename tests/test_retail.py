import tempfile, unittest, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fame_scm.settings import load_settings
from fame_scm.fingerprints.features import temporal_features, Forecastability
from fame_scm.datasets.retail import prepare_m5, prepare_favorita, load_retail_panel
from fame_scm.competence.scaling import FrozenRowScaling
from fame_scm.competence.trainers import SCMEnsemble
from fame_scm.evaluation.statistics import paired_bootstrap
from retail_fixtures import make_retail_fixtures


def fixture_config(path, dates):
    c = load_settings()
    c["data"].update(format="retail_panel", path=str(path), max_series=None)
    c["forecast"].update(
        horizon=28, lookback=49, origin_stride=28, max_origins_per_stage=None
    )
    c["split"]["cutoffs"] = [str(dates[x].date()) for x in [489, 524, 559]]
    return c


class RetailContracts(unittest.TestCase):
    def test_lifecycle_uses_birth_not_global_calendar(self):
        y = np.ones(100)
        valid = np.arange(100) >= 80
        self.assertEqual(temporal_features(y, valid)[0], 20)
        self.assertEqual(temporal_features(y, valid)[1], 20)

    def test_scaling_rejects_accidental_broadcast(self):
        s = FrozenRowScaling(np.ones(2), np.ones(2))
        with self.assertRaises(ValueError):
            s.restore(np.ones((1, 3)))

    def test_new_rows_have_zero_offset_explicit_scaling_and_correct_shape(self):
        c = load_settings()
        c["scm"].update(epochs=1, members=2)
        torch.set_num_threads(1)
        z = np.zeros((2, 3), np.float32)
        loss = np.array([[1.0, 2.0], [2.0, 1.0]])
        s = FrozenRowScaling(np.ones(2), np.ones(2))
        est = SCMEnsemble(c, 42).fit(z, loss, np.ones((2, 2), bool), s)
        for n in [1, 3]:
            test = np.ones((n, 3), np.float32)
            with self.assertRaises(ValueError):
                est.predict(test)
            mu, sd = est.predict(test, scaling=FrozenRowScaling(np.ones(n), np.ones(n)))
            self.assertEqual(mu.shape, (n, 2))
            self.assertTrue(np.isfinite(sd).all())
        self.assertEqual(est.predict(z[:1], row_ids=[1])[0].shape, (1, 2))

    def test_m5_price_lifecycle_and_future_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            m5, _, dates = make_retail_fixtures(d, n=4)
            prices = pd.read_csv(m5 / "sell_prices.csv")
            prices = prices[~((prices.item_id == "item_0") & (prices.wm_yr_wk < 10))]
            prices.to_csv(m5 / "sell_prices.csv", index=False)
            out = Path(d) / "prepared"
            prepare_m5(m5, out, train_end=str(dates[489].date()), max_series=4)
            c = fixture_config(out, dates)
            p = load_retail_panel(c)
            i = p.ids.index('["CA_1","item_0"]')
            self.assertFalse(p.valid[i, 69])
            self.assertTrue(p.valid[i, 70])
            f = Forecastability(c).fit(p)
            raw = f.raw(p, i, 489)
            self.assertEqual(raw[0], 420)
            self.assertEqual(len(p.origins["test"]), 5)
            # Numeric parse errors retain a missing flag, not a legal-zero flag.
            p.frames[i].numeric["sell_price"] = np.full(700, np.nan)
            bad = f.raw(p, i, 489)
            self.assertEqual(bad[27], 1.0)
            with self.assertRaisesRegex(ValueError, "cutoff"):
                c["split"]["cutoffs"][0] = str(dates[488].date())
                load_retail_panel(c)

    def test_favorita_chunking_returns_zeros_holidays_and_training_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            _, fav, dates = make_retail_fixtures(d, n=4)
            out = Path(d) / "prepared"
            meta = prepare_favorita(
                fav,
                out,
                train_end=str(dates[489].date()),
                start=str(dates[0].date()),
                end=str(dates[-1].date()),
                chunksize=103,
            )
            self.assertEqual(meta["negative_rows_in_date_range"], 1)
            p = load_retail_panel(fixture_config(out, dates))
            i = p.ids.index('["1","0"]')
            self.assertEqual(p.y[i, 2], 0)
            self.assertEqual(p.y[i, 11], 0)
            self.assertTrue(p.valid[i, 11])
            self.assertEqual(p.frames[i].iloc[10]["holiday"], 1)
            self.assertEqual(p.frames[i].iloc[12]["holiday"], 0)
            self.assertEqual(p.frames[i].iloc[13]["holiday"], 0)
            self.assertTrue(np.isnan(p.frames[i].iloc[11]["onpromotion"]))
            # Test-only rows cannot enter training cohort.
            extra = pd.read_csv(fav / "train.csv")
            late = extra[extra.date > str(dates[489].date())].copy()
            late["item_nbr"] = 999
            pd.concat([extra, late]).to_csv(fav / "train.csv", index=False)
            other = prepare_favorita(
                fav,
                Path(d) / "other",
                train_end=str(dates[489].date()),
                start=str(dates[0].date()),
                end=str(dates[-1].date()),
                chunksize=257,
            )
            self.assertEqual(meta["ids"], other["ids"])

    def test_duplicate_favorita_across_chunks_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, fav, dates = make_retail_fixtures(d, n=2)
            x = pd.read_csv(fav / "train.csv")
            pd.concat([x, x.iloc[:1]]).to_csv(fav / "train.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prepare_favorita(
                    fav,
                    Path(d) / "p",
                    train_end=str(dates[489].date()),
                    start=str(dates[0].date()),
                    end=str(dates[-1].date()),
                    chunksize=71,
                )

    def test_constrained_column_updates_preserve_old_parameters(self):
        from fame_scm.competence.insertion import append_competence, append_router
        from fame_scm.competence.torch_models import SparseRouter

        c = load_settings()
        c["scm"].update(epochs=2, members=2)
        c["router"]["epochs"] = 2
        torch.set_num_threads(1)
        z = np.random.default_rng(4).normal(size=(4, 3)).astype(np.float32)
        losses = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 3.0], [3.0, 2.0]])
        scaling = FrozenRowScaling(np.ones(4), np.ones(4))
        est = SCMEnsemble(c, 42).fit(z, losses, np.ones_like(losses, bool), scaling)
        before = est.predict(z)[0]
        enlarged = np.c_[losses, [0.5, np.nan, 1.0, np.nan]]
        observed = np.isfinite(enlarged)
        inserted = append_competence(est, z, enlarged, observed, scaling)
        np.testing.assert_array_equal(inserted.predict(z)[0][:, :2], before)
        old = SparseRouter(3, 2)
        cache = np.ones((4, 3, 2), np.float32)
        cache[~observed] = np.nan
        router = append_router(
            old,
            c,
            z,
            np.full((4, 3), 1 / 3, np.float32),
            cache,
            observed,
            np.ones((4, 2), np.float32),
            np.ones((4, 2), bool),
            np.ones(3),
        )
        self.assertEqual(len(router.heads), 3)
        self.assertEqual(len(old.heads), 2)

    def test_frozen_cost_profile_is_reused_and_signature_checked(self):
        from fame_scm.evaluation.costs import profile_costs
        from types import SimpleNamespace
        from unittest.mock import Mock

        c = load_settings()
        c["cost_profiling"] = {"enabled": True, "series": 1, "repeats": 1}
        panel = SimpleNamespace(
            ids=["a", "b"],
            audit={"sha256": "fixed"},
            stages={"train": SimpleNamespace(end=100)},
            origins={"mining": [100, 114]},
        )
        pool = SimpleNamespace(
            names=c["experts"]["names"],
            lookback=49,
            h=14,
            device=torch.device("cpu"),
            availability=lambda origins: np.ones((2, 5), bool),
            predict=Mock(return_value=np.zeros(14)),
        )
        with tempfile.TemporaryDirectory() as d:
            c["cost_profiling"]["profile_path"] = str(Path(d) / "cost.json")
            first = profile_costs(pool, panel, c)
            calls = pool.predict.call_count
            second = profile_costs(pool, panel, c)
            self.assertEqual(calls, pool.predict.call_count)
            self.assertEqual(first["online"], second["online"])
            self.assertEqual(second["seconds"], 0.0)
            panel.audit["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "Frozen cost"):
                profile_costs(pool, panel, c)

    def test_identity_holdout_is_group_disjoint(self):
        from fame_scm.orchestration.transfer import identity_split

        ids = ['["a","x"]', '["b","x"]', '["a","y"]', '["b","y"]']
        train, test = identity_split(ids, 1, 0.5, 42)
        self.assertFalse(
            {json.loads(ids[i])[1] for i in train}
            & {json.loads(ids[i])[1] for i in test}
        )

    def test_paired_bootstrap_recomputes_pooled_metric(self):
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for i in range(4):
                for t in range(56):
                    rows.append(
                        {
                            "entity_id": str(i),
                            "origin": "2020-01-01",
                            "horizon_step": t + 1,
                            "target_date": str(
                                (
                                    pd.Timestamp("2020-01-02") + pd.Timedelta(days=t)
                                ).date()
                            ),
                            "target": 2.0,
                            "valid": True,
                            "weight": 1.0,
                        }
                    )
            for method, pred in [("a", 3.0), ("b", 4.0)]:
                dest = Path(d) / method / "mask_0" / "seed_0"
                dest.mkdir(parents=True)
                x = pd.DataFrame(rows)
                x["prediction"] = pred
                x.to_csv(dest / "predictions.csv", index=False)
            result = paired_bootstrap(d, "a", "b", replicates=20)
            self.assertEqual(result["delta_method_minus_reference"], -0.5)
            self.assertEqual(result["paired_percentile_95ci_delta"], [-0.5, -0.5])
            x.loc[0, "target"] = 99
            x.to_csv(dest / "predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Nonpaired"):
                paired_bootstrap(d, "a", "b", replicates=20)
