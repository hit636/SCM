import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from fame_scm.settings import load_settings
from fame_scm.datasets.wide import load_wide_panel
from fame_scm.experts.forecast_experts import ExpertPool, tree_rows
from fame_scm.datasets.temporal import timestamp_text

ROOT = Path(__file__).resolve().parents[1]


class WideContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "wide.csv"
        self.dates = pd.date_range("2024-01-01", periods=300, freq="h")
        self.frame = pd.DataFrame(
            {
                "date": self.dates,
                "a": np.arange(300, dtype=float),
                "b": np.sin(np.arange(300) / 3),
            }
        )
        self.frame.to_csv(self.path, index=False)
        self.cfg = load_settings(ROOT / "configs/experiments/electricity_smoke.yaml")
        self.cfg["data"].update(
            path=str(self.path), min_observed_train_steps=24, max_series=None
        )
        self.cfg["forecast"].update(lookback=24, horizon=12, origin_stride=12)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hourly_grid_splits_and_training_only_scaling(self):
        p = load_wide_panel(self.cfg)
        self.assertEqual(len(p.dates), 300)
        self.assertEqual(p.dates[1] - p.dates[0], pd.Timedelta(hours=1))
        self.assertAlmostEqual(p.y[0, :180].mean(), 0)
        self.assertAlmostEqual(p.y[0, :180].std(), 1)
        for name, origins in p.origins.items():
            for o in origins:
                self.assertGreaterEqual(o + 1, p.stages[name].start)
                self.assertLessEqual(o + 12, p.stages[name].end)
        scaler = p.audit["scalers"]
        self.frame.loc[180:, "a"] += 1e6
        self.frame.to_csv(self.path, index=False)
        self.assertEqual(scaler, load_wide_panel(self.cfg).audit["scalers"])

    def test_duplicate_hour_rejected(self):
        pd.concat([self.frame, self.frame.iloc[:1]]).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate timestamps"):
            load_wide_panel(self.cfg)

    def test_gap_is_masked_not_zero_observation(self):
        self.frame.drop(10).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "missing time steps"):
            load_wide_panel(self.cfg)
        self.cfg["data"]["missing_timestamps"] = "reindex"
        p = load_wide_panel(self.cfg)
        self.assertEqual(len(p.dates), 300)
        self.assertFalse(p.valid[0, 10])
        self.assertTrue(p.valid[0, 11])

    def test_inference_uses_frozen_scaler(self):
        p = load_wide_panel(self.cfg)
        self.frame.iloc[-48:].to_csv(self.path, index=False)
        q = load_wide_panel(self.cfg, inference=True, scaling=p.audit["scalers"])
        np.testing.assert_allclose(p.y[:, -48:], q.y)
        with self.assertRaisesRegex(ValueError, "frozen training scaler"):
            load_wide_panel(self.cfg, inference=True)

    def test_subhourly_and_misaligned_frequency(self):
        self.frame["date"] = pd.date_range("2024-01-01", periods=300, freq="15min")
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "aligned"):
            load_wide_panel(self.cfg)
        self.cfg["data"]["frequency"] = "15min"
        p = load_wide_panel(self.cfg)
        self.assertEqual(p.dates[1].minute, 15)

    def test_negative_predictions_and_daily_compatibility(self):
        p = load_wide_panel(self.cfg)

        class Negative:
            def predict(self, x):
                return np.full(len(x), -2.0)

        pool = ExpertPool(self.cfg, p)
        pool.models["linear"] = Negative()
        np.testing.assert_array_equal(pool.predict("linear", 0, 100), -np.ones(12) * 2)
        pool.cfg["forecast"]["clip_nonnegative"] = True
        np.testing.assert_array_equal(pool.predict("linear", 0, 100), np.zeros(12))
        a = tree_rows(np.ones(24), pd.Timestamp("2024-01-01 22:00"), 3, "h")
        self.assertNotEqual(a[0, -1], a[1, -1])
        daily = load_settings()
        self.assertEqual(timestamp_text(p.dates[1], daily), "2024-01-01")
        self.assertIn("T01:00:00", timestamp_text(p.dates[1], self.cfg))

    def test_no_future_based_cohort_selection(self):
        self.frame.loc[180:195, "a"] = np.nan
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "observed targets"):
            load_wide_panel(self.cfg)

    def test_headers_and_non_numeric_values(self):
        self.path.write_text("date,a,a\n2024-01-01,1,2\n2024-01-02,2,3\n")
        with self.assertRaisesRegex(ValueError, "unique header"):
            load_wide_panel(self.cfg)

    def test_all_supplied_configs_resolve(self):
        for dataset in ["snbc", "electricity", "traffic"]:
            for profile in ["smoke", "main"]:
                cfg = load_settings(
                    ROOT / f"configs/experiments/{dataset}_{profile}.yaml"
                )
                self.assertIn(dataset, cfg["data"]["path"])
                if profile == "main":
                    self.assertEqual(len(cfg["evaluation"]["mask_seeds"]), 5)
                    self.assertEqual(cfg["experts"]["neural_epochs"], 50)

    def test_fingerprints_describe_raw_signal_under_standardization(self):
        from fame_scm.fingerprints.features import Forecastability
        from copy import deepcopy

        standardized = load_wide_panel(self.cfg)
        rawcfg = deepcopy(self.cfg)
        rawcfg["data"]["standardization"] = "none"
        raw = load_wide_panel(rawcfg)
        a = Forecastability(self.cfg).raw(standardized, 0, 179)
        b = Forecastability(rawcfg).raw(raw, 0, 179)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
