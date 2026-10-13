"""Current-manuscript regression tests and evidence isolation contracts."""

import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from fame_scm.acquisition.evidence import MiningStore
from fame_scm.contracts import RequestKey, ForecastEvidence
from fame_scm.baselines.recovery import expected_improvement
from fame_scm.competence.calibration import estimate_rank_margin, count_rank_pairs
from fame_scm.competence.scaling import FrozenRowScaling
from fame_scm.competence.torch_models import CompetenceModel
from fame_scm.competence.torch_objectives import competence_objective
from fame_scm.evaluation.theory import stability_diagnostics
from fame_scm.evaluation.tables import export_tables
from fame_scm.routing.consumers import sparsemax, operator_logits
from fame_scm.orchestration.suite import make_plan
from fame_scm.settings import load_settings
from fame_scm.reproduction import validate_protocol, protocol_snapshot
from fame_scm.evaluation.resources import ResourceMonitor
from fame_scm.checkpoints import state_training_seed
import psutil


def store_with_cap(cap=None):
    def provider(key):
        return ForecastEvidence(
            key, float(int(key.origin) + (key.expert_id == "b")), np.array([1.0]), 0.1
        )

    return MiningStore(
        ["x", "y"],
        ["a", "b"],
        np.ones((2, 2), bool),
        4,
        provider,
        {"a": "1", "b": "2"},
        "protocol",
        construction_request_budget=cap,
    )


class CurrentManuscriptContracts(unittest.TestCase):
    def test_saved_consumer_seed_preserves_replicate(self):
        cfg = load_settings()
        self.assertEqual(
            state_training_seed({"cfg": cfg, "evidence_mask_id": 3, "neural_seed": 2}),
            cfg["seed"] + 30002,
        )
        cfg["evaluation"]["mask_seeds"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "lacks mask/neural IDs"):
            state_training_seed({"cfg": cfg})

    def test_resource_unavailable_is_reported_not_zero(self):
        with patch(
            "fame_scm.evaluation.resources.psutil.Process",
            side_effect=psutil.NoSuchProcess(2),
        ):
            monitor = ResourceMonitor().start()
            result = monitor.finish()
            self.assertIsNone(result["peak_process_tree_rss_bytes"])
            self.assertIn("pid=2", result["memory_sampling_error"].lower())
            self.assertIs(monitor.finish(), result)

    def test_paper_claim_requires_author_protocol(self):
        cfg = load_settings()
        cfg["result_kind"] = "paper_reproduction"
        panel = SimpleNamespace(
            audit={"sha256": "abc"},
            ids=["a"],
            origins={"mining": [1]},
            stages={"train": SimpleNamespace(start=0, end=1)},
        )
        with self.assertRaisesRegex(ValueError, "author_protocol"):
            validate_protocol(cfg, panel)

    def test_scale_plans_do_not_reuse_wrong_cohort_cost_profiles(self):
        for job in make_plan(load_settings(), "scale")["jobs"]:
            self.assertIsNone(job["config"]["cost_profiling"]["profile_path"])
            self.assertEqual(
                job["config"]["data"]["required_series_count"],
                job["config"]["data"]["max_series"],
            )

    def test_protocol_matching_binds_selector_settings(self):
        cfg = load_settings()
        panel = SimpleNamespace(
            audit={"sha256": "abc"},
            ids=["a"],
            origins={"mining": [1]},
            stages={"train": SimpleNamespace(start=0, end=1)},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol_snapshot(cfg, panel)))
            cfg["reproduction"] = {"strict": True, "author_protocol": str(path)}
            self.assertEqual(
                validate_protocol(cfg, panel)["status"], "author_protocol_matched"
            )
            cfg["teacher"]["tau"] = 0.9
            with self.assertRaisesRegex(ValueError, "selector_protocol_hash"):
                validate_protocol(cfg, panel)

    def test_total_budget_includes_calibration_and_cached_keys(self):
        store = store_with_cap(2)
        store.acquire("x", "a", ["1"])
        key = RequestKey("calibration", "x", "a", "5", "1", "protocol")
        store.fetch(key)
        store.fetch(key)
        with self.assertRaisesRegex(ValueError, "Total construction"):
            store.fetch(RequestKey("calibration", "y", "a", "5", "1", "protocol"))
        account = store.evidence_account()
        self.assertEqual(account["construction_unique_origin_requests"], 2)
        self.assertEqual(account["stages"]["calibration"]["cache_hits"], 1)

    def test_test_audit_do_not_consume_construction_cap(self):
        store = store_with_cap(0)
        store.fetch(RequestKey("test", "x", "a", "5", "1", "protocol"))
        store.fetch(RequestKey("research_audit", "x", "a", "1", "1", "protocol"))
        self.assertEqual(
            store.evidence_account()["construction_unique_origin_requests"], 0
        )

    def test_zero_uncertainty_ei_cost_units(self):
        pred = np.array([[2.0, 1.0, 0.0]])
        observed = np.array([[True, False, False]])
        loss = np.array([[3.0, np.nan, np.nan]])
        scores = expected_improvement(
            pred,
            np.zeros_like(pred),
            loss,
            observed,
            np.ones_like(observed),
            np.array([1, 2, 6]),
        )
        self.assertTrue(np.isneginf(scores[0, 0]))
        self.assertAlmostEqual(scores[0, 1], 1)
        self.assertAlmostEqual(scores[0, 2], 0.5)

    def test_ei_rejects_zero_cost(self):
        with self.assertRaises(ValueError):
            expected_improvement(
                np.zeros((1, 2)),
                np.ones((1, 2)),
                np.array([[1, np.nan]]),
                np.array([[True, False]]),
                np.ones((1, 2), bool),
                [1, 0],
            )

    def test_margin_uses_acquired_cache_only(self):
        store = store_with_cap()
        store.acquire("x", "a", ["1", "2"])
        store.acquire("x", "b", ["1", "2"])
        value, record = estimate_rank_margin(
            store, FrozenRowScaling(np.zeros(2), np.ones(2))
        )
        self.assertEqual(value, 0)
        self.assertEqual(record["pair_histories"], 1)
        self.assertEqual(len(store.ledger.events), 4)

    def test_margin_single_origin_fallback(self):
        store = store_with_cap()
        store.acquire("x", "a", ["1"])
        value, record = estimate_rank_margin(
            store, FrozenRowScaling(np.zeros(2), np.ones(2))
        )
        self.assertEqual(value, 0.05)
        self.assertTrue(record["fallback_used"])

    def test_rank_pairs_chunked_and_hidden_independent(self):
        labels = np.array([[1, 3, np.nan], [5, np.nan, 2]])
        mask = np.isfinite(labels)
        self.assertEqual(count_rank_pairs(labels, mask, 0, chunk_size=1), 2)
        labels[~mask] = 1e20
        self.assertEqual(count_rank_pairs(labels, mask, 0), 2)

    def test_mse_reconstruction_ablation(self):
        model = CompetenceModel(1, 1, 2)
        prediction = torch.tensor([[2.0, 1.0]], requires_grad=True)
        labels = torch.tensor([[0.0, 0.0]])
        loss, _ = competence_objective(
            model,
            prediction,
            labels,
            torch.ones((1, 2), dtype=torch.bool),
            rank_margin=0,
            lambda_factor=0,
            lambda_encoder=0,
            lambda_rank=0,
            reconstruction="mse",
        )
        self.assertAlmostEqual(float(loss), 2.5)

    def test_sparsemax_probability_and_gradient(self):
        logits = torch.tensor([[3.0, 1.0, 0.0], [0.2, 0.1, -0.2]], requires_grad=True)
        p = sparsemax(logits)
        torch.testing.assert_close(p.sum(1), torch.ones(2))
        self.assertEqual(float(p[0, 2]), 0)
        p.square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_operators_have_at_most_r_nontrivial_weights(self):
        for operator in ["straight_through", "gumbel_top_r"]:
            scores = operator_logits(
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]), operator, r=2
            )
            p = scores.softmax(1)
            self.assertEqual(int((p > 1e-6).sum()), 2)

    def test_theory_bound_and_tail_are_audit_only(self):
        loss = np.array([[1.0, 4.0], [2.0, 3.0]])
        p = np.array([[0.8, 0.2], [0.1, 0.9]])
        rows, summary = stability_diagnostics(
            loss,
            loss.copy(),
            p,
            p,
            np.ones((2, 2), bool),
            np.array([[True, False], [False, True]]),
            tau=0.3,
            bound=2,
        )
        self.assertTrue(
            all(r["component_loss_deviation"] <= r["upper_bound"] for r in rows)
        )
        self.assertEqual(summary["bound_B"], 2)
        self.assertGreater(rows[1]["raw_tail_selected"], 0)

    def test_theory_partial_rows_excluded(self):
        rows, summary = stability_diagnostics(
            np.array([[1, np.nan]]),
            np.array([[1, 2]]),
            np.array([[1, 0]]),
            np.array([[1, 0]]),
            np.array([[True, False]]),
            np.array([[True, False]]),
            tau=1,
        )
        self.assertEqual(rows, [])
        self.assertEqual(summary["excluded_partial_rows"], 1)

    def test_export_reads_measured_results_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "run").mkdir()
            (root / "paper_targets.csv").write_text("rmse\n0.0001\n")
            (root / "run" / "summary.json").write_text(
                json.dumps(
                    {"method": "actual", "metrics": {"rmse": 7.0}, "actual_rho": 0.4}
                )
            )
            export_tables(root, root / "export")
            self.assertIn("7.0", (root / "export" / "measured_runs.csv").read_text())
            self.assertNotIn(
                "0.0001", (root / "export" / "measured_runs.csv").read_text()
            )

    def test_new_suite_profiles_and_ablation_complete(self):
        cfg = load_settings()
        cfg["output"] = "runs/test"
        self.assertEqual(len(make_plan(cfg, "scale")["jobs"]), 5)
        self.assertEqual(len(make_plan(cfg, "pools")["jobs"]), 5)
        names = {j["name"] for j in make_plan(cfg, "ablations")["jobs"]}
        self.assertTrue({"no_huber", "no_row_normalization"} <= names)

    def test_total_budget_plan_requires_explicit_limits(self):
        with self.assertRaisesRegex(ValueError, "comparison_requests"):
            make_plan(load_settings(), "total_budget")


if __name__ == "__main__":
    unittest.main()
