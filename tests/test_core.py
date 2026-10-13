import tempfile
import unittest
import numpy as np
from fame_scm.acquisition import partition_remainder, acquisition_scores, select_batch
from fame_scm.contracts import ForecastEvidence, RequestKey
from fame_scm.datasets.data import (
    StageInterval,
    admissible_origins,
    validate_stage_intervals,
    basic_sparse_features,
)
from fame_scm.acquisition.evidence import (
    MiningStore,
    AcquiredEvidenceView,
    weighted_work_ratio,
)
from fame_scm.evaluation.metrics import (
    forecast_metrics,
    summarize_nested_replicates,
    full_observation_gap,
)
from fame_scm.reporting import (
    check_comparable,
    check_pooled_metric_identity,
    COMPARISON_HASHES,
)
from fame_scm.routing import tkde_gate, fuse, hit_at_r
from fame_scm.competence.scaling import FrozenRowScaling
from fame_scm.routing.teacher import competence_teacher
from fame_scm.orchestration.smoke import run_smoke


class CoreInvariants(unittest.TestCase):
    def test_full_horizon_within_stage(self):
        self.assertEqual(
            admissible_origins(range(0, 30), StageInterval(10, 20), 5).tolist(),
            list(range(9, 16)),
        )

    def test_stage_overlap_rejected(self):
        with self.assertRaises(ValueError):
            validate_stage_intervals(
                {
                    "train": StageInterval(0, 10),
                    "mining": StageInterval(10, 20),
                    "calibration": StageInterval(21, 30),
                    "test": StageInterval(31, 40),
                }
            )

    def test_valid_zero_and_censor_distinction(self):
        x = basic_sparse_features([0, 4, 0, 999], [True, True, True, False])
        self.assertAlmostEqual(x["zero_ratio"], 2 / 3)
        self.assertEqual(x["adi"], 3)

    def test_seed_floor_and_four_batches(self):
        self.assertEqual(partition_remainder(40, 13), [7, 7, 7, 6])
        with self.assertRaises(ValueError):
            partition_remainder(12, 13)

    def test_fixed_scale_and_single_anchor(self):
        loss = np.array([[4.0, np.nan], [2.0, 6.0], [np.nan, np.nan]])
        scale = FrozenRowScaling.from_seed(
            loss,
            np.isfinite(loss),
            [8, 8, 8],
            [3, 3, 3],
            floor=0.01,
            mad_multiplier=1.0,
        )
        np.testing.assert_allclose(scale.center, [4, 4, 8])
        np.testing.assert_allclose(scale.scale, [3, 2, 3])
        np.testing.assert_allclose(
            scale.restore(scale.normalize(loss)), loss, equal_nan=True
        )
        self.assertFalse(scale.scale.flags.writeable)

    def test_budget_and_cache_stage_isolation(self):
        calls = []

        def provider(k):
            calls.append(k)
            return ForecastEvidence(k, 1.0, np.ones(2), 0.1)

        store = MiningStore(
            ["a"],
            ["x", "y"],
            np.ones((1, 2), bool),
            1,
            provider,
            {"x": "c", "y": "c"},
            "p",
        )
        store.acquire("a", "x", ["o"])
        store.acquire("a", "x", ["o"])
        with self.assertRaises(ValueError):
            store.acquire("a", "y", ["o"])
        loss, mask = AcquiredEvidenceView(store).snapshot()
        self.assertTrue(np.isnan(loss[0, 1]))
        self.assertEqual(mask.sum(), 1)
        key = RequestKey("calibration", "a", "x", "o", "c", "p")
        store.fetch(key)
        store.fetch(key)
        self.assertEqual(len(calls), 2)
        self.assertEqual(store.remaining, 0)
        self.assertEqual(sum(e["cache_hit"] for e in store.ledger.events), 1)

    def test_changed_origins_rejected(self):
        def provider(k):
            return ForecastEvidence(k, 1.0, np.ones(2), 0.1)

        store = MiningStore(["a"], ["x"], [[True]], 1, provider, {"x": "c"}, "p")
        store.acquire("a", "x", ["o"])
        with self.assertRaises(ValueError):
            store.acquire("a", "x", ["new"])

    def test_acquisition_never_reads_hidden_measured_values(self):
        measured = np.array([[2.0, np.nan, 123456.0]])
        obs = np.array([[True, False, False]])
        avail = np.array([[True, True, False]])
        scores = acquisition_scores(
            [[2, 1, 0]], [[0.1, 0.2, 0.3]], measured, obs, avail, [1, 1, 1]
        )
        self.assertEqual(select_batch(scores, 1), [(0, 1)])
        self.assertTrue(np.isneginf(scores[0, 0]))
        self.assertTrue(np.isneginf(scores[0, 2]))

    def test_observed_teacher_ignores_prediction_uncertainty(self):
        observed = np.array([[True, False]])
        a = competence_teacher(
            [[1, np.nan]], observed, [[900, 2]], [[900, 0.2]], [[True, True]], [1, 1]
        )
        b = competence_teacher(
            [[1, np.nan]], observed, [[0, 2]], [[0, 0.2]], [[True, True]], [1, 1]
        )
        np.testing.assert_allclose(a, b)

    def test_full_observation_teacher_independent_of_recovery(self):
        obs = np.ones((1, 2), bool)
        a = competence_teacher([[1, 2]], obs, [[9, 8]], [[5, 3]], obs, [1, 1])
        b = competence_teacher([[1, 2]], obs, [[0, 0]], [[0, 0]], obs, [1, 1])
        np.testing.assert_allclose(a, b)

    def test_gate_version_threshold_witness(self):
        p = np.array([[0.70, 0.04] + [0.0325] * 8])
        g = tkde_gate(p, np.ones_like(p, bool), r=2, threshold=0.05)
        self.assertEqual(g.top_r_before_pruning.sum(), 2)
        self.assertEqual(g.retained.sum(), 1)
        self.assertGreater(
            p[0, 1] / (p[0, 0] + p[0, 1]), 0.05
        )  # ICDM threshold would retain two

    def test_availability_precedes_topk_and_fallback(self):
        g = tkde_gate([[0.8, 0.15, 0.05]], [[False, True, True]], r=2, threshold=0.99)
        np.testing.assert_allclose(g.weights, [[0, 1, 0]])
        with self.assertRaises(ValueError):
            tkde_gate([[1, 0]], [[False, True]])

    def test_fusion_ignores_unexecuted_nan(self):
        out = fuse([[1, 0]], [[[2, 3], [np.nan, np.nan]]])
        np.testing.assert_allclose(out, [[2, 3]])

    def test_hit_computed_before_pruning(self):
        g = tkde_gate([[0.96, 0.04]], [[True, True]], threshold=0.05)
        self.assertEqual(
            hit_at_r(g.top_r_before_pruning, [[2, 1]], [[True, True]]), 1.0
        )
        self.assertFalse(g.retained[0, 1])

    def test_weighted_cost_not_cell_ratio(self):
        self.assertAlmostEqual(
            weighted_work_ratio([[True, False]], [[True, True]], [1, 9]), 0.1
        )

    def test_common_target_metric_identity(self):
        metrics = forecast_metrics([1, 3, 100], [2, 1, np.nan], [True, True, False])
        check_pooled_metric_identity(metrics)
        self.assertEqual(metrics["target_abs_mean"], 2.0)
        self.assertEqual(metrics["wape"], 0.75)

    def test_zero_wape_denominator_is_missing(self):
        self.assertIsNone(forecast_metrics([0, 0], [1, 0], [True, True])["wape"])

    def test_nested_sd_uses_mask_means(self):
        result = summarize_nested_replicates([[1, 3], [3, 5]])
        self.assertEqual(result["mask_means"], [2.0, 4.0])
        self.assertAlmostEqual(result["mask_sd"], np.sqrt(2))

    def test_pairing_requires_protocol_match(self):
        a = {k: "x" for k in COMPARISON_HASHES}
        check_comparable(a, a)
        b = dict(a)
        b["gate_version"] = "icdm"
        with self.assertRaises(ValueError):
            check_comparable(a, b)

    def test_gap_arithmetic(self):
        self.assertAlmostEqual(full_observation_gap(1.398, 1.384), 0.010115606936416)

    def test_synthetic_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_smoke(directory)
        self.assertFalse(result["is_paper_reproduction"])
        self.assertEqual(result["cell_budget"], result["acquired_cells"])
        self.assertEqual(result["rho"], 0.4)
        self.assertLessEqual(result["execution_count_mean"], 2.0)
        self.assertLess(result["normalized_roundtrip_max_error"], 1e-10)


if __name__ == "__main__":
    unittest.main()
