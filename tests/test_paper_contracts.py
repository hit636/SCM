"""Regression tests for paper-aligned feature and evidence contracts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fame_scm.datasets.snbc import make_demo, load_panel
from fame_scm.fingerprints.features import Forecastability, temporal_features
from fame_scm.settings import load_settings


class PaperContracts(unittest.TestCase):
    def test_spectral_entropy_uses_full_fft_including_dc(self):
        vector = temporal_features(np.full(28, 4.0), np.ones(28, bool))
        self.assertAlmostEqual(vector[19], 0.0, places=6)
        self.assertAlmostEqual(vector[20], 1.0, places=6)
        self.assertAlmostEqual(vector[21] + vector[22], 0.0, places=6)

    def test_training_fill_remains_frozen_after_future_values_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            make_demo(path, series=4, days=420)
            cfg = load_settings()
            cfg["data"]["path"] = str(path)
            panel = load_panel(cfg)
            features = Forecastability(cfg).fit(panel)
            fill = dict(features.training_fill)
            panel.y[:, panel.stages["train"].end + 1 :] = 1e6
            features.raw(panel, 0, panel.origins["test"][0])
            self.assertEqual(features.training_fill, fill)
            self.assertEqual(features.global_fill, float(np.mean(list(fill.values()))))
            self.assertEqual(features.state()["training_fill"], fill)

    def test_explicit_fill_changes_only_grid_based_descriptors(self):
        y = np.arange(56, dtype=float)
        valid = np.ones(56, bool)
        valid[20:30] = False
        a = temporal_features(y, valid, fill_value=0.0)
        b = temporal_features(y, valid, fill_value=100.0)
        np.testing.assert_array_equal(a[:14], b[:14])
        self.assertNotEqual(a[19], b[19])

    def test_invalid_training_limits_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("scm:\n  epochs: 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scm.epochs"):
                load_settings(config)


if __name__ == "__main__":
    unittest.main()
