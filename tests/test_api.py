"""Source-checkout API behavior and bilingual executable documentation."""

from pathlib import Path
import unittest
from unittest.mock import patch

from fame_scm.api import PROJECT_ROOT, load_config, train


class SourceAPIContracts(unittest.TestCase):
    def test_default_config_paths_are_independent_of_callers_cwd(self):
        cfg = load_config()
        self.assertTrue(Path(cfg["data"]["path"]).is_absolute())
        self.assertTrue(Path(cfg["output"]).is_absolute())
        self.assertTrue((PROJECT_ROOT / "fame_scm/api.py").is_file())

    def test_training_preserves_callers_configuration(self):
        cfg = load_config()
        original = cfg["output"]
        with patch(
            "fame_scm.orchestration.experiment.run_experiment", return_value="measured"
        ) as run:
            self.assertEqual(
                train(cfg, out="runs/api_contract", methods=["lightgbm"]), "measured"
            )
            called = run.call_args.args[0]
            self.assertNotEqual(called["output"], original)
            self.assertEqual(cfg["output"], original)
            self.assertEqual(run.call_args.kwargs["methods"], ["lightgbm"])


if __name__ == "__main__":
    unittest.main()
