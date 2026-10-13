"""Device failure boundaries and placement, including optional real CUDA checks."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from fame_scm.devices import resolve_device, device_report
from fame_scm.settings import load_settings
from fame_scm.experts.forecast_experts import ExpertPool
from fame_scm.experts.extended_experts import ExtendedExpert


class DeviceContracts(unittest.TestCase):
    def test_cuda_unavailable_fails_before_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "No CPU fallback"):
                resolve_device("cuda:0")

    def test_invalid_and_invisible_devices_fail(self):
        for value in ["cuda:-1", "cuda:abc", "mps", "auto"]:
            with self.assertRaises(ValueError):
                resolve_device(value)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            self.assertEqual(str(resolve_device("cuda:1")), "cuda:1")
            with self.assertRaisesRegex(ValueError, "not visible"):
                resolve_device("cuda:2")

    def test_cpu_probe_runs_matrix_multiplication(self):
        result = device_report("cpu", probe=True)
        self.assertEqual(result["selected"], "cpu")
        self.assertEqual(result["probe"], "passed")

    def test_main_configs_require_cuda_smoke_configs_allow_cpu(self):
        root = Path(__file__).resolve().parents[1] / "configs/experiments"
        for dataset in ["snbc", "m5", "favorita", "electricity", "traffic"]:
            self.assertEqual(
                load_settings(root / f"{dataset}_main.yaml")["device"], "cuda:0"
            )
            self.assertEqual(
                load_settings(root / f"{dataset}_smoke.yaml")["device"], "cpu"
            )

    def test_nested_expert_move_preserves_portable_deployment(self):
        cfg = load_settings()
        cfg["device"] = "cuda:1"
        nested = ExpertPool.__new__(ExpertPool)
        nested.cfg = cfg
        nested.models = {"dlinear": torch.nn.Linear(3, 2)}
        wrapper = ExtendedExpert("dlinear__variant1", cfg)
        wrapper.kind = "dlinear"
        wrapper.model = nested
        wrapper.cpu()
        self.assertEqual(str(nested.device), "cpu")
        self.assertEqual(str(next(nested.models["dlinear"].parameters()).device), "cpu")
        self.assertEqual(cfg["device"], "cuda:1")

    def test_neuralforecast_cpu_move_updates_prediction_trainer(self):
        wrapper = ExtendedExpert("nhits", load_settings())
        model = torch.nn.Linear(3, 2)
        model.trainer_kwargs = {"accelerator": "gpu", "devices": [1]}
        wrapper.model = SimpleNamespace(models=[model])
        wrapper.cpu()
        self.assertEqual(model.trainer_kwargs, {"accelerator": "cpu", "devices": 1})

    def test_neuralforecast_preserves_selected_visible_gpu_index(self):
        from unittest.mock import Mock

        wrapper = ExtendedExpert("nhits", load_settings())
        model = Mock(trainer_kwargs={"accelerator": "cpu", "devices": 1})
        wrapper.model = SimpleNamespace(models=[model])
        with patch(
            "fame_scm.devices.prepare_device", return_value=torch.device("cuda:1")
        ):
            wrapper.to("cuda:1")
        model.to.assert_called_once_with(torch.device("cuda:1"))
        self.assertEqual(model.trainer_kwargs, {"accelerator": "gpu", "devices": [1]})

    @unittest.skipUnless(torch.cuda.is_available(), "Real CUDA hardware unavailable")
    def test_cuda_recovery_and_network_backward(self):
        from fame_scm.competence.torch_models import SparseRouter
        from fame_scm.competence.scaling import FrozenRowScaling
        from fame_scm.baselines.recovery import recover_static, GPLatentRecovery

        device = resolve_device("cuda:0")
        model = SparseRouter(4, 3, dropout=0.0).to(device)
        x = torch.ones(5, 4, device=device)
        loss = model(x).square().mean()
        loss.backward()
        self.assertEqual(model.heads[0].weight.grad.device, device)
        cfg = load_settings()
        cfg["device"] = str(device)
        cfg["baselines"]["epochs"] = 2
        values = np.arange(15).reshape(5, 3).astype(float) + 1
        mask = np.ones_like(values, dtype=bool)
        scaling = FrozenRowScaling(np.zeros(5), np.ones(5))
        for method in ["imc", "pmf_a"]:
            if method == "imc":
                mean, _ = recover_static(
                    method, np.ones((5, 4)), values, mask, scaling, cfg, 42
                )
            else:
                mean, _ = (
                    GPLatentRecovery(cfg, 42)
                    .fit(np.ones((5, 4)), values, mask, scaling)
                    .predict(np.ones((5, 4)))
                )
            self.assertTrue(np.isfinite(mean).all())


if __name__ == "__main__":
    unittest.main()
