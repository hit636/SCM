"""Real CUDA smoke validation for experts, recovery, routing and deployment."""

from pathlib import Path
import argparse
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fame_scm.api import load_config, make_demo, train, audit, predict
from fame_scm.checkpoints import load_state
from fame_scm.devices import device_report
from fame_scm.reporting import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="runs/verify_gpu")
    args = parser.parse_args()
    hardware = device_report(args.device, probe=True)
    if not hardware["selected"].startswith("cuda:"):
        raise ValueError("GPU validation requires a real CUDA device")
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh --out directory")
    output.mkdir(parents=True, exist_ok=True)
    data = output / "synthetic_daily.csv"
    make_demo(data, series=12, days=420)
    cfg = load_config()
    cfg["device"] = hardware["selected"]
    cfg["scm"].update(epochs=2, members=2, batches=2)
    cfg["router"].update(epochs=2)
    cfg["baselines"].update(epochs=2)
    cfg["experts"].update(
        neural_epochs=1, tree_estimators=5, max_train_windows=36, sarima_maxiter=3
    )
    cfg.setdefault("research", {})["save_state"] = True
    cfg["result_kind"] = "synthetic_gpu_validation"
    methods = ["lightgbm", "full_observation", "imc", "pmf_a", "fame_scm_active"]
    results = train(cfg, data=data, out=output / "experiment", methods=methods)
    state = load_state(
        output / "experiment/fame_scm_active/mask_0/seed_0/build_state.pkl"
    )
    model_devices = {
        name: str(next(model.parameters()).device)
        for name, model in state["pool"].models.items()
        if isinstance(model, torch.nn.Module)
    }
    model_devices["router"] = str(next(state["router"].parameters()).device)
    for index, model in enumerate(state["estimator"].models):
        model_devices[f"competence_{index}"] = str(next(model.parameters()).device)
    if not model_devices or any(
        d != hardware["selected"] for d in model_devices.values()
    ):
        raise AssertionError(f"Model placement mismatch: {model_devices}")
    checkpoint = output / "experiment/fame_scm_active/mask_0/seed_0/deployment.pkl"
    predict(
        checkpoint, data, output / "gpu_predictions.csv", device=hardware["selected"]
    )
    predict(checkpoint, data, output / "cpu_predictions.csv", device="cpu")
    report = {
        "status": "passed",
        "kind": "synthetic_only",
        "hardware": hardware,
        "model_devices": model_devices,
        "methods": results["method"].tolist(),
        "audit": audit(output / "experiment"),
        "gpu_and_cpu_deployment": "passed",
        "scope": "Core Torch CUDA execution; extended experts, DDP, paper scores and GPU-hours telemetry are not validated by this check",
    }
    write_json(output / "gpu_validation.json", report)
    print(results[["method", "rmse", "actual_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
