"""Executable synthetic backend checks, including backend save/reload equality.

Use --all-experts to validate all configured forecasting backends after installing
requirements.txt.
"""

from pathlib import Path
import argparse, sys, pickle, json
import numpy as np
import pandas as pd
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1] / "tests"),
]
from fame_scm.settings import load_settings
from fame_scm.datasets.snbc import make_demo
from fame_scm.orchestration.experiment import load_panel, run_experiment
from fame_scm.experts.forecast_experts import ExpertPool
from fame_scm.experts.pools import configure_pool
from fame_scm.reporting import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--all-experts", action="store_true")
    a = p.parse_args()
    root = Path(a.out)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Fresh output required")
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    data = root / "demo.csv"
    make_demo(data, series=12, days=420, seed=42)
    cfg = load_settings()
    cfg["data"]["path"] = str(data.resolve())
    cfg["data"]["max_series"] = 12
    cfg["result_kind"] = "synthetic_backend_validation"
    cfg["forecast"]["max_origins_per_stage"] = 1
    cfg["experts"].update(
        names=["tsb", "lightgbm", "dlinear"],
        neural_epochs=1,
        tree_estimators=5,
        max_train_windows=12,
        nf_steps=1,
        sarima_maxiter=3,
    )
    cfg["scm"].update(epochs=2, members=2, batches=2, rho=0.6)
    cfg["router"].update(epochs=2, calibration_thresholds=[0, 0.05])
    cfg["evaluation"]["hidden_audit"] = False
    cfg["cost_profiling"] = {"enabled": True, "series": 1, "repeats": 1}
    cfg["research"] = {"save_state": True}
    if a.all_experts:
        ec = configure_pool(cfg, 40)
        panel = load_panel(ec)
        pool = ExpertPool(ec, panel).fit()
        origin = panel.origins["test"][0]
        preds = {n: pool.predict(n, 0, origin) for n in pool.names}
        path = root / "all_experts.pkl"
        with path.open("wb") as f:
            pickle.dump(pool.deployment_copy(), f)
        with path.open("rb") as f:
            loaded = pickle.load(f)
        loaded.panel = panel
        records = []
        for n, pred in preds.items():
            again = loaded.predict(n, 0, origin)
            np.testing.assert_allclose(pred, again, rtol=1e-5, atol=1e-5)
            records.append(
                {
                    "expert": n,
                    "finite": bool(np.isfinite(pred).all()),
                    "shape": list(pred.shape),
                    "save_reload_max_difference": float(np.abs(pred - again).max()),
                }
            )
        write_json(root / "all_expert_checks.json", records)
        path.unlink()
    cfg["output"] = str(root / "base")
    run_experiment(cfg, methods=["fame_scm_active"])
    checkpoint = root / "base/fame_scm_active/mask_0/seed_0/build_state.pkl"
    from fame_scm.orchestration.screening import run_insertion

    run_insertion(checkpoint, "linear", root / "insertion", ratios=[0.2, 0.4, 1.0])
    from fame_scm.orchestration.timed import run_equal_time

    # One deliberately too-small budget and one completed construction.
    results = run_equal_time(
        cfg, root / "equal_time", [0.05, 60], ["fame_scm_active"], rhos=[0.6]
    )
    assert results[0]["status"] == "no_complete_checkpoint"
    assert results[1]["status"] == "completed_checkpoint", results[1]
    from fame_scm.routing.inference import predict_file

    path = root / "insertion/scm_0.2/deployment.pkl"
    predict_file(path, data, root / "insertion_future.csv")
    from test_deployment import DeploymentContracts
    from fame_scm.evaluation.inventory import run_inventory

    pred, daily, policy = DeploymentContracts().fixtures()
    pred.to_csv(root / "inventory_predictions.csv", index=False)
    daily.to_csv(root / "inventory_input.csv", index=False)
    run_inventory(
        root / "inventory_predictions.csv",
        root / "inventory_input.csv",
        policy,
        root / "inventory",
    )
    write_json(
        root / "validation_summary.json",
        {
            "kind": "synthetic_only",
            "extended_backends": 40 if a.all_experts else 0,
            "insertion_runs": 7,
            "equal_time_runs": 2,
            "deployment_predict": True,
            "inventory_replay": True,
        },
    )


if __name__ == "__main__":
    main()
