"""Synthetic integration verification, not public-dataset benchmark results."""

from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fame_scm.settings import load_settings
from fame_scm.orchestration.experiment import run_experiment
from fame_scm.evaluation.audit import audit_directory
from fame_scm.routing.inference import predict_file
from fame_scm.datasets.snbc import make_demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/timeseries_validation")
    args = p.parse_args()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty validation output directory")
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for dataset in ["snbc", "electricity", "traffic"]:
        cfg = load_settings(ROOT / f"configs/experiments/{dataset}_smoke.yaml")
        path = out / f"{dataset}_synthetic.csv"
        if dataset == "snbc":
            make_demo(path, series=12, days=420, seed=42)
        else:
            rng = np.random.default_rng(42)
            t = np.arange(1200)
            frame = pd.DataFrame(
                {"date": pd.date_range("2024-01-01", periods=len(t), freq="h")}
            )
            for i in range(12):
                frame[f"series_{i}"] = (
                    10
                    + i
                    + 3 * np.sin(2 * np.pi * t / 24 + i)
                    + rng.normal(0, 0.2, len(t))
                )
            frame.to_csv(path, index=False)
        cfg["data"]["path"] = str(path)
        cfg["output"] = str(out / dataset)
        cfg["result_kind"] = "synthetic_integration_test"
        run_experiment(cfg)
        audit = audit_directory(cfg["output"])
        checkpoint = (
            Path(cfg["output"]) / "fame_scm_active/mask_0/seed_0/deployment.pkl"
        )
        future = out / f"{dataset}_future.csv"
        predict_file(checkpoint, str(path), str(future))
        predictions = pd.read_csv(
            Path(cfg["output"]) / "fame_scm_active/mask_0/seed_0/predictions.csv"
        )
        assert not predictions.duplicated(
            ["entity_id", "origin", "horizon_step", "target_date"]
        ).any()
        forecast = pd.read_csv(future)
        assert np.isfinite(forecast.forecast_quantity).all()
        dates = pd.to_datetime(
            forecast[forecast.entity_id == forecast.entity_id.iloc[0]].date
        )
        step = pd.Timedelta(days=1) if dataset == "snbc" else pd.Timedelta(hours=1)
        assert (dates.diff().dropna() == step).all()
        reports[dataset] = {
            "audit": audit,
            "prediction_rows": len(predictions),
            "future_rows": len(forecast),
            "frequency_verified": str(step),
        }
    (out / "validation_summary.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
