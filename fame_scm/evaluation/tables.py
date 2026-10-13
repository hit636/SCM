"""Export measured runs only. This module never reads paper target metadata."""

from pathlib import Path
import json
import pandas as pd
from fame_scm.reporting import write_json


def export_tables(root, out):
    root, out = Path(root), Path(out)
    rows, evidence = [], []
    for path in sorted(root.rglob("summary.json")):
        result = json.loads(path.read_text())
        if not result.get("metrics"):
            continue
        row = {
            "source": str(path.relative_to(root)),
            "method": result.get(
                "method", result.get("consumer", result.get("strategy", "unknown"))
            ),
            "dataset": result.get("dataset", "separate_control"),
            "n": result.get("n"),
            "m": result.get("m"),
            "rho": result.get("actual_rho", result.get("actual_ratio")),
            "mask": result.get("evidence_mask_id", 0),
            "seed": result.get("neural_seed", 0),
            "result_kind": result.get("result_kind", "measured_control"),
            **result["metrics"],
            "weighted_cost_ratio": result.get(
                "weighted_cost_ratio", result.get("new_column_weighted_cost_ratio")
            ),
            "build_seconds": result.get("seconds", {}).get("build"),
        }
        rows.append(row)
        account = result.get("evidence_account", {})
        for stage, values in account.get("stages", {}).items():
            evidence.append(
                {
                    "source": row["source"],
                    "method": row["method"],
                    "stage": stage,
                    **values,
                }
            )
    if not rows:
        raise ValueError("No measured summary results found")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "measured_runs.csv", index=False)
    pd.DataFrame(evidence).to_csv(out / "evidence_by_stage.csv", index=False)
    identity = ["dataset", "method", "n", "m", "rho", "result_kind"]
    numeric = [
        k
        for k in [
            "rmse",
            "mae",
            "mse",
            "wape",
            "smape",
            "weighted_cost_ratio",
            "build_seconds",
        ]
        if k in frame
    ]
    masks = frame.groupby(identity + ["mask"], dropna=False)[numeric].mean()
    aggregate = masks.groupby(identity, dropna=False).agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(k) for k in aggregate.columns]
    aggregate.to_csv(out / "measured_aggregate.csv")
    manifest = {
        "measured_runs": len(rows),
        "files": [
            "measured_runs.csv",
            "measured_aggregate.csv",
            "evidence_by_stage.csv",
        ],
        "aggregation": "Neural-seed means within masks; mean/SD across masks; cohorts, candidate counts, budgets and result kinds remain separate",
        "scope": "Measured outputs only; paper reference targets are never read or copied",
    }
    write_json(out / "export_manifest.json", manifest)
    return manifest
