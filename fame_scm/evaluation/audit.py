"""Recompute saved results and check common evaluation targets across methods."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from fame_scm.evaluation.metrics import forecast_metrics
from fame_scm.reporting import check_pooled_metric_identity, write_json


def audit_directory(directory):
    directory = Path(directory)
    reports = []
    reference = None
    cfg = yaml.safe_load(
        (directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    r = cfg["router"]["r"]
    for summary_path in sorted(directory.glob("*/mask_*/seed_*/summary.json")):
        root = summary_path.parent
        summary = json.loads(summary_path.read_text())
        pred = (
            pd.read_csv(root / "predictions.csv")
            .sort_values(["entity_id", "origin", "horizon_step"])
            .reset_index(drop=True)
        )
        metrics = forecast_metrics(
            pred.target.to_numpy(),
            pred.prediction.to_numpy(),
            pred.valid.to_numpy(bool),
            pred.weight.to_numpy(),
        )
        check_pooled_metric_identity(metrics)
        for name in ["rmse", "mae", "mse", "wape", "smape"]:
            actual = metrics[name]
            reported = summary["metrics"][name]
            if actual is None:
                if reported is not None:
                    raise ValueError(f"{root}: {name} undefined but reported")
            elif not np.isclose(actual, reported, rtol=1e-9, atol=1e-10):
                raise ValueError(f"{root}: {name} mismatch")
        comparison = pred[
            [
                "entity_id",
                "origin",
                "horizon_step",
                "target_date",
                "target",
                "valid",
                "weight",
            ]
        ]
        if reference is None:
            reference = comparison
        else:
            pd.testing.assert_frame_equal(comparison, reference)
        if summary["acquired_cells"] > summary["cell_budget"]:
            raise ValueError("Budget exceeded")
        ledger = pd.read_csv(root / "execution_ledger.csv")
        if "evidence_account" in summary:
            recorded = summary["evidence_account"]
            for stage, account in recorded["stages"].items():
                events = ledger[ledger.stage == stage]
                keys_here = [
                    "entity_id",
                    "expert_id",
                    "origin",
                    "config_hash",
                    "protocol_hash",
                ]
                requests = len(events[events.success].drop_duplicates(keys_here))
                if requests != account["unique_origin_requests"]:
                    raise ValueError("Stage-origin evidence account mismatch")
            cap = recorded["construction_request_budget"]
            if (
                cap is not None
                and recorded["construction_unique_origin_requests"] > cap
            ):
                raise ValueError("Construction evidence budget exceeded")
        actual = ledger[~ledger.cache_hit]
        for stage, count in summary["execution_requests"].items():
            if int((actual.stage == stage).sum()) != count:
                raise ValueError("Execution ledger count mismatch")
        keys = [
            "stage",
            "entity_id",
            "expert_id",
            "origin",
            "config_hash",
            "protocol_hash",
        ]
        # A request key is executed once; subsequent requests must be cache hits.
        if actual.duplicated(keys).any():
            raise ValueError("Uncharged/repeated request key execution")
        if summary["acquired_cells"]:
            evidence = pd.read_csv(root / "evidence_cells.csv")
            if (
                len(evidence) != summary["acquired_cells"]
                or evidence.duplicated(["entity", "expert"]).any()
            ):
                raise ValueError("Evidence cell ledger mismatch")
            pack = np.load(root / "competence.npz")
            if int(pack["observed"].sum()) != summary["acquired_cells"]:
                raise ValueError("Saved mask differs from budget ledger")
            if "availability" in pack:
                from fame_scm.acquisition.evidence import weighted_work_ratio

                profile = json.loads((directory / "cost_profile.json").read_text())
                ratio = weighted_work_ratio(
                    pack["observed"],
                    pack["availability"],
                    (
                        np.load(root / "evaluator_cell_work_seconds.npy")
                        if summary.get("weighted_cost_reference")
                        == "measured_selected_and_hidden_audit_work"
                        else [profile["offline"][x] for x in summary["expert_names"]]
                    ),
                )
                if not np.isclose(ratio, summary["weighted_cost_ratio"]):
                    raise ValueError("Weighted cost mismatch")
        routes = json.loads((root / "routing.json").read_text())
        for row in routes:
            if not 1 <= len(row["selected_experts"]) <= r:
                raise ValueError("Routing count exceeds configured r")
            if not np.isclose(sum(row["weights"]), 1):
                raise ValueError("Fusion weights do not sum to 1")
        reports.append(
            {
                "method": summary["method"],
                "mask": summary["evidence_mask_id"],
                "seed": summary["neural_seed"],
                "prediction_records": len(pred),
                "rmse": metrics["rmse"],
                "status": "passed",
            }
        )
    if not reports:
        raise ValueError("No completed method results found")
    output = {"runs_checked": len(reports), "common_targets": True, "checks": reports}
    write_json(directory / "consistency_audit.json", output)
    return output
