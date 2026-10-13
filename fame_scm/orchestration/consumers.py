"""Fit consumers on exactly the same acquired evidence and soft teacher."""

from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import pandas as pd
from fame_scm.checkpoints import load_state, state_training_seed
from fame_scm.acquisition.evidence import ExecutionLedger
from fame_scm.orchestration.execution import build_router_training, evaluate_stage
from fame_scm.routing.training import fit_router
from fame_scm.routing.consumers import DirectTeacher
from fame_scm.reporting import content_hash, write_json


def run_consumers(checkpoint, out):
    base = load_state(checkpoint)
    training_seed = state_training_seed(base)
    if base["estimator"] is None or base["teacher"] is None:
        raise ValueError("Consumers require a saved SCM build checkpoint")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Consumer output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    configurations = {
        "complete": {},
        "kl_only": {"lambda_pred": 0.0},
        "direct_teacher": {"consumer": "direct_teacher"},
        "straight_through": {"operator": "straight_through"},
        "gumbel_top_r": {"operator": "gumbel_top_r"},
        "sparsemax": {"operator": "sparsemax"},
    }
    summaries = []
    for name, override in configurations.items():
        cfg = deepcopy(base["cfg"])
        cfg["router"].update(override)
        store = deepcopy(base["store"])
        store.ledger = ExecutionLedger(out / name / "execution_events.jsonl")
        # Re-run calibration in a separate physical ledger; mining evidence is shared.
        store.cache = {k: v for k, v in store.cache.items() if k.stage == "mining"}
        store._construction_keys = {
            k for k in store._construction_keys if k.stage == "mining"
        }
        dest = out / name
        dest.mkdir()
        panel, feature = base["panel"], base["feature"]
        measured, observed = store.snapshot()
        cost = np.array([base["pool"].cost_profile["online"][e] for e in store.experts])
        started = time.perf_counter()
        args = build_router_training(panel, feature, store, base["teacher"], cfg)
        if name == "direct_teacher":
            router = DirectTeacher(
                deepcopy(base["estimator"]),
                base["scaling"],
                measured,
                observed,
                store.availability,
                cost,
                cfg["teacher"],
                entities=panel.ids,
            )
            logs = []
        elif name == "complete":
            router, logs = deepcopy(base["router"]), []
        else:
            router, logs = fit_router(cfg, *args, cost, training_seed)
        fitting = time.perf_counter() - started
        best, selected, trials = None, None, []
        calibration_start = time.perf_counter()
        for threshold in cfg["router"]["calibration_thresholds"]:
            metrics, _, _ = evaluate_stage(
                "calibration", panel, feature, store, router, cfg, threshold
            )
            score = (
                metrics[cfg["evaluation"].get("primary_metric", "rmse")],
                metrics["execution_count_mean"],
                threshold,
            )
            trials.append({"threshold": threshold, **metrics})
            if best is None or score < best:
                best, selected = score, threshold
        calibration_seconds = time.perf_counter() - calibration_start
        metrics, rows, routing = evaluate_stage(
            "test", panel, feature, store, router, cfg, selected
        )
        summary = {
            "consumer": name,
            "router_training_seed": training_seed,
            "dataset": panel.audit.get("dataset", "snbc_format"),
            "n": len(panel.ids),
            "m": len(store.experts),
            "result_kind": cfg["result_kind"],
            "evidence_mask_id": base.get("evidence_mask_id", 0),
            "neural_seed": base.get("neural_seed", 0),
            "actual_rho": float(observed.sum() / store.availability.sum()),
            "metrics": metrics,
            "threshold": selected,
            "router_fit_seconds": fitting,
            "calibration_seconds": calibration_seconds,
            "observed_mask_hash": content_hash(observed.tolist()),
            "teacher_hash": content_hash(base["teacher"].tolist()),
            "evidence_account": store.evidence_account(),
            "shared_source_evidence_account": base["store"].evidence_account(),
            "scope": "Shared charged base mining evidence; consumer-specific fitting and calibration; no new mining labels",
        }
        pd.DataFrame(rows).to_csv(dest / "predictions.csv", index=False)
        pd.DataFrame(store.ledger.events).to_csv(
            dest / "execution_ledger.csv", index=False
        )
        pd.DataFrame(logs).to_csv(dest / "training.csv", index=False)
        write_json(dest / "summary.json", summary)
        write_json(dest / "routing.json", routing)
        write_json(dest / "calibration_trials.json", trials)
        summaries.append(summary)
    if (
        len({s["observed_mask_hash"] for s in summaries}) != 1
        or len({s["teacher_hash"] for s in summaries}) != 1
    ):
        raise AssertionError("Consumers changed shared evidence/teacher")
    write_json(out / "consumer_results.json", summaries)
    return summaries
