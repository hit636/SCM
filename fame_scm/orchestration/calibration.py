"""Teacher/router hyperparameter selection on calibration targets only."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import time
import numpy as np
import pandas as pd
from fame_scm.checkpoints import load_state, save_state, state_training_seed
from fame_scm.acquisition.evidence import ExecutionLedger
from fame_scm.routing.teacher import competence_teacher
from fame_scm.routing.training import fit_router
from fame_scm.orchestration.execution import build_router_training, evaluate_stage
from fame_scm.reporting import write_json


def calibrate_checkpoint(checkpoint, out, full_grid=False):
    state = load_state(checkpoint)
    training_seed = state_training_seed(state)
    if state["estimator"] is None:
        raise ValueError("Calibration search requires fitted competence state")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Calibration output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(state["cfg"])
    store = deepcopy(state["store"])
    store.ledger = ExecutionLedger(out / "search_events.jsonl")
    panel, features = state["panel"], state["feature"]
    measured, observed = store.snapshot()
    mean, sd = state["estimator"].predict(state["z"])
    costs = np.array([state["pool"].cost_profile["online"][e] for e in store.experts])
    grid = {
        "tau": [0.1, 0.3, 1.0],
        "eta": [0.0, 0.5, 2.0],
        "nu": [0.0, 0.05, 0.1],
        "lambda_balance": [0.0, 0.01, 0.05],
        "lambda_cost": [0.0, 0.01, 0.05],
    }
    if not full_grid:
        grid = {
            key: [cfg["teacher"][key] if key in cfg["teacher"] else cfg["router"][key]]
            for key in grid
        }
    grid.update(cfg.get("calibration", {}).get("grid", {}))
    if not all(isinstance(values, list) and values for values in grid.values()):
        raise ValueError("Calibration grid must contain nonempty value lists")
    trials, selected, best = [], None, None
    start = time.perf_counter()
    for trial_id, values in enumerate(product(*(grid[k] for k in grid))):
        c = deepcopy(cfg)
        parameters = dict(zip(grid, values))
        for key, value in parameters.items():
            c["teacher" if key in c["teacher"] else "router"][key] = value
        store.ledger.context = {"trial_id": trial_id}
        trial_start = time.perf_counter()
        teacher = competence_teacher(
            measured,
            observed,
            mean,
            sd,
            store.availability,
            costs,
            **{k: c["teacher"][k] for k in ["tau", "eta", "nu"]},
        )
        args = build_router_training(panel, features, store, teacher, c)
        router, _ = fit_router(c, *args, costs, training_seed)
        for threshold in c["router"]["calibration_thresholds"]:
            metrics, _, _ = evaluate_stage(
                "calibration", panel, features, store, router, c, threshold
            )
            score = (
                metrics[c["evaluation"].get("primary_metric", "rmse")],
                metrics["execution_count_mean"],
                threshold,
                trial_id,
            )
            record = {
                "trial_id": trial_id,
                "parameters": parameters,
                "threshold": threshold,
                "calibration_metrics": metrics,
            }
            trials.append(record)
            if best is None or score < best:
                best, selected = score, {
                    "cfg": deepcopy(c),
                    "router": deepcopy(router),
                    "teacher": teacher.copy(),
                    "threshold": threshold,
                    "trial_id": trial_id,
                }
        trials[-1]["fit_and_calibration_seconds"] = time.perf_counter() - trial_start
        write_json(out / "calibration_search.json", trials)
    search_seconds = time.perf_counter() - start
    store.ledger.context = {"trial_id": selected["trial_id"], "selection_frozen": True}
    metrics, records, routing = evaluate_stage(
        "test",
        panel,
        features,
        store,
        selected["router"],
        selected["cfg"],
        selected["threshold"],
    )
    pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(store.ledger.events).to_csv(out / "execution_ledger.csv", index=False)
    write_json(out / "routing.json", routing)
    result = {
        "method": "calibration_search",
        "router_training_seed": training_seed,
        "dataset": panel.audit.get("dataset", "snbc_format"),
        "n": len(panel.ids),
        "m": len(store.experts),
        "result_kind": cfg["result_kind"],
        "evidence_mask_id": state.get("evidence_mask_id", 0),
        "neural_seed": state.get("neural_seed", 0),
        "actual_rho": float(observed.sum() / store.availability.sum()),
        "selected_trial": selected["trial_id"],
        "selected_teacher": selected["cfg"]["teacher"],
        "selected_threshold": selected["threshold"],
        "metrics": metrics,
        "grid": grid,
        "trials_fitted": len(list(product(*(grid[k] for k in grid)))),
        "search_seconds": search_seconds,
        "evidence_account": store.evidence_account(),
        "shared_source_evidence_account": state["store"].evidence_account(),
        "scope": "All trial fit/calibration work charged; shared source-build preparation/evidence separate; test scored only after selection",
    }
    write_json(out / "summary.json", result)
    save_state(
        out / "deployment.pkl",
        {
            "config": selected["cfg"],
            "features": features,
            "expert_pool": state["pool"].deployment_copy(),
            "router": selected["router"].cpu(),
            "threshold": selected["threshold"],
        },
    )
    return result
