"""Current-manuscript synthetic integration checks; never paper benchmark scores."""

import argparse
from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fame_scm.settings import load_settings
from fame_scm.datasets.snbc import make_demo
from fame_scm.orchestration.experiment import run_experiment
from fame_scm.orchestration.consumers import run_consumers
from fame_scm.orchestration.calibration import calibrate_checkpoint
from fame_scm.evaluation.audit import audit_directory
from fame_scm.evaluation.statistics import paired_bootstrap
from fame_scm.evaluation.tables import export_tables
from fame_scm.reporting import write_json
from fame_scm.checkpoints import load_state, save_state
from fame_scm.reproduction import validate_protocol, protocol_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Fresh validation output required")
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_settings()
    cfg["threads"] = 1
    cfg["data"].update(path=str(root / "daily.csv"), max_series=12)
    make_demo(cfg["data"]["path"], series=12, days=700, seed=cfg["seed"])
    cfg["result_kind"] = "synthetic_revision_validation"
    cfg["output"] = str(root / "main")
    cfg["scm"].update(
        epochs=3, members=2, batches=2, rank_margin_mode="acquired_rolling"
    )
    cfg["router"].update(epochs=3, calibration_thresholds=[0, 0.05])
    cfg["evaluation"].update(cost_reference="measured_audit")
    cfg["research"] = {"save_state": True}
    cfg["experts"].update(
        neural_epochs=1, tree_estimators=5, sarima_maxiter=3, max_train_windows=36
    )
    cfg["baselines"]["epochs"] = 3
    methods = [
        "lightgbm",
        "full_observation",
        "random_partial",
        "softimpute",
        "imc",
        "independent_lgbm",
        "independent_neural",
        "pmf_a",
        "nn_ei_c",
        "scm_passive",
        "fame_scm_active",
    ]
    run_experiment(cfg, methods=methods)
    audit = audit_directory(root / "main")
    checkpoint = root / "main/fame_scm_active/mask_0/seed_0/build_state.pkl"
    consumers = run_consumers(checkpoint, root / "consumers")
    state = load_state(checkpoint)
    state["cfg"]["calibration"] = {"grid": {"tau": [0.1, 0.3]}}
    search_checkpoint = root / "search_state.pkl"
    save_state(search_checkpoint, state)
    calibration = calibrate_checkpoint(search_checkpoint, root / "calibration")
    ledger = json.loads((root / "calibration/summary.json").read_text())[
        "evidence_account"
    ]
    assert (
        calibration["trials_fitted"] == 2
        and ledger["stages"]["test"]["executed_events"] > 0
    )
    expected = protocol_snapshot(state["cfg"], state["panel"])
    protocol = root / "author_protocol.json"
    write_json(protocol, expected)
    strict = dict(
        state["cfg"], reproduction={"strict": True, "author_protocol": str(protocol)}
    )
    assert (
        validate_protocol(strict, state["panel"])["status"] == "author_protocol_matched"
    )
    expected["cohort_hash"] = "invalid"
    write_json(protocol, expected)
    try:
        validate_protocol(strict, state["panel"])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid protocol accepted")
    statistics = paired_bootstrap(
        root / "main",
        "fame_scm_active",
        "nn_ei_c",
        metric="rmse",
        replicates=10000,
        block_days=7,
    )
    write_json(root / "comparison.json", statistics)
    tables = export_tables(root, root / "tables")
    write_json(
        root / "validation_summary.json",
        {
            "kind": "synthetic_only",
            "status": "passed",
            "methods": len(methods),
            "audit": audit,
            "consumers": len(consumers),
            "calibration_trials": 2,
            "bootstrap_replicates": 10000,
            "tables": tables,
            "strict_protocol_match_and_mismatch": True,
        },
    )
    print("REVISION INTEGRATION PASSED")


if __name__ == "__main__":
    main()
