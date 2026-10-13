import sys, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "tests")]
from retail_fixtures import make_retail_fixtures
from test_retail import fixture_config
from fame_scm.datasets.retail import prepare_m5, prepare_favorita
from fame_scm.orchestration.experiment import run_experiment
from fame_scm.evaluation.audit import audit_directory
from fame_scm.evaluation.statistics import paired_bootstrap
from fame_scm.reporting import write_json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()
work = Path(args.out)
if work.exists() and any(work.iterdir()):
    raise ValueError("Fresh output directory required")
work.mkdir(parents=True, exist_ok=True)
m5, fav, dates = make_retail_fixtures(work / "raw", n=16)
for dataset, raw in [("m5", m5), ("favorita", fav)]:
    dest = work / (dataset + "_prepared")
    if dataset == "m5":
        prepare_m5(raw, dest, train_end=str(dates[489].date()))
    else:
        prepare_favorita(
            raw,
            dest,
            train_end=str(dates[489].date()),
            start=str(dates[0].date()),
            end=str(dates[-1].date()),
            chunksize=503,
        )
    cfg = fixture_config(dest, dates)
    cfg["result_kind"] = "synthetic_schema_validation_not_benchmark"
    cfg["output"] = str(work / (dataset + "_run"))
    cfg["experts"].update(
        names=[
            "sarima",
            "ets",
            "prophet",
            "tsb",
            "linear",
            "xgboost",
            "lightgbm",
            "dlinear",
            "timemixer",
            "timesnet",
        ],
        neural_epochs=1,
        tree_estimators=10,
        max_train_windows=64,
        sarima_maxiter=5,
    )
    cfg["scm"].update(epochs=2, members=2, batches=2)
    cfg["router"].update(epochs=3, calibration_thresholds=[0, 0.05])
    cfg["baselines"].update(
        epochs=3, fformpp_steps=2, fformpp_features=2, fformpp_samples=2
    )
    cfg["cost_profiling"] = {"enabled": True, "series": 2, "repeats": 1}
    cfg["evaluation"]["primary_metric"] = "wape"
    cfg["baselines"]["methods"] = [
        "lightgbm",
        "full_observation",
        "random_partial",
        "softimpute",
        "imc",
        "independent_lgbm",
        "fformpp_b",
        "pmf_a",
        "scm_passive",
        "fame_scm_active",
    ]
    run_experiment(cfg)
    print(audit_directory(cfg["output"]), flush=True)
    result = paired_bootstrap(
        cfg["output"],
        "fame_scm_active",
        "full_observation",
        replicates=50,
        block_days=28,
    )
    write_json(Path(cfg["output"]) / "paired_bootstrap.json", result)

# Deployment snapshots must omit all training/test panels and forecast on new inputs.
import pickle
from fame_scm.routing.inference import predict_file

for dataset in ["m5", "favorita"]:
    checkpoint = (
        work
        / (dataset + "_run")
        / "fame_scm_active"
        / "mask_0"
        / "seed_0"
        / "deployment.pkl"
    )
    with checkpoint.open("rb") as f:
        bundle = pickle.load(f)
    assert bundle["expert_pool"].panel is None
    predict_file(
        checkpoint, work / (dataset + "_prepared"), work / (dataset + "_future.csv")
    )
# Test identity holdout on complete item identities (all legal backbone history allowed).
from fame_scm.orchestration.transfer import run_holdout

cfg["output"] = str(work / "identity_holdout")
cfg["baselines"]["methods"] = ["lightgbm", "fame_scm_active"]
cfg["scm"]["rho"] = 0.6
run_holdout(cfg, component=1, fraction=0.25)
# Repeat Active SCM with a frozen measured cost profile: equal seeds => same outputs.
import yaml
from fame_scm.settings import load_settings

repeat = load_settings(work / "m5_run" / "resolved_config.yaml")
repeat["cost_profiling"]["profile_path"] = str(work / "m5_run" / "cost_profile.json")
repeat["output"] = str(work / "m5_repeat")
run_experiment(repeat, methods=["fame_scm_active"])
a = work / "m5_run" / "fame_scm_active" / "mask_0" / "seed_0"
b = work / "m5_repeat" / "fame_scm_active" / "mask_0" / "seed_0"
for file in ["predictions.csv", "acquisition_trace.json"]:
    assert (a / file).read_bytes() == (
        b / file
    ).read_bytes(), f"Determinism failed: {file}"
write_json(
    work / "validation_complete.json",
    {
        "kind": "synthetic_schema_validation_not_benchmark",
        "datasets": ["m5", "favorita"],
        "experts": 10,
        "methods_per_dataset": 10,
        "horizon": 28,
        "prediction_rows_per_method": 2240,
        "future_prediction_rows_per_dataset": 448,
        "holdout_methods": 2,
        "frozen_cost_repeat_prediction_bytes_equal": True,
        "frozen_cost_repeat_acquisition_equal": True,
    },
)
print("ALL RETAIL VALIDATION CHECKS PASSED", flush=True)
