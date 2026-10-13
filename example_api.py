"""Run a synthetic Python-API example without installing FAME-SCM."""

import argparse
from pathlib import Path

from fame_scm.api import audit, load_config, make_demo, train
from fame_scm.reporting import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/api_example")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:<index>")
    args = parser.parse_args()
    from fame_scm.devices import prepare_device

    selected_device = prepare_device(args.device)
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh --out directory")
    output.mkdir(parents=True, exist_ok=True)
    data = output / "synthetic_daily.csv"
    make_demo(data, series=12, days=420, seed=42)
    config = load_config()
    config["result_kind"] = "synthetic_api_validation"
    config["scm"].update(epochs=2, members=2, batches=2)
    config["router"].update(epochs=2)
    config["experts"].update(
        neural_epochs=1, tree_estimators=5, max_train_windows=36, sarima_maxiter=3
    )
    table = train(
        config,
        data=data,
        out=output / "experiment",
        methods=["lightgbm", "fame_scm_active"],
        device=selected_device,
    )
    report = audit(output / "experiment")
    write_json(
        output / "api_validation.json",
        {
            "kind": "synthetic_only",
            "status": "passed",
            "methods": len(table),
            "audit": report,
        },
    )
    print(table[["method", "rmse", "actual_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
