import argparse, json
from pathlib import Path
from fame_scm.settings import load_settings


def main():
    parser = argparse.ArgumentParser(
        description="Sparse competence mining and budgeted forecasting"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "inspect", "demo"]:
        p = commands.add_parser(name)
        p.add_argument("--config", default="configs/experiments/quick_demo.yaml")
        p.add_argument("--data")
        p.add_argument("--out")
        p.add_argument("--methods", nargs="+")
        if name == "demo":
            p.add_argument("--series", type=int, default=32)
            p.add_argument("--days", type=int, default=420)
    p = commands.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/future_predictions.csv")
    commands.add_parser("status")
    p = commands.add_parser("audit")
    p.add_argument("--out", required=True)
    p = commands.add_parser("prepare-retail")
    p.add_argument("--dataset", choices=["m5", "favorita"], required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--max-series", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-days", type=int, default=15)
    p.add_argument("--sales-file", default="sales_train_evaluation.csv")
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2017-08-15")
    p.add_argument("--chunksize", type=int, default=500000)
    p.add_argument("--returns", choices=["clip_zero", "error"], default="clip_zero")
    p = commands.add_parser("suite")
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    p.add_argument(
        "--profile",
        choices=[
            "budgets",
            "mechanism",
            "ablations",
            "sensitivity",
            "recovery",
            "scale",
            "pools",
            "total_budget",
        ],
        required=True,
    )
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="archive unfinished job outputs and rerun them; completed jobs are skipped",
    )
    p = commands.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument(
        "--metric", choices=["wape", "smape", "rmse", "mae", "mse"], default="wape"
    )
    p.add_argument("--replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-days", type=int, default=28)
    p.add_argument("--group-component", type=int)
    p = commands.add_parser("holdout")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--component", type=int, default=1)
    p.add_argument("--fraction", type=float, default=0.2)
    p = commands.add_parser("pool-config")
    p.add_argument("--config", required=True)
    p.add_argument("--size", type=int, choices=[5, 10, 15, 20, 40], required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("insertion")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expert", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.4, 1.0])
    p.add_argument("--strategies", nargs="+", default=["scm", "random", "softimpute"])
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("equal-time")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", nargs="+", type=float, required=True)
    p.add_argument(
        "--methods", nargs="+", default=["fame_scm_active", "scm_passive", "pmf_a"]
    )
    p.add_argument("--rhos", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument(
        "--preparation", choices=["included", "shared_excluded"], default="included"
    )
    p = commands.add_parser("inventory")
    p.add_argument("--predictions", required=True)
    p.add_argument("--daily", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("consumers")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("calibrate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--full-grid", action="store_true")
    p = commands.add_parser("tables")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("doctor")
    p.add_argument("--config", required=True)
    p.add_argument("--check-data", action="store_true")
    for command in [
        "run",
        "inspect",
        "demo",
        "doctor",
        "suite",
        "holdout",
        "equal-time",
        "pool-config",
        "predict",
    ]:
        commands.choices[command].add_argument(
            "--device", help="cpu, cuda or cuda:<visible GPU index>; overrides config"
        )
    args = parser.parse_args()

    def configured():
        cfg = load_settings(args.config)
        if getattr(args, "device", None) is not None:
            cfg["device"] = args.device
        return cfg

    if args.command == "calibrate":
        from fame_scm.orchestration.calibration import calibrate_checkpoint

        print(
            json.dumps(
                calibrate_checkpoint(args.checkpoint, args.out, args.full_grid),
                indent=2,
            )
        )
        return
    if args.command == "consumers":
        from fame_scm.orchestration.consumers import run_consumers

        print(json.dumps(run_consumers(args.checkpoint, args.out), indent=2))
        return
    if args.command == "tables":
        from fame_scm.evaluation.tables import export_tables

        print(json.dumps(export_tables(args.root, args.out), indent=2))
        return
    if args.command == "doctor":
        from fame_scm.doctor import check_environment

        report = check_environment(configured(), args.check_data)
        print(json.dumps(report, indent=2))
        if report["status"] != "ready":
            raise SystemExit(1)
        return
    if args.command == "pool-config":
        import yaml
        from fame_scm.experts.pools import configure_pool

        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(configure_pool(configured(), args.size), sort_keys=False),
            encoding="utf-8",
        )
        print(path)
        return
    if args.command == "insertion":
        from fame_scm.orchestration.screening import run_insertion

        print(
            json.dumps(
                run_insertion(
                    args.checkpoint,
                    args.expert,
                    args.out,
                    args.ratios,
                    args.strategies,
                    args.seed,
                ),
                indent=2,
            )
        )
        return
    if args.command == "equal-time":
        from fame_scm.orchestration.timed import run_equal_time

        cfg = configured()
        cfg.setdefault("equal_time", {})["preparation_scope"] = args.preparation
        print(
            json.dumps(
                run_equal_time(cfg, args.out, args.seconds, args.methods, args.rhos),
                indent=2,
            )
        )
        return
    if args.command == "inventory":
        import yaml
        from fame_scm.evaluation.inventory import run_inventory

        print(
            json.dumps(
                run_inventory(
                    args.predictions,
                    args.daily,
                    yaml.safe_load(Path(args.policy).read_text()),
                    args.out,
                ),
                indent=2,
            )
        )
        return
    if args.command == "holdout":
        from fame_scm.orchestration.transfer import run_holdout

        cfg = configured()
        cfg["output"] = args.out
        print(json.dumps(run_holdout(cfg, args.component, args.fraction), indent=2))
        return
    if args.command == "prepare-retail":
        from fame_scm.datasets.retail import prepare_m5, prepare_favorita

        kw = dict(
            train_end=args.train_end,
            max_series=(
                args.max_series
                if args.max_series is not None
                else (30490 if args.dataset == "m5" else 18000)
            ),
            seed=args.seed,
            min_train_days=args.min_train_days,
        )
        report = (
            prepare_m5(args.raw, args.out, sales_file=args.sales_file, **kw)
            if args.dataset == "m5"
            else prepare_favorita(
                args.raw,
                args.out,
                start=args.start,
                end=args.end,
                chunksize=args.chunksize,
                returns=args.returns,
                **kw,
            )
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ["ids", "dates", "static"]
                },
                indent=2,
            )
        )
        return
    if args.command == "suite":
        from fame_scm.orchestration.suite import run_suite

        cfg = configured()
        if args.out:
            cfg["output"] = args.out
        print(
            run_suite(
                cfg,
                args.profile,
                args.execute,
                restart_incomplete=args.restart_incomplete,
            )
        )
        return
    if args.command == "compare":
        from fame_scm.evaluation.statistics import paired_bootstrap
        from fame_scm.reporting import write_json

        result = paired_bootstrap(
            args.out,
            args.method,
            args.reference,
            metric=args.metric,
            replicates=args.replicates,
            seed=args.seed,
            block_days=args.block_days,
            group_component=args.group_component,
        )
        write_json(
            Path(args.out)
            / f"comparison_{args.method}_vs_{args.reference}_{args.metric}.json",
            result,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "audit":
        from fame_scm.evaluation.audit import audit_directory

        print(json.dumps(audit_directory(args.out), ensure_ascii=False, indent=2))
        return
    if args.command == "status":
        from fame_scm.baselines.status import implementation_status

        print(json.dumps(implementation_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "predict":
        from fame_scm.routing.inference import predict_file

        print(predict_file(args.checkpoint, args.data, args.out, device=args.device))
        return
    cfg = configured()
    if args.data:
        cfg["data"]["path"] = args.data
    if args.out:
        cfg["output"] = args.out
    if args.command == "inspect":
        from fame_scm.datasets.snbc import inspect_sales

        if cfg["data"].get("format") == "wide_csv":
            from fame_scm.datasets.wide import load_wide_panel

            report = load_wide_panel(cfg).audit
        elif cfg["data"].get("format") == "retail_panel":
            from fame_scm.datasets.retail import load_retail_panel

            report = load_retail_panel(cfg).audit
        else:
            report = inspect_sales(cfg)
        if args.out:
            from fame_scm.reporting import write_json

            write_json(Path(args.out) / "data_inspection.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "demo":
        if cfg["data"].get("format") == "wide_csv":
            raise ValueError(
                "demo generates daily SNBC data only; use scripts/validate_timeseries.py for hourly smoke tests"
            )
        from fame_scm.datasets.snbc import make_demo
        from fame_scm.reporting import write_json

        if Path(cfg["output"]).exists() and any(Path(cfg["output"]).iterdir()):
            raise ValueError("Demo output directory must be empty")
        cfg["result_kind"] = "synthetic_demo"
        cfg["data"]["path"] = str(Path(cfg["output"]) / "demo_daily_sales.csv")
        spec = make_demo(
            cfg["data"]["path"], series=args.series, days=args.days, seed=cfg["seed"]
        )
        write_json(Path(cfg["output"]) / "demo_data_manifest.json", spec)
    from fame_scm.orchestration.experiment import run_experiment

    table = run_experiment(cfg, methods=args.methods)
    print(
        table[
            ["method", "rmse", "mae", "wape", "actual_rho", "weighted_cost_ratio"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
