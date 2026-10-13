"""Generate/execute pool sweeps and all five independent new-expert insertions."""

from pathlib import Path
import sys, argparse, json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fame_scm.settings import load_settings
from fame_scm.experts.pools import configure_pool
from fame_scm.reporting import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["pools", "insertion"])
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--checkpoint",
        help="Path template containing {mask} and {seed} for repeated builds",
    )
    p.add_argument("--execute", action="store_true")
    p.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 15, 20, 40])
    p.add_argument(
        "--experts",
        nargs="+",
        default=["catboost", "patchtst", "nhits", "deepar", "chronos"],
    )
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_settings(a.config)
    plan = []
    if a.task == "pools":
        import yaml

        # Target evaluations per series in Table C7, not identical rho across pools.
        ratios = {5: 0.56, 10: 0.4, 15: 5 / 15, 20: 6.1 / 20, 40: 7.9 / 40}
        for n in a.sizes:
            c = configure_pool(cfg, n)
            c["scm"]["rho"] = ratios[n]
            c["output"] = str(root / f"pool_{n}")
            path = root / f"pool_{n}.yaml"
            path.write_text(yaml.safe_dump(c, sort_keys=False))
            plan.append({"size": n, "config": str(path), "rho": ratios[n]})
            if a.execute:
                from fame_scm.orchestration.experiment import run_experiment

                run_experiment(c, methods=["full_observation", "fame_scm_active"])
    else:
        if not a.checkpoint:
            p.error("--checkpoint is required for insertion")
        repeated = (
            len(cfg["evaluation"]["mask_seeds"])
            * len(cfg["evaluation"]["neural_seeds"])
            > 1
        )
        if repeated and not all(
            token in a.checkpoint for token in ["{mask}", "{seed}"]
        ):
            p.error(
                "Repeated insertion needs a matching checkpoint for every mask/seed; provide a template with {mask} and {seed}"
            )
        for expert in a.experts:
            for mask in cfg["evaluation"]["mask_seeds"]:
                for seed in cfg["evaluation"]["neural_seeds"]:
                    dest = root / expert / f"mask_{mask}" / f"seed_{seed}"
                    checkpoint = a.checkpoint.format(mask=mask, seed=seed)
                    plan.append(
                        {
                            "expert": expert,
                            "out": str(dest),
                            "mask": mask,
                            "seed": seed,
                            "checkpoint": checkpoint,
                        }
                    )
                    if a.execute:
                        from fame_scm.orchestration.screening import run_insertion

                        run_insertion(
                            checkpoint,
                            expert,
                            dest,
                            seed=cfg["seed"] + 10000 * mask + seed,
                        )
    write_json(root / "suite_plan.json", plan)
    if a.execute and a.task == "insertion":
        import pandas as pd

        rows = []
        for path in root.glob("*/mask_*/seed_*/*/summary.json"):
            s = json.loads(path.read_text())
            rows.append(
                {
                    "path": str(path),
                    "expert": s["new_expert"],
                    "strategy": s["strategy"],
                    "rho": s["requested_ratio"],
                    "rmse": s["metrics"]["rmse"],
                    "use_fraction": s["new_expert_use_fraction"],
                    "update_seconds": s["update_seconds"],
                }
            )
        table = pd.DataFrame(rows)
        table.to_csv(root / "results.csv", index=False)
        table.groupby(["strategy", "rho"])[
            ["rmse", "use_fraction", "update_seconds"]
        ].agg(["mean", "std"]).to_csv(root / "aggregate.csv")


if __name__ == "__main__":
    main()
