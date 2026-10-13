"""Run independent dataset/horizon experiments; never overwrite finished runs."""

from pathlib import Path
import argparse
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fame_scm.settings import load_settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["snbc", "m5", "favorita", "electricity", "traffic"],
        default=["snbc", "m5", "favorita", "electricity", "traffic"],
    )
    p.add_argument("--profile", choices=["smoke", "main"], default="smoke")
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        help="Hourly datasets only; SNBC retains its configured 14-day horizon",
    )
    p.add_argument("--out", default="runs/three_datasets")
    p.add_argument("--device", help="cpu, cuda or cuda:<visible GPU index>")
    p.add_argument("--max-series", type=int)
    p.add_argument("--dry-run", action="store_true")
    for name in ["snbc", "m5", "favorita", "electricity", "traffic"]:
        p.add_argument(f"--{name}-data")
    args = p.parse_args()
    if args.horizons and (
        min(args.horizons) < 1 or len(set(args.horizons)) != len(args.horizons)
    ):
        p.error("horizons must be distinct positive integers")
    if len(set(args.datasets)) != len(args.datasets):
        p.error("datasets must be distinct")
    if args.max_series is not None and args.max_series < 1:
        p.error("max-series must be positive")
    # Resolve all input/output paths before changing cwd in subprocesses.
    output = Path(args.out).resolve()
    jobs = []
    for name in args.datasets:
        cfg = load_settings(ROOT / f"configs/experiments/{name}_{args.profile}.yaml")
        override = getattr(args, f"{name}_data")
        path = Path(override).resolve() if override else ROOT / cfg["data"]["path"]
        if not path.exists():
            raise FileNotFoundError(f"{name}: missing {path}; provide --{name}-data")
        cfg["data"]["path"] = str(path)
        if args.device:
            cfg["device"] = args.device
        if args.max_series:
            cfg["data"]["max_series"] = args.max_series
        for horizon in (
            [cfg["forecast"]["horizon"]]
            if name in ["snbc", "m5", "favorita"] or not args.horizons
            else args.horizons
        ):
            from copy import deepcopy

            job = deepcopy(cfg)
            job["forecast"]["horizon"] = horizon
            job["output"] = str(output / f"{name}_H{horizon}_{args.profile}")
            dest = Path(job["output"])
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError(f"Choose fresh --out: {dest} already exists")
            jobs.append(job)
    plans = output / "configs"
    plans.mkdir(parents=True, exist_ok=True)
    for cfg in jobs:
        name = Path(cfg["output"]).name
        file = plans / f"{name}.yaml"
        file.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        cmd = [sys.executable, str(ROOT / "run.py"), "run", "--config", str(file)]
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
