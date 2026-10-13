"""Inspectable experiment plans with restartable completed jobs."""

from pathlib import Path
from copy import deepcopy
import json
from datetime import datetime, timezone
import uuid
import yaml
from fame_scm.settings import merge
from fame_scm.reporting import write_json, content_hash


def make_plan(cfg, profile):
    cfg = deepcopy(cfg)
    if cfg.get("cost_profiling", {}).get("enabled"):
        cfg["cost_profiling"]["profile_path"] = str(
            Path(cfg["output"]) / "shared_cost_profile.json"
        )
    jobs = []

    def add(name, override):
        c = merge(cfg, override)
        c["output"] = str(Path(cfg["output"]) / name)
        jobs.append({"name": name, "config": c, "config_hash": content_hash(c)})

    if profile == "budgets":
        for rho in [0.2, 0.4, 0.6, 0.8]:
            add(f"rho_{rho:.1f}", {"scm": {"rho": rho}})
    elif profile == "mechanism":
        for rank in [0.0, 0.2]:
            for policy in ["scm_passive", "fame_scm_active"]:
                add(
                    f"rank_{rank}_{policy}",
                    {
                        "scm": {"lambda_rank": rank, "rho": 0.4},
                        "baselines": {"methods": [policy]},
                    },
                )
    elif profile == "ablations":
        variants = {
            "complete": {},
            "no_rank": {"scm": {"lambda_rank": 0}},
            "no_huber": {"scm": {"reconstruction": "mse"}},
            "no_row_normalization": {"scm": {"row_normalization": False}},
            "no_fingerprint": {"features": {"disabled": True}},
            "no_teacher_uncertainty": {"teacher": {"eta": 0}},
            "no_balance_cost": {"router": {"lambda_balance": 0, "lambda_cost": 0}},
            "kl_only": {"router": {"lambda_pred": 0}},
            "no_teacher_kl": {"router": {"lambda_teacher": 0}},
            "no_coverage": {"scm": {"beta": 0}},
            "no_acquisition_cost": {"scm": {"zeta": 0}},
            "uncertainty_only": {"scm": {"alpha": 0, "beta": 0, "zeta": 0}},
            "improvement_only": {
                "scm": {"uncertainty_weight": 0, "beta": 0, "zeta": 0}
            },
            "uncertainty_improvement": {"scm": {"beta": 0, "zeta": 0}},
        }
        for name, override in variants.items():
            add(name, merge({"baselines": {"methods": ["fame_scm_active"]}}, override))
    elif profile == "sensitivity":
        for param, values in [("alpha", [0.5, 1.0, 2.0]), ("beta", [0.1, 0.5, 1.0])]:
            for value in values:
                add(
                    f"{param}_{value}",
                    {
                        "scm": {param: value},
                        "baselines": {"methods": ["fame_scm_active"]},
                    },
                )
        for r in [1, 2, 3]:
            add(
                f"top_{r}",
                {"router": {"r": r}, "baselines": {"methods": ["fame_scm_active"]}},
            )
        for parameter, values in [("tau", [0.1, 0.3, 1.0]), ("eta", [0, 0.5, 2.0])]:
            for value in values:
                add(
                    f"{parameter}_{value}",
                    {
                        "teacher": {parameter: value},
                        "baselines": {"methods": ["fame_scm_active"]},
                    },
                )
    elif profile == "recovery":
        for method in ["independent_neural", "scm_passive"]:
            for rank in [0, 0.2] if method == "scm_passive" else [0]:
                add(
                    f"{method}_{rank}",
                    {"scm": {"lambda_rank": rank}, "baselines": {"methods": [method]}},
                )
    elif profile == "scale":
        from fame_scm.experts.pools import configure_pool

        for count in [5000, 10000, 25000, 50000, 100000]:
            add(
                f"N_{count}_M20",
                {
                    "data": {"max_series": count, "required_series_count": count},
                    "experts": configure_pool(cfg, 20)["experts"],
                    "cost_profiling": {"enabled": True, "profile_path": None},
                    "scm": {"rho": 0.305},
                    "baselines": {"methods": ["full_observation", "fame_scm_active"]},
                },
            )
    elif profile == "pools":
        from fame_scm.experts.pools import configure_pool

        for size, rho in [(5, 0.56), (10, 0.4), (15, 1 / 3), (20, 0.305), (40, 0.1975)]:
            add(
                f"pool_{size}",
                {
                    "experts": configure_pool(cfg, size)["experts"],
                    "cost_profiling": {"enabled": True, "profile_path": None},
                    "scm": {"rho": rho},
                    "baselines": {"methods": ["full_observation", "fame_scm_active"]},
                },
            )
    elif profile == "total_budget":
        # Absolute origin-request budgets are explicit, not inferred from cell rho.
        limits = cfg.get("budgets", {}).get("comparison_requests", [])
        if not limits:
            raise ValueError("total_budget requires budgets.comparison_requests")
        for limit in limits:
            add(f"requests_{limit}", {"budgets": {"construction_requests": limit}})
    else:
        raise ValueError("Unknown experiment profile")
    return {
        "schema_version": 1,
        "profile": profile,
        "jobs": jobs,
        "selection": "Configurations fixed before test; no result-based job selection",
    }


def _archive_incomplete_job(root, dest, job):
    if not dest.is_dir() or dest.is_symlink():
        raise ValueError(f"Expected a regular job directory: {dest}")
    archive_root = root / "_incomplete" / job["name"]
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archived = archive_root / f"{timestamp}_{uuid.uuid4().hex[:8]}"
    dest.rename(archived)
    write_json(
        archived / "suite_restart.json",
        {
            "job": job["name"],
            "original_output": str(dest),
            "config_hash": job["config_hash"],
            "archived_at_utc": timestamp,
            "reason": "Incomplete job retained before an explicitly requested restart",
            "accounting_scope": "Previous attempt evidence is retained here; suite_costs.json accounts for completed jobs",
        },
    )
    print(f"Archived incomplete job: {dest} -> {archived}", flush=True)
    return archived


def run_suite(cfg, profile, execute=False, *, restart_incomplete=False):
    if restart_incomplete and not execute:
        raise ValueError("--restart-incomplete requires --execute")
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(cfg, profile)
    planfile = root / f"{profile}_plan.json"
    if planfile.exists() and json.loads(planfile.read_text()) != plan:
        raise ValueError("Existing plan differs; choose new output")
    write_json(planfile, plan)
    for job in plan["jobs"]:
        (root / (job["name"] + ".yaml")).write_text(
            yaml.safe_dump(job["config"], sort_keys=False)
        )
        if not execute:
            continue
        dest = Path(job["config"]["output"])
        completion = dest / "job_complete.json"
        if completion.exists():
            try:
                completed = json.loads(completion.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                completed = {}
            if not isinstance(completed, dict):
                completed = {}
            recorded_hash = completed.get("config_hash")
            if recorded_hash and recorded_hash != job["config_hash"]:
                raise ValueError("Resume hash mismatch")
            if recorded_hash and all(
                (dest / name).is_file() for name in ["run_manifest.json", "results.csv"]
            ):
                continue
        if dest.exists() and any(dest.iterdir()):
            if not restart_incomplete:
                raise ValueError(
                    f"Incomplete job at {dest}. After stopping the previous run, "
                    "rerun this suite with --execute --restart-incomplete to "
                    "archive its outputs and restart the job."
                )
            _archive_incomplete_job(root, dest, job)
        from fame_scm.orchestration.experiment import run_experiment

        run_experiment(deepcopy(job["config"]))
        write_json(completion, {"config_hash": job["config_hash"]})
    if execute:
        ledgers = []
        for job in plan["jobs"]:
            dest = Path(job["config"]["output"])
            manifest = json.loads((dest / "run_manifest.json").read_text())
            import pandas as pd

            results = pd.read_csv(dest / "results.csv")
            ledgers.append(
                {
                    "job": job["name"],
                    "selected_build_seconds_sum": float(results.build_seconds.sum()),
                    "shared_expert_fit_seconds": manifest["shared_expert_fit_seconds"],
                    "feature_seconds": manifest["feature_seconds"],
                    "cost_profile_seconds": manifest["cost_profile_seconds"],
                }
            )
        write_json(
            root / "suite_costs.json",
            {
                "jobs": ledgers,
                "search_inclusive_seconds": sum(
                    sum(v for k, v in row.items() if k != "job") for row in ledgers
                ),
                "scope": "All completed builds plus each job preparation/profile; test and research audit excluded",
            },
        )
    return str(planfile)
