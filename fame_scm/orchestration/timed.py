"""Hard wall-clock construction budgets with atomic completed checkpoints.

Each budget/method starts a fresh process. Preparation is charged. Increasing
cell-budget attempts are restarted, with all repeated work charged. The last
fully calibrated checkpoint completed by the deadline is scored afterwards.
No test metrics participate in stopping or checkpoint selection.
"""

import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
from copy import deepcopy
import pandas as pd
import torch
import yaml
from fame_scm.reporting import write_json


def build_worker(config, method, out, rhos):
    from fame_scm.settings import load_settings
    from fame_scm.orchestration.experiment import (
        load_panel,
        Forecastability,
        ExpertPool,
        profile_costs,
        run_method,
    )

    cfg = load_settings(config)
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    cfg.setdefault("research", {}).update(build_only=True, save_state=True)
    cfg["evaluation"]["hidden_audit"] = False
    out = Path(out)
    t = time.perf_counter()
    if cfg.get("equal_time", {}).get("prepared_checkpoint"):
        from fame_scm.checkpoints import load_state

        prepared = load_state(cfg["equal_time"]["prepared_checkpoint"])
        panel, feature, pool = prepared["panel"], prepared["feature"], prepared["pool"]
    else:
        panel = load_panel(cfg)
        feature = Forecastability(cfg).fit(panel)
        pool = ExpertPool(cfg, panel).fit()
        pool.cost_profile = profile_costs(pool, panel, cfg)
    write_json(
        out / "preparation.json",
        {
            "seconds": time.perf_counter() - t,
            "cost_profile": pool.cost_profile,
            "expert_fit_log": pool.logs,
        },
    )
    for k, rho in enumerate([1.0] if method == "full_observation" else rhos):
        cfg["scm"]["rho"] = rho
        run_method(cfg, panel, pool, feature, method, 0, 0, out / f"attempt_{k:03d}")


def score_checkpoint(checkpoint, out):
    from fame_scm.checkpoints import load_state
    from fame_scm.orchestration.experiment import evaluate_stage

    state = load_state(checkpoint)
    cfg = state["cfg"]
    torch.set_num_threads(cfg["threads"])
    store = state["store"]
    store.ledger.path = str(Path(out) / "score_events.jsonl")
    before = len(store.ledger.events)
    t = time.perf_counter()
    met, rows, routing = evaluate_stage(
        "test",
        state["panel"],
        state["feature"],
        store,
        state["router"],
        cfg,
        state["threshold"],
        fixed_expert=(
            state["method"] if state["method"] in state["pool"].names else None
        ),
    )
    pd.DataFrame(rows).to_csv(Path(out) / "predictions.csv", index=False)
    write_json(Path(out) / "routing.json", routing)
    pd.DataFrame(store.ledger.events[before:]).to_csv(
        Path(out) / "test_execution_ledger.csv", index=False
    )
    result = {
        "metrics": met,
        "test_seconds_excluded": time.perf_counter() - t,
        "checkpoint": str(checkpoint),
    }
    write_json(Path(out) / "score.json", result)
    return result


def run_equal_time(cfg, out, budgets, methods, rhos=(0.2, 0.4, 0.6, 0.8, 1.0)):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Equal-time output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    if any(b <= 0 for b in budgets):
        raise ValueError("Budgets must be positive seconds")
    if any(not 0 < r <= 1 for r in rhos) or list(rhos) != sorted(set(rhos)):
        raise ValueError("rho schedule must be unique and increasing")
    cfg = deepcopy(cfg)
    scope = cfg.get("equal_time", {}).get("preparation_scope", "included")
    if scope not in ["included", "shared_excluded"]:
        raise ValueError("Invalid preparation_scope")
    if cfg.get("equal_time", {}).get("prepared_checkpoint"):
        raise ValueError(
            "Prepared checkpoints are generated internally to bind data and config"
        )
    if scope == "shared_excluded":
        from fame_scm.orchestration.experiment import (
            load_panel,
            Forecastability,
            ExpertPool,
            profile_costs,
        )
        from fame_scm.checkpoints import save_state

        torch.set_num_threads(cfg["threads"])
        prep_start = time.perf_counter()
        panel = load_panel(cfg)
        feature = Forecastability(cfg).fit(panel)
        pool = ExpertPool(cfg, panel).fit()
        pool.cost_profile = profile_costs(pool, panel, cfg)
        path = out / "shared_preparation.pkl"
        save_state(path, {"panel": panel, "feature": feature, "pool": pool})
        cfg.setdefault("equal_time", {})["prepared_checkpoint"] = str(path.resolve())
        write_json(
            out / "shared_preparation.json",
            {
                "seconds_excluded": time.perf_counter() - prep_start,
                "expert_fit_log": pool.logs,
                "cost_profile": pool.cost_profile,
            },
        )
    config = out / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    results = []
    for seconds in budgets:
        for method in methods:
            dest = out / f"{method}_{seconds:g}s"
            dest.mkdir()
            started = time.monotonic()
            deadline = started + seconds
            wall_deadline = time.time() + seconds
            peak_rss = 0
            args = [
                sys.executable,
                "-m",
                "fame_scm.orchestration.timed",
                "build",
                "--config",
                str(config.resolve()),
                "--method",
                method,
                "--out",
                str(dest.resolve()),
                "--rhos",
                ",".join(map(str, rhos)),
            ]
            with (dest / "worker.log").open("w") as log:
                worker_env = os.environ.copy()
                source_root = str(Path(__file__).resolve().parents[2])
                worker_env["PYTHONPATH"] = (
                    source_root + os.pathsep + worker_env.get("PYTHONPATH", "")
                )
                process = subprocess.Popen(
                    args,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name == "posix"),
                    env=worker_env,
                )
                try:
                    import psutil

                    monitor = psutil.Process(process.pid)
                except (ImportError, ProcessLookupError):
                    monitor = None
                except Exception as exc:
                    if type(exc).__name__ not in ["NoSuchProcess", "AccessDenied"]:
                        raise
                    monitor = None
                timed_out = False
                while process.poll() is None:
                    if monitor:
                        try:
                            peak_rss = max(
                                peak_rss,
                                sum(
                                    p.memory_info().rss
                                    for p in [monitor]
                                    + monitor.children(recursive=True)
                                ),
                            )
                        except psutil.Error:
                            pass
                    if time.monotonic() >= deadline:
                        timed_out = True
                        try:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                        except ProcessLookupError:
                            pass
                        break
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                process.wait()
            elapsed = time.monotonic() - started
            # Completion is attested by the atomic build_summary written AFTER checkpoint serialization.
            complete = sorted(
                p
                for p in dest.glob("attempt_*/build_summary.json")
                if p.stat().st_mtime <= wall_deadline
            )
            selected = complete[-1].parent / "build_state.pkl" if complete else None
            entry = {
                "method": method,
                "budget_seconds": seconds,
                "actual_construction_seconds": elapsed,
                "timeout": timed_out,
                "worker_exit_code": process.returncode,
                "completed_attempts": len(complete),
                "peak_rss_bytes": peak_rss or None,
                "preparation_scope": scope,
                "boundary": (
                    "includes interpreter, data/features, backbone fit, cost profiling, repeated mining/recovery/router/calibration and checkpoint writing; test excluded"
                    if scope == "included"
                    else "shared feature/backbone/profile preparation excluded and reported once; includes interpreter/checkpoint loading, all repeated mining/recovery/router/calibration/checkpoint writing; test excluded"
                ),
                "status": (
                    "no_complete_checkpoint"
                    if selected is None
                    else "completed_checkpoint"
                ),
            }
            if selected:
                entry["selected_build"] = json.loads(
                    (selected.parent / "build_summary.json").read_text()
                )
                entry.update(score_checkpoint(selected, dest))
            if process.returncode and not timed_out:
                entry["worker_error"] = (
                    "See worker.log; this run must not be treated as a successful fair comparison"
                )
            # All completed/partial attempts are charged, not just selected checkpoint.
            events = []
            for p in dest.glob("attempt_*/execution_events.jsonl"):
                for line in p.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # possible interrupted final write
            entry["executed_requests_all_attempts"] = sum(
                not e["cache_hit"] for e in events
            )
            entry["expert_executed_seconds_all_attempts"] = sum(
                e["elapsed_seconds"] for e in events if not e["cache_hit"]
            )
            write_json(dest / "budget_result.json", entry)
            results.append(entry)
    write_json(out / "equal_time_results.json", results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["build"])
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rhos", required=True)
    a = p.parse_args()
    build_worker(a.config, a.method, a.out, [float(x) for x in a.rhos.split(",")])
