"""End-to-end new-column screening over a common frozen old-column cache."""

from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from fame_scm.checkpoints import load_state, save_state
from fame_scm.acquisition.evidence import MiningStore, ExecutionLedger
from fame_scm.acquisition.scores import acquisition_scores
from fame_scm.experts.forecast_experts import ExpertPool
from fame_scm.evaluation.costs import profile_costs
from fame_scm.orchestration.execution import (
    TimedProvider,
    build_router_training,
    evaluate_stage,
)
from fame_scm.competence.insertion import append_competence, append_router
from fame_scm.routing.teacher import competence_teacher
from fame_scm.reporting import write_json, content_hash
from fame_scm.experts.pools import REVISION


def run_insertion(
    checkpoint,
    new_expert,
    out,
    ratios=(0.1, 0.2, 0.4, 1.0),
    strategies=("scm", "random", "softimpute"),
    seed=42,
):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Insertion output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    base = load_state(checkpoint)
    cfg = deepcopy(base["cfg"])
    cfg.setdefault("research", {})["build_only"] = False
    if base["estimator"] is None or not hasattr(base["estimator"], "models"):
        raise ValueError(
            "Base checkpoint must contain SCM ensemble; run SCM with research.save_state=true"
        )
    torch.set_num_threads(cfg["threads"])
    panel = base["panel"]
    old_pool = base["pool"]
    old_store = base["store"]
    names = old_pool.names + [new_expert]
    if new_expert in old_pool.names:
        raise ValueError("New expert already present")
    nc = deepcopy(cfg)
    nc["experts"]["names"] = [new_expert]
    nc["experts"]["anchor"] = new_expert
    nc["experts"].setdefault("chronos_model", "amazon/chronos-t5-tiny")
    nc["experts"].setdefault("chronos_revision", REVISION)
    nc["cost_profiling"] = {
        "enabled": True,
        "series": cfg.get("cost_profiling", {}).get("series", 8),
        "repeats": cfg.get("cost_profiling", {}).get("repeats", 2),
    }
    t = time.perf_counter()
    new_pool = ExpertPool(nc, panel).fit()
    fit_seconds = time.perf_counter() - t
    new_pool.cost_profile = profile_costs(new_pool, panel, nc)
    # Common frozen old backbones plus one independently fitted new backbone.
    pool = deepcopy(old_pool)
    pool.panel = panel
    pool.names = names
    pool.models.update(new_pool.models)
    pool.cfg = deepcopy(cfg)
    pool.cfg["experts"]["names"] = names
    pool.cfg["experts"]["chronos_model"] = nc["experts"]["chronos_model"]
    pool.cfg["experts"]["chronos_revision"] = nc["experts"]["chronos_revision"]
    pool.cost_profile = deepcopy(old_pool.cost_profile)
    for field in ["offline", "online"]:
        pool.cost_profile[field][new_expert] = new_pool.cost_profile[field][new_expert]
    # New profile is normalized to its own anchor. Convert its measured seconds to OLD anchor units.
    if "median_seconds_per_request" in old_pool.cost_profile:
        anchor_sec = old_pool.cost_profile["median_seconds_per_request"][
            cfg["experts"]["anchor"]
        ]
        pool.cost_profile["online"][new_expert] = (
            new_pool.cost_profile["median_seconds_per_request"][new_expert] / anchor_sec
        )
    else:
        raise ValueError(
            "Insertion requires a measured base cost profile for consistent online cost units"
        )
    cfg["experts"] = deepcopy(pool.cfg["experts"])
    available = pool.availability(panel.origins["mining"])
    eligible = np.flatnonzero(available[:, -1])
    n = len(panel.ids)
    if not len(eligible):
        raise ValueError("No executable new-column cells")
    old_loss, old_mask = old_store.snapshot()
    origins = [str(o) for o in panel.origins["mining"]]
    costs = np.array([pool.cost_profile["online"][name] for name in names])
    feature = base["feature"]
    z = base["z"]
    summaries = []
    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise ValueError("Insertion ratio outside (0,1]")
        for strategy in ["full"] if ratio == 1 else strategies:
            if strategy not in ["scm", "random", "softimpute", "full"]:
                raise ValueError(strategy)
            dest = out / f"{strategy}_{ratio:g}"
            dest.mkdir()
            rng = np.random.default_rng(seed)
            budget = max(1, int(np.floor(ratio * len(eligible))))
            ledger = ExecutionLedger(dest / "execution_events.jsonl")
            configs = {
                **old_store.config_hashes,
                new_expert: content_hash(nc["experts"]),
            }
            store = MiningStore(
                panel.ids,
                names,
                available,
                int(old_mask.sum()) + budget,
                TimedProvider(panel, pool),
                configs,
                old_store.protocol_hash,
                ledger=ledger,
                cache=dict(old_store.cache),
            )
            store._cells = dict(
                old_store._cells
            )  # fixed common acquired evidence, no new charge
            trace = []
            update_seconds = 0.0
            estimator = None
            remaining = list(rng.permutation(eligible))
            seed_count = min(budget, max(1, int(np.ceil(0.05 * len(eligible)))))
            batches = [seed_count] + [
                len(a)
                for a in np.array_split(
                    np.arange(budget - seed_count), cfg["scm"]["batches"]
                )
                if len(a)
            ]
            for step, size in enumerate(batches):
                if step == 0 or strategy in ["random", "softimpute", "full"]:
                    selected = remaining[:size]
                else:
                    loss, mask = store.snapshot()
                    mu, sd = estimator.predict(z)
                    offline = np.array(
                        [pool.cost_profile["offline"][name] for name in names]
                    )
                    offline /= offline[names.index(cfg["experts"]["anchor"])]
                    score = acquisition_scores(
                        mu,
                        sd,
                        loss,
                        mask,
                        available,
                        offline,
                        alpha=cfg["scm"]["alpha"],
                        beta=cfg["scm"]["beta"],
                        zeta=cfg["scm"]["zeta"],
                        uncertainty_weight=cfg["scm"].get("uncertainty_weight", 1.0),
                    )[:, -1]
                    selected = sorted(remaining, key=lambda i: (-score[i], i))[:size]
                for i in selected:
                    store.acquire(panel.ids[int(i)], new_expert, origins)
                    remaining.remove(i)
                loss, mask = store.snapshot()
                t = time.perf_counter()
                # Refit only the appended factor from identical frozen base parameters.
                estimator = append_competence(
                    base["estimator"], z, loss, mask, base["scaling"], seed=seed
                )
                update_seconds += time.perf_counter() - t
                trace.append(
                    {
                        "round": step,
                        "new_rows": [panel.ids[int(i)] for i in selected],
                        "new_cells": int(mask[:, -1].sum()),
                    }
                )
                write_json(dest / "screening_trace.json", trace)
            mu, sd = estimator.predict(z)
            if strategy == "softimpute":
                from fame_scm.baselines.recovery import recover_static

                recovered, recovered_sd = recover_static(
                    "softimpute", z, loss, mask, base["scaling"], cfg, seed
                )
                # Existing columns are frozen; only the new column comes from recovery.
                mu[:, -1], sd[:, -1] = recovered[:, -1], recovered_sd[:, -1]
            tc = cfg["teacher"]
            teacher = competence_teacher(
                loss,
                mask,
                mu,
                sd,
                available,
                costs,
                **{k: tc[k] for k in ["tau", "eta", "nu"]},
            )
            args = build_router_training(panel, feature, store, teacher, cfg)
            t = time.perf_counter()
            router = append_router(base["router"], cfg, *args, costs, seed=seed)
            update_seconds += time.perf_counter() - t
            trials = []
            best = None
            threshold = None
            for th in cfg["router"]["calibration_thresholds"]:
                met, _, _ = evaluate_stage(
                    "calibration", panel, feature, store, router, cfg, th
                )
                score = (
                    met[cfg["evaluation"].get("primary_metric", "rmse")],
                    met["execution_count_mean"],
                    th,
                )
                trials.append({"threshold": th, **met})
                if best is None or score < best:
                    best = score
                    threshold = th
            met, records, routing = evaluate_stage(
                "test", panel, feature, store, router, cfg, threshold
            )
            summary = {
                "new_expert": new_expert,
                "strategy": strategy,
                "dataset": panel.audit.get("dataset", "snbc_format"),
                "n": len(panel.ids),
                "m": len(names),
                "result_kind": cfg["result_kind"],
                "evidence_mask_id": base.get("evidence_mask_id", 0),
                "neural_seed": base.get("neural_seed", 0),
                "requested_ratio": ratio,
                "new_cells": int(mask[:, -1].sum()),
                "eligible_new_cells": len(eligible),
                "actual_ratio": float(mask[:, -1].sum() / len(eligible)),
                "new_column_weighted_cost_ratio": float(
                    mask[:, -1].sum() / len(eligible)
                ),
                "cost_note": "constant per-expert cell weight; actual screening/calibration/test seconds reported separately",
                "update_seconds": update_seconds,
                "new_backbone_fit_seconds_shared": fit_seconds,
                "metrics": met,
                "new_expert_use_fraction": float(
                    np.mean([new_expert in r["selected_experts"] for r in routing])
                ),
                "executed_seconds": {
                    s: ledger.executed_seconds(s)
                    for s in ["mining", "calibration", "test"]
                },
                "old_column_cache_sha256": content_hash(old_loss.tolist()),
                "old_logits_and_parameters_frozen": True,
                "new_acquired_row_indices": np.flatnonzero(mask[:, -1]).tolist(),
                "evidence_account": store.evidence_account(),
            }
            if cfg["evaluation"].get("hidden_audit", False):
                from fame_scm.contracts import RequestKey
                from fame_scm.routing.gate import tkde_gate, hit_at_r
                from fame_scm.routing.training import router_probabilities

                dense = loss.copy()
                for i, j in zip(*np.where(available & ~mask)):
                    vals = [
                        store.fetch(
                            RequestKey(
                                "research_audit",
                                panel.ids[i],
                                names[j],
                                str(o),
                                configs[names[j]],
                                store.protocol_hash,
                            )
                        ).loss
                        for o in panel.origins["mining"]
                    ]
                    dense[i, j] = float(np.mean(vals))
                gate = tkde_gate(
                    router_probabilities(router, z),
                    available,
                    r=cfg["router"]["r"],
                    threshold=threshold,
                )
                summary["hit_at_2_validation"] = hit_at_r(
                    gate.top_r_before_pruning, dense, available
                )
                summary["research_audit_seconds_excluded"] = ledger.executed_seconds(
                    "research_audit"
                )
                summary["audit_scope"] = (
                    "hidden mining labels accessed only after acquisition, calibration and test predictions fixed"
                )
                np.save(dest / "dense_audit.npy", dense)
            pd.DataFrame(records).to_csv(dest / "predictions.csv", index=False)
            pd.DataFrame(ledger.events).to_csv(
                dest / "execution_ledger.csv", index=False
            )
            summary["evidence_account"] = store.evidence_account()
            summary["shared_source_evidence_account"] = base["store"].evidence_account()
            write_json(dest / "summary.json", summary)
            write_json(dest / "calibration_trials.json", trials)
            write_json(dest / "routing.json", routing)
            save_state(
                dest / "deployment.pkl",
                {
                    "config": cfg,
                    "features": feature,
                    "expert_pool": pool.deployment_copy(),
                    "router": router.cpu(),
                    "threshold": threshold,
                },
            )
            np.savez_compressed(
                dest / "competence.npz",
                measured=loss,
                observed=mask,
                predicted=mu,
                uncertainty=sd,
                teacher=teacher,
            )
            summaries.append(summary)
    # References are evaluated only for reporting; never drive acquisition/calibration.
    old_store.ledger.path = str(out / "old_reference_events.jsonl")
    old_metrics, old_records, _ = evaluate_stage(
        "test",
        panel,
        feature,
        old_store,
        base["router"],
        base["cfg"],
        base["threshold"],
    )
    pd.DataFrame(old_records).to_csv(out / "old_pool_predictions.csv", index=False)
    full = next((s for s in summaries if s["strategy"] == "full"), None)
    if full is not None:
        full_dest = out / f"full_{full['requested_ratio']:g}"
        frame = pd.read_csv(full_dest / "execution_ledger.csv")
        frame = frame[
            (frame.stage == "mining")
            & (frame.expert_id == new_expert)
            & ~frame.cache_hit
            & frame.success
        ]
        cell_weights = frame.groupby("entity_id").elapsed_seconds.sum().to_dict()
        weights = np.array([cell_weights.get(e, 0.0) for e in panel.ids])
        write_json(
            out / "full_column_reference_work.json",
            {
                "weights_seconds": cell_weights,
                "scope": "Full-column measured evaluator reference, used only after all selections are frozen",
            },
        )
        for s in summaries:
            s["new_column_weighted_cost_ratio"] = float(
                weights[s["new_acquired_row_indices"]].sum() / weights.sum()
            )
            s["cost_note"] = (
                "Post-selection full-column measured work reference; actual trajectory execution times separately reported"
            )
    for s in summaries:
        s["rmse_gap_percent_to_old"] = (
            100 * (s["metrics"]["rmse"] / old_metrics["rmse"] - 1)
            if old_metrics["rmse"]
            else None
        )
        s["rmse_gap_percent_to_full_inserted"] = (
            100 * (s["metrics"]["rmse"] / full["metrics"]["rmse"] - 1)
            if full and full["metrics"]["rmse"]
            else None
        )
        write_json(
            out / f"{s['strategy']}_{s['requested_ratio']:g}" / "summary.json", s
        )
    write_json(out / "insertion_results.json", summaries)
    write_json(out / "new_cost_profile.json", new_pool.cost_profile)
    return summaries
