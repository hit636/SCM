"""End-to-end train/mining/calibration/test orchestration with explicit records."""

from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import json, pickle, time, platform, importlib.metadata
import numpy as np
import pandas as pd
import torch
import yaml
from fame_scm.datasets.loader import load_panel
from fame_scm.orchestration.execution import (
    stamp,
    TimedProvider,
    build_router_training,
    evaluate_stage,
)
from fame_scm.datasets.retail import load_retail_panel
from fame_scm.datasets.temporal import timestamp_text
from fame_scm.evaluation.costs import profile_costs
from fame_scm.fingerprints.features import Forecastability, training_fallback_scales
from fame_scm.experts.forecast_experts import ExpertPool
from fame_scm.contracts import RequestKey, ForecastEvidence, CellKey
from fame_scm.acquisition.evidence import (
    MiningStore,
    AcquiredEvidenceView,
    weighted_work_ratio,
)
from fame_scm.competence.scaling import FrozenRowScaling
from fame_scm.acquisition.scores import (
    acquisition_scores,
    select_batch,
    partition_remainder,
    random_remainder,
)
from fame_scm.routing.teacher import competence_teacher
from fame_scm.routing.gate import tkde_gate, fuse, hit_at_r
from fame_scm.evaluation.metrics import forecast_metrics, full_observation_gap
from fame_scm.reporting import write_json, content_hash, check_pooled_metric_identity
from fame_scm.competence.trainers import SCMEnsemble
from fame_scm.routing.training import fit_router, router_probabilities
from fame_scm.baselines.recovery import (
    recover_static,
    GPLatentRecovery,
    expected_improvement,
)


def save_torch_states(estimator, path):
    if isinstance(estimator, SCMEnsemble):
        for j, model in enumerate(estimator.models):
            torch.save(model.state_dict(), path / f"competence_{j}.pt")


def _run_method(
    cfg, panel, pool, feature, method, mask_seed, neural_seed, out, resource_monitor
):
    cfg = deepcopy(cfg)
    started = stamp()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(panel.ids)
    names = pool.names
    m = len(names)
    available = pool.availability(panel.origins["mining"])
    full = method == "full_observation"
    single = method in names
    budget = (
        int(available.sum())
        if full
        else (0 if single else int(np.floor(cfg["scm"]["rho"] * available.sum())))
    )
    profile = pool.cost_profile
    costs = np.array([profile["online"][name] for name in names])
    offline_raw = np.array([profile["offline"][name] for name in names])
    offline = offline_raw / offline_raw[names.index(cfg["experts"]["anchor"])]
    protocol_hash = content_hash(
        {
            "input": panel.audit["sha256"],
            "ids": panel.ids,
            "origins": panel.origins,
            "forecast": cfg["forecast"],
            "missing": cfg["data"]["missing_dates"],
            "availability": content_hash(available.tolist()),
        }
    )
    configs = {
        name: content_hash(
            {"expert": name, "config": cfg["experts"], "train_seed": cfg["seed"]}
        )
        for name in names
    }
    store = MiningStore(
        panel.ids,
        names,
        available,
        budget,
        TimedProvider(panel, pool),
        configs,
        protocol_hash,
        construction_request_budget=cfg.get("budgets", {}).get("construction_requests"),
    )
    store.ledger.path = str(out / "execution_events.jsonl")
    rng = np.random.default_rng(cfg["seed"] + mask_seed)
    seed = cfg["seed"] + 10000 * mask_seed + neural_seed
    origins = [str(x) for x in panel.origins["mining"]]
    anchor = names.index(cfg["experts"]["anchor"])
    z = feature.transform(panel, panel.stages["train"].end)
    trace = []
    training = []
    estimator = None
    teacher = None
    router = None
    fit_seconds = 0.0
    scan_seconds = 0.0
    seed_cells = 0
    router_seconds = 0.0
    row_order = rng.permutation(n)
    if not available[:, anchor].all():
        raise ValueError(
            "Mining cohort contains non-executable anchors; adjust train-only cohort/history protocol"
        )
    if not single:
        seeds = [(i, anchor) for i in range(n)]
        for j in range(m):
            candidates = np.flatnonzero(available[:, j])
            if j != anchor and len(candidates):
                seeds.append((int(rng.choice(candidates)), j))
        if budget < len(seeds):
            raise ValueError(
                f"Budget {budget} < charged initialization {len(seeds)}; increase rho or N"
            )
        for i, j in seeds:
            store.acquire(panel.ids[i], names[j], origins)
        seed_cells = len(seeds)
        initial_loss, initial_mask = store.snapshot()
        fallback, fallback_spec = training_fallback_scales(panel, feature, cfg)
        scaling = FrozenRowScaling.from_seed(
            initial_loss,
            initial_mask,
            fallback,
            fallback,
            floor=cfg["scm"]["scale_floor"],
            mad_multiplier=cfg["scm"]["mad_multiplier"],
        )
        if not cfg["scm"].get("row_normalization", True):
            scaling = FrozenRowScaling(np.zeros(n), np.ones(n))
        if cfg["scm"].get("rank_margin_mode", "fixed") == "acquired_rolling":
            from fame_scm.competence.calibration import estimate_rank_margin

            cfg["scm"]["rank_margin"], margin_record = estimate_rank_margin(
                store, scaling, fallback=cfg["scm"]["rank_margin"]
            )
        else:
            margin_record = {
                "value": cfg["scm"]["rank_margin"],
                "definition": "configuration-fixed normalized margin",
            }
        write_json(out / "rank_margin.json", margin_record)
        write_json(
            out / "scaling.json",
            {
                "center": scaling.center.tolist(),
                "scale": scaling.scale.tolist(),
                "fallback": fallback_spec,
            },
        )
        write_json(
            out / "initialization.json",
            {
                "cells": [[panel.ids[i], names[j]] for i, j in seeds],
                "policy": "anchor every row; seed each other executable expert on a seeded random eligible row",
            },
        )
        batches = partition_remainder(budget, seed_cells, cfg["scm"]["batches"])
        for round_id, size in enumerate(batches):
            if size == 0:
                continue
            loss, mask = store.snapshot()
            t = stamp()
            if method in ["fame_scm_active", "nn_ei_c", "pmf_a"]:
                fit_cfg = deepcopy(cfg)
                if method == "nn_ei_c":
                    fit_cfg["scm"]["lambda_rank"] = 0.0
                estimator = (
                    SCMEnsemble(fit_cfg, seed)
                    if method in ["fame_scm_active", "nn_ei_c"]
                    else GPLatentRecovery(cfg, seed)
                )
                estimator.fit(z, loss, mask, scaling)
                mu, sd = estimator.predict(z)
                fit_seconds += stamp() - t
                t = stamp()
                score = (
                    acquisition_scores(
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
                    )
                    if method == "fame_scm_active"
                    else expected_improvement(
                        mu,
                        sd,
                        loss,
                        mask,
                        available,
                        offline if method == "nn_ei_c" else None,
                    )
                )
                chosen = select_batch(score, size)
            elif method == "fformpp_b":
                # Complete-row acquisition: fill rows without exceeding the total
                # budget. Residual partial evidence is charged but not fit labels.
                candidates = [
                    (i, j)
                    for i in row_order
                    for j in range(m)
                    if available[i, j] and not mask[i, j]
                ]
                chosen = candidates[:size]
            else:
                chosen = random_remainder(mask, available, size, rng)
            scan_seconds += stamp() - t
            for i, j in chosen:
                store.acquire(panel.ids[i], names[j], origins)
            trace.append(
                {
                    "round": round_id,
                    "batch_cells": len(chosen),
                    "acquired_cells": len(store.observed_cells()),
                    "cells": [[panel.ids[i], names[j]] for i, j in chosen],
                }
            )
        loss, mask = store.snapshot()
        t = stamp()
        if full:
            mu = loss.copy()
            sd = np.zeros_like(loss)
        elif method in ["fame_scm_active", "scm_passive", "nn_ei_c"]:
            fit_cfg = deepcopy(cfg)
            if method == "nn_ei_c":
                fit_cfg["scm"]["lambda_rank"] = 0.0
            estimator = SCMEnsemble(fit_cfg, seed).fit(z, loss, mask, scaling)
            mu, sd = estimator.predict(z)
            training = estimator.logs
        elif method == "pmf_a":
            estimator = GPLatentRecovery(cfg, seed).fit(z, loss, mask, scaling)
            mu, sd = estimator.predict(z)
        elif method == "independent_neural":
            from fame_scm.baselines.neural import independent_neural

            mu, sd = independent_neural(z, loss, mask, scaling, cfg, seed)
        else:
            mu, sd = recover_static(method, z, loss, mask, scaling, cfg, seed)
        fit_seconds += stamp() - t
        tc = cfg["teacher"]
        teacher = competence_teacher(
            loss,
            mask,
            mu,
            sd,
            available,
            costs,
            tau=tc["tau"],
            eta=tc["eta"],
            nu=tc["nu"],
        )
        args = build_router_training(panel, feature, store, teacher, cfg)
        t = stamp()
        if cfg["router"].get("consumer", "distilled") == "direct_teacher":
            from fame_scm.routing.consumers import DirectTeacher

            router = DirectTeacher(
                estimator,
                scaling,
                loss,
                mask,
                available,
                costs,
                cfg["teacher"],
                entities=panel.ids,
            )
            router_history = []
        else:
            router, router_history = fit_router(cfg, *args, costs, seed)
        router_seconds = stamp() - t
        pd.DataFrame(router_history).to_csv(out / "router_training.csv", index=False)
        torch.save(router.state_dict(), out / "router.pt")
        save_torch_states(estimator, out)
        np.savez_compressed(
            out / "competence.npz",
            availability=available,
            measured=loss,
            observed=mask,
            predicted=mu,
            uncertainty=sd,
            teacher=teacher,
        )
        with (out / "deployment.pkl").open("wb") as file:
            pickle.dump(
                {
                    "data_scalers": panel.audit.get("scalers"),
                    "config": cfg,
                    "features": feature,
                    "expert_pool": pool.deployment_copy(),
                    "router": router.to("cpu"),
                    "threshold": cfg["router"]["threshold"],
                },
                file,
            )
        router.to(cfg["device"])
    mining_seconds = stamp() - started - router_seconds
    # Calibration charges stage-specific requests and selects the configured primary metric.
    t = stamp()
    trials = []
    best_threshold = cfg["router"]["threshold"]
    best = None
    thresholds = [best_threshold] if single else cfg["router"]["calibration_thresholds"]
    for threshold in thresholds:
        metrics, _, _ = evaluate_stage(
            "calibration",
            panel,
            feature,
            store,
            router,
            cfg,
            threshold,
            fixed_expert=method if single else None,
        )
        trials.append(
            {
                "threshold": threshold,
                "rmse": metrics["rmse"],
                "selection_metric": cfg["evaluation"].get("primary_metric", "rmse"),
                "selection_value": metrics[
                    cfg["evaluation"].get("primary_metric", "rmse")
                ],
                "execution_count_mean": metrics["execution_count_mean"],
            }
        )
        # Equal error favors fewer executions; exact remaining ties use lower threshold.
        score = (
            metrics[cfg["evaluation"].get("primary_metric", "rmse")],
            metrics["execution_count_mean"],
            threshold,
        )
        if score[0] is None:
            raise ValueError("Undefined calibration metric; cannot select threshold")
        if best is None or score < best:
            best = score
            best_threshold = threshold
    calibration_seconds = stamp() - t
    if not single:
        # Update deployment with the selected calibration result.
        with (out / "deployment.pkl").open("rb") as file:
            bundle = pickle.load(file)
        bundle["threshold"] = best_threshold
        with (out / "deployment.pkl").open("wb") as file:
            pickle.dump(bundle, file)
    if cfg.get("research", {}).get("save_state") or cfg.get("research", {}).get(
        "build_only"
    ):
        from fame_scm.checkpoints import save_state

        # Write only after calibration; an interrupted build is never selectable.
        save_state(
            out / "build_state.pkl",
            {
                "cfg": cfg,
                "panel": panel,
                "pool": pool,
                "feature": feature,
                "store": store,
                "estimator": estimator,
                "router": router,
                "scaling": scaling if not single else None,
                "teacher": teacher,
                "z": z,
                "threshold": best_threshold,
                "method": method,
                "evidence_mask_id": mask_seed,
                "neural_seed": neural_seed,
                "completed_at": time.time(),
                "mining_seconds": mining_seconds,
                "router_seconds": router_seconds,
                "calibration_seconds": calibration_seconds,
            },
        )
    if cfg.get("research", {}).get("build_only"):
        loss, mask = store.snapshot()
        summary = {
            "method": method,
            "acquired_cells": int(mask.sum()),
            "cell_budget": budget,
            "selected_threshold": best_threshold,
            "status": "build_complete_test_not_accessed",
            "build_seconds": stamp() - started,
            "resources": resource_monitor.finish(),
        }
        write_json(out / "build_summary.json", summary)
        pd.DataFrame(store.ledger.events).to_csv(
            out / "execution_ledger.csv", index=False
        )
        return summary
    t = stamp()
    metrics, records, routing = evaluate_stage(
        "test",
        panel,
        feature,
        store,
        router,
        cfg,
        best_threshold,
        fixed_expert=method if single else None,
    )
    test_seconds = stamp() - t
    loss, mask = store.snapshot()
    diagnostics = {}
    measured_cost_matrix = None
    if cfg["evaluation"]["hidden_audit"] and not single:
        dense = np.full((n, m), np.nan)
        measured_cost_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if not available[i, j]:
                    continue
                if mask[i, j]:
                    dense[i, j] = loss[i, j]
                    measured_cost_matrix[i, j] = sum(
                        store.cache[k].elapsed_seconds
                        for k in store._cells[CellKey(panel.ids[i], names[j])].requests
                    )
                else:
                    values = []
                    for origin in origins:
                        key = RequestKey(
                            "research_audit",
                            panel.ids[i],
                            names[j],
                            origin,
                            configs[names[j]],
                            protocol_hash,
                        )
                        evidence = store.fetch(key)
                        values.append(evidence.loss)
                        measured_cost_matrix[i, j] += evidence.elapsed_seconds
                    dense[i, j] = np.mean(values)
        p = router_probabilities(router, z)
        gate = tkde_gate(p, available, r=cfg["router"]["r"], threshold=best_threshold)
        hidden = available & ~mask
        diagnostics = {
            "hit_at_2_validation": hit_at_r(
                gate.top_r_before_pruning, dense, available
            ),
            "recovery_mae_hidden": (
                float(np.mean(abs(mu[hidden] - dense[hidden])))
                if hidden.any()
                else None
            ),
            "component_loss_regret": float(
                np.mean(
                    np.sum(gate.weights * np.nan_to_num(dense), axis=1)
                    - np.where(available, dense, np.inf).min(1)
                )
            ),
            "teacher_kl": float(
                np.mean(
                    np.sum(teacher * np.log((teacher + 1e-12) / (p + 1e-12)), axis=1)
                )
            ),
            "singleton_fraction": float(np.mean(mask.sum(1) == 1)),
            "audit_scope": "hidden mining losses requested only after model/configuration selection and test predictions are frozen",
        }
        from fame_scm.evaluation.theory import stability_diagnostics

        theory_rows, theory_summary = stability_diagnostics(
            dense,
            mu,
            teacher,
            p,
            available,
            gate.retained,
            tau=cfg["teacher"]["tau"],
            bound=cfg["evaluation"].get("theory_bound"),
        )
        pd.DataFrame(theory_rows).to_csv(out / "stability_terms.csv", index=False)
        write_json(out / "stability_summary.json", theory_summary)
        pd.DataFrame(
            {
                "entity_id": panel.ids,
                "observed_experts": mask.sum(1),
                "recovered_experts": (available & ~mask).sum(1),
                "retained_mass": (p * gate.retained).sum(1),
            }
        ).to_csv(out / "row_diagnostics.csv", index=False)
        np.save(out / "evaluator_validation_losses.npy", dense)
        np.save(out / "evaluator_cell_work_seconds.npy", measured_cost_matrix)
    event_frame = pd.DataFrame(store.ledger.events)
    executed = event_frame[~event_frame.cache_hit & event_frame.success]
    stage_cells = {
        stage: int(
            executed[executed.stage == stage][["entity_id", "expert_id"]]
            .drop_duplicates()
            .shape[0]
        )
        for stage in ["mining", "calibration", "test", "research_audit"]
    }
    construction_pairs = int(
        executed[executed.stage.isin(["mining", "calibration"])][
            ["entity_id", "expert_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )
    summary = {
        "method": method,
        "result_kind": cfg["result_kind"],
        "evidence_mask_id": mask_seed,
        "neural_seed": neural_seed,
        "dataset": panel.audit.get("dataset", "snbc_format"),
        "n": n,
        "m": m,
        "horizon": cfg["forecast"]["horizon"],
        "gate_version": "tkde_eq17",
        "protocol_hash": protocol_hash,
        "expert_names": names,
        "seed_cells": seed_cells,
        "cell_budget": budget,
        "acquired_cells": int(mask.sum()),
        "actual_rho": float(mask.sum() / available.sum()) if not single else None,
        "weighted_cost_ratio": (
            weighted_work_ratio(
                mask,
                available,
                (
                    measured_cost_matrix
                    if cfg["evaluation"].get("cost_reference") == "measured_audit"
                    and measured_cost_matrix is not None
                    else offline_raw
                ),
            )
            if not single
            else None
        ),
        "cost_profile_kind": profile["kind"],
        "weighted_cost_reference": (
            "measured_selected_and_hidden_audit_work"
            if cfg["evaluation"].get("cost_reference") == "measured_audit"
            and measured_cost_matrix is not None
            else "frozen_training_profile_per_expert"
        ),
        "resources": resource_monitor.finish(),
        "evidence_account": store.evidence_account(),
        "router_consumer": cfg["router"].get("consumer", "distilled"),
        "router_operator": cfg["router"].get("operator", "soft_kl"),
        "competence_label_metric": cfg["scm"].get("competence_loss", "rmse"),
        "observed_mask_hash": content_hash(mask.tolist()),
        "evidence_cells_by_stage": stage_cells,
        "construction_distinct_series_expert_pairs": construction_pairs,
        "selected_threshold": best_threshold,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "seconds": {
            "mining": mining_seconds,
            "recovery_fit": fit_seconds,
            "candidate_scan": scan_seconds,
            "router": router_seconds,
            "calibration": calibration_seconds,
            "build": mining_seconds + router_seconds + calibration_seconds,
            "test": test_seconds,
            "expert_mining": store.ledger.executed_seconds("mining"),
            "expert_calibration": store.ledger.executed_seconds("calibration"),
            "expert_test": store.ledger.executed_seconds("test"),
            "research_audit": store.ledger.executed_seconds("research_audit"),
        },
        "execution_requests": {
            s: int(
                sum(e["stage"] == s and not e["cache_hit"] for e in store.ledger.events)
            )
            for s in ["mining", "calibration", "test", "research_audit"]
        },
    }
    write_json(out / "summary.json", summary)
    write_json(out / "routing.json", routing)
    write_json(out / "acquisition_trace.json", trace)
    write_json(out / "calibration_trials.json", trials)
    pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
    event_frame.to_csv(out / "execution_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "entity": v.key.entity_id,
                "expert": v.key.expert_id,
                "loss": v.loss,
                "origins": [k.origin for k in v.requests],
            }
            for v in store.observed_cells()
        ]
    ).to_csv(out / "evidence_cells.csv", index=False)
    if training:
        pd.DataFrame(training).to_csv(out / "competence_training.csv", index=False)
    return summary


def run_method(cfg, panel, pool, feature, method, mask_seed, neural_seed, out):
    from fame_scm.evaluation.resources import ResourceMonitor

    monitor = ResourceMonitor().start()
    try:
        return _run_method(
            cfg, panel, pool, feature, method, mask_seed, neural_seed, out, monitor
        )
    except BaseException as error:
        resources = monitor.finish()
        write_json(
            Path(out) / "failed_run.json",
            {
                "method": method,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "resources": resources,
                "scope": "Failed/incomplete runs are not valid experiment results; partial work remains charged in execution_events.jsonl",
            },
        )
        raise


def run_experiment(cfg, *, methods=None):
    import random, os
    from fame_scm.devices import device_report, prepare_device

    cfg = deepcopy(cfg)
    cfg["device"] = str(prepare_device(cfg["device"]))
    random.seed(cfg["seed"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(cfg["threads"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    # Do not overwrite an existing set of results by accident.
    if any(out.glob("*/mask_*/seed_*")) or (out / "results.csv").exists():
        raise ValueError(
            f"Results already exist at {out}; choose a fresh --out directory"
        )
    (out / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    t = stamp()
    panel = load_panel(cfg)
    required_count = cfg["data"].get("required_series_count")
    if required_count is not None and len(panel.ids) != required_count:
        raise ValueError(
            f"Requested exact cohort of {required_count} series, got {len(panel.ids)}"
        )
    from fame_scm.reproduction import validate_protocol

    protocol = validate_protocol(cfg, panel)
    write_json(out / "reproduction_protocol.json", protocol)
    feature = Forecastability(cfg).fit(panel)
    feature_seconds = stamp() - t
    write_json(out / "data_audit.json", panel.audit)
    write_json(out / "features.json", feature.state())
    write_json(out / "cohort.json", {"ids": panel.ids, "hash": content_hash(panel.ids)})
    print(
        f'Data: {len(panel.ids)} series, {len(panel.dates)} time steps, {len(pool_names:=cfg["experts"]["names"])} experts',
        flush=True,
    )
    t = stamp()
    pool = ExpertPool(cfg, panel).fit()
    expert_fit_seconds = stamp() - t
    pool.save(out / "expert_checkpoints")
    pool.cost_profile = profile_costs(pool, panel, cfg)
    write_json(out / "cost_profile.json", pool.cost_profile)
    (out / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"Expert fitting completed in {expert_fit_seconds:.2f}s", flush=True)
    summaries = []
    selected = methods or cfg["baselines"]["methods"]
    for mask_seed in cfg["evaluation"]["mask_seeds"]:
        for neural_seed in cfg["evaluation"]["neural_seeds"]:
            for method in selected:
                print(
                    f"Running {method} mask={mask_seed} seed={neural_seed}", flush=True
                )
                result = run_method(
                    cfg,
                    panel,
                    pool,
                    feature,
                    method,
                    mask_seed,
                    neural_seed,
                    out / method / f"mask_{mask_seed}" / f"seed_{neural_seed}",
                )
                summaries.append(result)
                print(
                    f'  RMSE={result["metrics"]["rmse"]:.6f}, MAE={result["metrics"]["mae"]:.6f}, cells={result["acquired_cells"]}',
                    flush=True,
                )
    rows = []
    for result in summaries:
        reference = next(
            (
                x
                for x in summaries
                if x["method"] == "full_observation"
                and x["evidence_mask_id"] == result["evidence_mask_id"]
                and x["neural_seed"] == result["neural_seed"]
            ),
            None,
        )
        rows.append(
            {
                "method": result["method"],
                "mask": result["evidence_mask_id"],
                "seed": result["neural_seed"],
                "result_kind": result["result_kind"],
                "metric_scale": panel.audit.get("metric_scale", "raw"),
                **result["metrics"],
                "actual_rho": result["actual_rho"],
                "weighted_cost_ratio": result["weighted_cost_ratio"],
                "build_seconds": result["seconds"]["build"],
                "rmse_gap_to_full": (
                    full_observation_gap(
                        result["metrics"]["rmse"], reference["metrics"]["rmse"]
                    )
                    if reference and reference["metrics"]["rmse"] > 0
                    else None
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "results.csv", index=False)
    # Within-mask mean over neural-seed metrics, then SD across mask means.
    metric_cols = [
        "rmse",
        "mae",
        "mse",
        "wape",
        "smape",
        "weighted_cost_ratio",
        "build_seconds",
        "rmse_gap_to_full",
    ]
    mask_means = table.groupby(["method", "mask"])[metric_cols].mean()
    aggregate = mask_means.groupby("method").agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate.to_csv(out / "aggregate.csv")
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "pandas",
            "torch",
            "lightgbm",
            "statsmodels",
            "scikit-learn",
        ]
    }
    write_json(
        out / "run_manifest.json",
        {
            "result_kind": cfg["result_kind"],
            "python": platform.python_version(),
            "versions": versions,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": device_report(cfg["device"]),
                "threads": cfg["threads"],
            },
            "data_sha256": panel.audit["sha256"],
            "config_hash": content_hash(cfg),
            "feature_seconds": feature_seconds,
            "code_hash": content_hash(
                {
                    str(p.relative_to(Path(__file__).parents[1])): __import__("hashlib")
                    .sha256(p.read_bytes())
                    .hexdigest()
                    for p in sorted(Path(__file__).parents[1].rglob("*.py"))
                }
            ),
            "cost_profile_seconds": pool.cost_profile["seconds"],
            "shared_expert_fit_seconds": expert_fit_seconds,
            "statistical_unit": "mask means after averaging neural-seed metrics",
            "cost_boundary": "shared expert/feature preparation separate; mining/router/calibration included in build; test/audit separate",
            "reproduction_protocol": protocol,
        },
    )
    return table
