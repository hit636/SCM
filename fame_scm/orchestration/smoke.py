"""Synthetic kernel integration only; no paper performance claims."""

from dataclasses import asdict
from pathlib import Path
import csv
import numpy as np
from fame_scm.contracts import ForecastEvidence, RequestKey
from fame_scm.acquisition.evidence import (
    MiningStore,
    AcquiredEvidenceView,
    weighted_work_ratio,
)
from fame_scm.acquisition.scores import (
    partition_remainder,
    acquisition_scores,
    select_batch,
)
from fame_scm.competence.scaling import FrozenRowScaling
from fame_scm.routing.teacher import competence_teacher
from fame_scm.routing.gate import tkde_gate, fuse, hit_at_r
from fame_scm.evaluation.metrics import forecast_metrics
from fame_scm.reporting import write_json, check_pooled_metric_identity, content_hash


def run_smoke(out):
    rng = np.random.default_rng(20261004)  # engineering seed, not an author seed
    n, m, horizon = 24, 5, 7
    entities, experts = [f"entity_{i}" for i in range(n)], [
        f"expert_{j}" for j in range(m)
    ]
    avail = np.ones((n, m), bool)
    costs = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    y_mining = rng.poisson(5, (n, horizon)).astype(float)
    y_test = rng.poisson(5, (n, horizon)).astype(float)
    valid = np.ones_like(y_test, bool)
    valid[0, 0] = False
    # Evaluator/provider-owned tensors. Never pass these to recovery/acquisition.
    private_mining = np.maximum(
        0.0,
        y_mining[:, None, :]
        + rng.normal(0, 1, (n, m, horizon)) * np.arange(1, m + 1)[None, :, None],
    )
    private_test = np.maximum(
        0.0,
        y_test[:, None, :]
        + rng.normal(0, 1, (n, m, horizon)) * np.arange(1, m + 1)[None, :, None],
    )
    private_loss = np.sqrt(
        np.mean((private_mining - y_mining[:, None, :]) ** 2, axis=2)
    )
    ei, mi = {x: i for i, x in enumerate(entities)}, {
        x: i for i, x in enumerate(experts)
    }

    def provider(key):
        i, j = ei[key.entity_id], mi[key.expert_id]
        if key.stage == "mining":
            pred, loss = private_mining[i, j], private_loss[i, j]
        elif key.stage == "test":
            pred = private_test[i, j]
            loss = np.sqrt(np.mean((pred - y_test[i]) ** 2))
        else:
            raise ValueError("Synthetic demo defines mining/test only")
        # Explicit simulated expert time: tests ledger semantics, not real runtime.
        return ForecastEvidence(key, float(loss), pred.copy(), float(costs[j] / 1000))

    budget = int(0.4 * avail.sum())
    hashes = {x: content_hash({"expert": x, "demo": True}) for x in experts}
    store = MiningStore(
        entities,
        experts,
        avail,
        budget,
        provider,
        hashes,
        content_hash({"dataset": "synthetic_kernel_demo", "h": horizon}),
    )
    # Synthetic seed design: all row anchors + one cell per remaining expert.
    for entity in entities:
        store.acquire(entity, experts[0], ["origin_mining"])
    for j in range(1, m):
        store.acquire(entities[j], experts[j], ["origin_mining"])
    learner_view = AcquiredEvidenceView(store)
    losses, observed = learner_view.snapshot()
    scaling = FrozenRowScaling.from_seed(
        losses, observed, 0.0, 1.0, floor=1e-6, mad_multiplier=1.0
    )

    def demo_recover(view):
        acquired, mask = view.snapshot()
        # A transparent surrogate exercises interfaces; no neural SCM claim.
        mean = np.nanmean(acquired, axis=0)
        sd = np.array([np.std(acquired[mask[:, j], j]) + 0.1 for j in range(m)])
        return np.broadcast_to(mean, (n, m)).copy(), np.broadcast_to(sd, (n, m)).copy()

    batches = partition_remainder(budget, len(store.observed_cells()), 4)
    for size in batches:
        predicted, sd = demo_recover(learner_view)
        acquired, mask = learner_view.snapshot()
        scores = acquisition_scores(
            predicted, sd, acquired, mask, avail, costs / costs.mean()
        )
        for i, j in select_batch(scores, size):
            store.acquire(entities[i], experts[j], ["origin_mining"])
    measured, observed = learner_view.snapshot()
    predicted, sd = demo_recover(learner_view)
    teacher = competence_teacher(measured, observed, predicted, sd, avail, costs)
    # Direct teacher consumer here; a trained router is not simulated.
    gate = tkde_gate(teacher, avail)
    selected = np.full((n, m, horizon), np.nan)
    for i, j in zip(*np.where(gate.retained)):
        key = RequestKey(
            "test",
            entities[i],
            experts[j],
            "origin_test",
            hashes[experts[j]],
            store.protocol_hash,
        )
        selected[i, j] = store.fetch(key).prediction
    fused = fuse(gate.weights, selected)
    metrics = forecast_metrics(y_test, fused, valid)
    check_pooled_metric_identity(metrics)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "synthetic_kernel_demo",
        "method": "kernel_demo_column_mean_direct_teacher",
        "result_kind": "synthetic_smoke",
        "is_paper_reproduction": False,
        "n": n,
        "m": m,
        "horizon": horizon,
        "seed_cells": n + m - 1,
        "cell_budget": budget,
        "acquired_cells": int(observed.sum()),
        "rho": float(observed.sum() / avail.sum()),
        "remainder_batches": batches,
        "weighted_work_ratio": weighted_work_ratio(observed, avail, costs),
        "gate_version": "tkde_eq17",
        "execution_count_mean": float(gate.retained.sum(1).mean()),
        "hit_at_2_validation_evaluator_only": hit_at_r(
            gate.top_r_before_pruning, private_loss, avail
        ),
        "metrics": metrics,
        "expert_time_kind": "simulated_for_ledger_test",
        "mining_expert_seconds_simulated": store.ledger.executed_seconds("mining"),
        "test_expert_seconds_simulated": store.ledger.executed_seconds("test"),
        "normalized_roundtrip_max_error": float(
            np.nanmax(np.abs(scaling.restore(scaling.normalize(measured)) - measured))
        ),
    }
    write_json(out / "summary.json", summary)
    write_json(out / "execution_ledger.json", store.ledger.events)
    write_json(
        out / "evidence_manifest.json",
        [
            {
                "entity": x.key.entity_id,
                "expert": x.key.expert_id,
                "loss": x.loss,
                "request_keys": [asdict(k) for k in x.requests],
            }
            for x in store.observed_cells()
        ],
    )
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "entity_id",
                "origin",
                "horizon_step",
                "target",
                "prediction",
                "valid",
                "weight",
            ]
        )
        for i in range(n):
            for h in range(horizon):
                writer.writerow(
                    [
                        entities[i],
                        "origin_test",
                        h + 1,
                        y_test[i, h],
                        fused[i, h],
                        bool(valid[i, h]),
                        1.0,
                    ]
                )
    return summary
