"""Forecast requests, cached router supervision, and stage evaluation."""

import time
import numpy as np
import torch
from fame_scm.contracts import RequestKey, ForecastEvidence
from fame_scm.datasets.temporal import timestamp_text
from fame_scm.evaluation.metrics import forecast_metrics
from fame_scm.reporting import check_pooled_metric_identity
from fame_scm.routing.gate import tkde_gate, fuse
from fame_scm.routing.training import router_probabilities


def stamp():
    return time.perf_counter()


class TimedProvider:
    def __init__(self, panel, pool):
        self.panel = panel
        self.pool = pool
        self.ids = {x: i for i, x in enumerate(panel.ids)}

    def __call__(self, key):
        i = self.ids[key.entity_id]
        origin = int(key.origin)
        expected = "mining" if key.stage == "research_audit" else key.stage
        if origin not in self.panel.origins[expected]:
            raise ValueError("Request origin outside its stage")
        if self.pool.device.type == "cuda":
            torch.cuda.synchronize(self.pool.device)
        started = stamp()
        pred = self.pool.predict(key.expert_id, i, origin)
        if self.pool.device.type == "cuda":
            torch.cuda.synchronize(self.pool.device)
        elapsed = stamp() - started
        y, valid = self.panel.target(i, origin, self.pool.h)
        metric = self.pool.cfg.get("scm", {}).get("competence_loss", "rmse")
        loss = forecast_metrics(y, pred, valid)[metric]
        if loss is None:
            raise ValueError(f"Undefined competence label: {metric}")
        return ForecastEvidence(key, loss, pred, elapsed)


def build_router_training(panel, feature, store, teacher, cfg):
    xs = []
    qs = []
    preds = []
    observed = []
    ys = []
    masks = []
    loss, mask = store.snapshot()
    n, m = mask.shape
    h = cfg["forecast"]["horizon"]
    for origin in panel.origins["mining"]:
        z = feature.transform(panel, origin)
        for i in range(n):
            cache = np.full((m, h), np.nan)
            for j in np.flatnonzero(mask[i]):
                key = RequestKey(
                    "mining",
                    panel.ids[i],
                    store.experts[j],
                    str(origin),
                    store.config_hashes[store.experts[j]],
                    store.protocol_hash,
                )
                # Direct acquired cache read: router fitting executes zero experts.
                cache[j] = store.cache[key].prediction
            y, valid = panel.target(i, origin, h)
            xs.append(z[i])
            qs.append(teacher[i])
            preds.append(cache)
            observed.append(mask[i])
            ys.append(y)
            masks.append(valid)
    return [np.asarray(x) for x in [xs, qs, preds, observed, ys, masks]]


def evaluate_stage(
    stage, panel, feature, store, router, cfg, threshold, *, fixed_expert=None
):
    records = []
    routing = []
    targets = []
    predictions = []
    valids = []
    counts = []
    selected_missing = []
    retained_mass = []
    online_work = []
    _, observed = store.snapshot()
    m = len(store.experts)
    h = cfg["forecast"]["horizon"]
    for origin in panel.origins[stage]:
        z = feature.transform(panel, origin)
        if fixed_expert is not None:
            p = np.zeros((len(panel.ids), m))
            p[:, store.experts.index(fixed_expert)] = 1.0
            gate = tkde_gate(
                p, store.provider.pool.availability([origin]), r=1, threshold=threshold
            )
        else:
            p = router_probabilities(router, z)
            gate = tkde_gate(
                p,
                store.provider.pool.availability([origin]),
                r=cfg["router"]["r"],
                threshold=threshold,
            )
        forecasts = np.full((len(panel.ids), m, h), np.nan)
        for i, j in zip(*np.where(gate.retained)):
            key = RequestKey(
                stage,
                panel.ids[i],
                store.experts[j],
                str(origin),
                store.config_hashes[store.experts[j]],
                store.protocol_hash,
            )
            forecasts[i, j] = store.fetch(key).prediction
        combined = fuse(gate.weights, forecasts)
        for i, entity in enumerate(panel.ids):
            y, valid = panel.target(i, origin, h)
            targets.append(y)
            predictions.append(combined[i])
            valids.append(valid)
            online_work.append(
                float(
                    sum(
                        store.provider.pool.cost_profile["online"][store.experts[j]]
                        for j in np.flatnonzero(gate.retained[i])
                    )
                )
            )
            counts.append(int(gate.retained[i].sum()))
            selected_missing.extend((~observed[i, gate.retained[i]]).tolist())
            retained_mass.append(
                float(np.sum(gate.available_probability[i, gate.retained[i]]))
            )
            routing.append(
                {
                    "entity_id": entity,
                    "origin": timestamp_text(panel.dates[origin], cfg),
                    "probabilities": p[i].tolist(),
                    "top_r_before_pruning": np.flatnonzero(
                        gate.top_r_before_pruning[i]
                    ).tolist(),
                    "selected_experts": [
                        store.experts[j] for j in np.flatnonzero(gate.retained[i])
                    ],
                    "weights": gate.weights[i].tolist(),
                    "retained_probability_mass": retained_mass[-1],
                }
            )
            for step in range(h):
                records.append(
                    {
                        "entity_id": entity,
                        "origin": timestamp_text(panel.dates[origin], cfg),
                        "horizon_step": step + 1,
                        "target_date": timestamp_text(
                            panel.dates[origin + step + 1], cfg
                        ),
                        "target": float(y[step]),
                        "prediction": float(combined[i, step]),
                        "valid": bool(valid[step]),
                        "weight": 1.0,
                    }
                )
    metrics = forecast_metrics(
        np.asarray(targets), np.asarray(predictions), np.asarray(valids)
    )
    check_pooled_metric_identity(metrics)
    metrics.update(
        {
            "execution_count_mean": float(np.mean(counts)),
            "online_execution_cost_mean": float(np.mean(online_work)),
            "selected_unobserved_fraction": float(np.mean(selected_missing)),
            "retained_probability_mass_mean": float(np.mean(retained_mass)),
        }
    )
    return metrics, records, routing
