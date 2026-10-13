"""Paired two-way cluster bootstrap on exported predictions, never averaged forecasts."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from fame_scm.reporting import write_json

KEY = ["entity_id", "origin", "horizon_step", "target_date"]


def paired_bootstrap(
    root,
    method,
    reference,
    *,
    metric="wape",
    replicates=10000,
    seed=42,
    block_days=28,
    group_component=None,
):
    if metric not in ["rmse", "mse", "mae", "wape", "smape"]:
        raise ValueError("Unsupported metric")
    if replicates < 2 or block_days < 1:
        raise ValueError("Invalid bootstrap parameters")
    root = Path(root)
    pairs = []
    base_keys = None
    layout = None
    files = sorted((root / method).glob("mask_*/seed_*/predictions.csv"))
    reference_files = sorted((root / reference).glob("mask_*/seed_*/predictions.csv"))
    if not files or {p.relative_to(root / method) for p in files} != {
        p.relative_to(root / reference) for p in reference_files
    }:
        raise ValueError("Both methods must have exactly the same mask/seed runs")
    for left in files:
        right = root / reference / left.relative_to(root / method)
        a = pd.read_csv(left).sort_values(KEY).reset_index(drop=True)
        b = pd.read_csv(right).sort_values(KEY).reset_index(drop=True)
        if a.duplicated(KEY).any() or b.duplicated(KEY).any():
            raise ValueError("Duplicate forecast key")
        if not a[KEY].equals(b[KEY]):
            raise ValueError("Nonpaired prediction keys")
        for col in ["target", "valid", "weight"]:
            if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
                raise ValueError(f"Nonpaired {col}")
        if base_keys is None:
            base_keys = a[KEY].copy()
            layout = a
        elif not base_keys.equals(a[KEY]):
            raise ValueError("Different targets across seeds")
        pairs.append(
            (
                left.parent.parent.name,
                left.parent.name,
                a.prediction.to_numpy(),
                b.prediction.to_numpy(),
            )
        )
    groups = layout.entity_id.astype(str)
    if group_component is not None:
        groups = groups.map(lambda x: str(json.loads(x)[group_component]))
    gi, gn = pd.factorize(groups, sort=True)
    dates = pd.to_datetime(layout.target_date)
    block = ((dates - dates.min()).dt.days // block_days).to_numpy()
    ti, tn = pd.factorize(block, sort=True)
    if len(gn) < 2 or len(tn) < 2:
        raise ValueError(
            "Two-way bootstrap requires >=2 entity groups and >=2 time blocks; reduce block_days or use more origins"
        )
    y = layout.target.to_numpy(float)
    valid = layout.valid.to_numpy(bool)
    basew = layout.weight.to_numpy(float) * valid

    def score(pred, w):
        den = w.sum()
        if den <= 0:
            raise ValueError("Bootstrap sample has no valid targets")
        error = pred - y
        if metric == "wape":
            d = np.sum(w * abs(y))
            return np.sum(w * abs(error)) / d if d > 0 else np.nan
        if metric == "smape":
            return (
                np.sum(
                    w
                    * np.divide(
                        2 * abs(error),
                        abs(y) + abs(pred),
                        out=np.zeros_like(y),
                        where=(abs(y) + abs(pred)) > 0,
                    )
                )
                / den
            )
        if metric == "mae":
            return np.sum(w * abs(error)) / den
        mse = np.sum(w * error**2) / den
        return np.sqrt(mse) if metric == "rmse" else mse

    def estimator(w):
        # Average neural seeds within each mask; then average masks equally.
        bymask = {}
        for mask, _, a, b in pairs:
            bymask.setdefault(mask, []).append([score(a, w), score(b, w)])
        return np.mean([np.mean(v, axis=0) for v in bymask.values()], axis=0)

    estimate = estimator(basew)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        gc = np.bincount(rng.integers(len(gn), size=len(gn)), minlength=len(gn))
        tc = np.bincount(rng.integers(len(tn), size=len(tn)), minlength=len(tn))
        draws.append(estimator(basew * gc[gi] * tc[ti]))
    draws = np.asarray(draws)
    if not np.isfinite(draws).all():
        raise ValueError(
            "Undefined bootstrap metric (e.g. all-zero WAPE denominator); do not silently discard replicates"
        )
    delta = draws[:, 0] - draws[:, 1]
    observed_delta = float(estimate[0] - estimate[1])
    centered_p = float(
        (1 + np.sum(np.abs(delta - observed_delta) >= abs(observed_delta)))
        / (replicates + 1)
    )
    return {
        "method": method,
        "reference": reference,
        "metric": metric,
        "method_estimate": float(estimate[0]),
        "reference_estimate": float(estimate[1]),
        "delta_method_minus_reference": float(estimate[0] - estimate[1]),
        "D_reference_minus_method": -observed_delta,
        "paired_percentile_95ci_D": np.quantile(-delta, [0.025, 0.975]).tolist(),
        "centered_two_sided_p_value": centered_p,
        "p_resolution": 1 / (replicates + 1),
        "paired_percentile_95ci_delta": np.quantile(delta, [0.025, 0.975]).tolist(),
        "relative_error_reduction": (
            float((estimate[1] - estimate[0]) / estimate[1]) if estimate[1] else None
        ),
        "replicates": replicates,
        "seed": seed,
        "entity_groups": len(gn),
        "time_blocks": len(tn),
        "block_days": block_days,
        "group_component": group_component,
        "mask_count": len(set(p[0] for p in pairs)),
        "estimator": "paired two-way entity-group x nonoverlapping target-date block bootstrap; seeds averaged within masks; no origins-as-replicates",
        "scope": "sampling diagnostic conditional on fitted models and observed masks; no multiple-testing correction",
    }
