from dataclasses import dataclass
import numpy as np


@dataclass
class GateResult:
    available_probability: np.ndarray
    top_r_before_pruning: np.ndarray
    retained: np.ndarray
    weights: np.ndarray


def tkde_gate(probability, availability, *, r=2, threshold=0.05):
    """TKDE Eq. (17): availability -> global available normalization -> Top-r
    -> threshold using those globally normalized probabilities -> fallback.
    Fusion weights are renormalized over the retained experts.
    """
    p, avail = np.asarray(probability, float), np.asarray(availability, bool)
    if p.ndim != 2 or p.shape != avail.shape or r < 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid gate input")
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    mass = np.where(avail, p, 0.0).sum(axis=1, keepdims=True)
    if np.any(~avail.any(axis=1)) or np.any(mass <= 0):
        raise ValueError("No available probability mass")
    filtered = np.where(avail, p, 0.0) / mass
    top, retained = np.zeros_like(avail), np.zeros_like(avail)
    for i in range(p.shape[0]):
        candidates = np.flatnonzero(avail[i])
        order = candidates[np.argsort(-filtered[i, candidates], kind="stable")[:r]]
        top[i, order] = True
        active = order[filtered[i, order] >= threshold]
        if not active.size:
            active = order[:1]
        retained[i, active] = True
    weights = np.where(retained, filtered, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return GateResult(filtered, top, retained, weights)


def fuse(weights, selected_forecasts):
    """Forecast tensor N x M x ... may contain NaN at unexecuted experts."""
    weights, forecasts = np.asarray(weights, float), np.asarray(
        selected_forecasts, float
    )
    if forecasts.ndim < 3 or forecasts.shape[:2] != weights.shape:
        raise ValueError("Fusion shape mismatch")
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(1), 1)
    ):
        raise ValueError("Weights must form valid row distributions")
    expanded = weights[(...,) + (None,) * (forecasts.ndim - 2)]
    selected = expanded > 0
    if not np.isfinite(forecasts[np.broadcast_to(selected, forecasts.shape)]).all():
        raise ValueError("Executed expert forecasts missing")
    safe = np.where(selected, forecasts, 0.0)
    return (expanded * safe).sum(axis=1)


def hit_at_r(top_before_pruning, executable_losses, availability):
    """Evaluator-only diagnostic; best expert is from validation evidence.
    Stable lowest-index tie resolution is an explicit engineering convention.
    """
    top, loss, avail = (
        np.asarray(top_before_pruning, bool),
        np.asarray(executable_losses, float),
        np.asarray(availability, bool),
    )
    if not (top.shape == loss.shape == avail.shape) or np.any(~avail.any(axis=1)):
        raise ValueError("Hit@r shape/availability mismatch")
    if not np.isfinite(loss[avail]).all():
        raise ValueError("Evaluator needs executable validation losses")
    best = np.argmin(np.where(avail, loss, np.inf), axis=1)
    return float(top[np.arange(top.shape[0]), best].mean())
