import numpy as np


def partition_remainder(total_budget, seed_cells, batches=4):
    remainder = int(total_budget) - int(seed_cells)
    if remainder < 0 or batches < 1:
        raise ValueError("Budget is below initialization or invalid batch count")
    q, rem = divmod(remainder, batches)
    return [q + (i < rem) for i in range(batches)]


def acquisition_scores(
    predicted,
    uncertainty,
    measured,
    observed,
    availability,
    normalized_offline_cost,
    *,
    alpha=1.0,
    beta=0.5,
    zeta=0.1,
    uncertainty_weight=1.0,
):
    """TKDE Eq. (12). predicted/uncertainty/measured use original loss units."""
    predicted, sd = np.asarray(predicted, float), np.asarray(uncertainty, float)
    measured, obs, avail = (
        np.asarray(measured, float),
        np.asarray(observed, bool),
        np.asarray(availability, bool),
    )
    if not (predicted.shape == sd.shape == measured.shape == obs.shape == avail.shape):
        raise ValueError("Acquisition shape mismatch")
    if np.any(obs & ~avail) or np.any(obs.sum(axis=1) == 0):
        raise ValueError("Every mining row needs an executable acquired anchor")
    if not np.isfinite(measured[obs]).all():
        raise ValueError("Observed loss missing")
    if (
        not np.isfinite(predicted[avail]).all()
        or not np.isfinite(sd[avail]).all()
        or np.any(sd[avail] < 0)
    ):
        raise ValueError("Invalid competence predictions/uncertainty")
    costs = np.broadcast_to(np.asarray(normalized_offline_cost, float), avail.shape)
    if not np.isfinite(costs[avail]).all() or np.any(costs[avail] < 0):
        raise ValueError("Invalid offline costs")
    best = np.min(np.where(obs, measured, np.inf), axis=1)
    improvement = np.maximum(0, best[:, None] - predicted)
    coverage = (obs.sum(axis=0) + 1.0) ** -0.5
    scores = (
        uncertainty_weight * sd
        + alpha * improvement
        + beta * coverage[None, :]
        - zeta * costs
    )
    return np.where(avail & ~obs, scores, -np.inf)


def select_batch(scores, count):
    """Frozen-batch stable ordering; tie rule must be recorded in the manifest."""
    scores = np.asarray(scores, float)
    eligible = np.flatnonzero(np.isfinite(scores.ravel()))
    if count < 0 or count > eligible.size:
        raise ValueError("Batch exceeds executable unobserved cells")
    order = eligible[np.argsort(-scores.ravel()[eligible], kind="stable")[:count]]
    return list(zip(*np.unravel_index(order, scores.shape)))


def random_remainder(observed, availability, count, rng):
    candidates = np.flatnonzero(
        (np.asarray(availability, bool) & ~np.asarray(observed, bool)).ravel()
    )
    if count < 0 or count > candidates.size:
        raise ValueError("Invalid random batch size")
    selected = rng.choice(candidates, size=count, replace=False)
    return list(zip(*np.unravel_index(selected, observed.shape)))
