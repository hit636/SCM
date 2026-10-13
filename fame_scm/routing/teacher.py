import numpy as np


def masked_softmax(logits, availability):
    logits, avail = np.asarray(logits, float), np.asarray(availability, bool)
    if logits.ndim != 2 or logits.shape != avail.shape or np.any(~avail.any(axis=1)):
        raise ValueError("Every row needs at least one available expert")
    if not np.isfinite(logits[avail]).all():
        raise ValueError("Available logits must be finite")
    x = np.where(avail, logits, -np.inf)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.where(avail, np.exp(x), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def competence_teacher(
    measured,
    observed,
    predicted,
    uncertainty,
    availability,
    online_cost,
    *,
    tau=0.3,
    eta=0.5,
    nu=0.05,
):
    """Eqs. (13–15). Offline acquisition cost and online cost stay separate."""
    measured, obs = np.asarray(measured, float), np.asarray(observed, bool)
    predicted, sd = np.asarray(predicted, float), np.asarray(uncertainty, float)
    avail = np.asarray(availability, bool)
    if not (measured.shape == obs.shape == predicted.shape == sd.shape == avail.shape):
        raise ValueError("Teacher shape mismatch")
    if tau <= 0 or eta < 0 or nu < 0 or np.any(obs & ~avail):
        raise ValueError("Invalid teacher parameters/evidence")
    if not np.isfinite(measured[obs]).all():
        raise ValueError("Observed loss missing")
    if (
        not np.isfinite(predicted[avail]).all()
        or not np.isfinite(sd[avail]).all()
        or np.any(sd[avail] < 0)
    ):
        raise ValueError("Invalid predicted competence")
    cost = np.broadcast_to(np.asarray(online_cost, float), avail.shape)
    if not np.isfinite(cost[avail]).all() or np.any(cost[avail] < 0):
        raise ValueError("Invalid online costs")
    adjusted = np.where(obs, measured, predicted + eta * sd) + nu * cost
    return masked_softmax(-adjusted / tau, avail)
