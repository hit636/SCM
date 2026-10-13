"""Evaluator-only Eq. (29) diagnostics; never available to acquisition."""

import numpy as np
from scipy.special import softmax


def stability_diagnostics(
    loss, predicted, teacher, probability, availability, retained, *, tau, bound=None
):
    if tau <= 0:
        raise ValueError("Temperature must be positive")
    loss, predicted, teacher, probability = [
        np.asarray(a, float) for a in [loss, predicted, teacher, probability]
    ]
    availability, retained = np.asarray(availability, bool), np.asarray(retained, bool)
    if not (
        loss.shape
        == predicted.shape
        == teacher.shape
        == probability.shape
        == availability.shape
        == retained.shape
    ):
        raise ValueError("Stability diagnostic shape mismatch")
    if not np.isfinite(loss[availability]).all():
        raise ValueError("Audit losses must be finite")
    b = float(max(loss[availability].max(), 1e-12) if bound is None else bound)
    if not np.isfinite(b) or b <= 0:
        raise ValueError("Loss bound must be positive finite")
    clipped, recovered = np.clip(loss, 0, b), np.clip(predicted, 0, b)
    full_rows = availability.all(1)
    rows = []
    for i in np.flatnonzero(full_rows):
        p, q = probability[i], teacher[i]
        qacc = softmax(-clipped[i] / tau)
        qhat = softmax(-recovered[i] / tau)
        kept_mass = float(p[retained[i]].sum())
        if kept_mass <= 0:
            raise ValueError("Nonempty retained mass required")
        pk = p * retained[i] / kept_mass
        epsilon = float(np.max(abs(recovered[i] - clipped[i])))
        kl = float(
            np.sum(
                np.where(
                    q > 0,
                    q * (np.log(np.maximum(q, 1e-300)) - np.log(np.maximum(p, 1e-300))),
                    0,
                )
            )
        )
        delta = 1 - kept_mass
        adjustment = float(abs(q - qhat).sum())
        imitation, recovery, truncation, modulation = (
            b * np.sqrt(max(0, 2 * kl)),
            2 * b * epsilon / tau,
            2 * b * delta,
            b * adjustment,
        )
        upper = imitation + recovery + truncation + modulation
        deviation = float(pk @ clipped[i] - qacc @ clipped[i])
        if deviation > upper + 1e-8:
            raise AssertionError("Eq. (29) violated")
        rows.append(
            {
                "row": int(i),
                "epsilon_inf": epsilon,
                "teacher_kl": kl,
                "omitted_mass": delta,
                "teacher_adjustment_l1": adjustment,
                "imitation_term": imitation,
                "recovery_term": recovery,
                "truncation_term": truncation,
                "adjustment_term": modulation,
                "upper_bound": upper,
                "component_loss_deviation": deviation,
                "raw_tail_selected": float(pk @ np.maximum(loss[i] - b, 0)),
                "nonvacuous_vs_B": upper < b,
            }
        )
    return rows, {
        "bound_B": b,
        "bound_source": "observed audit maximum" if bound is None else "configuration",
        "fully_executable_rows": len(rows),
        "excluded_partial_rows": int((~full_rows).sum()),
        "mean_upper_bound": (
            float(np.mean([r["upper_bound"] for r in rows])) if rows else None
        ),
        "nonvacuous_fraction": (
            float(np.mean([r["nonvacuous_vs_B"] for r in rows])) if rows else None
        ),
        "scope": "Evaluator-only clipped component loss diagnostics, not a fused-RMSE or sample-complexity guarantee",
    }
