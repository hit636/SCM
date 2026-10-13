import numpy as np


def forecast_metrics(target, prediction, valid, weights=None):
    """One shared target index/weight set for pooled metrics; ratios use 0..1.
    sMAPE = mean(2|e|/(|y|+|pred|)); both-zero contributions are zero.
    Zero WAPE denominator returns None, not a fabricated zero.
    """
    y, pred, valid = (
        np.asarray(target, float),
        np.asarray(prediction, float),
        np.asarray(valid, bool),
    )
    if not (y.shape == pred.shape == valid.shape):
        raise ValueError("Target/prediction/mask shape mismatch")
    w = (
        np.ones(y.shape)
        if weights is None
        else np.broadcast_to(np.asarray(weights, float), y.shape)
    )
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be finite and nonnegative")
    valid = valid & (w > 0)
    if (
        not valid.any()
        or not np.isfinite(y[valid]).all()
        or not np.isfinite(pred[valid]).all()
    ):
        raise ValueError("No valid finite target/prediction records")
    y, pred, w = y[valid], pred[valid], w[valid]
    absolute = np.abs(pred - y)
    total_w, absolute_y = w.sum(), np.sum(w * np.abs(y))
    mse, mae = float(np.sum(w * (pred - y) ** 2) / total_w), float(
        np.sum(w * absolute) / total_w
    )
    smape_den = np.abs(y) + np.abs(pred)
    smape_values = np.divide(
        2 * absolute, smape_den, out=np.zeros_like(absolute), where=smape_den > 0
    )
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "wape": None if absolute_y == 0 else float(np.sum(w * absolute) / absolute_y),
        "smape": float(np.sum(w * smape_values) / total_w),
        "target_abs_mean": float(absolute_y / total_w),
        "valid_targets": int(valid.sum()),
        "total_weight": float(total_w),
    }


def full_observation_gap(error, full_error):
    if full_error <= 0:
        raise ValueError("Full-observation reference must be positive")
    return float((error - full_error) / full_error)


def relative_reduction(error, baseline_error):
    if baseline_error <= 0:
        raise ValueError("Baseline reference must be positive")
    return float((baseline_error - error) / baseline_error)


def summarize_nested_replicates(errors):
    """Array = evidence-mask x neural-seed; paper SD is over mask means."""
    x = np.asarray(errors, float)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("Need >=2 masks with equal finite seed records")
    means = x.mean(axis=1)
    return {
        "mean": float(means.mean()),
        "mask_sd": float(means.std(ddof=1)),
        "mask_means": means.tolist(),
        "masks": x.shape[0],
        "neural_seeds_per_mask": x.shape[1],
    }
