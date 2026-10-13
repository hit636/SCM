"""Masked Huber/ranking and router objectives with explicit configuration."""

import torch
from torch.nn import functional as F


def competence_objective(
    model,
    predicted_normalized,
    measured_normalized,
    observed,
    *,
    rank_margin,
    lambda_factor,
    lambda_encoder,
    lambda_rank=0.2,
    huber_delta=1.0,
    reconstruction="huber",
    reconstruction_normalizer=None,
    ranking_normalizer=None,
):
    observed = observed.bool()
    if not observed.any() or not torch.isfinite(measured_normalized[observed]).all():
        raise ValueError("Need finite acquired loss labels")
    if rank_margin < 0 or lambda_factor < 0 or lambda_encoder < 0:
        raise ValueError("Invalid explicit objective parameters")
    if reconstruction == "huber":
        rec_values = F.huber_loss(
            predicted_normalized[observed],
            measured_normalized[observed],
            reduction="none",
            delta=huber_delta,
        )
    elif reconstruction == "mse":
        rec_values = (
            predicted_normalized[observed] - measured_normalized[observed]
        ).square()
    else:
        raise ValueError("reconstruction must be huber or mse")
    rec = rec_values.sum() / (reconstruction_normalizer or int(observed.sum()))
    # Mini-batch x expert-a x expert-b. Hidden labels never form ranking pairs.
    labels = torch.where(
        observed, measured_normalized, torch.zeros_like(measured_normalized)
    )
    pairs = (
        observed[:, :, None]
        & observed[:, None, :]
        & (labels[:, :, None] + rank_margin < labels[:, None, :])
    )
    differences = predicted_normalized[:, :, None] - predicted_normalized[:, None, :]
    rank = (
        F.softplus(differences[pairs]).sum() / (ranking_normalizer or int(pairs.sum()))
        if pairs.any()
        else predicted_normalized.sum() * 0.0
    )
    factor_reg = sum(v.square().sum() for v in model.factors)
    encoder_reg = sum(v.square().sum() for v in model.encoder.parameters())
    total = (
        rec
        + lambda_rank * rank
        + lambda_factor * factor_reg
        + lambda_encoder * encoder_reg
    )
    return total, {
        "reconstruction": rec,
        "ranking": rank,
        "ranking_pairs": int(pairs.sum()),
    }


def sparse_router_objective(
    logits,
    teacher,
    cached_forecasts,
    observed,
    targets,
    target_valid,
    online_cost,
    *,
    prediction_loss,
    lambda_pred=1.0,
    lambda_teacher=1.0,
    lambda_balance=0.01,
    lambda_cost=0.01,
):
    observed, valid = observed.bool(), target_valid.bool()
    if torch.any(observed.sum(1) == 0) or not valid.any():
        raise ValueError("Need acquired forecasts per row and valid targets")
    eligible = observed[:, :, None] & valid[:, None, :]
    if (
        not torch.isfinite(cached_forecasts[eligible]).all()
        or not torch.isfinite(targets[valid]).all()
    ):
        raise ValueError("Acquired valid predictions/targets must be finite")
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    # Equivalent to p_m / sum_{observed} p_j, but stable when observed mass is tiny.
    weights = F.softmax(logits.masked_fill(~observed, -torch.inf), dim=1)
    safe_cache = torch.where(
        eligible, cached_forecasts, torch.zeros_like(cached_forecasts)
    )
    mixture = (weights[:, :, None] * safe_cache).sum(dim=1)
    if prediction_loss == "mse":
        pred = F.mse_loss(mixture[valid], targets[valid])
    elif prediction_loss == "mae":
        pred = F.l1_loss(mixture[valid], targets[valid])
    elif prediction_loss == "huber":
        pred = F.huber_loss(mixture[valid], targets[valid], delta=1.0)
    else:
        raise ValueError("prediction_loss must be mse, mae, or huber")
    # KL, balance, and cost follow Eqs. (22)-(24).
    kl = F.kl_div(log_p, teacher.detach(), reduction="batchmean")
    balance = p.shape[1] * p.mean(0).square().sum()
    cost = (p * online_cost).sum(1).mean()
    total = (
        lambda_pred * pred
        + lambda_teacher * kl
        + lambda_balance * balance
        + lambda_cost * cost
    )
    return total, {
        "prediction": pred,
        "teacher_kl": kl,
        "balance": balance,
        "online_cost": cost,
        "singleton_rows": int((observed.sum(1) == 1).sum()),
    }
