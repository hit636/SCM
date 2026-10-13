"""Constrained new-column parameter updates (supplement C-D).

This API consumes ONLY an acquired enlarged cache. It does not evaluate experts
or assume that freezing logits preserves their softmax probabilities. Experiment
orchestration must charge every supplied new-column loss and forecast separately.
"""

from copy import deepcopy
import numpy as np
import torch
from fame_scm.competence.torch_objectives import (
    competence_objective,
    sparse_router_objective,
)


def append_competence(ensemble, z, loss, observed, scaling, seed=42):
    if loss.shape[1] != len(ensemble.models[0].factors) + 1:
        raise ValueError("Exactly one new column required")
    if not np.any(observed[:, -1]):
        raise ValueError("New column requires acquired evidence")
    c = ensemble.cfg["scm"]
    device = ensemble.cfg["device"]
    norm = scaling.normalize(loss)
    x = torch.as_tensor(z, dtype=torch.float32, device=device)
    labels = torch.as_tensor(norm, dtype=torch.float32, device=device)
    obs = torch.as_tensor(observed, dtype=torch.bool, device=device)
    ids = torch.arange(len(z), device=device)
    result = deepcopy(ensemble)
    result.scaling = scaling
    result.train_z = np.array(z, copy=True)
    result.logs = []
    for member, model in enumerate(result.models):
        torch.manual_seed(seed + member * 997)
        new = model.freeze_old_and_append()
        old_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not k.startswith(
                (
                    f"factors.{len(model.factors)-1}",
                    f"expert_biases.{len(model.factors)-1}",
                )
            )
        }
        opt = torch.optim.Adam(new, lr=c["learning_rate"])
        best = float("inf")
        state = None
        wait = 0
        rng = np.random.default_rng(seed + member * 997)
        for epoch in range(c["epochs"]):
            total = 0.0
            model.eval()
            for ix in np.array_split(
                rng.permutation(len(z)), max(1, int(np.ceil(len(z) / c["batch_size"])))
            ):
                opt.zero_grad()
                pred = model(x[ix], ids[ix])
                # Old-old objective terms are constant; gradients affect new bias/factor only.
                objective, _ = competence_objective(
                    model,
                    pred,
                    labels[ix],
                    obs[ix],
                    rank_margin=c["rank_margin"],
                    lambda_factor=c["lambda_factor"],
                    lambda_encoder=c["lambda_encoder"],
                    lambda_rank=c["lambda_rank"],
                    reconstruction=c.get("reconstruction", "huber"),
                    huber_delta=c.get("huber_delta", 1.0),
                )
                if not torch.isfinite(objective):
                    raise ValueError("Nonfinite insertion objective")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(new, 5.0)
                opt.step()
                total += float(objective.detach()) * len(ix)
            total /= len(z)
            result.logs.append({"member": member, "epoch": epoch, "objective": total})
            if total < best - 1e-6:
                best = total
                state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            if wait >= c["patience"]:
                break
        model.load_state_dict(state)
        model.eval()
        for k, v in old_state.items():
            if not torch.equal(v, model.state_dict()[k]):
                raise AssertionError("Old competence parameters changed")
    return result


def append_router(
    router, cfg, z, teacher, forecasts, observed, targets, valid, cost, seed=42
):
    if teacher.shape[1] != len(router.heads) + 1:
        raise ValueError("Exactly one new router head required")
    torch.manual_seed(seed)
    model = deepcopy(router)
    old_logits = None
    device = next(model.parameters()).device
    arrays = [
        torch.as_tensor(
            a,
            device=device,
            dtype=torch.bool if np.asarray(a).dtype == bool else torch.float32,
        )
        for a in [z, teacher, forecasts, observed, targets, valid]
    ]
    x, q, cache, obs, y, mask = arrays
    model.eval()
    with torch.no_grad():
        old_logits = model(x).detach().clone()
    new = model.freeze_old_and_append()
    c = cfg["router"]
    opt = torch.optim.Adam(new, lr=c["learning_rate"])
    costs = torch.as_tensor(cost, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best = float("inf")
    wait = 0
    state = None
    for epoch in range(c["epochs"]):
        model.train_appended_head_only()
        total = 0.0
        for ix in np.array_split(
            rng.permutation(len(z)), max(1, int(np.ceil(len(z) / c["batch_size"])))
        ):
            opt.zero_grad()
            logits = model(x[ix])
            loss, _ = sparse_router_objective(
                logits,
                q[ix],
                cache[ix],
                obs[ix],
                y[ix],
                mask[ix],
                costs,
                prediction_loss=c["prediction_loss"],
                lambda_pred=c["lambda_pred"],
                lambda_teacher=c["lambda_teacher"],
                lambda_balance=c["lambda_balance"],
                lambda_cost=c["lambda_cost"],
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite insertion router loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new, 5.0)
            opt.step()
            total += float(loss.detach()) * len(ix)
        total /= len(z)
        if total < best - 1e-6:
            best = total
            state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= c["patience"]:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        if not torch.equal(old_logits, model(x)[:, :-1]):
            raise AssertionError("Old logits changed during constrained update")
    return model
