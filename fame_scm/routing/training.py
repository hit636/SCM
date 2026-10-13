from copy import deepcopy
import numpy as np
import torch
from fame_scm.competence.torch_models import SparseRouter
from fame_scm.competence.torch_objectives import sparse_router_objective


def fit_router(cfg, z, teacher, forecasts, observed, targets, valid, cost, seed):
    c = cfg["router"]
    device = cfg["device"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SparseRouter(
        z.shape[1],
        teacher.shape[1],
        c["dropout"],
        c.get("operator", "soft_kl"),
        c["r"],
        c.get("operator_temperature", 1.0),
    ).to(device)
    arrays = [z, teacher, forecasts, observed, targets, valid]
    x, q, cache, obs, y, mask = [
        torch.tensor(
            a, device=device, dtype=torch.bool if a.dtype == bool else torch.float32
        )
        for a in arrays
    ]
    costs = torch.tensor(cost, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    best = np.inf
    state = None
    wait = 0
    logs = []
    for epoch in range(c["epochs"]):
        model.train()
        total = 0.0
        count = 0
        for ix in np.array_split(
            rng.permutation(len(z)), max(1, int(np.ceil(len(z) / c["batch_size"])))
        ):
            opt.zero_grad()
            out = model(x[ix])
            loss, parts = sparse_router_objective(
                out,
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
                raise ValueError("Nonfinite router objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(ix)
            count += len(ix)
        score = total / count
        logs.append({"epoch": epoch, "objective": score})
        if score < best - 1e-6:
            best = score
            wait = 0
            state = deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= c["patience"]:
            break
    model.load_state_dict(state)
    model.eval()
    return model, logs


def router_probabilities(model, z):
    if hasattr(model, "probabilities"):
        return model.probabilities(np.asarray(z, np.float32))
    with torch.no_grad():
        return (
            torch.softmax(
                model(
                    torch.tensor(
                        z, dtype=torch.float32, device=next(model.parameters()).device
                    )
                ),
                1,
            )
            .cpu()
            .numpy()
            .astype(float)
        )
