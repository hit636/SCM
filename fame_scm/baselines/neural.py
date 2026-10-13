"""Independent nonlinear loss regressors with acquired-label-only fitting."""

import numpy as np
import torch
from torch import nn


def independent_neural(z, loss, observed, scaling, cfg, seed):
    y = scaling.normalize(loss)
    n, m = y.shape
    device = cfg["device"]
    x = torch.as_tensor(z, dtype=torch.float32, device=device)
    predictions = []
    for member in range(cfg["scm"]["members"]):
        matrix = np.empty((n, m), dtype=float)
        rng = np.random.default_rng(seed + 997 * member)
        for j in range(m):
            rows = np.flatnonzero(observed[:, j])
            if not len(rows):
                matrix[:, j] = np.nanmean(y[observed])
                continue
            torch.manual_seed(seed + 997 * member + j)
            keep = rows[rng.random(len(rows)) < cfg["scm"]["ensemble_keep_probability"]]
            if not len(keep):
                keep = rows[:1]
            model = nn.Sequential(
                nn.Linear(z.shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg["scm"]["learning_rate"]
            )
            labels = torch.as_tensor(y[:, j], dtype=torch.float32, device=device)
            best, wait, state = float("inf"), 0, None
            for _ in range(cfg["scm"]["epochs"]):
                optimizer.zero_grad()
                predicted = model(x[keep])[:, 0]
                objective = torch.nn.functional.huber_loss(
                    predicted, labels[keep], delta=cfg["scm"].get("huber_delta", 1.0)
                )
                objective.backward()
                optimizer.step()
                value = float(objective.detach())
                if value < best - 1e-6:
                    best, wait = value, 0
                    state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                else:
                    wait += 1
                if wait >= cfg["scm"]["patience"]:
                    break
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                matrix[:, j] = model(x)[:, 0].cpu().numpy()
        predictions.append(matrix)
    array = np.stack(predictions)
    return scaling.restore(array.mean(0)), scaling.restore_sd(
        array.std(0, ddof=min(1, len(array) - 1))
    )
