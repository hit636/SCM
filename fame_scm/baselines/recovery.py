"""Recovery baselines; all fit inputs are acquired labels only."""

import numpy as np
import torch
from lightgbm import LGBMRegressor
from scipy.special import ndtr
from sklearn.neighbors import NearestNeighbors


def recover_static(method, z, loss, mask, scaling, cfg, seed):
    y = scaling.normalize(loss)
    n, m = y.shape
    c = cfg["baselines"]
    rng = np.random.default_rng(seed)
    mean = np.array(
        [
            np.mean(y[mask[:, j], j]) if mask[:, j].any() else np.nanmean(y[mask])
            for j in range(m)
        ]
    )
    pred = np.tile(mean, (n, 1))
    sd = np.ones_like(pred)
    if method == "random_partial":
        # Chunk queries; never construct the N x N distance matrix.
        for j in range(m):
            known = np.flatnonzero(mask[:, j])
            if not len(known):
                continue
            nn = NearestNeighbors(
                n_neighbors=min(c["neighbors"], len(known)), n_jobs=cfg["threads"]
            ).fit(z[known])
            for lo in range(0, n, c.get("query_chunk_size", 512)):
                hi = min(n, lo + c.get("query_chunk_size", 512))
                dist, idx = nn.kneighbors(z[lo:hi])
                values = y[known[idx], j]
                weights = 1 / (dist + 1e-4)
                weights /= weights.sum(1, keepdims=True)
                pred[lo:hi, j] = (weights * values).sum(1)
                sd[lo:hi, j] = np.sqrt(
                    (weights * (values - pred[lo:hi, j, None]) ** 2).sum(1) + 1e-4
                )
    elif method == "softimpute":
        filled = np.where(mask, y, pred)
        for _ in range(c["epochs"]):
            u, s, v = np.linalg.svd(filled, full_matrices=False)
            s = np.maximum(s - c["softimpute_lambda"], 0)
            reconstructed = (u * s) @ v
            if np.linalg.norm(reconstructed - pred) < 1e-5:
                pred = reconstructed
                break
            pred = reconstructed
            filled = np.where(mask, y, pred)
        sd[:] = max(float(np.std(y[mask] - pred[mask])), 0.01)
    elif method == "independent_lgbm":
        for j in range(m):
            ids = np.flatnonzero(mask[:, j])
            if len(ids) < 2:
                continue
            model = LGBMRegressor(
                n_estimators=cfg["experts"]["tree_estimators"],
                num_leaves=7,
                min_child_samples=2,
                verbosity=-1,
                n_jobs=cfg["threads"],
                random_state=seed,
            )
            model.fit(z[ids], y[ids, j])
            pred[:, j] = model.booster_.predict(z)
            sd[:, j] = max(float(np.std(y[ids, j] - pred[ids, j])), 0.01)
    elif method == "imc":
        torch.manual_seed(seed)
        rank = min(c["rank"], m)
        device = cfg["device"]
        x = torch.tensor(np.c_[np.ones(n), z], dtype=torch.float32, device=device)
        a = torch.nn.Parameter(torch.randn(x.shape[1], rank, device=device) * 0.05)
        b = torch.nn.Parameter(torch.randn(m, rank, device=device) * 0.05)
        target = torch.tensor(np.nan_to_num(y), dtype=torch.float32, device=device)
        obs = torch.tensor(mask, device=device)
        opt = torch.optim.Adam([a, b], lr=0.01)
        for _ in range(c["epochs"]):
            opt.zero_grad()
            estimate = x @ a @ b.T
            objective = (estimate[obs] - target[obs]).square().mean() + 1e-4 * (
                a.square().sum() + b.square().sum()
            )
            objective.backward()
            opt.step()
        pred = (x @ a @ b.T).detach().cpu().numpy()
        sd[:] = max(float(np.std(y[mask] - pred[mask])), 0.01)
    elif method == "fformpp_b":
        import jax
        from movingknots import fformpp
        from sklearn.decomposition import PCA

        complete = np.flatnonzero(mask.all(1))
        if len(complete) < 2:
            raise ValueError(
                "FFORMPP-B requires >=2 fully acquired rows; increase rho or series count"
            )
        dimensions = min(c["fformpp_features"], z.shape[1], max(1, n - 1))
        xp = PCA(n_components=dimensions, random_state=seed).fit_transform(z)
        fit = fformpp.fit(
            xp[complete],
            y[complete],
            model_names=[str(j) for j in range(m)],
            surface_knots=1,
            additive_knots=1,
            key=jax.random.PRNGKey(seed),
            fit_kwargs={
                "n_steps": c["fformpp_steps"],
                "n_samples": 1,
                "learning_rate": 0.005,
                "init_scale": 0.005,
            },
        )
        pred = np.asarray(
            fformpp.predict(
                fit,
                xp,
                key=jax.random.PRNGKey(seed + 1),
                n_samples=c["fformpp_samples"],
                estimate="mean",
                return_frame=False,
            )
        )
        sd[:] = max(float(np.std(y[complete] - pred[complete])), 0.01)
    else:
        raise ValueError(f"Unsupported static recovery: {method}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{method} produced nonfinite recovery")
    return scaling.restore(pred), scaling.restore_sd(sd)


class GPLatentRecovery:
    """Fusi-style GP over learned expert embeddings, independent entity functions.
    Uses acquired normalized losses and exact observed-submatrix conditioning.
    This is an explicitly documented entity-level adaptation, not author code.
    """

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed

    def fit(self, z, loss, mask, scaling):
        torch.manual_seed(self.seed)
        n, m = loss.shape
        c = self.cfg["baselines"]
        device = self.cfg["device"]
        self.mask = mask.copy()
        self.y = scaling.normalize(loss)
        self.scaling = scaling
        latent = torch.nn.Parameter(
            torch.randn(m, c["pmf_latent_dim"], device=device) * 0.3
        )
        log_length = torch.nn.Parameter(torch.tensor(0.0, device=device))
        log_noise = torch.nn.Parameter(torch.tensor(-2.0, device=device))
        opt = torch.optim.Adam([latent, log_length, log_noise], lr=0.02)
        groups = {}
        for i, row in enumerate(mask):
            groups.setdefault(tuple(np.flatnonzero(row)), []).append(i)
        for _ in range(c["epochs"]):
            opt.zero_grad()
            kernel = torch.exp(
                -torch.cdist(latent, latent).square() / (2 * log_length.exp().square())
            )
            total = latent.square().mean() * 0.01
            for columns, rows in groups.items():
                idx = torch.tensor(columns, dtype=torch.long, device=device)
                targets = torch.tensor(
                    self.y[np.ix_(rows, columns)], dtype=torch.float32, device=device
                ).T
                covariance = kernel[idx][:, idx] + (
                    log_noise.exp().square() + 1e-4
                ) * torch.eye(len(columns), device=device)
                chol = torch.linalg.cholesky(covariance)
                solution = torch.cholesky_solve(targets, chol)
                total = (
                    total
                    + (targets * solution).sum() / (2 * n)
                    + len(rows) * torch.log(torch.diag(chol)).sum() / n
                )
            total.backward()
            opt.step()
            with torch.no_grad():
                log_noise.clamp_(-5, 2)
                log_length.clamp_(-3, 3)
        with torch.no_grad():
            self.kernel = (
                torch.exp(
                    -torch.cdist(latent, latent).square()
                    / (2 * log_length.exp().square())
                )
                .cpu()
                .numpy()
            )
            self.noise = float(log_noise.exp().square()) + 1e-4
        return self

    def predict(self, z):
        n, m = self.y.shape
        mean = np.zeros((n, m))
        sd = np.zeros((n, m))
        for i in range(n):
            idx = np.flatnonzero(self.mask[i])
            k = self.kernel[np.ix_(idx, idx)] + self.noise * np.eye(len(idx))
            cross = self.kernel[:, idx]
            mean[i] = cross @ np.linalg.solve(k, self.y[i, idx])
            sd[i] = np.sqrt(
                np.maximum(
                    1.0 - np.sum(cross * np.linalg.solve(k, cross.T).T, axis=1), 1e-8
                )
            )
        return self.scaling.restore(mean), self.scaling.restore_sd(sd)


def expected_improvement(mean, sd, loss, mask, available, costs=None):
    best = np.min(np.where(mask, loss, np.inf), axis=1)[:, None]
    original_sd = np.asarray(sd)
    sd = np.maximum(sd, 1e-8)
    delta = best - mean
    z = delta / sd
    ei = delta * ndtr(z) + sd * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ei = np.where(original_sd <= 0, np.maximum(delta, 0), ei)
    if costs is not None:
        cost = np.broadcast_to(np.asarray(costs, float), ei.shape)
        if np.any(cost[available] <= 0) or not np.isfinite(cost[available]).all():
            raise ValueError("Expected-improvement costs must be positive finite")
        ei /= np.maximum(cost, 1e-6)
    return np.where(available & ~mask, ei, -np.inf)
