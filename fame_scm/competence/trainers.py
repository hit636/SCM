from copy import deepcopy
import numpy as np
import torch
from fame_scm.competence.torch_models import CompetenceModel, SparseRouter
from fame_scm.competence.torch_objectives import (
    competence_objective,
    sparse_router_objective,
)


class SCMEnsemble:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed
        self.models = []
        self.logs = []

    def fit(self, z, losses, observed, scaling):
        c = self.cfg["scm"]
        device = self.cfg["device"]
        n, m = losses.shape
        self.scaling = scaling
        self.models = []
        self.train_z = np.array(z, copy=True)
        self.logs = []
        norm = scaling.normalize(losses)
        x = torch.tensor(z, device=device)
        labels = torch.tensor(norm, dtype=torch.float32, device=device)
        for member in range(c["members"]):
            seed = self.seed + member * 997
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            mask = observed & (
                rng.random(observed.shape) < c["ensemble_keep_probability"]
            )
            for i in range(n):
                if not mask[i].any():
                    mask[i, np.flatnonzero(observed[i])[0]] = True
            model = CompetenceModel(
                z.shape[1], n, m, initial_mean=float(np.nanmean(norm[mask]))
            ).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
            from fame_scm.competence.calibration import count_rank_pairs

            total_pairs = count_rank_pairs(norm, mask, c["rank_margin"])
            best = np.inf
            wait = 0
            state = None
            for epoch in range(c["epochs"]):
                model.train()
                total = 0.0
                count = 0
                for ix in np.array_split(
                    rng.permutation(n), max(1, int(np.ceil(n / c["batch_size"])))
                ):
                    ids = torch.tensor(ix, dtype=torch.long, device=device)
                    opt.zero_grad()
                    pred = model(x[ids], ids)
                    objective, _ = competence_objective(
                        model,
                        pred,
                        labels[ids],
                        torch.tensor(mask[ix], device=device),
                        rank_margin=c["rank_margin"],
                        lambda_factor=c["lambda_factor"],
                        lambda_encoder=c["lambda_encoder"],
                        lambda_rank=c["lambda_rank"],
                        reconstruction=c.get("reconstruction", "huber"),
                        huber_delta=c.get("huber_delta", 1.0),
                        reconstruction_normalizer=max(1e-12, mask.sum() * len(ix) / n),
                        ranking_normalizer=max(1e-12, total_pairs * len(ix) / n),
                    )
                    if not torch.isfinite(objective):
                        raise ValueError("Nonfinite SCM objective")
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total += float(objective.detach()) * len(ix)
                    count += len(ix)
                score = total / count
                self.logs.append({"member": member, "epoch": epoch, "objective": score})
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
            self.models.append(model)
        return self

    def predict(self, z, row_ids=None, scaling=None):
        """Known rows require explicit row IDs; unseen rows use zero row offset.

        The exact fitted matrix identifies all
        training rows. New rows require training-derived scaling supplied by caller.
        """
        from fame_scm.competence.scaling import FrozenRowScaling

        z = np.asarray(z, np.float32)
        if row_ids is None:
            if z.shape == self.train_z.shape and np.array_equal(z, self.train_z):
                row_ids = np.arange(len(z))
            else:
                row_ids = np.full(len(z), -1)
        row_ids = np.asarray(row_ids, int)
        if (
            row_ids.shape != (len(z),)
            or np.any(row_ids >= len(self.train_z))
            or np.any(row_ids < -1)
        ):
            raise ValueError("Invalid row IDs; use -1 for unseen rows")
        if scaling is None:
            if np.any(row_ids < 0):
                raise ValueError(
                    "Unseen rows require explicit training-only fallback scaling"
                )
            scaling = FrozenRowScaling(
                self.scaling.center[row_ids], self.scaling.scale[row_ids]
            )
        if len(scaling.center) != len(z) or len(scaling.scale) != len(z):
            raise ValueError("Prediction scaling row mismatch")
        with torch.no_grad():
            x = torch.tensor(z, device=self.cfg["device"])
            ids = torch.tensor(row_ids, device=x.device)
            estimates = np.stack([model(x, ids).cpu().numpy() for model in self.models])
        return scaling.restore(estimates.mean(0)), scaling.restore_sd(
            estimates.std(0, ddof=min(self.cfg["scm"]["sd_ddof"], len(estimates) - 1))
        )


from fame_scm.routing.training import fit_router, router_probabilities
