"""Direct teacher control and alternative sparse probability operators."""

from copy import deepcopy
import numpy as np
import torch
from fame_scm.routing.teacher import competence_teacher


class DirectTeacher:
    def __init__(
        self,
        estimator,
        scaling,
        measured,
        observed,
        available,
        costs,
        config,
        entities=None,
    ):
        if estimator is None:
            raise ValueError("Direct teacher requires a fitted competence estimator")
        self.estimator = estimator
        self.scaling = scaling
        self.measured = np.array(measured, copy=True)
        self.observed = np.array(observed, copy=True)
        self.available = np.array(available, copy=True)
        self.costs = np.array(costs, copy=True)
        self.config = deepcopy(config)
        self.entities = None if entities is None else list(entities)

    def probabilities(self, z):
        if len(z) != len(self.measured):
            raise ValueError(
                "Direct teacher control requires its frozen row identities"
            )
        mean, sd = self.estimator.predict(
            z, row_ids=np.arange(len(z)), scaling=self.scaling
        )
        return competence_teacher(
            self.measured,
            self.observed,
            mean,
            sd,
            self.available,
            self.costs,
            **{k: self.config[k] for k in ["tau", "eta", "nu"]},
        )

    def to(self, device):
        for model in self.estimator.models:
            model.to(device)
        self.estimator.cfg["device"] = str(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def eval(self):
        for model in self.estimator.models:
            model.eval()
        return self

    def state_dict(self):
        return {
            f"member_{i}": model.state_dict()
            for i, model in enumerate(self.estimator.models)
        }


def sparsemax(logits):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    ordered = shifted.sort(dim=1, descending=True).values
    k = torch.arange(1, logits.shape[1] + 1, device=logits.device, dtype=logits.dtype)[
        None
    ]
    cumulative = ordered.cumsum(dim=1)
    support = 1 + k * ordered > cumulative
    size = support.sum(dim=1, keepdim=True)
    tau = (cumulative.gather(1, size - 1) - 1) / size
    return (shifted - tau).clamp_min(0)


def operator_logits(logits, operator="soft_kl", r=2, temperature=1.0, training=False):
    """Return log weights; ST/Gumbel use a hard support and soft surrogate gradient."""
    if operator == "soft_kl":
        return logits
    if operator == "sparsemax":
        return sparsemax(logits).clamp_min(1e-8).log()
    if operator not in ["straight_through", "gumbel_top_r"]:
        raise ValueError(f"Unknown router operator: {operator}")
    if temperature <= 0:
        raise ValueError("Operator temperature must be positive")
    scores = logits
    if training and operator == "gumbel_top_r":
        u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        scores = logits - (-u.log()).log()
    soft = torch.softmax(scores / temperature, dim=1)
    support = torch.zeros_like(soft).scatter_(
        1, scores.topk(min(r, logits.shape[1]), dim=1).indices, 1
    )
    hard = soft * support
    hard = hard / hard.sum(dim=1, keepdim=True)
    weights = hard.detach() - soft.detach() + soft if training else hard
    return weights.clamp_min(1e-8).log()
