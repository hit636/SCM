"""Neural competence and router architectures for Eqs.6/16."""

import torch
from torch import nn


class CompetenceModel(nn.Module):
    def __init__(self, feature_dim, rows, experts, width=32, initial_mean=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, width),
        )
        self.global_mean = nn.Parameter(torch.tensor(float(initial_mean)))
        self.row_bias = nn.Embedding(rows, 1)
        nn.init.zeros_(self.row_bias.weight)
        self.factors = nn.ParameterList()
        self.expert_biases = nn.ParameterList()
        self.width = width
        for _ in range(experts):
            self.append_expert()

    def append_expert(self):
        reference = self.global_mean
        factor = nn.Parameter(
            torch.randn(self.width, device=reference.device, dtype=reference.dtype)
            * 0.02
        )
        bias = nn.Parameter(
            torch.zeros((), device=reference.device, dtype=reference.dtype)
        )
        self.factors.append(factor)
        self.expert_biases.append(bias)
        return [factor, bias]

    def freeze_old_and_append(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self.append_expert()

    def forward(self, features, row_ids=None):
        h = self.encoder(features)
        offsets = torch.zeros((features.shape[0], 1), device=h.device, dtype=h.dtype)
        if row_ids is not None:
            # -1 denotes unseen rows; no competence evidence -> b_i = 0.
            known = row_ids >= 0
            offsets = torch.where(
                known[:, None], self.row_bias(row_ids.clamp_min(0)), offsets
            )
        return (
            self.global_mean
            + offsets
            + torch.stack(list(self.expert_biases))[None, :]
            + h @ torch.stack(list(self.factors)).T
        )


class SparseRouter(nn.Module):
    def __init__(
        self,
        feature_dim,
        experts,
        dropout=0.1,
        operator="soft_kl",
        r=2,
        temperature=1.0,
    ):
        super().__init__()
        # Dropout follows each hidden ReLU.
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(experts)])
        self.operator, self.r, self.temperature = operator, r, temperature

    def forward(self, features):
        h = self.encoder(features)
        logits = torch.cat([head(h) for head in self.heads], dim=1)
        from fame_scm.routing.consumers import operator_logits

        return operator_logits(
            logits, self.operator, self.r, self.temperature, self.training
        )

    def freeze_old_and_append(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        reference = next(self.parameters())
        head = nn.Linear(128, 1).to(device=reference.device, dtype=reference.dtype)
        self.heads.append(head)
        return list(head.parameters())

    def train_appended_head_only(self):
        # requires_grad alone does not disable dropout in a frozen encoder.
        self.eval()
        self.heads[-1].train()
