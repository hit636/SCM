from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    family: str
    config_hash: str
    upstream_commit: str | None
    fit_scope: str  # global_training_stage / local_legal_history / pretrained


class ExpertAdapter(Protocol):
    """All experts emit predictions in the same declared target space."""

    spec: ExpertSpec

    def fit(self, training_data) -> None: ...
    def can_execute(self, history, known_future_covariates) -> bool: ...
    def predict(self, history, horizon: int, known_future_covariates) -> np.ndarray: ...


class ExpertRegistry:
    def __init__(self):
        self._experts = {}

    def register(self, expert: ExpertAdapter):
        if expert.spec.expert_id in self._experts:
            raise ValueError("Duplicate expert ID")
        self._experts[expert.spec.expert_id] = expert

    def get(self, expert_id: str):
        if expert_id not in self._experts:
            raise NotImplementedError(f"Expert adapter not implemented: {expert_id}")
        return self._experts[expert_id]
