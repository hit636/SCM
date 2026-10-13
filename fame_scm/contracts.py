from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np

STAGES = {"mining", "calibration", "test", "research_audit"}


@dataclass(frozen=True)
class RequestKey:
    stage: str
    entity_id: str
    expert_id: str
    origin: str
    config_hash: str
    protocol_hash: str

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"Unknown stage {self.stage}")
        if not all(
            (
                self.entity_id,
                self.expert_id,
                self.origin,
                self.config_hash,
                self.protocol_hash,
            )
        ):
            raise ValueError("Request key fields cannot be empty")


@dataclass(frozen=True)
class CellKey:
    entity_id: str
    expert_id: str


@dataclass
class ForecastEvidence:
    key: RequestKey
    loss: float
    prediction: np.ndarray
    elapsed_seconds: float


@dataclass
class CellEvidence:
    key: CellKey
    requests: tuple[RequestKey, ...]
    loss: float


class ForecastProvider(Protocol):
    def __call__(self, key: RequestKey) -> ForecastEvidence: ...


class EvidenceView(Protocol):
    """The learner receives acquired evidence, never a dense hidden matrix."""

    def observed_cells(self) -> Sequence[CellEvidence]: ...
    def snapshot(self) -> tuple[np.ndarray, np.ndarray]: ...


class CompetenceEstimator(Protocol):
    def fit(
        self,
        features: np.ndarray,
        observed: EvidenceView,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> None: ...
    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return original-unit mean and standard deviation, both N x M."""
        ...
