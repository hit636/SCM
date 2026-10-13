from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class StageInterval:
    """Inclusive integer timestamp indices; origins are last observed times."""

    start: int
    end: int


def admissible_origins(candidates, stage: StageInterval, horizon: int):
    if horizon <= 0 or stage.start > stage.end:
        raise ValueError("Invalid horizon/stage")
    origins = np.asarray(candidates, dtype=int)
    return origins[(origins + 1 >= stage.start) & (origins + horizon <= stage.end)]


def validate_stage_intervals(stages: dict[str, StageInterval]):
    required = ["train", "mining", "calibration", "test"]
    for name in required:
        if name not in stages or stages[name].start > stages[name].end:
            raise ValueError(f"Missing/invalid {name}")
    for a, b in zip(required, required[1:]):
        if stages[a].end >= stages[b].start:
            raise ValueError(f"Overlapping or unordered target stages: {a}, {b}")


class DatasetAdapter(Protocol):
    """Implement one adapter per dataset; retain zeros and explicit censor masks."""

    def prepare(self, raw_path: str, split_manifest: str) -> None: ...
    def history(self, entity_id: str, origin: str) -> dict: ...
    def targets(self, entity_id: str, origin: str, horizon: int) -> dict: ...
    def availability(self, stage: str) -> np.ndarray: ...


class FingerprintExtractor(Protocol):
    """fit uses training history only; transform uses information at origin."""

    def fit(self, training_histories) -> None: ...
    def transform(self, histories, origins) -> np.ndarray: ...


def basic_sparse_features(values, valid):
    """Three explicit supplementary features, not the complete fingerprint."""
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    x = values[valid]
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("No finite valid history")
    positive = x[x > 0]
    return {
        "zero_ratio": float(np.mean(x == 0)),
        "adi": None if positive.size == 0 else float(x.size / positive.size),
        "cv2_positive": (
            None
            if positive.size == 0
            else float((positive.std(ddof=0) / positive.mean()) ** 2)
        ),
    }
