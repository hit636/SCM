from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FrozenRowScaling:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_seed(
        cls,
        losses,
        seed_mask,
        fallback_center,
        fallback_scale,
        *,
        floor,
        mad_multiplier,
    ):
        """Fallbacks MUST be frozen training-only stratum/global estimates.
        MAD multiplier and positive floor are supplied by the configuration.
        Single-anchor rows keep their observed center and use fallback scale.
        """
        losses, mask = np.asarray(losses, float), np.asarray(seed_mask, bool)
        n = losses.shape[0]
        if losses.ndim != 2 or mask.shape != losses.shape:
            raise ValueError("Seed shape mismatch")
        centers = np.broadcast_to(np.asarray(fallback_center, float), (n,)).copy()
        scales = np.broadcast_to(np.asarray(fallback_scale, float), (n,)).copy()
        if floor <= 0 or mad_multiplier <= 0:
            raise ValueError("Scale parameters must be positive")
        if not np.isfinite(centers).all() or not np.isfinite(scales).all():
            raise ValueError("Fallbacks must be finite")
        scales = np.maximum(scales, floor)
        for i in range(n):
            x = losses[i, mask[i]]
            if not np.isfinite(x).all():
                raise ValueError("Acquired seed losses must be finite")
            if x.size:
                centers[i] = np.median(x)
            if x.size >= 2:
                mad = mad_multiplier * np.median(np.abs(x - centers[i]))
                if mad > 0 and np.isfinite(mad):
                    scales[i] = max(mad, floor)
        centers.setflags(write=False)
        scales.setflags(write=False)
        return cls(centers, scales)

    def _validate(self, values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[0] != len(self.center):
            raise ValueError(
                "Row scaling requires a 2D array with matching row identities"
            )

    def normalize(self, losses):
        self._validate(losses)
        return (np.asarray(losses) - self.center[:, None]) / self.scale[:, None]

    def restore(self, predicted_normalized):
        self._validate(predicted_normalized)
        return self.center[:, None] + self.scale[:, None] * predicted_normalized

    def restore_sd(self, sd_normalized):
        self._validate(sd_normalized)
        return self.scale[:, None] * sd_normalized
