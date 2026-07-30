from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


@dataclass(frozen=True)
class BranchPrediction:
    """Prediction-only branch input. Ground truth is intentionally absent."""

    sample_ids: np.ndarray
    image_scores: np.ndarray
    pixel_maps: np.ndarray
    image_uncertainty: np.ndarray | None = None
    pixel_uncertainty: np.ndarray | None = None

    def validate(self) -> "BranchPrediction":
        ids = np.asarray(self.sample_ids).reshape(-1)
        scores = _finite("image_scores", self.image_scores).reshape(-1)
        maps = _finite("pixel_maps", self.pixel_maps)
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0]
        if maps.ndim != 3:
            raise ValueError(f"pixel_maps must be [N,H,W], got {maps.shape}")
        if not (len(ids) == len(scores) == len(maps)):
            raise ValueError("sample_ids, image_scores and pixel_maps must have equal N")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("sample_ids must be unique")
        if self.image_uncertainty is not None:
            uncertainty = _finite("image_uncertainty", self.image_uncertainty).reshape(-1)
            if len(uncertainty) != len(ids):
                raise ValueError("image_uncertainty must have N values")
        if self.pixel_uncertainty is not None:
            uncertainty_map = _finite("pixel_uncertainty", self.pixel_uncertainty)
            if uncertainty_map.shape != maps.shape:
                raise ValueError("pixel_uncertainty must match pixel_maps")
        return self


@dataclass(frozen=True)
class FusionResult:
    sample_ids: np.ndarray
    image_scores: np.ndarray
    pixel_maps: np.ndarray
    visual_weights: np.ndarray
    visual_pixel_weights: np.ndarray
    decisions: np.ndarray
    features: dict[str, np.ndarray]
