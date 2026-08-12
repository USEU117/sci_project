from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class NormalReferencePrediction:
    """Prediction-only observations made from allowed normal reference shots."""

    sample_ids: np.ndarray
    source_ids: np.ndarray
    augmentation_ids: np.ndarray
    image_scores: np.ndarray
    pixel_maps: np.ndarray

    def validate(self, min_views_per_source: int = 1) -> "NormalReferencePrediction":
        if min_views_per_source <= 0:
            raise ValueError("min_views_per_source must be positive")
        sample_ids = np.asarray(self.sample_ids).reshape(-1)
        source_ids = np.asarray(self.source_ids).reshape(-1)
        augmentation_ids = np.asarray(self.augmentation_ids).reshape(-1)
        image_scores = _finite("image_scores", self.image_scores).reshape(-1)
        pixel_maps = _finite("pixel_maps", self.pixel_maps)
        if pixel_maps.ndim == 4 and pixel_maps.shape[1] == 1:
            pixel_maps = pixel_maps[:, 0]
        if pixel_maps.ndim != 3:
            raise ValueError(f"pixel_maps must be [N,H,W], got {pixel_maps.shape}")
        lengths = {
            len(sample_ids),
            len(source_ids),
            len(augmentation_ids),
            len(image_scores),
            len(pixel_maps),
        }
        if len(lengths) != 1:
            raise ValueError("all normal-reference arrays must have equal N")
        if len(sample_ids) == 0:
            raise ValueError("normal-reference cache must not be empty")
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        if any(not str(value).strip() for value in source_ids):
            raise ValueError("source_ids must not be empty")
        unique_sources, counts = np.unique(source_ids, return_counts=True)
        if np.any(counts < min_views_per_source):
            deficient = unique_sources[counts < min_views_per_source].tolist()
            raise ValueError(
                f"each source requires at least {min_views_per_source} views; "
                f"deficient={deficient[:5]}"
            )
        return self

    @property
    def source_count(self) -> int:
        return len(set(np.asarray(self.source_ids).reshape(-1).tolist()))
