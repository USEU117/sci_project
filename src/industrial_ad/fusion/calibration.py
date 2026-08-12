from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import BranchPrediction


EPS = 1e-6
PIXEL_REFERENCE_QUANTILE = 0.99


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class RobustNormalCalibrator:
    """Map raw scores to probabilities using normal-reference data only.

    ``center`` and ``scale`` must be fitted outside the target test set.  The
    robust MAD scale avoids a single reference image dominating calibration.
    """

    center: float
    scale: float
    temperature: float = 1.0

    @classmethod
    def fit(
        cls, normal_reference: np.ndarray, temperature: float = 1.0
    ) -> "RobustNormalCalibrator":
        reference = _finite(normal_reference, "normal_reference").reshape(-1)
        if reference.size == 0:
            raise ValueError("normal_reference must not be empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        center = float(np.median(reference))
        mad = float(np.median(np.abs(reference - center)))
        scale = max(1.4826 * mad, EPS)
        return cls(center=center, scale=scale, temperature=float(temperature))

    def transform(self, values: np.ndarray) -> np.ndarray:
        raw = _finite(values, "values")
        logits = (raw - self.center) / (self.scale * self.temperature)
        positive = logits >= 0
        output = np.empty_like(logits, dtype=np.float64)
        output[positive] = 1.0 / (1.0 + np.exp(-np.minimum(logits[positive], 700.0)))
        negative = ~positive
        exp_logits = np.exp(np.maximum(logits[negative], -700.0))
        output[negative] = exp_logits / (1.0 + exp_logits)
        return np.clip(output, EPS, 1.0 - EPS)

    def to_dict(self) -> dict[str, float]:
        return {
            "center": self.center,
            "scale": self.scale,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "RobustNormalCalibrator":
        return cls(
            center=float(values["center"]),
            scale=float(values["scale"]),
            temperature=float(values["temperature"]),
        )


@dataclass(frozen=True)
class BranchCalibration:
    image: RobustNormalCalibrator
    pixel: RobustNormalCalibrator

    @classmethod
    def fit(
        cls,
        normal_image_scores: np.ndarray,
        normal_pixel_maps: np.ndarray,
        temperature: float = 1.0,
    ) -> "BranchCalibration":
        maps = _finite(normal_pixel_maps, "normal_pixel_maps")
        if maps.ndim < 2:
            raise ValueError("normal_pixel_maps must include a per-image spatial dimension")
        if maps.shape[0] != np.asarray(normal_image_scores).reshape(-1).shape[0]:
            raise ValueError("normal image scores and pixel maps must have equal N")
        # Pixel maps for normal images are usually sparse: flattening all pixels
        # makes their median and MAD exactly zero.  Fit one robust scale from
        # each view's upper-tail response instead, then apply it to every pixel.
        # This remains normal-reference-only and captures view-to-view response
        # variation without treating background zeros as calibration evidence.
        per_view_tail = np.quantile(
            maps.reshape(maps.shape[0], -1), PIXEL_REFERENCE_QUANTILE, axis=1
        )
        return cls(
            image=RobustNormalCalibrator.fit(normal_image_scores, temperature),
            pixel=RobustNormalCalibrator.fit(per_view_tail, temperature),
        )

    def apply(self, branch: BranchPrediction) -> BranchPrediction:
        branch.validate()
        return BranchPrediction(
            sample_ids=np.asarray(branch.sample_ids),
            image_scores=self.image.transform(branch.image_scores).astype(np.float32),
            pixel_maps=self.pixel.transform(branch.pixel_maps).astype(np.float32),
            image_uncertainty=branch.image_uncertainty,
            pixel_uncertainty=branch.pixel_uncertainty,
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"image": self.image.to_dict(), "pixel": self.pixel.to_dict()}

    @classmethod
    def from_dict(
        cls, values: dict[str, dict[str, float]]
    ) -> "BranchCalibration":
        return cls(
            image=RobustNormalCalibrator.from_dict(values["image"]),
            pixel=RobustNormalCalibrator.from_dict(values["pixel"]),
        )


def load_category_calibrations(
    payload: dict[str, Any], category: str
) -> tuple[BranchCalibration, BranchCalibration]:
    """Load one category only after checking the no-test-data fit contract."""

    if payload.get("status") != "passed":
        raise ValueError("calibration report status must be 'passed'")
    if payload.get("test_predictions_used") is not False:
        raise ValueError("calibration must explicitly record test_predictions_used=false")
    if payload.get("test_labels_used") is not False:
        raise ValueError("calibration must explicitly record test_labels_used=false")
    categories = payload.get("categories")
    if not isinstance(categories, dict) or category not in categories:
        raise ValueError(f"calibration category missing: {category}")
    values = categories[category]
    if not isinstance(values, dict) or "visual" not in values or "text" not in values:
        raise ValueError(f"calibration branches missing for category: {category}")
    return (
        BranchCalibration.from_dict(values["visual"]),
        BranchCalibration.from_dict(values["text"]),
    )
