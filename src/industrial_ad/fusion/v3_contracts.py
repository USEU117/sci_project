from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .contracts import BranchPrediction


class V3Decision(str, Enum):
    VISUAL_CONFIDENT = "visual_confident"
    TEXT_INELIGIBLE = "text_ineligible"
    NO_RESCUE_OPPORTUNITY = "no_rescue_opportunity"
    TEXT_RESCUE_APPLIED = "text_rescue_applied"
    RESCUE_BUDGET_CLIPPED = "rescue_budget_clipped"
    VISUAL_SAFE_FALLBACK = "visual_safe_fallback"


@dataclass(frozen=True)
class BranchEvidenceV3:
    """Label-free branch evidence used by the V3 router.

    Labels and masks are intentionally absent.  Anomaly evidence and reliability
    are separate arrays so a high anomaly score is never treated as a reliability
    failure merely because it lies above the normal-reference range.
    """

    prediction: BranchPrediction
    image_anomaly_evidence: np.ndarray
    pixel_anomaly_evidence: np.ndarray
    image_reliability: np.ndarray
    pixel_reliability: np.ndarray

    def validate(self) -> "BranchEvidenceV3":
        self.prediction.validate()
        sample_ids = np.asarray(self.prediction.sample_ids).reshape(-1)
        maps = np.asarray(self.prediction.pixel_maps)
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0]

        image_evidence = np.asarray(self.image_anomaly_evidence, dtype=np.float64).reshape(-1)
        image_reliability = np.asarray(self.image_reliability, dtype=np.float64).reshape(-1)
        pixel_evidence = np.asarray(self.pixel_anomaly_evidence, dtype=np.float64)
        pixel_reliability = np.asarray(self.pixel_reliability, dtype=np.float64)

        if len(image_evidence) != len(sample_ids) or len(image_reliability) != len(sample_ids):
            raise ValueError("V3 image evidence and reliability must have N values")
        if pixel_evidence.shape != maps.shape or pixel_reliability.shape != maps.shape:
            raise ValueError("V3 pixel evidence and reliability must match pixel maps")
        for name, values in (
            ("image_anomaly_evidence", image_evidence),
            ("pixel_anomaly_evidence", pixel_evidence),
            ("image_reliability", image_reliability),
            ("pixel_reliability", pixel_reliability),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if np.any(image_reliability < 0) or np.any(image_reliability > 1):
            raise ValueError("image_reliability must be in [0, 1]")
        if np.any(pixel_reliability < 0) or np.any(pixel_reliability > 1):
            raise ValueError("pixel_reliability must be in [0, 1]")
        return self


@dataclass(frozen=True)
class RescueBudgetV3:
    max_image_residual: float = 0.05
    max_pixel_residual: float = 0.15

    def validate(self) -> "RescueBudgetV3":
        if not 0 <= self.max_image_residual <= 0.25:
            raise ValueError("max_image_residual must be in [0, 0.25]")
        if not 0 <= self.max_pixel_residual <= 0.50:
            raise ValueError("max_pixel_residual must be in [0, 0.50]")
        return self


@dataclass(frozen=True)
class RescueResultV3:
    sample_ids: np.ndarray
    image_scores: np.ndarray
    pixel_maps: np.ndarray
    image_residual: np.ndarray
    pixel_residual: np.ndarray
    image_rescue_allowed: np.ndarray
    pixel_rescue_allowed: np.ndarray
    decisions: np.ndarray
    features: dict[str, np.ndarray]

    def validate(self) -> "RescueResultV3":
        ids = np.asarray(self.sample_ids).reshape(-1)
        image = np.asarray(self.image_scores, dtype=np.float64).reshape(-1)
        pixel = np.asarray(self.pixel_maps, dtype=np.float64)
        image_residual = np.asarray(self.image_residual, dtype=np.float64).reshape(-1)
        pixel_residual = np.asarray(self.pixel_residual, dtype=np.float64)
        if pixel.ndim != 3:
            raise ValueError("V3 result pixel_maps must be [N,H,W]")
        if not (len(ids) == len(image) == len(pixel) == len(image_residual)):
            raise ValueError("V3 result arrays must share N")
        if pixel_residual.shape != pixel.shape:
            raise ValueError("V3 pixel residual must match pixel maps")
        if np.any(image_residual < 0) or np.any(pixel_residual < 0):
            raise ValueError("V3 rescue residuals must be non-negative")
        if not np.isfinite(image).all() or not np.isfinite(pixel).all():
            raise ValueError("V3 result contains NaN or infinity")
        return self
