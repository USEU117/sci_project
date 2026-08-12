from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import BranchPrediction


EPS = np.finfo(np.float64).eps
PIXEL_TAIL_QUANTILE = 0.99


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class RankPreservingCalibrator:
    """Strictly monotonic normal-reference calibration for DynamicFusion V2.

    The arctangent mapping never clips finite query scores to a fixed boundary,
    so it preserves the ordering of the source branch.  All fitted statistics
    come from allowed normal-reference predictions.
    """

    center: float
    scale: float
    lower_support: float
    upper_support: float
    scale_floor: float
    reference_count: int
    degenerate_reference: bool
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    @classmethod
    def fit(
        cls,
        normal_reference: np.ndarray,
        *,
        minimum_scale: float = 1e-6,
        scale_floor_fraction: float = 0.05,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> "RankPreservingCalibrator":
        reference = _finite(normal_reference, "normal_reference").reshape(-1)
        if reference.size == 0:
            raise ValueError("normal_reference must not be empty")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        if scale_floor_fraction < 0:
            raise ValueError("scale_floor_fraction must be non-negative")
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("support quantiles must satisfy 0 <= lower < upper <= 1")

        center = float(np.median(reference))
        mad_scale = float(1.4826 * np.median(np.abs(reference - center)))
        q25, q75 = np.quantile(reference, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349)
        reference_range = float(np.max(reference) - np.min(reference))
        scale_floor = max(
            float(minimum_scale),
            float(reference_range * scale_floor_fraction),
        )
        robust_scale = max(mad_scale, iqr_scale)
        scale = max(robust_scale, scale_floor)
        return cls(
            center=center,
            scale=float(scale),
            lower_support=float(np.quantile(reference, lower_quantile)),
            upper_support=float(np.quantile(reference, upper_quantile)),
            scale_floor=float(scale_floor),
            reference_count=int(reference.size),
            degenerate_reference=bool(robust_scale < scale_floor),
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        raw = _finite(values, "values")
        standardized = (raw - self.center) / self.scale
        # atan is strictly increasing for finite values and avoids sigmoid's
        # practical 0/1 saturation under the large V1 z-scores.
        return 0.5 + np.arctan(standardized) / np.pi

    def upper_support_evidence(
        self, values: np.ndarray, *, tolerance: float = 3.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return out-of-support flag, confidence and normalized excess.

        Industrial anomaly scores are directional: only an unexpectedly high
        response is treated as extrapolation.  A low background response is not
        incorrectly flagged as unsupported.
        """

        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        raw = _finite(values, "values")
        excess = np.maximum(raw - self.upper_support, 0.0) / self.scale
        out_of_support = excess > tolerance
        confidence = 1.0 / (1.0 + excess)
        return out_of_support, confidence, excess

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "rank_preserving_arctan",
            "center": self.center,
            "scale": self.scale,
            "lower_support": self.lower_support,
            "upper_support": self.upper_support,
            "scale_floor": self.scale_floor,
            "reference_count": self.reference_count,
            "degenerate_reference": self.degenerate_reference,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RankPreservingCalibrator":
        if values.get("type") != "rank_preserving_arctan":
            raise ValueError("unsupported V2 calibration type")
        return cls(
            center=float(values["center"]),
            scale=float(values["scale"]),
            lower_support=float(values["lower_support"]),
            upper_support=float(values["upper_support"]),
            scale_floor=float(values["scale_floor"]),
            reference_count=int(values["reference_count"]),
            degenerate_reference=bool(values["degenerate_reference"]),
            lower_quantile=float(values["lower_quantile"]),
            upper_quantile=float(values["upper_quantile"]),
        )


@dataclass(frozen=True)
class BranchV2Calibration:
    image: RankPreservingCalibrator
    pixel: RankPreservingCalibrator
    pixel_tail_quantile: float = PIXEL_TAIL_QUANTILE

    @classmethod
    def fit(
        cls,
        normal_image_scores: np.ndarray,
        normal_pixel_maps: np.ndarray,
        **kwargs: Any,
    ) -> "BranchV2Calibration":
        scores = _finite(normal_image_scores, "normal_image_scores").reshape(-1)
        maps = _finite(normal_pixel_maps, "normal_pixel_maps")
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0]
        if maps.ndim != 3:
            raise ValueError(f"normal_pixel_maps must be [N,H,W], got {maps.shape}")
        if len(scores) != len(maps):
            raise ValueError("normal image scores and pixel maps must have equal N")
        per_view_tail = np.quantile(
            maps.reshape(len(maps), -1), PIXEL_TAIL_QUANTILE, axis=1
        )
        return cls(
            image=RankPreservingCalibrator.fit(scores, **kwargs),
            pixel=RankPreservingCalibrator.fit(per_view_tail, **kwargs),
        )

    @property
    def calibration_warning(self) -> bool:
        return self.image.degenerate_reference or self.pixel.degenerate_reference

    def apply(self, branch: BranchPrediction) -> BranchPrediction:
        branch.validate()
        return BranchPrediction(
            sample_ids=np.asarray(branch.sample_ids),
            image_scores=self.image.transform(branch.image_scores).astype(np.float32),
            pixel_maps=self.pixel.transform(branch.pixel_maps).astype(np.float32),
            image_uncertainty=branch.image_uncertainty,
            pixel_uncertainty=branch.pixel_uncertainty,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image.to_dict(),
            "pixel": self.pixel.to_dict(),
            "pixel_tail_quantile": self.pixel_tail_quantile,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BranchV2Calibration":
        return cls(
            image=RankPreservingCalibrator.from_dict(values["image"]),
            pixel=RankPreservingCalibrator.from_dict(values["pixel"]),
            pixel_tail_quantile=float(values.get("pixel_tail_quantile", PIXEL_TAIL_QUANTILE)),
        )


def load_v2_category_calibrations(
    payload: dict[str, Any], category: str
) -> tuple[BranchV2Calibration, BranchV2Calibration]:
    """Load a V2 calibration only after enforcing its leakage-safe contract."""

    if payload.get("schema_version") != 2:
        raise ValueError("V2 calibration schema_version must be 2")
    if payload.get("status") != "passed":
        raise ValueError("V2 calibration report status must be 'passed'")
    for field in (
        "test_predictions_used",
        "test_labels_used",
        "test_masks_used",
        "test_set_statistics_used",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"V2 calibration must explicitly record {field}=false")
    categories = payload.get("categories")
    if not isinstance(categories, dict) or category not in categories:
        raise ValueError(f"V2 calibration category missing: {category}")
    values = categories[category]
    if not isinstance(values, dict) or "visual" not in values or "text" not in values:
        raise ValueError(f"V2 calibration branches missing for category: {category}")
    return (
        BranchV2Calibration.from_dict(values["visual"]),
        BranchV2Calibration.from_dict(values["text"]),
    )
