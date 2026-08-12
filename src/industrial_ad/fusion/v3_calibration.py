from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = np.finfo(np.float64).eps


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class GroupedReferenceCalibratorV3:
    """Order-preserving calibration with group-aware K-shot shrinkage.

    Multiple deterministic views from one normal reference remain one effective
    reference group.  This prevents a 1-shot reference from appearing to be a
    large independent calibration set merely because it has many augmentations.
    Reliability depends only on normal-reference quality, never on how high a
    query anomaly score is.
    """

    center: float
    scale: float
    source_center: float
    source_scale: float
    effective_group_count: int
    view_count: int
    target_weight: float
    within_group_instability: float
    reference_reliability: float
    target_scale_degenerate: bool

    @classmethod
    def fit(
        cls,
        reference_values: np.ndarray,
        group_ids: np.ndarray,
        *,
        source_center: float = 0.0,
        source_scale: float = 1.0,
        prior_strength: float = 2.0,
        minimum_scale: float = 1e-6,
    ) -> "GroupedReferenceCalibratorV3":
        values = _finite(reference_values, "reference_values")
        groups = np.asarray(group_ids).astype(str).reshape(-1)
        if len(values) != len(groups):
            raise ValueError("reference_values and group_ids must have equal length")
        if len(set(groups.tolist())) == 0:
            raise ValueError("group_ids must not be empty")
        if not np.isfinite(source_center):
            raise ValueError("source_center must be finite")
        if not np.isfinite(source_scale) or source_scale <= 0:
            raise ValueError("source_scale must be finite and positive")
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")

        unique_groups = list(dict.fromkeys(groups.tolist()))
        group_medians: list[float] = []
        group_instabilities: list[float] = []
        for group in unique_groups:
            subset = values[groups == group]
            group_medians.append(float(np.median(subset)))
            group_instabilities.append(float(1.4826 * np.median(np.abs(subset - np.median(subset)))))

        summaries = np.asarray(group_medians, dtype=np.float64)
        target_center = float(np.median(summaries))
        target_mad = float(1.4826 * np.median(np.abs(summaries - target_center)))
        if len(summaries) >= 2:
            q25, q75 = np.quantile(summaries, [0.25, 0.75])
            target_iqr = float((q75 - q25) / 1.349)
        else:
            target_iqr = 0.0
        target_scale = max(target_mad, target_iqr)
        target_scale_degenerate = target_scale < minimum_scale

        group_count = len(unique_groups)
        target_weight = (
            group_count / (group_count + prior_strength)
            if group_count + prior_strength > 0
            else 1.0
        )
        center = target_weight * target_center + (1.0 - target_weight) * float(source_center)
        safe_target_scale = max(target_scale, minimum_scale)
        if target_scale_degenerate:
            safe_target_scale = float(source_scale)
        log_scale = (
            target_weight * np.log(safe_target_scale)
            + (1.0 - target_weight) * np.log(float(source_scale))
        )
        scale = max(float(np.exp(log_scale)), minimum_scale)

        within_instability = float(np.median(group_instabilities))
        stability_factor = 1.0 / (1.0 + within_instability / max(scale, minimum_scale))
        diversity_factor = target_weight
        degeneracy_factor = 0.5 if target_scale_degenerate else 1.0
        reliability = float(np.clip(stability_factor * diversity_factor * degeneracy_factor, 0.0, 1.0))

        return cls(
            center=float(center),
            scale=float(scale),
            source_center=float(source_center),
            source_scale=float(source_scale),
            effective_group_count=group_count,
            view_count=len(values),
            target_weight=float(target_weight),
            within_group_instability=within_instability,
            reference_reliability=reliability,
            target_scale_degenerate=bool(target_scale_degenerate),
        )

    def standardize(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("query values contain NaN or infinity")
        return (raw - self.center) / self.scale

    def signed_evidence(self, values: np.ndarray) -> np.ndarray:
        """Compress extreme z-scores without clipping or changing their order."""

        return np.arcsinh(self.standardize(values))

    def anomaly_evidence(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(self.signed_evidence(values), 0.0)

    def reliability(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, self.reference_reliability, dtype=np.float64)
