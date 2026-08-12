"""V3.2 data contracts: multi-branch evidence, candidate regions, hierarchical routing.

V3.2 separates AdaptCLIP into its constituent branches (text adapter, visual
adapter, PQ adapter, final fusion) plus AnomalyDINO as the visual anchor.
The router uses region proposals instead of pixel-level comparisons, and
requires multiple independent reliability signals before allowing text rescue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class V3_2ReasonCode(str, Enum):
    """Reason codes for V3.2 routing decisions."""

    NO_TEXT_OPPORTUNITY = "no_text_opportunity"
    BACKGROUND_RISK = "background_risk"
    TEXT_UNSTABLE = "text_unstable"
    PROMPT_DISAGREEMENT = "prompt_disagreement"
    PQ_DISAGREEMENT = "pq_disagreement"
    VISUAL_CONFIDENT_NORMAL = "visual_confident_normal"
    VISUAL_CONFIDENT_ANOMALY = "visual_confident_anomaly"
    REGION_TOO_SMALL = "region_too_small"
    REGION_TOO_LARGE = "region_too_large"
    RELIABILITY_TOO_LOW = "reliability_too_low"
    AUGMENTATION_UNSTABLE = "augmentation_unstable"
    RESCUE_APPLIED = "rescue_applied"
    RESCUE_BUDGET_CLIPPED = "rescue_budget_clipped"
    TEXT_NOT_EXCEEDING = "text_not_exceeding"
    VISUAL_FALLBACK = "visual_fallback"
    NO_CANDIDATE_REGIONS = "no_candidate_regions"


class V3_2DecisionLevel(str, Enum):
    """Three-level routing hierarchy."""

    L1_OPPORTUNITY = "l1_opportunity"
    L2_RELIABILITY = "l2_reliability"
    L3_MODIFICATION = "l3_modification"


@dataclass(frozen=True)
class AdaptCLIPBranches:
    """Decomposed AdaptCLIP internal outputs.

    Attributes
    ----------
    text_adapter : normalized [N] image scores from textual adapter only.
    text_adapter_maps : [N, H, W] pixel anomaly maps from textual adapter.
    visual_adapter : normalized [N] image scores from visual adapter only.
    visual_adapter_maps : [N, H, W] pixel maps from visual adapter.
    pq_adapter : normalized [N] image scores from PQ adapter only.
    pq_adapter_maps : [N, H, W] pixel maps from PQ adapter.
    final_fused : normalized [N] image scores from AdaptCLIP internal fusion.
    final_fused_maps : [N, H, W] pixel maps from AdaptCLIP internal fusion.
    align_scores : [N, H, W] alignment/query-match pixel maps (optional).
    sample_ids : [N] sample identifier strings.
    """

    sample_ids: np.ndarray
    text_adapter: np.ndarray
    text_adapter_maps: np.ndarray
    visual_adapter: np.ndarray
    visual_adapter_maps: np.ndarray
    pq_adapter: np.ndarray
    pq_adapter_maps: np.ndarray
    final_fused: np.ndarray
    final_fused_maps: np.ndarray
    align_scores: np.ndarray | None = None
    prompt_scores: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    augmented_views: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def validate(self) -> "AdaptCLIPBranches":
        ids = np.asarray(self.sample_ids).reshape(-1)
        n = len(ids)
        for name, arr in [
            ("text_adapter", self.text_adapter),
            ("visual_adapter", self.visual_adapter),
            ("pq_adapter", self.pq_adapter),
            ("final_fused", self.final_fused),
        ]:
            a = np.asarray(arr, dtype=np.float64).reshape(-1)
            if len(a) != n:
                raise ValueError(f"{name} image scores have wrong length")
            if not np.isfinite(a).all():
                raise ValueError(f"{name} contains NaN/inf")
        for name, maps in [
            ("text_adapter_maps", self.text_adapter_maps),
            ("visual_adapter_maps", self.visual_adapter_maps),
            ("pq_adapter_maps", self.pq_adapter_maps),
            ("final_fused_maps", self.final_fused_maps),
        ]:
            m = np.asarray(maps, dtype=np.float64)
            if m.ndim != 3 or m.shape[0] != n:
                raise ValueError(f"{name} must be [N,H,W]")
            if not np.isfinite(m).all():
                raise ValueError(f"{name} contains NaN/inf")
        if self.align_scores is not None:
            a = np.asarray(self.align_scores, dtype=np.float64)
            if a.ndim != 3 or a.shape[0] != n:
                raise ValueError("align_scores must be [N,H,W]")
        return self


@dataclass(frozen=True)
class CandidateRegion:
    """A proposed anomalous region for text rescue evaluation.

    Attributes
    ----------
    mask : [H, W] boolean mask of the region.
    center_yx : (row, col) center of mass.
    area : pixel count.
    compactness : area / bounding_box_area.
    text_score_max : peak text adapter anomaly evidence in region.
    pq_score_max : peak PQ adapter anomaly evidence in region.
    visual_score_mean : mean visual (AnomalyDINO) anomaly evidence.
    visual_score_max : peak visual evidence (may be weak).
    source_branches : which branches nominated this region.
    """

    mask: np.ndarray
    center_yx: tuple[float, float]
    area: int
    compactness: float
    text_score_max: float
    pq_score_max: float
    visual_score_mean: float
    visual_score_max: float
    source_branches: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.area <= 0:
            raise ValueError("region area must be positive")
        if not 0 <= self.compactness <= 1:
            raise ValueError("compactness must be in [0, 1]")


@dataclass(frozen=True)
class ReliabilityEvidenceV3_2:
    """Five reliability evidence signals for a candidate region.

    All scores are in [0, 1] where higher = more reliable.
    These are label-free and use only frozen normal references.
    """

    normal_range_excess: float
    spatial_stability: float
    augmentation_consistency: float
    prompt_consistency: float
    branch_consistency: float
    background_risk: float

    def __post_init__(self):
        for name in [
            "normal_range_excess",
            "spatial_stability",
            "augmentation_consistency",
            "prompt_consistency",
            "branch_consistency",
            "background_risk",
        ]:
            v = getattr(self, name)
            if not 0 <= v <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    def combined_reliability(self, weights: dict[str, float] | None = None) -> float:
        """Weighted combination of reliability signals."""
        if weights is None:
            weights = {
                "normal_range_excess": 0.25,
                "spatial_stability": 0.15,
                "augmentation_consistency": 0.20,
                "prompt_consistency": 0.20,
                "branch_consistency": 0.20,
            }
        total = 0.0
        w_sum = 0.0
        for key, w in weights.items():
            total += getattr(self, key) * w
            w_sum += w
        return total / w_sum if w_sum > 0 else 0.0

    def combined_risk(self) -> float:
        """Risk score: inverse of reliability, weighted by background risk."""
        reliability = self.combined_reliability()
        return (1.0 - reliability) * (0.5 + 0.5 * self.background_risk)


@dataclass(frozen=True)
class RegionRoutingDecision:
    """Per-region routing outcome with reason code and metadata."""

    region: CandidateRegion
    reason: V3_2ReasonCode
    reliability: ReliabilityEvidenceV3_2 | None = None
    rescue_applied: bool = False
    rescue_residual: float = 0.0
    max_allowed_residual: float = 0.0
    level_reached: V3_2DecisionLevel = V3_2DecisionLevel.L1_OPPORTUNITY


@dataclass(frozen=True)
class V3_2FusionResult:
    """Final fusion output with full audit trail.

    Attributes
    ----------
    sample_ids : [N] sample identifiers.
    pixel_maps : [N, H, W] final fused pixel anomaly maps.
    image_scores : [N] final image anomaly scores (visual unchanged per protocol).
    pixel_residual : [N, H, W] non-negative text rescue residual added to visual.
    rescue_mask : [N, H, W] boolean mask of pixels where rescue was applied.
    visual_fallback : [N] boolean: True if text rescue was fully rejected.
    decisions : [N] list of per-region routing decisions.
    stats : dict with aggregate statistics.
    """

    sample_ids: np.ndarray
    pixel_maps: np.ndarray
    image_scores: np.ndarray
    pixel_residual: np.ndarray
    rescue_mask: np.ndarray
    visual_fallback: np.ndarray
    decisions: list[list[RegionRoutingDecision]]
    stats: dict = field(default_factory=dict)

    def validate(self) -> "V3_2FusionResult":
        ids = np.asarray(self.sample_ids)
        n = len(ids)
        pixel = np.asarray(self.pixel_maps, dtype=np.float64)
        residual = np.asarray(self.pixel_residual, dtype=np.float64)
        rescue = np.asarray(self.rescue_mask, dtype=bool)
        if pixel.ndim != 3 or pixel.shape[0] != n:
            raise ValueError("pixel_maps must be [N,H,W]")
        if residual.shape != pixel.shape:
            raise ValueError("pixel_residual shape mismatch")
        if rescue.shape != pixel.shape:
            raise ValueError("rescue_mask shape mismatch")
        if np.any(residual < 0):
            raise ValueError("pixel_residual must be non-negative")
        if np.any(rescue & ~(residual > 0)):
            raise ValueError("rescue_mask true but no residual")
        if np.any((residual > 0) & ~rescue):
            raise ValueError("residual positive but rescue_mask false")
        if not np.isfinite(pixel).all():
            raise ValueError("pixel_maps contain NaN/inf")
        return self


@dataclass(frozen=True)
class V3_2Config:
    """Configuration for V3.2 hierarchical selective rescue router.

    Parameters are grouped by routing level.
    """

    # L1: Region proposal
    min_region_area: int = 16
    max_region_area_fraction: float = 0.05
    min_region_compactness: float = 0.1
    text_excess_threshold: float = 1.5  # z-score beyond normal ref
    pq_excess_threshold: float = 1.2  # z-score beyond normal ref

    # L2: Reliability
    min_combined_reliability: float = 0.55
    min_branch_consistency: float = 0.3
    max_background_risk: float = 0.60
    min_augmentation_consistency: float = 0.4
    min_prompt_consistency: float = 0.4

    # L3: Modification budget
    max_pixel_residual: float = 0.12
    max_rescue_area_fraction: float = 0.01  # per-image max rescue area
    reliability_residual_scale: bool = True

    # Visual anchor
    visual_ambiguous_low: float = 0.5   # below this: visual says normal
    visual_ambiguous_high: float = 2.0  # above this: visual says anomaly
    visual_fallback: bool = True

    # Image-level (disabled by default per protocol)
    enable_image_rescue: bool = False

    # Branch normalization (V3.2 fix: scale branches to comparable dynamic range)
    normalize_branches: bool = True

    def validate(self) -> "V3_2Config":
        if self.min_region_area < 4:
            raise ValueError("min_region_area must be >= 4")
        if not 0 < self.max_region_area_fraction <= 0.5:
            raise ValueError("max_region_area_fraction must be in (0, 0.5]")
        if not 0 <= self.min_combined_reliability <= 1:
            raise ValueError("min_combined_reliability must be in [0, 1]")
        if self.max_pixel_residual < 0:
            raise ValueError("max_pixel_residual must be non-negative")
        if not 0 < self.max_rescue_area_fraction <= 0.1:
            raise ValueError("max_rescue_area_fraction must be in (0, 0.1]")
        return self
