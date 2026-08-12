from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure

from .v3_contracts import (
    BranchEvidenceV3,
    RescueBudgetV3,
    RescueResultV3,
    V3Decision,
)


@dataclass(frozen=True)
class SelectiveRescueConfigV3:
    minimum_text_reliability: float = 0.50
    minimum_evidence_gap: float = 0.10
    image_visual_ambiguity: float = 1.00
    pixel_visual_ambiguity: float = 1.50
    minimum_region_pixels: int = 4
    maximum_region_fraction: float = 0.25
    enable_image_rescue: bool = False
    enable_pixel_rescue: bool = True
    budget: RescueBudgetV3 = RescueBudgetV3()

    def validate(self) -> "SelectiveRescueConfigV3":
        for name, value in (
            ("minimum_text_reliability", self.minimum_text_reliability),
            ("minimum_evidence_gap", self.minimum_evidence_gap),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.image_visual_ambiguity < 0 or self.pixel_visual_ambiguity < 0:
            raise ValueError("visual ambiguity limits must be non-negative")
        if self.minimum_region_pixels < 1:
            raise ValueError("minimum_region_pixels must be positive")
        if not 0 < self.maximum_region_fraction <= 1:
            raise ValueError("maximum_region_fraction must be in (0, 1]")
        self.budget.validate()
        return self


def _filter_regions(candidate: np.ndarray, config: SelectiveRescueConfigV3) -> np.ndarray:
    allowed = np.zeros_like(candidate, dtype=bool)
    maximum_pixels = max(int(np.prod(candidate.shape) * config.maximum_region_fraction), 1)
    components = measure.label(candidate.astype(bool), connectivity=1)
    for region_id in range(1, int(components.max()) + 1):
        region = components == region_id
        size = int(np.sum(region))
        if config.minimum_region_pixels <= size <= maximum_pixels:
            allowed |= region
    return allowed


class HierarchicalSelectiveRescueV3:
    """Label-free visual-anchored text rescue.

    The public interface contains predictions, calibrated evidence and
    reliability only. It cannot receive labels or masks. Text assistance is a
    bounded non-negative residual, so it never suppresses visual evidence.
    """

    def __init__(self, config: SelectiveRescueConfigV3 | None = None) -> None:
        self.config = (config or SelectiveRescueConfigV3()).validate()

    def fuse(
        self, visual: BranchEvidenceV3, text: BranchEvidenceV3
    ) -> RescueResultV3:
        visual.validate()
        text.validate()
        visual_ids = np.asarray(visual.prediction.sample_ids)
        text_ids = np.asarray(text.prediction.sample_ids)
        if not np.array_equal(visual_ids, text_ids):
            raise ValueError("V3 branch sample IDs or ordering differ")

        visual_image = np.asarray(visual.image_anomaly_evidence, dtype=np.float64)
        text_image = np.asarray(text.image_anomaly_evidence, dtype=np.float64)
        visual_pixel = np.asarray(visual.pixel_anomaly_evidence, dtype=np.float64)
        text_pixel = np.asarray(text.pixel_anomaly_evidence, dtype=np.float64)
        visual_image_rel = np.asarray(visual.image_reliability, dtype=np.float64)
        text_image_rel = np.asarray(text.image_reliability, dtype=np.float64)
        text_pixel_rel = np.asarray(text.pixel_reliability, dtype=np.float64)

        image_gap = text_image - visual_image
        image_eligible = text_image_rel >= self.config.minimum_text_reliability
        image_opportunity = (
            self.config.enable_image_rescue
            & image_eligible
            & (np.abs(visual_image) <= self.config.image_visual_ambiguity)
            & (image_gap >= self.config.minimum_evidence_gap)
        )
        image_residual = np.where(
            image_opportunity,
            np.clip(image_gap, 0.0, self.config.budget.max_image_residual),
            0.0,
        )

        pixel_gap = text_pixel - visual_pixel
        pixel_candidate = (
            self.config.enable_pixel_rescue
            & (text_pixel_rel >= self.config.minimum_text_reliability)
            & (visual_pixel <= self.config.pixel_visual_ambiguity)
            & (pixel_gap >= self.config.minimum_evidence_gap)
        )
        pixel_allowed = np.zeros_like(pixel_candidate, dtype=bool)
        if self.config.enable_pixel_rescue:
            for index in range(len(pixel_candidate)):
                pixel_allowed[index] = _filter_regions(pixel_candidate[index], self.config)
        pixel_residual = np.where(
            pixel_allowed,
            np.clip(pixel_gap, 0.0, self.config.budget.max_pixel_residual),
            0.0,
        )

        decisions = np.full(len(visual_ids), V3Decision.NO_RESCUE_OPPORTUNITY.value, dtype="<U32")
        ineligible = (~image_eligible) & ~np.any(pixel_allowed, axis=(1, 2))
        decisions[ineligible] = V3Decision.TEXT_INELIGIBLE.value
        applied = (image_residual > 0) | np.any(pixel_residual > 0, axis=(1, 2))
        decisions[applied] = V3Decision.TEXT_RESCUE_APPLIED.value
        visual_confident = (
            visual_image_rel >= text_image_rel
        ) & (np.abs(visual_image) > self.config.image_visual_ambiguity) & ~applied
        decisions[visual_confident] = V3Decision.VISUAL_CONFIDENT.value

        result = RescueResultV3(
            sample_ids=visual_ids,
            image_scores=(visual_image + image_residual).astype(np.float32),
            pixel_maps=(visual_pixel + pixel_residual).astype(np.float32),
            image_residual=image_residual.astype(np.float32),
            pixel_residual=pixel_residual.astype(np.float32),
            image_rescue_allowed=np.asarray(image_opportunity, dtype=bool),
            pixel_rescue_allowed=pixel_allowed,
            decisions=decisions,
            features={
                "image_evidence_gap": image_gap.astype(np.float32),
                "pixel_evidence_gap": pixel_gap.astype(np.float32),
                "image_text_eligible": image_eligible,
                "pixel_candidate": pixel_candidate,
                "image_budget_clipped": image_gap > self.config.budget.max_image_residual,
                "pixel_budget_clipped": pixel_gap > self.config.budget.max_pixel_residual,
            },
        )
        return result.validate()

