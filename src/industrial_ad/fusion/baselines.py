from __future__ import annotations

import numpy as np

from .calibration import BranchCalibration
from .contracts import BranchPrediction, FusionResult
from .features import as_probability, extract_pair_features


def single_branch_fusion(
    branch: BranchPrediction,
    source: str,
    calibration: BranchCalibration | None = None,
) -> FusionResult:
    """Return one branch through the common fusion-result contract."""

    if source not in {"visual", "text"}:
        raise ValueError("source must be 'visual' or 'text'")
    branch.validate()
    if calibration is not None:
        branch = calibration.apply(branch)
    image_scores = as_probability(branch.image_scores).astype(np.float32)
    pixel_maps = as_probability(branch.pixel_maps).astype(np.float32)
    visual_weight = 1.0 if source == "visual" else 0.0
    return FusionResult(
        sample_ids=np.asarray(branch.sample_ids),
        image_scores=image_scores,
        pixel_maps=pixel_maps,
        visual_weights=np.full(len(image_scores), visual_weight, dtype=np.float32),
        visual_pixel_weights=np.full(
            pixel_maps.shape, visual_weight, dtype=np.float32
        ),
        decisions=np.full(len(image_scores), source, dtype="<U16"),
        features={},
    )


class FixedWeightFusion:
    """Declared fixed-weight control that never reads labels or test statistics."""

    def __init__(
        self,
        image_visual_weight: float = 0.5,
        pixel_visual_weight: float | None = None,
        visual_calibration: BranchCalibration | None = None,
        text_calibration: BranchCalibration | None = None,
    ) -> None:
        pixel_visual_weight = (
            image_visual_weight
            if pixel_visual_weight is None
            else pixel_visual_weight
        )
        if not 0.0 <= image_visual_weight <= 1.0:
            raise ValueError("image_visual_weight must be in [0, 1]")
        if not 0.0 <= pixel_visual_weight <= 1.0:
            raise ValueError("pixel_visual_weight must be in [0, 1]")
        self.image_visual_weight = float(image_visual_weight)
        self.pixel_visual_weight = float(pixel_visual_weight)
        self.visual_calibration = visual_calibration
        self.text_calibration = text_calibration

    def fuse(
        self, visual: BranchPrediction, text: BranchPrediction
    ) -> FusionResult:
        visual.validate()
        text.validate()
        if self.visual_calibration is not None:
            visual = self.visual_calibration.apply(visual)
        if self.text_calibration is not None:
            text = self.text_calibration.apply(text)
        features = extract_pair_features(visual, text)
        image_scores = (
            self.image_visual_weight * as_probability(visual.image_scores)
            + (1.0 - self.image_visual_weight) * as_probability(text.image_scores)
        )
        pixel_maps = (
            self.pixel_visual_weight * as_probability(visual.pixel_maps)
            + (1.0 - self.pixel_visual_weight) * as_probability(text.pixel_maps)
        )
        return FusionResult(
            sample_ids=np.asarray(visual.sample_ids),
            image_scores=image_scores.astype(np.float32),
            pixel_maps=pixel_maps.astype(np.float32),
            visual_weights=np.full(
                len(image_scores), self.image_visual_weight, dtype=np.float32
            ),
            visual_pixel_weights=np.full(
                pixel_maps.shape, self.pixel_visual_weight, dtype=np.float32
            ),
            decisions=np.full(len(image_scores), "fixed_weight", dtype="<U16"),
            features=features,
        )
