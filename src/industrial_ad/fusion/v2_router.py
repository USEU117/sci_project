from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import BranchPrediction, FusionResult
from .features import binary_entropy, spatial_response_concentration
from .v2_calibration import BranchV2Calibration


def _smooth_3x3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (1, 1), (1, 1)), mode="edge")
    total = np.zeros_like(values, dtype=np.float64)
    for row in range(3):
        for column in range(3):
            total += padded[:, row : row + values.shape[1], column : column + values.shape[2]]
    return total / 9.0


@dataclass(frozen=True)
class SafeRouterV2Config:
    support_tolerance: float = 3.0
    minimum_disagreement: float = 0.05
    uncertainty_margin: float = 0.05
    concentration_tolerance: float = 0.10
    max_image_text_weight: float = 0.15
    max_pixel_text_weight: float = 0.35
    smooth_pixel_weights: bool = True

    def validate(self) -> "SafeRouterV2Config":
        if self.support_tolerance < 0:
            raise ValueError("support_tolerance must be non-negative")
        if not 0 <= self.minimum_disagreement <= 1:
            raise ValueError("minimum_disagreement must be in [0, 1]")
        if not 0 <= self.uncertainty_margin <= 1:
            raise ValueError("uncertainty_margin must be in [0, 1]")
        if not 0 <= self.concentration_tolerance <= 1:
            raise ValueError("concentration_tolerance must be in [0, 1]")
        if not 0 <= self.max_image_text_weight <= 0.5:
            raise ValueError("max_image_text_weight must be in [0, 0.5]")
        if not 0 <= self.max_pixel_text_weight <= 0.5:
            raise ValueError("max_pixel_text_weight must be in [0, 0.5]")
        return self


class SafeVisualDefaultRouterV2:
    """Conservative split router with a visual-default failure path.

    It accepts predictions and normal-reference calibration only.  Ground-truth
    labels and masks are absent from the public interface by construction.
    """

    def __init__(
        self,
        visual_calibration: BranchV2Calibration,
        text_calibration: BranchV2Calibration,
        config: SafeRouterV2Config | None = None,
    ) -> None:
        self.visual_calibration = visual_calibration
        self.text_calibration = text_calibration
        self.config = (config or SafeRouterV2Config()).validate()

    def fuse(self, visual: BranchPrediction, text: BranchPrediction) -> FusionResult:
        visual.validate()
        text.validate()
        if not np.array_equal(visual.sample_ids, text.sample_ids):
            raise ValueError("branch sample_ids or ordering differ")
        if np.asarray(visual.pixel_maps).shape != np.asarray(text.pixel_maps).shape:
            raise ValueError("branch pixel-map shapes differ")

        calibrated_visual = self.visual_calibration.apply(visual)
        calibrated_text = self.text_calibration.apply(text)
        visual_image = np.asarray(calibrated_visual.image_scores, dtype=np.float64)
        text_image = np.asarray(calibrated_text.image_scores, dtype=np.float64)
        visual_pixel = np.asarray(calibrated_visual.pixel_maps, dtype=np.float64)
        text_pixel = np.asarray(calibrated_text.pixel_maps, dtype=np.float64)

        vi_oos, vi_support, vi_excess = self.visual_calibration.image.upper_support_evidence(
            visual.image_scores, tolerance=self.config.support_tolerance
        )
        ti_oos, ti_support, ti_excess = self.text_calibration.image.upper_support_evidence(
            text.image_scores, tolerance=self.config.support_tolerance
        )
        vp_oos, vp_support, vp_excess = self.visual_calibration.pixel.upper_support_evidence(
            visual.pixel_maps, tolerance=self.config.support_tolerance
        )
        tp_oos, tp_support, tp_excess = self.text_calibration.pixel.upper_support_evidence(
            text.pixel_maps, tolerance=self.config.support_tolerance
        )

        calibration_warning = bool(
            self.visual_calibration.calibration_warning
            or self.text_calibration.calibration_warning
        )
        visual_image_uncertainty = binary_entropy(visual_image)
        text_image_uncertainty = binary_entropy(text_image)
        image_disagreement = np.abs(visual_image - text_image)
        image_advantage = visual_image_uncertainty - text_image_uncertainty
        image_allowed = (
            (~vi_oos)
            & (~ti_oos)
            & (image_disagreement >= self.config.minimum_disagreement)
            & (image_advantage >= self.config.uncertainty_margin)
            & (not calibration_warning)
        )
        image_strength = np.clip(
            (image_advantage - self.config.uncertainty_margin)
            / max(1.0 - self.config.uncertainty_margin, 1e-12),
            0.0,
            1.0,
        )
        image_text_weight = np.where(
            image_allowed,
            self.config.max_image_text_weight * image_strength * ti_support,
            0.0,
        )
        image_visual_weight = 1.0 - image_text_weight
        fused_image = image_visual_weight * visual_image + image_text_weight * text_image

        visual_pixel_uncertainty = binary_entropy(visual_pixel)
        text_pixel_uncertainty = binary_entropy(text_pixel)
        pixel_disagreement = np.abs(visual_pixel - text_pixel)
        pixel_advantage = visual_pixel_uncertainty - text_pixel_uncertainty
        visual_concentration = spatial_response_concentration(visual_pixel)
        text_concentration = spatial_response_concentration(text_pixel)
        spatially_safe = (
            text_concentration + self.config.concentration_tolerance
            >= visual_concentration
        )[:, None, None]
        pixel_allowed = (
            (~vp_oos)
            & (~tp_oos)
            & spatially_safe
            & (pixel_disagreement >= self.config.minimum_disagreement)
            & (pixel_advantage >= self.config.uncertainty_margin)
            & (not calibration_warning)
        )
        pixel_strength = np.clip(
            (pixel_advantage - self.config.uncertainty_margin)
            / max(1.0 - self.config.uncertainty_margin, 1e-12),
            0.0,
            1.0,
        )
        pixel_text_weight = np.where(
            pixel_allowed,
            self.config.max_pixel_text_weight * pixel_strength * tp_support,
            0.0,
        )
        if self.config.smooth_pixel_weights:
            pixel_text_weight = _smooth_3x3(pixel_text_weight)
            # Keep the pixel route independent from the image route. Reapply
            # the pixel-level gate after smoothing so neighboring pixels cannot
            # be activated when their own evidence failed.
            pixel_text_weight = np.where(pixel_allowed, pixel_text_weight, 0.0)
        pixel_text_weight = np.clip(
            pixel_text_weight, 0.0, self.config.max_pixel_text_weight
        )
        pixel_visual_weight = 1.0 - pixel_text_weight
        fused_pixel = pixel_visual_weight * visual_pixel + pixel_text_weight * text_pixel

        decisions = np.full(len(image_visual_weight), "use_visual_default", dtype="<U24")
        decisions[image_text_weight > 0] = "allow_text_assist"
        decisions[vi_oos | ti_oos] = "out_of_support"
        if calibration_warning:
            decisions[:] = "calibration_warning"

        return FusionResult(
            sample_ids=np.asarray(visual.sample_ids),
            image_scores=fused_image.astype(np.float32),
            pixel_maps=fused_pixel.astype(np.float32),
            visual_weights=image_visual_weight.astype(np.float32),
            visual_pixel_weights=pixel_visual_weight.astype(np.float32),
            decisions=decisions,
            features={
                "visual_image_out_of_support": vi_oos,
                "text_image_out_of_support": ti_oos,
                "visual_pixel_out_of_support": vp_oos,
                "text_pixel_out_of_support": tp_oos,
                "visual_image_support_confidence": vi_support,
                "text_image_support_confidence": ti_support,
                "visual_pixel_support_confidence": vp_support,
                "text_pixel_support_confidence": tp_support,
                "visual_image_support_excess": vi_excess,
                "text_image_support_excess": ti_excess,
                "visual_pixel_support_excess": vp_excess,
                "text_pixel_support_excess": tp_excess,
                "image_disagreement": image_disagreement,
                "pixel_disagreement": pixel_disagreement,
                "visual_image_uncertainty": visual_image_uncertainty,
                "text_image_uncertainty": text_image_uncertainty,
                "visual_pixel_uncertainty": visual_pixel_uncertainty,
                "text_pixel_uncertainty": text_pixel_uncertainty,
                "visual_response_concentration": visual_concentration,
                "text_response_concentration": text_concentration,
                "image_text_assist_allowed": image_allowed,
                "pixel_text_assist_allowed": pixel_allowed,
                "calibration_warning": np.full(
                    len(image_visual_weight), calibration_warning, dtype=bool
                ),
            },
        )
