from __future__ import annotations

import numpy as np

from .calibration import BranchCalibration
from .contracts import BranchPrediction, FusionResult
from .features import as_probability, extract_pair_features


class ConfidenceRouter:
    """Ground-truth-free router based on branch uncertainty.

    The implementation is deliberately deterministic. Seed-0 development may
    replace its hyperparameters only before the design is locked.
    """

    def __init__(
        self,
        temperature: float = 0.20,
        image_temperature: float | None = None,
        pixel_temperature: float | None = None,
        min_weight: float = 0.05,
        decision_margin: float = 0.15,
        pixel_level: bool = True,
        visual_calibration: BranchCalibration | None = None,
        text_calibration: BranchCalibration | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if image_temperature is not None and image_temperature <= 0:
            raise ValueError("image_temperature must be positive")
        if pixel_temperature is not None and pixel_temperature <= 0:
            raise ValueError("pixel_temperature must be positive")
        if not 0 <= min_weight < 0.5:
            raise ValueError("min_weight must be in [0, 0.5)")
        self.temperature = temperature
        # Keep the original single-temperature contract as the default.  A
        # split temperature is an explicit seed-0 ablation, not a silent
        # behaviour change for existing runs.
        self.image_temperature = image_temperature or temperature
        self.pixel_temperature = pixel_temperature or temperature
        self.min_weight = min_weight
        self.decision_margin = decision_margin
        self.pixel_level = pixel_level
        self.visual_calibration = visual_calibration
        self.text_calibration = text_calibration

    def _weight(
        self, visual_u: np.ndarray, text_u: np.ndarray, temperature: float
    ) -> np.ndarray:
        visual_logit = -np.asarray(visual_u, dtype=np.float64) / temperature
        text_logit = -np.asarray(text_u, dtype=np.float64) / temperature
        maximum = np.maximum(visual_logit, text_logit)
        visual_exp = np.exp(visual_logit - maximum)
        text_exp = np.exp(text_logit - maximum)
        weight = visual_exp / (visual_exp + text_exp)
        return np.clip(weight, self.min_weight, 1.0 - self.min_weight)

    def fuse(self, visual: BranchPrediction, text: BranchPrediction) -> FusionResult:
        if self.visual_calibration is not None:
            visual = self.visual_calibration.apply(visual)
        if self.text_calibration is not None:
            text = self.text_calibration.apply(text)
        features = extract_pair_features(visual, text)
        image_weight = self._weight(
            features["visual_image_uncertainty"],
            features["text_image_uncertainty"],
            self.image_temperature,
        )
        if self.pixel_level:
            pixel_weight = self._weight(
                features["visual_pixel_uncertainty"],
                features["text_pixel_uncertainty"],
                self.pixel_temperature,
            )
        else:
            pixel_weight = image_weight[:, None, None]
        image_scores = image_weight * as_probability(visual.image_scores) + (
            1.0 - image_weight
        ) * as_probability(text.image_scores)
        pixel_maps = pixel_weight * as_probability(visual.pixel_maps) + (
            1.0 - pixel_weight
        ) * as_probability(text.pixel_maps)
        decisions = np.full(len(image_weight), "weighted_fusion", dtype="<U16")
        decisions[image_weight >= 0.5 + self.decision_margin] = "visual"
        decisions[image_weight <= 0.5 - self.decision_margin] = "text"
        return FusionResult(
            sample_ids=np.asarray(visual.sample_ids),
            image_scores=image_scores.astype(np.float32),
            pixel_maps=pixel_maps.astype(np.float32),
            visual_weights=image_weight.astype(np.float32),
            visual_pixel_weights=pixel_weight.astype(np.float32),
            decisions=decisions,
            features=features,
        )
