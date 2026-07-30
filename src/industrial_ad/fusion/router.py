from __future__ import annotations

import numpy as np

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
        min_weight: float = 0.05,
        decision_margin: float = 0.15,
        pixel_level: bool = True,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= min_weight < 0.5:
            raise ValueError("min_weight must be in [0, 0.5)")
        self.temperature = temperature
        self.min_weight = min_weight
        self.decision_margin = decision_margin
        self.pixel_level = pixel_level

    def _weight(self, visual_u: np.ndarray, text_u: np.ndarray) -> np.ndarray:
        visual_logit = -np.asarray(visual_u, dtype=np.float64) / self.temperature
        text_logit = -np.asarray(text_u, dtype=np.float64) / self.temperature
        maximum = np.maximum(visual_logit, text_logit)
        visual_exp = np.exp(visual_logit - maximum)
        text_exp = np.exp(text_logit - maximum)
        weight = visual_exp / (visual_exp + text_exp)
        return np.clip(weight, self.min_weight, 1.0 - self.min_weight)

    def fuse(self, visual: BranchPrediction, text: BranchPrediction) -> FusionResult:
        features = extract_pair_features(visual, text)
        image_weight = self._weight(
            features["visual_image_uncertainty"], features["text_image_uncertainty"]
        )
        if self.pixel_level:
            pixel_weight = self._weight(
                features["visual_pixel_uncertainty"],
                features["text_pixel_uncertainty"],
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
