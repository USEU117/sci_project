from __future__ import annotations

import numpy as np

from .contracts import BranchPrediction


EPS = 1e-6


def as_probability(values: np.ndarray) -> np.ndarray:
    """Use already calibrated probabilities, clipping only for stability."""
    return np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)


def binary_entropy(values: np.ndarray) -> np.ndarray:
    probability = as_probability(values)
    return -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))


def branch_uncertainty(branch: BranchPrediction) -> tuple[np.ndarray, np.ndarray]:
    image = (
        np.asarray(branch.image_uncertainty, dtype=np.float64)
        if branch.image_uncertainty is not None
        else binary_entropy(branch.image_scores)
    )
    pixel = (
        np.asarray(branch.pixel_uncertainty, dtype=np.float64)
        if branch.pixel_uncertainty is not None
        else binary_entropy(branch.pixel_maps)
    )
    return np.clip(image, 0.0, 1.0), np.clip(pixel, 0.0, 1.0)


def extract_pair_features(
    visual: BranchPrediction, text: BranchPrediction
) -> dict[str, np.ndarray]:
    visual.validate()
    text.validate()
    if not np.array_equal(visual.sample_ids, text.sample_ids):
        raise ValueError("branch sample_ids or ordering differ")
    if np.asarray(visual.pixel_maps).shape != np.asarray(text.pixel_maps).shape:
        raise ValueError("branch pixel-map shapes differ")
    visual_image_u, visual_pixel_u = branch_uncertainty(visual)
    text_image_u, text_pixel_u = branch_uncertainty(text)
    return {
        "visual_image_uncertainty": visual_image_u,
        "text_image_uncertainty": text_image_u,
        "visual_pixel_uncertainty": visual_pixel_u,
        "text_pixel_uncertainty": text_pixel_u,
        "image_disagreement": np.abs(
            as_probability(visual.image_scores) - as_probability(text.image_scores)
        ),
        "pixel_disagreement": np.abs(
            as_probability(visual.pixel_maps) - as_probability(text.pixel_maps)
        ),
    }
