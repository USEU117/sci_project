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


def spatial_response_concentration(pixel_maps: np.ndarray) -> np.ndarray:
    """Return 0 for uniform maps and values near 1 for concentrated maps."""

    maps = as_probability(pixel_maps)
    if maps.ndim != 3:
        raise ValueError(f"pixel_maps must be [N,H,W], got {maps.shape}")
    flattened = maps.reshape(len(maps), -1)
    mass = flattened / np.maximum(flattened.sum(axis=1, keepdims=True), EPS)
    entropy = -(mass * np.log(np.maximum(mass, EPS))).sum(axis=1)
    maximum_entropy = np.log(flattened.shape[1])
    if maximum_entropy <= 0:
        return np.ones(len(maps), dtype=np.float64)
    return np.clip(1.0 - entropy / maximum_entropy, 0.0, 1.0)


def augmentation_consistency(
    source_ids: np.ndarray,
    image_probabilities: np.ndarray,
    pixel_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure prediction variation across deterministic views of each source."""

    sources = np.asarray(source_ids).astype(str).reshape(-1)
    image = as_probability(image_probabilities).reshape(-1)
    pixel = as_probability(pixel_probabilities)
    if pixel.ndim != 3:
        raise ValueError(f"pixel_probabilities must be [N,H,W], got {pixel.shape}")
    if not (len(sources) == len(image) == len(pixel)):
        raise ValueError("source_ids, image_probabilities and pixel_probabilities need equal N")
    unique_sources = np.asarray(list(dict.fromkeys(sources.tolist())))
    image_std: list[float] = []
    pixel_std: list[np.ndarray] = []
    view_counts: list[int] = []
    for source in unique_sources:
        index = np.flatnonzero(sources == source)
        if len(index) < 2:
            raise ValueError(f"source {source} requires at least 2 views")
        image_std.append(float(np.std(image[index], ddof=0)))
        pixel_std.append(np.std(pixel[index], axis=0, ddof=0))
        view_counts.append(len(index))
    return {
        "source_ids": unique_sources,
        "view_counts": np.asarray(view_counts, dtype=np.int32),
        "image_view_std": np.asarray(image_std, dtype=np.float64),
        "pixel_view_std": np.stack(pixel_std).astype(np.float64),
    }


def shot_sensitivity(
    image_probabilities: np.ndarray, pixel_probabilities: np.ndarray
) -> dict[str, np.ndarray]:
    """Measure variation across frozen shot settings, with shape [K,N,...]."""

    image = as_probability(image_probabilities)
    pixel = as_probability(pixel_probabilities)
    if image.ndim != 2:
        raise ValueError(f"image_probabilities must be [K,N], got {image.shape}")
    if pixel.ndim != 4:
        raise ValueError(f"pixel_probabilities must be [K,N,H,W], got {pixel.shape}")
    if image.shape[:2] != pixel.shape[:2]:
        raise ValueError("image and pixel shot arrays must share [K,N]")
    if image.shape[0] < 2:
        raise ValueError("shot sensitivity requires at least 2 shot settings")
    return {
        "image_shot_std": np.std(image, axis=0, ddof=0),
        "pixel_shot_std": np.std(pixel, axis=0, ddof=0),
    }


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
        "image_agreement": 1.0
        - np.abs(
            as_probability(visual.image_scores) - as_probability(text.image_scores)
        ),
        "pixel_agreement": 1.0
        - np.abs(
            as_probability(visual.pixel_maps) - as_probability(text.pixel_maps)
        ),
        "visual_response_concentration": spatial_response_concentration(
            visual.pixel_maps
        ),
        "text_response_concentration": spatial_response_concentration(
            text.pixel_maps
        ),
    }
