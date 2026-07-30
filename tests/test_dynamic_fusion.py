from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_ad.fusion import BranchPrediction, ConfidenceRouter


def branch(scores: list[float], maps: np.ndarray | None = None) -> BranchPrediction:
    values = np.asarray(scores, dtype=np.float32)
    if maps is None:
        maps = np.broadcast_to(values[:, None, None], (len(values), 4, 5)).copy()
    return BranchPrediction(
        sample_ids=np.asarray([f"sample-{index}" for index in range(len(values))]),
        image_scores=values,
        pixel_maps=maps,
    )


def test_shapes_and_finite_outputs() -> None:
    result = ConfidenceRouter().fuse(branch([0.1, 0.9]), branch([0.2, 0.8]))
    assert result.image_scores.shape == (2,)
    assert result.pixel_maps.shape == (2, 4, 5)
    assert result.visual_weights.shape == (2,)
    assert result.visual_pixel_weights.shape == (2, 4, 5)
    assert np.isfinite(result.image_scores).all()
    assert np.isfinite(result.pixel_maps).all()


def test_confident_branch_receives_more_weight() -> None:
    visual = branch([0.99, 0.50])
    text = branch([0.50, 0.01])
    result = ConfidenceRouter().fuse(visual, text)
    assert result.visual_weights[0] > 0.5
    assert result.visual_weights[1] < 0.5


def test_extreme_values_are_numerically_stable() -> None:
    result = ConfidenceRouter(temperature=0.01).fuse(
        branch([-1000.0, 1000.0]), branch([1000.0, -1000.0])
    )
    assert np.isfinite(result.image_scores).all()
    assert np.isfinite(result.pixel_maps).all()
    assert ((0 <= result.image_scores) & (result.image_scores <= 1)).all()


def test_rejects_misaligned_samples() -> None:
    visual = branch([0.1, 0.9])
    text = BranchPrediction(
        sample_ids=np.asarray(["other-0", "other-1"]),
        image_scores=np.asarray([0.1, 0.9]),
        pixel_maps=np.zeros((2, 4, 5)),
    )
    with pytest.raises(ValueError, match="sample_ids"):
        ConfidenceRouter().fuse(visual, text)


def test_rejects_nan() -> None:
    maps = np.zeros((2, 4, 5))
    maps[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ConfidenceRouter().fuse(branch([0.1, 0.9], maps), branch([0.1, 0.9]))
