from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_ad.fusion import (
    BranchCalibration,
    BranchPrediction,
    ConfidenceRouter,
    FixedWeightFusion,
    NormalReferencePrediction,
    RobustNormalCalibrator,
    augmentation_consistency,
    build_alignment_plan,
    canonical_sample_id,
    load_category_calibrations,
    shot_sensitivity,
    single_branch_fusion,
    spatial_response_concentration,
)


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


def test_split_temperatures_change_only_the_requested_weight_level() -> None:
    visual = branch([0.9, 0.4])
    text = branch([0.2, 0.6])
    baseline = ConfidenceRouter(temperature=0.20).fuse(visual, text)
    split = ConfidenceRouter(
        temperature=0.20, image_temperature=0.50, pixel_temperature=0.20
    ).fuse(visual, text)
    assert not np.allclose(split.visual_weights, baseline.visual_weights)
    assert np.allclose(split.visual_pixel_weights, baseline.visual_pixel_weights)


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


def test_canonical_sample_id_accepts_path_and_compact_forms() -> None:
    assert canonical_sample_id("candle/test/bad/000.JPG") == "candle-bad-000"
    assert canonical_sample_id(r"candle\test\bad\000.JPG") == "candle-bad-000"
    assert canonical_sample_id("candle/Data/Images/Anomaly/000.JPG") == "candle-bad-000"
    assert canonical_sample_id("candle/Data/Images/Normal/0011.JPG") == "candle-good-0011"
    assert canonical_sample_id("candle-bad-000") == "candle-bad-000"


def test_alignment_plan_reorders_candidate() -> None:
    plan = build_alignment_plan(
        np.asarray(["candle/test/bad/000.JPG", "candle/test/good/001.JPG"]),
        np.asarray(["candle-good-001", "candle-bad-000"]),
    )
    assert plan.candidate_order.tolist() == [1, 0]
    assert plan.order_already_equal is False


def test_alignment_plan_rejects_missing_samples() -> None:
    with pytest.raises(ValueError, match="sample sets differ"):
        build_alignment_plan(
            np.asarray(["candle-bad-000", "candle-good-001"]),
            np.asarray(["candle-bad-000", "candle-good-002"]),
        )


def test_alignment_plan_rejects_normalized_duplicates() -> None:
    with pytest.raises(ValueError, match="not unique"):
        build_alignment_plan(
            np.asarray(["candle/test/bad/000.JPG", "candle-bad-000"]),
            np.asarray(["candle-bad-000", "candle-good-001"]),
        )


def test_normal_reference_calibrator_is_finite_and_bounded() -> None:
    calibrator = RobustNormalCalibrator.fit(np.asarray([-3.0, -2.0, -1.0, 0.0]))
    output = calibrator.transform(np.asarray([-1000.0, 0.0, 1000.0]))
    assert np.isfinite(output).all()
    assert ((output > 0) & (output < 1)).all()
    assert output[0] < output[1] < output[2]


def test_router_accepts_reference_only_calibration() -> None:
    visual_calibration = BranchCalibration.fit(
        np.asarray([0.1, 0.2, 0.3]),
        np.zeros((3, 4, 5), dtype=np.float32),
    )
    text_calibration = BranchCalibration.fit(
        np.asarray([10.0, 11.0, 12.0]),
        np.ones((3, 4, 5), dtype=np.float32),
    )
    result = ConfidenceRouter(
        visual_calibration=visual_calibration,
        text_calibration=text_calibration,
    ).fuse(
        branch([0.2, 0.8]),
        branch([11.0, 10.0]),
    )
    assert np.isfinite(result.image_scores).all()
    assert np.isfinite(result.pixel_maps).all()


def test_normal_reference_contract_requires_multiple_views() -> None:
    cache = NormalReferencePrediction(
        sample_ids=np.asarray(["ref-a-view-0", "ref-a-view-1"]),
        source_ids=np.asarray(["ref-a", "ref-a"]),
        augmentation_ids=np.asarray(["original", "flip"]),
        image_scores=np.asarray([0.1, 0.2]),
        pixel_maps=np.zeros((2, 4, 5)),
    )
    assert cache.validate(min_views_per_source=2).source_count == 1
    with pytest.raises(ValueError, match="at least 3 views"):
        cache.validate(min_views_per_source=3)


def test_calibration_round_trip_dict() -> None:
    calibration = BranchCalibration.fit(
        np.asarray([0.1, 0.2, 0.4]),
        np.asarray([[[0.0]], [[0.2]], [[0.5]]]),
    )
    restored = BranchCalibration.from_dict(calibration.to_dict())
    values = np.asarray([-1.0, 0.0, 1.0])
    assert np.allclose(
        calibration.image.transform(values),
        restored.image.transform(values),
    )


def test_pixel_calibration_uses_per_view_tail_not_background_zeros() -> None:
    maps = np.zeros((5, 4, 4), dtype=np.float64)
    maps[:, 0, 0] = np.asarray([0.01, 0.02, 0.04, 0.08, 0.16])
    calibration = BranchCalibration.fit(
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]), maps
    )
    assert calibration.pixel.scale > 1e-6


def test_load_category_calibrations_enforces_no_test_data() -> None:
    calibration = BranchCalibration.fit(
        np.asarray([0.1, 0.2, 0.4]),
        np.asarray([[[0.0]], [[0.2]], [[0.5]]]),
    )
    payload = {
        "status": "passed",
        "test_predictions_used": False,
        "test_labels_used": False,
        "categories": {
            "candle": {
                "visual": calibration.to_dict(),
                "text": calibration.to_dict(),
            }
        },
    }
    visual, text = load_category_calibrations(payload, "candle")
    assert visual == calibration
    assert text == calibration
    payload["test_labels_used"] = True
    with pytest.raises(ValueError, match="test_labels_used=false"):
        load_category_calibrations(payload, "candle")


def test_load_category_calibrations_rejects_missing_category() -> None:
    payload = {
        "status": "passed",
        "test_predictions_used": False,
        "test_labels_used": False,
        "categories": {},
    }
    with pytest.raises(ValueError, match="category missing"):
        load_category_calibrations(payload, "candle")


def test_single_branch_controls_use_common_contract() -> None:
    visual = single_branch_fusion(branch([0.1, 0.9]), "visual")
    text = single_branch_fusion(branch([0.1, 0.9]), "text")
    assert np.all(visual.visual_weights == 1.0)
    assert np.all(text.visual_weights == 0.0)
    assert visual.decisions.tolist() == ["visual", "visual"]
    assert text.decisions.tolist() == ["text", "text"]


def test_fixed_weight_fusion_matches_declared_formula() -> None:
    visual = branch([0.2, 0.8])
    text = branch([0.6, 0.4])
    result = FixedWeightFusion(
        image_visual_weight=0.25,
        pixel_visual_weight=0.75,
    ).fuse(visual, text)
    assert np.allclose(result.image_scores, [0.5, 0.5])
    assert np.allclose(result.pixel_maps[:, 0, 0], [0.3, 0.7])
    assert np.all(result.visual_weights == 0.25)
    assert np.all(result.visual_pixel_weights == 0.75)


def test_fixed_weight_fusion_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FixedWeightFusion(image_visual_weight=1.1)


def test_spatial_response_concentration_distinguishes_peak_from_uniform() -> None:
    uniform = np.full((1, 4, 4), 0.5)
    peak = np.full((1, 4, 4), 1e-6)
    peak[0, 0, 0] = 1.0
    values = spatial_response_concentration(np.concatenate([uniform, peak]))
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert values[1] > 0.9


def test_augmentation_consistency_returns_per_source_variation() -> None:
    result = augmentation_consistency(
        np.asarray(["a", "a", "b", "b"]),
        np.asarray([0.1, 0.3, 0.7, 0.7]),
        np.asarray(
            [
                np.zeros((2, 2)),
                np.ones((2, 2)),
                np.full((2, 2), 0.5),
                np.full((2, 2), 0.5),
            ]
        ),
    )
    assert result["source_ids"].tolist() == ["a", "b"]
    assert result["view_counts"].tolist() == [2, 2]
    assert result["image_view_std"][0] > result["image_view_std"][1]
    assert result["pixel_view_std"].shape == (2, 2, 2)


def test_augmentation_consistency_requires_two_views() -> None:
    with pytest.raises(ValueError, match="at least 2 views"):
        augmentation_consistency(
            np.asarray(["a"]),
            np.asarray([0.1]),
            np.zeros((1, 2, 2)),
        )


def test_shot_sensitivity_shapes_and_zero_case() -> None:
    image = np.asarray([[0.2, 0.8], [0.2, 0.4], [0.2, 0.6]])
    pixel = np.broadcast_to(image[:, :, None, None], (3, 2, 2, 3)).copy()
    result = shot_sensitivity(image, pixel)
    assert result["image_shot_std"].shape == (2,)
    assert result["pixel_shot_std"].shape == (2, 2, 3)
    assert result["image_shot_std"][0] == pytest.approx(0.0)
    assert result["image_shot_std"][1] > 0.0


def test_pair_features_include_agreement_and_concentration() -> None:
    result = ConfidenceRouter().fuse(branch([0.2, 0.8]), branch([0.6, 0.4]))
    assert np.allclose(
        result.features["image_agreement"]
        + result.features["image_disagreement"],
        1.0,
    )
    assert result.features["visual_response_concentration"].shape == (2,)
    assert result.features["text_response_concentration"].shape == (2,)
