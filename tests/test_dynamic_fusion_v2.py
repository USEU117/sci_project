from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_ad.fusion import (
    BranchPrediction,
    BranchV2Calibration,
    RankPreservingCalibrator,
    SafeRouterV2Config,
    SafeVisualDefaultRouterV2,
    calibration_diagnostics,
    load_v2_category_calibrations,
    run_v2_minimum_ablation,
)


def prediction(scores: np.ndarray, maps: np.ndarray | None = None) -> BranchPrediction:
    scores = np.asarray(scores, dtype=np.float32)
    if maps is None:
        maps = np.broadcast_to(scores[:, None, None], (len(scores), 4, 5)).copy()
    return BranchPrediction(
        sample_ids=np.asarray([f"sample-{index}" for index in range(len(scores))]),
        image_scores=scores,
        pixel_maps=np.asarray(maps, dtype=np.float32),
    )


def calibration(offset: float = 0.0) -> BranchV2Calibration:
    image = np.linspace(0.0 + offset, 1.0 + offset, 20)
    maps = np.broadcast_to(image[:, None, None], (20, 4, 5)).copy()
    return BranchV2Calibration.fit(image, maps)


def test_rank_preserving_calibration_keeps_strict_order() -> None:
    fitted = RankPreservingCalibrator.fit(np.linspace(-2.0, 2.0, 31))
    raw = np.asarray([-1e6, -10.0, -1.0, 0.0, 1.0, 10.0, 1e6])
    transformed = fitted.transform(raw)
    assert np.all(np.diff(transformed) > 0)
    assert np.array_equal(np.argsort(raw), np.argsort(transformed))
    assert np.all((transformed > 0) & (transformed < 1))


def test_calibration_diagnostics_report_no_saturation_or_rank_loss() -> None:
    raw = np.linspace(-100.0, 100.0, 101)
    fitted = RankPreservingCalibrator.fit(np.linspace(-2.0, 2.0, 31))
    diagnostics = calibration_diagnostics(raw, fitted.transform(raw))
    assert diagnostics["spearman_raw_vs_calibrated"] == pytest.approx(1.0)
    assert diagnostics["unique_value_ratio"] == pytest.approx(1.0)
    assert diagnostics["largest_tie_group"] == 1
    assert diagnostics["lower_boundary_rate"] < 0.01
    assert diagnostics["upper_boundary_rate"] < 0.01


def test_rank_preserving_calibration_constant_reference_is_safe_warning() -> None:
    fitted = RankPreservingCalibrator.fit(np.ones(8))
    assert fitted.degenerate_reference is True
    assert fitted.scale > 0
    assert np.isfinite(fitted.transform(np.asarray([-1.0, 1.0, 2.0]))).all()


def test_upper_support_is_directional() -> None:
    fitted = RankPreservingCalibrator.fit(np.linspace(0.0, 1.0, 20))
    out, confidence, excess = fitted.upper_support_evidence(
        np.asarray([-100.0, 0.5, 100.0]), tolerance=3.0
    )
    assert out.tolist() == [False, False, True]
    assert confidence[0] == pytest.approx(1.0)
    assert excess[0] == pytest.approx(0.0)


def test_v2_router_shapes_and_finite_outputs() -> None:
    router = SafeVisualDefaultRouterV2(calibration(), calibration())
    result = router.fuse(prediction(np.asarray([0.2, 0.8])), prediction(np.asarray([0.3, 0.7])))
    assert result.image_scores.shape == (2,)
    assert result.pixel_maps.shape == (2, 4, 5)
    assert result.visual_weights.shape == (2,)
    assert result.visual_pixel_weights.shape == (2, 4, 5)
    assert np.isfinite(result.image_scores).all()
    assert np.isfinite(result.pixel_maps).all()


def test_v2_defaults_to_visual_when_evidence_is_not_complementary() -> None:
    router = SafeVisualDefaultRouterV2(calibration(), calibration())
    visual = prediction(np.asarray([0.2, 0.8]))
    result = router.fuse(visual, prediction(np.asarray([0.2, 0.8])))
    expected = calibration().apply(visual)
    assert np.all(result.visual_weights == 1.0)
    assert np.all(result.visual_pixel_weights == 1.0)
    assert np.allclose(result.image_scores, expected.image_scores)


def test_v2_text_assistance_is_capped() -> None:
    config = SafeRouterV2Config(
        minimum_disagreement=0.01,
        uncertainty_margin=0.01,
        max_image_text_weight=0.15,
        max_pixel_text_weight=0.35,
    )
    router = SafeVisualDefaultRouterV2(calibration(), calibration(), config)
    visual_maps = np.full((1, 4, 5), 0.50)
    text_maps = np.full((1, 4, 5), 0.05)
    result = router.fuse(
        prediction(np.asarray([0.50]), visual_maps),
        prediction(np.asarray([0.05]), text_maps),
    )
    assert np.any(result.visual_weights < 1.0)
    assert np.all(result.visual_weights >= 0.85)
    assert np.all(result.visual_pixel_weights >= 0.65)


def test_v2_out_of_support_forces_visual_fallback() -> None:
    router = SafeVisualDefaultRouterV2(calibration(), calibration())
    visual = prediction(np.asarray([100.0]))
    result = router.fuse(visual, prediction(np.asarray([0.1])))
    assert result.visual_weights[0] == 1.0
    assert result.decisions[0] == "out_of_support"
    assert result.features["visual_image_out_of_support"][0]


def test_v2_degenerate_calibration_forces_visual_fallback() -> None:
    constant = BranchV2Calibration.fit(np.ones(4), np.ones((4, 3, 3)))
    result = SafeVisualDefaultRouterV2(constant, calibration()).fuse(
        prediction(np.asarray([1.0])), prediction(np.asarray([0.1]))
    )
    assert result.visual_weights[0] == 1.0
    assert result.decisions[0] == "calibration_warning"


def test_v2_rejects_sample_order_mismatch() -> None:
    text = prediction(np.asarray([0.2, 0.8]))
    text = BranchPrediction(
        sample_ids=text.sample_ids[::-1],
        image_scores=text.image_scores,
        pixel_maps=text.pixel_maps,
    )
    with pytest.raises(ValueError, match="sample_ids"):
        SafeVisualDefaultRouterV2(calibration(), calibration()).fuse(
            prediction(np.asarray([0.2, 0.8])), text
        )


def test_v2_rejects_map_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="pixel-map shapes"):
        SafeVisualDefaultRouterV2(calibration(), calibration()).fuse(
            prediction(np.asarray([0.2]), np.zeros((1, 4, 5))),
            prediction(np.asarray([0.2]), np.zeros((1, 3, 5))),
        )


def test_v2_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        SafeVisualDefaultRouterV2(calibration(), calibration()).fuse(
            prediction(np.asarray([np.nan])), prediction(np.asarray([0.2]))
        )


def test_v2_calibration_loader_enforces_all_no_test_flags() -> None:
    fitted = calibration()
    payload = {
        "schema_version": 2,
        "status": "passed",
        "test_predictions_used": False,
        "test_labels_used": False,
        "test_masks_used": False,
        "test_set_statistics_used": False,
        "categories": {"part": {"visual": fitted.to_dict(), "text": fitted.to_dict()}},
    }
    visual, text = load_v2_category_calibrations(payload, "part")
    assert visual == fitted
    assert text == fitted
    payload["test_set_statistics_used"] = True
    with pytest.raises(ValueError, match="test_set_statistics_used=false"):
        load_v2_category_calibrations(payload, "part")


def test_v2_image_and_pixel_routes_are_independent() -> None:
    config = SafeRouterV2Config(
        minimum_disagreement=0.01,
        uncertainty_margin=0.01,
        smooth_pixel_weights=False,
    )
    visual_maps = np.full((1, 4, 5), 0.50)
    text_maps = np.full((1, 4, 5), 0.05)
    result = SafeVisualDefaultRouterV2(calibration(), calibration(), config).fuse(
        prediction(np.asarray([0.2]), visual_maps),
        prediction(np.asarray([0.2]), text_maps),
    )
    assert result.visual_weights[0] == 1.0
    assert np.any(result.visual_pixel_weights < 1.0)


def test_v2_smoothed_pixel_route_is_not_suppressed_by_image_oos() -> None:
    config = SafeRouterV2Config(
        minimum_disagreement=0.01,
        uncertainty_margin=0.01,
        smooth_pixel_weights=True,
    )
    visual_maps = np.full((1, 4, 5), 0.50)
    text_maps = np.full((1, 4, 5), 0.05)
    result = SafeVisualDefaultRouterV2(calibration(), calibration(), config).fuse(
        prediction(np.asarray([100.0]), visual_maps),
        prediction(np.asarray([0.10]), text_maps),
    )
    assert result.features["visual_image_out_of_support"][0]
    assert result.visual_weights[0] == 1.0
    assert np.any(result.visual_pixel_weights < 1.0)


def test_v2_minimum_ablation_has_separate_image_and_pixel_variants() -> None:
    visual = prediction(np.asarray([0.5]), np.full((1, 4, 5), 0.5))
    text = prediction(np.asarray([0.05]), np.full((1, 4, 5), 0.05))
    results = run_v2_minimum_ablation(
        visual,
        text,
        calibration(),
        calibration(),
        SafeRouterV2Config(minimum_disagreement=0.01, uncertainty_margin=0.01),
    )
    assert set(results) == {
        "raw_visual",
        "raw_text",
        "rank_calibrated_visual",
        "rank_calibrated_fixed_050",
        "v2_image_only",
        "v2_pixel_only",
        "v2_complete_split",
    }
    assert np.all(results["v2_image_only"].visual_pixel_weights == 1.0)
    assert np.all(results["v2_pixel_only"].visual_weights == 1.0)
