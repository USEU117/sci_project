from __future__ import annotations

import numpy as np
import pytest

from industrial_ad.fusion.contracts import BranchPrediction
from industrial_ad.fusion.v3_contracts import BranchEvidenceV3, RescueBudgetV3
from industrial_ad.fusion.v3_calibration import GroupedReferenceCalibratorV3
from industrial_ad.fusion.v3_router import HierarchicalSelectiveRescueV3, SelectiveRescueConfigV3


def _prediction() -> BranchPrediction:
    return BranchPrediction(
        sample_ids=np.asarray(["a", "b"]),
        image_scores=np.asarray([0.1, 0.9], dtype=np.float32),
        pixel_maps=np.asarray(
            [
                [[0.1, 0.2], [0.2, 0.1]],
                [[0.1, 0.8], [0.7, 0.1]],
            ],
            dtype=np.float32,
        ),
    )


def test_v3_contract_keeps_anomaly_evidence_separate_from_reliability() -> None:
    prediction = _prediction()
    evidence = BranchEvidenceV3(
        prediction=prediction,
        image_anomaly_evidence=np.asarray([0.2, 8.0]),
        pixel_anomaly_evidence=np.asarray(prediction.pixel_maps) * 10,
        image_reliability=np.asarray([0.8, 0.9]),
        pixel_reliability=np.full((2, 2, 2), 0.75),
    )

    assert evidence.validate() is evidence
    assert evidence.image_anomaly_evidence[1] > 1
    assert evidence.image_reliability[1] == pytest.approx(0.9)


def test_v3_contract_rejects_invalid_reliability() -> None:
    prediction = _prediction()
    evidence = BranchEvidenceV3(
        prediction=prediction,
        image_anomaly_evidence=np.asarray([0.2, 0.8]),
        pixel_anomaly_evidence=np.asarray(prediction.pixel_maps),
        image_reliability=np.asarray([0.8, 1.1]),
        pixel_reliability=np.full((2, 2, 2), 0.75),
    )

    with pytest.raises(ValueError, match="image_reliability"):
        evidence.validate()


def test_v3_contract_rejects_pixel_shape_mismatch() -> None:
    prediction = _prediction()
    evidence = BranchEvidenceV3(
        prediction=prediction,
        image_anomaly_evidence=np.asarray([0.2, 0.8]),
        pixel_anomaly_evidence=np.zeros((2, 3, 3)),
        image_reliability=np.asarray([0.8, 0.9]),
        pixel_reliability=np.full((2, 2, 2), 0.75),
    )

    with pytest.raises(ValueError, match="pixel evidence"):
        evidence.validate()


@pytest.mark.parametrize(
    ("image_budget", "pixel_budget"),
    [(-0.01, 0.1), (0.3, 0.1), (0.1, -0.01), (0.1, 0.6)],
)
def test_v3_budget_rejects_unsafe_values(image_budget: float, pixel_budget: float) -> None:
    with pytest.raises(ValueError):
        RescueBudgetV3(image_budget, pixel_budget).validate()


def test_v3_budget_accepts_conservative_defaults() -> None:
    budget = RescueBudgetV3()
    assert budget.validate() is budget


def test_v3_grouped_calibration_counts_references_not_augmented_views() -> None:
    calibrator = GroupedReferenceCalibratorV3.fit(
        np.asarray([0.10, 0.11, 0.09, 0.105]),
        np.asarray(["reference_0"] * 4),
        source_center=0.0,
        source_scale=0.2,
    )

    assert calibrator.effective_group_count == 1
    assert calibrator.view_count == 4
    assert calibrator.target_scale_degenerate


def test_v3_grouped_calibration_standardization_preserves_order() -> None:
    calibrator = GroupedReferenceCalibratorV3.fit(
        np.asarray([0.10, 0.11, 0.20, 0.19]),
        np.asarray(["reference_0", "reference_0", "reference_1", "reference_1"]),
        source_center=0.0,
        source_scale=0.2,
    )
    query = np.asarray([-2.0, 0.1, 0.4, 100.0])
    transformed = calibrator.standardize(query)

    assert np.all(np.diff(transformed) > 0)
    assert np.all(np.diff(calibrator.signed_evidence(query)) > 0)


def test_v3_high_anomaly_evidence_does_not_reduce_reference_reliability() -> None:
    calibrator = GroupedReferenceCalibratorV3.fit(
        np.asarray([0.10, 0.11, 0.20, 0.19]),
        np.asarray(["reference_0", "reference_0", "reference_1", "reference_1"]),
        source_center=0.0,
        source_scale=0.2,
    )

    reliability = calibrator.reliability((2,))
    evidence = calibrator.anomaly_evidence(np.asarray([0.2, 100.0]))
    assert evidence[1] > evidence[0]
    assert reliability[1] == pytest.approx(reliability[0])


def _branch_evidence(
    image: np.ndarray, pixel: np.ndarray, reliability: float
) -> BranchEvidenceV3:
    prediction = BranchPrediction(
        sample_ids=np.asarray(["a", "b"]),
        image_scores=np.asarray(image, dtype=np.float32),
        pixel_maps=np.asarray(pixel, dtype=np.float32),
    )
    return BranchEvidenceV3(
        prediction=prediction,
        image_anomaly_evidence=np.asarray(image, dtype=np.float64),
        pixel_anomaly_evidence=np.asarray(pixel, dtype=np.float64),
        image_reliability=np.full(2, reliability),
        pixel_reliability=np.full(np.asarray(pixel).shape, reliability),
    )


def test_v3_default_disables_image_rescue_and_preserves_visual_image_scores() -> None:
    visual_pixel = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixel = np.zeros((2, 4, 4), dtype=np.float64)
    visual = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixel, 0.7)
    text = _branch_evidence(np.asarray([0.8, 0.9]), text_pixel, 0.9)

    result = HierarchicalSelectiveRescueV3().fuse(visual, text)
    assert np.array_equal(result.image_scores, np.asarray([0.1, 0.2], dtype=np.float32))
    assert not np.any(result.image_rescue_allowed)


def test_v3_region_rescue_is_non_negative_and_budgeted() -> None:
    visual_pixel = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixel = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixel[0, 1:3, 1:3] = 4.0
    visual = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixel, 0.7)
    text = _branch_evidence(np.asarray([0.1, 0.2]), text_pixel, 0.9)

    result = HierarchicalSelectiveRescueV3().fuse(visual, text)
    assert np.all(result.pixel_maps >= visual_pixel)
    assert np.max(result.pixel_residual) == pytest.approx(0.15)
    assert np.sum(result.pixel_rescue_allowed[0]) == 4


def test_v3_region_filter_rejects_single_pixel_text_noise() -> None:
    visual_pixel = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixel = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixel[0, 1, 1] = 4.0
    visual = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixel, 0.7)
    text = _branch_evidence(np.asarray([0.1, 0.2]), text_pixel, 0.9)

    result = HierarchicalSelectiveRescueV3().fuse(visual, text)
    assert not np.any(result.pixel_rescue_allowed)
    assert np.all(result.pixel_residual == 0)


def test_v3_router_rejects_misaligned_sample_ids() -> None:
    visual_pixel = np.zeros((2, 4, 4), dtype=np.float64)
    visual = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixel, 0.7)
    text = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixel, 0.9)
    text_prediction = BranchPrediction(
        sample_ids=np.asarray(["b", "a"]),
        image_scores=text.prediction.image_scores,
        pixel_maps=text.prediction.pixel_maps,
    )
    text = BranchEvidenceV3(
        prediction=text_prediction,
        image_anomaly_evidence=text.image_anomaly_evidence,
        pixel_anomaly_evidence=text.pixel_anomaly_evidence,
        image_reliability=text.image_reliability,
        pixel_reliability=text.pixel_reliability,
    )

    with pytest.raises(ValueError, match="sample IDs"):
        HierarchicalSelectiveRescueV3().fuse(visual, text)


def test_v3_image_rescue_requires_explicit_enable_and_is_budgeted() -> None:
    pixels = np.zeros((2, 4, 4), dtype=np.float64)
    visual = _branch_evidence(np.asarray([0.1, 2.0]), pixels, 0.7)
    text = _branch_evidence(np.asarray([3.0, 4.0]), pixels, 0.9)
    config = SelectiveRescueConfigV3(enable_image_rescue=True)

    result = HierarchicalSelectiveRescueV3(config).fuse(visual, text)
    assert result.image_residual[0] == pytest.approx(0.05)
    assert result.image_residual[1] == pytest.approx(0.0)


def test_v3_counterfactual_unreliable_text_cannot_change_output() -> None:
    pixels = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixels = np.full((2, 4, 4), 1000.0, dtype=np.float64)
    visual = _branch_evidence(np.asarray([0.1, 0.2]), pixels, 0.9)
    unreliable_text = _branch_evidence(np.asarray([1000.0, 1000.0]), text_pixels, 0.0)

    result = HierarchicalSelectiveRescueV3().fuse(visual, unreliable_text)
    assert np.array_equal(result.image_scores, visual.image_anomaly_evidence.astype(np.float32))
    assert np.array_equal(result.pixel_maps, visual.pixel_anomaly_evidence.astype(np.float32))
    assert not np.any(result.pixel_rescue_allowed)


def test_v3_counterfactual_is_sample_permutation_equivariant() -> None:
    visual_pixels = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixels = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixels[0, 1:3, 1:3] = 2.0
    text_pixels[1, 0:2, 0:2] = 3.0
    visual = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixels, 0.7)
    text = _branch_evidence(np.asarray([0.3, 0.4]), text_pixels, 0.9)
    direct = HierarchicalSelectiveRescueV3().fuse(visual, text)

    order = np.asarray([1, 0])
    def permute(branch: BranchEvidenceV3) -> BranchEvidenceV3:
        return BranchEvidenceV3(
            prediction=BranchPrediction(
                sample_ids=branch.prediction.sample_ids[order],
                image_scores=branch.prediction.image_scores[order],
                pixel_maps=branch.prediction.pixel_maps[order],
            ),
            image_anomaly_evidence=branch.image_anomaly_evidence[order],
            pixel_anomaly_evidence=branch.pixel_anomaly_evidence[order],
            image_reliability=branch.image_reliability[order],
            pixel_reliability=branch.pixel_reliability[order],
        )

    permuted = HierarchicalSelectiveRescueV3().fuse(permute(visual), permute(text))
    assert np.array_equal(permuted.image_scores, direct.image_scores[order])
    assert np.array_equal(permuted.pixel_maps, direct.pixel_maps[order])
    assert np.array_equal(permuted.pixel_rescue_allowed, direct.pixel_rescue_allowed[order])


def test_v3_counterfactual_is_spatial_flip_equivariant() -> None:
    visual_pixels = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixels = np.zeros((2, 4, 4), dtype=np.float64)
    text_pixels[0, 0:2, 1:3] = 2.0
    visual = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixels, 0.7)
    text = _branch_evidence(np.asarray([0.3, 0.4]), text_pixels, 0.9)
    direct = HierarchicalSelectiveRescueV3().fuse(visual, text)

    flipped_visual = _branch_evidence(np.asarray([0.1, 0.2]), visual_pixels[:, :, ::-1], 0.7)
    flipped_text = _branch_evidence(np.asarray([0.3, 0.4]), text_pixels[:, :, ::-1], 0.9)
    flipped = HierarchicalSelectiveRescueV3().fuse(flipped_visual, flipped_text)
    assert np.array_equal(flipped.pixel_maps, direct.pixel_maps[:, :, ::-1])
    assert np.array_equal(flipped.pixel_rescue_allowed, direct.pixel_rescue_allowed[:, :, ::-1])
