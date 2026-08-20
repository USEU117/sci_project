"""CPU tests for V4 contracts (src/industrial_ad/fusion/v4_contracts.py).

Proves the physical RouterInput / EvaluationTarget separation and the
prediction-hash GT independence required by DYNAMIC_FUSION_NEXT_STEPS.md G1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_ad.fusion.v4_contracts import (  # noqa: E402
    EvaluationTarget,
    LEAKAGE_FLAGS,
    RouterInput,
)


def make_router(n: int = 3, h: int = 8, w: int = 8, with_text: bool = False) -> RouterInput:
    rng = np.random.default_rng(0)
    text = rng.random((n, h, w)).astype(np.float32) if with_text else None
    return RouterInput(
        sample_ids=np.array([f"im_{i}" for i in range(n)]),
        visual_map=rng.random((n, h, w)).astype(np.float32),
        grid=(h, w),
        text_map=text,
        model_metadata={
            "normal_prompt": "a photo of a flawless object",
            "abnormal_prompt": "a photo of a defective object",
            "text_order": "normal_then_abnormal",
        },
    )


def make_target(n: int = 3, h: int = 8, w: int = 8) -> EvaluationTarget:
    rng = np.random.default_rng(1)
    return EvaluationTarget(
        sample_ids=np.array([f"im_{i}" for i in range(n)]),
        image_labels=np.array([0, 1, 0], dtype=np.int64),
        pixel_masks=(rng.random((n, h, w)) > 0.5).astype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Leakage flags
# ---------------------------------------------------------------------------

def test_leakage_flags_all_false() -> None:
    assert set(LEAKAGE_FLAGS) == {
        "test_predictions_used_for_parameter_fit",
        "test_labels_used_for_parameter_fit",
        "test_masks_used_for_parameter_fit",
        "test_dataset_statistics_used_for_calibration",
        "test_normal_selection_used",
    }
    assert all(v is False for v in LEAKAGE_FLAGS.values())


# ---------------------------------------------------------------------------
# Structural GT independence
# ---------------------------------------------------------------------------

def test_router_input_has_no_gt_attributes() -> None:
    router = make_router()
    for forbidden in ("image_labels", "pixel_masks", "gt_sp", "imgs_masks", "labels", "masks"):
        assert not hasattr(router, forbidden), f"RouterInput must not expose {forbidden}"


def test_prediction_hash_independent_of_evaluation_target() -> None:
    router = make_router()
    h_before = router.prediction_hash()

    target = make_target()
    # delete
    del target
    assert router.prediction_hash() == h_before

    # replace / shuffle GT values while keeping router fixed
    t1 = make_target()
    t2 = EvaluationTarget(
        sample_ids=np.array([f"im_{i}" for i in range(3)]),
        image_labels=np.array([1, 0, 1], dtype=np.int64),
        pixel_masks=(np.random.default_rng(9).random((3, 8, 8)) > 0.5).astype(np.uint8),
    )
    t1.validate()
    t2.validate()
    assert router.prediction_hash() == h_before


def test_prediction_hash_deterministic() -> None:
    a = make_router()
    b = make_router()
    assert a.prediction_hash() == b.prediction_hash()


# ---------------------------------------------------------------------------
# Failure injection
# ---------------------------------------------------------------------------

def test_duplicate_sample_ids_fail() -> None:
    router = make_router()
    with pytest.raises(ValueError):
        RouterInput(
            sample_ids=np.array(["im_0", "im_0", "im_2"]),
            visual_map=router.visual_map,
            grid=router.grid,
            text_map=router.text_map,
            model_metadata=router.model_metadata,
        ).validate()


def test_sample_id_map_misalignment_fail() -> None:
    router = make_router()
    with pytest.raises(ValueError):
        RouterInput(
            sample_ids=np.array(["im_0", "im_1"]),  # N=2 vs map N=3
            visual_map=router.visual_map,
            grid=router.grid,
        ).validate()


def test_grid_mismatch_fail() -> None:
    router = make_router(h=8, w=8)
    with pytest.raises(ValueError):
        RouterInput(
            sample_ids=router.sample_ids,
            visual_map=router.visual_map,
            grid=(4, 4),
        ).validate()


def test_nan_inf_fail() -> None:
    router = make_router()
    bad = router.visual_map.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        RouterInput(
            sample_ids=router.sample_ids, visual_map=bad, grid=router.grid
        ).validate()


def test_empty_prompt_fail() -> None:
    router = make_router(with_text=True)
    with pytest.raises(ValueError):
        RouterInput(
            sample_ids=router.sample_ids,
            visual_map=router.visual_map,
            grid=router.grid,
            text_map=router.text_map,
            model_metadata={"normal_prompt": "", "abnormal_prompt": "x"},
        ).validate()


def test_wrong_text_order_fail() -> None:
    router = make_router(with_text=True)
    with pytest.raises(ValueError):
        RouterInput(
            sample_ids=router.sample_ids,
            visual_map=router.visual_map,
            grid=router.grid,
            text_map=router.text_map,
            model_metadata={"normal_prompt": "a", "abnormal_prompt": "b", "text_order": "sideways"},
        ).validate()


def test_text_missing_is_valid_visual_only_fallback() -> None:
    # text_map=None is the precise visual-only fallback state; must validate.
    router = make_router(with_text=False)
    router.validate()
    assert router.text_map is None


def test_text_map_spatial_mismatch_fail() -> None:
    router = make_router(h=8, w=8)
    text = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        RouterInput(
            sample_ids=router.sample_ids,
            visual_map=router.visual_map,
            grid=router.grid,
            text_map=text,
            model_metadata=router.model_metadata,
        ).validate()


def test_evaluation_target_validation() -> None:
    make_target().validate()
    with pytest.raises(ValueError):
        EvaluationTarget(
            sample_ids=np.array(["a", "a"]),
            image_labels=np.array([0, 1]),
            pixel_masks=np.zeros((2, 8, 8), dtype=np.uint8),
        ).validate()
