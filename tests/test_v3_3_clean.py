"""CPU tests for V3.3-clean leakage-safe fusion (docs 阶段二)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from industrial_ad.fusion.v3_3_clean import (
    DEFAULT_ANCHOR,
    EvaluationTarget,
    RouterInput,
    branch_is_reliable,
    compute_z_score,
    estimate_reference_stats,
    evaluate_clean,
    sanitize_finite,
    validate_router_input,
    weighted_ensemble_clean,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_router(
    n: int = 8,
    h: int = 16,
    w: int = 16,
    k: int = 4,
    nan_in_text: bool = False,
) -> RouterInput:
    rng = np.random.RandomState(0)
    ids = np.asarray([f"s-{i}" for i in range(n)])
    visual_maps = rng.uniform(0.0, 1.0, (n, h, w)).astype(np.float32)
    text_maps = rng.uniform(0.0, 1.0, (n, h, w)).astype(np.float32)
    if nan_in_text:
        text_maps[2, 3, 3] = np.nan
        text_maps[5, 0, 0] = np.inf
    ref_visual = rng.uniform(0.0, 0.2, (k, h, w)).astype(np.float32)
    ref_text = rng.uniform(0.0, 0.2, (k, h, w)).astype(np.float32)
    return RouterInput(
        branches={DEFAULT_ANCHOR: visual_maps, "anomalyclip_text": text_maps},
        reference_maps={DEFAULT_ANCHOR: ref_visual, "anomalyclip_text": ref_text},
        sample_ids=ids,
        category="bracket_black",
        seed=0,
        shot=1,
        metadata={"cache_hash": "abc"},
    )


def make_target(ri: RouterInput, labels: np.ndarray | None = None) -> EvaluationTarget:
    n = ri.sample_ids.size
    if labels is None:
        labels = np.zeros(n, dtype=np.uint8)
        labels[::2] = 1
    masks = np.zeros((n, 16, 16), dtype=np.uint8)
    masks[1::2, 4:10, 5:11] = 1
    return EvaluationTarget(
        gt_labels=np.asarray(labels, dtype=np.uint8),
        gt_masks=masks,
        sample_ids=ri.sample_ids.copy(),
    )


# ---------------------------------------------------------------------------
# 1. RouterInput must not carry test ground truth
# ---------------------------------------------------------------------------

def test_router_input_has_no_test_truth_fields() -> None:
    ri = make_router()
    for banned in ("gt_labels", "gt_masks", "gt_sp", "imgs_masks"):
        assert not hasattr(ri, banned), f"RouterInput must not expose {banned}"
    assert list(RouterInput.__dataclass_fields__.keys()) == [
        "branches",
        "reference_maps",
        "sample_ids",
        "category",
        "seed",
        "shot",
        "metadata",
    ]


# ---------------------------------------------------------------------------
# 2. Changing GT must not change predictions
# ---------------------------------------------------------------------------

def test_changing_ground_truth_does_not_change_fused_maps() -> None:
    ri = make_router()
    t1 = make_target(ri)
    masks2 = t1.gt_masks.copy()
    # (0,0) is sampled by stride=8, so this flips even-sample cell (0,0) positive
    masks2[::2, 0:2, 0:2] = 1  # different pixel ground truth
    t2 = EvaluationTarget(
        gt_labels=t1.gt_labels.copy(),
        gt_masks=masks2,
        sample_ids=t1.sample_ids.copy(),
    )
    fused1, _ = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 0.6, "anomalyclip_text": 0.4})
    fused2, _ = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 0.6, "anomalyclip_text": 0.4})
    assert np.array_equal(fused1, fused2)
    # evaluation differs only because the target changed
    m1 = evaluate_clean(ri, t1, fused1)
    m2 = evaluate_clean(ri, t2, fused2)
    assert m1["pixel_auroc"] != m2["pixel_auroc"]


# ---------------------------------------------------------------------------
# 3. sample ID misalignment / missing / duplicates must fail
# ---------------------------------------------------------------------------

def test_duplicate_sample_ids_fail() -> None:
    ri = make_router()
    ids = ri.sample_ids.copy()
    ids[1] = ids[0]
    ri = RouterInput(**{**ri.__dict__, "sample_ids": ids})
    with pytest.raises(ValueError, match="duplicates"):
        validate_router_input(ri)


def test_sample_count_mismatch_fails() -> None:
    ri = make_router()
    branches = dict(ri.branches)
    branches[DEFAULT_ANCHOR] = branches[DEFAULT_ANCHOR][:-1]  # one fewer sample
    ri = RouterInput(**{**ri.__dict__, "branches": branches})
    with pytest.raises(ValueError, match="expected \\[N="):
        validate_router_input(ri)


def test_missing_reference_branch_fails() -> None:
    ri = make_router()
    refs = dict(ri.reference_maps)
    del refs["anomalyclip_text"]
    ri = RouterInput(**{**ri.__dict__, "reference_maps": refs})
    with pytest.raises(ValueError, match="missing reference maps"):
        validate_router_input(ri)


def test_reference_spatial_mismatch_fails() -> None:
    ri = make_router()
    refs = dict(ri.reference_maps)
    refs[DEFAULT_ANCHOR] = refs[DEFAULT_ANCHOR][:, :, :7]  # wrong width
    ri = RouterInput(**{**ri.__dict__, "reference_maps": refs})
    with pytest.raises(ValueError, match="spatial shape"):
        validate_router_input(ri)


# ---------------------------------------------------------------------------
# 4. NaN/Inf safety
# ---------------------------------------------------------------------------

def test_reference_stats_reject_nan_inf() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        estimate_reference_stats(np.asarray([[[0.0, np.nan]]]))
    with pytest.raises(ValueError, match="NaN or infinity"):
        estimate_reference_stats(np.asarray([[[0.0, np.inf]]]))


def test_nan_in_text_branch_is_sanitised_and_falls_back() -> None:
    ri = make_router(nan_in_text=True)
    fused, diag = weighted_ensemble_clean(
        ri, {DEFAULT_ANCHOR: 0.6, "anomalyclip_text": 0.4}
    )
    assert np.isfinite(fused).all()
    assert "anomalyclip_text" in diag["fallback_branches"]


def test_sanitize_finite_replaces_nonfinite_with_floor() -> None:
    arr = np.asarray([[1.0, np.nan, np.inf, -np.inf]])
    out = sanitize_finite(arr)
    assert np.isfinite(out).all()
    assert out[0, 1] == 1.0  # replaced by the finite minimum floor


# ---------------------------------------------------------------------------
# 5. Visual fallback when text unreliable
# ---------------------------------------------------------------------------

def test_fully_degenerate_text_falls_back_to_visual() -> None:
    ri = make_router()
    # constant reference text maps -> degenerate scale
    refs = dict(ri.reference_maps)
    refs["anomalyclip_text"] = np.full((4, 16, 16), 0.5, dtype=np.float32)
    ri = RouterInput(**{**ri.__dict__, "reference_maps": refs})
    fused, diag = weighted_ensemble_clean(
        ri, {DEFAULT_ANCHOR: 0.5, "anomalyclip_text": 0.5}
    )
    assert "anomalyclip_text" in diag["fallback_branches"]
    # anchor only -> fused equals visual z-scores
    assert np.allclose(fused, compute_z_score(ri.branches[DEFAULT_ANCHOR], *[np.median(ri.reference_maps[DEFAULT_ANCHOR].ravel()), max(np.subtract(*np.percentile(ri.reference_maps[DEFAULT_ANCHOR].ravel(), [75, 25])), 1e-8)]))


def test_branch_is_reliable_flags_degenerate() -> None:
    stats = {"scale": 0.0, "mad": 0.0}
    assert branch_is_reliable(stats, np.zeros((2, 4, 4))) is False
    stats2 = {"scale": 0.1, "mad": 0.1}
    assert branch_is_reliable(stats2, np.zeros((2, 4, 4))) is True


def test_all_branches_unreliable_returns_raw_visual() -> None:
    ri = make_router()
    refs = dict(ri.reference_maps)
    for k in refs:
        refs[k] = np.full((4, 16, 16), 0.5, dtype=np.float32)
    ri = RouterInput(**{**ri.__dict__, "reference_maps": refs})
    fused, diag = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 0.5, "anomalyclip_text": 0.5})
    assert diag["fallback"] == "all"
    assert np.array_equal(fused, ri.branches[DEFAULT_ANCHOR].astype(np.float64))


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

def test_fusion_is_deterministic() -> None:
    ri = make_router()
    f1, d1 = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 0.6, "anomalyclip_text": 0.4})
    f2, d2 = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 0.6, "anomalyclip_text": 0.4})
    assert np.array_equal(f1, f2)
    assert d1["reliable"] == d2["reliable"]


# ---------------------------------------------------------------------------
# 7. Reference stats sanity
# ---------------------------------------------------------------------------

def test_reference_stats_median_iqr_mad_quantiles() -> None:
    vals = np.linspace(0.0, 1.0, 101)
    maps = np.broadcast_to(vals[:, None, None], (101, 2, 2))
    stats = estimate_reference_stats(maps)
    assert stats["center"] == pytest.approx(0.5, abs=1e-6)
    assert stats["scale"] == pytest.approx(0.5, abs=1e-6)  # q75-q25
    assert stats["mad"] > 0.0
    assert stats["q95"] == pytest.approx(0.95, abs=1e-6)
    assert stats["q99"] == pytest.approx(0.99, abs=1e-6)


def test_evaluate_clean_rejects_mismatched_target_ids() -> None:
    ri = make_router()
    target = make_target(ri)
    target = EvaluationTarget(
        gt_labels=target.gt_labels,
        gt_masks=target.gt_masks,
        sample_ids=np.asarray([f"other-{i}" for i in range(ri.sample_ids.size)]),
    )
    fused, _ = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 1.0})
    with pytest.raises(ValueError, match="do not match"):
        evaluate_clean(ri, target, fused)
