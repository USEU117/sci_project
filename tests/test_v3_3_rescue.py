"""CPU tests for V3.3 visual-anchored text local rescue (docs 阶段四)."""

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
    evaluate_clean,
)
from industrial_ad.fusion.v3_3_rescue import (
    BOUNDED_TEXT_RESIDUAL,
    NO_VISUAL_CANDIDATE,
    PROMPT_UNSTABLE,
    REFERENCE_IN_SUPPORT,
    VISUAL_FALLBACK,
    LocalRescueConfig,
    background_reject_mask,
    local_rescue_fusion,
    prompt_stability,
    visual_candidate_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ri(
    n: int = 6,
    h: int = 32,
    w: int = 32,
    k: int = 5,
    text_unstable: bool = False,
    blob_value: float = 5.0,
) -> RouterInput:
    rng = np.random.RandomState(7)
    ids = np.asarray([f"s-{i}" for i in range(n)])
    # low, mostly-flat background + one strong anomaly blob
    visual_maps = rng.uniform(0.0, 0.05, (n, h, w)).astype(np.float32)
    text_maps = rng.uniform(0.0, 0.05, (n, h, w)).astype(np.float32)
    visual_maps[:, 8:24, 8:24] = blob_value
    text_maps[:, 8:24, 8:24] = blob_value
    ref_visual = rng.uniform(0.0, 0.05, (k, h, w)).astype(np.float32)
    if text_unstable:
        # one view wildly different -> large view spread -> unstable
        ref_text = np.zeros((k, h, w), dtype=np.float32)
        ref_text[0] = 1.0
        ref_text[1:] = rng.uniform(0.0, 0.01, (k - 1, h, w)).astype(np.float32)
    else:
        ref_text = rng.uniform(0.0, 0.05, (k, h, w)).astype(np.float32)
    return RouterInput(
        branches={DEFAULT_ANCHOR: visual_maps, "anomalyclip_text": text_maps},
        reference_maps={DEFAULT_ANCHOR: ref_visual, "anomalyclip_text": ref_text},
        sample_ids=ids,
        category="connector",
        seed=0,
        shot=1,
        metadata={"cache_hash": "test"},
    )


def make_target(ri: RouterInput) -> EvaluationTarget:
    n = ri.sample_ids.size
    labels = np.zeros(n, dtype=np.uint8)
    labels[::2] = 1
    masks = np.zeros((n, 32, 32), dtype=np.uint8)
    masks[::2, 10:22, 10:22] = 1
    return EvaluationTarget(gt_labels=labels, gt_masks=masks, sample_ids=ri.sample_ids.copy())


# ---------------------------------------------------------------------------
# 1. RouterInput data boundary reused from clean protocol
# ---------------------------------------------------------------------------

def test_rescue_reuses_router_input_boundary() -> None:
    ri = make_ri()
    for banned in ("gt_labels", "gt_masks"):
        assert not hasattr(ri, banned)


# ---------------------------------------------------------------------------
# 2. Prompt/aug stability
# ---------------------------------------------------------------------------

def test_prompt_stability_stable_views() -> None:
    rng = np.random.RandomState(0)
    refs = rng.uniform(0.1, 0.12, (5, 16, 16)).astype(np.float32)  # near-identical views
    assert prompt_stability(refs) is True


def test_prompt_stability_unstable_views() -> None:
    refs = np.zeros((5, 16, 16), dtype=np.float32)
    refs[0] = 1.0
    refs[1:] = 0.001
    assert prompt_stability(refs) is False


def test_prompt_stability_too_few_views() -> None:
    assert prompt_stability(np.zeros((1, 8, 8), dtype=np.float32), min_views=2) is False


# ---------------------------------------------------------------------------
# 3. Candidate mask & background rejection
# ---------------------------------------------------------------------------

def test_visual_candidate_mask_rejects_border() -> None:
    z = np.zeros((64, 64), dtype=np.float64)
    z[30, 30] = 10.0
    support = np.full((64, 64), 0.0)
    cand = visual_candidate_mask(z, support, margin=4)
    assert bool(cand[30, 30])
    assert not cand[:4].any() and not cand[-4:].any()
    assert not cand[:, :4].any() and not cand[:, -4:].any()


def test_background_reject_flat_pixels() -> None:
    rng = np.random.RandomState(1)
    maps = np.zeros((2, 32, 32), dtype=np.float64)
    maps[0, 10:20, 10:20] = 1.0  # textured region
    cand = np.ones((2, 32, 32), dtype=bool)
    keep = background_reject_mask(maps, cand, min_visual_std=1e-3)
    # flat image-2 region rejected entirely
    assert not keep[1].any()
    # image-1 flat area rejected, high-contrast area kept
    assert keep[0, 10:20, 10:20].any()


# ---------------------------------------------------------------------------
# 4. Local rescue: reason codes & bounded residual
# ---------------------------------------------------------------------------

def test_local_rescue_reason_codes() -> None:
    ri = make_ri()
    fused, diag = local_rescue_fusion(ri)
    counts = diag["reason_counts"]
    # every pixel must have a reason
    assert sum(counts.values()) == ri.sample_ids.size * 32 * 32
    # at least some text rescue happens in this setup
    assert counts["bounded_text_residual"] > 0
    # reason codes cover expected set
    assert set(counts) == {
        "no_visual_candidate",
        "reference_in_support",
        "prompt_unstable",
        "background_rejected",
        "bounded_text_residual",
        "visual_fallback",
    }
    assert np.isfinite(fused).all()


def test_local_rescue_residual_is_bounded() -> None:
    ri = make_ri()
    # force a huge text anomaly -> residual must be capped
    ri2 = RouterInput(
        **{
            **ri.__dict__,
            "branches": {
                DEFAULT_ANCHOR: ri.branches[DEFAULT_ANCHOR],
                "anomalyclip_text": np.full_like(ri.branches["anomalyclip_text"], 50.0),
            },
        }
    )
    fused, diag = local_rescue_fusion(ri2, config=LocalRescueConfig(residual_cap=2.0))
    # residual in z-units never exceeds the cap
    assert diag["max_text_residual"] <= 2.0 + 1e-6
    assert diag["accepted_pixels"] > 0
    # capped residual => fused never exceeds visual-z by more than the cap
    assert np.isfinite(fused).all()


def test_local_rescue_unstable_text_falls_back_to_visual() -> None:
    ri = make_ri(text_unstable=True)
    fused, diag = local_rescue_fusion(ri)
    assert diag["text_stable"] is False
    counts = diag["reason_counts"]
    # no text residual accepted; candidate pixels -> prompt_unstable
    assert counts["bounded_text_residual"] == 0
    assert counts["prompt_unstable"] > 0


def test_local_rescue_visual_fallback_on_nan() -> None:
    ri = make_ri()
    ri2 = RouterInput(
        **{
            **ri.__dict__,
            "branches": {
                DEFAULT_ANCHOR: ri.branches[DEFAULT_ANCHOR],
                "anomalyclip_text": ri.branches["anomalyclip_text"].copy(),
            },
        }
    )
    ri2.branches["anomalyclip_text"][0, 0, 0] = np.nan
    fused, diag = local_rescue_fusion(ri2, config=LocalRescueConfig(fill=-1.0))
    assert np.isfinite(fused).all()


# ---------------------------------------------------------------------------
# 5. Determinism & leakage safety
# ---------------------------------------------------------------------------

def test_local_rescue_deterministic() -> None:
    ri = make_ri()
    f1, d1 = local_rescue_fusion(ri)
    f2, d2 = local_rescue_fusion(ri)
    assert np.array_equal(f1, f2)
    assert d1["reason_counts"] == d2["reason_counts"]


def test_changing_ground_truth_does_not_change_rescue_maps() -> None:
    ri = make_ri()
    t1 = make_target(ri)
    masks2 = t1.gt_masks.copy()
    masks2[::2, 0:2, 0:2] = 1  # (0,0) sampled by stride=8
    t2 = EvaluationTarget(gt_labels=t1.gt_labels.copy(), gt_masks=masks2, sample_ids=t1.sample_ids.copy())
    f1, _ = local_rescue_fusion(ri)
    f2, _ = local_rescue_fusion(ri)
    assert np.array_equal(f1, f2)
    m1 = evaluate_clean(ri, t1, f1)
    m2 = evaluate_clean(ri, t2, f2)
    assert m1["pixel_auroc"] != m2["pixel_auroc"]


def test_evaluation_target_id_mismatch_rejected() -> None:
    ri = make_ri()
    target = make_target(ri)
    target = EvaluationTarget(
        gt_labels=target.gt_labels,
        gt_masks=target.gt_masks,
        sample_ids=np.asarray([f"other-{i}" for i in range(ri.sample_ids.size)]),
    )
    fused, _ = local_rescue_fusion(ri)
    with pytest.raises(ValueError, match="do not match"):
        evaluate_clean(ri, target, fused)
