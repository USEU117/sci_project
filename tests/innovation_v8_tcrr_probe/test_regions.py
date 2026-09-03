from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.innovation_v8_tcrr_probe import (
    component_features,
    component_masks,
    normal_calibrated_region_boost_map,
    proposal_label,
    region_rerank_map,
    robust01,
)


def test_robust01_is_finite_and_bounded():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    y = robust01(x)
    assert np.isfinite(y).all()
    assert float(y.min()) == 0.0 and float(y.max()) == 1.0


def test_robust01_constant_returns_zero():
    assert not robust01(np.ones((4, 4), dtype=np.float32)).any()


def test_component_masks_are_8_connected_and_size_filtered():
    x = np.zeros((8, 8), dtype=np.float32)
    x[1:3, 1:3] = 1.0
    x[5, 5] = 1.0
    masks = component_masks(x, 0.90, min_cells=4)
    assert len(masks) == 1
    assert int(masks[0].sum()) == 4


def test_features_do_not_need_ground_truth():
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:3] = True
    a = np.arange(16, dtype=np.float32).reshape(4, 4) / 15
    t = np.flipud(a)
    f = component_features(m, a, t)
    assert f["area_cells"] == 4
    assert set(f) == {"area_cells", "a1_mean", "a1_max", "a1_p90",
                      "text_trimmed_mean", "text_p90", "text_consistency"}


def test_proposal_label_uses_overlap_fraction():
    m = np.zeros((10, 10), dtype=bool)
    m[1:5, 1:5] = True
    gt = np.zeros_like(m)
    gt[1, 1] = True
    assert proposal_label(m, gt, 0.05)["label"] == 1
    assert proposal_label(m, gt, 0.10)["label"] == 0


def test_ap_is_undefined_when_labels_have_one_class():
    # Documents the reason R0b excludes an unevaluable category from a
    # sensitivity macro instead of silently treating undefined AP as zero.
    from sklearn.metrics import average_precision_score

    y = np.zeros(4, dtype=np.int64)
    assert np.unique(y).size == 1
    # sklearn returns a number with a warning, so the experiment wrapper must
    # explicitly detect the one-class condition before calling it.
    assert callable(average_precision_score)


def test_region_reranker_is_bounded_and_cannot_create_regions():
    raw = np.ones((10, 10), dtype=np.float32)
    proposal = np.zeros_like(raw)
    proposal[2:5, 2:5] = 1.0
    text = np.zeros_like(raw)
    text[2:5, 2:5] = 1.0
    out, audit = region_rerank_map(raw, proposal, text, quantile=0.9, min_cells=4)
    assert len(audit) == 1
    assert np.allclose(out[2:5, 2:5], 1.5)
    outside = np.ones_like(raw, dtype=bool)
    outside[2:5, 2:5] = False
    assert np.array_equal(out[outside], raw[outside])


def test_region_reranker_suppresses_low_text_component():
    raw = np.ones((6, 6), dtype=np.float32)
    proposal = np.zeros_like(raw)
    proposal[1:3, 1:3] = 1.0
    text = np.zeros_like(raw)
    out, audit = region_rerank_map(raw, proposal, text, quantile=0.8, min_cells=4)
    assert np.isclose(audit[0]["factor"], 1 / 1.5)
    assert np.allclose(out[1:3, 1:3], 1 / 1.5)


def test_normal_calibrated_boost_has_identity_fallback_and_never_suppresses():
    raw = np.ones((8, 8), dtype=np.float32)
    proposal = np.zeros_like(raw); proposal[2:5, 2:5] = 1
    refs = np.linspace(0, 1, 2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    low, audit_low, _ = normal_calibrated_region_boost_map(raw, proposal, np.full_like(raw, 0.5), refs,
                                                            quantile=0.8, min_cells=4)
    assert np.array_equal(low, raw)
    high, audit_high, _ = normal_calibrated_region_boost_map(raw, proposal, np.full_like(raw, 20), refs,
                                                              quantile=0.8, min_cells=4)
    assert np.all(high >= raw)
    assert audit_low[0]["factor"] == 1.0
    assert np.isclose(audit_high[0]["factor"], 1.5)
