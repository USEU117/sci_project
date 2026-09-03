"""Route D NORC: gating correctness + no-leakage tests (task book 19 §7)."""

from __future__ import annotations

import numpy as np
import pytest

from industrial_ad.innovation_v10_portfolio import norc


def test_conformal_p_rank():
    calib = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert norc.conformal_p(0.5, calib) == pytest.approx(1.0 / 5.0)  # 0 calib >= 0.5
    assert norc.conformal_p(0.05, calib) == pytest.approx(5.0 / 5.0)  # all calib >= 0.05


def test_region_max_scores_basic():
    grid = np.zeros((16, 16), dtype=np.float32)
    grid[2:5, 2:5] = 0.9   # region 1 above theta
    grid[10:12, 10:12] = 0.7  # region 2 above theta
    grid[0:1, 0:1] = 0.95   # isolated above theta
    lbl, scores, counts = norc.region_max_scores(grid, theta=0.5)
    assert len(scores) == 3
    assert set(counts.tolist()) == {9, 4, 1}
    assert scores.max() == pytest.approx(0.95)


def test_gate_delta_identity_when_no_region_significant():
    a1 = np.zeros((16, 16), dtype=np.float32)
    a1[2:4, 2:4] = 0.6  # above theta but below all calibration scores -> p = 1
    calib = np.array([0.8, 0.9, 0.95, 1.0], dtype=np.float32)
    delta = np.ones_like(a1) * 0.25
    gated, stats = norc.gate_delta(delta, a1, theta=0.5, calib_region_max=calib)
    assert float(gated.max()) == 0.0          # no modification anywhere
    assert stats["n_activated"] == 0
    # identity: a1 + gated == a1 exactly
    assert np.array_equal(a1 + gated, a1)


def test_gate_delta_applies_only_to_significant_regions():
    a1 = np.zeros((16, 16), dtype=np.float32)
    a1[2:5, 2:5] = 0.99   # extreme -> p small -> activated
    a1[8:10, 8:10] = 0.6  # moderate -> not activated
    # n=20 calibration units: p_min = 1/21 ~ 0.048 <= 0.05 (reachable)
    calib = np.linspace(0.4, 0.7, 20, dtype=np.float32)
    delta = np.ones_like(a1) * 0.5
    gated, stats = norc.gate_delta(delta, a1, theta=0.5, calib_region_max=calib)
    assert stats["n_regions"] == 2
    assert stats["n_activated"] == 1
    assert float(gated[3, 3]) == pytest.approx(0.5)
    assert float(gated[9, 9]) == 0.0


def test_gate_never_fires_with_tiny_calibration():
    """MPDD k4 -> 4 units -> p_min = 1/5 = 0.2 > 0.05: gate is fully conservative."""
    a1 = np.zeros((16, 16), dtype=np.float32)
    a1[2:5, 2:5] = 0.99
    calib = np.array([0.5, 0.55, 0.6, 0.65], dtype=np.float32)
    gated, stats = norc.gate_delta(np.ones_like(a1) * 0.5, a1, theta=0.5,
                                   calib_region_max=calib)
    assert stats["n_activated"] == 0
    assert float(gated.max()) == 0.0


def test_loo_theta_requires_k2():
    ref = np.random.default_rng(0).normal(size=(1, 4, 4, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        norc.loo_theta(ref)


def test_gate_functions_have_no_gt_params():
    import inspect

    for name in ("gate_delta", "significant_region_mask", "region_max_scores",
                 "conformal_p", "loo_theta"):
        sig = inspect.signature(getattr(norc, name))
        assert not any("gt" in p for p in sig.parameters), name
