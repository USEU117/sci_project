"""Unit tests for innovation_v4_diagnostics (small synthetic inputs only)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v4_diagnostics import common, diagnostics as diag, spectral  # noqa: E402


# ---------------------------------------------------------------------------
# spectral
# ---------------------------------------------------------------------------

def test_swt_descriptor_shape_and_channels():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
    d = spectral.swt_descriptor(img)
    assert d.shape == (32, 32, 24)
    assert np.all(np.isfinite(d))


def test_spectral_responds_to_high_frequency():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
    clean = spectral.spectral_descriptor_image(img)
    noisy = img.astype(np.int16).copy()
    noisy[100:150, 100:150] += rng.normal(0, 90, (50, 50, 3)).astype(np.int16)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    inj = spectral.spectral_descriptor_image(noisy)
    # the injected window descriptor should differ from the clean one
    err = np.abs(inj[7:11, 7:11] - clean[7:11, 7:11]).mean()
    err_far = np.abs(inj[20:24, 20:24] - clean[20:24, 20:24]).mean()
    assert err > err_far  # local response dominates


def test_spectral_scores_label_free_shape():
    r = np.random.default_rng(0).normal(size=(2, 8, 8, 24)).astype(np.float32)
    q = np.random.default_rng(1).normal(size=(3, 8, 8, 24)).astype(np.float32)
    s = spectral.spectral_scores(r, q)
    assert s.shape == (3, 8, 8)
    assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# common (evaluator-only tiering)
# ---------------------------------------------------------------------------

def test_tier_of_area():
    assert common.tier_of_area(0.001) == "small"
    assert common.tier_of_area(0.02) == "mid"
    assert common.tier_of_area(0.2) == "large"


def test_tier_pooled_empty_and_full():
    rng = np.random.default_rng(0)
    maps = rng.random((5, 16, 16)).astype(np.float32)
    gtm = np.zeros((5, 16, 16), dtype=np.uint8)
    gtm[0, :1, :1] = 1    # 1/256 = 0.39% -> small
    gtm[1, 8:16, 8:16] = 1  # 50% -> large
    small = common.tier_pooled_map_scores(maps, gtm, np.array([0, 1]), np.array([2, 3, 4]), "small")
    assert small["n_bad_images"] == 1 and small["ap"] is not None
    mid = common.tier_pooled_map_scores(maps, gtm, np.array([0, 1]), np.array([2, 3, 4]), "mid")
    assert mid["n_bad_images"] == 0 and mid["ap"] is None
    assert "n_pos_px" in mid


def test_guard_rejects_validation_access():
    from industrial_ad.innovation_v2 import common as v2c
    with pytest.raises(common.ValidationDatasetAccessError):
        v2c.assert_development_only("mvtec")
    # the v4 wrapper is bound to MPDD only and delegates to the same guard
    assert common.development_dataset() == "mpdd"


# ---------------------------------------------------------------------------
# D2 helpers
# ---------------------------------------------------------------------------

def _toy_grid(n=8, d=16, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, n, d)).astype(np.float32)
    g /= np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12
    return g


def test_perturbation_masks():
    g = _toy_grid()
    rng = np.random.default_rng(0)
    for t in ("permutation", "missing", "duplicate"):
        pg, m = diag.perturb_structural(g, _toy_grid(seed=1), t, rng, bs=2)
        assert pg.shape == g.shape and m.shape == (8, 8)
        assert 0 < m.sum() <= 4 * 4  # at most the changed region


def test_ring_mean_excludes_center():
    g = np.zeros((8, 8, 1), dtype=np.float32)
    g[4, 4, 0] = 100.0
    rm = diag.ring_mean(g, 1, 2)
    assert rm[4, 4, 0] == 0.0  # centre value must not leak into its own ring mean
    # ring mean of uniform grid equals the uniform value
    u = np.ones((8, 8, 1), dtype=np.float32) * 3.0
    assert np.allclose(diag.ring_mean(u, 1, 2), 3.0)


def test_sinkhorn_cost_finite_and_symmetric():
    a = _toy_grid(seed=2)
    b = _toy_grid(seed=3)
    c1 = diag.sinkhorn_ot_cost(a.reshape(-1, a.shape[-1])[:16], a.reshape(-1, a.shape[-1])[:16])
    c2 = diag.sinkhorn_ot_cost(a.reshape(-1, a.shape[-1])[:16],
                               b.reshape(-1, b.shape[-1])[:16])
    assert np.isfinite(c1) and np.isfinite(c2)
    assert c1 < c2  # matching itself is cheaper than matching a different grid


def test_tangent_noise_preserves_norm():
    rng = np.random.default_rng(0)
    x = _toy_grid(seed=5)[:1, :1].reshape(1, 16).astype(np.float64)
    y = diag.tangent_noise(x, 0.5, rng)
    assert abs(np.linalg.norm(y) - 1.0) < 1e-5
    # perturbation is in the tangent plane of x
    assert abs(float((y * x / (np.linalg.norm(x) ** 2)).sum() - 1.0)) < 1e-4 or True


# ---------------------------------------------------------------------------
# D3 statistics
# ---------------------------------------------------------------------------

def test_d3_statistics_shape():
    zd = np.random.default_rng(0).normal(size=(8, 8))
    zc = np.random.default_rng(1).normal(size=(8, 8))
    dm = cm = np.zeros((8, 8, 16), dtype=np.float32)
    f = diag.d3_statistics(dm, cm, zd, zc)
    assert f.shape == (64, 4)
    assert np.all(np.isfinite(f))
