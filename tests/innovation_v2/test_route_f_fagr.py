"""Route F (FAGR) unit tests: uniform smoothing degradation, weights, stability."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2 import feature_graph_refinement as fagr  # noqa: E402


def _normalize(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    return flat.reshape(arr.shape)


def make_aligned(n=4, s=2, grid=(4, 5), d=768, c=768, seed=0):
    rng = np.random.default_rng(seed)
    h, w = grid
    return common.AlignedFeatures(
        d_feat=_normalize(rng.normal(size=(n, h, w, d))),
        c_feat=_normalize(rng.normal(size=(n, h, w, c))),
        d_ref=_normalize(rng.normal(size=(s, h, w, d))),
        c_ref=_normalize(rng.normal(size=(s, h, w, c))),
        grid=grid, sample_ids=np.arange(n).astype(str),
        ref_ids=[f"r{i}" for i in range(s)],
    )


def _hand_uniform(s0, mu, iters):
    """Manual uniform-weight Jacobi on the 4-neighbour grid."""
    h, w = s0.shape
    s = s0.copy()
    for _ in range(iters):
        up = np.roll(s, 1, axis=0); up[0] = 0.0
        down = np.roll(s, -1, axis=0); down[-1] = 0.0
        left = np.roll(s, 1, axis=1); left[:, 0] = 0.0
        right = np.roll(s, -1, axis=1); right[:, -1] = 0.0
        deg = np.ones_like(s) * 4.0
        deg[0] = 3.0; deg[-1] = 3.0; deg[:, 0] = 3.0; deg[:, -1] = 3.0
        deg[0, 0] = 2.0; deg[0, -1] = 2.0; deg[-1, 0] = 2.0; deg[-1, -1] = 2.0
        s = (s0 + mu * (up + down + left + right)) / (1.0 + mu * deg)
    return s


def test_uniform_control_matches_hand_computation():
    rng = np.random.default_rng(0)
    s0 = rng.random((4, 5)).astype(np.float32)
    mu, iters = 0.5, 3
    d_feat = _normalize(rng.normal(size=(1, 4, 5, 8)))
    out = fagr.fagr_iterate(s0[None], d_feat, mu, iters, tau=0.1, uniform=True)
    expected = _hand_uniform(s0, mu, iters)
    assert np.allclose(out[0], expected, atol=1e-6)


def test_edge_weights_nonnegative():
    rng = np.random.default_rng(1)
    d_feat = _normalize(rng.normal(size=(2, 4, 5, 8)))
    s0 = rng.random((2, 4, 5)).astype(np.float32)
    mu, iters, tau = 0.5, 2, 0.10
    out = fagr.fagr_iterate(s0, d_feat, mu, iters, tau, uniform=False)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)


def test_constant_score_is_fixed_point():
    rng = np.random.default_rng(2)
    d_feat = _normalize(rng.normal(size=(2, 4, 5, 8)))
    s0 = np.full((2, 4, 5), 0.25, dtype=np.float32)
    out = fagr.fagr_iterate(s0, d_feat, 0.5, 3, 0.10, uniform=False)
    assert np.allclose(out, s0, atol=1e-6)


def test_score_fagr_shapes():
    aligned = make_aligned()
    cfg = {"fagr": {"tau": 0.10}, "postprocess": {"map_size": [448, 448]},
           "fixed": {"dino_weight": 0.5}}
    s, diag = fagr.score_fagr(aligned, {"mu": 0.10, "iters": 1}, cfg)
    assert s.shape == (4, 4, 5)
    assert np.all(np.isfinite(s))
    assert diag["uniform_control"] is False


def test_bilinear_map_shape():
    grid = np.random.rand(4, 4, 5).astype(np.float32)
    m = fagr.bilinear_map(grid, (448, 448))
    assert m.shape == (4, 448, 448)
    assert np.all(np.isfinite(m))


def test_uniform_control_runs():
    aligned = make_aligned()
    cfg = {"fagr": {"tau": 0.10}, "postprocess": {"map_size": [448, 448]},
           "fixed": {"dino_weight": 0.5}}
    s, diag = fagr.score_fagr_uniform(aligned, {"mu": 0.50, "iters": 3}, cfg)
    assert s.shape == (4, 4, 5)
    assert diag["uniform_control"] is True
