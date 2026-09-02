"""Route A (LNDC) unit tests: LOO exclusion and hand-computed ratios."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2 import local_density  # noqa: E402
from industrial_ad.innovation_v2.local_density import (  # noqa: E402
    _loo_exclusion_mask, fused_flat, lndc_scores, ref_density_rho,
)


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


def test_loo_mask_shot2_excludes_same_image():
    h, w, s = 2, 2, 2
    mask = _loo_exclusion_mask(s * h * w, (h, w), s, shot=2)
    assert mask[0, 1].item() is True        # same image
    assert mask[0, h * w].item() is False   # different image


def test_loo_mask_shot1_excludes_self_and_chebyshev1():
    h, w, s = 5, 5, 1
    mask = _loo_exclusion_mask(h * w, (h, w), s, shot=1)
    center = 2 * w + 2
    # self and 8 neighbours excluded
    assert int(mask[center].sum()) == 9
    assert mask[center, 0].item() is False  # corner (0,0) at Chebyshev distance 2


def test_rho_equals_hand_computed_median():
    rng = np.random.default_rng(3)
    h, w, s = 2, 2, 2
    grid = (h, w)
    d = 8
    refs = rng.normal(size=(s, h, w, d)).astype(np.float32)
    refs = refs / np.linalg.norm(refs, axis=-1, keepdims=True)
    z = refs.reshape(-1, d).astype(np.float32)
    k = 2
    rho = ref_density_rho(z, grid, s, shot=2, k=k, epsilon=1e-6)
    # brute force: exclude same image, take median of allowed k distances
    n = z.shape[0]
    D = 0.5 * np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1) ** 2
    img = np.arange(n) // (h * w)
    for i in range(n):
        allowed = D[i, img != img[i]]
        order = np.argsort(allowed)[:k]
        expected = float(np.median(allowed[order]))
        assert rho[i] == pytest.approx(expected, rel=1e-5)


def test_lndc_scores_monotone_under_constant_rho():
    rng = np.random.default_rng(7)
    n_q, n_r, d = 20, 10, 8
    q = rng.normal(size=(n_q, d)).astype(np.float32)
    r = rng.normal(size=(n_r, d)).astype(np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)
    k = 3
    s1 = lndc_scores(q, r, rho=1.0, k=k, epsilon=1e-6)
    s2 = lndc_scores(q, r, rho=2.0, k=k, epsilon=1e-6)
    assert np.allclose(s1, s2 * 2.0, rtol=1e-4)


def test_score_lndc_shapes_and_diagnostics():
    aligned = make_aligned(n=4, s=2, grid=(4, 5))
    cfg = {"lndc": {"epsilon": 1e-6}, "_shot": 2,
           "postprocess": {"map_size": [448, 448]}, "fixed": {"dino_weight": 0.5}}
    s, diag = local_density.score_lndc(aligned, {"k": 3}, cfg)
    assert s.shape == (4, 4, 5)
    assert np.all(np.isfinite(s))
    assert diag["k"] == 3


def test_score_lndc_dino_descriptor():
    aligned = make_aligned(n=4, s=1, grid=(4, 5))
    cfg = {"lndc": {"epsilon": 1e-6}, "_shot": 1,
           "postprocess": {"map_size": [448, 448]}, "fixed": {"dino_weight": 0.5}}
    s, diag = local_density.score_lndc(aligned, {"k": 5}, cfg, descriptor="dino")
    assert s.shape == (4, 4, 5)
    assert diag["descriptor"] == "dino"


def test_global_sham_uses_single_rho():
    aligned = make_aligned(n=4, s=2, grid=(4, 5))
    cfg = {"lndc": {"epsilon": 1e-6}, "_shot": 2,
           "postprocess": {"map_size": [448, 448]}, "fixed": {"dino_weight": 0.5}}
    s, diag = local_density.score_lndc_global_sham(aligned, {"k": 3}, cfg)
    assert s.shape == (4, 4, 5)
    assert diag["sham"] is True
    assert diag["global_rho"] > 0
