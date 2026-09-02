"""Route D (DEVA) unit tests: transforms, inverse warp, equivariance filter."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2.equivariant_augmentation import (  # noqa: E402
    apply_transform, build_augmented_memory, equivariance_scores,
    inverse_warp_features, make_transforms,
)

CFG = {
    "deva": {
        "geometry": {"shift_frac": 0.04, "max_rotation_deg": 5.0},
        "photometric": {"brightness": [0.90, 1.10], "contrast": [0.90, 1.10]},
    }
}


def _normalize(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    return flat.reshape(arr.shape)


def make_aligned(n=4, s=2, grid=(32, 32), d=768, c=768, seed=0):
    rng = np.random.default_rng(seed)
    h, w = grid
    return common.AlignedFeatures(
        d_feat=_normalize(rng.normal(size=(n, h, w, d))),
        c_feat=_normalize(rng.normal(size=(n, h, w, c))),
        d_ref=_normalize(rng.normal(size=(s, h, w, d))),
        c_ref=_normalize(rng.normal(size=(s, 37, 37, c))),
        grid=grid, sample_ids=np.arange(n).astype(str),
        ref_ids=[f"r{i}" for i in range(s)], category="toy",
    )


def test_transform_counts():
    assert len(make_transforms("geometry", CFG)) == 6
    assert len(make_transforms("photometric", CFG)) == 4
    assert len(make_transforms("combined", CFG)) == 1


def test_identity_warp_is_identity():
    rng = np.random.default_rng(0)
    grid = _normalize(rng.normal(size=(32, 32, 64))).astype(np.float32)
    ident = np.float32([[1, 0, 0], [0, 1, 0]])
    warped, valid = inverse_warp_features(grid, ident, stride=14.0)
    assert valid.all()
    assert np.all(np.isfinite(warped))
    assert np.allclose(warped, grid, atol=1e-4)


def test_photometric_apply():
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    t = make_transforms("photometric", CFG)[0]  # b0.9_c0.9
    out = apply_transform(img, t, 32)
    assert out.shape == img.shape
    # contrast 0.9 dims the image slightly
    assert float(out.mean()) < float(img.mean()) + 1e-3


def test_equivariance_scores_nan_invalid():
    rng = np.random.default_rng(1)
    a = _normalize(rng.normal(size=(4, 5, 8)))
    b = a.copy()
    valid = np.ones((4, 5), dtype=bool)
    valid[0, 0] = False
    e = equivariance_scores(a, b, valid)
    assert np.isnan(e[0, 0])
    assert np.allclose(e[1:, :], 1.0, atol=1e-5)


def test_build_memory_acceptance_bounds():
    aligned = make_aligned(n=2, s=2, grid=(8, 8))
    h, w = aligned.grid
    rng = np.random.default_rng(2)
    # 3 augmentations sourced from refs 0,1,0
    aug_d = np.stack([aligned.d_ref[0] + 0.01, aligned.d_ref[1] + 0.01,
                      aligned.d_ref[0] + 0.5])
    aug_c = np.stack([aligned.c_ref[0] + 0.01, aligned.c_ref[1] + 0.01,
                      aligned.c_ref[0] + 0.5])
    aug_d = _normalize(aug_d.reshape(-1, 768)).reshape(aug_d.shape)
    aug_c = _normalize(aug_c.reshape(-1, 768)).reshape(aug_c.shape)
    vd = np.ones((3, h, w), dtype=bool)
    vc = np.ones((3, 37, 37), dtype=bool)
    src = np.asarray([0, 1, 0], dtype=np.int32)

    # tau=-1: accept everything valid -> memory grows
    d_all, c_all, meta = build_augmented_memory(aligned, aug_d, aug_c, vd, vc, src, -1.0)
    n_orig = aligned.n_references * h * w
    assert d_all.shape[0] > n_orig
    assert meta["n_accepted_patches_total"] == 3 * h * w

    # tau=1.0: accept nothing (cos < 1 for the +0.5 noise; +0.01 gives ~1.0 cos
    # but after renormalisation slightly below 1) -> memory equals original.
    d_all2, c_all2, meta2 = build_augmented_memory(aligned, aug_d, aug_c, vd, vc, src, 1.0)
    assert d_all2.shape[0] == n_orig
    assert meta2["n_accepted_patches_total"] == 0


def test_combined_transform_has_geometry_and_photometric():
    t = make_transforms("combined", CFG)[0]
    assert t.angle_deg != 0.0
    assert t.contrast is not None
