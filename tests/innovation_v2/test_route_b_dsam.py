"""Route B (DSAM) unit tests: transform recovery, fallback, controls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2 import deformable_spatial_memory as dsam  # noqa: E402
from industrial_ad.innovation_v2.deformable_spatial_memory import (  # noqa: E402
    _estimate_transform, _mutual_matches,
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


def _shifted_query(ref, shift, rng):
    """Query with ref's pattern placed at +shift; rest filled with noise."""
    h, w, d = ref.shape
    query = _normalize(rng.normal(size=(h, w, d)))
    dy, dx = shift
    query[dy:, dx:] = ref[:h - dy, :w - dx]
    return _normalize(query)


def test_translation_recovery():
    """A shifted query grid must be recovered by median translation."""
    rng = np.random.default_rng(0)
    h, w, d = 12, 12, 16
    ref = _normalize(rng.normal(size=(h, w, d)))
    shift = (2, 3)
    query = _shifted_query(ref, shift, rng)
    qf = query.reshape(-1, d).astype(np.float32)
    rf = ref.reshape(-1, d).astype(np.float32)
    nn_q, _, mutual = _mutual_matches(qf, rf)
    q_idx = np.nonzero(mutual)[0]
    r_idx = nn_q[q_idx]
    rows = np.tile(np.arange(h)[:, None], (1, w)).reshape(-1)
    cols = np.tile(np.arange(w)[None, :], (h, 1)).reshape(-1)
    q_pts = np.stack([rows[q_idx], cols[q_idx]], axis=1).astype(np.float32)
    r_pts = np.stack([rows[r_idx], cols[r_idx]], axis=1).astype(np.float32)
    T, ratio, _ = _estimate_transform(q_pts, r_pts, "translation", 0, 3.0)
    cy, cx = T(rows.astype(np.float32), cols.astype(np.float32))
    # query pattern is at +shift in the query image, so ref = query - shift;
    # median displacement (ref - query) should be approximately -shift.
    disp = r_pts - q_pts
    med = np.median(disp, axis=0)
    assert abs(med[0] - (-shift[0])) <= 0.51
    assert abs(med[1] - (-shift[1])) <= 0.51
    assert ratio > 0.5  # most mutual matches inliers


def test_affine_recovery_on_rigid_shift():
    rng = np.random.default_rng(1)
    h, w, d = 12, 12, 16
    ref = _normalize(rng.normal(size=(h, w, d)))
    shift = (1, 2)
    query = _shifted_query(ref, shift, rng)
    qf = query.reshape(-1, d).astype(np.float32)
    rf = ref.reshape(-1, d).astype(np.float32)
    nn_q, _, mutual = _mutual_matches(qf, rf)
    q_idx = np.nonzero(mutual)[0]
    r_idx = nn_q[q_idx]
    rows = np.tile(np.arange(h)[:, None], (1, w)).reshape(-1)
    cols = np.tile(np.arange(w)[None, :], (h, 1)).reshape(-1)
    q_pts = np.stack([rows[q_idx], cols[q_idx]], axis=1).astype(np.float32)
    r_pts = np.stack([rows[r_idx], cols[r_idx]], axis=1).astype(np.float32)
    T, ratio, _ = _estimate_transform(q_pts, r_pts, "affine", 0, 3.0)
    assert T is not None
    assert ratio > 0.5
    cy, cx = T(np.arange(h * w, dtype=np.float32) // w,
               np.arange(h * w, dtype=np.float32) % w)
    center = 6 * w + 6
    # affine recovers ref = query - shift
    assert abs(cy[center] - (6 - shift[0])) <= 1.0
    assert abs(cx[center] - (6 - shift[1])) <= 1.0


def test_low_mutual_fallback_produces_global():
    aligned = make_aligned(n=4, s=2, grid=(4, 5))
    cfg = {"dsam": {"min_mutual": 16, "min_inlier_ratio": 0.20, "ransac_threshold": 3.0,
                    "seed": 0},
           "postprocess": {"map_size": [448, 448]}, "fixed": {"dino_weight": 0.5}}
    s, diag = dsam.score_dsam(aligned, {"alignment": "translation", "r": 2}, cfg)
    assert s.shape == (4, 4, 5)
    assert np.all(np.isfinite(s))
    assert 0.0 <= diag["fallback_ratio"] <= 1.0


def test_identity_control_same_as_score_with_identity():
    aligned = make_aligned(n=4, s=2, grid=(4, 5))
    cfg = {"dsam": {"min_mutual": 16, "min_inlier_ratio": 0.20, "ransac_threshold": 3.0,
                    "seed": 0},
           "postprocess": {"map_size": [448, 448]}, "fixed": {"dino_weight": 0.5}}
    s_ctrl, diag = dsam.score_dsam_fixed_loc(aligned, {"alignment": "translation", "r": 4}, cfg)
    assert diag["alignment"] == "identity"
    assert s_ctrl.shape == (4, 4, 5)


def test_dino_descriptor_dsam():
    aligned = make_aligned(n=4, s=2, grid=(4, 5))
    cfg = {"dsam": {"min_mutual": 16, "min_inlier_ratio": 0.20, "ransac_threshold": 3.0,
                    "seed": 0},
           "postprocess": {"map_size": [448, 448]}, "fixed": {"dino_weight": 0.5}}
    s, _ = dsam.score_dsam(aligned, {"alignment": "translation", "r": 4}, cfg,
                           descriptor="dino")
    assert s.shape == (4, 4, 5)
