"""Unit tests for Wave2a probe geometry (inverse-warp consistency + z math).

These guard the frozen reliability probe: geometric light-normal versions are
applied in the 672 frame and their 48x48 residual grids inverse-warped back to
the identity frame; a coding error here would silently corrupt U_aug / the
per-pixel calibration pools.
"""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
W2A = ROOT / "scripts" / "innovation_v6_dgsafe" / "run_wave2a_build_reliability.py"
spec = importlib.util.spec_from_file_location("wave2a_probe", W2A)
w2a = importlib.util.module_from_spec(spec)
sys.modules["wave2a_probe"] = w2a
spec.loader.exec_module(w2a)


def _smooth672(rng, res=672):
    base = rng.random((48, 48)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3.0)          # low-frequency field
    return cv2.resize(base, (res, res), interpolation=cv2.INTER_CUBIC)


def _grid48(img672):
    return cv2.resize(img672, (48, 48), interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("kind", ["t2p", "t2n", "rot_p", "rot_n"])
def test_geo_inverse_warp_roundtrip(kind):
    rng = np.random.default_rng(0)
    img = _smooth672(rng)
    m_id = _grid48(img).astype(np.float64)
    if kind == "t2p":
        A, t = w2a.translate_affine(0.02 * 672, 0.0)
    elif kind == "t2n":
        A, t = w2a.translate_affine(-0.02 * 672, 0.0)
    elif kind == "rot_p":
        A, t = w2a.rotation_affine(2.0, 672)
    else:
        A, t = w2a.rotation_affine(-2.0, 672)
    M = w2a.affine_dest2src(A, t)
    warped = cv2.warpAffine(img, M, (672, 672), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    m_warp = _grid48(warped).astype(np.float64)
    m_orig = w2a.inv_warp_grid(np.asarray(m_warp, np.float32), A, t)
    # crop borders (replicate padding) for the comparison
    c = 5
    a, b = m_id[c:-c, c:-c], m_orig[c:-c, c:-c]
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert corr > 0.90, f"{kind}: corr={corr}"   # gross sanity (low-contrast field)
    assert float(np.median(np.abs(a - b))) < 0.02, kind
    # residual geometric shift between reconstructed and identity maps ~ 0
    resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    dx, dy = resp[0]
    assert abs(dx) < 0.3 and abs(dy) < 0.3, f"{kind}: residual shift ({dx:.3f},{dy:.3f})"


def test_z_from_pool_monotone():
    # pool with two low scores and one high score -> z monotone in score
    grids = [np.full((48, 48), 0.1), np.full((48, 48), 0.1),
             np.full((48, 48), 5.0)]
    zs = w2a.z_from_pool(grids)
    assert zs[2].mean() > zs[0].mean()
    assert float(zs[2].max()) <= w2a.Z_CAP + 1e-9
    # doc formula: even the floor sample keeps p=(1+n)/(2+n) -> bounded z>0
    z_floor = -np.log((1 + 3) / (2 + 3))
    assert abs(float(zs[0].mean()) - z_floor) < 1e-9


def test_percentile_ranks_risk():
    pr = w2a.percentile_ranks_risk([1.0, 2.0, 3.0, 4.0])
    assert pr == [0.25, 0.5, 0.75, 1.0]
    pr_tie = w2a.percentile_ranks_risk([1.0, 1.0, 3.0])
    assert pr_tie == [0.5, 0.5, 1.0]   # ties averaged
