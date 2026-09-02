"""Unit tests for innovation_v6_dgsafe maps tooling (task book 16 S0).

Synthetic/equivalence tests only; no real MPDD feature load, no GT handed to
algorithm code paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402


def test_align_perm_identity_and_reorder():
    a = np.asarray(["x/1.png", "x/2.png", "x/3.png"])
    b = np.asarray(["x/2.png", "x/3.png", "x/1.png"])
    perm = maps.align_perm(b, a)
    np.testing.assert_array_equal(b[perm], a)
    with pytest.raises(ValueError):
        maps.align_perm(np.asarray(["x/1.png", "x/1.png"]), np.asarray(["x/1.png", "x/2.png"]))
    with pytest.raises(ValueError):
        maps.align_perm(np.asarray(["x/9.png"]), a)


def test_grid_to_map448_deterministic_shape():
    g = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
    m1 = maps.grid_to_map448(g)
    m2 = maps.grid_to_map448(g)
    assert m1.shape == (448, 448)
    np.testing.assert_array_equal(m1, m2)
    assert m1.dtype == np.float32


def test_a1_maps448_stack_shape():
    pm = np.zeros((3, 32, 32), dtype=np.float32)
    assert maps.a1_maps448(pm).shape == (3, 448, 448)


def test_pixel_metrics_matches_manual_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(1)
    maps_arr = rng.normal(size=(4, 64, 64)).astype(np.float32)
    gtm = (rng.random((4, 64, 64)) > 0.95).astype(np.uint8)
    got = maps.pixel_metrics_448(maps_arr, gtm)
    y = gtm.ravel()[::8]
    s = maps_arr.ravel()[::8]
    assert got["pixel_ap"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert got["pixel_auroc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    # all-good/no-pos -> None
    assert maps.pixel_metrics_448(maps_arr, np.zeros_like(gtm))["pixel_ap"] is None


def test_gt_masks_good_is_zero_and_missing_raises(tmp_path):
    ids = np.asarray(["fake/test/good/1.png"])
    m = maps.gt_masks_for(ids, data_root=tmp_path)
    assert m.shape == (1, 448, 448) and m.sum() == 0
    bad = np.asarray(["fake/test/bad/1.png"])
    with pytest.raises(FileNotFoundError):
        maps.gt_masks_for(bad, data_root=tmp_path)


def test_protocol_pre_registered_values():
    assert maps.MAP_SIZE == (448, 448)
    assert maps.STRIDE == 8
    assert maps.PROTOCOL["subspacead"]["pca_ev"] == 0.99
    assert maps.PROTOCOL["subspacead"]["image_res"] == 672
    assert maps.PROTOCOL["identity_replay_tolerance_pixel_ap"] == 0.0005


def test_development_dataset_only():
    maps.assert_development_only()  # must not raise
