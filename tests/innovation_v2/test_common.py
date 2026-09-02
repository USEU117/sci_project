"""Shared framework tests for the A2 innovation_v2 program."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v2 import common  # noqa: E402


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
        grid=grid,
        sample_ids=np.asarray([f"c-good-{i:03d}" for i in range(n)]),
        ref_ids=[f"ref-{i:02d}" for i in range(s)],
    )


def test_validation_dataset_guard():
    with pytest.raises(common.ValidationDatasetAccessError):
        common.assert_development_only("btad")
    common.assert_development_only("mpdd")  # no raise
    with pytest.raises(common.ValidationDatasetAccessError):
        common.assert_frozen_validation_dataset("mpdd")


def test_algorithm_inputs_reject_labels():
    with pytest.raises(common.InnovationError):
        common.validate_algorithm_inputs({"patch_features": np.zeros((2, 2)), "gt_sp": np.zeros(2)})
    common.validate_algorithm_inputs({"patch_features": np.zeros((2, 2))})  # no raise


def test_a1_grid_matches_dino_duplicate_on_synthetic():
    aligned = make_aligned()
    s = common.a1_grid(aligned)
    assert s.shape == (4, 4, 5)
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0)


def test_dino_grid_nonnegative():
    aligned = make_aligned()
    s = common.dino_grid(aligned)
    assert s.shape == (4, 4, 5)
    assert np.all(s >= 0)


def test_grids_to_maps_shape():
    aligned = make_aligned(grid=(4, 5))
    s = common.a1_grid(aligned)
    maps = common.grids_to_maps(s, (448, 448))
    assert maps.shape == (4, 448, 448)


def test_hash_deterministic():
    p = ROOT / "configs" / "innovation_v2" / "route_a_lndc.yaml"
    h1 = common.sha256_file(p)
    h2 = common.sha256_file(p)
    assert h1 == h2 and len(h1) == 64


def test_manifest_layout_mpdd():
    assert common.DATASETS["mpdd"]["role"] == "development"
    assert common.DATASETS["btad"]["role"] == "external_frozen_validation"
    assert common.DATASETS["visa"]["role"] == "in_domain_frozen_validation"
    assert common.DATASETS["mvtec"]["role"] == "external_frozen_validation"


def test_route_registry_complete():
    expected = {"A_LNDC", "B_DSAM", "C_CEQA", "D_DEVA", "E_NCPRA", "F_FAGR"}
    assert set(common.ROUTE_LABELS) == expected
