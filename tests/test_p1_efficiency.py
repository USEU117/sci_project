from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "p1_c_efficiency.py"
    spec = importlib.util.spec_from_file_location("p1_c_efficiency", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bank_stats_counts_all_spatial_patches(tmp_path):
    mod = _module()
    cache_root = tmp_path / "cache"
    branch = cache_root / "toy_s0_k2" / "visual"
    branch.mkdir(parents=True)
    np.savez_compressed(
        branch / "category_a.npz",
        ref_patch_features=np.zeros((2, 32, 32, 768), dtype=np.float16),
    )
    mod.CACHE_ROOT = cache_root
    stats = mod.bank_stats({"toy": "toy_s{seed}_k{shot}/visual"}, "toy", 0, 2)
    assert stats == {"n_ref_patches": 2 * 32 * 32, "n_ref_images": 2, "dim": 768}
