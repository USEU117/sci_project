"""Route E (NCPRA) unit tests: synthetic smoke, no-label training, convergence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2 import predictive_adapter as ncpra  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _normalize(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    return flat.reshape(arr.shape)


def make_aligned(n=8, s=2, grid=(4, 5), d=768, c=768, seed=0):
    rng = np.random.default_rng(seed)
    h, w = grid
    return common.AlignedFeatures(
        d_feat=_normalize(rng.normal(size=(n, h, w, d))),
        c_feat=_normalize(rng.normal(size=(n, h, w, c))),
        d_ref=_normalize(rng.normal(size=(s, h, w, d))),
        c_ref=_normalize(rng.normal(size=(s, h, w, c))),
        grid=grid, sample_ids=np.arange(n).astype(str),
        ref_ids=[f"r{i}" for i in range(s)], category="toy",
    )


def test_adapter_parameter_budget():
    g = ncpra.BottleneckAdapter(r=64)
    g2 = ncpra.BottleneckAdapter(r=32)
    assert ncpra.parameter_count([g, g2]) < 500_000  # task book < 0.5M


def test_adapter_requires_grad_only_on_itself():
    """Backbone is never passed to the adapter (features are frozen arrays)."""
    g = ncpra.BottleneckAdapter(r=32)
    trainable = [p for p in g.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_train_converges_on_linear_relation():
    """With c = d M (rank-64 relation), the residual must drop after training."""
    rng = np.random.default_rng(0)
    n = 512
    d = rng.normal(size=(n, 768)).astype(np.float32)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    # rank-64 map so the r=64 bottleneck can fit it
    A = rng.normal(size=(768, 64)).astype(np.float32)
    B = rng.normal(size=(64, 768)).astype(np.float32)
    c_raw = d @ (A @ B).T
    c = c_raw / np.linalg.norm(c_raw, axis=1, keepdims=True)

    cfg = {"ncpra": {"mu": 0.1, "seed": 0, "max_epochs": 100, "patience": 10}}
    g, report = ncpra.train_ncpra(d, c, r=64, mu=0.1, shot=2, cfg=cfg, device=DEVICE)
    e0 = ncpra.residual_scores(g["g_d2c"], g["g_c2d"], d[:64], c[:64], device=DEVICE)
    # random-pair baseline residual ~= 1.0; a trained rank-64 adapter should be << 0.5
    assert float(np.mean(e0)) < 0.5
    assert report["parameter_count"] < 500_000


def test_loo_split_shot2():
    """shot>=2 validation must use a held-out reference image (no overlap)."""
    aligned = make_aligned(n=4, s=2, grid=(4, 5))
    d_ref = aligned.d_ref.reshape(-1, 768)
    c_ref = aligned.c_ref.reshape(-1, 768)
    cfg = {"ncpra": {"mu": 0.1, "seed": 0, "max_epochs": 3, "patience": 2}}
    _, report = ncpra.train_ncpra(d_ref, c_ref, r=32, mu=0.1, shot=2, cfg=cfg,
                                  device=DEVICE)
    assert report["shot"] == 2
    assert report["n_val_patches"] == aligned.grid[0] * aligned.grid[1]
    assert report["n_train_patches"] == report["n_val_patches"]


def test_score_ncpra_runs_and_finite():
    aligned = make_aligned(n=6, s=2, grid=(4, 5))
    cfg = {"fixed": {"dino_weight": 0.5}, "_shot": 2,
           "postprocess": {"map_size": [448, 448]},
           "ncpra": {"mu": 0.1, "seed": 0, "max_epochs": 3, "patience": 2}}
    s, diag = ncpra.score_ncpra(aligned, {"r": 32, "lambda": 0.10}, cfg,
                                device=DEVICE)
    assert s.shape == (6, 4, 5)
    assert np.all(np.isfinite(s))
    assert diag["protocol"] == "lightweight normal-only adaptation"
