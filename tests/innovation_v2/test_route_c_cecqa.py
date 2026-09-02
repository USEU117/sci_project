"""Route C (CE-CQA) unit tests: consensus intersection, shift cap, fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2 import consensus_query_adaptation as cecqa  # noqa: E402


def _normalize(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    return flat.reshape(arr.shape)


def make_aligned(n=6, s=2, grid=(4, 5), d=768, c=768, seed=0):
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


def test_score_cecqa_runs_and_finite():
    aligned = make_aligned()
    cfg = {"fixed": {"dino_weight": 0.5}, "_shot": 2,
           "postprocess": {"map_size": [448, 448]}}
    s, diag = cecqa.score_cecqa(aligned, {"q": 0.10, "eta": 0.25}, cfg)
    assert s.shape == (6, 4, 5)
    assert np.all(np.isfinite(s))
    assert diag["selection"] == "consensus"


def test_shift_is_capped():
    """Combined shift norm must never exceed the LOO 95th-percentile cap."""
    aligned = make_aligned(n=3, s=2, grid=(24, 24), seed=11)
    cfg = {"fixed": {"dino_weight": 0.5}, "_shot": 2,
           "postprocess": {"map_size": [448, 448]}}
    # Force a large deviation between query and ref so the shift would exceed cap.
    aligned.d_feat = _normalize(aligned.d_feat + 2.0)
    aligned.c_feat = _normalize(aligned.c_feat - 2.0)
    s, diag = cecqa.score_cecqa(aligned, {"q": 0.20, "eta": 0.50}, cfg)
    assert diag["mean_shift_norm"] is not None
    assert diag["mean_shift_norm"] <= diag["cap_loo_p95"] + 1e-5


def test_fallback_when_consensus_empty():
    """When |P_q| < 16 for every image the output must equal A1."""
    rng = np.random.default_rng(5)
    n, s, h, w = 4, 1, 2, 2  # tiny grid -> consensus set can never reach 16
    d = 768
    aligned = common.AlignedFeatures(
        d_feat=_normalize(rng.normal(size=(n, h, w, d))),
        c_feat=_normalize(rng.normal(size=(n, h, w, d))),
        d_ref=_normalize(rng.normal(size=(s, h, w, d))),
        c_ref=_normalize(rng.normal(size=(s, h, w, d))),
        grid=(h, w), sample_ids=np.arange(n).astype(str), ref_ids=["r0"],
    )
    cfg = {"fixed": {"dino_weight": 0.5}, "_shot": 1,
           "postprocess": {"map_size": [448, 448]}}
    s_new, diag = cecqa.score_cecqa(aligned, {"q": 0.10, "eta": 0.25}, cfg)
    assert diag["fallback_image_ratio"] == 1.0
    a1 = common.a1_grid(aligned)
    assert np.allclose(s_new, a1, atol=1e-6)


def test_a1_rank_control_runs():
    aligned = make_aligned()
    cfg = {"fixed": {"dino_weight": 0.5}, "_shot": 2,
           "postprocess": {"map_size": [448, 448]}}
    s, diag = cecqa.score_cecqa_a1_rank_only(aligned, {"q": 0.10, "eta": 0.25}, cfg)
    assert s.shape == (6, 4, 5)
    assert diag["selection"] == "a1_rank"


def test_never_self_match():
    """The adapted memory excludes the query itself (no zero-distance patches)."""
    aligned = make_aligned(n=6, s=2, grid=(4, 5), seed=2)
    cfg = {"fixed": {"dino_weight": 0.5}, "_shot": 2,
           "postprocess": {"map_size": [448, 448]}}
    s, _ = cecqa.score_cecqa(aligned, {"q": 0.10, "eta": 0.25}, cfg)
    # With all-random data the A1 distance is strictly positive; the adapted
    # score must be bounded away from 0 unless the memory actually contained
    # the query (which it must not).
    assert float(s.min()) > 1e-6
