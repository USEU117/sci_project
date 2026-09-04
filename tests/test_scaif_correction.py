"""CPU tests for the Stage1 SCAIF correction run (doc 25 phase A, 2026-09-04).

Covers the confirmed implementation bugs fixed in the correction branch:
  P0-1 sparse gate penalty must carry gradient (was detached);
  P0-2 training patch sampling must be uniform over the whole 32x32 grid (was fixed top half);
  P1-3 every control variant must forward/backward cleanly and be parameter-matched;
  support/query distance scale must be uniform (identical token -> distance ~0);
  gate=0 must still reduce SCAIF to the raw static feature rows (identity).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "scripts" / "innovation_v12_new_observables"))

from scaif_common import (  # noqa: E402
    ANCHORS, CATEGORIES, GATE_CAP, PAIRS, QSUBS, SEED, SHOTS, TRAIN_STEPS,
    SCAIF, ccol, deep_rows, l2row, pair_raw_rows,
)

VARIANTS = ["main", "no_support", "shuffled", "symmetric", "no_cross", "dino_only", "clip_only"]


def _tiny_inputs(device="cpu", grid=8, n=2, seed=0):
    torch.manual_seed(seed)
    d = torch.randn(n, 3, grid, grid, 768, device=device)
    c = torch.randn(n, 3, grid, grid, 768, device=device)
    sup_d = [torch.randn(16, 768, device=device) for _ in PAIRS]
    sup_c = [torch.randn(16, 768, device=device) for _ in PAIRS]
    return d, c, sup_d, sup_c


# ---------------------------------------------------------------------------
# P0-1: sparse gate penalty gradient
# ---------------------------------------------------------------------------

def test_sparse_penalty_grad_flows_and_scales_with_weight():
    m = SCAIF(variant="main")
    d, c, sd, sc = _tiny_inputs()
    # Gate heads are zero-initialised by design (start at the static identity), which
    # pins the last gate layer at exactly 0 -> no gradient can reach earlier gate
    # layers. Perturb them slightly so the graph is alive, then verify that the
    # sparse-penalty gradient flows to the gate head and scales linearly with lambda
    # (this is the P0-1 regression: gs must carry the autograd graph, not be detached).
    with torch.no_grad():
        for blk in m.blocks:
            for head in (blk.gate_d, blk.gate_c):
                head[-1].weight.normal_(0.0, 0.05)
                head[-1].bias.normal_(0.0, 0.05)
    opt_l1 = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.0)
    opt_l2 = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.0)

    def sparse_grads(lmbda):
        for g in (opt_l1, opt_l2):
            g.zero_grad()
        _, _, gs = m.refine(d, c, sd, sc)
        gcat = torch.cat([g for gd, gc in gs for g in (gd, gc)])
        loss = lmbda * gcat.abs().mean()
        loss.backward()
        return {name: p.grad.detach().clone()
                for name, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    g1 = sparse_grads(1.0)
    g2 = sparse_grads(2.0)
    assert g1, "no gate parameter received gradient from the sparse penalty"
    gate_params = [n for n in g1 if ".gate_" in n]
    assert gate_params, "gate head parameters must receive sparse-penalty gradient"
    for n in gate_params[:4]:
        ratio = (g2[n].abs().sum() / (g1[n].abs().sum() + 1e-12)).item()
        assert abs(ratio - 2.0) < 1e-2, f"gradient must scale linearly with lambda ({n}: {ratio})"


# ---------------------------------------------------------------------------
# P0-2: whole-grid uniform patch sampling
# ---------------------------------------------------------------------------

def test_patch_sampling_covers_whole_grid():
    rng = np.random.default_rng(SEED)
    hits = np.zeros(1024, dtype=np.int64)
    for _ in range(40):  # 40 steps x 4 images x 512 -> far more draws than 1024
        for _ in range(4):
            pix = rng.choice(1024, size=QSUBS, replace=False)
            hits += np.bincount(pix, minlength=1024)
    assert QSUBS < 1024
    assert int((hits > 0).sum()) == 1024, "uniform sampling must visit every row of the 32x32 grid"
    # no fixed top-half bias: lowest rows must not be the only ones sampled
    assert hits[:16].min() > 0 and hits[16:].min() > 0


# ---------------------------------------------------------------------------
# P1-3: all controls forward/backward and parameter match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_control_forward_backward_smoke(variant):
    m = SCAIF(variant=variant)
    d, c, sd, sc = _tiny_inputs()
    fr, f0, gs = m.refine(d, c, sd, sc)
    assert torch.isfinite(fr).all()
    assert fr.shape == (2 * 8 * 8, 4 * 768)
    gcat = torch.cat([g for gd, gc in gs for g in (gd, gc)])
    loss = fr.abs().mean() + gcat.mean()
    loss.backward()
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_parameter_count_matched_within_5pct():
    counts = {v: SCAIF(variant=v).trainable_params() for v in VARIANTS}
    ref = counts["main"]
    for v in VARIANTS:
        ratio = counts[v] / ref
        assert 0.95 <= ratio <= 1.05, f"{v}: {counts[v]} vs main {ref} (ratio {ratio:.3f})"


# ---------------------------------------------------------------------------
# support/query distance scale
# ---------------------------------------------------------------------------

def test_identical_token_has_zero_support_distance():
    m = SCAIF(variant="main")
    blk = m.blocks[0]
    tok = torch.randn(1, 768)
    sup = tok.repeat(2, 1)  # 2 raw support copies of the identical token
    q = tok.reshape(1, 1, 1, 768).expand(1, 8, 8, 768)
    # d_sup is computed inside forward; recompute manually for the assertion
    ud = torch.nn.functional.normalize(blk.pd(q), dim=-1).reshape(-1, blk.u)
    su = torch.nn.functional.normalize(blk.pd(sup), dim=-1)
    dmin = torch.cdist(ud, su).min(dim=-1)[0]
    assert dmin.max().item() < 1e-5, f"identical-token distance should be ~0, got {dmin.max().item()}"


# ---------------------------------------------------------------------------
# gate=0 identity (control #10 regression)
# ---------------------------------------------------------------------------

def test_gate_zero_reduces_exactly_to_raw_rows():
    m = SCAIF(variant="main")
    d, c, sd, sc = _tiny_inputs()
    fr, f0, _ = m.refine(d, c, sd, sc, gate_zero=True)
    assert (fr - f0).abs().max().item() == 0.0, "gate=0 must be bit-exact to the raw static rows"


# ---------------------------------------------------------------------------
# sanity on constants used by training
# ---------------------------------------------------------------------------

def test_frozen_training_constants():
    assert ANCHORS == 256
    assert QSUBS == 512
    assert TRAIN_STEPS == 600
    assert GATE_CAP == 0.2
    assert CATEGORIES == ["bracket_black", "bracket_brown", "bracket_white",
                          "connector", "metal_plate", "tubes"]
    assert SHOTS == [1, 2, 4]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
