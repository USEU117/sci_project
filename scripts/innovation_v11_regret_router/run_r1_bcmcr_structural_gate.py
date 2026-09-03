"""V11 BC-MCR R0 - blind-center masked-context repair, structural gate (doc 21 s6.3).

Per-class tiny net predicts a grid cell's fused centre from its BLIND local context
(7x7 window minus the centre 3x3 = 40 cells) plus top-M support centre tokens
retrieved by context-mean similarity from the 2 reference grids; score =
1 - cos(predicted centre, actual centre).

Normal-material protocol mirrors the archived D2 diagnostic (MPDD seed0 shot2,
normal test/good grids synthetically perturbed: permutation / missing / duplicate,
block 4, dilate 1, seeded) so numbers are comparable. Controls per doc 21 s6.3:
  CTRL_COPY   centre 3x3 visible (49 slots)
  CTRL_CTX    no support tokens (query context only)
  CTRL_POS    no model: old D2 non-parametric ring->centre retrieval
  CTRL_SHUFFLE broken (context, centre) pairing in training
No GT, no bad images, no test aggregates anywhere.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v11_regret_router\\run_r1_bcmcr_structural_gate.py [--category X]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

from industrial_ad.innovation_v4_diagnostics import common, diagnostics as diag

SEED = 0
SHOT = 2
BS = 4
N_GOOD_MAX = 16
RNG_SEED = 20260902
H = W = 32
D = 1536
DIM = 128
R_OUT = 3
M_SUP = 16
LR = 2e-3
EPOCHS = 30
BATCH = 128
PATIENCE = 4
CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- context utils

def template_offsets(mode: str) -> np.ndarray:
    """(L,2) offsets in [-3,3]; blind keeps max|.| in {2,3} (40 cells); copy keeps all 49."""
    out = []
    for dr in range(-R_OUT, R_OUT + 1):
        for dc in range(-R_OUT, R_OUT + 1):
            if mode == "blind" and max(abs(dr), abs(dc)) <= 1:
                continue
            out.append((dr, dc))
    return np.asarray(out, dtype=np.int64)


OFF_BLIND = template_offsets("blind")
OFF_COPY = template_offsets("copy")


def qpos_slot(offsets: np.ndarray) -> np.ndarray:
    return ((offsets[:, 0] + R_OUT) * (2 * R_OUT + 1) + (offsets[:, 1] + R_OUT)).astype(np.int64)


QSLOT_BLIND = qpos_slot(OFF_BLIND)
QSLOT_COPY = qpos_slot(OFF_COPY)


def pad_grid(g: np.ndarray) -> np.ndarray:
    return np.pad(g, ((R_OUT, R_OUT), (R_OUT, R_OUT), (0, 0)), mode="edge")


def context_stack(padded: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """padded [H+6,W+6,D] -> ctx [H*W, L, D] (rows NOT re-normalised)."""
    parts = []
    for (dr, dc) in offsets:
        sl = padded[R_OUT + dr:R_OUT + dr + H, R_OUT + dc:R_OUT + dc + W]
        parts.append(sl.reshape(-1, D))
    return np.stack(parts, axis=1)


def grid_ctx_mean(ctx: np.ndarray) -> np.ndarray:
    """[H*W, L, D] -> [H*W, D] L2-unit context-mean (for support retrieval)."""
    m = ctx.mean(axis=1)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    return (m / n).astype(np.float32)


def l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n < 1e-9] = 1.0
    return (x / n).astype(np.float32)


# ---------------------------------------------------------------- model

class BCNet(nn.Module):
    def __init__(self, dim_in: int = D, dim: int = DIM):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim)
        self.ln = nn.LayerNorm(dim)
        self.qpos = nn.Embedding(49, dim)
        self.spos = nn.Embedding(H * W, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=256,
                                           dropout=0.0, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, dim_in)

    def forward(self, ctx: torch.Tensor, qpidx: torch.Tensor,
                sup: torch.Tensor | None, spos_idx: torch.Tensor | None) -> torch.Tensor:
        t = self.ln(self.proj(ctx)) + self.qpos(qpidx)       # [B,L,dim]
        n_q = t.shape[1]
        if sup is not None:
            s = self.ln(self.proj(sup)) + self.spos(spos_idx)
            t = torch.cat([t, s], dim=1)
        h = self.attn(t)[:, :n_q]
        pred = self.head(h.mean(dim=1))
        return nn.functional.normalize(pred, dim=-1)


def _faiss_topm(q: np.ndarray, bank: np.ndarray, m: int) -> np.ndarray:
    import faiss
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(np.ascontiguousarray(bank, dtype=np.float32))
    _, idx = index.search(np.ascontiguousarray(q, dtype=np.float32), m)
    return idx.astype(np.int64)


def _val_cells():
    yy, xx = np.meshgrid(np.arange(10, 22), np.arange(10, 22), indexing="ij")
    return (yy * W + xx).ravel()


# ---------------------------------------------------------------- training

def train_model(ctx_all: np.ndarray, centers_all: np.ndarray, bank_cm: np.ndarray,
                qslot: np.ndarray, use_support: bool, shuffle_targets: bool,
                rng: np.random.Generator) -> BCNet:
    """ctx_all [S*H*W, L, D]; centers_all [S*H*W, D] (targets, L2-unit)."""
    model = BCNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_ref = centers_all.shape[0] // (H * W)
    val_idx = np.concatenate([_val_cells() + r * H * W for r in range(n_ref)])
    tr_idx = np.setdiff1d(np.arange(centers_all.shape[0]), val_idx)
    targets = centers_all.copy()
    if shuffle_targets:
        targets[tr_idx] = centers_all[rng.permutation(tr_idx)]

    ctx_t = torch.from_numpy(ctx_all).float()
    tgt_t = torch.from_numpy(targets).float()
    pos_bank = np.concatenate([np.arange(H * W) for _ in range(n_ref)]).astype(np.int64)
    qslot_t = torch.from_numpy(qslot).long()

    def step(idx: np.ndarray, train: bool) -> float:
        c = ctx_t[torch.from_numpy(idx).long()].to(DEV)
        y = tgt_t[torch.from_numpy(idx).long()].to(DEV)
        qp = qslot_t.to(DEV).repeat(len(c), 1)
        if use_support:
            qm = grid_ctx_mean(c.cpu().numpy())
            nbr = _faiss_topm(qm, bank_cm, M_SUP)
            sup = torch.from_numpy(centers_all[nbr]).float().to(DEV)
            sp = torch.from_numpy(pos_bank[nbr]).long().to(DEV)
        else:
            sup = sp = None
        pred = model(c, qp, sup, sp)
        loss = (1.0 - (pred * y).sum(-1)).mean()
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach().cpu())

    best_val, best_state, bad = 1e9, None, 0
    for _ in range(EPOCHS):
        order = rng.permutation(tr_idx)
        for s in range(0, len(order), BATCH):
            step(order[s:s + BATCH], True)
        vsum = 0.0
        for s in range(0, len(val_idx), BATCH):
            vsum += step(val_idx[s:s + BATCH], False)
        vloss = vsum / max(1, int(np.ceil(len(val_idx) / BATCH)))
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------- scoring

def score_grid(model: BCNet, padded: np.ndarray, offsets: np.ndarray,
               use_support: bool, bank_cm: np.ndarray, centers_all: np.ndarray,
               pos_bank: np.ndarray) -> np.ndarray:
    """-> [H, W] residual grid (1 - cos(pred, centre))."""
    ctx = context_stack(padded, offsets)                       # [1024, L, D]
    qm = grid_ctx_mean(ctx)
    if use_support:
        nbr = _faiss_topm(qm, bank_cm, M_SUP)
    preds = []
    with torch.no_grad():
        for s in range(0, ctx.shape[0], BATCH):
            c = torch.from_numpy(ctx[s:s + BATCH]).float().to(DEV)
            qp = torch.from_numpy(np.broadcast_to(offsets_slot(offsets),
                                                  (len(c), len(offsets)))).long().to(DEV)
            if use_support:
                sup = torch.from_numpy(centers_all[nbr[s:s + BATCH]]).float().to(DEV)
                sp = torch.from_numpy(pos_bank[nbr[s:s + BATCH]]).long().to(DEV)
            else:
                sup = sp = None
            preds.append(model(c, qp, sup, sp).cpu().numpy())
    pred = np.concatenate(preds)
    center = padded[R_OUT:-R_OUT, R_OUT:-R_OUT].reshape(-1, D)
    cos = (pred * l2_rows(center.astype(np.float32))).sum(axis=1)
    return np.clip(1.0 - cos, 0.0, 2.0).reshape(H, W).astype(np.float32)


_SLOT_CACHE = {}


def offsets_slot(offsets: np.ndarray) -> np.ndarray:
    key = offsets.tobytes()
    if key not in _SLOT_CACHE:
        _SLOT_CACHE[key] = qpos_slot(offsets)
    return _SLOT_CACHE[key]


# ---------------------------------------------------------------- category

def _cat_seed(cat: str) -> int:
    """Deterministic per-category offset (hash() is salted per-process)."""
    return sum(ord(ch) for ch in cat) % 1000


def run_category(cat: str) -> dict:
    aligned = common.aligned_category("mpdd", SEED, SHOT, cat)
    good_ids = [i for i, sid in enumerate(aligned.sample_ids) if "/good/" in str(sid)]
    rng_ids = good_ids[:N_GOOD_MAX]
    rng = np.random.default_rng(RNG_SEED + _cat_seed(cat))
    torch.manual_seed(4321 + _cat_seed(cat))

    q = diag.fused_grid(aligned.d_feat, aligned.c_feat)   # [N,32,32,1536]
    r = diag.fused_grid(aligned.d_ref, aligned.c_ref)     # [S,32,32,1536]
    good_grids = [q[i] for i in rng_ids]

    pert = {t: [] for t in ("permutation", "missing", "duplicate")}
    for gi, g in enumerate(good_grids):
        donor = good_grids[(gi + 1) % len(good_grids)]
        for t in pert:
            pg, m = diag.perturb_structural(g, donor, t, rng, bs=BS)
            pert[t].append((pg, diag.dilate_mask(m, iters=1)))

    # normal memory from the two refs
    ref_padded = [pad_grid(g) for g in r]
    centers_all = np.concatenate([g.reshape(-1, D) for g in r]).astype(np.float32)
    centers_all = l2_rows(centers_all)
    bank_cm = np.concatenate([grid_ctx_mean(context_stack(p, OFF_BLIND)) for p in ref_padded])
    pos_bank = np.concatenate([np.arange(H * W) for _ in range(len(ref_padded))]).astype(np.int64)
    ctx_blind = np.concatenate([context_stack(p, OFF_BLIND) for p in ref_padded])
    ctx_copy = np.concatenate([context_stack(p, OFF_COPY) for p in ref_padded])

    # CTRL-POS old fixed retrieval bank (D2 parity)
    ring_bank = diag.ring_bank_context(r, r_in=1, r_out=2)
    center_bank_all = r.reshape(-1, D)

    # ---- train
    print(f"[{cat}] training FULL/CTRL_COPY/CTRL_CTX/CTRL_SHUFFLE", flush=True)
    t0 = time.time()
    m_full = train_model(ctx_blind, centers_all, bank_cm, QSLOT_BLIND, True, False, rng)
    print(f"    FULL done {time.time()-t0:.0f}s", flush=True)
    m_copy = train_model(ctx_copy, centers_all, bank_cm, QSLOT_COPY, True, False, rng)
    m_ctx = train_model(ctx_blind, centers_all, bank_cm, QSLOT_BLIND, False, False, rng)
    m_shuf = train_model(ctx_blind, centers_all, bank_cm, QSLOT_BLIND, True, True, rng)
    torch.cuda.empty_cache()

    def run_variant(model, ctx_offsets, use_sup, grids) -> np.ndarray:
        return np.stack([score_grid(model, pad_grid(g), ctx_offsets, use_sup,
                                    bank_cm, centers_all, pos_bank) for g in grids])

    per_type = {}
    for t in ("permutation", "missing", "duplicate"):
        grids = np.stack([p for p, _ in pert[t]])
        masks = np.stack([m for _, m in pert[t]])
        pos_gt = (masks.ravel() > 0)
        neg_gt = ~pos_gt
        a1v = diag.memory_dist1(grids, r)
        q_ring = np.stack([diag.ring_mean(p, 1, 2) for p in grids])
        cpos = diag.context_residual(grids, q_ring, ring_bank, center_bank_all)
        res = {
            "A1": diag.patch_auroc(np.concatenate([s.ravel() for s in a1v]), pos_gt, neg_gt),
            "CTRL_POS": diag.patch_auroc(np.concatenate([s.ravel() for s in cpos]),
                                         pos_gt, neg_gt),
            "FULL": None, "CTRL_COPY": None, "CTRL_CTX": None, "CTRL_SHUFFLE": None,
        }
        res["FULL"] = diag.patch_auroc(
            np.concatenate([s.ravel() for s in run_variant(m_full, OFF_BLIND, True, grids)]),
            pos_gt, neg_gt)
        res["CTRL_COPY"] = diag.patch_auroc(
            np.concatenate([s.ravel() for s in run_variant(m_copy, OFF_COPY, True, grids)]),
            pos_gt, neg_gt)
        res["CTRL_CTX"] = diag.patch_auroc(
            np.concatenate([s.ravel() for s in run_variant(m_ctx, OFF_BLIND, False, grids)]),
            pos_gt, neg_gt)
        res["CTRL_SHUFFLE"] = diag.patch_auroc(
            np.concatenate([s.ravel() for s in run_variant(m_shuf, OFF_BLIND, True, grids)]),
            pos_gt, neg_gt)
        per_type[t] = res
        print(f"    {t}: " + " ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    return {"category": cat, "seed": SEED, "shot": SHOT, "per_type": per_type}


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v11_regret_router/bc_mcr")
    args = parser.parse_args()
    common.assert_development_only()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    cats = [args.category] if args.category else CATEGORIES
    rows = []
    for cat in cats:
        print(f"[BC-MCR] {cat}", flush=True)
        rows.append(run_category(cat))

    keys = ["A1", "CTRL_POS", "FULL", "CTRL_COPY", "CTRL_CTX", "CTRL_SHUFFLE"]
    ptypes = ["permutation", "missing", "duplicate"]

    def mean_auroc(ptype, key):
        vals = [r["per_type"][ptype][key] for r in rows]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    per_type_mean = {t: {k: mean_auroc(t, k) for k in keys} for t in ptypes}
    report = {
        "route": "V11-BCMCR",
        "pipeline": "v11_bcmcr_structural_gate",
        "seed": SEED, "shot": SHOT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "per_type_mean": per_type_mean,
        "gates": {
            "g1_each_type_ge_075": all((per_type_mean[t]["FULL"] or 0.0) >= 0.75 for t in ptypes),
            "g2_each_type_ge_A1_plus_010": all(
                (per_type_mean[t]["FULL"] or 0.0) >= (per_type_mean[t]["A1"] or 0.0) + 0.10
                for t in ptypes),
            "g3_missing_ge_CTRLPOS_plus_015": (per_type_mean["missing"]["FULL"] or 0.0) >= (
                per_type_mean["missing"]["CTRL_POS"] or 0.0) + 0.15,
            "g4_FULL_ge_each_ctrl_per_type": all(
                (per_type_mean[t]["FULL"] or 0.0) >= (per_type_mean[t][c] or 0.0) for t in ptypes
                for c in ("CTRL_COPY", "CTRL_CTX", "CTRL_POS", "CTRL_SHUFFLE")),
            "g5_FULL_minus_best_ctrl_mean_ge_005": round(float(np.mean([
                (per_type_mean[t]["FULL"] or 0.0) - max(per_type_mean[t][c] or 0.0 for c in
                                                        ("CTRL_COPY", "CTRL_CTX", "CTRL_POS",
                                                         "CTRL_SHUFFLE"))
                for t in ptypes])), 4) >= 0.05,
        },
        "decision": "PENDING",
    }
    (out_dir / "R0_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["per_type_mean"], indent=1))
    print("gates:", report["gates"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
