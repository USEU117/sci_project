"""N3 / DNC - defect-vs-nuisance channel selection (doc 27 s7; CPU, k1 caches).

R0 gate ONLY on synthetic held-out families (doc27 s7 R0), normal-only fitting:
channels selected from TWO synthetic families (leave-one-family-out, 3 rotations),
evaluated by pooled pixel-AP on the THIRD family's masks with the reduced-feature
fused-A1 KNN path. GT masks are SYNTHETIC (cutpaste/local_erasure/thin_scratch on
GOOD images); no real anomaly mask is used in selection.

Channel response (frozen, per branch b, channel j):
  def_resp(g,k)  = robust_median(value[inside mask]) - robust_median(value[all cells])
                   (self-centred: how the defect moves the channel within that image)
  nui_resp(p)    = robust median over ref cells of |value_p - value_ref|   (15
                   photometric variants on the SUPPORT image; global nuisance scale)
  normal scale s_j = robust std (MAD*1.4826) of the support normal cells.
  q_j = median_g,k(|def_resp|/s_j) / (median_p(nui_resp/s_j) + 0.05)

Selections (KEEP_D=256/branch, total 512, single-branch controls 512 same total):
  DNC-I  : top-256 per branch by q_j
  DNC-C  : greedy with cross-branch redundancy penalty (lambda=0.3) over episode
           response correlation matrix (per-branch quotas preserved)
  random : 3 fixed per-branch masks (seeds 11,12,13)
  highvar: top-256 per branch by normal-cell variance
  low_nui: top-256 per branch by -nui_resp
  dino_only/clip_only: single-branch top-512 by q_j
  full   : all channels (frozen A1 bypass reference)

R0 pass rule (doc27 s7): held-out defect AP(DNC) - AP(random or low_nui) >= +0.02
on >=2 of the 3 rotations (macro over 6 cats). Fail -> no real-mask tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    sys.path.insert(0, p)

import faiss  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
NTOF_D = ROOT / "outputs/dynamic_fusion/ntof_features_dino_s0_k1"
NTOF_C = ROOT / "outputs/dynamic_fusion/ntof_features_clip_s0_k1"
MASK_D = ROOT / "outputs/dynamic_fusion/ntof_syn_masks_s0_k1"
SYN = ["cutpaste", "local_erasure", "thin_scratch"]
KEEP = 256
LAMBDA = 0.3
EPS_Q = 0.05
RANDOM_SEEDS = [11, 12, 13]
GRID32 = 32


def _robust_std(x):
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _cell_mask1024(m1024):
    """1024x1024 mask -> 32x32 cells (any-positive in 32px block)."""
    m = m1024.reshape(32, 32, 32, 32).any(axis=(1, 3))
    return m.astype(np.uint8)


def load_cat(cat):
    zd = np.load(NTOF_D / f"{cat}.npz", allow_pickle=False)
    zc = np.load(NTOF_C / f"{cat}.npz", allow_pickle=False)
    zm = np.load(MASK_D / f"{cat}.npz", allow_pickle=False)
    d_ref = np.asarray(zd["ref_orig_feat"])[0].astype(np.float64)       # [32,32,768]
    d_syn = np.asarray(zd["good_syn_feat"]).astype(np.float64)          # [G,3,32,32,768]
    d_held = np.asarray(zd["good_held_feat"]).astype(np.float64)        # [G,5,32,32,768]
    d_refv = np.asarray(zd["ref_var_feat"])[0].astype(np.float64)       # [15,32,32,768]
    c_ref = np.asarray(zc["ref_orig_feat"])[0]                          # [37,37,768]
    c_syn = np.asarray(zc["good_syn_feat"])                             # [G,3,37,37,768]
    c_held = np.asarray(zc["good_held_feat"])                           # [G,5,37,37,768]
    c_refv = np.asarray(zc["ref_var_feat"])[0]                          # [15,37,37,768]
    c_ref = resize_patches(c_ref[None], (GRID32, GRID32))[0].astype(np.float64)
    c_syn = resize_patches(c_syn.reshape(-1, 37, 37, 768), (GRID32, GRID32))
    c_syn = c_syn.reshape(*d_syn.shape).astype(np.float64)
    c_held = resize_patches(c_held.reshape(-1, 37, 37, 768), (GRID32, GRID32))
    c_held = c_held.reshape(*d_held.shape).astype(np.float64)
    c_refv = resize_patches(c_refv.reshape(-1, 37, 37, 768), (GRID32, GRID32))
    c_refv = c_refv.reshape(*d_refv.shape).astype(np.float64)
    syn_masks = np.asarray(zm["syn_masks"])                              # [G,3,1024,1024]
    masks32 = np.stack([np.stack([_cell_mask1024(syn_masks[g, k])
                                  for k in range(3)]) for g in range(syn_masks.shape[0])])
    return (d_ref, d_syn, d_held, d_refv, c_ref, c_syn, c_held, c_refv, masks32)


def _channel_stats(ref):                      # ref [32,32,768]
    cells = ref.reshape(-1, ref.shape[-1])    # [1024,768]
    med = np.median(cells, axis=0)
    s = 1.4826 * np.median(np.abs(cells - med), axis=0)
    return med, np.maximum(s, 1e-8)


def _resp_matrix(syn, masks32, sel_idx):      # syn [G,K,32,32,768] (K=3)
    """def_resp per episode & channel: median_inside - median_all, per channel."""
    G = syn.shape[0]
    resps = []
    for g in range(G):
        for k in sel_idx:
            m = masks32[g, k] > 0
            im = syn[g, k][m]
            am = syn[g, k].reshape(-1, syn.shape[-1])
            if im.shape[0] == 0:
                continue
            r = np.median(im, axis=0) - np.median(am, axis=0)
            resps.append(r)
    return np.stack(resps) if resps else np.zeros((1, syn.shape[-1]))    # [E,768]


def _nui_score(refv, ref):
    """photometric shift per channel: median over variants of |v-ref| cellwise."""
    shifts = []
    for p in range(refv.shape[0]):
        d = np.abs(refv[p] - ref).reshape(-1, ref.shape[-1])
        shifts.append(np.median(d, axis=0))
    return np.median(np.stack(shifts), axis=0)


def select_methods(qD, qC, varD, varC, nuiD, nuiC, respD, respC, rng_seeds):
    """Return dict method -> (selD, selC) index arrays."""
    out = {}
    top = lambda q: np.argsort(-q)[:KEEP]
    out["dnc_i"] = (top(qD), top(qC))
    # DNC-C greedy with cross-branch redundancy penalty
    nD, nC = respD.shape[1], respC.shape[1]
    E = respD.shape[0]
    if E > 2:
        Z = np.hstack([respD, respC]).T          # [1536, E]: samples in columns
        corr = np.corrcoef(Z)                    # [1536,1536]
        corrDC = corr[:nD, nD:]                  # [768D, 768C]
    else:
        corrDC = np.zeros((nD, nC))
    chosenD, chosenC = [], []
    qqD, qqC = qD.copy(), qC.copy()
    while len(chosenD) < KEEP or len(chosenC) < KEEP:
        best = None
        if len(chosenD) < KEEP and len(qqD) > 0:
            j = int(np.argmax(qqD))
            pen = max([abs(corrDC[j, k]) for k in chosenC], default=0.0)
            best = ("D", j, qqD[j] - LAMBDA * pen)
        if len(chosenC) < KEEP and len(qqC) > 0:
            k = int(np.argmax(qqC))
            pen = max([abs(corrDC[jj, k]) for jj in chosenD], default=0.0)
            cand = ("C", k, qqC[k] - LAMBDA * pen)
            if best is None or cand[2] > best[2]:
                best = cand
        if best is None:
            break
        b, idx, _ = best
        if b == "D":
            chosenD.append(idx)
            qqD[idx] = -np.inf
        else:
            chosenC.append(idx)
            qqC[idx] = -np.inf
    out["dnc_c"] = (np.asarray(chosenD), np.asarray(chosenC))
    for sd in RANDOM_SEEDS:
        rg = np.random.default_rng(sd)
        out[f"random{sd}"] = (rg.permutation(nD)[:KEEP], rg.permutation(nC)[:KEEP])
    out["highvar"] = (np.argsort(-varD)[:KEEP], np.argsort(-varC)[:KEEP])
    out["low_nui"] = (np.argsort(nuiD)[:KEEP], np.argsort(nuiC)[:KEEP])
    out["dino_only"] = (np.argsort(-qD)[:2 * KEEP], np.asarray([], dtype=int))
    out["clip_only"] = (np.asarray([], dtype=int), np.argsort(-qC)[:2 * KEEP])
    out["full"] = (np.arange(nD), np.arange(nC))
    return out


def fused_bank(d_ref, c_ref, selD, selC):
    """Reduced fused ref/query bank (A1 protocol on the reduced channel set)."""
    ncell = d_ref.reshape(-1, 768).shape[0]
    d = d_ref.reshape(-1, 768)[:, selD] if selD.size else np.zeros((ncell, 0))
    c = c_ref.reshape(-1, 768)[:, selC] if selC.size else np.zeros((ncell, 0))
    if d.shape[1] == 0:
        f = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
    elif c.shape[1] == 0:
        f = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    else:
        dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
        cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
        f = np.concatenate([0.5 * dn, 0.5 * cn], axis=-1)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    return np.ascontiguousarray(f, dtype=np.float32)


def run_cat(cat):
    (d_ref, d_syn, d_held, d_refv, c_ref, c_syn, c_held, c_refv, masks32) = load_cat(cat)
    medD, sD = _channel_stats(d_ref)
    medC, sC = _channel_stats(c_ref)
    varD = np.var(d_ref.reshape(-1, 768), axis=0)
    varC = np.var(c_ref.reshape(-1, 768), axis=0)
    nuiD = _nui_score(d_refv, d_ref) / sD
    nuiC = _nui_score(c_refv, c_ref) / sC
    rows = []
    for held in range(3):                    # rotate held-out family
        sel_idx = [k for k in range(3) if k != held]
        rD = _resp_matrix(d_syn, masks32, sel_idx)   # [E,768]
        rC = _resp_matrix(c_syn, masks32, sel_idx)
        qD = np.median(np.abs(rD) / sD, axis=0) / (nuiD + EPS_Q)
        qC = np.median(np.abs(rC) / sC, axis=0) / (nuiC + EPS_Q)
        methods = select_methods(qD, qC, varD, varC, nuiD, nuiC, rD, rC, RANDOM_SEEDS)
        # ref reduced banks per method
        for name, (selD, selC) in methods.items():
            ref_bank = fused_bank(d_ref, c_ref, selD, selC)
            idx = faiss.IndexFlatL2(ref_bank.shape[1])
            idx.add(ref_bank)
            # held-out family episodes AP
            aps = []
            for g in range(d_syn.shape[0]):
                dq = d_syn[g, held].reshape(-1, 768)
                cq = c_syn[g, held].reshape(-1, 768)
                m = masks32[g, held].ravel() > 0
                qb = fused_bank(dq.reshape(32, 32, 768), cq.reshape(32, 32, 768), selD, selC)
                dists, _ = idx.search(qb, 1)
                sc = (dists[:, 0] / 2.0)
                if m.sum() and (~m).sum():
                    aps.append(average_precision_score(m.astype(int), sc))
            # nuisance-FP proxy: mean p99 distance over photometric (defect-free) episodes
            fp99 = []
            for g in range(d_held.shape[0]):
                for h in range(d_held.shape[1]):
                    dq = d_held[g, h].reshape(-1, 768)
                    cq = c_held[g, h].reshape(-1, 768)
                    qb = fused_bank(dq.reshape(32, 32, 768), cq.reshape(32, 32, 768), selD, selC)
                    dists, _ = idx.search(qb, 1)
                    fp99.append(np.percentile(dists[:, 0] / 2.0, 99))
            rows.append({"cat": cat, "held_family": SYN[held], "method": name,
                         "ap": float(np.mean(aps)) if aps else float("nan"),
                         "fp_p99": float(np.mean(fp99))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    args = ap.parse_args()
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v13_overnight_20260904/N3_dnc"
    out_root.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    all_rows = []
    for cat in cats:
        all_rows += run_cat(cat)
        print(f"  done {cat}", flush=True)
    payload = {"rows": all_rows,
               "config": {"keep": KEEP, "lambda": LAMBDA, "eps_q": EPS_Q,
                          "families": SYN}}
    (out_root / "R0.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    # macro summary per (method, held_family) over cats
    summ = {}
    for r in all_rows:
        key = (r["held_family"], r["method"])
        summ.setdefault(key, []).append(r["ap"])
    agg = {f"{k[0]}|{k[1]}": round(float(np.mean(v)), 4) for k, v in summ.items()}
    print("AGG", json.dumps(agg, indent=0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
