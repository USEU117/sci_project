"""N4 / PMC - fidelity-preserving coreset of dual-branch normal memory (doc27 s8).

REDUNDANT-first: check whether a tri-distance (dino+clip+concat normalised coverage)
minimax greedy differs from concat-only greedy at equal budget. If near-identical
(Jaccard high), mark REDUNDANT and stop there (doc27 s8).

Budget ratios 25% / 50% of the SUPPORT fused bank. Controls: full A1 bank, same-ratio
random (3 seeds), concat greedy coreset, tri greedy, per-branch-merge (each branch
greedy over half the budget, union). Metric = frozen A1 protocol pooled Pixel-AP@56
vs real masks on MPDD six classes, k2 & k4 (CPU; faiss).

R1 engineering gate (doc27 s8): at 50% memory macro-AP loss <= +0.003 (i.e. drop of
at most 0.003), worst class >= -0.01, matching-stage speedup >=25% (bank-size proxy).
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
from src.utils import dists2map  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
ML = ROOT / "outputs/dynamic_fusion/v12_early_fusion"
RATIOS = [0.25, 0.50]
SEEDS = [21, 22, 23]
GRID = 32


def load_final(cat, shot):
    zd = np.load(ML / f"ml_dino_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    zc = np.load(ML / f"ml_clip_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    d_feat = np.asarray(zd["patch_features"])[:, 2]
    d_ref = np.asarray(zd["ref_patch_features"])[2]
    c_feat = np.asarray(zc["patch_features"])[:, 3]
    c_ref = np.asarray(zc["ref_patch_features"])[3]
    masks = np.asarray(zd["imgs_masks"], dtype=np.uint8)
    c_feat32 = resize_patches(c_feat, (GRID, GRID))
    c_ref32 = resize_patches(c_ref, (GRID, GRID))
    return d_feat, d_ref, c_feat32, c_ref32, masks


def _row_l2(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    f = x.reshape(-1, x.shape[-1])
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    return f.reshape(x.shape)


def fused_rows(d, c):
    dn = _row_l2(d).reshape(-1, d.shape[-1])
    cn = _row_l2(c).reshape(-1, c.shape[-1])
    f = np.concatenate([0.5 * dn, 0.5 * cn], axis=-1)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


def nn_dists(q, bank):
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(np.ascontiguousarray(bank, dtype=np.float32))
    d2, _ = idx.search(np.ascontiguousarray(q, dtype=np.float32), 1)
    return d2[:, 0] / 2.0


def _row_dists(X, i):
    """L2sq/2 from row i to every row (rows pre-normalised)."""
    d = ((X - X[i]) ** 2).sum(axis=1) / 2.0
    return d


def greedy_farthest(X, k, seed):
    """k-center-ish farthest-first on rows of X. Returns indices."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(n))]
    dmin = np.full(n, np.inf)
    while len(sel) < k:
        d = _row_dists(X, sel[-1])
        dmin = np.minimum(dmin, d)
        j = int(np.argmax(dmin))
        sel.append(j)
    return np.asarray(sel)


def greedy_tri(d_ref, c_ref, k, seed):
    """minimax over three normalised coverage errors (dino/clip/concat)."""
    n = d_ref.shape[0]
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(n))]
    dd = _row_l2(d_ref).reshape(n, -1)
    cc = _row_l2(c_ref).reshape(n, -1)
    ff = fused_rows(d_ref, c_ref)
    # static normalisation scales (median of sampled pairwise distances)
    r = rng.integers(0, n, 64)
    scale = lambda X: float(np.median(_row_dists(X, r[0])[r]))  # noqa: E731
    sD, sC, sF = scale(dd), scale(cc), scale(ff)
    eD = np.full(n, np.inf)
    eC = np.full(n, np.inf)
    eF = np.full(n, np.inf)
    while len(sel) < k:
        last = sel[-1]
        eD = np.minimum(eD, _row_dists(dd, last))
        eC = np.minimum(eC, _row_dists(cc, last))
        eF = np.minimum(eF, _row_dists(ff, last))
        err = np.maximum(eD / max(sD, 1e-8), np.maximum(eC / max(sC, 1e-8), eF / max(sF, 1e-8)))
        err[sel] = -np.inf
        sel.append(int(np.argmax(err)))
    return np.asarray(sel)


def pooled_ap(dist_maps56, m56):
    y = (m56.ravel() > 0.5).astype(np.int32)
    s = dist_maps56.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def run_cat_shot(cat, shot):
    d_feat, d_ref, c_feat32, c_ref32, masks = load_final(cat, shot)
    K = d_ref.shape[0]
    cells = K * GRID * GRID
    d_ref_f = d_ref.reshape(cells, -1)
    c_ref_f = c_ref32.reshape(cells, -1)
    d_feat_f = d_feat.reshape(-1, d_feat.shape[-1])
    c_feat_f = c_feat32.reshape(-1, c_feat32.shape[-1])
    f_ref = fused_rows(d_ref_f, c_ref_f)
    nq = d_feat.shape[0]
    # full query fused
    fq = fused_rows(d_feat_f, c_feat_f)
    m56 = (masks[:, ::8, ::8] > 0.5).astype(np.uint8)

    def ap_for(sel):
        bank = f_ref[sel]
        dists = nn_dists(fq, bank)
        maps56 = np.stack([dists2map((dists.reshape(nq, GRID, GRID)[i]), (448, 448))[::8, ::8]
                           for i in range(nq)])
        return pooled_ap(maps56, m56), len(sel)

    rows = []
    ap_full, nfull = ap_for(np.arange(cells))
    rows.append({"method": "full", "ratio": 1.0, "ap": ap_full, "n": nfull})
    # redundancy + fidelity at budgets
    for ratio in RATIOS:
        k = max(int(round(cells * ratio)), 1)
        # random (3 seeds)
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            sel = rng.permutation(cells)[:k]
            ap, n = ap_for(sel)
            rows.append({"method": "random", "seed": sd, "ratio": ratio, "ap": ap, "n": n})
        # concat greedy
        sel = greedy_farthest(f_ref, k, seed=0)
        ap, n = ap_for(sel)
        rows.append({"method": "concat_greedy", "ratio": ratio, "ap": ap, "n": n})
        # tri greedy
        sel = greedy_tri(d_ref_f, c_ref_f, k, seed=0)
        ap, n = ap_for(sel)
        rows.append({"method": "tri_greedy", "ratio": ratio, "ap": ap, "n": n})
        # branch-merge: half concat-greedy on dino, half on clip, union cells
        kh = max(int(round(k / 2)), 1)
        selD = greedy_farthest(_row_l2(d_ref_f).reshape(cells, -1), kh, seed=0)
        selC = greedy_farthest(_row_l2(c_ref_f).reshape(cells, -1), kh, seed=1)
        sel = np.unique(np.concatenate([selD, selC]))
        ap, n = ap_for(sel)
        rows.append({"method": "branch_merge", "ratio": ratio, "ap": ap, "n": n,
                     "n_union": int(len(sel))})
        # redundancy check: overlap concat vs tri (same k)
        s1 = greedy_farthest(f_ref, k, seed=0)
        s2 = greedy_tri(d_ref_f, c_ref_f, k, seed=0)
        jac = float(len(np.intersect1d(s1, s2)) / k)
        rows.append({"method": "_jaccard_concat_vs_tri", "ratio": ratio, "jaccard": jac, "k": k})
    return {"cat": cat, "shot": shot, "rows": rows, "cells": cells}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v13_overnight_20260904/N4_pmc"
    out_root.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    all_rows = []
    for shot in args.shots:
        for cat in cats:
            out = run_cat_shot(cat, shot)
            for r in out["rows"]:
                r["cat"] = cat
                r["shot"] = shot
            all_rows += out["rows"]
            print(f"  done {cat} k{shot} cells={out['cells']}", flush=True)
    payload = {"rows": all_rows}
    (out_root / "R0.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    # loss at 50% vs full macro (per cat-shot then macro)
    agg = {}
    for r in all_rows:
        if "ap" not in r:
            continue
        key = (r["method"], r["ratio"])
        agg.setdefault(str(key), []).append(r["ap"])
    print("AGG", json.dumps({k: round(float(np.nanmean(v)), 4) for k, v in agg.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
