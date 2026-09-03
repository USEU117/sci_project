"""V12 CECW - Cross-Encoder Conformity Work (doc 22 s4) R0 runner.

Reproduces published ANoCo (arXiv:2605.28428, CVPR 2026) as strong
training-free baselines on the FROZEN project features and then evaluates the
minimal closed-form CECW coupling (doc 22 s4.1) with the required controls.

Pre-registered protocol:
    experiments/dynamic_fusion/innovation_v12_new_observables/cecw/R0_PROTOCOL.json

Run (seed0 development only, MPDD):
    .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r1_cecw.py --shot 1
    .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r1_cecw.py --shot 2
    .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r1_cecw.py --shot 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import faiss
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize as sk_normalize

from industrial_ad.innovation_v10_portfolio.common import (
    MAP_SIZE,
    STRIDE,
    build_fused_blocks,
    load_features,
    resize_patches,
)
from industrial_ad.innovation_v8_tcrr_probe.regions import robust01
from src.utils import dists2map

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
M_PREFIX = 64          # candidate prefix for ANoCo anchor-driven retrieval
LAMBDA_Q = 1.0         # ANoCo query feature stabilization coefficient (paper-flat default)
R_RANK = 16            # CECW alignment rank (support-only fit)
LAMBDA_C = 1.0         # CECW conflict-coupling weight
CHUNK = 256            # patch chunk size for memory bounded batch solves
RNG_SEED = 0

METHOD_ORDER = [
    "a1",
    "anoco_dino", "anoco_clip", "anoco_concat", "fixed_mean",
    "cecw", "ctrl_shuffled", "ctrl_noconflict", "ctrl_qq", "ctrl_smoothing",
]
BASELINES = ["anoco_dino", "anoco_clip", "anoco_concat", "fixed_mean"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ helpers

def _l2rows(x: np.ndarray) -> np.ndarray:
    return sk_normalize(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)


def to56(grid_map: np.ndarray) -> np.ndarray:
    return dists2map(grid_map, MAP_SIZE)[::STRIDE, ::STRIDE].astype(np.float32)


def a1_grid_distance(feat: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """fused A1 1-NN squared-L2/2 grid map [N,32,32] (identical to mtcoa E0)."""
    hh, ww, d = feat.shape[1], feat.shape[2], feat.shape[-1]
    q = feat.reshape(-1, d).astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(bank.astype(np.float32))
    dists, _ = index.search(q, k=1)
    return (dists[:, 0] / 2.0).reshape(feat.shape[0], hh, ww).astype(np.float32)


def _pooled_ap(maps56: np.ndarray, masks56: np.ndarray) -> float:
    y = (masks56.ravel() > 0.5).astype(np.int32)
    s = maps56.ravel().astype(np.float64)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def _pooled_auroc(maps56: np.ndarray, masks56: np.ndarray) -> float:
    y = (masks56.ravel() > 0.5).astype(np.int32)
    s = maps56.ravel().astype(np.float64)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(roc_auc_score(y, s))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled per-patch Spearman rank correlation (tie-robust via rank mean)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    def rankdata(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
        # average ties
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = float(np.mean(ranks[order[i:j + 1]]))
            i = j + 1
        return ranks

    ra = rankdata(a)
    rb = rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------- ANoCo internals

def anoco_internals(q_img: np.ndarray, bank: np.ndarray, m: int = M_PREFIX):
    """ANoCo bipartite anchored graph internals for one image.

    q_img [P, D] unit-norm query rows; bank [B, D] unit-norm ref rows.
    Returns per-patch a (stiffness D_i + LAMBDA_Q), b (D_i*f_q - sum w f_r),
    and the count of retrieved neighbors, using cosine affinity with the
    pre-registered anchor-driven rule (M-prefix, positive edges).
    """
    p, d = q_img.shape
    q32 = q_img.astype(np.float32)
    b32 = bank.astype(np.float32)
    index = faiss.IndexFlatIP(d)
    index.add(b32)
    a_vec = np.empty((p,), dtype=np.float64)
    b_mat = np.empty((p, d), dtype=np.float64)
    deg = np.empty((p,), dtype=np.float64)
    n_nei = np.empty((p,), dtype=np.int64)
    for s in range(0, p, CHUNK):
        e = min(s + CHUNK, p)
        sub = q32[s:e]
        S, I = index.search(sub, k=m)          # cosine, descending
        nbr = b32[I]                            # [c, m, d]
        anch = nbr[:, 0, :]                     # anchor = argmax row
        a_cos = np.einsum("cd,cmd->cm", anch, nbr)  # cos(anchor, candidate)
        top = S[:, 0]
        mask = (a_cos > top[:, None]) & (S > 0.0)
        mask[:, 0] = True                        # anchor always included
        W = np.where(mask, S, 0.0).astype(np.float32)
        D = W.sum(axis=1)                        # [c]
        c = np.einsum("cm,cmd->cd", W, nbr)      # [c, d]
        n_nei[s:e] = (W > 0).sum(axis=1)
        deg[s:e] = D
        a_vec[s:e] = D + LAMBDA_Q
        b_mat[s:e] = D[:, None] * sub.astype(np.float64) - c.astype(np.float64)
    return a_vec, b_mat, deg, n_nei


def anoco_score(a_vec: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
    """score_i = ||b_i|| / a_i (magnitude of the closed-form update)."""
    valid = a_vec > 0.0
    score = np.full(a_vec.shape[0], np.nan, dtype=np.float64)
    score[valid] = np.linalg.norm(b_mat[valid], axis=1) / a_vec[valid]
    return score


# ------------------------------------------------------- CECW joint coupling

def fit_alignment(ref_d: np.ndarray, ref_c: np.ndarray, r: int = R_RANK,
                  shuffle: bool = False, seed: int = RNG_SEED):
    """Support-only low-rank orthogonal alignment (top-r SVD of cross-cov).

    ref_d/ref_c: [n_patches, d] unit-norm matched DINO/CLIP support rows.
    Returns Pd (r x d), Pc (r x d), row-orthonormal.
    """
    xd = ref_d.astype(np.float64)
    xc = ref_c.astype(np.float64)
    xd = xd - xd.mean(axis=0, keepdims=True)
    xc = xc - xc.mean(axis=0, keepdims=True)
    if shuffle:
        rng = np.random.RandomState(seed)
        xc = xc[rng.permutation(xc.shape[0])]
    m = xc.T @ xd
    u, _s, vh = np.linalg.svd(m, full_matrices=False)
    pd = vh[:r].astype(np.float32)
    pc = (u[:, :r]).T.astype(np.float32)
    return pd, pc


def _batch_solve(bD, bC, aD, aC, Pd, Pc, lam):
    """Closed-form block solve of the CECW joint quadratic over a batch.

    min  aD||dD||^2 + 2 bD.dD + aC||dC||^2 + 2 bC.dC
       + lam ||Pd dD - Pc dC||^2
    Returns dD, dC, work, conflict (all [batch, ...]).
    """
    bD = bD.astype(np.float64)
    bC = bC.astype(np.float64)
    aD = aD.astype(np.float64)
    aC = aC.astype(np.float64)
    n = bD.shape[0]
    r = Pd.shape[0]
    d = bD.shape[1]
    SD = (Pd.astype(np.float64) @ Pd.astype(np.float64).T)
    SC = (Pc.astype(np.float64) @ Pc.astype(np.float64).T)
    Ir = np.eye(r, dtype=np.float64)

    if lam == 0.0:
        dD = -bD / aD[:, None]
        dC = -bC / aC[:, None]
    else:
        inv_lam = 1.0 / lam
        # per-patch K_D, T_D (r x r)
        KD = np.linalg.inv(inv_lam * Ir[None] + (1.0 / aD)[:, None, None] * SD[None])
        TD = np.linalg.inv(aD[:, None, None] * Ir[None] + lam * SD[None])
        # K_C
        KC = np.linalg.inv(inv_lam * Ir[None] + (1.0 / aC)[:, None, None] * SC[None])
        # P_D b_D  (batch r-vec) and (P_D B_D b_D)
        pDb = bD @ Pd.T                       # [n, r]
        KDpDb = np.einsum("nij,nj->ni", KD, pDb)
        S_KDpDb = np.einsum("ij,nj->ni", SD, KDpDb)
        PDBDb = (1.0 / aD)[:, None] * pDb - (1.0 / (aD ** 2))[:, None] * S_KDpDb  # P_D B_D b_D
        # B_D b_D (full d-vec)
        BDb = (1.0 / aD)[:, None] * bD - (1.0 / (aD ** 2))[:, None] * (KDpDb @ Pd)
        dD0 = -BDb
        # Q = lam (I - lam T_D S_D)
        TSD = np.einsum("nij,jk->nik", TD, SD)
        Q = lam * (Ir[None] - lam * TSD)
        Qinv = np.linalg.inv(Q)
        # F_C = (Q^{-1} + aC^{-1} S_C)^{-1}
        FC = np.linalg.inv(Qinv + (1.0 / aC)[:, None, None] * SC[None])
        # rhs_C = -bC - lam P_C^T (P_D B_D b_D)
        rhsC = -bC - lam * (PDBDb @ Pc.astype(np.float64))   # PDBDb[n,r] @ Pc[r,d]
        pCrhs = rhsC @ Pc.T                                   # [n, r]
        FCpCrhs = np.einsum("nij,nj->ni", FC, pCrhs)
        dC = (1.0 / aC)[:, None] * rhsC - (1.0 / (aC ** 2))[:, None] * (FCpCrhs @ Pc)
        # dD = dD0 + lam P_D^T T_D (P_C dC)
        pCdC = dC @ Pc.T                                      # [n, r]
        TDpCdC = np.einsum("nij,nj->ni", TD, pCdC)
        dD = dD0 + lam * (TDpCdC @ Pd.astype(np.float64))

    pDdD = dD @ Pd.T
    pCdC = dC @ Pc.T
    diff = pDdD - pCdC
    sq = (aD * (dD ** 2).sum(axis=1) + aC * (dC ** 2).sum(axis=1)
          + lam * (diff ** 2).sum(axis=1))
    work = np.sqrt(np.maximum(sq, 0.0))
    conflict = np.linalg.norm(diff, axis=1)
    return dD, dC, work, conflict


def cecw_map(aD, bD, aC, bC, Pd, Pc, grid, fallback_grid, lam=LAMBDA_C):
    """CECW work map [P] for one image with A1-distance fallback on degenerate."""
    p = bD.shape[0]
    n_fb = 0
    valid = (aD > 0.0) & (aC > 0.0) & np.isfinite(bD).all(axis=1) & np.isfinite(bC).all(axis=1)
    work = np.full((p,), np.nan, dtype=np.float64)
    idx = np.where(valid)[0]
    for s in range(0, len(idx), CHUNK):
        sel = idx[s:s + CHUNK]
        _dD, _dC, w, _c = _batch_solve(bD[sel], bC[sel], aD[sel], aC[sel], Pd, Pc, lam)
        work[sel] = w
    bad = ~valid
    n_fb = int(bad.sum())
    if n_fb:
        work[bad] = fallback_grid[bad]      # fused A1 distance for that patch
    return work, n_fb


# ------------------------------------------------------------------ category

def run_category(dino_cache: Path, clip_cache: Path, cat: str, shot: int) -> dict:
    t0 = time.time()
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    grid = tuple(int(v) for v in dino["grid_size"])          # (32, 32)
    masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
    masks56 = (masks[:, ::STRIDE, ::STRIDE] > 0.5).astype(np.uint8)
    n = dino["patch_features"].shape[0]
    s = dino["ref_patch_features"].shape[0]

    # ---- per-branch unit-norm grids (CLIP resized to the DINO grid)
    d_feat = _l2rows(np.asarray(dino["patch_features"], dtype=np.float32))
    d_ref = _l2rows(np.asarray(dino["ref_patch_features"], dtype=np.float32))
    c_feat_raw = np.asarray(clip["patch_features"], dtype=np.float32)
    c_ref_raw = np.asarray(clip["ref_patch_features"], dtype=np.float32)
    if c_feat_raw.shape[1:3] != grid:
        c_feat_raw = resize_patches(c_feat_raw, grid)
    if c_ref_raw.shape[1:3] != grid:
        c_ref_raw = resize_patches(c_ref_raw, grid)
    c_feat = _l2rows(c_feat_raw)
    c_ref = _l2rows(c_ref_raw)

    # test-order alignment: reorder CLIP test features to the DINO sample order
    di = dino["sample_ids"]
    ci = clip["sample_ids"]
    if not np.array_equal(di, ci):
        pos = {str(v): k for k, v in enumerate(ci)}
        c_feat = c_feat[np.asarray([pos[str(v)] for v in di], dtype=np.int64)]

    # ---- A1 fused identity (concat grid + bank)
    ffeat, fref, _ids, _m, _g = build_fused_blocks(dino, clip, dino_weight=0.5)
    fbank = fref.reshape(-1, fref.shape[-1]).astype(np.float32)
    a1_gridv = a1_grid_distance(ffeat, fbank)                 # [n,32,32]
    a1_map = np.stack([to56(g) for g in a1_gridv]).astype(np.float32)

    # ---- reference banks per branch
    d_bank = d_ref.reshape(-1, d_ref.shape[-1]).astype(np.float32)
    c_bank = c_ref.reshape(-1, c_ref.shape[-1]).astype(np.float32)

    # ---- support-only alignment fit (matched DINO<->CLIP ref patches)
    pd_, pc_ = fit_alignment(d_ref.reshape(-1, d_ref.shape[-1]),
                             c_ref.reshape(-1, c_ref.shape[-1]), R_RANK, shuffle=False)
    pd_s, pc_s = fit_alignment(d_ref.reshape(-1, d_ref.shape[-1]),
                               c_ref.reshape(-1, c_ref.shape[-1]), R_RANK, shuffle=True)

    # ---- per-image per-method grid maps
    maps56 = {m: np.zeros((n, 56, 56), dtype=np.float32) for m in METHOD_ORDER}
    n_fallback = {m: 0 for m in METHOD_ORDER}
    n_nei_stats = {"dino": [], "clip": []}
    flat_cecw, flat_a1, flat_concat = [], [], []

    for i in range(n):
        qd = d_feat[i].reshape(-1, d_feat.shape[-1])
        qc = c_feat[i].reshape(-1, c_feat.shape[-1])
        qf = ffeat[i].reshape(-1, ffeat.shape[-1])

        # ---- ANoCo single branches: internals are shared with CECW
        aD, bD, degD, neiD = anoco_internals(qd, d_bank)
        aC, bC, degC, neiC = anoco_internals(qc, c_bank)
        n_nei_stats["dino"].append(int(neiD.mean()))
        n_nei_stats["clip"].append(int(neiC.mean()))

        scD = anoco_score(aD, bD)
        scC = anoco_score(aC, bC)
        # fallback to A1 distance where ANoCo degenerate
        fbD = a1_gridv[i].ravel()
        fbd = ~np.isfinite(scD)
        scD[fbd] = fbD[fbd]
        n_fallback["anoco_dino"] += int(fbd.sum())
        fbc = ~np.isfinite(scC)
        scC[fbc] = fbD[fbc]
        n_fallback["anoco_clip"] += int(fbc.sum())

        # ---- ANoCo on A1 concat features
        aF, bF, _degF, _neiF = anoco_internals(qf, fbank)
        scF = anoco_score(aF, bF)
        fbf = ~np.isfinite(scF)
        scF[fbf] = fbD[fbf]
        n_fallback["anoco_concat"] += int(fbf.sum())

        # ---- fixed mean of robust01 single-branch maps
        m01 = 0.5 * (robust01(scD.reshape(grid)) + robust01(scC.reshape(grid)))
        maps56["fixed_mean"][i] = to56(m01)
        # ---- ANoCo displacement-only grids (recorded as maps too)
        maps56["anoco_dino"][i] = to56(scD.reshape(grid))
        maps56["anoco_clip"][i] = to56(scC.reshape(grid))
        maps56["anoco_concat"][i] = to56(scF.reshape(grid))

        # ---- CECW joint work (correct and shuffled alignment)
        wok, nf = cecw_map(aD, bD, aC, bC, pd_, pc_, grid, fbD, LAMBDA_C)
        n_fallback["cecw"] += nf
        maps56["cecw"][i] = to56(wok.reshape(grid))
        wsh, nfs = cecw_map(aD, bD, aC, bC, pd_s, pc_s, grid, fbD, LAMBDA_C)
        n_fallback["ctrl_shuffled"] += nfs
        maps56["ctrl_shuffled"][i] = to56(wsh.reshape(grid))

        # ---- no-conflict control (lambda_c = 0 -> independent optima)
        wn, _ = cecw_map(aD, bD, aC, bC, pd_, pc_, grid, fbD, 0.0)
        maps56["ctrl_noconflict"][i] = to56(wn.reshape(grid))

        # ---- q-q smoothing control (1 round, 4-neighbour, on the CECW work grid)
        wg = wok.reshape(grid).copy()
        wg_p = np.pad(wg, 1, mode="edge")
        agg = (wg_p[0:-2, 1:-1] + wg_p[2:, 1:-1] + wg_p[1:-1, 0:-2] + wg_p[1:-1, 2:]) / 4.0
        maps56["ctrl_qq"][i] = to56(0.5 * wg + 0.5 * agg)
        # ---- extra Laplacian smoothing control (another pass of the same)
        sm = 0.5 * wg + 0.5 * agg
        sm_p = np.pad(sm, 1, mode="edge")
        sm2 = (sm_p[0:-2, 1:-1] + sm_p[2:, 1:-1] + sm_p[1:-1, 0:-2] + sm_p[1:-1, 2:]) / 4.0
        maps56["ctrl_smoothing"][i] = to56(0.5 * sm + 0.5 * sm2)

        # ---- pooled Spearman source (grid-level concatenation)
        flat_cecw.append(wok.ravel())
        flat_a1.append(a1_gridv[i].ravel())
        flat_concat.append(scF.ravel())

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"      [{cat} k{shot}] img {i + 1}/{n} "
                  f"meanN_d={np.mean(n_nei_stats['dino']):.0f}/c={np.mean(n_nei_stats['clip']):.0f} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    a1_ap = _pooled_ap(a1_map, masks56)
    a1_auroc = _pooled_auroc(a1_map, masks56)
    row = {
        "category": cat, "shot": shot, "n_images": n, "n_refs": s,
        "a1_ap": round(a1_ap, 6), "a1_auroc": round(a1_auroc, 6),
        "n_fallback": n_fallback,
        "mean_neighbors_dino": float(np.mean(n_nei_stats["dino"])),
        "mean_neighbors_clip": float(np.mean(n_nei_stats["clip"])),
        "methods": {},
        "spearman_vs_a1": {},
    }
    cw = np.concatenate(flat_cecw)
    ca = np.concatenate(flat_a1)
    cf = np.concatenate(flat_concat)
    for m in METHOD_ORDER:
        if m == "a1":
            continue
        ap = _pooled_ap(maps56[m], masks56)
        au = _pooled_auroc(maps56[m], masks56)
        row["methods"][m] = {
            "pixel_ap": (round(ap, 6) if ap == ap else None),
            "pixel_auroc": (round(au, 6) if au == au else None),
            "delta_ap_vs_a1": (round(ap - a1_ap, 6) if (ap == ap and a1_ap == a1_ap) else None),
        }
    if len(cw):
        row["spearman_vs_a1"]["cecw"] = round(_spearman(cw, ca), 6)
        row["spearman_vs_a1"]["anoco_concat"] = round(_spearman(cf, ca), 6)
    print(f"    [{cat} k{shot}] done in {time.time() - t0:.0f}s", flush=True)
    return row


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--category", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT /
                        "experiments/dynamic_fusion/innovation_v12_new_observables/cecw")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    base = ROOT / "outputs/dynamic_fusion/v3_direction_a"
    dino_dir = base / f"features_vitb14_s0_k{args.shot}/anomalydino_visual"
    clip_dir = base / f"features_s0_k{args.shot}/anomalyclip_text"

    cats = [args.category] if args.category else CATEGORIES
    rows = []
    for cat in cats:
        if not (dino_dir / f"{cat}.npz").is_file():
            print(f"[skip {cat}] cache missing", flush=True)
            continue
        print(f"[CECW k{args.shot}] {cat}", flush=True)
        rows.append(run_category(dino_dir, clip_dir, cat, args.shot))
        r = rows[-1]
        md = r["methods"]
        print("    " + "  ".join(f"{m}={md[m]['delta_ap_vs_a1']:+.4f}"
                                 for m in ["anoco_dino", "anoco_clip", "anoco_concat",
                                           "fixed_mean", "cecw", "ctrl_shuffled",
                                           "ctrl_noconflict", "ctrl_qq", "ctrl_smoothing"]),
              flush=True)

    def mean_delta(m: str) -> float:
        vals = [r["methods"][m]["delta_ap_vs_a1"] for r in rows
                if r["methods"][m]["delta_ap_vs_a1"] is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    base_deltas = {m: mean_delta(m) for m in BASELINES}
    cecw_delta = mean_delta("cecw")
    best_base = max(base_deltas.values())
    best_base_name = max(base_deltas, key=base_deltas.get)
    shuffled_delta = mean_delta("ctrl_shuffled")
    noconflict_delta = mean_delta("ctrl_noconflict")
    qq_delta = mean_delta("ctrl_qq")
    sm_delta = mean_delta("ctrl_smoothing")
    n_cat_pos = sum(1 for r in rows if r["methods"]["cecw"]["delta_ap_vs_a1"] is not None
                    and r["methods"]["cecw"]["delta_ap_vs_a1"] > 0.0)
    worst = min((r["methods"]["cecw"]["delta_ap_vs_a1"] for r in rows
                 if r["methods"]["cecw"]["delta_ap_vs_a1"] is not None), default=None)
    n_fb_total = sum(sum(r["n_fallback"].values()) for r in rows)
    spears = [r["spearman_vs_a1"]["cecw"] for r in rows
              if r["spearman_vs_a1"].get("cecw") is not None]
    mean_spear = round(float(np.mean(spears)), 6) if spears else None
    total_patches = sum(r["n_images"] * 1024 for r in rows)
    fb_frac = (n_fb_total / total_patches) if total_patches else 0.0

    g1 = (cecw_delta is not None and cecw_delta >= 0.006 and n_cat_pos >= 4
          and worst is not None and worst >= -0.010)
    g2 = (cecw_delta is not None and best_base is not None
          and cecw_delta - best_base >= 0.003)
    g3 = (cecw_delta is not None and shuffled_delta is not None
          and cecw_delta - shuffled_delta >= 0.003)
    g4_rescale = (mean_spear is not None and mean_spear > 0.98
                  and cecw_delta is not None and abs(cecw_delta) < 0.003)

    report = {
        "route": "V12-CECW", "seed": 0, "shot": args.shot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "constants": {"M_prefix": M_PREFIX, "Lambda_q": LAMBDA_Q, "rank": R_RANK,
                      "lambda_conflict": LAMBDA_C, "chunk": CHUNK},
        "per_category": rows,
        "mean_a1_ap": (round(float(np.mean([r["a1_ap"] for r in rows if r["a1_ap"] == r["a1_ap"]])), 6)
                       if rows else None),
        "mean_delta_ap_vs_a1": {m: mean_delta(m) for m in METHOD_ORDER if m != "a1"},
        "cecw_mean_delta": cecw_delta,
        "best_baseline": {"name": best_base_name, "mean_delta": best_base},
        "cecw_gain_over_best_baseline": (round(cecw_delta - best_base, 6)
                                         if cecw_delta is not None and best_base is not None else None),
        "shuffled_mean_delta": shuffled_delta,
        "noconflict_mean_delta": noconflict_delta,
        "qq_mean_delta": qq_delta,
        "smoothing_mean_delta": sm_delta,
        "n_categories_delta_positive": n_cat_pos,
        "worst_category_delta": worst,
        "cecw_spearman_vs_a1_mean": mean_spear,
        "fallback_fraction": round(fb_frac, 8),
        "gates": {
            "g1_headroom": g1,
            "g2_coupling_gain_ge_0.003": g2,
            "g3_shuffled_drop_ge_0.003": g3,
            "g4_spearman_rescale_veto": g4_rescale,
        },
        "decision": "PENDING",
    }
    (out_dir / f"R0_RESULT_k{args.shot}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- per_category.csv / per_seed_shot.csv
    with open(out_dir / "per_category.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "shot", "method", "pixel_ap", "delta_ap_vs_a1", "pixel_auroc"])
        for r in rows:
            for m in METHOD_ORDER:
                if m == "a1":
                    continue
                mm = r["methods"][m]
                w.writerow([r["category"], r["shot"], m, mm["pixel_ap"], mm["delta_ap_vs_a1"], mm["pixel_auroc"]])
    with open(out_dir / "per_seed_shot.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "shot", "method", "mean_delta_ap", "n_cats_positive", "worst_delta"])
        for m in METHOD_ORDER:
            if m == "a1":
                continue
            md = mean_delta(m)
            w.writerow([0, args.shot, m,
                        md if md is not None else "",
                        sum(1 for r in rows if r["methods"][m]["delta_ap_vs_a1"] is not None
                            and r["methods"][m]["delta_ap_vs_a1"] > 0.0),
                        min((r["methods"][m]["delta_ap_vs_a1"] for r in rows
                             if r["methods"][m]["delta_ap_vs_a1"] is not None), default="")])
    # ---- controls.json
    controls = {
        "shuffled_correspondence_drop": (round(cecw_delta - shuffled_delta, 6)
                                         if cecw_delta is not None and shuffled_delta is not None else None),
        "no_conflict_delta": noconflict_delta,
        "qq_edges_delta": qq_delta,
        "extra_smoothing_delta": sm_delta,
        "best_anoco_baseline": best_base_name,
        "fallback_fraction": round(fb_frac, 8),
        "mean_neighbors": {k: (round(float(np.mean([r[k] for r in rows])), 2) if rows else None)
                           for k in ("mean_neighbors_dino", "mean_neighbors_clip")},
    }
    (out_dir / "controls.json").write_text(
        json.dumps(controls, ensure_ascii=False, indent=1), encoding="utf-8")
    # ---- environment.json
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "n/a"
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "sklearn": __import__("sklearn").__version__,
        "torch": torch_v,
        "cv2": cv2.__version__,
    }
    (out_dir / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    # ---- input_manifest.json
    manifest = {}
    for cat in cats:
        for role, path in (("dino", dino_dir / f"{cat}.npz"), ("clip", clip_dir / f"{cat}.npz")):
            if path.is_file():
                manifest.setdefault(cat, {})[role] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
    (out_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nk{args.shot}: meanΔ cecw={cecw_delta} best_base={best_base_name} {best_base} "
          f"gain={report['cecw_gain_over_best_baseline']} shuffled={shuffled_delta} "
          f"noconf={noconflict_delta} qq={qq_delta} sm={sm_delta} n_pos={n_cat_pos}/6 "
          f"worst={worst} spear={mean_spear} fb={fb_frac:.2e} gates={report['gates']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
