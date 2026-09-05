"""N1 / JTD - joint-tail dependence (doc 27 s5; CPU, ~50 min budget).

Hypothesis: DINO & CLIP single-branch nearest-neighbour scores, taken jointly,
can be rare on normal data even when each alone looks ordinary. Implementation:
  u_D,u_C  = normal-only empirical CDFs of per-patch dino-only & clip-only
             final-layer L2 distances (clip aligned to 32 grid as in A1);
  8x8 2D histogram on (u_D,u_C) over NORMAL calibration patches (support
             leave-one-image-out for k2/k4) with fixed Dirichlet smoothing;
  R = max(0, log(p_D*p_C/p_joint)) gated on max(u_D,u_C)>0.9, capped at 5;
  candidate = rankF + 0.1*R, rankF = normal-only CDF of the frozen-A1 fused
             distance (support leave-one-image-out as well).
Controls: A1-rank-only; independent-tail u_D+u_C / max; shuffled D/C pairing
(fit joint on permuted pairs, marginals preserved); no-gate 2D rarity (diag).
R0 toy: 2D gaussian tail-dependence vs independence - pairing recoverable,
shuffle flattens; monotone scaling invariance on both branches.
R1 (macro over 6 cats): candidate - strongest independent tail/rank-only
>= +0.003 AND candidate - shuffled >= +0.003, per shot k2/k4 and combined.
Fitting is normal-only; GT only used at evaluation. k1 NOT auto-applied.
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
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from src.utils import dists2map  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
ML = ROOT / "outputs/dynamic_fusion/v12_early_fusion"
N_BINS = 8
ALPHA = 1.0            # fixed Dirichlet smoothing mass (uniform prior mass = ALPHA over 64 cells)
GATE = 0.9
CAP = 5.0
W = 0.1                # candidate coefficient on R


def _l2(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    shp = x.shape
    flat = x.reshape(-1, shp[-1])
    n = np.linalg.norm(flat, axis=1, keepdims=True)
    return (flat / np.maximum(n, 1e-8)).reshape(shp)


def _knn_dist(q_flat, bank):
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(bank.astype(np.float32))
    d2, _ = idx.search(q_flat.astype(np.float32), 1)
    return (d2[:, 0] / 2.0).astype(np.float64)          # 1-cos (rows L2-normalised)


def load_final(cat, shot):
    zd = np.load(ML / f"ml_dino_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    zc = np.load(ML / f"ml_clip_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    d_feat = np.asarray(zd["patch_features"])[:, 2]        # [N,32,32,768] L11
    d_ref = np.asarray(zd["ref_patch_features"])[2]        # [K,32,32,768]
    c_feat = np.asarray(zc["patch_features"])[:, 3]        # [N,37,37,768] L24
    c_ref = np.asarray(zc["ref_patch_features"])[3]        # [K,37,37,768]
    masks = np.asarray(zd["imgs_masks"], dtype=np.uint8)   # [N,448,448]
    c_feat32 = resize_patches(c_feat, (32, 32))
    c_ref32 = resize_patches(c_ref, (32, 32))
    del zd, zc
    return d_feat, d_ref, c_feat32, c_ref32, masks


def _patch_dist_matrix(feat, ref):
    """Per-patch 1-cos to nearest support row: returns [N,32,32] and ref rows."""
    n = feat.shape[0]
    bank = _l2(ref).reshape(-1, ref.shape[-1])
    out = np.empty((n, 32 * 32), dtype=np.float64)
    for i in range(n):
        q = _l2(feat[i]).reshape(-1, feat.shape[-1])
        out[i] = _knn_dist(q, bank)
    return out.reshape(n, 32, 32)


def _cal_patch_dists(d_ref, c_ref32, K, loo):
    """Calibration pools from support with leave-one-image-out: dino/clip/fused."""
    D0, C0, F0 = [], [], []
    for k in range(K):
        oth = [j for j in range(K) if j != k]
        db = _l2(d_ref[oth]).reshape(-1, d_ref.shape[-1]) if oth else None
        cb = _l2(c_ref32[oth]).reshape(-1, c_ref32.shape[-1]) if oth else None
        # fused support bank (per-image preserved) - build from other images
        dk = d_ref[k].reshape(-1, d_ref.shape[-1])
        ck = c_ref32[k].reshape(-1, c_ref32.shape[-1])
        # fused rows for calibration image k vs fused bank of others
        def _fuse_rows(a, b):
            na = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
            return np.concatenate([0.5 * na, 0.5 * nb], axis=-1).astype(np.float32)
        fq = _fuse_rows(dk, ck)
        fbank_rows = []
        for j in oth:
            dj = d_ref[j].reshape(-1, d_ref.shape[-1])
            cj = c_ref32[j].reshape(-1, c_ref32.shape[-1])
            fbank_rows.append(_fuse_rows(dj, cj))
        fbank = np.concatenate(fbank_rows, 0) if fbank_rows else None
        faiss.normalize_L2(fq)
        if fbank is not None:
            faiss.normalize_L2(fbank)
            F0.append(_knn_dist(fq, fbank))
        D0.append(_knn_dist(_l2(dk), db))
        C0.append(_knn_dist(_l2(ck), cb))
    D0 = np.concatenate(D0) if D0 else np.array([])
    C0 = np.concatenate(C0) if C0 else np.array([])
    F0 = np.concatenate(F0) if F0 else np.array([])
    return D0, C0, F0


def _cdf_ext(vals):
    """Empirical CDF over `vals` with boundary-slope linear tail extension.

    In-range: rank / (n+1) (average ties), u ALIGNED with sorted x. Out-of-range
    keeps STRICT monotonicity by continuing the last segment slope beyond max
    (and first below min), so test distances outside the narrow support-LOO
    calibration range do NOT tie at 1.0 (doc27 s5 out-of-range rule recorded).
    """
    v = np.sort(np.asarray(vals, dtype=np.float64))
    n = v.size
    u = rankdata(v, method="average") / (n + 1.0)   # ascending, aligned with v
    x0 = v[0]
    x1 = v[-1]
    u0 = float(u[0])
    u1 = float(u[-1])
    if n >= 2:
        s_top = (u[-1] - u[-2]) / max(v[-1] - v[-2], 1e-12)
        s_bot = (u[1] - u[0]) / max(v[1] - v[0], 1e-12)
    else:
        s_top = s_bot = 1.0 / (max(x1 - x0, 1e-12) * 2.0)

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.interp(x, v, u, left=u0, right=u1)
        over = x > x1
        under = x < x0
        out = np.where(over, u1 + s_top * (x - x1), out)
        out = np.where(under, u0 - s_bot * (x0 - x), out)
        return out

    return f


def _hist2d(uD, uC):
    h, e = np.histogramdd(np.stack([uD, uC], 1), bins=N_BINS, range=[[0, 1], [0, 1]])
    h = h + ALPHA / (N_BINS * N_BINS)
    h /= h.sum()
    pD = h.sum(1)
    pC = h.sum(0)
    return h, pD, pC


def _bin_idx(u):
    return np.clip(np.floor(u * N_BINS).astype(int), 0, N_BINS - 1)


def _surprise(uDq, uCq, h, pD, pC, gate=True):
    i = _bin_idx(uDq)
    j = _bin_idx(uCq)
    pj = h[i, j]
    pd = pD[i]
    pc = pC[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.maximum(0.0, np.log(np.maximum(pd * pc, 1e-12) / np.maximum(pj, 1e-12)))
    if gate:
        R = np.where(np.maximum(uDq, uCq) > GATE, R, 0.0)
    return np.minimum(R, CAP)


def run_category(cat, shot):
    d_feat, d_ref, c_feat32, c_ref32, masks = load_final(cat, shot)
    K = d_ref.shape[0]
    n = d_feat.shape[0]
    if K < 2:
        return {"category": cat, "shot": shot, "skip": "K<2"}
    # calibration pools (support LOO)
    D0, C0, F0 = _cal_patch_dists(d_ref, c_ref32, K, loo=True)
    if len(D0) < 200 or len(F0) < 200:
        return {"category": cat, "shot": shot, "skip": "calibration-too-small",
                "n_cal": int(len(D0))}
    # quantile transforms on calibration (tail-extended, strict-monotone)
    uD_f = _cdf_ext(D0)
    uC_f = _cdf_ext(C0)
    rankF_f = _cdf_ext(F0)
    uD0 = uD_f(D0)
    uC0 = uC_f(C0)
    h, pD, pC = _hist2d(uD0, uC0)
    # R0 block stability: half-split calibration fits vs full fit (corner R + cell diff)
    n2 = len(uD0) // 2
    hA, pDA, pCA = _hist2d(uD0[:n2], uC0[:n2])
    hB, pDB, pCB = _hist2d(uD0[n2:], uC0[n2:])
    def _corner_r(hh, ppD, ppC):
        return float(_surprise(1.0 - 1e-9, 1.0 - 1e-9, hh, ppD, ppC, gate=False))
    block = {"corner_R_full": _corner_r(h, pD, pC), "corner_R_halfA": _corner_r(hA, pDA, pCA),
             "corner_R_halfB": _corner_r(hB, pDB, pCB),
             "max_cell_abs_diff_A": float(np.abs(hA - h).max()),
             "max_cell_abs_diff_B": float(np.abs(hB - h).max())}
    # shuffled joint (marginals preserved)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(uC0))
    hS, pDS, pCS = _hist2d(uD0, uC0[perm])
    # query patch scores
    dQ = _patch_dist_matrix(d_feat, d_ref)          # [N,32,32]
    cQ = _patch_dist_matrix(c_feat32, c_ref32)
    nq = dQ.shape[0]
    uD = uD_f(dQ.ravel())
    uC = uC_f(cQ.ravel())
    R = _surprise(uD, uC, h, pD, pC, gate=True).reshape(nq, 32, 32)
    Rng = _surprise(uD, uC, h, pD, pC, gate=False).reshape(nq, 32, 32)
    Rs = _surprise(uD, uC, hS, pDS, pCS, gate=True).reshape(nq, 32, 32)
    # fused A1 distance for query
    def _fuse_rows(a, b):
        na = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.concatenate([0.5 * na, 0.5 * nb], axis=-1).astype(np.float32)
    fbank_rows = [_fuse_rows(d_ref[j].reshape(-1, 768), c_ref32[j].reshape(-1, 768))
                  for j in range(K)]
    fbank = np.concatenate(fbank_rows, 0)
    faiss.normalize_L2(fbank)
    FQ = np.empty((nq, 32 * 32), dtype=np.float64)
    for i in range(nq):
        fq = _fuse_rows(d_feat[i].reshape(-1, 768), c_feat32[i].reshape(-1, 768))
        faiss.normalize_L2(fq)
        FQ[i] = _knn_dist(fq, fbank)
    FQ = FQ.reshape(nq, 32, 32)
    rankF = rankF_f(FQ.ravel()).reshape(nq, 32, 32)
    u_sum = (uD.reshape(nq, 32, 32) + uC.reshape(nq, 32, 32)) / 2.0
    u_max = np.maximum(uD.reshape(nq, 32, 32), uC.reshape(nq, 32, 32))

    scores = {"a1_raw": FQ, "a1_rank": rankF, "cand": rankF + W * R,
              "u_sum": u_sum, "u_max": u_max, "shuf": rankF + W * Rs,
              "no_gate": rankF + W * Rng}
    m56 = (masks[:, ::8, ::8] > 0.5).astype(np.uint8)
    aps = {}
    for name, s in scores.items():
        maps = np.stack([dists2map(s[i], (448, 448))[::8, ::8] for i in range(nq)])
        y = m56.ravel() > 0.5
        aps[name] = float(average_precision_score(y.astype(int), maps.ravel()))
    fq_flat = FQ.ravel()
    from scipy.stats import spearmanr  # noqa: PLC0415
    rho = float(spearmanr(fq_flat, rankF.ravel()).statistic)
    # raw-32-grid pooled AP (no dists2map) for raw vs rank to isolate transform vs map path
    m32 = (masks[:, ::14, ::14] > 0.5).astype(np.uint8)
    y32 = m32.ravel() > 0.5
    ap32_raw = float(average_precision_score(y32.astype(int), FQ.ravel()))
    ap32_rank = float(average_precision_score(y32.astype(int), rankF.ravel()))
    diag = {"F0": {"min": float(D0.min()), "p50": float(np.percentile(D0, 50)),
                   "p90": float(np.percentile(D0, 90)), "max": float(D0.max())},
            "FQ": {"min": float(fq_flat.min()), "p50": float(np.percentile(fq_flat, 50)),
                   "p90": float(np.percentile(fq_flat, 90)), "max": float(fq_flat.max())},
            "frac_FQ_gt_maxF0": float((fq_flat > D0.max()).mean()),
            "rankF_p": {"p1": float(np.percentile(rankF, 1)), "p50": float(np.percentile(rankF, 50)),
                        "p90": float(np.percentile(rankF, 90)), "p99": float(np.percentile(rankF, 99))},
            "spearman_FQ_rankF": rho, "ap32_raw": ap32_raw, "ap32_rank": ap32_rank}
    return {"category": cat, "shot": shot, "aps": aps, "n": n, "K": K,
            "n_cal": int(len(D0)), "diag": diag, "block": block}


def toy_test():
    """R0 toy (doc 27 s5): pairing/shuffle identifiability + monotone-scale invariance."""
    rng = np.random.default_rng(1)
    n = 20000
    z = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n)
    uD, uC = _cdf_ext(z[:, 0])(z[:, 0]), _cdf_ext(z[:, 1])(z[:, 1])
    zi = rng.normal(size=(n, 2))
    wD, wC = _cdf_ext(zi[:, 0])(zi[:, 0]), _cdf_ext(zi[:, 1])(zi[:, 1])

    def tail_gap(a, b, seed=3):
        h, pD, pC = _hist2d(a, b)
        perm = rng.permutation(len(b))
        hS, pDS, pCS = _hist2d(a, b[perm])
        Rt = _surprise(a, b, h, pD, pC, True)
        Rs = _surprise(a, b, hS, pDS, pCS, True)
        sel = np.maximum(a, b) > GATE
        return float(np.abs(Rt - Rs)[sel].mean()) if sel.any() else 0.0

    res = {"tail_mean_abs_R_minus_Rshuf": {"dep": tail_gap(uD, uC),
                                           "ind": tail_gap(wD, wC)}}
    # monotone-scale invariance on the QUANTILE pair: per-branch monotone maps of the
    # scores leave (uD,uC) unchanged -> R identical
    d = rng.random(n)
    c = rng.random(n)
    dd, cc = _cdf_ext(d)(d), _cdf_ext(c)(c)
    h1, pD1, pC1 = _hist2d(dd, cc)
    t1 = _surprise(dd, cc, h1, pD1, pC1, False)
    d2, c2 = _cdf_ext(np.cbrt(d))(np.cbrt(d)), _cdf_ext(np.sqrt(c))(np.sqrt(c))
    h2, pD2, pC2 = _hist2d(d2, c2)
    t3 = _surprise(d2, c2, h2, pD2, pC2, False)
    res["monotone_scale_invariance_mean_abs_Rdiff"] = float(np.abs(t1 - t3).mean())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--cats", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v13_overnight_20260904/N1_jtd"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.toy:
        out = toy_test()
        (out_root / "TOY.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("TOY", json.dumps(out), flush=True)
        return 0
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    rows = []
    for shot in args.shots:
        for cat in cats:
            r = run_category(cat, shot)
            rows.append(r)
            print(" ", cat, shot, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                              for k, v in r.items()}), flush=True)
    payload = {"rows": rows, "note": "JTD k2/k4 normal-only calibration (support LOO)"}
    (out_root / "RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
