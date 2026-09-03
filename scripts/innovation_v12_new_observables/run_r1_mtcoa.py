"""V12 MTCOA - Macro-Tail Calibrated Oracle Audit (doc 22 s2.3).

One-shot corrected re-audit of the v11 RSR oracle over the same expert pool
{E0 A1, E1 text-evidence, E2 LLSE, E3 CSS}, MPDD development seed0 x shot{1,2,4}.

Fixes vs v11:
  A. selection and output both on a per-expert calibrated [0,1] map fit from
     NORMAL-SUPPORT leave-one-out scores (E0/E2: LOO over references; E3: refs'
     intra-image css; E1: per-image robust01 - documented approximation);
  B. per GT component choose argmin_e L_e(c) = (1 - AUROC(c vs outside)) +
     1.0 * mean(outside calibrated score);
  C. report pixel-micro, category-macro, component-macro and size strata.

Evaluator-only capacity audit; GT used only offline. No router is trained.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r1_mtcoa.py --shot 1
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
from sklearn.metrics import average_precision_score, roc_auc_score

from industrial_ad.innovation_v10_portfolio import common
from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, load_features
from industrial_ad.innovation_v8_tcrr_probe.regions import robust01
from src.utils import dists2map

K_NEIGH = 8
REG = 1e-3
RADIUS = 2
CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
EXPERT_NAMES = ["E0_A1", "E1_TEXT", "E2_LLSE", "E3_CSS"]
BINS = [(1, 4), (5, 64), (65, 10 ** 6)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- expert grids

def a1_grid(feat_img: np.ndarray, bank: np.ndarray) -> np.ndarray:
    import faiss
    hh, ww, d = feat_img.shape
    q = feat_img.reshape(-1, d).astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(bank.astype(np.float32))
    dists, _ = index.search(q, k=1)
    return (dists[:, 0] / 2.0).reshape(hh, ww).astype(np.float32)


def _residual_from_nbr(q, nbr, k):
    nd = nbr.astype(np.float64)
    G = nd @ nd.transpose(0, 2, 1) + REG * np.eye(k, dtype=np.float64)
    rhs = np.matmul(nd, q.astype(np.float64)[..., None])[..., 0]
    coef = np.linalg.solve(G, rhs)
    resid2 = np.maximum(1.0 - (coef * rhs).sum(axis=-1), 0.0)
    return np.sqrt(resid2).astype(np.float32)


def llse_grid(feat_img: np.ndarray, bank: np.ndarray) -> np.ndarray:
    import faiss
    hh, ww, d = feat_img.shape
    q = feat_img.reshape(-1, d).astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(bank.astype(np.float32))
    _, idx = index.search(q, k=K_NEIGH)
    nbr = bank[idx.astype(np.int64)].astype(np.float32)
    resid = _residual_from_nbr(q, nbr, K_NEIGH)
    return resid.reshape(hh, ww)


def css_grid(feat_img: np.ndarray) -> np.ndarray:
    hh, ww, d = feat_img.shape
    pad = np.pad(feat_img, ((RADIUS, RADIUS), (RADIUS, RADIUS), (0, 0)), mode="edge")
    acc = np.zeros((hh, ww), dtype=np.float64)
    n_off = 0
    for dr in range(-RADIUS, RADIUS + 1):
        for dc in range(-RADIUS, RADIUS + 1):
            if dr == 0 and dc == 0:
                continue
            sl = pad[RADIUS + dr:RADIUS + dr + hh, RADIUS + dc:RADIUS + dc + ww]
            acc += 1.0 - np.einsum("hwd,hwd->hw", feat_img.astype(np.float64), sl.astype(np.float64))
            n_off += 1
    return (acc / n_off).astype(np.float32)


def to56(grid: np.ndarray) -> np.ndarray:
    return dists2map(grid, common.MAP_SIZE)[::common.STRIDE, ::common.STRIDE].astype(np.float32)


def align_perm(src_ids: np.ndarray, dst_ids: np.ndarray) -> np.ndarray:
    pos = {str(v): i for i, v in enumerate(src_ids)}
    return np.asarray([pos[str(v)] for v in dst_ids], dtype=np.int64)


def _calibrator(normal_vals: np.ndarray):
    """Strictly monotone empirical-CDF calibrator over normal-support scores.

    Linear interpolation on unique order statistics + linear extrapolation beyond
    the fitted range, so NO top saturation: distinct raw scores keep distinct
    calibrated values (identity AP preserved up to float32 rounding).
    """
    ns = np.sort(np.asarray(normal_vals, dtype=np.float64).ravel())
    xs, cnt = np.unique(ns, return_counts=True)
    if xs.size < 2:
        return lambda x: np.clip(x, 0.0, 1.0).astype(np.float32)
    F = np.cumsum(cnt).astype(np.float64) / float(cnt.sum())
    sl = min((F[1] - F[0]) / (xs[1] - xs[0]), 10.0)
    sr = min((F[-1] - F[-2]) / (xs[-1] - xs[-2]), 10.0)

    def cal(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.interp(x, xs, F)
        up = x > xs[-1]
        lo = x < xs[0]
        if up.any():
            y[up] = F[-1] + (x[up] - xs[-1]) * sr
        if lo.any():
            y[lo] = np.maximum(0.0, F[0] + (x[lo] - xs[0]) * sl)
        return y.astype(np.float32)

    return cal


def _pooled_ap(maps: np.ndarray, masks56: np.ndarray) -> float:
    y = (masks56.ravel() > 0.5).astype(np.int32)
    s = maps.ravel().astype(np.float64)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


# ---------------------------------------------------------------- category

def run_category(dino_cache: Path, clip_cache: Path, text_dir: Path, cat: str,
                 shot: int) -> dict:
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    feat, ref, sample_ids, masks, grid = build_fused_blocks(dino, clip, dino_weight=0.5)
    n = feat.shape[0]
    s = ref.shape[0]                      # number of reference images == shot
    masks56 = (masks[:, ::common.STRIDE, ::common.STRIDE] > 0.5).astype(np.uint8)

    # E1 text maps (aligned)
    with np.load(text_dir / f"{cat}.npz", allow_pickle=False) as z:
        text_maps = np.asarray(z["anomaly_maps"], dtype=np.float32)
        text_ids = np.asarray(z["sample_ids"])
    perm = align_perm(text_ids, sample_ids)
    text56_raw = np.stack([
        cv2.resize(m, (448, 448), interpolation=cv2.INTER_LINEAR)[::common.STRIDE, ::common.STRIDE]
        for m in text_maps[perm]
    ]).astype(np.float32)
    text56 = np.stack([robust01(m) for m in text56_raw]).astype(np.float32)

    # normal-support calibration values (per expert), leave-self-out over the
    # FULL reference bank so the pool is non-degenerate even at shot=1
    ref_flat = ref.reshape(-1, feat.shape[-1]).astype(np.float32)  # [S*P, D]
    s_p = ref_flat.shape[0]
    import faiss
    _idx = faiss.IndexFlatL2(feat.shape[-1])
    _idx.add(ref_flat)
    _, nn2 = _idx.search(ref_flat, k=2)               # self is always nearest (dist 0)
    self_lo = nn2[:, 1]                                # leave-self-out nearest id
    e0_loo = np.zeros((s, 56, 56), dtype=np.float32)
    e2_loo = np.zeros((s, 56, 56), dtype=np.float32)
    for r in range(s):
        rows = np.arange(r * 32 * 32, (r + 1) * 32 * 32)
        d2 = (ref_flat[rows] - ref_flat[self_lo[rows]]) ** 2
        e0_loo[r] = to56((0.5 * d2.sum(axis=1)).reshape(32, 32))
    _, nn9 = _idx.search(ref_flat, k=9)
    for r in range(s):
        rows = np.arange(r * 32 * 32, (r + 1) * 32 * 32)
        nbr = ref_flat[nn9[rows, 1:9]]                # exclude self (rank 0)
        res = _residual_from_nbr(ref_flat[rows].astype(np.float32), nbr, 8)
        e2_loo[r] = to56(res.reshape(32, 32))
    e3_css_ref = [to56(css_grid(ref[r])).ravel() for r in range(s)]

    # per-test-image expert maps (raw at 56)
    bank = ref.reshape(-1, feat.shape[-1]).astype(np.float32)
    raw = {e: np.empty((n, 56, 56), dtype=np.float32) for e in EXPERT_NAMES}
    raw["E1_TEXT"] = text56
    for i in range(n):
        raw["E0_A1"][i] = to56(a1_grid(feat[i], bank))
        raw["E2_LLSE"][i] = to56(llse_grid(feat[i], bank))
        raw["E3_CSS"][i] = to56(css_grid(feat[i]))

    # common normal calibration pool per expert = support leave-self-out scores +
    # the class's NORMAL test/good image scores (no bad, no mask, no label). A
    # support-only CDF saturates on classes whose test-normal FPs exceed the
    # reference LOO spread (documented amendment).
    good_idx = [i for i, sid in enumerate(sample_ids) if "/good/" in str(sid)]
    pool = {
        "E0_A1": np.concatenate([e0_loo[r].ravel() for r in range(s)]
                                + [raw["E0_A1"][g].ravel() for g in good_idx]),
        "E2_LLSE": np.concatenate([e2_loo[r].ravel() for r in range(s)]
                                  + [raw["E2_LLSE"][g].ravel() for g in good_idx]),
        "E3_CSS": np.concatenate(e3_css_ref + [raw["E3_CSS"][g].ravel() for g in good_idx]),
        "E1_TEXT": np.concatenate([text56[g].ravel() for g in good_idx])
        if good_idx else np.asarray([], dtype=np.float32),
    }
    calib = {e: _calibrator(pool[e]) for e in EXPERT_NAMES}
    # calibrated maps (same scale for selection and output)
    cal = {e: np.stack([calib[e](m) for m in raw[e]]) for e in EXPERT_NAMES}

    a1_raw_ap = _pooled_ap(raw["E0_A1"], masks56)
    a1_cal_ap = _pooled_ap(cal["E0_A1"], masks56)
    single_ap = {e: _pooled_ap(cal[e], masks56) for e in EXPERT_NAMES}

    # ---- component-level oracle selection
    comps_all = []          # (image idx, size_bin, chosen expert)
    per_img_oracle = {}

    def build_oracle(exclude: set[str]) -> tuple[np.ndarray, list]:
        names = [e for e in EXPERT_NAMES if e not in exclude]
        oracle = cal["E0_A1"].copy()
        chosen = []
        for i in range(n):
            lbl = masks56[i]
            if lbl.sum() == 0:
                continue
            num, comp = cv2.connectedComponents(lbl, connectivity=8)
            outside = lbl == 0
            for cid in range(1, num):
                cm = comp == cid
                area = int(cm.sum())
                if outside.sum() < 2:
                    best = "E0_A1"
                else:
                    losses = {}
                    for e in names:
                        pc = cal[e][i][cm]
                        po = cal[e][i][outside]
                        if pc.size == 0 or po.size == 0:
                            losses[e] = float("inf")
                            continue
                        eps = 1e-7
                        # doc 21 L_e(r)=BCE(M_r,S_e)+a[1-AP]+b*FP; M_r is the REGION
                        # indicator, so the BCE hit term is over region pixels only
                        # (all-ones labels); background is penalised by the FP term.
                        bce = -float(np.mean(np.log(np.clip(pc, eps, 1.0))))
                        scores = np.r_[pc, po]
                        y = np.r_[np.ones(len(pc)), np.zeros(len(po))]
                        try:
                            ap = float(average_precision_score(y, scores))
                        except ValueError:
                            ap = 0.5
                        losses[e] = bce + (1.0 - ap) + float(po.mean())
                    best = min(losses, key=lambda e: losses[e]) if losses else "E0_A1"
                    if not np.isfinite(losses.get(best, float("inf"))):
                        best = "E0_A1"
                oracle[i][cm] = cal[best][i][cm]
                chosen.append((i, area, best))
        return oracle, chosen

    oracle_full, chosen_full = build_oracle(set())
    oracle_ap = _pooled_ap(oracle_full, masks56)

    # size-stratified AP (positives = pixels in components of that bin)
    strata_ap = {}
    for lo, hi in BINS:
        posm = np.zeros_like(masks56, dtype=bool)
        for i in range(n):
            if masks56[i].sum() == 0:
                continue
            num, comp = cv2.connectedComponents(masks56[i], connectivity=8)
            for cid in range(1, num):
                area = int((comp == cid).sum())
                if lo <= area <= hi:
                    posm[i] |= (comp == cid)
        if posm.sum() == 0:
            strata_ap[f"{lo}-{hi}"] = {"a1_ap": None, "oracle_ap": None, "delta": None}
            continue
        y = posm.ravel()
        negm = ~y
        s_a1 = cal["E0_A1"].ravel()[y]
        s_or = oracle_full.ravel()[y]
        neg_s_a1 = cal["E0_A1"].ravel()[negm]
        neg_s_or = oracle_full.ravel()[negm]
        def ap(p, neg):
            if p.size == 0 or neg.size == 0:
                return None
            return float(average_precision_score(np.r_[np.ones(len(p)), np.zeros(len(neg))],
                                                 np.r_[p, neg]))
        a1v = ap(s_a1, neg_s_a1)
        orv = ap(s_or, neg_s_or)
        strata_ap[f"{lo}-{hi}"] = {"a1_ap": a1v, "oracle_ap": orv,
                                   "delta": (orv - a1v) if (a1v is not None and orv is not None) else None}

    # component shares
    share = {e: 0 for e in EXPERT_NAMES}
    for _, area, e in chosen_full:
        share[e] += 1
    total = max(1, len(chosen_full))
    share = {e: (v / total if v else 0.0) for e, v in share.items()}

    # LOO for core experts (non-A1 with >=15% component share)
    loo_drop = {}
    core = [e for e in ("E1_TEXT", "E2_LLSE", "E3_CSS") if share[e] >= 0.15]
    for e in core:
        o_loo, _ = build_oracle({e})
        loo_drop[f"without_{e}"] = round((oracle_ap - a1_cal_ap) -
                                         (_pooled_ap(o_loo, masks56) - a1_cal_ap), 6)

    return {
        "category": cat, "n_images": n, "shot": shot, "n_refs": s,
        "a1_raw_ap": round(a1_raw_ap, 6), "a1_cal_ap": round(a1_cal_ap, 6),
        "identity_diff": round(abs(a1_cal_ap - a1_raw_ap), 8),
        "single_cal_ap": {e: (round(v, 6) if v == v else None) for e, v in single_ap.items()},
        "oracle_cal_ap": round(oracle_ap, 6),
        "delta_oracle": round(oracle_ap - a1_cal_ap, 6),
        "component_share": {e: round(v, 6) for e, v in share.items()},
        "n_components": len(chosen_full),
        "core_experts_loo_drop": loo_drop,
        "strata_ap": {k: {kk: (round(vv, 6) if vv is not None and vv == vv else None)
                          for kk, vv in v.items()} for k, v in strata_ap.items()},
    }


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--category", default=None)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v12_new_observables/mtcoa")
    parser.add_argument("--text-dir", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/innovation_v8_tcrr_probe/text_maps")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    base = ROOT / "outputs/dynamic_fusion/v3_direction_a"
    dino_cache = base / f"features_vitb14_s0_k{args.shot}/anomalydino_visual"
    clip_cache = base / f"features_s0_k{args.shot}/anomalyclip_text"

    cats = [args.category] if args.category else CATEGORIES
    rows = []
    t0 = time.time()
    for cat in cats:
        if not (dino_cache / f"{cat}.npz").is_file():
            print(f"[skip {cat}] cache missing", flush=True)
            continue
        print(f"[MTCOA k{args.shot}] {cat}", flush=True)
        rows.append(run_category(dino_cache, clip_cache, args.text_dir, cat, args.shot))
        r = rows[-1]
        print(f"    A1 raw={r['a1_raw_ap']:.4f} cal={r['a1_cal_ap']:.4f} "
              f"oracle={r['oracle_cal_ap']:.4f} d={r['delta_oracle']:+.4f} "
              f"shares " + " ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in r["component_share"].items())
              + f" ncomp={r['n_components']}", flush=True)

    def meanf(key):
        vals = [r[key] for r in rows if r[key] == r[key]]
        return round(float(np.mean(vals)), 6) if vals else None

    mean_delta = meanf("delta_oracle")
    mean_identity = max(abs(r["identity_diff"]) for r in rows)
    n_cls_ge_010 = sum(1 for r in rows if r["delta_oracle"] >= 0.010)
    agg_share = {e: sum(r["component_share"][e] * r["n_components"] for r in rows)
                 for e in EXPERT_NAMES}
    tot = sum(agg_share.values()) or 1
    agg_share = {e: v / tot for e, v in agg_share.items()}
    pos_head = sum(max(r["delta_oracle"], 0.0) for r in rows)
    top_cls = max(max(r["delta_oracle"], 0.0) / pos_head for r in rows) if pos_head > 0 else 1.0

    # g2 alt: second best non-A1 expert small-stratum delta >= +0.020
    small_deltas = {}
    for e, key in (("E1_TEXT", "E1"), ("E2_LLSE", "E2"), ("E3_CSS", "E3")):
        vals = [r["strata_ap"]["1-4"]["delta"] for r in rows
                if r["strata_ap"]["1-4"]["delta"] is not None]
        small_deltas[key] = round(float(np.mean(vals)), 6) if vals else None
    non_a1_experts_ge15 = [e for e in ("E1_TEXT", "E2_LLSE", "E3_CSS") if agg_share[e] >= 0.15]
    n_nonA1_ge15 = len(non_a1_experts_ge15)
    second_best_small = sorted([v for v in small_deltas.values() if v is not None], reverse=True)[:2]
    alt_small = second_best_small[1] >= 0.020 if len(second_best_small) >= 2 else False

    # LOO drop aggregated over classes for each core expert
    loo_agg = {}
    for e in ("E1_TEXT", "E2_LLSE", "E3_CSS"):
        vals = [r["core_experts_loo_drop"][f"without_{e}"] for r in rows
                if f"without_{e}" in r["core_experts_loo_drop"]]
        loo_agg[f"without_{e}"] = round(float(np.mean(vals)), 6) if vals else None

    report = {
        "route": "V12-MTCOA", "seed": 0, "shot": args.shot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "mean_a1_cal_ap": meanf("a1_cal_ap"),
        "mean_delta_oracle": mean_delta,
        "identity_max_diff": mean_identity,
        "n_classes_delta_ge_010": n_cls_ge_010,
        "agg_component_share": {e: round(v, 6) for e, v in agg_share.items()},
        "n_nonA1_component_share_ge_15pct": n_nonA1_ge15,
        "small_stratum_delta_by_expert": small_deltas,
        "second_expert_small_stratum_ge_020": alt_small,
        "loo_drop_core_experts": loo_agg,
        "top_class_share_of_positive_headroom": round(top_cls, 6),
        "gates": {
            "g0_identity_le_1e-4": mean_identity <= 1e-4,
            "g1_headroom_ge_020_and_4of6_ge_010": (mean_delta is not None and mean_delta >= 0.020
                                                   and n_cls_ge_010 >= 4),
            "g2_contribution": n_nonA1_ge15 >= 2 or alt_small,
            "g3_loo": all(v is not None and v >= 0.003 for v in loo_agg.values())
                      if loo_agg else False,
            "g4_dominance_le_50pct": top_cls <= 0.50,
        },
        "decision": "PENDING",
    }
    (out_dir / f"R0_RESULT_k{args.shot}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nk{args.shot}: identity_max={mean_identity:.2e} meanΔ={mean_delta} "
          f"n_cls≥0.01={n_cls_ge_010}/6 agg_shares={ {k: round(v,3) for k,v in agg_share.items()} } "
          f"nonA1≥15%: {n_nonA1_ge15} small2nd={second_best_small} loo={loo_agg} "
          f"top={top_cls:.2f} gates={report['gates']} elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
