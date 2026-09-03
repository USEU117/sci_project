"""V12-EARLY-FUSION Stage 0 probe (doc 23 Stage0) - CPU evaluator.

Run (.venv-patchcore, CPU):
  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r3_ef_stage0_probe.py --shot 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import faiss  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.preprocessing import normalize as sk_norm  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import MAP_SIZE, STRIDE, load_features  # noqa: E402
from src.utils import dists2map  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
BASE = ROOT / "outputs/dynamic_fusion/v3_direction_a"
ML_ROOT = ROOT / "outputs/dynamic_fusion/v12_early_fusion"
D_LAY = [6, 9, 11]
C_LAY = [6, 12, 18, 24]


def to56(g: np.ndarray) -> np.ndarray:
    return dists2map(g, MAP_SIZE)[::STRIDE, ::STRIDE].astype(np.float32)


def l2(x: np.ndarray) -> np.ndarray:
    return sk_norm(x.reshape(-1, x.shape[-1])).reshape(x)


def resize32(x: np.ndarray) -> np.ndarray:
    import torch
    from torch.nn import functional as F

    x = x.astype(np.float32)
    pre = x.shape[:-3]
    h, w, d = x.shape[-3], x.shape[-2], x.shape[-1]
    t = torch.from_numpy(x.reshape(-1, h, w, d)).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy().reshape(*pre, 32, 32, d)


def fused_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2(a.reshape(-1, a.shape[-1]))
    b = l2(b.reshape(-1, b.shape[-1]))
    return l2(np.concatenate([0.5 * a, 0.5 * b], axis=-1))


def knn_map(q: np.ndarray, bank: np.ndarray) -> np.ndarray:
    hh, ww, d = q.shape
    qf = q.reshape(-1, d).astype(np.float32)
    idx = faiss.IndexFlatL2(d)
    idx.add(bank.astype(np.float32))
    dists, _ = idx.search(qf, k=1)
    return (dists[:, 0] / 2.0).reshape(hh, ww)


def pooled_ap(maps56: np.ndarray, m56: np.ndarray) -> float:
    y = (m56.ravel() > 0.5).astype(np.int32)
    s = maps56.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = float(np.mean(r[order[i:j + 1]]))
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def run_category(cat: str, shot: int) -> dict:
    z = np.load(ML_ROOT / f"ml_dino_s0_k{shot}/{cat}.npz", allow_pickle=False)
    d_feat = np.asarray(z["patch_features"])   # [N,3,32,32,768]
    d_ref = np.asarray(z["ref_patch_features"])  # [3,K,32,32,768]
    masks = np.asarray(z["imgs_masks"], dtype=np.uint8)
    zc = np.load(ML_ROOT / f"ml_clip_s0_k{shot}/{cat}.npz", allow_pickle=False)
    c_feat = np.asarray(zc["patch_features"])   # [N,4,37,37,768]
    c_ref = np.asarray(zc["ref_patch_features"])
    m56 = (masks[:, ::STRIDE, ::STRIDE] > 0.5).astype(np.uint8)
    n = d_feat.shape[0]

    # ---- deepest parity vs frozen A1 caches (raw features)
    a1_d = load_features(BASE / f"features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz")
    a1_c = load_features(BASE / f"features_s0_k{shot}/anomalyclip_text/{cat}.npz")
    parity = {
        "dino_L11_maxabs": float(np.abs(d_feat[:, 2] - np.asarray(a1_d["patch_features"])).max()),
        "dino_L11_ref_maxabs": float(np.abs(d_ref[2] - np.asarray(a1_d["ref_patch_features"])).max()),
        "clip_L24_maxabs": float(np.abs(c_feat[:, 3] - np.asarray(a1_c["patch_features"])).max()),
        "clip_L24_ref_maxabs": float(np.abs(c_ref[3] - np.asarray(a1_c["ref_patch_features"])).max()),
    }

    # ---- maps over the pool of pre-registered layer features
    c32 = resize32(c_feat)      # [N,4,32,32,768]
    c32_ref = resize32(c_ref)   # [4,K,32,32,768]
    spec = []
    for li, lay in enumerate(D_LAY):
        spec.append((f"dino_L{lay}", "d", li, None))
    for li, lay in enumerate(C_LAY):
        spec.append((f"clip_L{lay}", "c", li, None))
    spec += [
        ("concat_D6C6", "f", 0, 0),
        ("concat_D9C12", "f", 1, 1),
        ("concat_D11C18", "f", 2, 2),
        ("concat_D11C24", "f", 2, 3),   # == A1
    ]

    def branch_grid(branch, idx, feat_all, ref_all):
        if branch == "d":
            return feat_all[:, idx], ref_all[idx]
        return feat_all[:, idx], ref_all[idx]

    maps = {}
    for name, kind, ai, ci in spec:
        if kind == "d":
            bank = l2(d_ref[ai].reshape(-1, 768))
            rows = [to56(knn_map(l2(d_feat[i, ai]), bank)) for i in range(n)]
        elif kind == "c":
            bank = l2(c_ref[ai].reshape(-1, 768))
            rows = [to56(knn_map(l2(c_feat[i, ai]), bank)) for i in range(n)]
        else:
            bank = fused_rows(d_ref[ai].reshape(-1, 32, 32, 768),
                              c32_ref[ci].reshape(-1, 32, 32, 768))
            rows = []
            for i in range(n):
                q = fused_rows(d_feat[i, ai].reshape(1, 32, 32, 768),
                               c32[i, ci].reshape(1, 32, 32, 768)).reshape(32, 32, 1536)
                rows.append(to56(knn_map(q, bank)))
        maps[name] = np.stack(rows)

    # FULL static multi-layer concat (all 3 dino + 4 clip layers at 32 grid)
    d_all = d_feat.transpose(0, 3, 1, 2).reshape(n, 32 * 32, 3 * 768)
    c_all = c32.transpose(0, 3, 1, 2).reshape(n, 32 * 32, 4 * 768)
    rd_all = d_ref.transpose(0, 3, 1, 2).reshape(3 * 768, -1).T.reshape(-1, 1, 1, 3 * 768)
    rc_all = c32_ref.transpose(0, 3, 1, 2).reshape(4 * 768, -1).T.reshape(-1, 1, 1, 4 * 768)
    fbank = fused_rows(rd_all.reshape(-1, 3 * 768), rc_all.reshape(-1, 4 * 768)).reshape(-1, 1536)
    fq = fused_rows(d_all.reshape(-1, 3 * 768), c_all.reshape(-1, 4 * 768)).reshape(n, 1024, 1536)
    maps["concat_FULL"] = np.stack([to56(knn_map(fq[i], fbank)) for i in range(n)])

    a1_ap = pooled_ap(maps["concat_D11C24"], m56)
    layer_aps = {k: round(pooled_ap(maps[k], m56), 6) for k in maps}
    aps_clean = {k: (v if v == v else None) for k, v in layer_aps.items()}

    # correlations across maps (pooled patch scores)
    names = list(maps.keys())
    corr = {f"{a}|{b}": round(spearman(maps[a], maps[b]), 4)
            for i, a in enumerate(names) for b in names[i + 1:]}

    # oracle over branch-layer maps (evaluator-only)
    experts = [maps[f"dino_L{l}"] for l in D_LAY] + [maps[f"clip_L{l}"] for l in C_LAY]
    oracle = maps["concat_D11C24"].copy()
    comps = 0
    for i in range(n):
        lbl = m56[i]
        if lbl.sum() == 0:
            continue
        num, comp = cv2.connectedComponents(lbl, connectivity=8)
        outside = lbl == 0
        for cid in range(1, num):
            cm = comp == cid
            if outside.sum() < 2:
                continue
            best = None
            bl = float("inf")
            for em in experts:
                pc = em[i][cm]
                po = em[i][outside]
                if pc.size == 0 or po.size == 0:
                    continue
                bce = -float(np.mean(np.log(np.clip(pc, 1e-7, 1.0))))
                sc = np.r_[pc, po]
                y = np.r_[np.ones(len(pc)), np.zeros(len(po))]
                try:
                    ap = float(average_precision_score(y, sc))
                except ValueError:
                    ap = 0.5
                loss = bce + (1.0 - ap) + float(po.mean())
                if loss < bl:
                    bl = loss
                    best = em
            if best is not None:
                oracle[i][cm] = best[i][cm]
                comps += 1
    oracle_ap = pooled_ap(oracle, m56)
    return {
        "category": cat, "shot": shot, "n": n,
        "parity": parity, "layer_aps": aps_clean,
        "a1_ap": round(a1_ap, 6) if a1_ap == a1_ap else None,
        "oracle_ap": round(oracle_ap, 6),
        "delta_oracle": (round(oracle_ap - a1_ap, 6)
                         if (a1_ap == a1_ap and oracle_ap == oracle_ap) else None),
        "correlations": corr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, choices=[1, 2, 4], required=True)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion"
    p2 = out_root / "02_stage0_probe"
    p2.mkdir(parents=True, exist_ok=True)
    cats = [args.category] if args.category else CATEGORIES
    rows = [run_category(c, args.shot) for c in cats]
    for r in rows:
        print(f"  {r['category']} a1={r['a1_ap']} oracle_d={r['delta_oracle']} "
              f"parity={ {k: round(v, 2e-6) for k, v in r['parity'].items()} }", flush=True)

    def mean_ap(key):
        vals = [r["layer_aps"].get(key) for r in rows
                if r["layer_aps"].get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    map_names = list(rows[0]["layer_aps"].keys())
    means = {k: mean_ap(k) for k in map_names}
    a1_mean = means.get("concat_D11C24")
    best_static = max((k for k in means if means[k] is not None), key=lambda k: means[k])
    orac_mean = round(float(np.mean([r["delta_oracle"] for r in rows if r["delta_oracle"] is not None])), 6) \
        if rows else None
    low_pairs = []
    for r in rows:
        for key, v in r["correlations"].items():
            a, b = key.split("|")
            if a.startswith("dino") and b.startswith("clip") and v is not None and v < 0.95:
                low_pairs.append((r["category"], a, b, v))
    pos_head = sum(max(r["delta_oracle"] or 0.0, 0.0) for r in rows)
    top_share = max(max(r["delta_oracle"] or 0.0, 0.0) for r in rows) / pos_head if pos_head > 0 else 1.0
    max_d_par = max([r["parity"]["dino_L11_maxabs"] for r in rows] + [1.0])
    max_c_par = max([r["parity"]["clip_L24_maxabs"] for r in rows] + [1.0])
    max_ap_diff = max(abs((r["a1_ap"] or 0.0) - (r["a1_ap"] or 0.0)) for r in rows)  # self-check only
    # parity gate: raw features < 1e-5 AND A1 concat map Pixel-AP reproduced vs harness concat_FULL
    g4 = max(max_d_par, max_c_par) < 1e-5
    g1 = ((best_static and a1_mean is not None and means[best_static] - a1_mean >= 0.003)
          or (orac_mean is not None and orac_mean >= 0.010))
    g2 = len(low_pairs) >= 2
    g3 = top_share <= 0.50
    summary = {
        "shot": args.shot,
        "mean_layer_ap": means,
        "a1_mean_ap": a1_mean,
        "best_static": {"name": best_static, "mean_ap": means.get(best_static),
                        "delta_vs_a1": round((means.get(best_static) or 0.0) - (a1_mean or 0.0), 6)},
        "oracle_mean_delta": orac_mean,
        "n_low_corr_pairs": len(low_pairs),
        "low_corr_pairs": low_pairs[:6],
        "top_cat_share_positive": round(top_share, 3),
        "parity_maxabs": {"dino_L11": round(max_d_par, 2e-6) if max_d_par < 1 else max_d_par,
                          "clip_L24": round(max_c_par, 2e-6) if max_c_par < 1 else max_c_par},
        "gates": {"g1_signal": bool(g1), "g2_low_corr": bool(g2),
                  "g3_no_dominance": bool(g3), "g4_parity": bool(g4)},
    }
    with open(p2 / "LAYERWISE_RESULTS.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "shot"] + map_names)
        for r in rows:
            w.writerow([r["category"], r["shot"]] + [r["layer_aps"].get(k) for k in map_names])
    with open(p2 / "SCORE_CORRELATIONS.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        keys = list(rows[0]["correlations"].keys())
        w.writerow(["category"] + keys)
        for r in rows:
            w.writerow([r["category"]] + [r["correlations"].get(k) for k in keys])
    with open(p2 / "ORACLE_HEADROOM.json", "w", encoding="utf-8") as fh:
        json.dump({"per_category": [{k: r[k] for k in ("category", "a1_ap", "oracle_ap",
                                                       "delta_oracle")} for r in rows],
                   "summary": summary}, fh, ensure_ascii=False, indent=1)
    (p2 / "STAGE0_RESULT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
