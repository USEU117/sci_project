"""Breadth probe portfolio (2026-09-08): 10 candidates over 6 mechanism families.

Frozen-feature, real MPDD seed0 gate (k2 and k4), A1-compatible scoring
(1-max-cos via faiss L2/2 on unit rows -> 448 maps -> Pixel metrics stride 8).
Families: channel standardisation (CS), cross-image consistency (AGR2), KNN
depth top-5 (K5), within-image robust self-normalisation (WZ), centre-surround
residual descriptor (CSR, alpha in {0.25, 0.5}), branch score-level fusion
(MAP min/max/mean). Control must reproduce frozen macro AP (k2 .343706 / k4
.388328). Gates: family lead macro dAP >= +0.01 AND worst-cat >= -0.03 AND
macro dAUROC >= -0.005 (both shots; lead = best macro among pre-registered
variants of a family). Discipline: support-fit only, no test labels used to
fit/select/rout, frozen A1 untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import faiss  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

import evaluate_a1_feature_fusion as A1  # noqa: E402
from src.utils import dists2map  # noqa: E402

FEAT = ROOT / "outputs/dynamic_fusion/v3_direction_a"
OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908"
CATS = ("bracket_black", "bracket_brown", "bracket_white", "connector",
        "metal_plate", "tubes")
MAP_SIZE = (448, 448)
EPS = np.float32(1e-8)
REF_AP = {2: 0.343706, 4: 0.388328}
GATES = {"macro_ap": 0.01, "worst_cat_ap": -0.03, "macro_auroc": -0.005}
CELLS = 32 * 32
W = 0.5


def _fin(x):
    return None if x is None else float(x)


def _unit(x):
    return np.ascontiguousarray(normalize(np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])))


def _topk(q_flat, b_flat, k):
    q = np.ascontiguousarray(q_flat, dtype=np.float32)
    b = np.ascontiguousarray(b_flat, dtype=np.float32)
    faiss.normalize_L2(q)
    faiss.normalize_L2(b)
    idx = faiss.IndexFlatL2(b.shape[1])
    idx.add(b)
    sq, _ = idx.search(q, k=k)
    return (sq / 2.0).astype(np.float32)  # 1 - cos for unit rows


def _load(cat: str, shot: int):
    dp = FEAT / f"features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz"
    cp = FEAT / f"features_s0_k{shot}/anomalyclip_text/{cat}.npz"
    with np.load(dp, allow_pickle=False) as z:
        dino_feat = np.asarray(z["patch_features"], dtype=np.float32).copy()
        dino_ref = np.asarray(z["ref_patch_features"], dtype=np.float32).copy()
        masks = np.asarray(z["imgs_masks"], dtype=np.uint8).copy()
        grid = tuple(int(v) for v in z["grid_size"])
    with np.load(cp, allow_pickle=False) as z:
        clip_feat = np.asarray(z["patch_features"], dtype=np.float32).copy()
        clip_ref = np.asarray(z["ref_patch_features"], dtype=np.float32).copy()
    n = dino_feat.shape[0]
    Dd = dino_feat.shape[-1]
    # aligned branches on the dino 32x32 grid (identical to frozen concat prep)
    cf = A1.resize_patches(clip_feat, grid)
    cr = A1.resize_patches(clip_ref, grid)
    df_u = _unit(dino_feat).reshape(n, *grid, Dd)
    dr_u = _unit(dino_ref).reshape(dino_ref.shape)
    cf_u = _unit(cf.reshape(-1, cf.shape[-1])).reshape(cf.shape)
    cr_u = _unit(cr.reshape(-1, cr.shape[-1])).reshape(cr.shape)
    q_flat = _unit((np.concatenate([W * df_u, (1 - W) * cf_u], axis=-1)).reshape(-1, Dd + cf_u.shape[-1]))
    r_flat = _unit((np.concatenate([W * dr_u, (1 - W) * cr_u], axis=-1)).reshape(-1, Dd + cr_u.shape[-1]))
    return (q_flat, r_flat, masks, n, grid,
            df_u, dr_u, cf_u, cr_u)


def _neigh_mean(gf):
    """gf [.,32,32,D] -> [.,32,32,D] 3x3 reflect mean."""
    x = np.transpose(gf, (0, 3, 1, 2)).astype(np.float32)
    p = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
    acc = np.zeros_like(x)
    for di in range(3):
        for dj in range(3):
            acc += p[:, :, di:di + 32, dj:dj + 32]
    return np.transpose(acc / 9.0, (0, 2, 3, 1)).astype(np.float32)


def _maps(dists, n, grid):
    d = np.asarray(dists, dtype=np.float32).reshape(n, *grid)
    return np.stack([dists2map(x, MAP_SIZE) for x in d]).astype(np.float32)


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = _load(cat, shot)
    D = q_flat.shape[-1]
    dist: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    dz = _topk(q_flat, r_flat, 1)[:, 0]
    dist["C0"] = dz
    dist["CS"] = _topk(_unit((q_flat - np.mean(r_flat, 0)) / (np.std(r_flat, 0) + EPS)),
                      _unit((r_flat - np.mean(r_flat, 0)) / (np.std(r_flat, 0) + EPS)), 1)[:, 0]
    dist["K5"] = np.mean(_topk(q_flat, r_flat, 5), axis=1)
    K = int(r_flat.shape[0] // CELLS)
    if K > 1:
        per = np.stack([_topk(q_flat, r_flat[i * CELLS:(i + 1) * CELLS], 1)[:, 0] for i in range(K)], axis=1)
        srt = np.sort(per, axis=1)
        dist["AGR2"] = (srt[:, 0] + srt[:, 1]) / 2.0
    g = dz.reshape(n, CELLS)
    p75 = np.percentile(g, 75, axis=1, keepdims=True).astype(np.float32)
    p25 = np.percentile(g, 25, axis=1, keepdims=True).astype(np.float32)
    med = np.median(g, axis=1, keepdims=True).astype(np.float32)
    spread = np.maximum(p75 - p25, EPS)
    dist["WZ"] = ((g - med) / spread).astype(np.float32).reshape(-1)
    # CSR residual descriptor
    nq = q_flat.reshape(n, 32, 32, D)
    nr = r_flat.reshape(-1, 32, 32, D)
    q_res = _unit((nq - _neigh_mean(nq)).reshape(-1, D))
    r_res = _unit((nr - _neigh_mean(nr)).reshape(-1, D))
    dg = _topk(q_res, r_res, 1)[:, 0]
    for alpha in (0.25, 0.5):
        dist[f"CSR_A{int(alpha * 100)}"] = dz + np.float32(alpha) * dg
    # branch score-level fusion (both resized to the same 32 grid)
    dd = _topk(_unit(df_u.reshape(-1, df_u.shape[-1])), _unit(dr_u.reshape(-1, dr_u.shape[-1])), 1)[:, 0]
    dc = _topk(_unit(cf_u.reshape(-1, cf_u.shape[-1])), _unit(cr_u.reshape(-1, cr_u.shape[-1])), 1)[:, 0]
    d_stack = np.stack([dd, dc])  # anomaly distance: higher = more anomalous
    dist["MAP_max"] = np.max(d_stack, axis=0)
    dist["MAP_min"] = np.min(d_stack, axis=0)
    dist["MAP_mean"] = np.mean(d_stack, axis=0)
    elapsed = time.perf_counter() - t0

    ctrl_maps = _maps(dist["C0"], n, grid)
    ctrl_m = A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods = {}
    for cand, d in dist.items():
        if cand == "C0":
            m = ctrl_m
        else:
            m = A1.compute_metrics(_maps(d, n, grid).astype(np.float64), masks)
        methods[cand] = {k: _fin(v) for k, v in m.items()}
    return {"category": cat, "n_test": int(n), "memory_rows": int(r_flat.shape[0]),
            "compute_s": _fin(elapsed),
            "control": {k: _fin(v) for k, v in ctrl_m.items()},
            "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    ap.add_argument("--cats", default=None)
    args = ap.parse_args()
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else list(CATS)
    rows = [run_cat(c, args.shot) for c in cats]

    def mean(xs):
        return _fin(float(np.mean([x for x in xs if x is not None]))) if xs else None

    cands = [c for c in ("C0", "CS", "K5", "AGR2", "WZ", "CSR_A25", "CSR_A50",
                         "MAP_max", "MAP_min", "MAP_mean") if c in rows[0]["methods"]]
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    # family leads (pre-registered): CSR -> best macro of its alphas
    family_lead = {}
    for fam, members in (("CS", ["CS"]), ("AGR2", ["AGR2"]), ("K5", ["K5"]),
                         ("WZ", ["WZ"]), ("CSR", ["CSR_A25", "CSR_A50"]),
                         ("MAP", ["MAP_max", "MAP_min", "MAP_mean"])):
        avail = [m for m in members if m in agg]
        lead = max(avail, key=lambda m: agg[m]["macro_ap_delta"]) if avail else None
        if lead:
            family_lead[fam] = lead
    gates = {}
    for fam, lead in family_lead.items():
        a = agg[lead]
        gates[fam] = {"lead": lead, "macro_ap": a["macro_ap_delta"],
                      "worst_cat_ap": a["worst_cat_ap_delta"], "macro_auroc": a["macro_auroc_delta"],
                      "pass": bool(a["macro_ap_delta"] is not None and a["macro_ap_delta"] >= GATES["macro_ap"]
                                   and a["worst_cat_ap_delta"] is not None and a["worst_cat_ap_delta"] >= GATES["worst_cat_ap"]
                                   and a["macro_auroc_delta"] is not None and a["macro_auroc_delta"] >= GATES["macro_auroc"])}
    ctrl_macro = agg["C0"]["macro_pixel_ap"]
    result = {"seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
              "parity": {"control_macro_pixel_ap": ctrl_macro, "frozen_ref": REF_AP[args.shot],
                         "ok": abs(ctrl_macro - REF_AP[args.shot]) <= 3e-4},
              "aggregate": agg, "family_gates": gates, "per_category": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"RESULTS_s0_k{args.shot}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"shot": args.shot, "parity": result["parity"],
                      "macro_ap_delta": {c: agg[c]["macro_ap_delta"] for c in agg if c != "C0"},
                      "worst_cat": {c: agg[c]["worst_cat_ap_delta"] for c in agg if c != "C0"},
                      "gates": gates}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
