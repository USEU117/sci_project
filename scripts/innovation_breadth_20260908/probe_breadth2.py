"""Breadth probe portfolio ROUND 2 (2026-09-08): 4 new mechanism families.

Reuses the frozen-feature real MPDD seed0 gate harness from probe_breadth.py.
New families (not overlapping round 1):
  RW  reference-typicality weighted bank metric (penalise outlier ref cells),
      score = min_i (d_i + lambda*w_i), w_i = ref-cell 2nd-NN self distance,
      lambda in {0.25, 0.5};
  LCN local (4x4 block) contrast normalisation of the per-image distance grid;
  RB  weighted sum of the fused-concat distance (dz) and DINO-only distance (dd),
      alpha in {0.25, 0.5};
  BS  bagged sub-bank median (3 deterministic 80% row samples, seeds 0/1/2).
Control parity and gates identical to round 1 (macro dAP >= +0.01 AND worst-cat
>= -0.03 AND dAUROC >= -0.005, both shots).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "innovation_breadth_20260908"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round2"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
CELLS = PB.CELLS
GRID = (32, 32)
EPS = PB.EPS


def _fin(x):
    return None if x is None else float(x)


def rw_scan(q, b, w, lam):
    """chunked min_i (1-cos(q,b_i) + lam*w_i); returns [nq]."""
    lam = np.float32(lam)
    out = np.empty(q.shape[0], dtype=np.float32)
    chunk = 16384
    for s in range(0, q.shape[0], chunk):
        qq = PB._unit(q[s:s + chunk])
        dots = qq @ b.T  # cos
        vals = dots - lam * w[None, :]
        mx = vals.max(axis=1)
        out[s:s + chunk] = (1.0 - mx).astype(np.float32)
    return out


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    dist: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    dz = PB._topk(q_flat, r_flat, 1)[:, 0]
    dist["C0"] = dz
    # RW: reference typicality penalty
    w_self = PB._topk(r_flat, r_flat, 2)[:, 1]
    for lam in (0.25, 0.5):
        dist[f"RW_A{int(lam * 100)}"] = rw_scan(q_flat, r_flat, w_self, lam)
    # LCN: 4x4 block contrast normalisation per query image
    g = dz.reshape(n, 32, 32)
    g4 = g.reshape(n, 8, 4, 8, 4)
    bm = g4.mean(axis=(2, 4), keepdims=True).astype(np.float32)
    bs = g4.std(axis=(2, 4), keepdims=True).astype(np.float32) + EPS
    lcn = ((g4 - bm) / bs).astype(np.float32).reshape(n, 32, 32)
    dist["LCN"] = lcn.reshape(-1)
    # RB: concat distance + DINO-only distance
    dd = PB._topk(PB._unit(df_u.reshape(-1, df_u.shape[-1])),
                  PB._unit(dr_u.reshape(-1, dr_u.shape[-1])), 1)[:, 0]
    for alpha in (0.25, 0.5):
        dist[f"RB_A{int(alpha * 100)}"] = ((1.0 - alpha) * dz + alpha * dd).astype(np.float32)
    # BS: bagged sub-bank median (80% rows, deterministic seeds)
    R = r_flat.shape[0]
    nkeep = int(round(0.8 * R))
    sub_dists = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        keep = rng.permutation(R)[:nkeep]
        sub_dists.append(PB._topk(q_flat, r_flat[keep], 1)[:, 0])
    dist["BS"] = np.median(np.stack(sub_dists), axis=0).astype(np.float32)
    elapsed = time.perf_counter() - t0

    ctrl_maps = PB._maps(dist["C0"], n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods = {}
    for cand, d in dist.items():
        if cand == "C0":
            m = ctrl_m
        else:
            m = PB.A1.compute_metrics(PB._maps(d, n, grid).astype(np.float64), masks)
        methods[cand] = {k: _fin(v) for k, v in m.items()}
    return {"category": cat, "n_test": int(n), "compute_s": _fin(elapsed),
            "control": {k: _fin(v) for k, v in ctrl_m.items()}, "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    args = ap.parse_args()
    rows = [run_cat(c, args.shot) for c in CATS]

    def mean(xs):
        return _fin(float(np.mean([x for x in xs if x is not None]))) if xs else None

    cands = [c for c in ("C0", "RW_A25", "RW_A50", "LCN", "RB_A25", "RB_A50", "BS")]
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("RW", ["RW_A25", "RW_A50"]), ("LCN", ["LCN"]),
                         ("RB", ["RB_A25", "RB_A50"]), ("BS", ["BS"])):
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
    result = {"round": 2, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
