"""Breadth probe portfolio ROUND 3 (2026-09-08): new families + WZ attribution.

Reuses probe_breadth harness. Families:
  COMB  weighted sum of fused-concat distance (dz) and CLIP-only distance (dc),
        alpha in {0.25, 0.5}  (round-2 RB used DINO-only; this is the CLIP side);
  GAP   nearest-neighbour gap d2 - d1 (match-ambiguity as signal), plus a
        pre-registered mix GAP_MIX = 0.5*d1 + 0.5*gap;
  WZD   WZ attribution (read-only): WZC = d - median (center only),
        WZS = d / spread (scale only), where spread = p75-p25 of the per-image
        patch-distance distribution.  Diagnoses which part drives round-1 WZ's
        macro-positive / worst-negative pattern.
Gates identical (macro dAP >= +0.01 AND worst-cat >= -0.03, both shots).
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
for p in (str(ROOT / "scripts"), str(SCRIPTS), str(ROOT / "src"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round3"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
EPS = PB.EPS


def _fin(x):
    return None if x is None else float(x)


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    dist: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    dz = PB._topk(q_flat, r_flat, 1)[:, 0]
    dist["C0"] = dz
    # COMB: fused + CLIP-only
    dc = PB._topk(PB._unit(cf_u.reshape(-1, cf_u.shape[-1])),
                  PB._unit(cr_u.reshape(-1, cr_u.shape[-1])), 1)[:, 0]
    for alpha in (0.25, 0.5):
        dist[f"COMB_A{int(alpha * 100)}"] = ((1.0 - alpha) * dz + alpha * dc).astype(np.float32)
    # GAP family: d2 - d1
    d2 = PB._topk(q_flat, r_flat, 2)[:, 1]
    gap = np.clip(d2 - dz, 0.0, None).astype(np.float32)
    dist["GAP"] = gap
    dist["GAP_MIX"] = (0.5 * dz + 0.5 * gap).astype(np.float32)
    # WZD attribution
    g = dz.reshape(n, 32, 32)
    med = np.median(g, axis=(1, 2), keepdims=True).astype(np.float32)
    p75 = np.percentile(g, 75, axis=(1, 2), keepdims=True).astype(np.float32)
    p25 = np.percentile(g, 25, axis=(1, 2), keepdims=True).astype(np.float32)
    spread = np.maximum(p75 - p25, EPS).astype(np.float32)
    dist["WZC"] = (g - med).astype(np.float32).reshape(-1)
    dist["WZS"] = (g / spread).astype(np.float32).reshape(-1)
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

    cands = ["C0", "COMB_A25", "COMB_A50", "GAP", "GAP_MIX", "WZC", "WZS"]
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("COMB", ["COMB_A25", "COMB_A50"]), ("GAP", ["GAP", "GAP_MIX"]),
                         ("WZD", ["WZC", "WZS"])):
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
    result = {"round": 3, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
