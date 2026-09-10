"""Breadth probe ROUND 12 (axis: region-level (connected-component) score pooling, 2026-09-09 night).

Last unexplored offline axis from the historical ledger: REGION-level map
post-processing (distinct from R5 MH pixel morphology and R7/R11 spatial
pooling of features):

  for each 448 map: threshold at the per-image quantile t=quantile(M,1-q)
  (q in {0.30, 0.50} pre-registered, no tuning), take 8-connected components,
  and for components with area >= 16 px replace every pixel's score by the
  component MEAN (region pooling); all other pixels keep their raw score.
  This is a per-image NON-monotone transform, hence it can affect the
  per-class Pixel-AP in this harness (per-class-macro metric), unlike any
  monotone calibration.

Pre-registered gates identical to all breadth rounds (lead macro dAP >= +0.01
AND worst >= -0.03 AND dAUROC >= -0.005, both shots; parity k2 .343706 /
k4 .388328).  Frozen features, no ground-truth fitting, no post-hoc tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "innovation_breadth_20260908"
for p in (str(ROOT / "scripts"), str(SCRIPTS), str(ROOT / "src"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round12"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
QS = (0.30, 0.50)
AREA_MIN = 16


def _fin(x):
    return None if x is None else float(x)


def _region_pool(mp: np.ndarray, q: float) -> np.ndarray:
    t = float(np.quantile(mp, 1.0 - q))
    b = (mp >= t).astype(np.uint8)
    if not b.any():
        return mp
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = mp.copy()
    for l in range(1, nlab):
        if int(stats[l, cv2.CC_STAT_AREA]) < AREA_MIN:
            continue
        sel = lab == l
        out[sel] = float(mp[sel].mean())
    return out


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, *_ = PB._load(cat, shot)
    t0 = time.perf_counter()
    methods = {}

    d1 = PB._topk(q_flat, r_flat, 1)[:, 0]
    ctrl_maps = PB._maps(d1, n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods["C0"] = {k: _fin(v) for k, v in ctrl_m.items()}

    for q in QS:
        pooled = np.stack([_region_pool(ctrl_maps[i], q) for i in range(n)]).astype(np.float32)
        methods[f"RGN_q{int(q * 100)}"] = {k: _fin(v) for k, v in
                                           PB.A1.compute_metrics(pooled.astype(np.float64), masks).items()}

    elapsed = time.perf_counter() - t0
    return {"category": cat, "n_test": int(n), "compute_s": _fin(elapsed),
            "control": {k: _fin(v) for k, v in ctrl_m.items()}, "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    args = ap.parse_args()
    rows = [run_cat(c, args.shot) for c in CATS]

    def mean(xs):
        return _fin(float(np.mean([x for x in xs if x is not None]))) if xs else None

    cands = ("C0", "RGN_q30", "RGN_q50")
    agg = {}
    for c in cands:
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    members = ["RGN_q30", "RGN_q50"]
    lead = max(members, key=lambda m: agg[m]["macro_ap_delta"])
    fam_lead = {"RGN": lead}
    gates = {}
    for fam, ld in fam_lead.items():
        a = agg[ld]
        gates[fam] = {"lead": ld, "macro_ap": a["macro_ap_delta"],
                      "worst_cat_ap": a["worst_cat_ap_delta"], "macro_auroc": a["macro_auroc_delta"],
                      "pass": bool(a["macro_ap_delta"] is not None and a["macro_ap_delta"] >= GATES["macro_ap"]
                                   and a["worst_cat_ap_delta"] is not None and a["worst_cat_ap_delta"] >= GATES["worst_cat_ap"]
                                   and a["macro_auroc_delta"] is not None and a["macro_auroc_delta"] >= GATES["macro_auroc"])}
    ctrl_macro = mean([r["control"]["pixel_ap"] for r in rows])
    result = {"round": 12, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
              "parity": {"control_macro_pixel_ap": ctrl_macro, "frozen_ref": REF_AP[args.shot],
                         "ok": abs(ctrl_macro - REF_AP[args.shot]) <= 3e-4},
              "aggregate": agg, "family_gates": gates, "per_category": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"RESULTS_s0_k{args.shot}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"shot": args.shot, "parity": result["parity"],
                      "macro_ap_delta": {c: agg[c]["macro_ap_delta"] for c in cands if c != "C0"},
                      "worst_cat": {c: agg[c]["worst_cat_ap_delta"] for c in cands if c != "C0"},
                      "gates": gates}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
