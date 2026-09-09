"""Breadth probe ROUND 5 (axis: post-processing / geometry, 2026-09-09).

Two further mechanism families computable offline on the frozen caches:
  MH   morphological top-hat on the 448 anomaly map (small-bright-bump emphasis,
       kernel disc radius {2,4}); large-area defects are expected to be harmed;
  GV   geometric rotation voting on the QUERY patch grid (rot0/90/180/270 of the
       32x32 feature grid -> top-1 bank distance per orientation -> min / mean).
       NOTE: rotating feature positions is NOT content-equivariant re-encoding,
       this is a falsification probe of a cheap geometry shortcut (no re-encode).
Gates identical to all breadth rounds (lead macro dAP >= +0.01 AND worst >= -0.03,
both shots; parity k2 .343706 / k4 .388328).
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

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round5"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
GRID = (32, 32)


def _fin(x):
    return None if x is None else float(x)


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    t0 = time.perf_counter()
    dz = PB._topk(q_flat, r_flat, 1)[:, 0]
    gz = np.asarray(dz, dtype=np.float32).reshape(n, 32, 32)

    # --- GV: geometric rotation voting on the QUERY patch grid (falsification probe)
    votes = np.stack([gz] + [np.rot90(gz, k, axes=(1, 2)) for k in (1, 2, 3)])  # [4,n,32,32]
    gv = {"GV_min": votes.min(axis=0).reshape(-1),
          "GV_mean": votes.mean(axis=0).reshape(-1)}
    elapsed = time.perf_counter() - t0

    methods = {}
    ctrl_maps = PB._maps(dz, n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods["C0"] = {k: _fin(v) for k, v in ctrl_m.items()}
    for cand, d in gv.items():
        maps = PB._maps(d, n, grid)
        m = PB.A1.compute_metrics(maps.astype(np.float64), masks)
        methods[cand] = {k: _fin(v) for k, v in m.items()}

    # --- MH: morphological top-hat on the 448 control maps (small-bright-bump emphasis)
    for radius in (2, 4):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        th = np.stack([np.clip(cv2.morphologyEx(mp, cv2.MORPH_TOPHAT, kernel), 0, None)
                       for mp in ctrl_maps]).astype(np.float32)
        methods[f"MH_r{radius}"] = {k: _fin(v) for k, v in
                                    PB.A1.compute_metrics(th.astype(np.float64), masks).items()}
    return {"category": cat, "n_test": int(n), "compute_s": _fin(elapsed),
            "control": {k: _fin(v) for k, v in ctrl_m.items()}, "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    args = ap.parse_args()
    rows = [run_cat(c, args.shot) for c in CATS]

    def mean(xs):
        return _fin(float(np.mean([x for x in xs if x is not None]))) if xs else None

    cands = [c for c in ("C0", "GV_min", "GV_mean", "MH_r2", "MH_r4") if c in rows[0]["methods"]]
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("GV", ["GV_min", "GV_mean"]), ("MH", ["MH_r2", "MH_r4"])):
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
    result = {"round": 5, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
