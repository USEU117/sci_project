"""Breadth probe ROUND 9 (axes: distance metric / positional prior, 2026-09-09 night).

Two further offline axes on the frozen A1 fused concat features (row-unit):

  DST  replace the frozen 1-cos (L2-on-unit) top-1 with L1 and Chebyshev(Linf)
       top-1 over the SAME unit rows.  Metric-space axis (never isolated).
  PLC  position-locked matching: cell (i,j) of a query is compared ONLY to the
       same (i,j) cells across the k-shot reference images (registration-free
       positional prior).  Pre-registered as a CONTROL falsification: MPDD
       bracket parts show rotations/pose differences (GV / COP / rotation-
       alignment evidence), so position-locking is expected to be harmful and
       closes the "positional prior without registration" evidence chain.

Pre-registered gates identical to all breadth rounds (lead macro dAP >= +0.01
AND worst >= -0.03 AND dAUROC >= -0.005, both shots; parity k2 .343706 /
k4 .388328).  Frozen features, no test-label fitting, no post-hoc tuning.
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

from scipy.spatial.distance import cdist  # noqa: E402

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round9"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
S = 32


def _fin(x):
    return None if x is None else float(x)


def _metric_dists(q, b, kind):
    """row-unit q [M,D], b [R,D]; top-1 per-row distance of given kind (cdist)."""
    q = np.ascontiguousarray(q, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    metric = "cityblock" if kind == "l1" else "chebyshev"
    M = q.shape[0]
    out = np.empty(M, dtype=np.float32)
    for s in range(0, M, 2048):
        out[s:s + 2048] = cdist(q[s:s + 2048], b, metric=metric).min(axis=-1)
    return out


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    t0 = time.perf_counter()
    methods = {}

    dz1 = PB._topk(q_flat, r_flat, 1)[:, 0]
    ctrl_maps = PB._maps(dz1, n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods["C0"] = {k: _fin(v) for k, v in ctrl_m.items()}

    # ---- DST metric-space top-1 ----
    for kind, lab in (("l1", "DST_L1"), ("cheb", "DST_cheb")):
        d = _metric_dists(q_flat, r_flat, kind)
        maps = PB._maps(d, n, grid)
        methods[lab] = {k: _fin(v) for k, v in
                        PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    # ---- PLC position-locked matching (control falsification) ----
    # fused rows are image-major then row-major over the 32x32 grid for both
    # query (n) and reference (k) sets; reshape keeps position indices aligned.
    D_TOT = q_flat.shape[-1]
    k = r_flat.shape[0] // (S * S)
    q3 = np.ascontiguousarray(q_flat, dtype=np.float32).reshape(n, S, S, D_TOT)
    r3 = np.ascontiguousarray(r_flat, dtype=np.float32).reshape(k, S, S, D_TOT)
    d_plc = np.zeros((n, S, S), dtype=np.float32)
    for i in range(S):
        for j in range(S):
            bank = np.ascontiguousarray(r3[:, i, j, :])   # [k,D]
            qq = np.ascontiguousarray(q3[:, i, j, :])     # [n,D]
            d_plc[:, i, j] = PB._topk(PB._unit(qq), PB._unit(bank), 1)[:, 0]
    maps = PB._maps(d_plc.reshape(-1), n, grid)
    methods["PLC"] = {k: _fin(v) for k, v in
                      PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

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

    cands = ("C0", "DST_L1", "DST_cheb", "PLC")
    agg = {}
    for c in cands:
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("DST", ["DST_L1", "DST_cheb"]), ("PLC", ["PLC"])):
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
    ctrl_macro = mean([r["control"]["pixel_ap"] for r in rows])
    result = {"round": 9, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
