"""Breadth probe ROUND 10 (axis: reciprocal-NN reliability, 2026-09-09 night).

Two offline, never-run mechanism families (identified by the historical axis
ledger; the first is explicitly proposed-but-not-run in
innovation_followup_20260908/neighborhood/AUDIT_CN.md):

  RCW  support-side reciprocal coverage weighting: for each memory row m,
       measure its stability s_m across the OTHER support images via forward
       NN + back-check mutuality (s_m = mutual-image fraction in [0,1]); score
       with a top-K=8 candidate reweighting  min_m d(q,m)*(1+beta*(1-s_m)).
       Uses NO query statistics at all (unlike every test-time adaptation
       family WZ/LCN/PA/FastRef).
  BRC  query<->memory bidirectional mutual-NN consistency: for the A1 top-1
       memory row m of patch q, check whether m's nearest patch INSIDE the
       same query image is q; d' = d*(1+beta*(1-recip)).  Anomaly-reinforcing
       direction (opposite to the PA/CB hard-class-rescue signature).

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

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "innovation_breadth_20260908"
for p in (str(ROOT / "scripts"), str(SCRIPTS), str(ROOT / "src"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round10"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
CELLS = PB.CELLS
BETAS = (0.10, 0.30)
TOPK_RCW = 8


def _fin(x):
    return None if x is None else float(x)


def _knn(q, b, k):
    """faiss cosine top-k on unit rows -> (dists 1-cos, indices)."""
    qq = np.ascontiguousarray(q, dtype=np.float32)
    bb = np.ascontiguousarray(b, dtype=np.float32)
    faiss.normalize_L2(qq)
    faiss.normalize_L2(bb)
    idx = faiss.IndexFlatL2(bb.shape[1])
    idx.add(bb)
    sq, ind = idx.search(qq, k=k)
    return (sq / 2.0).astype(np.float32), ind


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    t0 = time.perf_counter()
    methods = {}

    M = q_flat.shape[0]
    Dm = q_flat.shape[-1]
    k_shot = r_flat.shape[0] // CELLS

    d1, i1 = _knn(q_flat, r_flat, 1)
    ctrl_maps = PB._maps(d1[:, 0], n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods["C0"] = {k: _fin(v) for k, v in ctrl_m.items()}

    # ---------- RCW: support-side reciprocal coverage stability ----------
    Rb = np.ascontiguousarray(r_flat, dtype=np.float32).reshape(k_shot, CELLS, Dm)
    stab = np.zeros(r_flat.shape[0], dtype=np.float32)
    for p in range(k_shot):
        rows_p = Rb[p]                                   # [CELLS,D]
        for p2 in range(k_shot):
            if p2 == p:
                continue
            _, fwd = _knn(rows_p, Rb[p2], 1)             # nearest row in image p2
            sel = fwd[:, 0]
            _, bwd = _knn(Rb[p2][sel], rows_p, 1)        # back to image p
            mutual = (bwd[:, 0] == np.arange(CELLS, dtype=np.int32))
            stab[p * CELLS + np.arange(CELLS, dtype=np.int64)] += mutual.astype(np.float32)
    stab /= max(1, k_shot - 1)
    stab = np.clip(stab, 0.0, 1.0)

    dk, ik = _knn(q_flat, r_flat, TOPK_RCW)              # [M,K]
    for beta in BETAS:
        w = 1.0 + beta * (1.0 - stab[ik])
        dmod = (dk * w).min(axis=1)
        maps = PB._maps(dmod, n, grid)
        methods[f"RCW_{int(beta * 100)}"] = {k: _fin(v) for k, v in
                                             PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    # ---------- BRC: query<->memory bidirectional mutual consistency ----------
    recip = np.zeros(M, dtype=np.float32)
    for i in range(n):
        s, e = i * CELLS, (i + 1) * CELLS
        qi = q_flat[s:e]
        mm = i1[s:e, 0]
        uniq, inv = np.unique(mm, return_inverse=True)
        _, back = _knn(r_flat[uniq], qi, 1)              # each matched m -> nearest patch in this image
        hit = back[:, 0]
        recip[s:e] = (hit[inv] == np.arange(CELLS, dtype=np.int32)).astype(np.float32)
    for beta in BETAS:
        dmod = d1[:, 0] * (1.0 + beta * (1.0 - recip))
        maps = PB._maps(dmod, n, grid)
        methods[f"BRC_{int(beta * 100)}"] = {k: _fin(v) for k, v in
                                             PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    elapsed = time.perf_counter() - t0
    return {"category": cat, "n_test": int(n), "compute_s": _fin(elapsed),
            "mean_support_stability": _fin(float(stab.mean())),
            "mean_query_reciprocal": _fin(float(recip.mean())),
            "control": {k: _fin(v) for k, v in ctrl_m.items()}, "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    args = ap.parse_args()
    rows = [run_cat(c, args.shot) for c in CATS]

    def mean(xs):
        return _fin(float(np.mean([x for x in xs if x is not None]))) if xs else None

    cands = ("C0", "RCW_10", "RCW_30", "BRC_10", "BRC_30")
    agg = {}
    for c in cands:
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("RCW", ["RCW_10", "RCW_30"]), ("BRC", ["BRC_10", "BRC_30"])):
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
    result = {"round": 10, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
              "parity": {"control_macro_pixel_ap": ctrl_macro, "frozen_ref": REF_AP[args.shot],
                         "ok": abs(ctrl_macro - REF_AP[args.shot]) <= 3e-4},
              "aggregate": agg, "family_gates": gates, "per_category": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"RESULTS_s0_k{args.shot}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"shot": args.shot, "parity": result["parity"],
                      "macro_ap_delta": {c: agg[c]["macro_ap_delta"] for c in cands if c != "C0"},
                      "worst_cat": {c: agg[c]["worst_cat_ap_delta"] for c in cands if c != "C0"},
                      "gates": gates,
                      "diag": {r["category"]: (r["mean_support_stability"], r["mean_query_reciprocal"]) for r in rows}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
