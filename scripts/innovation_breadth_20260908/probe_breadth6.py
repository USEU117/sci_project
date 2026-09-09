"""Breadth probe ROUND 6 (axis: data / memory-bank side, 2026-09-09).

Two mechanism families suggested by web-derived few-shot AD literature
(VisionAD arXiv:2504.11895: class-aware one-for-all memory bank + dual
support/query augmentation; pseudo-label / self-training style bank growth)
that operate on a DIFFERENT axis from the R1-R5 probes (score transforms,
feature transforms, map post-processing, geometry):

  CB  cross-category one-for-all joint bank: the bank of every category is
      built from the SAME-shot reference patches across the whole MPDD
      split.  CB_full = union of all 6 categories; CB_selfplus = own bank +
      the single nearest other category by fused-space bank-centroid
      distance (selective transfer).  NOTE: faithful VisionAD also uses
      multi-layer feature integration + real geometric augmentation, both
      requiring GPU re-encoding (out of scope, no GPU): this is a
      falsification of the cheap offline union-bank shortcut.
  PA  pseudo-label support self-refinement (test-time self-training on the
      BANK side; no ground truth): query patches that are confident-normal
      (dino-branch top-1 dist < q_t AND clip-branch top-1 dist < q_t AND
      branch agreement |d_d - d_c| <= median) are appended to the reference
      bank and scoring is re-done with per-image leave-own-image-out banks
      (no trivial self-matching).  Distinct from FastRef (query-side OT to
      a STATIC prototype set): here the reference bank itself grows.

Gates identical to all breadth rounds (lead macro dAP >= +0.01 AND worst
>= -0.03 AND dAUROC >= -0.005, both shots; parity k2 .343706 / k4 .388328).
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

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round6"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
PA_QT = (0.20, 0.40)


def _fin(x):
    return None if x is None else float(x)


def _d1(q, b):
    """top-1 (1-cos) distance over unit rows (PB._topk k=1)."""
    return PB._topk(np.ascontiguousarray(q, dtype=np.float32),
                    np.ascontiguousarray(b, dtype=np.float32), 1)[:, 0]


def run_cat(cat: str, shot: int, r_all: dict) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    t0 = time.perf_counter()
    rows_per_img = q_flat.shape[0] // n
    img_ids = np.arange(n).repeat(rows_per_img)

    # ---- control: frozen A1 concat top-1 ----
    dz0 = _d1(q_flat, r_flat)
    ctrl_maps = PB._maps(dz0, n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods = {"C0": {k: _fin(v) for k, v in ctrl_m.items()}}

    # branch-raw unit rows (PA gating only; no fused-bank change)
    Dd = df_u.shape[-1]
    qd = np.asarray(df_u, dtype=np.float32).reshape(-1, Dd)
    rd = np.asarray(dr_u, dtype=np.float32).reshape(-1, Dd)
    qc = np.asarray(cf_u, dtype=np.float32).reshape(-1, cf_u.shape[-1])
    rc = np.asarray(cr_u, dtype=np.float32).reshape(-1, cr_u.shape[-1])
    d_d = _d1(qd, rd)
    d_c = _d1(qc, rc)

    # ---- CB: cross-category joint bank ----
    others = [c for c in CATS if c != cat]
    cb_full = np.concatenate([r_flat] + [r_all[c] for c in others]).astype(np.float32)
    own_cent = PB._unit(np.mean(r_flat, axis=0, keepdims=True))
    best, bd = None, np.inf
    for c in others:
        oc = PB._unit(np.mean(r_all[c], axis=0, keepdims=True))
        dd = float(np.linalg.norm(own_cent - oc))
        if dd < bd:
            bd, best = dd, c
    cb_selfplus = np.concatenate([r_flat, r_all[best]]).astype(np.float32)
    for name, bank in (("CB_full", cb_full), ("CB_selfplus", cb_selfplus)):
        maps = PB._maps(_d1(q_flat, bank), n, grid)
        methods[name] = {k: _fin(v) for k, v in
                         PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    # ---- PA: pseudo-label bank growth (per-image leave-own-image-out) ----
    agree = np.abs(d_d - d_c)
    a_t = float(np.median(agree))
    pa_notes = {}
    for qt in PA_QT:
        t_d = float(np.quantile(d_d, qt))
        t_c = float(np.quantile(d_c, qt))
        sel = (d_d <= t_d) & (d_c <= t_c) & (agree <= a_t)
        if int(sel.sum()) < 50:  # too few -> relax branch-agreement only
            sel = (d_d <= t_d) & (d_c <= t_c)
        sel_img = img_ids[sel]
        aug = np.asarray(q_flat[sel], dtype=np.float32)
        pa_notes[f"PA_q{int(qt * 100)}"] = int(sel.sum())
        d_pa = np.zeros(q_flat.shape[0], dtype=np.float32)
        for i in range(n):
            bank_i = np.concatenate([r_flat, aug[sel_img != i]]).astype(np.float32)
            q_i = q_flat[i * rows_per_img:(i + 1) * rows_per_img]
            d_pa[i * rows_per_img:(i + 1) * rows_per_img] = _d1(q_i, bank_i)
        maps = PB._maps(d_pa, n, grid)
        methods[f"PA_q{int(qt * 100)}"] = {k: _fin(v) for k, v in
                                           PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    elapsed = time.perf_counter() - t0
    return {"category": cat, "n_test": int(n), "compute_s": _fin(elapsed),
            "pa_selected": pa_notes,
            "control": {k: _fin(v) for k, v in ctrl_m.items()}, "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    args = ap.parse_args()

    r_all = {}
    for c in CATS:  # same-shot fused reference banks for cross-category CB
        q_, r_, *_ = PB._load(c, args.shot)
        r_all[c] = r_

    rows = [run_cat(c, args.shot, r_all) for c in CATS]

    def mean(xs):
        return _fin(float(np.mean([x for x in xs if x is not None]))) if xs else None

    cands = ("C0", "CB_full", "CB_selfplus", "PA_q20", "PA_q40")
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("CB", ["CB_full", "CB_selfplus"]),
                         ("PA", ["PA_q20", "PA_q40"])):
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
    result = {"round": 6, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
              "parity": {"control_macro_pixel_ap": ctrl_macro, "frozen_ref": REF_AP[args.shot],
                         "ok": abs(ctrl_macro - REF_AP[args.shot]) <= 3e-4},
              "aggregate": agg, "family_gates": gates, "per_category": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"RESULTS_s0_k{args.shot}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"shot": args.shot, "parity": result["parity"],
                      "macro_ap_delta": {c: agg[c]["macro_ap_delta"] for c in cands if c != "C0"},
                      "worst_cat": {c: agg[c]["worst_cat_ap_delta"] for c in cands if c != "C0"},
                      "gates": gates, "pa_selected": {r["category"]: r["pa_selected"] for r in rows}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
