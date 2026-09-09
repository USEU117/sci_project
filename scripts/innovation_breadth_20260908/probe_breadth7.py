"""Breadth probe ROUND 7 (axes: matching scale / channel selection / neighbor-rank density, 2026-09-09 night).

Three offline mechanism families never isolated before on the frozen A1 fused
features (concat of unit DINO-32grid + unit CLIP-text-32grid, 0.5/0.5):

  MRS  multi-resolution block-pooled matching: block-average the fused-grid
       cell features (query and k-shot reference, both grid-arranged in the
       cache) over p x p blocks, re-unit each pooled cell, re-fuse 0.5/0.5,
       top-1 in the coarser grid, map to 448.  p in {2,4} -> 16/8 grids.
       (distinct from R1's 3x3 descriptor smoothing at a FIXED 32 grid:
       here the matching granularity itself changes)
  CSP  reference-stability channel selection: from the k-shot reference rows
       keep the fraction f of channels with the SMALLEST cross-ref variance,
       then re-unit rows and top-1.  Selection only, no scaling/centering
       (unlike R1 CS / R4 SCA).  f in {0.50, 0.75}.
  NDW  neighbor-rank density: faiss top-2 -> d1, d2;  NDW_d2 = d2 (pure 2nd-
       neighbor distance), NDW_gm = sqrt(d1*d2).  (unlike R1 K5 = top-5 MEAN:
       this uses the joint information of the two nearest neighbors)

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

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round7"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
S = 32  # frozen fused grid side
D_TOT = 768 + 512  # dino 768 + clip-resized 512


def _fin(x):
    return None if x is None else float(x)


def _pool(x, p):
    """x [...,32,32,D] -> block-mean over p x p -> [...,g,g,D]."""
    sh = x.shape
    g = S // p
    xr = x.reshape(sh[:-3] + (g, p, g, p, sh[-1]))
    return xr.mean(axis=(-4, -2))


def _pu(x, p):
    """pool then unit-normalize each cell row (keeps leading shape)."""
    y = _pool(x, p)
    return PB._unit(y).reshape(y.shape)


def _fused(qd, qc, rd, rc):
    """fuse pooled unit cells with A1's 0.5/0.5 concat + row unit."""
    def flat(z):
        return np.asarray(z, dtype=np.float32).reshape(-1, z.shape[-1])
    q = PB._unit(np.concatenate([0.5 * flat(qd), 0.5 * flat(qc)], axis=-1))
    r = PB._unit(np.concatenate([0.5 * flat(rd), 0.5 * flat(rc)], axis=-1))
    return q, r


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    t0 = time.perf_counter()

    methods = {}
    dz1 = PB._topk(q_flat, r_flat, 1)[:, 0]
    ctrl_maps = PB._maps(dz1, n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods["C0"] = {k: _fin(v) for k, v in ctrl_m.items()}

    # ---- MRS: coarser-grid fused top-1 ----
    for p, lab in ((2, "MRS_p2"), (4, "MRS_p4")):
        g = S // p
        qm, rm = _fused(_pu(df_u, p), _pu(cf_u, p), _pu(dr_u, p), _pu(cr_u, p))
        d = PB._topk(qm, rm, 1)[:, 0]
        maps = PB._maps(d, n, (g, g))
        methods[lab] = {k: _fin(v) for k, v in
                        PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    # ---- CSP: reference-stability channel selection ----
    std_r = np.asarray(r_flat, dtype=np.float32).std(axis=0)
    order = np.argsort(std_r)
    for f, lab in ((0.50, "CSP_f50"), (0.75, "CSP_f75")):
        keepn = max(1, int(D_TOT * f))
        keep = np.sort(order[:keepn])
        qs = PB._unit(q_flat[:, keep])
        rs = PB._unit(r_flat[:, keep])
        d = PB._topk(qs, rs, 1)[:, 0]
        maps = PB._maps(d, n, grid)
        methods[lab] = {k: _fin(v) for k, v in
                        PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    # ---- NDW: top-2 neighbor density ----
    d2 = PB._topk(q_flat, r_flat, 2)
    d1, dd = d2[:, 0], d2[:, 1]
    ndw = {"NDW_d2": dd, "NDW_gm": np.sqrt(np.clip(d1 * dd, 0.0, None))}
    for lab, d in ndw.items():
        maps = PB._maps(d, n, grid)
        methods[lab] = {k: _fin(v) for k, v in
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

    cands = ("C0", "MRS_p2", "MRS_p4", "CSP_f50", "CSP_f75", "NDW_d2", "NDW_gm")
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("MRS", ["MRS_p2", "MRS_p4"]),
                         ("CSP", ["CSP_f50", "CSP_f75"]),
                         ("NDW", ["NDW_d2", "NDW_gm"])):
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
    result = {"round": 7, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
