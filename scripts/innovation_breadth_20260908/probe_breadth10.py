"""Breadth probe ROUND 11 (axes: memory densification / multi-granularity, 2026-09-09 night).

Two never-run offline families from the historical axis ledger gaps:

  MDN  memory densification by support-feature mixup: append synthetic
       reference rows normalize(0.5*(r_i + r_j)) for random pairs (fixed
       seed0) to the frozen bank, then top-1 as usual.  Reference-side only
       (no query statistics), distinct from coreset (removal), RW/BS
       (reweight/subsample) and PA (test-time growth).
  MG   multi-granularity coexistence: keep the 32-grid A1 distance AND the
       16-grid block-pooled distance, take min(d32, d16) per 32-cell
       (ensemble).  Distinct from R7 MRS which REPLACED the grid.

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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "innovation_breadth_20260908"
for p in (str(ROOT / "scripts"), str(SCRIPTS), str(ROOT / "src"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round11"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
S = 32
MDN_SIZES = (2048, 8192)
RNG_SEED = 0


def _fin(x):
    return None if x is None else float(x)


def _pool(x, p):
    sh = x.shape
    g = S // p
    xr = x.reshape(sh[:-3] + (g, p, g, p, sh[-1]))
    return xr.mean(axis=(-4, -2))


def _pu(x, p):
    y = _pool(x, p)
    return PB._unit(y).reshape(y.shape)


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    t0 = time.perf_counter()
    methods = {}

    d1 = PB._topk(q_flat, r_flat, 1)[:, 0]
    ctrl_maps = PB._maps(d1, n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods["C0"] = {k: _fin(v) for k, v in ctrl_m.items()}

    # ---------- MDN: reference-side feature mixup densification ----------
    rng = np.random.default_rng(RNG_SEED)
    R = r_flat.shape[0]
    for n_syn in MDN_SIZES:
        i = rng.integers(0, R, size=n_syn)
        j = rng.integers(0, R, size=n_syn)
        same = i == j
        j[same] = (j[same] + 1) % R
        syn = PB._unit(0.5 * (r_flat[i] + r_flat[j]))          # [n_syn,D] unit
        bank = np.concatenate([r_flat, syn], axis=0).astype(np.float32)
        d = PB._topk(q_flat, bank, 1)[:, 0]
        maps = PB._maps(d, n, grid)
        methods[f"MDN_{n_syn // 1024}k"] = {k: _fin(v) for k, v in
                                            PB.A1.compute_metrics(maps.astype(np.float64), masks).items()}

    # ---------- MG: 32-grid min 16-grid (coexistence ensemble) ----------
    qm, rm = (PB._unit(np.concatenate([0.5 * np.asarray(_pu(df_u, 2), dtype=np.float32).reshape(-1, df_u.shape[-1]),
                                       0.5 * np.asarray(_pu(cf_u, 2), dtype=np.float32).reshape(-1, cf_u.shape[-1])], axis=-1)),
              PB._unit(np.concatenate([0.5 * np.asarray(_pu(dr_u, 2), dtype=np.float32).reshape(-1, dr_u.shape[-1]),
                                       0.5 * np.asarray(_pu(cr_u, 2), dtype=np.float32).reshape(-1, cr_u.shape[-1])], axis=-1)))
    d16 = PB._topk(qm, rm, 1)[:, 0].reshape(n, S // 2, S // 2)
    d16_up = np.repeat(np.repeat(d16, 2, axis=1), 2, axis=2).reshape(-1)   # [n*32*32]
    dmg = np.minimum(d1, d16_up)
    maps = PB._maps(dmg, n, grid)
    methods["MG_min"] = {k: _fin(v) for k, v in
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

    cands = ("C0", "MDN_2k", "MDN_8k", "MG_min")
    agg = {}
    for c in cands:
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("MDN", ["MDN_2k", "MDN_8k"]), ("MG", ["MG_min"])):
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
    result = {"round": 11, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
