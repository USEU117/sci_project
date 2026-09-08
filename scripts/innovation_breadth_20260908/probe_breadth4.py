"""Breadth probe ROUND 4 + COMB alpha scan (2026-09-09 overnight).

Pre-registered (R4): five new label-free, frozen-feature mechanism families
  SCA  scale-only per-channel standardisation of the fused rows (no centring,
       preserves origin semantics; CS in round 1 centred AND scaled -> catastrophic,
       this isolates the scale component);
  RANK per-image rank copula ensemble of DINO-only and CLIP-only distances
       (removes cross-branch scale mismatch; different combination space vs
       distance-weighted COMB/MAP);
  IMGP image-level prototype gating: per-image mean-pool fused vector -> distance
       to nearest ref-image prototype t; score = dz + gamma*t (gamma .25/.5);
  CEN  reference-centroid distance (prototype vs exemplar; distance of each query
       patch to the mean ref cell);
  DIS  cross-branch disagreement |d_dino - d_clip| (pure) and mix with dz.
Plus COMB alpha fine scan a in {.05,.10,.15,.20,.25,.30} to confirm the round-3
+0.005 observation (development-style scan on s0; no family passed so far, so this
is confirmation only and cannot authorise seed expansion).
Gates identical to prior rounds (lead macro dAP >= +0.01 AND worst >= -0.03, both
shots; parity k2 .343706 / k4 .388328).
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

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round4"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
EPS = PB.EPS
IMGP_GAMMAS = (0.25, 0.5)
COMB_ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _fin(x):
    return None if x is None else float(x)


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    D = q_flat.shape[-1]
    dist: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    dz = PB._topk(q_flat, r_flat, 1)[:, 0]
    dist["C0"] = dz
    # branch distances (32 grid, unit)
    dd = PB._topk(PB._unit(df_u.reshape(-1, df_u.shape[-1])),
                  PB._unit(dr_u.reshape(-1, dr_u.shape[-1])), 1)[:, 0]
    dc = PB._topk(PB._unit(cf_u.reshape(-1, cf_u.shape[-1])),
                  PB._unit(cr_u.reshape(-1, cr_u.shape[-1])), 1)[:, 0]
    # SCA: divide by ref per-channel std only
    s = np.std(r_flat, axis=0, keepdims=True).astype(np.float32) + EPS
    q_s = PB._unit(q_flat / s)
    b_s = PB._unit(r_flat / s)
    dist["SCA"] = PB._topk(q_s, b_s, 1)[:, 0]
    # RANK: per-image percentile ranks of branch distances
    def ranks(x):
        xg = np.asarray(x, dtype=np.float32).reshape(n, 32 * 32)
        order = xg.argsort(axis=1)
        ranks_ = np.empty_like(xg, dtype=np.float32)
        rows = np.arange(n)[:, None]
        ranks_[rows, order] = np.arange(32 * 32)[None, :].astype(np.float32)
        return (ranks_ / (32 * 32)).astype(np.float32).reshape(-1)
    dist["RANK"] = (ranks(dd) + ranks(dc)).astype(np.float32) / 2.0
    # IMGP: image-level prototype gating
    q_img = PB._unit(q_flat.reshape(n, 32 * 32, D).mean(axis=1))
    r_img = PB._unit(r_flat.reshape(-1, 32 * 32, D).mean(axis=1))
    t_img = PB._topk(q_img, r_img, 1)[:, 0]  # [n]
    t_up = np.repeat(t_img, 32 * 32).astype(np.float32)
    for gam in IMGP_GAMMAS:
        dist[f"IMGP_G{int(gam * 100)}"] = (dz + np.float32(gam) * t_up).astype(np.float32)
    # CEN: centroid distance
    cent = PB._unit(r_flat.mean(axis=0, keepdims=True))
    qc = PB._unit(q_flat)
    dist["CEN"] = (1.0 - (qc @ cent.T)[:, 0]).astype(np.float32)
    # DIS: cross-branch disagreement
    ddis = np.abs(dd - dc).astype(np.float32)
    dist["DIS"] = ddis
    dist["DIS_MIX"] = (dz + np.float32(0.5) * ddis).astype(np.float32)
    # COMB fine scan (confirmation of round-3 +0.005 observation)
    for a in COMB_ALPHAS:
        dist[f"COMB_a{int(a * 100)}"] = ((1.0 - a) * dz + a * dc).astype(np.float32)
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

    cands = ["C0", "SCA", "RANK", "IMGP_G25", "IMGP_G50", "CEN", "DIS", "DIS_MIX"] + \
            [f"COMB_a{int(a * 100)}" for a in COMB_ALPHAS]
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("SCA", ["SCA"]), ("RANK", ["RANK"]),
                         ("IMGP", ["IMGP_G25", "IMGP_G50"]), ("CEN", ["CEN"]),
                         ("DIS", ["DIS", "DIS_MIX"]),
                         ("COMB_scan", [f"COMB_a{int(a * 100)}" for a in COMB_ALPHAS])):
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
    result = {"round": 4, "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
