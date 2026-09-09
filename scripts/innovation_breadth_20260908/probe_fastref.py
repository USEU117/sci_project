"""FastRef-style test-time prototype refinement probe (2026-09-09).

Faithful to FastRef (arXiv:2506.21398 / CVPR'26): per query IMAGE t, refine the
normal prototype bank toward the query statistics by EM alternating
  E: T <- Sinkhorn solution of OT_eps(p, q), where p = uniform over refined
     rows v, q = uniform over original normal prototypes B, cost
     C_ij = ||v_i - B_j||^2, entropic reg eps = rel_eps * mean(C_0).
  M: v_i <- (f_i + lam * sum_j T_ij B_j) / (1 + lam/m)   (closed form,
     uniform row marginal sum_j T_ij = 1/m).
Scoring variants on the refined set V (rows = one per query patch):
  min-to-set : s_i = min_j ||f_i - V_j||^2            (paper Algo. "min over M*_s")
  row-residual: s_i = ||f_i - V_i||^2                 (reconstruction residual)
lam pre-registered {0.1, 1.0, 10.0} (min-to-set) and 1.0 (row-residual);
outer iterations L=3, Sinkhorn iterations E=20, rel_eps=0.1 -- fixed.
INPUT: frozen fused (0.5 DINO + 0.5 CLIP, resized 32-grid) unit rows per query
image and support bank (label-free; no /test/good in bank; no label used in the
per-image optimisation). Control C0 = frozen A1 cosine top-1 (parity gate).
Gates: family lead macro dAP >= +0.01 AND worst >= -0.03 (both shots).
Runtime note: per-image loop; moderate CPU cost (matrix m x n per outer step).
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

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/fastref"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
CELLS = PB.CELLS
L_OUTER = 3
E_SINK = 20
REL_EPS = 0.1
LAMS = (0.1, 1.0, 10.0)


def _fin(x):
    return None if x is None else float(x)


def _sinkhorn(C, eps, m, n, iters=E_SINK):
    """entropy-regularised OT with uniform marginals 1/m, 1/n."""
    T = np.exp(-C / eps).astype(np.float32)
    for _ in range(iters):
        T *= (1.0 / m) / T.sum(axis=1, keepdims=True)
        T *= (1.0 / n) / T.sum(axis=0, keepdims=True)
    return T


def _refine_image(f, B, lam, L=L_OUTER):
    """f [m,D] query rows (unit), B [n,D] prototypes (unit) -> v [m,D]."""
    m = f.shape[0]
    v = np.asarray(f, dtype=np.float32)
    fB = f @ B.T  # reuse for first cost
    for _ in range(L):
        v2 = (v * v).sum(axis=1, keepdims=True).astype(np.float32)
        b2 = (B * B).sum(axis=1, keepdims=True).T.astype(np.float32)
        C = (v2 + b2 - 2.0 * (v @ B.T)).clip(min=0).astype(np.float32)
        eps = float(REL_EPS * C.mean())
        if not np.isfinite(eps) or eps <= 1e-9:
            eps = 1e-3
        T = _sinkhorn(C, eps, m, B.shape[0])
        tb = T @ B
        v = (f + np.float32(lam) * tb) / np.float32(1.0 + lam / m)
    return v.astype(np.float32)


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    D = q_flat.shape[-1]
    m = CELLS
    dist: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    dz = PB._topk(q_flat, r_flat, 1)[:, 0]
    dist["C0"] = dz
    # precompute per-image query rows for the whole category
    q_img = q_flat.reshape(n, m, D)
    q2 = (q_flat * q_flat).sum(axis=1).astype(np.float32)  # [n*m]
    V = {}
    for lam in LAMS:
        vs = np.empty_like(q_flat, dtype=np.float32)
        for i in range(n):
            vs[i * m:(i + 1) * m] = _refine_image(q_img[i], r_flat, lam)
        V[lam] = vs
        # min-to-set scoring: s_i = min_j ||f_i - V_j||^2 (within its own image)
        out = np.empty_like(dz, dtype=np.float32)
        for i in range(n):
            sl = slice(i * m, (i + 1) * m)
            f = q_img[i]
            vsq = (vs[sl] * vs[sl]).sum(axis=1)  # [m]
            d2 = q2[sl][:, None] + vsq[None, :] - 2.0 * (f @ vs[sl].T)
            out[sl] = d2.min(axis=1).clip(min=0)
        dist[f"FRF_a{int(lam * 10)}"] = out
    # row-residual at lam=1.0
    vs = V[1.0]
    rr = (q_flat - vs)
    dist["FRR_a1"] = (rr * rr).sum(axis=1).astype(np.float32)
    elapsed = time.perf_counter() - t0

    ctrl_maps = PB._maps(dist["C0"], n, grid)
    ctrl_m = PB.A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
    methods = {}
    for cand, d in dist.items():
        if cand == "C0":
            m_ = ctrl_m
        else:
            m_ = PB.A1.compute_metrics(PB._maps(d, n, grid).astype(np.float64), masks)
        methods[cand] = {k: _fin(v) for k, v in m_.items()}
    return {"category": cat, "n_test": int(n), "compute_s": _fin(elapsed),
            "control": {k: _fin(v) for k, v in ctrl_m.items()}, "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    ap.add_argument("--cats", default=None)
    args = ap.parse_args()
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else list(CATS)
    rows = [run_cat(c, args.shot) for c in cats]

    def mean(xs):
        return _fin(float(np.mean([x for x in xs if x is not None]))) if xs else None

    cands = ["C0", "FRF_a1", "FRF_a10", "FRF_a100", "FRR_a1"]
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("FRF", ["FRF_a1", "FRF_a10", "FRF_a100"]), ("FRR", ["FRR_a1"])):
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
    result = {"probe": "fastref", "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
