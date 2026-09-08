"""Web-derived breadth probe (2026-09-09 overnight): SUB subspace-residual + COP
component-consistency scoring.

Literature-driven, frozen-feature, label-free mechanisms:
  SUB   normal-variation PCA subspace reconstruction residual as anomaly score
        (SubspaceAD, arXiv:2602.23013): fit PCA on support refs ONLY, score =
        ||q - q_hat||_2 residual. Tested on the fused concat rows (r in {64,256})
        and on DINO-only rows (r=64, mirroring the paper's DINOv2 usage). NOTE:
        distinct from our archived PCA->projected-KNN ablation: here the residual
        norm itself is the score, not a KNN distance in projected space.
  COP   unsupervised component-consistency (UniVAD C3 / PSAD compositional logic):
        k-means (k=8, deterministic) over support ref patches -> component centres;
        appearance d_app = 1 - cos(q, nearest centre); layout ratio =
        ||pos(q) - mu_c|| / r_c where mu_c/r_c are the component's expected spatial
        centre/radius across the support images.  Misplaced-but-normal-looking parts
        (parts_mismatch family) are the target. Variants: COP_lay = relu(ratio-1)
        (layout only), COP_mix = d_app + 0.5*relu(ratio-1).
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "innovation_breadth_20260908"
for p in (str(ROOT / "scripts"), str(SCRIPTS), str(ROOT / "src"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import probe_breadth as PB  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/web1"
CATS = PB.CATS
REF_AP = PB.REF_AP
GATES = PB.GATES
EPS = PB.EPS
SUB_RS = (64, 256)
COP_K = 8


def _fin(x):
    return None if x is None else float(x)


def _residual(q, pca):
    qc = np.asarray(q, dtype=np.float32)
    rec = pca.inverse_transform(pca.transform(qc))
    return np.linalg.norm(qc - rec, axis=1).astype(np.float32)


def run_cat(cat: str, shot: int) -> dict:
    q_flat, r_flat, masks, n, grid, df_u, dr_u, cf_u, cr_u = PB._load(cat, shot)
    dist: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    dz = PB._topk(q_flat, r_flat, 1)[:, 0]
    dist["C0"] = dz
    # SUB on fused concat rows
    for r in SUB_RS:
        pca = PCA(n_components=r, svd_solver="randomized", random_state=0)
        pca.fit(np.asarray(r_flat, dtype=np.float32))
        dist[f"SUB_c_r{r}"] = _residual(q_flat, pca)
    # SUB on DINO-only rows (paper alignment)
    qd = PB._unit(df_u.reshape(-1, df_u.shape[-1]))
    rd = PB._unit(dr_u.reshape(-1, dr_u.shape[-1]))
    pca_d = PCA(n_components=64, svd_solver="randomized", random_state=0)
    pca_d.fit(np.asarray(rd, dtype=np.float32))
    dist["SUB_d_r64"] = _residual(qd, pca_d)
    # COP: component clustering on fused support refs (deterministic)
    km = KMeans(n_clusters=COP_K, n_init=3, random_state=0).fit(np.asarray(r_flat, dtype=np.float32))
    centers = PB._unit(km.cluster_centers_)
    R = r_flat.shape[0]
    pos = np.arange(R) % 1024
    xs = (pos // 32).astype(np.float32)
    ys = (pos % 32).astype(np.float32)
    lab = km.labels_
    mx = np.zeros(COP_K); my = np.zeros(COP_K); rad = np.zeros(COP_K)
    for c in range(COP_K):
        idx = np.where(lab == c)[0]
        if idx.size == 0:
            rad[c] = EPS
            continue
        mx[c] = xs[idx].mean(); my[c] = ys[idx].mean()
        rad[c] = float(np.mean(np.sqrt((xs[idx] - mx[c]) ** 2 + (ys[idx] - my[c]) ** 2))) + EPS
    cosmat = np.asarray(q_flat, dtype=np.float32) @ centers.T  # [nq, K]
    am = np.argmax(cosmat, axis=1)
    d_app = (1.0 - cosmat[np.arange(cosmat.shape[0]), am]).astype(np.float32)
    px = (np.arange(q_flat.shape[0]) % 1024 // 32).astype(np.float32)
    py = (np.arange(q_flat.shape[0]) % 32).astype(np.float32)
    dx = px - mx[am]; dy = py - my[am]
    ratio = np.sqrt(dx * dx + dy * dy) / rad[am]
    lay = np.clip(ratio - 1.0, 0.0, None).astype(np.float32)
    dist["COP_lay"] = lay
    dist["COP_mix"] = (d_app + np.float32(0.5) * lay).astype(np.float32)
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

    cands = ["C0", "SUB_c_r64", "SUB_c_r256", "SUB_d_r64", "COP_lay", "COP_mix"]
    agg = {}
    for c in cands:
        ap_c = [r["methods"][c]["pixel_ap"] for r in rows]
        ap0 = [r["methods"][c]["pixel_ap"] - r["control"]["pixel_ap"] for r in rows]
        auc0 = [r["methods"][c]["pixel_auroc"] - r["control"]["pixel_auroc"] for r in rows]
        agg[c] = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap0),
                  "worst_cat_ap_delta": _fin(float(np.min(ap0))) if ap0 else None,
                  "macro_auroc_delta": mean(auc0)}
    family_lead = {}
    for fam, members in (("SUB_c", ["SUB_c_r64", "SUB_c_r256"]), ("SUB_d", ["SUB_d_r64"]),
                         ("COP", ["COP_lay", "COP_mix"])):
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
    result = {"round": "web1", "seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
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
