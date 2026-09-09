"""ROUND 8 — evaluation-decomposition analysis (2026-09-09 night).

Not a gate mechanism: an ANALYSIS on the frozen A1 control (C0 concat top-1)
quantifying WHERE the residual pixel error concentrates, on the defect-SIZE
axis (different aspect from R3 attribution, D6 defect-type diagnosis, and
the 2026-08-09 resolution follow-up).  Frozen features, zero test-label fit.

Per category / shot (real MPDD s0 k2/k4):
  - A1 control 448 map per test image (PB harness -> parity exact);
  - connected components (8-conn) of the 448 ground-truth mask;
  - area buckets [1-9][10-31][32-99][100-315][316-999][1000+];
  - per bucket: component count + Pixel-AP / Pixel-AUROC computed over
    (bucket anomaly pixels + ALL normal pixels) so buckets are comparable;
  - global per-class full-resolution Pixel-AP/AUROC anchor + image AUROC;
  - per-class component-area percentiles (links worst classes to size).
Analysis only; nothing is promoted.
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "innovation_breadth_20260908"
for p in (str(ROOT / "scripts"), str(SCRIPTS), str(ROOT / "src"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import probe_breadth as PB  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_breadth_20260908/round8_evaldecomp"
CATS = PB.CATS
BUCKETS = ((1, 9), (10, 31), (32, 99), (100, 315), (316, 999), (1000, None))


def _bucket_key(a: int) -> str:
    for lo, hi in BUCKETS:
        if (hi is None and a >= lo) or (hi is not None and lo <= a <= hi):
            return f"{lo}_{hi if hi else 'inf'}"
    raise ValueError(a)


def _ap_auc(pos_scores, normal_scores):
    if len(pos_scores) == 0:
        return None
    s = np.concatenate([pos_scores, normal_scores])
    y = np.concatenate([np.ones(len(pos_scores), dtype=np.int8),
                        np.zeros(len(normal_scores), dtype=np.int8)])
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return {"pixel_ap": float(average_precision_score(y, s)),
            "pixel_auroc": float(roc_auc_score(y, s)),
            "n_pos": int(len(pos_scores))}


def _percentiles(a):
    if not a:
        return None
    return {f"p{p}": float(np.percentile(a, p)) for p in (10, 50, 90, 99)}


def run_shot(shot: int) -> dict:
    per_cat = {}
    t0 = time.perf_counter()
    for cat in CATS:
        q_flat, r_flat, masks, n, grid, *_ = PB._load(cat, shot)
        dz = PB._topk(q_flat, r_flat, 1)[:, 0]
        maps = PB._maps(dz, n, grid).astype(np.float64)  # (n,448,448)
        msk = np.asarray(masks, dtype=np.uint8) > 0

        H = maps.shape[-1]
        flat_m = maps.reshape(-1)
        flat_k = msk.reshape(-1)
        normal_idx = np.flatnonzero(~flat_k)
        normal_scores = flat_m[normal_idx]

        # image-level AUROC (context only)
        img_anom = msk.reshape(n, -1).any(axis=1)
        img_max = maps.reshape(n, -1).max(axis=1)
        i_auc = None
        if 0 < int(img_anom.sum()) < n:
            i_auc = float(roc_auc_score(img_anom, img_max))

        # one pass over components, collecting per-image flat indices per bucket
        bucket_px = {k: [] for k in (f"{lo}_{hi if hi else 'inf'}" for lo, hi in BUCKETS)}
        comp_areas = {k: [] for k in bucket_px}
        n_comp = 0
        for i in range(n):
            m8 = (msk[i].astype(np.uint8)) * 255
            if not m8.any():
                continue
            nlab, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
            off = i * H * H
            for l in range(1, nlab):
                ys, xs = np.nonzero(lab == l)
                a = int(stats[l, cv2.CC_STAT_AREA])
                key = _bucket_key(a)
                bucket_px[key].append(off + ys * H + xs)
                comp_areas[key].append(a)
                n_comp += 1
        for k in bucket_px:
            bucket_px[k] = (np.concatenate(bucket_px[k]) if bucket_px[k] else np.zeros(0, dtype=np.int64))

        bucket_res = {}
        for k in bucket_px:
            pos = flat_m[bucket_px[k]]
            n_px = int(len(pos))
            bucket_res[k] = {"n_comp": len(comp_areas[k]),
                             "n_px": n_px,
                             "area_pct": _percentiles(comp_areas[k])}
            if n_px:
                bucket_res[k].update(_ap_auc(pos, normal_scores))

        glob = _ap_auc(flat_m[flat_k], normal_scores)
        per_cat[cat] = {"n_test": int(n), "n_anomaly_px": int(flat_k.sum()),
                        "image_auroc": i_auc, "n_components": int(n_comp),
                        "global_pixel_ap": glob["pixel_ap"] if glob else None,
                        "global_pixel_auroc": glob["pixel_auroc"] if glob else None,
                        "buckets": bucket_res}
    return {"shot": shot, "created_utc": datetime.now(timezone.utc).isoformat(),
            "per_cat": per_cat, "elapsed_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    args = ap.parse_args()
    res = run_shot(args.shot)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"EVALDECOMP_s0_k{args.shot}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["per_cat"], ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
