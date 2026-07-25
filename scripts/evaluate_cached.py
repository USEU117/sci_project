"""Fast, checkpointable evaluation for AnomalyCLIP per-class NPZ caches.

The upstream AUPRO implementation repeats connected-component extraction for
every threshold.  This evaluator extracts regions once and uses sorted pixel
scores to calculate the same threshold statistics.  It writes each completed
class immediately, so interrupted runs remain auditable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import auc
from skimage import measure


def aupro_fast(masks: np.ndarray, maps: np.ndarray, steps: int) -> float:
    """Match upstream ``cal_pro_score`` without repeated region extraction."""
    masks = (masks.squeeze(1) if masks.ndim == 4 else masks).astype(bool)
    maps = maps.squeeze(1) if maps.ndim == 4 else maps
    lo, hi = float(maps.min()), float(maps.max())
    if hi <= lo:
        return 0.0

    normal_scores = np.sort(maps[~masks])
    region_scores = []
    for mask, anomaly_map in zip(masks, maps):
        labels = measure.label(mask)
        for region_id in range(1, int(labels.max()) + 1):
            region_scores.append(np.sort(anomaly_map[labels == region_id]))

    pros, fprs = [], []
    delta = (hi - lo) / steps
    for threshold in np.arange(lo, hi, delta):
        normal_fp = len(normal_scores) - np.searchsorted(
            normal_scores, threshold, side="right"
        )
        fprs.append(normal_fp / len(normal_scores) if len(normal_scores) else 0.0)
        overlaps = [
            (len(scores) - np.searchsorted(scores, threshold, side="right"))
            / len(scores)
            for scores in region_scores
        ]
        pros.append(float(np.mean(overlaps)) if overlaps else 0.0)

    fprs, pros = np.asarray(fprs), np.asarray(pros)
    keep = fprs < 0.30
    if keep.sum() < 2:
        return 0.0
    selected_fprs = fprs[keep]
    span = selected_fprs.max() - selected_fprs.min()
    if span <= 0:
        return 0.0
    selected_fprs = (selected_fprs - selected_fprs.min()) / span
    return float(auc(selected_fprs, pros[keep]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--apro-steps", type=int, default=200)
    args = ap.parse_args()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if set(row) == {
                    "category",
                    "image_auroc",
                    "image_ap",
                    "pixel_auroc",
                    "pixel_ap",
                    "aupro",
                }:
                    rows.append(
                        {
                            "category": row["category"],
                            **{key: float(value) for key, value in row.items() if key != "category"},
                        }
                    )
    completed = {row["category"] for row in rows}
    for path in sorted(args.cache.glob("*.npz")):
        if path.stem in completed:
            print(f"skip completed category: {path.stem}", flush=True)
            continue
        d = np.load(path)
        row = {
            "category": path.stem,
            "image_auroc": roc_auc_score(d["gt_sp"], d["pr_sp"]),
            "image_ap": average_precision_score(d["gt_sp"], d["pr_sp"]),
            "pixel_auroc": roc_auc_score(d["imgs_masks"].ravel(), d["anomaly_maps"].ravel()),
            "pixel_ap": average_precision_score(d["imgs_masks"].ravel(), d["anomaly_maps"].ravel()),
            "aupro": aupro_fast(d["imgs_masks"], d["anomaly_maps"], args.apro_steps),
        }
        rows.append(row)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(rows)
        print(row, flush=True)
    if rows:
        print("mean", {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "category"})


if __name__ == "__main__":
    main()
