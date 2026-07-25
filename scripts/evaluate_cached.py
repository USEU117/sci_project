"""Fast, checkpointable evaluation for AnomalyCLIP per-class NPZ caches.

The upstream AUPRO implementation uses 200 thresholds and repeated skimage
regionprops calls over the full-resolution test set.  This evaluator defaults
to 50 thresholds and writes one row per class as it finishes, so long runs can
be resumed and audited.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage import measure
from sklearn.metrics import auc


def aupro(masks: np.ndarray, maps: np.ndarray, steps: int) -> float:
    masks = masks.squeeze(1) if masks.ndim == 4 else masks
    maps = maps.squeeze(1) if maps.ndim == 4 else maps
    binary = np.zeros_like(maps, dtype=bool)
    lo, hi = float(maps.min()), float(maps.max())
    if hi <= lo:
        return 0.0
    pros, fprs = [], []
    for th in np.linspace(lo, hi, steps, endpoint=False):
        binary[:] = maps > th
        region_scores = []
        for binary_map, mask in zip(binary, masks):
            for region in measure.regionprops(measure.label(mask)):
                region_scores.append(binary_map[region.coords[:, 0], region.coords[:, 1]].mean())
        inverse = ~masks.astype(bool)
        denom = inverse.sum()
        fprs.append(np.logical_and(inverse, binary).sum() / denom if denom else 0.0)
        pros.append(float(np.mean(region_scores)) if region_scores else 0.0)
    fprs, pros = np.asarray(fprs), np.asarray(pros)
    keep = fprs <= 0.30
    if keep.sum() < 2:
        return 0.0
    x = fprs[keep]
    x = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    return float(auc(x, pros[keep]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--apro-steps", type=int, default=50)
    args = ap.parse_args()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.cache.glob("*.npz")):
        d = np.load(path)
        row = {
            "category": path.stem,
            "image_auroc": roc_auc_score(d["gt_sp"], d["pr_sp"]),
            "image_ap": average_precision_score(d["gt_sp"], d["pr_sp"]),
            "pixel_auroc": roc_auc_score(d["imgs_masks"].ravel(), d["anomaly_maps"].ravel()),
            "pixel_ap": average_precision_score(d["imgs_masks"].ravel(), d["anomaly_maps"].ravel()),
            "aupro_approx": aupro(d["imgs_masks"], d["anomaly_maps"], args.apro_steps),
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
