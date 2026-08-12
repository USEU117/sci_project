"""Derive temperature/margin route sensitivity from frozen T=0.20 weights.

The router's weight is a sigmoid of the uncertainty difference. Therefore the
uncertainty difference can be recovered from the frozen weight and transformed
for a new temperature without re-reading test labels or recomputing branches.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURES = (0.20, 0.35, 0.50)
MARGINS = (0.05, 0.10, 0.15)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def transform(weight: np.ndarray, temperature: float, min_weight: float = 0.05) -> np.ndarray:
    clipped = np.clip(weight.astype(float), min_weight, 1.0 - min_weight)
    logit = np.log(clipped / (1.0 - clipped))
    # The stored weights were produced at temperature 0.20.
    raw_delta = 0.20 * logit
    return np.clip(sigmoid(raw_delta / temperature), min_weight, 1.0 - min_weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "dynamic_fusion" / "20260805_sensitivity")
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for shot, version in ((2, "v3"), (4, "v1")):
        base = ROOT / "outputs" / "dynamic_fusion" / "development_matrix" / f"20260804_visa_s0_k{shot}_calibrated_development_matrix_{version}" / "dynamic"
        for path in sorted(base.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                image_weight = np.asarray(data["visual_weights"], dtype=float)
                pixel_weight = np.asarray(data["visual_pixel_weights"], dtype=float).reshape(len(image_weight), -1).mean(axis=1)
            for temperature in TEMPERATURES:
                image_new = transform(image_weight, temperature)
                pixel_new = transform(pixel_weight, temperature)
                for margin in MARGINS:
                    decisions = np.where(image_new >= 0.5 + margin, "visual", np.where(image_new <= 0.5 - margin, "text", "weighted_fusion"))
                    unique, counts = np.unique(decisions, return_counts=True)
                    count_map = dict(zip(unique.tolist(), counts.tolist()))
                    rows.append({"shot": shot, "category": path.stem, "temperature": temperature, "decision_margin": margin, "image_weight_mean": float(image_new.mean()), "image_weight_q10": float(np.quantile(image_new, .1)), "image_weight_q50": float(np.quantile(image_new, .5)), "image_weight_q90": float(np.quantile(image_new, .9)), "pixel_weight_mean": float(pixel_new.mean()), "image_pixel_weight_gap_mean": float(np.mean(np.abs(image_new-pixel_new))), "visual_count": int(count_map.get("visual", 0)), "text_count": int(count_map.get("text", 0)), "weighted_count": int(count_map.get("weighted_fusion", 0))})
    path = out / "theoretical_temperature_margin_sensitivity.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
