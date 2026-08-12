"""Summarize dynamic-fusion weights and route decisions without reading labels."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "outputs" / "dynamic_fusion" / "development_matrix" / "20260731_visa_s0_k1_calibrated_development_matrix" / "dynamic"
DEFAULT_OUT = ROOT / "experiments" / "dynamic_fusion" / "20260803_cpu_route_analysis"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows: list[dict[str, str]] = []
    for path in sorted(base.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        weights = np.asarray(data["visual_weights"], dtype=float)
        decisions = np.asarray(data["route_decisions"]).astype(str)
        pixel = np.asarray(data["visual_pixel_weights"], dtype=float)
        unique, counts = np.unique(decisions, return_counts=True)
        route_counts = dict(zip(unique.tolist(), counts.tolist()))
        rows.append(
            {
                "category": path.stem,
                "sample_count": str(len(weights)),
                "visual_weight_mean": f"{weights.mean():.8f}",
                "visual_weight_std": f"{weights.std(ddof=0):.8f}",
                "pixel_weight_mean": f"{pixel.mean():.8f}",
                "pixel_weight_std": f"{pixel.std(ddof=0):.8f}",
                "visual_count": str(route_counts.get("visual", 0)),
                "text_count": str(route_counts.get("text", 0)),
                "weighted_count": str(route_counts.get("weighted_fusion", 0)),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    report = out / "route_stats.csv"
    with report.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {report} ({len(rows)} categories)")
    all_weights = np.concatenate([np.load(p, allow_pickle=False)["visual_weights"] for p in sorted(base.glob("*.npz"))])
    all_decisions = np.concatenate([np.load(p, allow_pickle=False)["route_decisions"] for p in sorted(base.glob("*.npz"))]).astype(str)
    unique, counts = np.unique(all_decisions, return_counts=True)
    print({"samples": len(all_weights), "weight_mean": float(all_weights.mean()), "weight_std": float(all_weights.std()), "routes": dict(zip(unique.tolist(), counts.tolist()))})


if __name__ == "__main__":
    main()
