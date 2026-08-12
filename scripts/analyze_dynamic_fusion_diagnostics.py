"""Compare seed-0 dynamic fusion with declared fixed weights for K=2/K=4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODES = ["fixed_w025", "fixed_w05", "fixed_w075", "dynamic"]
FIXED = ["fixed_w025", "fixed_w05", "fixed_w075"]
METRICS = ["image_auroc", "pixel_auroc", "pixel_ap", "aupro"]


def read_metrics(base: Path, mode: str) -> pd.DataFrame:
    return pd.read_csv(base / mode / "evaluation" / "per_category.csv").set_index("category")


def route_stats(dynamic_dir: Path) -> dict:
    weights, pixel_weights, decisions = [], [], []
    for path in sorted(dynamic_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            weights.append(np.asarray(data["visual_weights"], dtype=float))
            pixel_weights.append(np.asarray(data["visual_pixel_weights"], dtype=float).reshape(len(data["visual_weights"]), -1).mean(axis=1))
            decisions.append(np.asarray(data["route_decisions"]).astype(str))
    w = np.concatenate(weights)
    pw = np.concatenate(pixel_weights)
    d = np.concatenate(decisions)
    unique, counts = np.unique(d, return_counts=True)
    return {
        "samples": int(len(w)),
        "image_weight_mean": float(w.mean()),
        "image_weight_std": float(w.std()),
        "image_weight_q10": float(np.quantile(w, 0.10)),
        "image_weight_q50": float(np.quantile(w, 0.50)),
        "image_weight_q90": float(np.quantile(w, 0.90)),
        "pixel_weight_mean": float(pw.mean()),
        "pixel_weight_std": float(pw.std()),
        "pixel_image_weight_gap_mean": float(np.mean(np.abs(w - pw))),
        "routes": dict(zip(unique.tolist(), counts.tolist())),
    }


def analyze_one(root: Path, shot: int) -> tuple[pd.DataFrame, dict]:
    base = root / "outputs" / "dynamic_fusion" / "development_matrix" / (f"20260804_visa_s0_k{shot}_calibrated_development_matrix_" + ("v3" if shot == 2 else "v1"))
    tables = {mode: read_metrics(base, mode) for mode in MODES}
    rows = []
    for category in tables["dynamic"].index:
        row = {"shot": shot, "category": category}
        fixed_values = {mode: tables[mode].loc[category] for mode in FIXED}
        for metric in METRICS:
            best_mode = max(FIXED, key=lambda mode: float(fixed_values[mode][metric]))
            best_value = float(fixed_values[best_mode][metric])
            dynamic_value = float(tables["dynamic"].loc[category, metric])
            row[f"best_fixed_{metric}"] = best_value
            row[f"best_fixed_{metric}_mode"] = best_mode
            row[f"dynamic_{metric}"] = dynamic_value
            row[f"dynamic_minus_best_{metric}"] = dynamic_value - best_value
        route = route_stats(base / "dynamic")
        row.update({
            "route_visual_count": int(route["routes"].get("visual", 0)),
            "route_text_count": int(route["routes"].get("text", 0)),
            "route_weighted_count": int(route["routes"].get("weighted_fusion", 0)),
        })
        with np.load(base / "dynamic" / f"{category}.npz", allow_pickle=False) as data:
            w = np.asarray(data["visual_weights"], dtype=float)
            pw = np.asarray(data["visual_pixel_weights"], dtype=float).reshape(len(w), -1).mean(axis=1)
            row.update({
                "image_weight_mean": float(w.mean()),
                "image_weight_std": float(w.std()),
                "image_weight_q10": float(np.quantile(w, .1)),
                "image_weight_q90": float(np.quantile(w, .9)),
                "pixel_weight_mean": float(pw.mean()),
                "pixel_weight_std": float(pw.std()),
                "pixel_image_weight_gap_mean": float(np.mean(np.abs(w - pw))),
            })
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "shot": shot,
        "base": str(base.relative_to(root)),
        "route": route_stats(base / "dynamic"),
        "category_count": int(len(frame)),
        "dynamic_below_best_fixed_count": {
            metric: int((frame[f"dynamic_minus_best_{metric}"] < 0).sum()) for metric in METRICS
        },
        "dynamic_above_best_fixed_count": {
            metric: int((frame[f"dynamic_minus_best_{metric}"] > 0).sum()) for metric in METRICS
        },
        "mean_dynamic_minus_best_fixed": {
            metric: float(frame[f"dynamic_minus_best_{metric}"].mean()) for metric in METRICS
        },
        "mean_image_pixel_weight_gap": float(frame["pixel_image_weight_gap_mean"].mean()),
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "dynamic_fusion" / "20260804_diagnostics")
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frames, summaries = [], []
    for shot in (2, 4):
        frame, summary = analyze_one(ROOT, shot)
        frame.to_csv(out / f"category_diagnostics_k{shot}.csv", index=False)
        frames.append(frame)
        summaries.append(summary)
    all_frame = pd.concat(frames, ignore_index=True)
    all_frame.to_csv(out / "category_diagnostics_k2_k4.csv", index=False)
    report = {
        "schema_version": 1,
        "scope": "visa_seed_0_development_only",
        "test_labels_used_for_router": False,
        "shots": summaries,
        "high_priority_categories": all_frame[
            (all_frame["dynamic_minus_best_image_auroc"] < -0.01)
            | (all_frame["dynamic_minus_best_aupro"] < -0.02)
        ][["shot", "category", "dynamic_minus_best_image_auroc", "dynamic_minus_best_aupro", "image_weight_mean", "pixel_weight_mean"]].to_dict(orient="records"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
