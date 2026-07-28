"""Method-independent evaluation for industrial anomaly predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import auc
from skimage import measure

REQUIRED_KEYS = {"gt_sp", "pr_sp", "imgs_masks", "anomaly_maps"}
METRIC_FIELDS = [
    "category",
    "sample_count",
    "image_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_auroc",
    "pixel_ap",
    "aupro",
]


def squeeze_maps(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [N,H,W] or [N,1,H,W], got {values.shape}")
    return values


def validate_binary_classes(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1, got {sorted(unique)}")
    if unique != {0, 1}:
        raise ValueError(f"{name} requires both normal and anomalous samples")
    return values.astype(np.uint8)


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    denominator = precision + recall
    values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    return float(values.max(initial=0.0))


def aupro_fast(masks: np.ndarray, maps: np.ndarray, steps: int = 200) -> float:
    masks = squeeze_maps(np.asarray(masks), "imgs_masks").astype(bool)
    maps = squeeze_maps(np.asarray(maps), "anomaly_maps")
    if masks.shape != maps.shape:
        raise ValueError(f"mask/map shape mismatch: {masks.shape} vs {maps.shape}")
    lo, hi = float(maps.min()), float(maps.max())
    if hi <= lo:
        return 0.0

    normal_scores = np.sort(maps[~masks])
    region_scores: list[np.ndarray] = []
    for mask, anomaly_map in zip(masks, maps):
        labels = measure.label(mask)
        for region_id in range(1, int(labels.max()) + 1):
            region_scores.append(np.sort(anomaly_map[labels == region_id]))

    pros: list[float] = []
    fprs: list[float] = []
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

    fprs_array = np.asarray(fprs)
    pros_array = np.asarray(pros)
    keep = fprs_array < 0.30
    if keep.sum() < 2:
        return 0.0
    selected = fprs_array[keep]
    span = selected.max() - selected.min()
    if span <= 0:
        return 0.0
    normalized = (selected - selected.min()) / span
    return float(auc(normalized, pros_array[keep]))


def evaluate_category(path: Path, apro_steps: int) -> tuple[dict, list[dict]]:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise ValueError(f"{path.name}: missing arrays {sorted(missing)}")
        labels = validate_binary_classes(data["gt_sp"], f"{path.name}/gt_sp")
        scores = np.asarray(data["pr_sp"], dtype=np.float64).reshape(-1)
        masks = squeeze_maps(data["imgs_masks"], f"{path.name}/imgs_masks")
        maps = squeeze_maps(data["anomaly_maps"], f"{path.name}/anomaly_maps")
        if not np.isfinite(scores).all() or not np.isfinite(maps).all():
            raise ValueError(f"{path.name}: predictions contain NaN or infinity")
        if len(labels) != len(scores) or len(labels) != len(masks):
            raise ValueError(
                f"{path.name}: sample count mismatch: labels={len(labels)}, "
                f"scores={len(scores)}, masks={len(masks)}, maps={len(maps)}"
            )
        if masks.shape != maps.shape:
            raise ValueError(
                f"{path.name}: mask/map shape mismatch: {masks.shape} vs {maps.shape}"
            )
        pixel_labels = (masks > 0).reshape(-1).astype(np.uint8)
        validate_binary_classes(pixel_labels, f"{path.name}/pixel labels")
        pixel_scores = maps.reshape(-1)
        sample_ids = (
            [str(value) for value in data["sample_ids"]]
            if "sample_ids" in data.files
            else [f"{path.stem}/{index:06d}" for index in range(len(labels))]
        )
        if len(sample_ids) != len(labels):
            raise ValueError(f"{path.name}: sample_ids length mismatch")

        row = {
            "category": path.stem,
            "sample_count": len(labels),
            "image_auroc": float(roc_auc_score(labels, scores)),
            "image_ap": float(average_precision_score(labels, scores)),
            "image_f1_max": f1_max(labels, scores),
            "pixel_auroc": float(roc_auc_score(pixel_labels, pixel_scores)),
            "pixel_ap": float(average_precision_score(pixel_labels, pixel_scores)),
            "aupro": aupro_fast(masks, maps, apro_steps),
        }
        images = [
            {
                "category": path.stem,
                "sample_id": sample_id,
                "label": int(label),
                "image_score": float(score),
            }
            for sample_id, label, score in zip(sample_ids, labels, scores)
        ]
        return row, images


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--apro-steps", type=int, default=200)
    args = ap.parse_args()
    paths = sorted(args.cache_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"No NPZ prediction files found in {args.cache_dir}")
    if args.apro_steps <= 1:
        raise SystemExit("--apro-steps must be greater than 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_rows: list[dict] = []
    image_rows: list[dict] = []
    for path in paths:
        print(f"evaluate {path.name}", flush=True)
        category, images = evaluate_category(path, args.apro_steps)
        category_rows.append(category)
        image_rows.extend(images)

    metric_names = METRIC_FIELDS[2:]
    summary = {
        "category": "macro_mean",
        "sample_count": sum(row["sample_count"] for row in category_rows),
        **{
            metric: float(np.mean([row[metric] for row in category_rows]))
            for metric in metric_names
        },
    }
    write_csv(
        args.output_dir / "per_image.csv",
        ["category", "sample_id", "label", "image_score"],
        image_rows,
    )
    write_csv(args.output_dir / "per_category.csv", METRIC_FIELDS, category_rows)
    write_csv(args.output_dir / "summary.csv", METRIC_FIELDS, [summary])
    report = {
        "schema_version": 1,
        "score_direction": "higher_is_more_anomalous",
        "cache_dir": str(args.cache_dir.resolve()),
        "categories": [path.stem for path in paths],
        "apro_steps": args.apro_steps,
        "aupro_max_fpr": 0.30,
        "category_count": len(category_rows),
        "sample_count": len(image_rows),
        "validation_errors": 0,
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
