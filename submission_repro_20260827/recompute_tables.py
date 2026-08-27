#!/usr/bin/env python3
"""Recompute the A1 paper tables from compact patch anomaly maps (standalone, CPU).

Package-local script for submission_repro_20260827. It does NOT import anything
from the source repository. It reads:

  predictions_compact/maps/{dataset}/s{seed}_k{shot}/{category}.npz

each containing: sample_ids, concat_patch_map (float16, low-res), dino_patch_map
(float16), grid_size, map_size, stride, ref_ids, seed/shot/dataset, and hashes of
the source feature caches. GT masks are NOT packaged; they are read from the
user-provided data root (MPDD/BTAD/VisA/MVTec) aligned by sample_id.

Pipeline replayed per category:
  patch map (float16) -> dists2map (gaussian_filter + INTER_LINEAR resize to 448)
  -> stride-8 subsample -> pixel AUROC / AP / AUPRO (concat and matched DINO-only).

Usage:
  # Full CPU recompute (needs data roots for GT masks)
  python recompute_tables.py --data-root mpdd=</abs/path> --data-root mvtec=</abs/path> \
      --data-root btad=</abs/path> --data-root visa=</abs/path> [--output-dir tables]

  # Structural verification only (no masks needed)
  python recompute_tables.py --verify-only

Dependencies: numpy, opencv-python, scipy, scikit-learn, scikit-image.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import auc, average_precision_score, roc_auc_score

STRIDE = 8
MAP_SIZE = (448, 448)
# Per-category replay tolerance for float16 compact maps. Worst observed single
# deviation is mvtec s1/k4 wood dino-only AUPRO at 3.58e-3 (texture class with
# large connected anomaly regions; AUPRO's per-threshold component FPR curve is
# the most quantization-sensitive). concat metrics and pixel-AP/AUROC for both
# branches stay within ~1e-5. Config-level / dataset-level aggregates (paper
# tables) are additionally checked against the packaged references at 5e-4
# (REBUILD_TOLERANCE below).
TOLERANCE = 5e-3
REBUILD_TOLERANCE = 5e-4
PACKAGE_ROOT = Path(__file__).resolve().parent
MAPS_ROOT = PACKAGE_ROOT / "predictions_compact" / "maps"
REFERENCE_ROOT = PACKAGE_ROOT / "evidence" / "per_config"
DATASETS = ("mpdd", "btad", "visa", "mvtec")
SEEDS = (0, 1, 2)
SHOTS = (1, 2, 4)


# ----------------------------------------------------------------------
# Metrics (identical to the frozen A1 evaluator)
# ----------------------------------------------------------------------

def squeeze_maps(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [N,H,W] or [N,1,H,W], got {values.shape}")
    return values


def aupro_fast(masks: np.ndarray, maps: np.ndarray, steps: int = 200) -> float:
    masks = squeeze_maps(masks, "imgs_masks").astype(bool)
    maps = squeeze_maps(maps, "anomaly_maps")
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
        normal_fp = len(normal_scores) - np.searchsorted(normal_scores, threshold, side="right")
        fprs.append(normal_fp / len(normal_scores) if len(normal_scores) else 0.0)
        overlaps = [
            (len(scores) - np.searchsorted(scores, threshold, side="right")) / len(scores)
            for scores in region_scores
        ]
        pros.append(float(np.mean(overlaps)) if overlaps else 0.0)

    fprs_array = np.asarray(fprs, dtype=np.float64)
    pros_array = np.asarray(pros, dtype=np.float64)
    keep = fprs_array < 0.30
    if keep.sum() < 2:
        return 0.0
    selected = fprs_array[keep]
    span = selected.max() - selected.min()
    if span <= 0:
        return 0.0
    normalized = (selected - selected.min()) / span
    return float(auc(normalized, pros_array[keep]))


def dists2map(dists: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return gaussian_filter(
        cv2.resize(dists, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR), sigma=4
    )


def compute_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    maps_strided = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_strided = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_strided.ravel()
    flat_labels = (masks_strided.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_strided, maps_strided)),
    }


def maps_to_448(patch_map_f16: np.ndarray) -> np.ndarray:
    return np.stack([dists2map(m.astype(np.float32), MAP_SIZE) for m in patch_map_f16])


# ----------------------------------------------------------------------
# GT mask derivation from user data (no masks in the package)
# ----------------------------------------------------------------------

def build_visa_mask_map(data_root: Path) -> dict[str, str | None]:
    meta = json.loads((data_root / "meta.json").read_text(encoding="utf-8"))
    mapping: dict[str, str | None] = {}
    for samples in meta["test"].values():
        for entry in samples:
            img_path = entry["img_path"].replace("\\", "/")
            mapping[img_path] = entry.get("mask_path")
    return mapping


def load_mask_for_sample(dataset: str, sample_id: str, data_root: Path,
                         visa_mask_map: dict[str, str | None] | None,
                         map_size: tuple[int, int]) -> np.ndarray:
    sample_id = sample_id.replace("\\", "/")
    if dataset in ("mpdd", "mvtec"):
        if "/test/good/" in sample_id:
            return np.zeros(map_size, dtype=np.uint8)
        parts = sample_id.split("/")
        cat, _, defect, name = parts[0], parts[1], parts[2], parts[3]
        mask_rel = Path(cat) / "ground_truth" / defect / f"{Path(name).stem}_mask.png"
        mask_path = data_root / mask_rel
    elif dataset == "btad":
        if "/test/ok/" in sample_id:
            return np.zeros(map_size, dtype=np.uint8)
        parts = sample_id.split("/")
        cat, _, defect, name = parts[0], parts[1], parts[2], parts[3]
        mask_root = data_root / cat / "ground_truth" / "ko"
        candidates = sorted(p for p in mask_root.glob(f"{Path(name).stem}.*")
                            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one BTAD mask for {sample_id}, found {len(candidates)}")
        mask_path = candidates[0]
    elif dataset == "visa":
        if visa_mask_map is None:
            raise SystemExit("visa requires --data-root visa=<path> with meta.json")
        rel_mask = visa_mask_map.get(sample_id)
        if not rel_mask:
            return np.zeros(map_size, dtype=np.uint8)
        mask_path = data_root / rel_mask.replace("\\", "/")
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    if not mask_path.is_file():
        raise FileNotFoundError(f"GT mask missing: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (map_size[1], map_size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# ----------------------------------------------------------------------
# Verification and recompute
# ----------------------------------------------------------------------

def verify_payload(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["sample_ids"])
        concat = np.asarray(data["concat_patch_map"])
        dino = np.asarray(data["dino_patch_map"])
        grid = tuple(int(v) for v in data["grid_size"])
        map_size = tuple(int(v) for v in data["map_size"])
        stride = int(data["stride"])
        ref_ids = np.asarray(data["ref_ids"])
        dataset = str(np.asarray(data["dataset"]))
        seed = int(data["seed"])
        shot = int(data["shot"])
    checks = {
        "sample_ids_present": ids.size > 0,
        "map_shape_matches_grid": concat.ndim == 3 and tuple(concat.shape[1:]) == grid
        and tuple(dino.shape[1:]) == grid,
        "map_count_matches_samples": concat.shape[0] == ids.size and dino.shape[0] == ids.size,
        "no_nan_inf": bool(np.isfinite(concat).all() and np.isfinite(dino).all()),
        "ref_ids_present": ref_ids.size > 0,
        "map_size_448": map_size == MAP_SIZE,
        "stride_8": stride == STRIDE,
    }
    return {
        "category": path.stem,
        "dataset": dataset,
        "seed": seed,
        "shot": shot,
        "grid": list(grid),
        "n_samples": int(ids.size),
        "n_refs": int(ref_ids.size),
        "checks": checks,
        "passed": all(checks.values()),
    }


def recompute_category(path: Path, dataset: str, data_root: Path,
                       visa_mask_map: dict[str, str | None] | None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["sample_ids"])
        concat_f16 = np.asarray(data["concat_patch_map"])
        dino_f16 = np.asarray(data["dino_patch_map"])
    masks = np.stack([
        load_mask_for_sample(dataset, str(sid), data_root, visa_mask_map, MAP_SIZE)
        for sid in ids
    ]).astype(np.uint8)
    concat_metrics = compute_metrics(maps_to_448(concat_f16), masks)
    dino_metrics = compute_metrics(maps_to_448(dino_f16), masks)
    return {
        "category": path.stem,
        "concat": concat_metrics,
        "feature_dino_only": dino_metrics,
        "delta_ap": round(concat_metrics["pixel_ap"] - dino_metrics["pixel_ap"], 6),
    }


def compare_with_reference(recomputed: dict, reference: dict, cat: str) -> dict:
    diffs = {}
    for key in ("pixel_auroc", "pixel_ap", "pixel_aupro"):
        diffs[f"concat_{key}"] = round(
            abs(recomputed["concat"][key] - reference["concat"][key]), 6)
        diffs[f"dino_{key}"] = round(
            abs(recomputed["feature_dino_only"][key] - reference["feature_dino_only"][key]), 6)
    max_abs = max(diffs.values())
    return {"category": cat, "max_abs_metric_diff": max_abs,
            "passed": max_abs <= TOLERANCE, **diffs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute A1 tables from compact patch maps")
    parser.add_argument("--maps-root", type=Path, default=MAPS_ROOT)
    parser.add_argument("--data-root", action="append", default=[],
                        help="dataset=abs_path, repeatable (mpdd/btad/visa/mvtec)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    parser.add_argument("--reference-tables", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--verify-only", action="store_true",
                        help="structural checks only (no data root needed)")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "tables_recomputed")
    args = parser.parse_args()

    data_roots = {}
    for item in args.data_root:
        ds, _, p = item.partition("=")
        data_roots[ds] = Path(p)

    needs_masks = not args.verify_only
    missing = [ds for ds in args.datasets if needs_masks and ds not in data_roots]
    if missing:
        raise SystemExit(f"missing --data-root for: {', '.join(missing)}")

    visa_mask_maps = {}
    for ds in args.datasets:
        if ds == "visa" and (data_roots.get("visa") is not None):
            visa_mask_maps[ds] = build_visa_mask_map(data_roots["visa"])

    config_rows = []
    verify_rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for shot in args.shots:
                cfg_dir = args.maps_root / dataset / f"s{seed}_k{shot}"
                if not cfg_dir.is_dir():
                    raise SystemExit(f"missing maps config dir: {cfg_dir}")
                cat_payloads = []
                for cat_path in sorted(cfg_dir.glob("*.npz")):
                    verification = verify_payload(cat_path)
                    verify_rows.append(verification)
                    if not verification["passed"]:
                        raise SystemExit(f"verify failed: {cat_path.name}: {verification['checks']}")
                    if needs_masks:
                        row = recompute_category(cat_path, dataset, data_roots[dataset],
                                                 visa_mask_maps.get(dataset))
                        ref_path = args.reference_tables / f"{dataset}_s{seed}_k{shot}.json"
                        if ref_path.is_file():
                            reference = json.loads(ref_path.read_text(encoding="utf-8"))
                            ref_by_cat = {r["category"]: r for r in reference["per_category"]}
                            if row["category"] in ref_by_cat:
                                row["replay_check"] = compare_with_reference(row, ref_by_cat[row["category"]],
                                                                              row["category"])
                                if not row["replay_check"]["passed"]:
                                    raise SystemExit(
                                        f"replay mismatch in {cat_path.name}: {row['replay_check']}")
                        cat_payloads.append(row)
                if needs_masks:
                    mean = {
                        key: round(float(np.mean([r["concat"][key] for r in cat_payloads])), 6)
                        for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
                    }
                    mean_dino = round(float(np.mean([r["feature_dino_only"]["pixel_ap"] for r in cat_payloads])), 6)
                    cfg = {
                        "dataset": dataset, "seed": seed, "shot": shot,
                        "mean_concat_pixel_ap": mean["pixel_ap"],
                        "mean_feature_dino_only_pixel_ap": mean_dino,
                        "mean_delta_ap_vs_feature_dino": round(
                            float(np.mean([r["delta_ap"] for r in cat_payloads])), 6),
                        "positive_categories": int(sum(1 for r in cat_payloads if r["delta_ap"] > 0)),
                        "n_categories": len(cat_payloads),
                    }
                    ref_path = args.reference_tables / f"{dataset}_s{seed}_k{shot}.json"
                    if ref_path.is_file():
                        ref = json.loads(ref_path.read_text(encoding="utf-8"))
                        cfg["aggregate_check"] = {
                            "delta_concat_ap": round(
                                abs(cfg["mean_concat_pixel_ap"] - ref["mean_concat_pixel_ap"]), 6),
                            "delta_delta_ap": round(
                                abs(cfg["mean_delta_ap_vs_feature_dino"]
                                    - ref["mean_delta_ap_vs_feature_dino"]), 6),
                            "tolerance": REBUILD_TOLERANCE,
                            "passed": abs(cfg["mean_concat_pixel_ap"] - ref["mean_concat_pixel_ap"])
                            <= REBUILD_TOLERANCE
                            and abs(cfg["mean_delta_ap_vs_feature_dino"]
                                    - ref["mean_delta_ap_vs_feature_dino"]) <= REBUILD_TOLERANCE,
                        }
                        if not cfg["aggregate_check"]["passed"]:
                            raise SystemExit(
                                f"aggregate replay mismatch for {dataset} s{seed}/k{shot}: {cfg['aggregate_check']}")
                    config_rows.append(cfg)

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "verify_only" if args.verify_only else "recompute",
        "map_size": list(MAP_SIZE),
        "stride": STRIDE,
        "tolerance": TOLERANCE,
        "verify": {
            "n_categories": len(verify_rows),
            "all_passed": all(r["passed"] for r in verify_rows),
            "details": verify_rows,
        },
    }
    if needs_masks:
        summary["tables"] = {
            "configs": config_rows,
            "by_dataset": {ds: [c for c in config_rows if c["dataset"] == ds] for ds in args.datasets},
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recompute_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "mode": summary["mode"],
                      "verified_categories": len(verify_rows),
                      "verify_all_passed": summary["verify"]["all_passed"],
                      "output": str((args.output_dir / "recompute_report.json"))}))
    return 0 if summary["verify"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
