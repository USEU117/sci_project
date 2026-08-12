"""Leakage-explicit MPDD metadata preparation for the V3 AdaptCLIP gate.

This module is deliberately independent from the frozen V1/V2 implementation.
It converts one frozen seed/shot manifest entry into AdaptCLIP-compatible
metadata without changing, resampling, or duplicating the selected references.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_adaptclip_mpdd_metadata(
    data_root: Path, manifest: dict[str, Any], seed: int, shot: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build metadata and an audit using only a frozen MPDD split manifest."""
    data_root = data_root.resolve()
    if manifest.get("dataset") != "mpdd":
        raise ValueError("only the frozen MPDD development manifest is allowed")
    if seed not in manifest.get("seeds", []) or shot not in manifest.get("shots", []):
        raise ValueError("seed/shot is outside the frozen MPDD manifest")

    categories = manifest.get("categories", {})
    metadata: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "test": {}}
    reference_ids: list[str] = []
    test_ids: list[str] = []
    anomalous_test_images = 0

    for category in sorted(categories):
        selected = categories[category][str(seed)][str(shot)]
        if len(selected) != shot or len(set(selected)) != shot:
            raise ValueError(f"{category}: frozen references are not {shot} unique items")
        train_rows = []
        for relative in selected:
            normalized = Path(relative).as_posix()
            if f"/{category}/train/good/" not in f"/{normalized}":
                raise ValueError(f"{category}: non-normal or cross-category reference: {relative}")
            if not (data_root / relative).is_file():
                raise FileNotFoundError(data_root / relative)
            reference_ids.append(normalized)
            train_rows.append(
                {"img_path": normalized, "mask_path": "", "cls_name": category,
                 "specie_name": "good", "anomaly": 0, "view_id": "0"}
            )
        metadata["train"][category] = train_rows

        test_rows = []
        test_root = data_root / category / "test"
        for anomaly_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            anomaly = int(anomaly_dir.name != "good")
            for image_path in sorted(
                path for path in anomaly_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ):
                mask = ""
                if anomaly:
                    mask_path = data_root / category / "ground_truth" / anomaly_dir.name / f"{image_path.stem}_mask.png"
                    if not mask_path.is_file():
                        raise FileNotFoundError(mask_path)
                    mask = _relative(mask_path, data_root)
                    anomalous_test_images += 1
                sample_id = _relative(image_path, data_root)
                test_ids.append(sample_id)
                test_rows.append(
                    {"img_path": sample_id, "mask_path": mask, "cls_name": category,
                     "specie_name": anomaly_dir.name, "anomaly": anomaly, "view_id": "0"}
                )
        metadata["test"][category] = test_rows

    audit = {
        "dataset": "mpdd",
        "dataset_role": "development",
        "seed": seed,
        "shot": shot,
        "categories": len(categories),
        "normal_references": len(reference_ids),
        "test_images": len(test_ids),
        "anomalous_test_images": anomalous_test_images,
        "reference_ids_unique": len(reference_ids) == len(set(reference_ids)),
        "test_ids_unique": len(test_ids) == len(set(test_ids)),
        "test_labels_used_by_router": False,
        "test_masks_used_by_router": False,
        "test_set_statistics_used_by_router": False,
        "btad_accessed": False,
    }
    return metadata, audit


def validate_adaptclip_prediction_payload(
    *, sample_ids: list[str], image_scores: np.ndarray, pixel_maps: np.ndarray,
    labels: np.ndarray, masks: np.ndarray
) -> dict[str, Any]:
    """Validate one category payload before a unified cache can be written.

    Labels and masks are carried only for the downstream evaluator. They are
    intentionally absent from all routing or calibration inputs.
    """
    count = len(sample_ids)
    if len(set(sample_ids)) != count:
        raise ValueError("sample_ids must be unique")
    if image_scores.shape != (count,):
        raise ValueError("image_scores shape does not match sample_ids")
    if pixel_maps.ndim != 3 or pixel_maps.shape[0] != count:
        raise ValueError("pixel_maps must have shape [N,H,W]")
    if labels.shape != (count,) or masks.shape != pixel_maps.shape:
        raise ValueError("evaluator labels/masks do not align with predictions")
    for name, values in (("image_scores", image_scores), ("pixel_maps", pixel_maps)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary evaluator-only values")
    return {
        "samples": count,
        "map_height": int(pixel_maps.shape[1]),
        "map_width": int(pixel_maps.shape[2]),
        "sample_ids_unique": True,
        "finite_predictions": True,
        "labels_evaluator_only": True,
        "test_labels_used_by_router": False,
        "test_masks_used_by_router": False,
    }


def write_adaptclip_prediction_cache(
    output: Path, *, category: str, seed: int, shot: int, sample_ids: list[str],
    image_scores: np.ndarray, pixel_maps: np.ndarray, labels: np.ndarray,
    masks: np.ndarray
) -> dict[str, Any]:
    """Write the same NPZ schema used by the frozen branch-cache evaluator."""
    audit = validate_adaptclip_prediction_payload(
        sample_ids=sample_ids, image_scores=image_scores, pixel_maps=pixel_maps,
        labels=labels, masks=masks
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gt_sp=labels.astype(np.int64, copy=False),
        pr_sp=image_scores.astype(np.float32, copy=False),
        imgs_masks=masks.astype(np.uint8, copy=False),
        anomaly_maps=pixel_maps.astype(np.float32, copy=False),
        sample_ids=np.asarray(sample_ids),
        dataset=np.asarray("mpdd"),
        dataset_role=np.asarray("development"),
        branch=np.asarray("adaptclip_text_v3"),
        category=np.asarray(category),
        seed=np.asarray(seed),
        shot=np.asarray(shot),
        score_direction=np.asarray("higher_is_more_anomalous"),
        test_labels_used_by_router=np.asarray(False),
        test_masks_used_by_router=np.asarray(False),
    )
    return audit
