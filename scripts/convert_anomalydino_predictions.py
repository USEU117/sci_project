"""Convert AnomalyDINO output files to the project's common NPZ schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def mean_top_one_percent(values: np.ndarray) -> float:
    flat = np.asarray(values).reshape(-1)
    count = int(len(flat) * 0.01)
    if count == 0:
        return float(np.max(flat))
    boundary = len(flat) - count
    return float(np.mean(np.partition(flat, boundary)[boundary:]))


def load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if mask.shape != shape:
        mask = np.asarray(
            Image.fromarray(mask).resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        )
    return (mask > 0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--anomaly-dir", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    category_root = args.data_root / args.category
    test_root = category_root / "test"
    prediction_root = args.anomaly_dir / args.category / "test"
    if not test_root.is_dir() or not prediction_root.is_dir():
        raise SystemExit("Missing test data or AnomalyDINO prediction directory")

    labels: list[int] = []
    image_scores: list[float] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    sample_ids: list[str] = []

    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect = defect_dir.name
        for image_path in sorted(
            path for path in defect_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            stem = image_path.stem
            map_path = prediction_root / defect / f"{stem}.tiff"
            patch_path = prediction_root / defect / f"{stem}.npy"
            if not map_path.is_file() or not patch_path.is_file():
                raise FileNotFoundError(f"Missing predictions for {defect}/{image_path.name}")
            anomaly_map = np.asarray(tifffile.imread(map_path), dtype=np.float32)
            if anomaly_map.ndim != 2:
                raise ValueError(f"Expected 2D anomaly map, got {anomaly_map.shape}")
            mask_path = (
                None
                if defect == "good"
                else category_root / "ground_truth" / defect / f"{stem}.png"
            )
            if mask_path is not None and not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            patch_distances = np.load(patch_path)

            labels.append(0 if defect == "good" else 1)
            image_scores.append(mean_top_one_percent(patch_distances))
            masks.append(load_mask(mask_path, anomaly_map.shape))
            maps.append(anomaly_map)
            sample_ids.append(f"{args.category}/test/{defect}/{image_path.name}")

    shapes = {value.shape for value in maps}
    if len(shapes) != 1:
        raise ValueError(
            f"Common NPZ requires one map shape per category, got {sorted(shapes)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gt_sp=np.asarray(labels, dtype=np.uint8),
        pr_sp=np.asarray(image_scores, dtype=np.float32),
        imgs_masks=np.stack(masks).astype(np.uint8),
        anomaly_maps=np.stack(maps).astype(np.float32),
        sample_ids=np.asarray(sample_ids),
    )
    print(
        f"wrote {args.output} samples={len(labels)} shape={next(iter(shapes))} "
        f"normal={labels.count(0)} anomalous={labels.count(1)}"
    )


if __name__ == "__main__":
    main()
