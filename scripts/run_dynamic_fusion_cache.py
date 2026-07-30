"""Fuse two frozen common-NPZ caches without exposing ground truth to the router."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import BranchPrediction, ConfidenceRouter


def canonical_id(value: str) -> str:
    value = value.replace("\\", "/")
    if "/" in value:
        parts = value.split("/")
        return f"{parts[0]}-{parts[-2]}-{Path(parts[-1]).stem}"
    return value


def resize_maps(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape[1:] == shape:
        return values.astype(np.float32)
    resized = [
        np.asarray(
            Image.fromarray(value.astype(np.float32), mode="F").resize(
                (shape[1], shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        for value in values
    ]
    return np.stack(resized)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"gt_sp", "pr_sp", "imgs_masks", "anomaly_maps", "sample_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        return {key: data[key] for key in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visual = load_cache(args.visual_cache)
    text = load_cache(args.text_cache)
    visual_ids = np.asarray([canonical_id(str(value)) for value in visual["sample_ids"]])
    text_ids = np.asarray([canonical_id(str(value)) for value in text["sample_ids"]])
    text_index = {value: index for index, value in enumerate(text_ids.tolist())}
    if set(visual_ids.tolist()) != set(text_ids.tolist()):
        raise ValueError("Frozen cache sample sets differ")
    order = np.asarray([text_index[value] for value in visual_ids.tolist()])
    if not np.array_equal(visual["gt_sp"], text["gt_sp"][order]):
        raise ValueError("Frozen cache labels differ after sample alignment")

    visual_maps = np.asarray(visual["anomaly_maps"])
    if visual_maps.ndim == 4 and visual_maps.shape[1] == 1:
        visual_maps = visual_maps[:, 0]
    text_maps = np.asarray(text["anomaly_maps"])[order]
    if text_maps.ndim == 4 and text_maps.shape[1] == 1:
        text_maps = text_maps[:, 0]
    text_maps = resize_maps(text_maps, visual_maps.shape[1:])

    # Ground truth is deliberately not passed to either BranchPrediction or router.
    result = ConfidenceRouter().fuse(
        BranchPrediction(visual_ids, visual["pr_sp"], visual_maps),
        BranchPrediction(visual_ids, text["pr_sp"][order], text_maps),
    )
    masks = np.asarray(visual["imgs_masks"])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gt_sp=visual["gt_sp"],
        pr_sp=result.image_scores,
        imgs_masks=masks,
        anomaly_maps=result.pixel_maps,
        sample_ids=result.sample_ids,
        visual_weights=result.visual_weights,
        visual_pixel_weights=result.visual_pixel_weights,
        route_decisions=result.decisions,
    )
    unique, counts = np.unique(result.decisions, return_counts=True)
    print(
        {
            "output": str(args.output),
            "samples": len(result.sample_ids),
            "routes": dict(zip(unique.tolist(), counts.tolist())),
        }
    )


if __name__ == "__main__":
    main()
