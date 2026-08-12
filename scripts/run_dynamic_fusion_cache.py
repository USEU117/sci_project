"""Fuse two frozen common-NPZ caches without exposing ground truth to the router."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import (
    BranchPrediction,
    ConfidenceRouter,
    FixedWeightFusion,
    load_category_calibrations,
    single_branch_fusion,
)
from industrial_ad.fusion.alignment import build_alignment_plan


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


def load_cache(path: Path, sidecar: Path | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"gt_sp", "pr_sp", "imgs_masks", "anomaly_maps"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        result = {key: data[key] for key in required}
        if "sample_ids" in data.files:
            result["sample_ids"] = data["sample_ids"]
    if "sample_ids" not in result:
        if sidecar is None:
            raise ValueError(f"{path}: sample_ids missing and no sidecar supplied")
        with np.load(sidecar, allow_pickle=False) as data:
            if set(data.files) != {"sample_ids"}:
                raise ValueError(f"{sidecar}: expected only sample_ids")
            result["sample_ids"] = data["sample_ids"]
    if len(result["sample_ids"]) != len(result["gt_sp"]):
        raise ValueError(f"{path}: sample_ids N does not match predictions")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--visual-sidecar", type=Path)
    parser.add_argument("--text-sidecar", type=Path)
    parser.add_argument(
        "--calibration-json",
        type=Path,
        help="Passed normal-reference-only calibration report.",
    )
    parser.add_argument(
        "--category",
        help="Calibration category; defaults to the visual cache filename stem.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("dynamic", "fixed", "visual", "text"),
        default="dynamic",
    )
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument(
        "--image-temperature",
        type=float,
        help="Optional image-level router temperature; defaults to --temperature.",
    )
    parser.add_argument(
        "--pixel-temperature",
        type=float,
        help="Optional pixel-level router temperature; defaults to --temperature.",
    )
    parser.add_argument("--decision-margin", type=float, default=0.15)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--image-visual-weight", type=float, default=0.5)
    parser.add_argument("--pixel-visual-weight", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visual = load_cache(args.visual_cache, args.visual_sidecar)
    text = load_cache(args.text_cache, args.text_sidecar)
    alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
    visual_ids = alignment.reference_ids
    order = alignment.candidate_order
    if not np.array_equal(visual["gt_sp"], text["gt_sp"][order]):
        raise ValueError("Frozen cache labels differ after sample alignment")

    visual_maps = np.asarray(visual["anomaly_maps"])
    if visual_maps.ndim == 4 and visual_maps.shape[1] == 1:
        visual_maps = visual_maps[:, 0]
    text_maps = np.asarray(text["anomaly_maps"])[order]
    if text_maps.ndim == 4 and text_maps.shape[1] == 1:
        text_maps = text_maps[:, 0]
    text_maps = resize_maps(text_maps, visual_maps.shape[1:])

    calibration_path = ""
    calibration_sha256 = ""
    calibration_category = ""
    visual_calibration = None
    text_calibration = None
    if args.calibration_json is not None:
        calibration_payload = json.loads(
            args.calibration_json.read_text(encoding="utf-8")
        )
        calibration_category = args.category or args.visual_cache.stem
        visual_calibration, text_calibration = load_category_calibrations(
            calibration_payload, calibration_category
        )
        calibration_path = str(args.calibration_json.resolve())
        calibration_sha256 = sha256(args.calibration_json)

    # Ground truth is deliberately not passed to either BranchPrediction or router.
    visual_branch = BranchPrediction(visual_ids, visual["pr_sp"], visual_maps)
    text_branch = BranchPrediction(
        visual_ids, text["pr_sp"][order], text_maps
    )
    if args.fusion_mode == "dynamic":
        result = ConfidenceRouter(
            temperature=args.temperature,
            image_temperature=args.image_temperature,
            pixel_temperature=args.pixel_temperature,
            min_weight=args.min_weight,
            decision_margin=args.decision_margin,
            visual_calibration=visual_calibration,
            text_calibration=text_calibration,
        ).fuse(visual_branch, text_branch)
    elif args.fusion_mode == "fixed":
        result = FixedWeightFusion(
            image_visual_weight=args.image_visual_weight,
            pixel_visual_weight=args.pixel_visual_weight,
            visual_calibration=visual_calibration,
            text_calibration=text_calibration,
        ).fuse(visual_branch, text_branch)
    elif args.fusion_mode == "visual":
        result = single_branch_fusion(
            visual_branch, "visual", calibration=visual_calibration
        )
    else:
        result = single_branch_fusion(
            text_branch, "text", calibration=text_calibration
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
        calibration_path=np.asarray(calibration_path),
        calibration_sha256=np.asarray(calibration_sha256),
        calibration_category=np.asarray(calibration_category),
        fusion_mode=np.asarray(args.fusion_mode),
        router_temperature=np.asarray(args.temperature, dtype=np.float32),
        router_image_temperature=np.asarray(
            args.image_temperature
            if args.image_temperature is not None
            else args.temperature,
            dtype=np.float32,
        ),
        router_pixel_temperature=np.asarray(
            args.pixel_temperature
            if args.pixel_temperature is not None
            else args.temperature,
            dtype=np.float32,
        ),
        router_decision_margin=np.asarray(args.decision_margin, dtype=np.float32),
        router_min_weight=np.asarray(args.min_weight, dtype=np.float32),
        declared_image_visual_weight=np.asarray(
            args.image_visual_weight, dtype=np.float32
        ),
        declared_pixel_visual_weight=np.asarray(
            args.image_visual_weight
            if args.pixel_visual_weight is None
            else args.pixel_visual_weight,
            dtype=np.float32,
        ),
    )
    unique, counts = np.unique(result.decisions, return_counts=True)
    print(
        {
            "output": str(args.output),
            "samples": len(result.sample_ids),
            "routes": dict(zip(unique.tolist(), counts.tolist())),
            "calibration": calibration_path or None,
            "calibration_category": calibration_category or None,
            "fusion_mode": args.fusion_mode,
        }
    )


if __name__ == "__main__":
    main()
