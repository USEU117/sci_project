"""Run DynamicFusion V2 on two frozen prediction caches with an audit sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import (
    BranchPrediction,
    SafeRouterV2Config,
    SafeVisualDefaultRouterV2,
    load_v2_category_calibrations,
)
from industrial_ad.fusion.alignment import build_alignment_plan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if len(result["sample_ids"]) != len(result["pr_sp"]):
        raise ValueError(f"{path}: sample_ids N does not match predictions")
    return result


def resize_maps(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape[1:] == shape:
        return values.astype(np.float32)
    return np.stack(
        [
            np.asarray(
                Image.fromarray(value.astype(np.float32), mode="F").resize(
                    (shape[1], shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            for value in values
        ]
    )


def reject_v1_output(path: Path) -> None:
    resolved = path.resolve()
    protected = [
        (ROOT / "outputs" / "dynamic_fusion" / "final_validation").resolve(),
        (ROOT / "experiments" / "dynamic_fusion" / "final_validation_audit_20260808").resolve(),
    ]
    for root in protected:
        if resolved == root or root in resolved.parents:
            raise ValueError(f"V2 output cannot overwrite protected V1 evidence: {resolved}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--visual-sidecar", type=Path)
    parser.add_argument("--text-sidecar", type=Path)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-role", choices=("development", "holdout", "retrospective"), required=True)
    parser.add_argument("--support-tolerance", type=float, default=3.0)
    parser.add_argument("--minimum-disagreement", type=float, default=0.05)
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    parser.add_argument("--concentration-tolerance", type=float, default=0.10)
    parser.add_argument("--max-image-text-weight", type=float, default=0.15)
    parser.add_argument("--max-pixel-text-weight", type=float, default=0.35)
    parser.add_argument("--no-smooth-pixel-weights", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reject_v1_output(args.output)

    visual = load_cache(args.visual_cache, args.visual_sidecar)
    text = load_cache(args.text_cache, args.text_sidecar)
    alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
    order = alignment.candidate_order
    if not np.array_equal(visual["gt_sp"], text["gt_sp"][order]):
        raise ValueError("cache labels differ after alignment")
    visual_maps = np.asarray(visual["anomaly_maps"])
    text_maps = np.asarray(text["anomaly_maps"])[order]
    if visual_maps.ndim == 4 and visual_maps.shape[1] == 1:
        visual_maps = visual_maps[:, 0]
    if text_maps.ndim == 4 and text_maps.shape[1] == 1:
        text_maps = text_maps[:, 0]
    text_maps = resize_maps(text_maps, visual_maps.shape[1:])

    payload = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    visual_calibration, text_calibration = load_v2_category_calibrations(payload, args.category)
    config = SafeRouterV2Config(
        support_tolerance=args.support_tolerance,
        minimum_disagreement=args.minimum_disagreement,
        uncertainty_margin=args.uncertainty_margin,
        concentration_tolerance=args.concentration_tolerance,
        max_image_text_weight=args.max_image_text_weight,
        max_pixel_text_weight=args.max_pixel_text_weight,
        smooth_pixel_weights=not args.no_smooth_pixel_weights,
    )
    router = SafeVisualDefaultRouterV2(visual_calibration, text_calibration, config)
    result = router.fuse(
        BranchPrediction(alignment.reference_ids, visual["pr_sp"], visual_maps),
        BranchPrediction(alignment.reference_ids, text["pr_sp"][order], text_maps),
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
        visual_image_out_of_support=result.features["visual_image_out_of_support"],
        text_image_out_of_support=result.features["text_image_out_of_support"],
        calibration_warning=result.features["calibration_warning"],
        run_id=np.asarray(args.run_id),
        schema_version=np.asarray(2),
    )
    audit = {
        "schema_version": 2,
        "status": "passed",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": args.dataset_role,
        "category": args.category,
        "visual_cache": str(args.visual_cache.resolve()),
        "visual_cache_sha256": sha256(args.visual_cache),
        "text_cache": str(args.text_cache.resolve()),
        "text_cache_sha256": sha256(args.text_cache),
        "calibration_json": str(args.calibration_json.resolve()),
        "calibration_sha256": sha256(args.calibration_json),
        "output": str(args.output.resolve()),
        "sample_count": len(result.sample_ids),
        "sample_alignment_reordered": not alignment.order_already_equal,
        "current_query_predictions_used_for_inference": True,
        "test_predictions_used_for_parameter_fit": False,
        "test_labels_used_by_router": False,
        "test_masks_used_by_router": False,
        "test_set_statistics_used_by_router": False,
        "router_config": vars(args),
    }
    audit["router_config"] = {key: str(value) if isinstance(value, Path) else value for key, value in audit["router_config"].items()}
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output), "audit": str(audit_path), "samples": len(result.sample_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
