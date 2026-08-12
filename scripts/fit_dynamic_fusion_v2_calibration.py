"""Fit leakage-safe DynamicFusion V2 calibration from normal references."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import BranchV2Calibration, NormalReferencePrediction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path: Path) -> NormalReferencePrediction:
    with np.load(path, allow_pickle=False) as data:
        value = NormalReferencePrediction(
            sample_ids=data["sample_ids"],
            source_ids=data["source_ids"],
            augmentation_ids=data["augmentation_ids"],
            image_scores=data["image_scores"],
            pixel_maps=data["pixel_maps"],
        )
    return value.validate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--visual-branch", default="anomalydino")
    parser.add_argument("--text-branch", default="anomalyclip")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-role",
        choices=("development", "holdout", "retrospective"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--minimum-scale", type=float, default=1e-6)
    parser.add_argument("--scale-floor-fraction", type=float, default=0.05)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visual_files = {path.stem: path for path in args.visual_dir.glob("*.npz")}
    text_files = {path.stem: path for path in args.text_dir.glob("*.npz")}
    if not visual_files or set(visual_files) != set(text_files):
        raise SystemExit("visual and text reference categories must be non-empty and equal")

    categories: dict[str, object] = {}
    failures: list[str] = []
    for category in sorted(visual_files):
        try:
            visual = load_reference(visual_files[category])
            text = load_reference(text_files[category])
            if set(visual.source_ids.tolist()) != set(text.source_ids.tolist()):
                raise ValueError("visual and text source_ids differ")
            fit_kwargs = {
                "minimum_scale": args.minimum_scale,
                "scale_floor_fraction": args.scale_floor_fraction,
                "lower_quantile": args.lower_quantile,
                "upper_quantile": args.upper_quantile,
            }
            visual_calibration = BranchV2Calibration.fit(
                visual.image_scores, visual.pixel_maps, **fit_kwargs
            )
            text_calibration = BranchV2Calibration.fit(
                text.image_scores, text.pixel_maps, **fit_kwargs
            )
            categories[category] = {
                "visual": visual_calibration.to_dict(),
                "text": text_calibration.to_dict(),
                "visual_cache": str(visual_files[category].resolve()),
                "text_cache": str(text_files[category].resolve()),
                "visual_cache_sha256": sha256(visual_files[category]),
                "text_cache_sha256": sha256(text_files[category]),
                "normal_source_count": visual.source_count,
                "visual_view_count": len(visual.sample_ids),
                "text_view_count": len(text.sample_ids),
            }
        except Exception as exc:
            failures.append(f"{category}: {exc}")

    report = {
        "schema_version": 2,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "seed": args.seed,
        "shot": args.shot,
        "visual_branch": args.visual_branch,
        "text_branch": args.text_branch,
        "method": "rank_preserving_arctan_with_normal_support",
        "fit_data": "target_normal_reference_shots_with_deterministic_views",
        "test_predictions_used": False,
        "test_labels_used": False,
        "test_masks_used": False,
        "test_set_statistics_used": False,
        "status": "passed" if categories and not failures else "failed",
        "failures": failures,
        "categories": categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "categories": len(categories), "failures": failures}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
