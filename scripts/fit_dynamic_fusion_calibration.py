"""Fit per-category calibration from audited normal-reference caches only."""

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

from industrial_ad.fusion import (
    PIXEL_REFERENCE_QUANTILE,
    BranchCalibration,
    NormalReferencePrediction,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> NormalReferencePrediction:
    with np.load(path, allow_pickle=False) as data:
        result = NormalReferencePrediction(
            sample_ids=data["sample_ids"],
            source_ids=data["source_ids"],
            augmentation_ids=data["augmentation_ids"],
            image_scores=data["image_scores"],
            pixel_maps=data["pixel_maps"],
        )
    return result.validate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--visual-branch", required=True)
    parser.add_argument("--text-branch", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visual_files = {path.stem: path for path in args.visual_dir.glob("*.npz")}
    text_files = {path.stem: path for path in args.text_dir.glob("*.npz")}
    if set(visual_files) != set(text_files):
        raise SystemExit("visual and text reference categories differ")

    categories: dict[str, object] = {}
    failures: list[str] = []
    for category in sorted(visual_files):
        try:
            visual = load(visual_files[category])
            text = load(text_files[category])
            if set(visual.source_ids.tolist()) != set(text.source_ids.tolist()):
                raise ValueError("visual and text source_ids differ")
            visual_calibration = BranchCalibration.fit(
                visual.image_scores, visual.pixel_maps, args.temperature
            )
            text_calibration = BranchCalibration.fit(
                text.image_scores, text.pixel_maps, args.temperature
            )
            scales = [
                visual_calibration.image.scale,
                visual_calibration.pixel.scale,
                text_calibration.image.scale,
                text_calibration.pixel.scale,
            ]
            if any(scale <= 1e-6 for scale in scales):
                raise ValueError(
                    "degenerate reference scale; export more varied deterministic views"
                )
            categories[category] = {
                "visual": visual_calibration.to_dict(),
                "text": text_calibration.to_dict(),
                "visual_cache": str(visual_files[category].resolve()),
                "text_cache": str(text_files[category].resolve()),
                "visual_cache_sha256": sha256(visual_files[category]),
                "text_cache_sha256": sha256(text_files[category]),
                "sources": visual.source_count,
                "visual_views": len(visual.sample_ids),
                "text_views": len(text.sample_ids),
            }
        except Exception as exc:
            failures.append(f"{category}: {exc}")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "shot": args.shot,
        "visual_branch": args.visual_branch,
        "text_branch": args.text_branch,
        "method": "robust_normal_reference_median_mad_sigmoid",
        "pixel_scale_fit_statistic": f"per_view_q{PIXEL_REFERENCE_QUANTILE:.2f}",
        "temperature": args.temperature,
        "fit_data": "target_normal_reference_shots_with_deterministic_views",
        "test_predictions_used": False,
        "test_labels_used": False,
        "status": "passed" if not failures and categories else "failed",
        "failures": failures,
        "categories": categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "categories": len(categories),
                "failures": len(failures),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
