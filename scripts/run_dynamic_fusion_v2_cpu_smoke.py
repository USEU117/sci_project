"""Deterministic CPU-only smoke test and evidence report for V2 contracts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import (
    BranchPrediction,
    BranchV2Calibration,
    SafeRouterV2Config,
    SafeVisualDefaultRouterV2,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    reference_maps = np.broadcast_to(reference[:, None, None], (20, 8, 8)).copy()
    visual_calibration = BranchV2Calibration.fit(reference, reference_maps)
    text_calibration = BranchV2Calibration.fit(reference, reference_maps)
    router = SafeVisualDefaultRouterV2(
        visual_calibration,
        text_calibration,
        SafeRouterV2Config(minimum_disagreement=0.01, uncertainty_margin=0.01),
    )
    sample_ids = np.asarray(["normal", "text-assist", "out-of-support"])
    visual_scores = np.asarray([0.2, 0.5, 100.0])
    text_scores = np.asarray([0.2, 0.05, 0.1])
    visual_maps = np.broadcast_to(visual_scores[:, None, None], (3, 8, 8)).copy()
    text_maps = np.broadcast_to(text_scores[:, None, None], (3, 8, 8)).copy()
    result = router.fuse(
        BranchPrediction(sample_ids, visual_scores, visual_maps),
        BranchPrediction(sample_ids, text_scores, text_maps),
    )
    checks = {
        "finite_image_scores": bool(np.isfinite(result.image_scores).all()),
        "finite_pixel_maps": bool(np.isfinite(result.pixel_maps).all()),
        "visual_weights_bounded": bool(np.all((result.visual_weights >= 0.85) & (result.visual_weights <= 1.0))),
        "pixel_weights_bounded": bool(np.all((result.visual_pixel_weights >= 0.65) & (result.visual_pixel_weights <= 1.0))),
        "no_disagreement_defaults_visual": bool(result.visual_weights[0] == 1.0),
        "complementary_text_can_assist": bool(result.visual_weights[1] < 1.0),
        "out_of_support_falls_back": bool(result.visual_weights[2] == 1.0 and result.decisions[2] == "out_of_support"),
        "test_predictions_used": False,
        "test_labels_used": False,
        "test_masks_used": False,
        "test_set_statistics_used": False,
    }
    positive_checks = (
        "finite_image_scores",
        "finite_pixel_maps",
        "visual_weights_bounded",
        "pixel_weights_bounded",
        "no_disagreement_defaults_visual",
        "complementary_text_can_assist",
        "out_of_support_falls_back",
    )
    forbidden_use_checks = (
        "test_predictions_used",
        "test_labels_used",
        "test_masks_used",
        "test_set_statistics_used",
    )
    passed = all(checks[name] is True for name in positive_checks) and all(
        checks[name] is False for name in forbidden_use_checks
    )
    report = {
        "schema_version": 2,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "device": "cpu",
        "synthetic_only": True,
        "checks": checks,
        "decisions": result.decisions.tolist(),
        "visual_weights": result.visual_weights.tolist(),
        "visual_pixel_weight_range": [float(result.visual_pixel_weights.min()), float(result.visual_pixel_weights.max())],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
