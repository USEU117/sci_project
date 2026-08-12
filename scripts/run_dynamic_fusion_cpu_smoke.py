"""CPU-only contract smoke for stage-two baselines and reliability features."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import (  # noqa: E402
    BranchPrediction,
    ConfidenceRouter,
    FixedWeightFusion,
    augmentation_consistency,
    shot_sensitivity,
    single_branch_fusion,
)


def bounded(values: np.ndarray) -> bool:
    values = np.asarray(values)
    return bool(
        np.isfinite(values).all()
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(20260731)
    sample_ids = np.asarray([f"synthetic-{index}" for index in range(6)])
    visual = BranchPrediction(
        sample_ids=sample_ids,
        image_scores=np.linspace(0.1, 0.9, 6, dtype=np.float32),
        pixel_maps=rng.uniform(0.05, 0.95, size=(6, 4, 5)).astype(np.float32),
    )
    text = BranchPrediction(
        sample_ids=sample_ids,
        image_scores=np.linspace(0.8, 0.2, 6, dtype=np.float32),
        pixel_maps=rng.uniform(0.05, 0.95, size=(6, 4, 5)).astype(np.float32),
    )

    visual_only = single_branch_fusion(visual, "visual")
    text_only = single_branch_fusion(text, "text")
    fixed_rows: list[dict[str, object]] = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = FixedWeightFusion(weight).fuse(visual, text)
        expected = weight * visual.image_scores + (1.0 - weight) * text.image_scores
        fixed_rows.append(
            {
                "visual_weight": weight,
                "formula_match": bool(np.allclose(result.image_scores, expected)),
                "image_bounded": bounded(result.image_scores),
                "pixel_bounded": bounded(result.pixel_maps),
            }
        )

    routed = ConfidenceRouter().fuse(visual, text)
    consistency = augmentation_consistency(
        np.asarray(["ref-a"] * 3 + ["ref-b"] * 3),
        np.asarray([0.10, 0.12, 0.11, 0.70, 0.50, 0.60]),
        rng.uniform(0.05, 0.95, size=(6, 4, 5)),
    )
    shot = shot_sensitivity(
        np.stack(
            [
                visual.image_scores,
                np.clip(visual.image_scores + 0.05, 0.0, 1.0),
                np.clip(visual.image_scores - 0.05, 0.0, 1.0),
            ]
        ),
        np.stack(
            [
                visual.pixel_maps,
                np.clip(visual.pixel_maps + 0.05, 0.0, 1.0),
                np.clip(visual.pixel_maps - 0.05, 0.0, 1.0),
            ]
        ),
    )

    checks = {
        "visual_only_weights_are_one": bool(
            np.all(visual_only.visual_weights == 1.0)
        ),
        "text_only_weights_are_zero": bool(np.all(text_only.visual_weights == 0.0)),
        "fixed_formulas_match": all(row["formula_match"] for row in fixed_rows),
        "fixed_outputs_bounded": all(
            row["image_bounded"] and row["pixel_bounded"] for row in fixed_rows
        ),
        "router_outputs_bounded": bounded(routed.image_scores)
        and bounded(routed.pixel_maps),
        "pair_features_finite": all(
            np.isfinite(value).all() for value in routed.features.values()
        ),
        "augmentation_consistency_shape": consistency["pixel_view_std"].shape
        == (2, 4, 5),
        "shot_sensitivity_shape": shot["pixel_shot_std"].shape == (6, 4, 5),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "purpose": "CPU-only stage-two interface and numerical smoke",
        "synthetic_data_only": True,
        "gpu_used": False,
        "ground_truth_used": False,
        "test_predictions_used": False,
        "test_labels_used": False,
        "fixed_weight_candidates": fixed_rows,
        "router_feature_names": sorted(routed.features),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "checks": checks}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
