"""Quantify which V2 safety gates suppress text assistance on MPDD."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.fusion import BranchPrediction, SafeRouterV2Config, SafeVisualDefaultRouterV2, load_v2_category_calibrations
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_v2_cache import load_cache, resize_maps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    categories = sorted(json.loads((ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))["categories"])
    config = SafeRouterV2Config(support_tolerance=4.0, minimum_disagreement=0.05, uncertainty_margin=0.05, concentration_tolerance=0.10, max_image_text_weight=0.15, max_pixel_text_weight=0.25)
    rows = []
    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            pair_id = f"v2_mpdd_s{seed}_k{shot}_full_v1"
            pair = args.prediction_root / pair_id
            calibration = json.loads((args.calibration_root / f"v2_mpdd_s{seed}_k{shot}_branch_cache_v1.json").read_text(encoding="utf-8"))
            for category in categories:
                visual = load_cache(pair / "anomalydino_visual" / f"{category}.npz")
                text = load_cache(pair / "anomalyclip_text" / f"{category}.npz")
                alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
                order = alignment.candidate_order
                visual_maps = np.asarray(visual["anomaly_maps"])
                text_maps = resize_maps(np.asarray(text["anomaly_maps"])[order], visual_maps.shape[1:])
                vc, tc = load_v2_category_calibrations(calibration, category)
                result = SafeVisualDefaultRouterV2(vc, tc, config).fuse(
                    BranchPrediction(alignment.reference_ids, visual["pr_sp"], visual_maps),
                    BranchPrediction(alignment.reference_ids, text["pr_sp"][order], text_maps),
                )
                f = result.features
                image_advantage = np.asarray(f["visual_image_uncertainty"]) - np.asarray(f["text_image_uncertainty"])
                pixel_advantage = np.asarray(f["visual_pixel_uncertainty"]) - np.asarray(f["text_pixel_uncertainty"])
                spatial_safe = np.asarray(f["text_response_concentration"]) + config.concentration_tolerance >= np.asarray(f["visual_response_concentration"])
                rows.append(
                    {
                        "seed": seed,
                        "shot": shot,
                        "category": category,
                        "samples": len(result.sample_ids),
                        "visual_image_oos_rate": float(np.mean(f["visual_image_out_of_support"])),
                        "text_image_oos_rate": float(np.mean(f["text_image_out_of_support"])),
                        "image_disagreement_pass_rate": float(np.mean(np.asarray(f["image_disagreement"]) >= config.minimum_disagreement)),
                        "image_uncertainty_advantage_pass_rate": float(np.mean(image_advantage >= config.uncertainty_margin)),
                        "image_allowed_rate": float(np.mean(f["image_text_assist_allowed"])),
                        "visual_pixel_oos_rate": float(np.mean(f["visual_pixel_out_of_support"])),
                        "text_pixel_oos_rate": float(np.mean(f["text_pixel_out_of_support"])),
                        "pixel_disagreement_pass_rate": float(np.mean(np.asarray(f["pixel_disagreement"]) >= config.minimum_disagreement)),
                        "pixel_uncertainty_advantage_pass_rate": float(np.mean(pixel_advantage >= config.uncertainty_margin)),
                        "spatial_safe_sample_rate": float(np.mean(spatial_safe)),
                        "pixel_allowed_rate": float(np.mean(f["pixel_text_assist_allowed"])),
                        "mean_image_text_weight": float(np.mean(1.0 - result.visual_weights)),
                        "mean_pixel_text_weight": float(np.mean(1.0 - result.visual_pixel_weights)),
                    }
                )
    metric_fields = [key for key in rows[0] if key not in {"seed", "shot", "category", "samples"}]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in metric_fields}
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "dataset": "mpdd",
        "dataset_role": "development",
        "config": config.__dict__,
        "rows": rows,
        "summary": summary,
        "test_labels_used": False,
        "test_masks_used": False,
        "btad_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "summary": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
