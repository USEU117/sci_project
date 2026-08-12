"""ARCHIVED — V3.2 Gate B (hierarchical rescue) is superseded by V3.3 weighted_ensemble.
See: experiments/dynamic_fusion/v3_3/overview.md
V3.2 Gate B Evaluation: test hierarchical selective rescue on MPDD cached data.
Gate B evaluates the V3.2 router end-to-end using frozen MPDD prediction
caches (AnomalyDINO + AnomalyCLIP). This is a CPU-only evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast
from industrial_ad.fusion.alignment import build_alignment_plan
from industrial_ad.fusion.contracts import BranchPrediction
from industrial_ad.fusion.v3_2_contracts import V3_2Config
from industrial_ad.fusion.v3_2_router import HierarchicalSelectiveRescueV3_2
from run_dynamic_fusion_v2_cache import load_cache, resize_maps


CANDIDATES = {
    "strict": V3_2Config(
        min_region_area=16, max_region_area_fraction=0.03, min_region_compactness=0.1,
        text_excess_threshold=2.0, pq_excess_threshold=1.5,
        min_combined_reliability=0.55, min_branch_consistency=0.3,
        max_background_risk=0.60, min_augmentation_consistency=0.4,
        min_prompt_consistency=0.4,
        max_pixel_residual=0.10, max_rescue_area_fraction=0.005,
        visual_ambiguous_low=0.5, visual_ambiguous_high=2.0,
    ),
    "balanced": V3_2Config(
        min_region_area=12, max_region_area_fraction=0.05, min_region_compactness=0.08,
        text_excess_threshold=1.5, pq_excess_threshold=1.0,
        min_combined_reliability=0.45, min_branch_consistency=0.25,
        max_background_risk=0.70, min_augmentation_consistency=0.3,
        min_prompt_consistency=0.3,
        max_pixel_residual=0.12, max_rescue_area_fraction=0.01,
        visual_ambiguous_low=0.3, visual_ambiguous_high=2.5,
    ),
    "permissive": V3_2Config(
        min_region_area=8, max_region_area_fraction=0.08, min_region_compactness=0.05,
        text_excess_threshold=1.0, pq_excess_threshold=0.8,
        min_combined_reliability=0.30, min_branch_consistency=0.2,
        max_background_risk=0.80, min_augmentation_consistency=0.2,
        min_prompt_consistency=0.2,
        max_pixel_residual=0.15, max_rescue_area_fraction=0.02,
        visual_ambiguous_low=0.2, visual_ambiguous_high=3.0,
    ),
}


def signed_evidence(values: np.ndarray, calibration: dict) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    scale = max(float(calibration.get("scale", 1.0)), 1e-12)
    center = float(calibration.get("center", 0.0))
    return np.arcsinh((raw - center) / scale)


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.2 Gate B evaluation")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT
        / "experiments"
        / "dynamic_fusion"
        / "v2"
        / "branch_cache_queue"
        / "runtime"
        / "calibrations",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--shots", type=int, nargs="+", default=[1])
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--apro-steps", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "dynamic_fusion"
        / "v3"
        / "v3_2_gate_b"
        / "report.json",
    )
    args = parser.parse_args()

    categories = sorted(
        json.loads(
            (ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )["categories"]
    )

    all_rows: list[dict] = []

    for seed in args.seeds:
        for shot in args.shots:
            pair_id = f"v2_mpdd_s{seed}_k{shot}_full_v1"
            pair_root = args.prediction_root / pair_id
            calibration_path = (
                args.calibration_root / f"v2_mpdd_s{seed}_k{shot}_branch_cache_v1.json"
            )
            if not calibration_path.exists():
                print(f"SKIP {pair_id}: no calibration")
                continue

            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration.get("status") != "passed" or calibration.get("dataset_role") != "development":
                raise ValueError(f"invalid calibration: {calibration_path}")

            for category in categories:
                visual_path = pair_root / "anomalydino_visual" / f"{category}.npz"
                text_path = pair_root / "anomalyclip_text" / f"{category}.npz"
                if not visual_path.exists() or not text_path.exists():
                    print(f"  SKIP {category}: missing cache")
                    continue

                visual = load_cache(visual_path)
                text = load_cache(text_path)
                alignment = build_alignment_plan(
                    visual["sample_ids"], text["sample_ids"]
                )
                order = alignment.candidate_order

                labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
                if not np.array_equal(labels, np.asarray(text["gt_sp"])[order]):
                    raise ValueError(f"labels differ: {pair_id}/{category}")

                stride = args.pixel_stride
                visual_maps = np.asarray(visual["anomaly_maps"], dtype=np.float32)[
                    :, ::stride, ::stride
                ]
                full_text_maps = resize_maps(
                    np.asarray(text["anomaly_maps"], dtype=np.float32)[order],
                    np.asarray(visual["anomaly_maps"]).shape[1:],
                )
                text_maps = np.asarray(full_text_maps, dtype=np.float32)[
                    :, ::stride, ::stride
                ]
                masks = np.asarray(visual["imgs_masks"], dtype=np.uint8)[
                    :, ::stride, ::stride
                ]

                cat_cal = calibration["categories"][category]

                # Signed evidence (calibrated anomaly scores)
                visual_image = signed_evidence(
                    visual["pr_sp"], cat_cal["visual"]["image"]
                )
                text_image = signed_evidence(
                    np.asarray(text["pr_sp"])[order], cat_cal["text"]["image"]
                )
                visual_pixel = signed_evidence(visual_maps, cat_cal["visual"]["pixel"])
                text_pixel = signed_evidence(text_maps, cat_cal["text"]["pixel"])

                sample_ids = np.asarray(visual["sample_ids"])

                # Normal reference stats per sample
                pixel_center = float(cat_cal["visual"]["pixel"].get("center", 0.0))
                pixel_scale = float(cat_cal["visual"]["pixel"].get("scale", 1.0))
                normal_ref_stats = {
                    str(i): {
                        "pixel_center": pixel_center,
                        "pixel_scale": pixel_scale,
                    }
                    for i in range(len(sample_ids))
                }

                pixel_labels = masks.astype(bool)
                visual_auc = float(
                    roc_auc_score(pixel_labels.reshape(-1), visual_pixel.reshape(-1))
                )
                visual_ap = float(
                    average_precision_score(
                        pixel_labels.reshape(-1), visual_pixel.reshape(-1)
                    )
                )
                visual_aupro = aupro_fast(masks, visual_pixel, args.apro_steps)
                anomaly_prevalence = float(np.mean(pixel_labels))

                for candidate_name, config in CANDIDATES.items():
                    router = HierarchicalSelectiveRescueV3_2(config)
                    result = router.fuse(
                        visual_pixel_maps=visual_pixel,
                        text_pixel_maps=text_pixel,
                        visual_image_scores=visual_image,
                        text_image_scores=text_image,
                        sample_ids=sample_ids,
                        pq_pixel_maps=None,  # No PQ data yet
                        normal_reference_stats=normal_ref_stats,
                    )

                    fused = np.asarray(result.pixel_maps)
                    fused_auc = float(
                        roc_auc_score(pixel_labels.reshape(-1), fused.reshape(-1))
                    )
                    fused_ap = float(
                        average_precision_score(
                            pixel_labels.reshape(-1), fused.reshape(-1)
                        )
                    )
                    fused_aupro = aupro_fast(masks, fused, args.apro_steps)

                    # Rescue statistics
                    rescue_mask = np.asarray(result.rescue_mask, dtype=bool)
                    rescued_count = int(np.sum(rescue_mask))
                    rescued_true = int(np.sum(rescue_mask & pixel_labels))
                    rescue_precision = (
                        rescued_true / rescued_count if rescued_count else 0.0
                    )
                    coverage = rescued_count / rescue_mask.size
                    anomaly_coverage = (
                        rescued_true / max(int(np.sum(pixel_labels)), 1)
                    )

                    # Routing decisions breakdown
                    region_count = sum(len(d) for d in result.decisions)
                    rescued_region_count = sum(
                        sum(1 for d in sample_d if d.rescue_applied)
                        for sample_d in result.decisions
                    )
                    reason_counts = {}
                    for sample_d in result.decisions:
                        for d in sample_d:
                            reason = d.reason.value
                            reason_counts[reason] = reason_counts.get(reason, 0) + 1

                    delta_ap = fused_ap - visual_ap
                    delta_auc = fused_auc - visual_auc

                    all_rows.append(
                        {
                            "candidate": candidate_name,
                            "seed": seed,
                            "shot": shot,
                            "category": category,
                            "samples": len(labels),
                            "pixel_stride": stride,
                            "visual_pixel_auroc": visual_auc,
                            "visual_pixel_ap": visual_ap,
                            "visual_aupro": visual_aupro,
                            "fused_pixel_auroc": fused_auc,
                            "fused_pixel_ap": fused_ap,
                            "fused_aupro": fused_aupro,
                            "delta_pixel_auroc": delta_auc,
                            "delta_pixel_ap": delta_ap,
                            "delta_aupro": fused_aupro - visual_aupro,
                            "rescue_pixel_count": rescued_count,
                            "rescue_true_pixel_count": rescued_true,
                            "rescue_coverage": coverage,
                            "anomaly_coverage": anomaly_coverage,
                            "rescue_precision": rescue_precision,
                            "harm_rate": 1.0 - rescue_precision if rescued_count else 0.0,
                            "anomaly_prevalence": anomaly_prevalence,
                            "rescue_precision_lift": (
                                rescue_precision / anomaly_prevalence
                                if anomaly_prevalence > 0
                                else 0.0
                            ),
                            "total_regions": region_count,
                            "rescued_regions": rescued_region_count,
                            "reason_counts": json.dumps(reason_counts),
                            "fallback_rate": float(
                                np.mean(result.visual_fallback)
                            ),
                            "router_stats": json.dumps(result.stats),
                            "test_labels_used_by_router": False,
                        }
                    )

    # Generate summary statistics
    summaries = []
    for candidate_name in CANDIDATES:
        subset = [r for r in all_rows if r["candidate"] == candidate_name]
        if not subset:
            continue

        positive_categories = 0
        for cat in categories:
            cat_rows = [r for r in subset if r["category"] == cat]
            if cat_rows:
                mean_ap = float(np.mean([r["delta_pixel_ap"] for r in cat_rows]))
                if mean_ap > 0:
                    positive_categories += 1

        summaries.append(
            {
                "candidate": candidate_name,
                "rows": len(subset),
                "mean_delta_pixel_auroc": float(
                    np.mean([r["delta_pixel_auroc"] for r in subset])
                ),
                "mean_delta_pixel_ap": float(
                    np.mean([r["delta_pixel_ap"] for r in subset])
                ),
                "mean_delta_aupro": float(
                    np.mean([r["delta_aupro"] for r in subset])
                ),
                "mean_rescue_coverage": float(
                    np.mean([r["rescue_coverage"] for r in subset])
                ),
                "mean_anomaly_coverage": float(
                    np.mean([r["anomaly_coverage"] for r in subset])
                ),
                "mean_rescue_precision": float(
                    np.mean([r["rescue_precision"] for r in subset])
                ),
                "mean_harm_rate": float(
                    np.mean([r["harm_rate"] for r in subset])
                ),
                "positive_category_count": positive_categories,
                "total_categories": len(categories),
            }
        )

    # Cross-category validation (leave-one-out)
    folds = []
    for heldout in categories:
        training = [r for r in all_rows if r["category"] != heldout]
        best_candidate = None
        best_ap = -float("inf")
        for candidate_name in CANDIDATES:
            candidate_training = [
                r for r in training if r["candidate"] == candidate_name
            ]
            if candidate_training:
                mean_ap = float(
                    np.mean([r["delta_pixel_ap"] for r in candidate_training])
                )
                if mean_ap > best_ap:
                    best_ap = mean_ap
                    best_candidate = candidate_name

        if best_candidate:
            heldout_rows = [
                r
                for r in all_rows
                if r["category"] == heldout and r["candidate"] == best_candidate
            ]
            folds.append(
                {
                    "heldout_category": heldout,
                    "selected_candidate": best_candidate,
                    "heldout_delta_pixel_auroc": float(
                        np.mean([r["delta_pixel_auroc"] for r in heldout_rows])
                    ),
                    "heldout_delta_pixel_ap": float(
                        np.mean([r["delta_pixel_ap"] for r in heldout_rows])
                    ),
                    "heldout_delta_aupro": float(
                        np.mean([r["delta_aupro"] for r in heldout_rows])
                    ),
                    "heldout_rescue_precision": float(
                        np.mean([r["rescue_precision"] for r in heldout_rows])
                    ),
                }
            )

    # Gate B pass/fail
    mean_fold_ap = float(np.mean([f["heldout_delta_pixel_ap"] for f in folds])) if folds else 0.0
    mean_fold_auc = (
        float(np.mean([f["heldout_delta_pixel_auroc"] for f in folds])) if folds else 0.0
    )
    positive_folds = sum(f["heldout_delta_pixel_ap"] > 0 for f in folds)

    # V3.2 uses stricter Gate B4 criteria:
    # - Pixel AP mean improvement
    # - At least 4/6 categories positive, min 3/6
    gate_passed = bool(
        mean_fold_ap > 0
        and mean_fold_auc >= -0.002
        and positive_folds >= 3
    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if gate_passed else "failed",
        "gate": "v3_2_gate_b",
        "run_id": "v3_2_20260812_gate_b_v1",
        "dataset": "mpdd",
        "dataset_role": "development",
        "candidate_configs": {
            name: {
                "min_region_area": cfg.min_region_area,
                "text_excess_threshold": cfg.text_excess_threshold,
                "min_combined_reliability": cfg.min_combined_reliability,
                "max_pixel_residual": cfg.max_pixel_residual,
                "max_rescue_area_fraction": cfg.max_rescue_area_fraction,
            }
            for name, cfg in CANDIDATES.items()
        },
        "rows": all_rows,
        "summaries": summaries,
        "cross_category_folds": folds,
        "gate_summary": {
            "mean_heldout_delta_pixel_auroc": mean_fold_auc,
            "mean_heldout_delta_pixel_ap": mean_fold_ap,
            "positive_heldout_category_count": positive_folds,
            "total_heldout_categories": len(categories),
            "gate_b_passed": gate_passed,
        },
        "pq_branch_available": False,
        "adaptclip_decomposed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if all_rows:
        csv_path = args.output.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)

    print(json.dumps({"status": report["status"], **report["gate_summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
