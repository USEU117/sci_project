"""ARCHIVED — V3.2 Gate B V2 is superseded by V3.3 weighted_ensemble.
See: experiments/dynamic_fusion/v3_3/overview.md
V3.2 Gate B V2: test hierarchical selective rescue with decomposed AdaptCLIP branches.
Uses 4 independent branch views:
- AnomalyDINO visual (anchor/fallback, never modified downward)
- adaptclip_textual_adapter (text rescue source)
- adaptclip_pq_adapter (PQ cross-validation evidence)
- adaptclip_visual_adapter (extra cross-check)
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
        text_excess_threshold=0.8, pq_excess_threshold=0.8,
        min_combined_reliability=0.55, min_branch_consistency=0.3,
        max_background_risk=0.60, min_augmentation_consistency=0.4,
        min_prompt_consistency=0.4,
        max_pixel_residual=0.10, max_rescue_area_fraction=0.005,
        visual_ambiguous_low=0.5, visual_ambiguous_high=2.0,
        normalize_branches=True,
    ),
    "balanced": V3_2Config(
        min_region_area=12, max_region_area_fraction=0.05, min_region_compactness=0.08,
        text_excess_threshold=0.5, pq_excess_threshold=0.5,
        min_combined_reliability=0.45, min_branch_consistency=0.25,
        max_background_risk=0.70, min_augmentation_consistency=0.3,
        min_prompt_consistency=0.3,
        max_pixel_residual=0.15, max_rescue_area_fraction=0.01,
        visual_ambiguous_low=0.3, visual_ambiguous_high=2.5,
        normalize_branches=True,
    ),
    "permissive": V3_2Config(
        min_region_area=8, max_region_area_fraction=0.08, min_region_compactness=0.05,
        text_excess_threshold=0.3, pq_excess_threshold=0.3,
        min_combined_reliability=0.30, min_branch_consistency=0.2,
        max_background_risk=0.80, min_augmentation_consistency=0.2,
        min_prompt_consistency=0.2,
        max_pixel_residual=0.20, max_rescue_area_fraction=0.02,
        visual_ambiguous_low=0.2, visual_ambiguous_high=3.0,
        normalize_branches=True,
    ),
    # Ultra-permissive: remove anchor bound to test text rescue potential
    "unbounded": V3_2Config(
        min_region_area=6, max_region_area_fraction=0.12, min_region_compactness=0.03,
        text_excess_threshold=0.1, pq_excess_threshold=0.1,
        min_combined_reliability=0.15, min_branch_consistency=0.1,
        max_background_risk=0.95, min_augmentation_consistency=0.1,
        min_prompt_consistency=0.1,
        max_pixel_residual=0.25, max_rescue_area_fraction=0.05,
        visual_ambiguous_low=-100.0, visual_ambiguous_high=100.0,
        normalize_branches=True,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.2 Gate B V2 evaluation (decomposed branches)")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions",
    )
    parser.add_argument(
        "--branch-root",
        type=Path,
        default=ROOT / "outputs" / "dynamic_fusion" / "v3_2_branches" / "v3_2_mpdd_s0_k1",
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
        / "v3_2_gate_b_v2"
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
    pair_id = "v2_mpdd_s0_k1_full_v1"
    pair_root = args.prediction_root / pair_id
    calibration_path = args.calibration_root / "v2_mpdd_s0_k1_branch_cache_v1.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    branch_names = [
        "adaptclip_textual_adapter",
        "adaptclip_visual_adapter",
        "adaptclip_pq_adapter",
    ]

    for category in categories:
        # Load AnomalyDINO visual anchor
        visual_path = pair_root / "anomalydino_visual" / f"{category}.npz"
        if not visual_path.exists():
            print(f"  SKIP {category}: missing DINO cache")
            continue
        visual = load_cache(visual_path)

        # Load decomposed AdaptCLIP branches
        branches = {}
        for bname in branch_names:
            bpath = args.branch_root / bname / f"{category}.npz"
            if bpath.exists():
                branches[bname] = load_cache(bpath)
            else:
                print(f"  WARNING {category}: missing branch {bname}")

        if "adaptclip_textual_adapter" not in branches:
            print(f"  SKIP {category}: no textual_adapter branch")
            continue

        # Align all branches to AnomalyDINO
        alignment = build_alignment_plan(
            visual["sample_ids"], branches["adaptclip_textual_adapter"]["sample_ids"]
        )
        order = alignment.candidate_order

        labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
        stride = args.pixel_stride
        visual_maps = np.asarray(visual["anomaly_maps"], dtype=np.float32)[
            :, ::stride, ::stride
        ]
        masks = np.asarray(visual["imgs_masks"], dtype=np.uint8)[:, ::stride, ::stride]
        sample_ids = np.asarray(visual["sample_ids"])

        # Resize branch maps to match DINO resolution
        dino_shape = np.asarray(visual["anomaly_maps"]).shape[1:]

        branch_maps = {}
        for bname in branch_names:
            if bname not in branches:
                continue
            full = resize_maps(
                np.asarray(branches[bname]["anomaly_maps"], dtype=np.float32)[order],
                dino_shape,
            )
            branch_maps[bname] = np.asarray(full, dtype=np.float32)[
                :, ::stride, ::stride
            ]

        text_maps = branch_maps["adaptclip_textual_adapter"]
        pq_maps = branch_maps.get("adaptclip_pq_adapter")

        # Use raw anomaly maps (logit values) directly for routing.
        # Calibration is applied for metric computation only.
        pixel_labels = masks.astype(bool)
        visual_auc = float(
            roc_auc_score(pixel_labels.reshape(-1), visual_maps.reshape(-1))
        )
        visual_ap = float(
            average_precision_score(
                pixel_labels.reshape(-1), visual_maps.reshape(-1)
            )
        )
        visual_aupro = aupro_fast(masks, visual_maps, args.apro_steps)
        anomaly_prevalence = float(np.mean(pixel_labels))

        # Lightweight normal-reference stats for the router
        # Uses the existing text calibration center/scale as approximation
        cat_cal = calibration["categories"][category]
        pixel_center = float(cat_cal["visual"]["pixel"].get("center", 0.0))
        pixel_scale = float(cat_cal["visual"]["pixel"].get("scale", 1.0))
        normal_ref_stats = {
            str(i): {"pixel_center": pixel_center, "pixel_scale": max(pixel_scale, 1e-6)}
            for i in range(len(sample_ids))
        }

        for candidate_name, config in CANDIDATES.items():
            router = HierarchicalSelectiveRescueV3_2(config)
            result = router.fuse(
                visual_pixel_maps=visual_maps,
                text_pixel_maps=text_maps,
                visual_image_scores=np.zeros(len(sample_ids)),
                text_image_scores=np.zeros(len(sample_ids)),
                sample_ids=sample_ids,
                pq_pixel_maps=pq_maps,
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
                    "seed": 0,
                    "shot": 1,
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
                    "fallback_rate": float(np.mean(result.visual_fallback)),
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

    gate_passed = bool(
        mean_fold_ap > 0
        and mean_fold_auc >= -0.002
        and positive_folds >= 3
    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if gate_passed else "failed",
        "gate": "v3_2_gate_b_v2",
        "run_id": "v3_2_20260812_gate_b_v2",
        "dataset": "mpdd",
        "dataset_role": "development",
        "candidate_configs": {
            name: {
                "min_region_area": cfg.min_region_area,
                "text_excess_threshold": cfg.text_excess_threshold,
                "min_combined_reliability": cfg.min_combined_reliability,
                "max_pixel_residual": cfg.max_pixel_residual,
                "max_rescue_area_fraction": cfg.max_rescue_area_fraction,
                "visual_ambiguous_low": cfg.visual_ambiguous_low,
                "visual_ambiguous_high": cfg.visual_ambiguous_high,
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
        "branches_used": {
            "visual_anchor": "anomalydino_visual",
            "text_rescue": "adaptclip_textual_adapter",
            "pq_evidence": "adaptclip_pq_adapter",
            "extra_check": "adaptclip_visual_adapter",
        },
        "pq_branch_available": pq_maps is not None,
        "adaptclip_decomposed": True,
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
