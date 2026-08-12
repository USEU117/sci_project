"""Gate A2: can label-free V3 features identify helpful MPDD text regions?"""

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

from evaluate_unified import aupro_fast  # noqa: E402
from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from industrial_ad.fusion.contracts import BranchPrediction  # noqa: E402
from industrial_ad.fusion.v3_contracts import BranchEvidenceV3, RescueBudgetV3  # noqa: E402
from industrial_ad.fusion.v3_router import (  # noqa: E402
    HierarchicalSelectiveRescueV3,
    SelectiveRescueConfigV3,
)
from run_dynamic_fusion_v2_cache import load_cache, resize_maps  # noqa: E402


CANDIDATES = {
    "region_permissive": {
        "minimum_text_reliability": 0.20,
        "minimum_evidence_gap": 0.10,
        "pixel_visual_ambiguity": 3.0,
        "minimum_region_pixels": 4,
        "maximum_region_fraction": 0.25,
        "max_pixel_residual": 0.15,
    },
    "region_balanced": {
        "minimum_text_reliability": 0.40,
        "minimum_evidence_gap": 0.25,
        "pixel_visual_ambiguity": 1.5,
        "minimum_region_pixels": 4,
        "maximum_region_fraction": 0.15,
        "max_pixel_residual": 0.15,
    },
    "region_strict": {
        "minimum_text_reliability": 0.40,
        "minimum_evidence_gap": 0.50,
        "pixel_visual_ambiguity": 1.0,
        "minimum_region_pixels": 8,
        "maximum_region_fraction": 0.10,
        "max_pixel_residual": 0.10,
    },
    "region_wide_visual": {
        "minimum_text_reliability": 0.40,
        "minimum_evidence_gap": 0.25,
        "pixel_visual_ambiguity": 3.0,
        "minimum_region_pixels": 8,
        "maximum_region_fraction": 0.15,
        "max_pixel_residual": 0.10,
    },
}


def signed_evidence(values: np.ndarray, calibration: dict) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    scale = max(float(calibration["scale"]), 1e-12)
    return np.arcsinh((raw - float(calibration["center"])) / scale)


def reference_reliability(category_payload: dict, branch: str, level: str) -> float:
    source_count = int(category_payload.get("normal_source_count", 0))
    base = source_count / (source_count + 2.0) if source_count > 0 else 0.0
    if category_payload[branch][level].get("degenerate_reference", False):
        base *= 0.5
    return float(np.clip(base, 0.0, 1.0))


def make_config(values: dict) -> SelectiveRescueConfigV3:
    return SelectiveRescueConfigV3(
        minimum_text_reliability=values["minimum_text_reliability"],
        minimum_evidence_gap=values["minimum_evidence_gap"],
        pixel_visual_ambiguity=values["pixel_visual_ambiguity"],
        minimum_region_pixels=values["minimum_region_pixels"],
        maximum_region_fraction=values["maximum_region_fraction"],
        enable_image_rescue=False,
        enable_pixel_rescue=True,
        budget=RescueBudgetV3(max_image_residual=0.0, max_pixel_residual=values["max_pixel_residual"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--apro-steps", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "dynamic_fusion"
        / "v3"
        / "gate_a2_reliability_predictability"
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
    gate_a1 = json.loads(
        (
            ROOT
            / "experiments"
            / "dynamic_fusion"
            / "v3"
            / "gate_a1_oracle_headroom"
            / "report.json"
        ).read_text(encoding="utf-8")
    )
    oracle_lookup = {
        (row["seed"], row["shot"], row["category"]): row for row in gate_a1["rows"]
    }

    rows: list[dict] = []
    for seed in args.seeds:
        for shot in args.shots:
            pair_id = f"v2_mpdd_s{seed}_k{shot}_full_v1"
            pair_root = args.prediction_root / pair_id
            calibration_path = (
                args.calibration_root / f"v2_mpdd_s{seed}_k{shot}_branch_cache_v1.json"
            )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration.get("status") != "passed" or calibration.get("dataset_role") != "development":
                raise ValueError(f"invalid calibration status/role: {calibration_path}")
            for forbidden in (
                "test_predictions_used",
                "test_labels_used",
                "test_masks_used",
                "test_set_statistics_used",
            ):
                if calibration.get(forbidden) is not False:
                    raise ValueError(f"calibration leakage flag differs: {forbidden}")

            for category in categories:
                visual = load_cache(pair_root / "anomalydino_visual" / f"{category}.npz")
                text = load_cache(pair_root / "anomalyclip_text" / f"{category}.npz")
                alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
                order = alignment.candidate_order
                labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
                if not np.array_equal(labels, np.asarray(text["gt_sp"])[order]):
                    raise ValueError(f"labels differ: {pair_id}/{category}")

                stride = args.pixel_stride
                visual_maps = np.asarray(visual["anomaly_maps"], dtype=np.float32)[:, ::stride, ::stride]
                full_text_maps = resize_maps(
                    np.asarray(text["anomaly_maps"], dtype=np.float32)[order],
                    np.asarray(visual["anomaly_maps"]).shape[1:],
                )
                text_maps = np.asarray(full_text_maps, dtype=np.float32)[:, ::stride, ::stride]
                masks = np.asarray(visual["imgs_masks"], dtype=np.uint8)[:, ::stride, ::stride]
                category_calibration = calibration["categories"][category]

                visual_image = signed_evidence(
                    visual["pr_sp"], category_calibration["visual"]["image"]
                )
                text_image = signed_evidence(
                    np.asarray(text["pr_sp"])[order], category_calibration["text"]["image"]
                )
                visual_pixel = signed_evidence(
                    visual_maps, category_calibration["visual"]["pixel"]
                )
                text_pixel = signed_evidence(
                    text_maps, category_calibration["text"]["pixel"]
                )
                visual_image_rel = reference_reliability(category_calibration, "visual", "image")
                text_image_rel = reference_reliability(category_calibration, "text", "image")
                visual_pixel_rel = reference_reliability(category_calibration, "visual", "pixel")
                text_pixel_rel = reference_reliability(category_calibration, "text", "pixel")

                visual_evidence = BranchEvidenceV3(
                    prediction=BranchPrediction(alignment.reference_ids, visual_image, visual_pixel),
                    image_anomaly_evidence=visual_image,
                    pixel_anomaly_evidence=visual_pixel,
                    image_reliability=np.full(len(labels), visual_image_rel),
                    pixel_reliability=np.full(visual_pixel.shape, visual_pixel_rel),
                )
                text_evidence = BranchEvidenceV3(
                    prediction=BranchPrediction(alignment.reference_ids, text_image, text_pixel),
                    image_anomaly_evidence=text_image,
                    pixel_anomaly_evidence=text_pixel,
                    image_reliability=np.full(len(labels), text_image_rel),
                    pixel_reliability=np.full(text_pixel.shape, text_pixel_rel),
                )

                pixel_labels = masks.astype(bool)
                visual_auc = float(roc_auc_score(pixel_labels.reshape(-1), visual_pixel.reshape(-1)))
                visual_ap = float(average_precision_score(pixel_labels.reshape(-1), visual_pixel.reshape(-1)))
                visual_aupro = aupro_fast(masks, visual_pixel, args.apro_steps)
                anomaly_prevalence = float(np.mean(pixel_labels))
                oracle_row = oracle_lookup[(seed, shot, category)]

                for candidate, values in CANDIDATES.items():
                    result = HierarchicalSelectiveRescueV3(make_config(values)).fuse(
                        visual_evidence, text_evidence
                    )
                    fused = np.asarray(result.pixel_maps)
                    fused_auc = float(roc_auc_score(pixel_labels.reshape(-1), fused.reshape(-1)))
                    fused_ap = float(average_precision_score(pixel_labels.reshape(-1), fused.reshape(-1)))
                    fused_aupro = aupro_fast(masks, fused, args.apro_steps)
                    rescued = np.asarray(result.pixel_rescue_allowed, dtype=bool)
                    rescued_count = int(np.sum(rescued))
                    rescued_true = int(np.sum(rescued & pixel_labels))
                    rescue_precision = rescued_true / rescued_count if rescued_count else 0.0
                    coverage = rescued_count / rescued.size
                    anomaly_coverage = rescued_true / max(int(np.sum(pixel_labels)), 1)
                    oracle_gain = float(oracle_row["oracle_pixel_delta_ap_vs_visual"])
                    realized_gain = fused_ap - visual_ap
                    rows.append(
                        {
                            "candidate": candidate,
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
                            "delta_pixel_auroc": fused_auc - visual_auc,
                            "delta_pixel_ap": realized_gain,
                            "delta_aupro": fused_aupro - visual_aupro,
                            "oracle_pixel_ap_gain": oracle_gain,
                            "routing_efficiency_ap": realized_gain / oracle_gain if oracle_gain > 0 else 0.0,
                            "rescue_pixel_count": rescued_count,
                            "rescue_true_pixel_count": rescued_true,
                            "rescue_coverage": coverage,
                            "anomaly_coverage": anomaly_coverage,
                            "rescue_precision": rescue_precision,
                            "harm_rate": 1.0 - rescue_precision if rescued_count else 0.0,
                            "anomaly_prevalence": anomaly_prevalence,
                            "rescue_precision_lift": rescue_precision / anomaly_prevalence if anomaly_prevalence > 0 else 0.0,
                            "image_rescue_enabled": False,
                            "test_labels_used_by_router": False,
                        }
                    )

    summaries = []
    for candidate in CANDIDATES:
        subset = [row for row in rows if row["candidate"] == candidate]
        positive_categories = 0
        for category in categories:
            category_rows = [row for row in subset if row["category"] == category]
            positive_categories += float(np.mean([row["delta_pixel_ap"] for row in category_rows])) > 0
        positive_seeds = 0
        for seed in args.seeds:
            seed_rows = [row for row in subset if row["seed"] == seed]
            positive_seeds += float(np.mean([row["delta_pixel_ap"] for row in seed_rows])) > 0
        summaries.append(
            {
                "candidate": candidate,
                "rows": len(subset),
                "mean_delta_pixel_auroc": float(np.mean([row["delta_pixel_auroc"] for row in subset])),
                "mean_delta_pixel_ap": float(np.mean([row["delta_pixel_ap"] for row in subset])),
                "mean_delta_aupro": float(np.mean([row["delta_aupro"] for row in subset])),
                "mean_routing_efficiency_ap": float(np.mean([row["routing_efficiency_ap"] for row in subset])),
                "mean_rescue_coverage": float(np.mean([row["rescue_coverage"] for row in subset])),
                "mean_anomaly_coverage": float(np.mean([row["anomaly_coverage"] for row in subset])),
                "mean_rescue_precision": float(np.mean([row["rescue_precision"] for row in subset])),
                "mean_harm_rate": float(np.mean([row["harm_rate"] for row in subset])),
                "mean_rescue_precision_lift": float(np.mean([row["rescue_precision_lift"] for row in subset])),
                "positive_seed_count": positive_seeds,
                "positive_category_count": positive_categories,
            }
        )

    folds = []
    for heldout in categories:
        training = [row for row in rows if row["category"] != heldout]
        candidate_training = []
        for candidate in CANDIDATES:
            subset = [row for row in training if row["candidate"] == candidate]
            candidate_training.append(
                {
                    "candidate": candidate,
                    "delta_pixel_ap": float(np.mean([row["delta_pixel_ap"] for row in subset])),
                    "delta_pixel_auroc": float(np.mean([row["delta_pixel_auroc"] for row in subset])),
                }
            )
        eligible = [row for row in candidate_training if row["delta_pixel_auroc"] >= -0.002]
        selected = max(eligible, key=lambda row: row["delta_pixel_ap"]) if eligible else max(
            candidate_training, key=lambda row: row["delta_pixel_auroc"]
        )
        heldout_rows = [
            row for row in rows if row["category"] == heldout and row["candidate"] == selected["candidate"]
        ]
        folds.append(
            {
                "heldout_category": heldout,
                "selected_candidate": selected["candidate"],
                "heldout_delta_pixel_auroc": float(np.mean([row["delta_pixel_auroc"] for row in heldout_rows])),
                "heldout_delta_pixel_ap": float(np.mean([row["delta_pixel_ap"] for row in heldout_rows])),
                "heldout_delta_aupro": float(np.mean([row["delta_aupro"] for row in heldout_rows])),
                "heldout_rescue_precision_lift": float(
                    np.mean([row["rescue_precision_lift"] for row in heldout_rows])
                ),
            }
        )

    mean_fold_ap = float(np.mean([row["heldout_delta_pixel_ap"] for row in folds]))
    mean_fold_auc = float(np.mean([row["heldout_delta_pixel_auroc"] for row in folds]))
    positive_folds = sum(row["heldout_delta_pixel_ap"] > 0 for row in folds)
    gate_passed = bool(mean_fold_ap > 0 and mean_fold_auc >= -0.002 and positive_folds >= 3)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "gate": "v3_gate_a2_reliability_predictability",
        "run_id": "v3_20260812_overnight_gate_a_v1",
        "dataset": "mpdd",
        "dataset_role": "development",
        "candidate_configs": CANDIDATES,
        "pixel_stride": args.pixel_stride,
        "apro_steps": args.apro_steps,
        "diagnostic_metrics_not_paper_metrics": True,
        "test_labels_used_by_router": False,
        "test_masks_used_by_router": False,
        "development_labels_used_for_offline_selection": True,
        "test_set_statistics_used_by_router": False,
        "btad_accessed": False,
        "rows": rows,
        "summaries": summaries,
        "cross_category_folds": folds,
        "gate_summary": {
            "mean_heldout_delta_pixel_auroc": mean_fold_auc,
            "mean_heldout_delta_pixel_ap": mean_fold_ap,
            "positive_heldout_category_count": positive_folds,
            "gate_a2_passed": gate_passed,
        },
        "decision_rule": "mean heldout pixel AP gain > 0, mean heldout pixel AUROC gain >= -0.002, and positive AP gain on at least 3/6 heldout categories",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": report["status"], **report["gate_summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

