"""Evaluate a small, predeclared V2 router grid on MPDD development caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from industrial_ad.fusion import BranchPrediction, SafeRouterV2Config, SafeVisualDefaultRouterV2, load_v2_category_calibrations
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_v2_cache import load_cache, resize_maps


CANDIDATES = {
    "visual_only": {"support_tolerance": 3.0, "minimum_disagreement": 0.05, "uncertainty_margin": 0.05, "concentration_tolerance": 0.10, "max_image_text_weight": 0.0, "max_pixel_text_weight": 0.0},
    "safe_default": {"support_tolerance": 3.0, "minimum_disagreement": 0.05, "uncertainty_margin": 0.05, "concentration_tolerance": 0.10, "max_image_text_weight": 0.15, "max_pixel_text_weight": 0.35},
    "pixel_only_w15": {"support_tolerance": 3.0, "minimum_disagreement": 0.05, "uncertainty_margin": 0.05, "concentration_tolerance": 0.10, "max_image_text_weight": 0.0, "max_pixel_text_weight": 0.15},
    "pixel_only_w25": {"support_tolerance": 3.0, "minimum_disagreement": 0.05, "uncertainty_margin": 0.05, "concentration_tolerance": 0.10, "max_image_text_weight": 0.0, "max_pixel_text_weight": 0.25},
    "pixel_only_w35": {"support_tolerance": 3.0, "minimum_disagreement": 0.05, "uncertainty_margin": 0.05, "concentration_tolerance": 0.10, "max_image_text_weight": 0.0, "max_pixel_text_weight": 0.35},
    "pixel_strict_w25": {"support_tolerance": 2.0, "minimum_disagreement": 0.10, "uncertainty_margin": 0.05, "concentration_tolerance": 0.10, "max_image_text_weight": 0.0, "max_pixel_text_weight": 0.25},
    "pixel_wide_w25": {"support_tolerance": 4.0, "minimum_disagreement": 0.05, "uncertainty_margin": 0.05, "concentration_tolerance": 0.10, "max_image_text_weight": 0.0, "max_pixel_text_weight": 0.25},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--candidates", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--apro-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    categories = sorted(json.loads((ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))["categories"])
    rows = []
    provenance = []
    for seed in args.seeds:
        for shot in args.shots:
            pair_id = f"v2_mpdd_s{seed}_k{shot}_full_v1"
            pair_root = args.prediction_root / pair_id
            calibration = args.calibration_root / f"v2_mpdd_s{seed}_k{shot}_branch_cache_v1.json"
            payload = json.loads(calibration.read_text(encoding="utf-8"))
            if payload.get("dataset_role") != "development":
                raise SystemExit(f"calibration role differs: {calibration}")
            provenance.append({"pair_id": pair_id, "calibration": str(calibration.resolve()), "calibration_sha256": sha256(calibration)})
            for category in categories:
                visual_path = pair_root / "anomalydino_visual" / f"{category}.npz"
                text_path = pair_root / "anomalyclip_text" / f"{category}.npz"
                visual = load_cache(visual_path)
                text = load_cache(text_path)
                alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
                order = alignment.candidate_order
                if not np.array_equal(visual["gt_sp"], text["gt_sp"][order]):
                    raise ValueError(f"labels differ: {pair_id}/{category}")
                visual_maps = np.asarray(visual["anomaly_maps"])
                text_maps = resize_maps(np.asarray(text["anomaly_maps"])[order], visual_maps.shape[1:])
                masks = np.asarray(visual["imgs_masks"])
                visual_calibration, text_calibration = load_v2_category_calibrations(payload, category)
                for name in args.candidates:
                    config = SafeRouterV2Config(**CANDIDATES[name])
                    result = SafeVisualDefaultRouterV2(visual_calibration, text_calibration, config).fuse(
                        BranchPrediction(alignment.reference_ids, visual["pr_sp"], visual_maps),
                        BranchPrediction(alignment.reference_ids, text["pr_sp"][order], text_maps),
                    )
                    labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
                    pixel_labels = (masks > 0).reshape(-1).astype(np.uint8)
                    pixel_scores = np.asarray(result.pixel_maps).reshape(-1)
                    rows.append(
                        {
                            "candidate": name,
                            "seed": seed,
                            "shot": shot,
                            "category": category,
                            "samples": len(labels),
                            "image_auroc": float(roc_auc_score(labels, result.image_scores)),
                            "image_ap": float(average_precision_score(labels, result.image_scores)),
                            "pixel_auroc": float(roc_auc_score(pixel_labels, pixel_scores)),
                            "pixel_ap": float(average_precision_score(pixel_labels, pixel_scores)),
                            "aupro": aupro_fast(masks, result.pixel_maps, args.apro_steps),
                            "mean_visual_weight": float(np.mean(result.visual_weights)),
                            "mean_visual_pixel_weight": float(np.mean(result.visual_pixel_weights)),
                            "text_route_fraction": float(np.mean(np.asarray(result.visual_weights) < 0.999999)),
                        }
                    )

    summaries = []
    for name in args.candidates:
        subset = [row for row in rows if row["candidate"] == name]
        summaries.append(
            {
                "candidate": name,
                "runs": len(subset),
                **{key: float(np.mean([row[key] for row in subset])) for key in ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro", "mean_visual_weight", "mean_visual_pixel_weight", "text_route_fraction")},
            }
        )
    baseline = next(row for row in summaries if row["candidate"] == "visual_only")
    for summary in summaries:
        for metric in ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro"):
            summary[f"delta_{metric}_vs_visual"] = summary[metric] - baseline[metric]
    eligible = [row for row in summaries if row["delta_image_auroc_vs_visual"] >= -0.002]
    selected = max(eligible, key=lambda row: (row["aupro"], row["pixel_ap"])) if eligible else None
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if selected else "failed",
        "dataset": "mpdd",
        "dataset_role": "development",
        "seeds": args.seeds,
        "shots": args.shots,
        "categories": categories,
        "candidate_configs": {name: CANDIDATES[name] for name in args.candidates},
        "selection_rule": "maximize macro AUPRO then pixel AP subject to image AUROC degradation <= 0.002 versus visual_only",
        "selected_candidate": selected["candidate"] if selected else None,
        "summaries": summaries,
        "test_predictions_used_for_router_inference": True,
        "development_labels_used_for_candidate_selection": True,
        "test_labels_used_by_router": False,
        "btad_accessed": False,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": report["status"], "selected_candidate": report["selected_candidate"], "rows": len(rows)}))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
