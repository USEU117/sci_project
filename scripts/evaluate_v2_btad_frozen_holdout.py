"""Evaluate the single frozen V2 decision on BTAD without tuning or selection."""

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
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--apro-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "parameters_frozen" or freeze.get("holdout_dataset") != "btad":
        raise SystemExit("invalid BTAD parameter freeze")
    if freeze.get("selected_candidate") != "visual_only_safe_fallback":
        raise SystemExit("unexpected frozen candidate; evaluator will not choose another candidate")
    if freeze.get("holdout_metrics_allowed_after_this_freeze") is not True:
        raise SystemExit("freeze does not permit holdout evaluation")
    if args.apro_steps != int(freeze.get("final_apro_steps", -1)):
        raise SystemExit("AUPRO steps must equal the frozen protocol")

    config = SafeRouterV2Config(**freeze["router_parameters"])
    categories = sorted(json.loads((ROOT / "data" / "splits" / "btad" / "manifest.json").read_text(encoding="utf-8"))["categories"])
    rows: list[dict] = []
    provenance: list[dict] = []
    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            pair_id = f"v2_btad_s{seed}_k{shot}_full_v1"
            audit = args.audit_root / f"{pair_id}.json"
            audit_payload = json.loads(audit.read_text(encoding="utf-8"))
            if audit_payload.get("status") != "passed" or audit_payload.get("metrics_computed") is not False:
                raise SystemExit(f"prediction audit has not passed cleanly: {audit}")
            calibration = args.calibration_root / f"v2_btad_s{seed}_k{shot}_branch_cache_v1.json"
            payload = json.loads(calibration.read_text(encoding="utf-8"))
            if payload.get("dataset_role") != "holdout":
                raise SystemExit(f"calibration role differs: {calibration}")
            provenance.append({
                "pair_id": pair_id,
                "audit": str(audit.resolve()),
                "audit_sha256": sha256(audit),
                "calibration": str(calibration.resolve()),
                "calibration_sha256": sha256(calibration),
            })
            pair_root = args.prediction_root / pair_id
            for category in categories:
                visual = load_cache(pair_root / "anomalydino_visual" / f"{category}.npz")
                text = load_cache(pair_root / "anomalyclip_text" / f"{category}.npz")
                alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
                order = alignment.candidate_order
                if not np.array_equal(visual["gt_sp"], text["gt_sp"][order]):
                    raise ValueError(f"labels differ: {pair_id}/{category}")
                visual_maps = np.asarray(visual["anomaly_maps"])
                text_maps = resize_maps(np.asarray(text["anomaly_maps"])[order], visual_maps.shape[1:])
                masks = np.asarray(visual["imgs_masks"])
                visual_calibration, text_calibration = load_v2_category_calibrations(payload, category)
                result = SafeVisualDefaultRouterV2(visual_calibration, text_calibration, config).fuse(
                    BranchPrediction(alignment.reference_ids, visual["pr_sp"], visual_maps),
                    BranchPrediction(alignment.reference_ids, text["pr_sp"][order], text_maps),
                )
                labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
                pixel_labels = (masks > 0).reshape(-1).astype(np.uint8)
                pixel_scores = np.asarray(result.pixel_maps).reshape(-1)
                rows.append({
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
                })

    metrics = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro")
    summary = {
        metric: {
            "macro_mean": float(np.mean([row[metric] for row in rows])),
            "macro_std": float(np.std([row[metric] for row in rows], ddof=0)),
        }
        for metric in metrics
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "dataset": "btad",
        "dataset_role": "holdout",
        "selected_candidate": freeze["selected_candidate"],
        "router_parameters": freeze["router_parameters"],
        "parameter_freeze": str(args.freeze.resolve()),
        "parameter_freeze_sha256": sha256(args.freeze),
        "seeds": [0, 1, 2],
        "shots": [1, 2, 4],
        "categories": categories,
        "apro_steps": args.apro_steps,
        "rows": len(rows),
        "summary": summary,
        "holdout_labels_used_for_final_metrics_only": True,
        "holdout_labels_used_by_router": False,
        "holdout_results_used_for_parameter_selection": False,
        "parameters_modified_after_holdout_access": False,
        "candidate_comparison_performed_on_holdout": False,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "passed", "rows": len(rows), "summary": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
