"""Seed-0 sensitivity sweep for router temperature and decision margin.

This intentionally reports image/pixel AUROC/AP and route statistics only. It
does not use test labels to choose parameters; labels are used only after each
fusion output is produced for development metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from industrial_ad.fusion import BranchPrediction, ConfidenceRouter, load_category_calibrations
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_cache import load_cache, resize_maps

CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
TEMPERATURES = (0.20, 0.35, 0.50)
MARGINS = (0.05, 0.10, 0.15)


def metrics(prediction, masks: np.ndarray) -> dict[str, float]:
    labels = prediction["labels"]
    image = prediction["image"]
    maps = prediction["maps"]
    pixel_labels = (masks > 0).reshape(-1).astype(np.uint8)
    pixel_scores = maps.reshape(-1)
    return {
        "image_auroc": float(roc_auc_score(labels, image)),
        "image_ap": float(average_precision_score(labels, image)),
        "pixel_auroc": float(roc_auc_score(pixel_labels, pixel_scores)),
        "pixel_ap": float(average_precision_score(pixel_labels, pixel_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "dynamic_fusion" / "20260805_sensitivity")
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    commands: list[str] = []
    for shot in (2, 4):
        version = "v3" if shot == 2 else "v1"
        calibration_path = ROOT / "outputs" / "dynamic_fusion" / "normal_reference_predictions" / f"20260804_visa_s0_k{shot}_real_reference_v1_q99" / "calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        visual_root = ROOT / "outputs" / "anomalydino" / "unified_matrix" / "seed_0_shot_1" / "predictions"
        text_root = ROOT / "outputs" / "anomalyclip" / "visa_all_518_cached"
        sidecar_root = ROOT / "outputs" / "dynamic_fusion" / "sidecars" / "anomalyclip_visa_518_verified"
        shot_root = out / f"k{shot}"
        shot_root.mkdir(exist_ok=True)
        for category in CATEGORIES:
            visual = load_cache(visual_root / f"{category}.npz")
            text = load_cache(text_root / f"{category}.npz", sidecar_root / f"{category}.sample_ids.npz")
            alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
            order = alignment.candidate_order
            visual_maps = np.asarray(visual["anomaly_maps"])
            if visual_maps.ndim == 4 and visual_maps.shape[1] == 1:
                visual_maps = visual_maps[:, 0]
            text_maps = np.asarray(text["anomaly_maps"])[order]
            if text_maps.ndim == 4 and text_maps.shape[1] == 1:
                text_maps = text_maps[:, 0]
            text_maps = resize_maps(text_maps, visual_maps.shape[1:])
            visual_branch = BranchPrediction(visual["sample_ids"], visual["pr_sp"], visual_maps)
            text_branch = BranchPrediction(visual["sample_ids"], text["pr_sp"][order], text_maps)
            masks = np.asarray(visual["imgs_masks"])
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            for temperature in TEMPERATURES:
                for margin in MARGINS:
                    router = ConfidenceRouter(temperature=temperature, min_weight=0.05, decision_margin=margin, visual_calibration=load_category_calibrations(calibration, category)[0], text_calibration=load_category_calibrations(calibration, category)[1])
                    result = router.fuse(visual_branch, text_branch)
                    decisions, counts = np.unique(result.decisions, return_counts=True)
                    prediction = {"labels": visual["gt_sp"], "image": result.image_scores, "maps": result.pixel_maps}
                    metric = metrics(prediction, masks)
                    iw = np.asarray(result.visual_weights, dtype=float)
                    pw = np.asarray(result.visual_pixel_weights, dtype=float).reshape(len(iw), -1).mean(axis=1)
                    row = {"shot": shot, "category": category, "temperature": temperature, "decision_margin": margin, "min_weight": 0.05, **metric, "image_weight_mean": float(iw.mean()), "image_weight_q10": float(np.quantile(iw, .1)), "image_weight_q50": float(np.quantile(iw, .5)), "image_weight_q90": float(np.quantile(iw, .9)), "pixel_weight_mean": float(pw.mean()), "image_pixel_weight_gap_mean": float(np.mean(np.abs(iw-pw))), "visual_count": int(dict(zip(decisions.tolist(), counts.tolist())).get("visual", 0)), "text_count": int(dict(zip(decisions.tolist(), counts.tolist())).get("text", 0)), "weighted_count": int(dict(zip(decisions.tolist(), counts.tolist())).get("weighted_fusion", 0))}
                    rows.append(row)
            del visual, text, visual_branch, text_branch
    frame_path = out / "sensitivity_results.csv"
    with frame_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for (shot, temperature, margin), group in __import__("pandas").DataFrame(rows).groupby(["shot", "temperature", "decision_margin"]):
        summary.append({"shot": int(shot), "temperature": float(temperature), "decision_margin": float(margin), **{metric: float(group[metric].mean()) for metric in ("image_auroc", "pixel_auroc", "pixel_ap", "image_weight_mean", "pixel_weight_mean", "image_pixel_weight_gap_mean", "visual_count", "text_count", "weighted_count")}})
    (out / "sensitivity_summary.json").write_text(json.dumps({"scope": "visa_seed_0_development_only", "min_weight": 0.05, "temperatures": TEMPERATURES, "decision_margins": MARGINS, "rows": len(rows), "summary": summary}, indent=2), encoding="utf-8")
    print(f"wrote {frame_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
