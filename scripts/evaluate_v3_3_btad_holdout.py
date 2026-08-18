"""V3.3 BTAD holdout validation — frozen weighted_ensemble (DINO=0.60, CLIP=0.40).

Runs ONLY the frozen V3.3 configuration. No candidate search / leave-one-out,
because BTAD is the independent holdout and must not be used for tuning.

Extends the MPDD pixel-only evaluation with image-level AUROC/AP, which are
derived from the fused anomaly maps via max pooling (the same aggregation is
applied to the DINO baseline so the delta is a fair comparison).

Frozen config (from experiments/dynamic_fusion/v3_3/cross_seed_report.json):
    strategy = weighted_ensemble
    weights  = {anomalydino_visual: 0.60, anomalyclip_text: 0.40}
    calibrate = True  (z-score per branch using its own normal-image stats)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast
from industrial_ad.fusion.alignment import build_alignment_plan
from industrial_ad.fusion.v3_3_strategies import (
    BranchData,
    weighted_ensemble_fusion,
)
from run_dynamic_fusion_v2_cache import load_cache, resize_maps
from sklearn.metrics import average_precision_score, roc_auc_score

STRIDE: int = 8  # matches V3.3 MPDD pixel-eval protocol
DINO_W: float = 0.60
CLIP_W: float = 0.40

CATEGORIES = ["01", "02", "03"]
BTAD_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v2_btad_predictions"


def load_branches(category: str, seed: int) -> dict[str, BranchData]:
    root = BTAD_ROOT / f"v2_btad_s{seed}_k1_full_v1"
    dino = load_cache(root / "anomalydino_visual" / f"{category}.npz")
    ref_ids = np.asarray(dino["sample_ids"])
    dino_maps = np.asarray(dino["anomaly_maps"], dtype=np.float32)
    dino_masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
    dino_gts = np.asarray(dino["gt_sp"], dtype=np.uint8)

    branches = {
        "anomalydino_visual": BranchData(
            name="anomalydino_visual",
            anomaly_maps=dino_maps,
            image_scores=np.asarray(dino["pr_sp"], dtype=np.float32),
            gt_labels=dino_gts,
            gt_masks=dino_masks,
            sample_ids=ref_ids,
        )
    }

    clip = load_cache(root / "anomalyclip_text" / f"{category}.npz")
    plan = build_alignment_plan(ref_ids, np.asarray(clip["sample_ids"]))
    clip_maps = np.asarray(clip["anomaly_maps"], dtype=np.float32)[plan.candidate_order]
    clip_maps = np.asarray(resize_maps(clip_maps, dino_maps.shape[1:]), dtype=np.float32)
    branches["anomalyclip_text"] = BranchData(
        name="anomalyclip_text",
        anomaly_maps=clip_maps,
        image_scores=np.zeros(len(ref_ids), dtype=np.float32),
        gt_labels=dino_gts,
        gt_masks=dino_masks,
        sample_ids=ref_ids,
    )
    return branches


def pixel_metrics(maps: np.ndarray, masks: np.ndarray) -> dict:
    maps_s = maps[:, ::STRIDE, ::STRIDE]
    masks_s = masks[:, ::STRIDE, ::STRIDE]
    flat_m = maps_s.ravel()
    flat_l = (masks_s.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_l, flat_m)),
        "pixel_ap": float(average_precision_score(flat_l, flat_m)),
        "pixel_aupro": float(aupro_fast(masks_s, maps_s)),
    }


def image_metrics(maps: np.ndarray, labels: np.ndarray) -> dict:
    scores = maps.max(axis=(1, 2)).astype(np.float64)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.3 BTAD holdout validation")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], help="Run a single seed only")
    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else [0, 1, 2]

    weights = {"anomalydino_visual": DINO_W, "anomalyclip_text": CLIP_W}
    rows = []

    for seed in seeds:
        for cat in CATEGORIES:
            branches = load_branches(cat, seed)
            dino = branches["anomalydino_visual"]
            fused = weighted_ensemble_fusion(branches, weights, calibrate=True)

            dino_pix = pixel_metrics(dino.anomaly_maps.astype(np.float64), dino.gt_masks)
            fused_pix = pixel_metrics(fused, dino.gt_masks)
            dino_img = image_metrics(dino.anomaly_maps.astype(np.float64), dino.gt_labels)
            fused_img = image_metrics(fused, dino.gt_labels)

            rows.append({
                "seed": seed,
                "category": cat,
                "samples": int(len(dino.sample_ids)),
                "dino_image_auroc": round(dino_img["image_auroc"], 6),
                "dino_image_ap": round(dino_img["image_ap"], 6),
                "fused_image_auroc": round(fused_img["image_auroc"], 6),
                "fused_image_ap": round(fused_img["image_ap"], 6),
                "delta_image_auroc": round(fused_img["image_auroc"] - dino_img["image_auroc"], 6),
                "delta_image_ap": round(fused_img["image_ap"] - dino_img["image_ap"], 6),
                "dino_pixel_auroc": round(dino_pix["pixel_auroc"], 6),
                "dino_pixel_ap": round(dino_pix["pixel_ap"], 6),
                "dino_pixel_aupro": round(dino_pix["pixel_aupro"], 6),
                "fused_pixel_auroc": round(fused_pix["pixel_auroc"], 6),
                "fused_pixel_ap": round(fused_pix["pixel_ap"], 6),
                "fused_pixel_aupro": round(fused_pix["pixel_aupro"], 6),
                "delta_pixel_auroc": round(fused_pix["pixel_auroc"] - dino_pix["pixel_auroc"], 6),
                "delta_pixel_ap": round(fused_pix["pixel_ap"] - dino_pix["pixel_ap"], 6),
                "delta_pixel_aupro": round(fused_pix["pixel_aupro"] - dino_pix["pixel_aupro"], 6),
            })
            print(
                f"  seed={seed} cat={cat}: dinoAP={dino_pix['pixel_ap']:.4f} "
                f"fusedAP={fused_pix['pixel_ap']:.4f} dAP={fused_pix['pixel_ap']-dino_pix['pixel_ap']:+.4f} "
                f"| img dAUROC={fused_img['image_auroc']-dino_img['image_auroc']:+.4f}",
                flush=True,
            )

    # Aggregate across seeds per category
    agg_by_cat = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        agg_by_cat[cat] = {
            "delta_image_auroc": round(float(np.mean([r["delta_image_auroc"] for r in cat_rows])), 6),
            "delta_image_ap": round(float(np.mean([r["delta_image_ap"] for r in cat_rows])), 6),
            "delta_pixel_auroc": round(float(np.mean([r["delta_pixel_auroc"] for r in cat_rows])), 6),
            "delta_pixel_ap": round(float(np.mean([r["delta_pixel_ap"] for r in cat_rows])), 6),
            "delta_pixel_aupro": round(float(np.mean([r["delta_pixel_aupro"] for r in cat_rows])), 6),
            "positive_pixel_ap_seeds": sum(1 for r in cat_rows if r["delta_pixel_ap"] > 0),
        }

    def mean(key: str) -> float:
        return round(float(np.mean([r[key] for r in rows])), 6)

    report = {
        "pipeline": "v3_3",
        "role": "btad_holdout_validation",
        "strategy": "weighted_ensemble",
        "weights": weights,
        "calibrate": True,
        "stride": STRIDE,
        "seeds": seeds,
        "categories": CATEGORIES,
        "note": "Frozen config only; no candidate search on BTAD. Image-level score = max(fused map). "
                "z-score calibration uses each category's normal test images (same protocol as MPDD V3.3).",
        "summary": {
            "mean_delta_image_auroc": mean("delta_image_auroc"),
            "mean_delta_image_ap": mean("delta_image_ap"),
            "mean_delta_pixel_auroc": mean("delta_pixel_auroc"),
            "mean_delta_pixel_ap": mean("delta_pixel_ap"),
            "mean_delta_pixel_aupro": mean("delta_pixel_aupro"),
            "positive_pixel_ap_seeds": sum(1 for r in rows if r["delta_pixel_ap"] > 0),
            "positive_image_ap_seeds": sum(1 for r in rows if r["delta_image_ap"] > 0),
        },
        "per_category": agg_by_cat,
        "rows": rows,
    }

    out_dir = ROOT / "experiments" / "dynamic_fusion" / "v3_3" / "btad_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
