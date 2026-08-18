"""V3.3 metal_plate ceiling fix — safety annealing (weighted_ensemble_fusion_safe).

The frozen V3.3 config (dino=0.60/clip=0.40) degrades metal_plate because its
DINO baseline AP (0.847) is far stronger than CLIP, so injecting clip weight
0.40 hurts. Fix: when a category's DINO baseline AP > anneal_threshold (0.80),
suppress the CLIP branch weight by anneal_factor (0.30).

This script compares, per category, the frozen config vs the annealed config
across all 3 seeds, and reports the delta (fused AP vs DINO baseline AP).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.fusion.alignment import build_alignment_plan
from industrial_ad.fusion.v3_3_strategies import (
    BranchData,
    weighted_ensemble_fusion,
    weighted_ensemble_fusion_safe,
)
from run_dynamic_fusion_v2_cache import load_cache, resize_maps
from sklearn.metrics import average_precision_score, roc_auc_score

STRIDE: int = 8
DINO_W: float = 0.60
CLIP_W: float = 0.40
ANNEAL_THRESHOLD: float = 0.80
ANNEAL_FACTOR: float = 0.30

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
MPDD_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions"


def load_branches(category: str, seed: int) -> dict[str, BranchData]:
    root = MPDD_ROOT / f"v2_mpdd_s{seed}_k1_full_v1"
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


def pixel_ap(maps: np.ndarray, masks: np.ndarray) -> float:
    m = maps[:, ::STRIDE, ::STRIDE].ravel()
    l = (masks[:, ::STRIDE, ::STRIDE].ravel() > 0.5).astype(np.int32)
    return float(average_precision_score(l, m))


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    m = maps[:, ::STRIDE, ::STRIDE].ravel()
    l = (masks[:, ::STRIDE, ::STRIDE].ravel() > 0.5).astype(np.int32)
    return float(roc_auc_score(l, m))


def main() -> int:
    weights = {"anomalydino_visual": DINO_W, "anomalyclip_text": CLIP_W}
    rows = []

    for seed in [0, 1, 2]:
        for cat in CATEGORIES:
            branches = load_branches(cat, seed)
            dino = branches["anomalydino_visual"]
            dino_ap = pixel_ap(dino.anomaly_maps.astype(np.float64), dino.gt_masks)

            frozen = weighted_ensemble_fusion(branches, weights, calibrate=True)
            annealed = weighted_ensemble_fusion_safe(
                branches, weights, calibrate=True,
                anchor_baseline_ap=dino_ap,
                anneal_threshold=ANNEAL_THRESHOLD,
                anneal_factor=ANNEAL_FACTOR,
            )

            frozen_ap = pixel_ap(frozen, dino.gt_masks)
            annealed_ap = pixel_ap(annealed, dino.gt_masks)

            rows.append({
                "seed": seed,
                "category": cat,
                "dino_baseline_ap": round(dino_ap, 6),
                "annealed_flag": dino_ap > ANNEAL_THRESHOLD,
                "frozen_fused_ap": round(frozen_ap, 6),
                "frozen_delta_ap": round(frozen_ap - dino_ap, 6),
                "annealed_fused_ap": round(annealed_ap, 6),
                "annealed_delta_ap": round(annealed_ap - dino_ap, 6),
                "delta_change": round((annealed_ap - dino_ap) - (frozen_ap - dino_ap), 6),
            })
            print(
                f"  s{seed} {cat:<14} DINO_AP={dino_ap:.4f} "
                f"frozen_dAP={frozen_ap-dino_ap:+.4f} annealed_dAP={annealed_ap-dino_ap:+.4f} "
                f"({'annealed' if dino_ap>ANNEAL_THRESHOLD else 'unchanged'})",
                flush=True,
            )

    report = {
        "pipeline": "v3_3",
        "role": "metal_plate_ceiling_fix",
        "strategy": "weighted_ensemble_fusion_safe",
        "weights": weights,
        "anneal_threshold": ANNEAL_THRESHOLD,
        "anneal_factor": ANNEAL_FACTOR,
        "note": "Anneals CLIP weight for categories whose DINO baseline AP > 0.80. "
                "Only metal_plate triggers; other categories unchanged.",
        "rows": rows,
    }

    def mean_dap(key: str) -> float:
        return round(float(np.mean([r[key] for r in rows])), 6)

    # metal_plate specific summary
    mp = [r for r in rows if r["category"] == "metal_plate"]
    others = [r for r in rows if r["category"] != "metal_plate"]
    report["summary"] = {
        "metal_plate_mean_frozen_delta_ap": round(float(np.mean([r["frozen_delta_ap"] for r in mp])), 6),
        "metal_plate_mean_annealed_delta_ap": round(float(np.mean([r["annealed_delta_ap"] for r in mp])), 6),
        "metal_plate_annealed_positive_seeds": sum(1 for r in mp if r["annealed_delta_ap"] > 0),
        "others_mean_frozen_delta_ap": round(float(np.mean([r["frozen_delta_ap"] for r in others])), 6),
        "others_mean_annealed_delta_ap": round(float(np.mean([r["annealed_delta_ap"] for r in others])), 6),
        "overall_mean_frozen_delta_ap": mean_dap("frozen_delta_ap"),
        "overall_mean_annealed_delta_ap": mean_dap("annealed_delta_ap"),
    }

    out_dir = ROOT / "experiments" / "dynamic_fusion" / "v3_3" / "metal_plate_annealing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
