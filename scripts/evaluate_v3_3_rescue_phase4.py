"""Phase-4 evaluation of V3.3 visual-anchored text local rescue on MPDD seed0/K1.

CPU only; reuses the same frozen caches as the phase-3 gate:
  - outputs/dynamic_fusion/v2_mpdd_predictions/v2_mpdd_s0_k1_full_v1/
  - outputs/dynamic_fusion/v2_branch_cache/v2_mpdd_s0_k1_branch_cache_v1/

Comparisons (all reference-only calibrated):
  - visual_only            : AnomalyDINO raw maps (default safe output)
  - v33_clean_w0.40        : phase-3 best fixed-weight clean ensemble
  - rescue_cap{1.0,2.0,4.0}: visual-anchored local rescue, residual-cap ablation
  - rescue_unstable        : rescue with text forced unstable (visual fallback only)

Reports per category and aggregated: pixel AUROC/AP/AUPRO (STRIDE=8), image
AUROC/AP/F1 (reference-derived threshold), reason-code counts, delta vs visual.
All five leakage flags false.
"""

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

from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from industrial_ad.fusion.v3_3_clean import (  # noqa: E402
    DEFAULT_ANCHOR,
    EvaluationTarget,
    RouterInput,
    estimate_reference_stats,
    evaluate_clean,
    weighted_ensemble_clean,
)
from industrial_ad.fusion.v3_3_rescue import (  # noqa: E402
    LocalRescueConfig,
    local_rescue_fusion,
)
from run_dynamic_fusion_v2_cache import load_cache, resize_maps  # noqa: E402

STRIDE = 8
RESCUE_CAPS = [1.0, 2.0, 4.0]


def image_scores_from_maps(maps: np.ndarray) -> np.ndarray:
    return np.asarray(maps, dtype=np.float64).max(axis=(1, 2))


def decision_threshold_from_reference(ref_maps: np.ndarray) -> float:
    return float(image_scores_from_maps(ref_maps).max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "dynamic_fusion" / "v3_3_clean" / "phase4_rescue_20260817" / "report.json",
    )
    args = parser.parse_args()

    pair_id = "v2_mpdd_s0_k1_full_v1"
    ref_pair_id = "v2_mpdd_s0_k1_branch_cache_v1"
    test_root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / pair_id
    ref_root = ROOT / "outputs" / "dynamic_fusion" / "v2_branch_cache" / ref_pair_id
    manifest_path = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
    categories = sorted(json.loads(manifest_path.read_text(encoding="utf-8"))["categories"])

    rows: list[dict] = []
    for category in categories:
        visual_path = test_root / "anomalydino_visual" / f"{category}.npz"
        text_path = test_root / "anomalyclip_text" / f"{category}.npz"
        ref_visual_path = ref_root / "anomalydino_visual" / f"{category}.npz"
        ref_text_path = ref_root / "anomalyclip_text" / f"{category}.npz"

        visual = load_cache(visual_path)
        text = load_cache(text_path)
        alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
        order = alignment.candidate_order
        labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
        visual_maps = np.asarray(visual["anomaly_maps"], dtype=np.float32)
        text_maps = resize_maps(
            np.asarray(text["anomaly_maps"], dtype=np.float32)[order], visual_maps.shape[1:]
        ).astype(np.float32)
        masks = np.asarray(visual["imgs_masks"], dtype=np.uint8)

        with np.load(ref_visual_path, allow_pickle=False) as data:
            ref_visual_maps = np.asarray(data["pixel_maps"], dtype=np.float32)
        with np.load(ref_text_path, allow_pickle=False) as data:
            ref_text_maps = resize_maps(
                np.asarray(data["pixel_maps"], dtype=np.float32), visual_maps.shape[1:]
            ).astype(np.float32)

        ri = RouterInput(
            branches={DEFAULT_ANCHOR: visual_maps, "anomalyclip_text": text_maps},
            reference_maps={DEFAULT_ANCHOR: ref_visual_maps, "anomalyclip_text": ref_text_maps},
            sample_ids=np.asarray(visual["sample_ids"]),
            category=category,
            seed=0,
            shot=1,
            metadata={"test_pair": pair_id, "ref_pair": ref_pair_id},
        )
        target = EvaluationTarget(
            gt_labels=labels, gt_masks=masks, sample_ids=np.asarray(visual["sample_ids"])
        )

        methods: dict[str, dict] = {}
        methods["visual_only"] = {"maps": visual_maps.astype(np.float64)}
        fused, _ = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 0.40, "anomalyclip_text": 0.60})
        methods["v33_clean_w0.40"] = {"maps": fused}
        for cap in RESCUE_CAPS:
            r_maps, r_diag = local_rescue_fusion(ri, config=LocalRescueConfig(residual_cap=cap))
            methods[f"rescue_cap{cap}"] = {"maps": r_maps, "diag": r_diag}
        # unstable-text control: text refs made degenerate -> pure visual fallback
        unstable_refs = {
            DEFAULT_ANCHOR: ref_visual_maps,
            "anomalyclip_text": np.full_like(ref_text_maps, 0.5, dtype=np.float32),
        }
        ri_unstable = RouterInput(
            branches=ri.branches,
            reference_maps=unstable_refs,
            sample_ids=ri.sample_ids,
            category=ri.category,
            seed=ri.seed,
            shot=ri.shot,
            metadata=ri.metadata,
        )
        u_maps, u_diag = local_rescue_fusion(ri_unstable, config=LocalRescueConfig(residual_cap=2.0))
        methods["rescue_unstable"] = {"maps": u_maps, "diag": u_diag}

        row: dict = {"category": category, "seed": 0, "shot": 1, "samples": len(labels), "anomaly_images": int(labels.sum()), "pixel_stride": STRIDE}
        from industrial_ad.fusion.v3_3_clean import compute_z_score
        rv_stats = estimate_reference_stats(ref_visual_maps)
        rt_stats = estimate_reference_stats(ref_text_maps)
        ref_visual_z = compute_z_score(ref_visual_maps, rv_stats["center"], rv_stats["scale"])
        ref_fused_clean_z = 0.40 * ref_visual_z + 0.60 * compute_z_score(
            ref_text_maps, rt_stats["center"], rt_stats["scale"]
        )
        for name, entry in methods.items():
            maps = entry["maps"]
            pixel = evaluate_clean(ri, target, maps, stride=STRIDE)
            if name == "visual_only":
                thr = decision_threshold_from_reference(ref_visual_maps)
            elif name.startswith("v33_clean"):
                thr = decision_threshold_from_reference(ref_fused_clean_z)
            else:  # rescue methods operate in visual-z scale
                thr = decision_threshold_from_reference(ref_visual_z)
            img_scores = image_scores_from_maps(maps)
            from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
            pred = (img_scores > thr).astype(np.uint8)
            row[f"{name}_pixel_auroc"] = pixel["pixel_auroc"]
            row[f"{name}_pixel_ap"] = pixel["pixel_ap"]
            row[f"{name}_pixel_aupro"] = pixel["pixel_aupro"]
            row[f"{name}_image_auroc"] = float(roc_auc_score(labels, img_scores))
            row[f"{name}_image_ap"] = float(average_precision_score(labels, img_scores))
            row[f"{name}_image_f1"] = float(f1_score(labels, pred, zero_division=0.0))
            if "diag" in entry:
                row[f"{name}_reason_counts"] = entry["diag"]["reason_counts"]
                row[f"{name}_text_stable"] = entry["diag"]["text_stable"]
                row[f"{name}_accepted_pixels"] = entry["diag"]["accepted_pixels"]
                row[f"{name}_max_text_residual"] = entry["diag"]["max_text_residual"]
            row[f"{name}_delta_pixel_ap_vs_visual"] = pixel["pixel_ap"] - row["visual_only_pixel_ap"]
        rows.append(row)

    # aggregate
    agg: dict = {"mean": {}, "positive_vs_visual": {}}
    for name in methods:
        agg["mean"][name] = {
            "pixel_ap": float(np.nanmean([r[f"{name}_pixel_ap"] for r in rows])),
            "pixel_auroc": float(np.nanmean([r[f"{name}_pixel_auroc"] for r in rows])),
            "pixel_aupro": float(np.nanmean([r[f"{name}_pixel_aupro"] for r in rows])),
            "image_auroc": float(np.nanmean([r[f"{name}_image_auroc"] for r in rows])),
            "image_ap": float(np.nanmean([r[f"{name}_image_ap"] for r in rows])),
            "image_f1": float(np.nanmean([r[f"{name}_image_f1"] for r in rows])),
        }
        if name != "visual_only":
            deltas = [r[f"{name}_delta_pixel_ap_vs_visual"] for r in rows]
            agg["positive_vs_visual"][name] = {
                "positive_count": int(sum(d > 0 for d in deltas)),
                "mean_delta": float(np.mean(deltas)),
                "max_regression": float(min(deltas)),
            }

    report = {
        "schema_version": 1,
        "run_id": "v3_3_clean_phase4_rescue_20260817_mpdd_s0_k1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "phase_4_visual_anchored_text_local_rescue",
        "dataset": "mpdd",
        "dataset_role": "development",
        "seed": 0,
        "shot": 1,
        "gpu": False,
        "leakage_flags": {
            "test_predictions_used": False,
            "test_labels_used": False,
            "test_masks_used": False,
            "test_dataset_statistics_used": False,
            "test_normal_selection_used": False,
        },
        "pre_registered": {
            "residual_caps": RESCUE_CAPS,
            "visual_candidate_quantile": LocalRescueConfig().visual_candidate_quantile,
            "text_support_quantile": LocalRescueConfig().text_support_quantile,
            "prompt_stability_cv": LocalRescueConfig().prompt_stability_cv,
            "background_reject_margin": LocalRescueConfig().background_reject_margin,
            "pixel_stride": STRIDE,
        },
        "aggregate": agg,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "n_rows": len(rows), "aggregate": agg["mean"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
