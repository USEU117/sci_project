"""Gate A1: evaluator-only upper bound for V3 text rescue on MPDD.

This diagnostic intentionally uses development labels only after both branch
predictions have been loaded and aligned.  Labels and masks are never exposed
to a router.  The oracle therefore measures possible headroom, not a deployable
method.  Pixel diagnostics are evaluated on a recorded spatial stride to keep
the overnight CPU gate bounded and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from run_dynamic_fusion_v2_cache import load_cache, resize_maps  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("rank input contains NaN or infinity")
    if len(flat) <= 1:
        return np.zeros_like(flat)
    return (rankdata(flat, method="average") - 1.0) / (len(flat) - 1.0)


def robust_map01(values: np.ndarray) -> np.ndarray:
    maps = np.asarray(values, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"maps must be [N,H,W], got {maps.shape}")
    flat = maps.reshape(len(maps), -1)
    lower = np.quantile(flat, 0.01, axis=1).astype(np.float32)
    upper = np.quantile(flat, 0.99, axis=1).astype(np.float32)
    scale = np.maximum(upper - lower, np.float32(1e-6))
    return np.clip(
        (maps - lower[:, None, None]) / scale[:, None, None], 0.0, 1.0
    ).astype(np.float32)


def label_informed_oracle(
    labels: np.ndarray, visual: np.ndarray, text: np.ndarray
) -> np.ndarray:
    """Optimistic evaluator-only selection; never valid as router inference."""

    truth = np.asarray(labels, dtype=bool)
    visual = np.asarray(visual)
    text = np.asarray(text)
    if truth.shape != visual.shape or truth.shape != text.shape:
        raise ValueError("oracle labels and scores must have equal shapes")
    return np.where(truth, np.maximum(visual, text), np.minimum(visual, text))


def region_diagnostics(
    masks: np.ndarray, visual: np.ndarray, text: np.ndarray
) -> tuple[int, int, float]:
    region_count = 0
    text_better = 0
    positive_gaps: list[float] = []
    for mask, visual_map, text_map in zip(masks, visual, text):
        components = measure.label(mask.astype(bool))
        for region_id in range(1, int(components.max()) + 1):
            select = components == region_id
            gap = float(np.mean(text_map[select]) - np.mean(visual_map[select]))
            region_count += 1
            if gap > 0:
                text_better += 1
                positive_gaps.append(gap)
    return region_count, text_better, float(np.mean(positive_gaps)) if positive_gaps else 0.0


def metric_pair(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    flat_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return (
        float(roc_auc_score(flat_labels, flat_scores)),
        float(average_precision_score(flat_labels, flat_scores)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "dynamic_fusion"
        / "v3"
        / "gate_a1_oracle_headroom"
        / "report.json",
    )
    args = parser.parse_args()
    if args.pixel_stride < 1:
        raise SystemExit("pixel-stride must be positive")

    manifest_path = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
    categories = sorted(
        json.loads(manifest_path.read_text(encoding="utf-8"))["categories"]
    )
    rows: list[dict] = []
    provenance: list[dict] = []

    for seed in args.seeds:
        for shot in args.shots:
            pair_id = f"v2_mpdd_s{seed}_k{shot}_full_v1"
            pair_root = args.prediction_root / pair_id
            for category in categories:
                visual_path = pair_root / "anomalydino_visual" / f"{category}.npz"
                text_path = pair_root / "anomalyclip_text" / f"{category}.npz"
                visual = load_cache(visual_path)
                text = load_cache(text_path)
                alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
                order = alignment.candidate_order
                labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
                if not np.array_equal(labels, np.asarray(text["gt_sp"])[order]):
                    raise ValueError(f"labels differ after alignment: {pair_id}/{category}")

                visual_maps = np.asarray(visual["anomaly_maps"], dtype=np.float32)
                text_maps = resize_maps(
                    np.asarray(text["anomaly_maps"], dtype=np.float32)[order],
                    visual_maps.shape[1:],
                ).astype(np.float32)
                masks = np.asarray(visual["imgs_masks"], dtype=np.uint8)
                if masks.shape != visual_maps.shape or masks.shape != text_maps.shape:
                    raise ValueError(f"mask/map mismatch: {pair_id}/{category}")

                visual_scores_raw = np.asarray(visual["pr_sp"], dtype=np.float64)
                text_scores_raw = np.asarray(text["pr_sp"], dtype=np.float64)[order]
                visual_scores = rank01(visual_scores_raw)
                text_scores = rank01(text_scores_raw)
                image_oracle = label_informed_oracle(labels, visual_scores, text_scores)

                stride = args.pixel_stride
                masks_small = masks[:, ::stride, ::stride].astype(bool)
                visual_small = robust_map01(visual_maps[:, ::stride, ::stride])
                text_small = robust_map01(text_maps[:, ::stride, ::stride])
                pixel_oracle = label_informed_oracle(
                    masks_small, visual_small, text_small
                )

                vi_auc_raw, vi_ap_raw = metric_pair(labels, visual_scores_raw)
                ti_auc_raw, ti_ap_raw = metric_pair(labels, text_scores_raw)
                vi_auc, vi_ap = metric_pair(labels, visual_scores)
                ti_auc, ti_ap = metric_pair(labels, text_scores)
                oi_auc, oi_ap = metric_pair(labels, image_oracle)
                vp_auc, vp_ap = metric_pair(masks_small, visual_small)
                tp_auc, tp_ap = metric_pair(masks_small, text_small)
                op_auc, op_ap = metric_pair(masks_small, pixel_oracle)
                region_count, text_better_regions, positive_region_gap = region_diagnostics(
                    masks_small, visual_small, text_small
                )

                image_text_help = np.where(
                    labels.astype(bool), text_scores > visual_scores, text_scores < visual_scores
                )
                image_text_harm = np.where(
                    labels.astype(bool), text_scores < visual_scores, text_scores > visual_scores
                )
                row = {
                    "seed": seed,
                    "shot": shot,
                    "category": category,
                    "samples": len(labels),
                    "pixel_stride": stride,
                    "visual_image_auroc_raw": vi_auc_raw,
                    "visual_image_ap_raw": vi_ap_raw,
                    "text_image_auroc_raw": ti_auc_raw,
                    "text_image_ap_raw": ti_ap_raw,
                    "visual_image_auroc_rank": vi_auc,
                    "visual_image_ap_rank": vi_ap,
                    "text_image_auroc_rank": ti_auc,
                    "text_image_ap_rank": ti_ap,
                    "oracle_image_auroc": oi_auc,
                    "oracle_image_ap": oi_ap,
                    "oracle_image_delta_auroc_vs_visual": oi_auc - vi_auc,
                    "oracle_image_delta_ap_vs_visual": oi_ap - vi_ap,
                    "image_text_help_count": int(np.sum(image_text_help)),
                    "image_text_harm_count": int(np.sum(image_text_harm)),
                    "visual_pixel_auroc_diagnostic": vp_auc,
                    "visual_pixel_ap_diagnostic": vp_ap,
                    "text_pixel_auroc_diagnostic": tp_auc,
                    "text_pixel_ap_diagnostic": tp_ap,
                    "oracle_pixel_auroc_diagnostic": op_auc,
                    "oracle_pixel_ap_diagnostic": op_ap,
                    "oracle_pixel_delta_auroc_vs_visual": op_auc - vp_auc,
                    "oracle_pixel_delta_ap_vs_visual": op_ap - vp_ap,
                    "anomaly_region_count": region_count,
                    "text_better_region_count": text_better_regions,
                    "text_better_region_fraction": (
                        text_better_regions / region_count if region_count else 0.0
                    ),
                    "mean_positive_text_region_gap": positive_region_gap,
                }
                rows.append(row)
                provenance.extend(
                    [
                        {
                            "pair_id": pair_id,
                            "category": category,
                            "branch": "visual",
                            "path": str(visual_path.resolve()),
                            "sha256": sha256(visual_path),
                        },
                        {
                            "pair_id": pair_id,
                            "category": category,
                            "branch": "text",
                            "path": str(text_path.resolve()),
                            "sha256": sha256(text_path),
                        },
                    ]
                )

    delta_fields = [
        "oracle_image_delta_auroc_vs_visual",
        "oracle_image_delta_ap_vs_visual",
        "oracle_pixel_delta_auroc_vs_visual",
        "oracle_pixel_delta_ap_vs_visual",
    ]
    seed_summaries = []
    for seed in args.seeds:
        subset = [row for row in rows if row["seed"] == seed]
        seed_summaries.append(
            {
                "seed": seed,
                **{field: float(np.mean([row[field] for row in subset])) for field in delta_fields},
                "text_better_region_fraction": float(
                    sum(row["text_better_region_count"] for row in subset)
                    / max(sum(row["anomaly_region_count"] for row in subset), 1)
                ),
            }
        )

    image_positive_seeds = sum(
        row["oracle_image_delta_auroc_vs_visual"] > 0.005 for row in seed_summaries
    )
    pixel_positive_seeds = sum(
        row["oracle_pixel_delta_ap_vs_visual"] > 0.005 for row in seed_summaries
    )
    total_regions = sum(row["anomaly_region_count"] for row in rows)
    text_better_regions = sum(row["text_better_region_count"] for row in rows)
    region_fraction = text_better_regions / max(total_regions, 1)
    sufficient = bool(
        image_positive_seeds >= 2
        or (pixel_positive_seeds >= 2 and region_fraction >= 0.10)
    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "gate": "v3_gate_a1_oracle_headroom",
        "run_id": "v3_20260812_overnight_gate_a_v1",
        "dataset": "mpdd",
        "dataset_role": "development",
        "analysis_type": "label_informed_evaluator_only_upper_bound",
        "router_labels_used": False,
        "router_masks_used": False,
        "development_labels_used_by_offline_oracle": True,
        "btad_accessed": False,
        "pixel_analysis_stride": args.pixel_stride,
        "pixel_metrics_are_diagnostic_not_paper_metrics": True,
        "rows": rows,
        "seed_summaries": seed_summaries,
        "summary": {
            "category_seed_shot_rows": len(rows),
            "image_positive_seed_count": image_positive_seeds,
            "pixel_positive_seed_count": pixel_positive_seeds,
            "anomaly_region_count": total_regions,
            "text_better_region_count": text_better_regions,
            "text_better_region_fraction": region_fraction,
            "anomalyclip_oracle_headroom_sufficient_for_gate_a2": sufficient,
        },
        "decision_rule": (
            "continue to Gate A2 if image oracle AUROC gain >0.005 on at least two seeds, "
            "or pixel oracle AP gain >0.005 on at least two seeds and text is better on "
            "at least 10% of anomaly regions"
        ),
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": len(rows),
                "sufficient_for_gate_a2": sufficient,
                "text_better_region_fraction": region_fraction,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

