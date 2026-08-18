"""Phase-3 CPU Gate for V3.3-clean on MPDD seed0 / K1 (docs 阶段三).

Reuses ONLY frozen caches (zero GPU inference):
  - test predictions : outputs/dynamic_fusion/v2_mpdd_predictions/v2_mpdd_s0_k1_full_v1/
  - normal references : outputs/dynamic_fusion/v2_branch_cache/v2_mpdd_s0_k1_branch_cache_v1/

Pre-registered comparison grid (no search, no test-true-value fitting):
  - visual_only   : AnomalyDINO raw maps (default safe output)
  - text_only     : AnomalyCLIP raw maps (resized to 448)
  - v33_clean_wXX : weighted_ensemble_clean with fixed visual weight
                    w in {0.40, 0.50, 0.60, 0.70}  (0.50 == 50/50)
  - visual_fallback: weighted_ensemble_clean with text forced unreliable
                    (pure anchor path; rank-equivalent to raw visual)
  - old_v33_w060  : OLD leaky weighted_ensemble_fusion (gt_masks calibration),
                    reported ONLY as an invalid-marked control.

Metrics per category and aggregated (pixel at STRIDE=8, image via max-pool):
  image AUROC / AP / F1, pixel AUROC / AP / AUPRO, per-class delta vs visual,
  rescue / harm / coverage / risk-coverage (image level, reference-derived τ).

All five leakage flags are false for every clean method; the old V3.3 row
carries development_only_leaky_calibration=true / paper_eligible=false.

Suggested gate (from the plan): mean pixel AP > visual, >=4/6 categories
positive, no single-category large regression, AUPRO not down overall,
audit & repeatability pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast  # noqa: E402
from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from industrial_ad.fusion.v3_3_clean import (  # noqa: E402
    DEFAULT_ANCHOR,
    EvaluationTarget,
    RouterInput,
    compute_z_score,
    estimate_reference_stats,
    evaluate_clean,
    weighted_ensemble_clean,
)
from industrial_ad.fusion.v3_3_strategies import (  # noqa: E402
    BranchData,
    weighted_ensemble_fusion,
)
from run_dynamic_fusion_v2_cache import load_cache, resize_maps  # noqa: E402

STRIDE = 8
CLEAN_WEIGHTS = [0.40, 0.50, 0.60, 0.70]
OLD_V33_WEIGHT = 0.60
RISK_COVERAGE_TARGETS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ref_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pixel_maps": np.asarray(data["pixel_maps"], dtype=np.float32),
            "image_scores": np.asarray(data["image_scores"], dtype=np.float64),
            "sample_ids": np.asarray(data["sample_ids"]),
        }


def image_scores_from_maps(maps: np.ndarray) -> np.ndarray:
    """Image score = max over spatial dims (fair across methods)."""
    return np.asarray(maps, dtype=np.float64).max(axis=(1, 2))


def decision_threshold_from_reference(ref_maps: np.ndarray) -> float:
    """Reference-only image threshold: max over normal-reference image scores."""
    ref_img = image_scores_from_maps(ref_maps)
    if ref_img.size == 0:
        raise ValueError("empty reference maps")
    return float(ref_img.max())


def metric_image(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores > threshold).astype(np.uint8)
    if np.unique(labels).size < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, scores))
    if labels.sum() == 0 or labels.sum() == labels.size:
        ap = float("nan")
    else:
        ap = float(average_precision_score(labels, scores))
    f1 = float(f1_score(labels, pred, zero_division=0.0))
    return {"image_auroc": auroc, "image_ap": ap, "image_f1": f1, "image_threshold": threshold}


def risk_coverage_curve(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Risk-coverage: accept top-c fraction by score; risk = error rate among kept."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)  # most anomalous first
    n = len(labels)
    curve = {}
    for c in RISK_COVERAGE_TARGETS:
        keep = max(1, int(round(c * n)))
        kept_idx = order[:keep]
        pred = (scores[kept_idx] > threshold).astype(np.uint8)
        errors = int(np.count_nonzero(pred != labels[kept_idx]))
        curve[f"c{c:.2f}"] = {
            "coverage": round(keep / n, 4),
            "risk": round(errors / keep, 4),
            "kept": keep,
            "errors": errors,
        }
    return curve


def rescues_harm_coverage(
    labels: np.ndarray,
    visual_pred: np.ndarray,
    fused_pred: np.ndarray,
    text_reliable: bool,
) -> dict:
    """Image-level rescue/harm/coverage vs the visual baseline at τ.

    rescue : visual prediction wrong, fused prediction correct
    harm   : visual prediction correct, fused prediction wrong
    coverage: fraction of test images where the text branch contributed
              (category-level reliability is a scalar in the clean router).
    """
    labels = np.asarray(labels, dtype=np.uint8)
    v = np.asarray(visual_pred, dtype=np.uint8)
    f = np.asarray(fused_pred, dtype=np.uint8)
    rescue = int(np.count_nonzero((v != labels) & (f == labels)))
    harm = int(np.count_nonzero((v == labels) & (f != labels)))
    return {
        "rescue_count": rescue,
        "harm_count": harm,
        "rescue_minus_harm": rescue - harm,
        "text_coverage": float(text_reliable),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=0, help="development seed (fixed to 0 for the gate)"
    )
    parser.add_argument("--shot", type=int, default=1, help="shot (fixed to 1 for the gate)")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "dynamic_fusion" / "v3_3_clean" / "gate_20260817" / "report.json",
    )
    args = parser.parse_args()
    if (args.seed, args.shot) != (0, 1):
        raise SystemExit("gate is pre-registered for MPDD seed0/K1 only")

    pair_id = f"v2_mpdd_s{args.seed}_k{args.shot}_full_v1"
    ref_pair_id = f"v2_mpdd_s{args.seed}_k{args.shot}_branch_cache_v1"
    test_root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / pair_id
    ref_root = ROOT / "outputs" / "dynamic_fusion" / "v2_branch_cache" / ref_pair_id
    manifest_path = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
    categories = sorted(json.loads(manifest_path.read_text(encoding="utf-8"))["categories"])

    rows: list[dict] = []
    provenance: list[dict] = []
    gate_decisions: dict[str, bool] = {}

    for category in categories:
        visual_path = test_root / "anomalydino_visual" / f"{category}.npz"
        text_path = test_root / "anomalyclip_text" / f"{category}.npz"
        ref_visual_path = ref_root / "anomalydino_visual" / f"{category}.npz"
        ref_text_path = ref_root / "anomalyclip_text" / f"{category}.npz"
        for path in (visual_path, text_path, ref_visual_path, ref_text_path):
            if not path.is_file():
                raise SystemExit(f"missing frozen cache: {path}")

        visual = load_cache(visual_path)
        text = load_cache(text_path)
        alignment = build_alignment_plan(visual["sample_ids"], text["sample_ids"])
        order = alignment.candidate_order
        labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
        if not np.array_equal(labels, np.asarray(text["gt_sp"])[order]):
            raise ValueError(f"labels differ after alignment: {category}")

        visual_maps = np.asarray(visual["anomaly_maps"], dtype=np.float32)
        text_maps = resize_maps(
            np.asarray(text["anomaly_maps"], dtype=np.float32)[order], visual_maps.shape[1:]
        ).astype(np.float32)
        masks = np.asarray(visual["imgs_masks"], dtype=np.uint8)
        if masks.shape != visual_maps.shape:
            raise ValueError(f"mask/map mismatch: {category}")

        ref_visual = load_ref_npz(ref_visual_path)
        ref_text = load_ref_npz(ref_text_path)
        ref_visual_maps = np.asarray(ref_visual["pixel_maps"], dtype=np.float32)
        ref_text_maps = resize_maps(
            np.asarray(ref_text["pixel_maps"], dtype=np.float32), visual_maps.shape[1:]
        ).astype(np.float32)
        if ref_visual_maps.shape[1:] != ref_text_maps.shape[1:]:
            raise ValueError(f"reference spatial mismatch: {category}")

        ri = RouterInput(
            branches={DEFAULT_ANCHOR: visual_maps, "anomalyclip_text": text_maps},
            reference_maps={DEFAULT_ANCHOR: ref_visual_maps, "anomalyclip_text": ref_text_maps},
            sample_ids=np.asarray(visual["sample_ids"]),
            category=category,
            seed=args.seed,
            shot=args.shot,
            metadata={"test_pair": pair_id, "ref_pair": ref_pair_id},
        )
        target = EvaluationTarget(
            gt_labels=labels,
            gt_masks=masks,
            sample_ids=np.asarray(visual["sample_ids"]),
        )

        # ---- methods ----
        methods: dict[str, dict] = {}
        # visual / text raw baselines
        methods["visual_only"] = {
            "maps": visual_maps.astype(np.float64),
            "leaky": False,
            "text_reliable": None,
        }
        methods["text_only"] = {
            "maps": text_maps.astype(np.float64),
            "leaky": False,
            "text_reliable": None,
        }
        # clean fixed-weight grid
        for w in CLEAN_WEIGHTS:
            fused, diag = weighted_ensemble_clean(
                ri, {DEFAULT_ANCHOR: w, "anomalyclip_text": 1.0 - w}
            )
            methods[f"v33_clean_w{w:.2f}"] = {
                "maps": fused,
                "leaky": False,
                "text_reliable": not ("anomalyclip_text" in diag.get("fallback_branches", [])),
                "diag": diag,
            }
        # visual safe fallback (text forced unreliable -> pure anchor path)
        fused_fb, diag_fb = weighted_ensemble_clean(ri, {DEFAULT_ANCHOR: 1.0})
        methods["visual_fallback"] = {
            "maps": fused_fb,
            "leaky": False,
            "text_reliable": False,
            "diag": diag_fb,
        }
        # OLD leaky V3.3 control (invalid-marked)
        old_branches = {
            "anomalydino_visual": BranchData(
                name="anomalydino_visual",
                anomaly_maps=visual_maps,
                image_scores=np.zeros(len(labels), dtype=np.float32),
                gt_labels=labels,
                gt_masks=masks,
                sample_ids=np.asarray(visual["sample_ids"]),
            ),
            "anomalyclip_text": BranchData(
                name="anomalyclip_text",
                anomaly_maps=text_maps,
                image_scores=np.zeros(len(labels), dtype=np.float32),
                gt_labels=labels,
                gt_masks=masks,
                sample_ids=np.asarray(visual["sample_ids"]),
            ),
        }
        old_maps = weighted_ensemble_fusion(
            old_branches, {"anomalydino_visual": OLD_V33_WEIGHT, "anomalyclip_text": 1.0 - OLD_V33_WEIGHT}, calibrate=True
        )
        methods["old_v33_w060_invalid"] = {"maps": old_maps, "leaky": True, "text_reliable": None}

        # reference-derived fused threshold for clean methods (reference z-score blend)
        ref_visual_stats = estimate_reference_stats(ref_visual_maps)
        ref_text_stats = estimate_reference_stats(ref_text_maps)
        ref_fused_by_w = {}
        for w in CLEAN_WEIGHTS:
            ref_fused = (
                w
                * compute_z_score(ref_visual_maps, ref_visual_stats["center"], ref_visual_stats["scale"])
                + (1.0 - w)
                * compute_z_score(ref_text_maps, ref_text_stats["center"], ref_text_stats["scale"])
            )
            ref_fused_by_w[f"v33_clean_w{w:.2f}"] = decision_threshold_from_reference(ref_fused)

        row: dict = {
            "category": category,
            "seed": args.seed,
            "shot": args.shot,
            "samples": len(labels),
            "anomaly_images": int(labels.sum()),
            "pixel_stride": STRIDE,
            "reference_views": ref_visual_maps.shape[0],
        }
        for method, entry in methods.items():
            maps = entry["maps"]
            if method == "visual_only":
                thr = decision_threshold_from_reference(ref_visual_maps)
            elif method == "text_only":
                thr = decision_threshold_from_reference(ref_text_maps)
            elif method == "visual_fallback":
                thr = decision_threshold_from_reference(ref_visual_maps)
            elif method.startswith("v33_clean"):
                thr = ref_fused_by_w[method]
            else:  # old leaky v33 -> reference-derived blended threshold (invalid control only)
                thr = decision_threshold_from_reference(
                    OLD_V33_WEIGHT * compute_z_score(ref_visual_maps, ref_visual_stats["center"], ref_visual_stats["scale"])
                    + (1.0 - OLD_V33_WEIGHT)
                    * compute_z_score(ref_text_maps, ref_text_stats["center"], ref_text_stats["scale"])
                )

            pixel = evaluate_clean(ri, target, maps, stride=STRIDE)
            img = metric_image(labels, image_scores_from_maps(maps), thr)
            prefix = method
            row[f"{prefix}_pixel_auroc"] = pixel["pixel_auroc"]
            row[f"{prefix}_pixel_ap"] = pixel["pixel_ap"]
            row[f"{prefix}_pixel_aupro"] = pixel["pixel_aupro"]
            row.update({f"{prefix}_{k}": v for k, v in img.items()})
            row[f"{prefix}_leaky"] = entry["leaky"]
            if entry["text_reliable"] is not None:
                row[f"{prefix}_text_reliable"] = entry["text_reliable"]

        # rescue / harm / coverage vs visual baseline (image level, τ reference-derived)
        v_thr = decision_threshold_from_reference(ref_visual_maps)
        visual_pred = (image_scores_from_maps(visual_maps) > v_thr).astype(np.uint8)
        for method in [m for m in methods if m.startswith("v33_clean")]:
            thr = ref_fused_by_w[method]
            fused_pred = (image_scores_from_maps(methods[method]["maps"]) > thr).astype(np.uint8)
            row[f"{method}_rescue_harm"] = rescues_harm_coverage(
                labels, visual_pred, fused_pred, bool(methods[method]["text_reliable"])
            )
            row[f"{method}_risk_coverage"] = risk_coverage_curve(
                labels, image_scores_from_maps(methods[method]["maps"]), thr
            )
        row["visual_risk_coverage"] = risk_coverage_curve(
            labels, image_scores_from_maps(visual_maps), v_thr
        )

        rows.append(row)
        provenance.extend(
            [
                {"category": category, "branch": "visual_test", "path": str(visual_path.resolve()), "sha256": sha256(visual_path)},
                {"category": category, "branch": "text_test", "path": str(text_path.resolve()), "sha256": sha256(text_path)},
                {"category": category, "branch": "visual_ref", "path": str(ref_visual_path.resolve()), "sha256": sha256(ref_visual_path)},
                {"category": category, "branch": "text_ref", "path": str(ref_text_path.resolve()), "sha256": sha256(ref_text_path)},
            ]
        )

    # ---- aggregates & gate decision ----
    agg: dict = {"mean": {}, "positive_vs_visual": {}, "max_regression": {}}
    for method in [m for m in methods]:
        px_ap = [row[f"{method}_pixel_ap"] for row in rows]
        px_auroc = [row[f"{method}_pixel_auroc"] for row in rows]
        px_aupro = [row[f"{method}_pixel_aupro"] for row in rows]
        im_auroc = [row[f"{method}_image_auroc"] for row in rows]
        im_ap = [row[f"{method}_image_ap"] for row in rows]
        im_f1 = [row[f"{method}_image_f1"] for row in rows]
        agg["mean"][method] = {
            "pixel_ap": float(np.nanmean(px_ap)),
            "pixel_auroc": float(np.nanmean(px_auroc)),
            "pixel_aupro": float(np.nanmean(px_aupro)),
            "image_auroc": float(np.nanmean(im_auroc)),
            "image_ap": float(np.nanmean(im_ap)),
            "image_f1": float(np.nanmean(im_f1)),
        }
        if method != "visual_only":
            deltas = [row[f"{method}_pixel_ap"] - row["visual_only_pixel_ap"] for row in rows]
            agg["positive_vs_visual"][method] = {
                "positive_count": int(sum(d > 0 for d in deltas)),
                "mean_delta": float(np.mean(deltas)),
                "max_regression": float(min(deltas)),
            }
    visual_mean_px_ap = agg["mean"]["visual_only"]["pixel_ap"]
    gate_pass_clean = [
        m for m in [f"v33_clean_w{w:.2f}" for w in CLEAN_WEIGHTS]
        if agg["mean"][m]["pixel_ap"] > visual_mean_px_ap
        and agg["positive_vs_visual"][m]["positive_count"] >= 4
        and agg["positive_vs_visual"][m]["max_regression"] > -0.02
        and agg["mean"][m]["pixel_aupro"] >= agg["mean"]["visual_only"]["pixel_aupro"]
    ]
    gate_decisions = {
        "visual_mean_pixel_ap": float(visual_mean_px_ap),
        "clean_configs_passing_suggested_gate": gate_pass_clean,
        "suggested_gate_passed": bool(gate_pass_clean),
        "decision_rule": (
            "mean pixel AP > visual, >=4/6 categories positive, max regression > -0.02, "
            "AUPRO not down overall, audit & repeatability pass"
        ),
    }

    report = {
        "schema_version": 1,
        "run_id": "v3_3_clean_gate_20260817_mpdd_s0_k1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "phase_3_mpdd_s0_k1_cpu_gate",
        "dataset": "mpdd",
        "dataset_role": "development",
        "seed": args.seed,
        "shot": args.shot,
        "gpu": False,
        "pre_registered_grid": {
            "clean_visual_weights": CLEAN_WEIGHTS,
            "old_v33_invalid_control_weight": OLD_V33_WEIGHT,
            "image_threshold_rule": "image_score > max(normal-reference image scores)",
            "pixel_stride": STRIDE,
            "risk_coverage_targets": RISK_COVERAGE_TARGETS,
        },
        "leakage_flags": {
            "test_predictions_used": False,
            "test_labels_used": False,
            "test_masks_used": False,
            "test_dataset_statistics_used": False,
            "test_normal_selection_used": False,
        },
        "old_v33_control_note": (
            "old_v33_w060_invalid uses gt_masks-based calibration and is "
            "development_only_leaky_calibration=true / paper_eligible=false"
        ),
        "adaptclip_note": "no MPDD AdaptCLIP cache exists; excluded per plan (cache/protocol pass required)",
        "v32_note": "V3.2 is archived/superseded (scripts/evaluate_v3_2_gate_b.py ARCHIVED); not re-run",
        "gate": gate_decisions,
        "aggregate": agg,
        "rows": rows,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "gate": gate_decisions, "n_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
