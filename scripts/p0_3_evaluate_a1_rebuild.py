"""P0-3: bounded rebuild of the A1 concat vs matched feature-DINO-only evidence.

Standalone (does NOT modify frozen scripts). Mirrors evaluate_a1_visa_frozen.py:
for every (seed, shot) and every category it computes BOTH the frozen A1 concat
KNN map and the matched feature-DINO-only KNN map with the exact same pipeline
(dual L2, concat w=0.5, KNN k=1, distance/2, map 448, stride 8), then reports
per-category and mean Pixel-AP deltas. No v2 score cache is required.

Layouts:
  mpdd/btad: outputs/dynamic_fusion/v3_direction_a/features_vitb14_s{S}_k{K}/anomalydino_visual
             outputs/dynamic_fusion/v3_direction_a/features_s{S}_k{K}/anomalyclip_text
  visa/mvtec: outputs/dynamic_fusion/v3_direction_a/{ds}_features_vitb14/s{S}_k{K}/anomalydino_visual
             outputs/dynamic_fusion/v3_direction_a/{ds}_features/s{S}_k{K}/anomalyclip_text
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_a1_feature_fusion import STRIDE, compute_metrics, fuse_category, load_features

FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"


def dirs_for(dataset: str, seed: int, shot: int) -> tuple[Path, Path]:
    if dataset == "mpdd":
        dino = FEATURES_ROOT / f"features_vitb14_s{seed}_k{shot}" / "anomalydino_visual"
        clip = FEATURES_ROOT / f"features_s{seed}_k{shot}" / "anomalyclip_text"
    elif dataset == "btad":
        dino = FEATURES_ROOT / f"features_vitb14_btad_s{seed}_k{shot}" / "anomalydino_visual"
        clip = FEATURES_ROOT / f"features_btad_s{seed}_k{shot}" / "anomalyclip_text"
    else:  # visa / mvtec
        dino = FEATURES_ROOT / f"{dataset}_features_vitb14" / f"s{seed}_k{shot}" / "anomalydino_visual"
        clip = FEATURES_ROOT / f"{dataset}_features" / f"s{seed}_k{shot}" / "anomalyclip_text"
    return dino, clip


def evaluate_config(dino_dir: Path, clip_dir: Path, dataset: str, seed: int, shot: int,
                    map_size: tuple[int, int]) -> dict:
    results = []
    for cat_path in sorted(dino_dir.glob("*.npz")):
        if cat_path.name == "export_report.json":
            continue
        cat = cat_path.stem
        clip_path = clip_dir / f"{cat}.npz"
        if not clip_path.is_file():
            raise SystemExit(f"missing clip features: {clip_path}")
        dino = load_features(cat_path)
        clip = load_features(clip_path)

        concat_maps = fuse_category(dino, clip, "concat", pca_dim=0, whiten=False,
                                    map_size=map_size, dino_weight=0.5)
        concat_metrics = compute_metrics(concat_maps.astype(np.float64), dino["imgs_masks"])
        dino_maps = fuse_category(dino, clip, "dino", pca_dim=0, whiten=False,
                                  map_size=map_size, dino_weight=0.5)
        dino_metrics = compute_metrics(dino_maps.astype(np.float64), dino["imgs_masks"])
        results.append(
            {
                "category": cat,
                "concat": concat_metrics,
                "feature_dino_only": dino_metrics,
                "delta_ap": round(concat_metrics["pixel_ap"] - dino_metrics["pixel_ap"], 6),
                "delta_auroc": round(concat_metrics["pixel_auroc"] - dino_metrics["pixel_auroc"], 6),
                "delta_aupro": round(concat_metrics["pixel_aupro"] - dino_metrics["pixel_aupro"], 6),
            }
        )
    mean = {
        key: round(float(np.mean([r["concat"][key] for r in results])), 6)
        for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "v3_direction_a_a1",
        "config": "concat pca_dim=0 whiten=0 dino_weight=0.5 KNN k=1 stride=8 map=448 (dinov2_vitb14 + AnomalyCLIP)",
        "dataset": dataset,
        "seed": seed,
        "shot": shot,
        "mean_concat_pixel_ap": mean["pixel_ap"],
        "mean_feature_dino_only_pixel_ap": round(
            float(np.mean([r["feature_dino_only"]["pixel_ap"] for r in results])), 6),
        "mean_delta_ap_vs_feature_dino": round(
            float(np.mean([r["delta_ap"] for r in results])), 6),
        "positive_categories": int(sum(1 for r in results if r["delta_ap"] > 0)),
        "baseline_source": "matched feature-level dino-only KNN (same pipeline)",
        "per_category": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mpdd", "btad", "visa", "mvtec"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "p0_rebuild_20260826")
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    map_size = (args.map_size, args.map_size)
    jobs = []
    for seed in args.seeds:
        for shot in args.shots:
            dino_dir, clip_dir = dirs_for(args.dataset, seed, shot)
            out = args.output_dir / f"{args.dataset}_s{seed}_k{shot}.json"
            jobs.append((seed, shot, dino_dir, clip_dir, out))

    missing = []
    for seed, shot, dino_dir, clip_dir, _ in jobs:
        d_cats = {p.stem for p in dino_dir.glob("*.npz") if p.stem != "export_report"}
        c_cats = {p.stem for p in clip_dir.glob("*.npz") if p.stem != "export_report"}
        if not d_cats or d_cats != c_cats:
            missing.append(
                f"s{seed}_k{shot}: dino={sorted(d_cats)} clip={sorted(c_cats)}")
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", "jobs": len(jobs)}))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, shot, dino_dir, clip_dir, out in jobs:
        report = evaluate_config(dino_dir, clip_dir, args.dataset, seed, shot, map_size)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{args.dataset} s{seed}/k{shot}] concat AP {report['mean_concat_pixel_ap']:.4f} | "
            f"dino AP {report['mean_feature_dino_only_pixel_ap']:.4f} | "
            f"ΔAP {report['mean_delta_ap_vs_feature_dino']:+.6f} ({report['positive_categories']}/{len(report['per_category'])} pos)",
            flush=True,
        )
        rows.append(report)

    overall_delta = round(float(np.mean([r["mean_delta_ap_vs_feature_dino"] for r in rows])), 6)
    overall_concat = round(float(np.mean([r["mean_concat_pixel_ap"] for r in rows])), 6)
    overall_dino = round(float(np.mean([r["mean_feature_dino_only_pixel_ap"] for r in rows])), 6)
    positive = int(sum(1 for r in rows if r["mean_delta_ap_vs_feature_dino"] > 0))
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "n_configs": len(rows),
        "overall_mean_concat_pixel_ap": overall_concat,
        "overall_mean_feature_dino_only_pixel_ap": overall_dino,
        "overall_mean_delta_ap_vs_feature_dino": overall_delta,
        "positive_configs": positive,
        "all_positive": positive == len(rows),
        "note": "9/9 means 9 reference-sampling configs on one test set, not 9 independent datasets.",
    }
    (args.output_dir / f"{args.dataset}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
