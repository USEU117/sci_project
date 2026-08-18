"""A1 frozen-config evaluation on VisA (post-freeze validation, 阶段七).

Baseline note: VisA has NO v2 score-level prediction cache, so the DINO
baseline here is the A1 *feature-level* dino-only KNN map (same pipeline, same
normal-reference memory bank). On MPDD s0/K1 the feature-level dino-only KNN
matches the v2 score-level baseline to within ~0.0008 AP, so the two deltas are
comparable. The report always records the baseline source explicitly.

New standalone script: does NOT touch the frozen A1 scripts in freeze_manifest.json
(imports functions from evaluate_a1_feature_fusion.py read-only).
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

SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]
FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
# dataset -> (dino prefix, clip prefix, experiment dir name)
DATASET_PREFIX = {
    "visa": ("visa_features_vitb14", "visa_features", "a1_visa_20260818"),
    "mvtec": ("mvtec_features_vitb14", "mvtec_features", "a1_mvtec_20260818"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("visa", "mvtec"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    parser.add_argument("--modes", choices=("concat", "dino", "clip"), nargs="+", default=["concat", "dino", "clip"])
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    dino_prefix, clip_prefix, exp_name = DATASET_PREFIX[args.dataset]
    dino_root = FEATURES_ROOT / dino_prefix
    clip_root = FEATURES_ROOT / clip_prefix
    EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / exp_name
    jobs = []
    for seed in args.seeds:
        for shot in args.shots:
            dino_dir = dino_root / f"s{seed}_k{shot}" / "anomalydino_visual"
            clip_dir = clip_root / f"s{seed}_k{shot}" / "anomalyclip_text"
            out_dir = EXPERIMENT_ROOT / f"seed{seed}_k{shot}"
            for mode in args.modes:
                marker = out_dir / f"{mode}_pca0_whiten0_w0.5_report.json"
                jobs.append(
                    {
                        "seed": seed,
                        "shot": shot,
                        "mode": mode,
                        "dino_dir": dino_dir,
                        "clip_dir": clip_dir,
                        "out_dir": out_dir,
                        "marker": marker,
                    }
                )

    missing = []
    seen = set()
    for job in jobs:
        key = (job["seed"], job["shot"])
        if key in seen:
            continue
        seen.add(key)
        for name, d in (("dino_dir", job["dino_dir"]), ("clip_dir", job["clip_dir"])):
            n_npz = len(list(d.glob("*.npz")))
            if n_npz < 12:
                missing.append(f"{name} {d} only {n_npz} npz")
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", "jobs": len(jobs)}))
        return 0

    map_size = (args.map_size, args.map_size)
    completed, failed = [], []
    for job in jobs:
        if job["marker"].is_file():
            completed.append({"seed": job["seed"], "shot": job["shot"], "mode": job["mode"], "status": "cached"})
            continue
        report = evaluate_config(job, map_size, args.dataset)
        job["out_dir"].mkdir(parents=True, exist_ok=True)
        job["marker"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[s{job['seed']}/k{job['shot']}/{job['mode']}] "
            f"mean fused AP {report['mean_fused']['pixel_ap']:.4f} | "
            f"DINO feature AP {report['mean_dino_baseline_ap']:.4f} | "
            f"ΔAP {report['mean_delta_ap_vs_dino']:+.4f}",
            flush=True,
        )
        completed.append({"seed": job["seed"], "shot": job["shot"], "mode": job["mode"], "status": "ok"})

    print(json.dumps({"completed": len(completed), "failed": failed}))
    return 0 if not failed else 1


def evaluate_config(job: dict, map_size: tuple[int, int], dataset: str) -> dict:
    results = []
    for cat_path in sorted(job["dino_dir"].glob("*.npz")):
        if cat_path.name == "export_report.json":
            continue
        cat = cat_path.stem
        clip_path = job["clip_dir"] / f"{cat}.npz"
        if not clip_path.is_file():
            raise SystemExit(f"missing clip features: {clip_path}")
        dino = load_features(cat_path)
        clip = load_features(clip_path)

        mode = job["mode"]
        maps = fuse_category(dino, clip, mode, pca_dim=0, whiten=False, map_size=map_size, dino_weight=0.5)
        fused_metrics = compute_metrics(maps.astype(np.float64), dino["imgs_masks"])

        if mode == "dino":
            baseline_metrics = fused_metrics
        else:
            # feature-level dino-only KNN = the VisA DINO baseline (no v2 cache).
            dino_maps = fuse_category(dino, clip, "dino", pca_dim=0, whiten=False, map_size=map_size, dino_weight=0.5)
            baseline_metrics = compute_metrics(dino_maps.astype(np.float64), dino["imgs_masks"])

        results.append(
            {
                "category": cat,
                "mode": mode,
                "pca_dim": 0,
                "whiten": False,
                "fused": fused_metrics,
                "baselines": {"anomalydino_visual_feature_knn": baseline_metrics},
                "delta_ap": round(fused_metrics["pixel_ap"] - baseline_metrics["pixel_ap"], 6),
                "delta_auroc": round(fused_metrics["pixel_auroc"] - baseline_metrics["pixel_auroc"], 6),
                "delta_aupro": round(fused_metrics["pixel_aupro"] - baseline_metrics["pixel_aupro"], 6),
            }
        )
    mean = {
        key: round(float(np.mean([r["fused"][key] for r in results])), 6)
        for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
    }
    return {
        "pipeline": "v3_direction_a_a1",
        "direction": "A_feature_level_fusion",
        "mode": job["mode"],
        "pca_dim": 0,
        "whiten": False,
        "seed": job["seed"],
        "stride": STRIDE,
        "dataset": dataset,
        "dataset_role": "holdout",
        "baseline_source": "feature_level_dino_only_knn (no v2 score cache on this holdout)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mean_fused": mean,
        "mean_dino_baseline_ap": round(float(np.mean([r["baselines"]["anomalydino_visual_feature_knn"]["pixel_ap"] for r in results])), 6),
        "mean_delta_ap_vs_dino": round(float(np.mean([r["delta_ap"] for r in results])), 6),
        "per_category": results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
