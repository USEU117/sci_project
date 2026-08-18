"""A1 dynamic-vs-fixed comparison on MPDD (docs 阶段三决策 / 阶段五 5.2).

Dynamic router (reference-only, per category):
    - For each branch, build a self-KNN index on its K normal-reference patch
      features (L2-normalized) and measure "compactness" = mean second-nearest
      L2 distance. A more compact reference set is treated as more reliable.
    - Pre-registered weight tiers: dino_weight in {0.4, 0.5, 0.6} selected by
      the compactness ratio with thresholds {1.15, 0.85}:
          dino/clip < 0.85  -> w=0.6 (DINO more compact)
          dino/clip > 1.15  -> w=0.4 (CLIP more compact)
          otherwise         -> w=0.5 (equal)
    - No test labels/masks/stats are used for routing; only normal references.

Fixed control: dino_weight = 0.5 on every category (frozen A1 config).

Reports per category: selected weight, compactness, fused metrics; and
aggregates for dynamic vs fixed (mean Pixel AP / Δ vs DINO).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_a1_feature_fusion import (  # noqa: E402
    compute_metrics,
    fuse_category,
    load_features,
)
from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402

STRIDE = 8
WEIGHT_TIERS = [0.4, 0.5, 0.6]
RATIO_UPPER = 1.15  # dino/clip above this -> clip more compact -> w=0.4
RATIO_LOWER = 0.85  # dino/clip below this -> dino more compact -> w=0.6


def ref_compactness(ref_patches: np.ndarray) -> float:
    """Mean second-nearest L2 distance within the normal-reference patch set.

    ref_patches: [K,H,W,D]. L2-normalized before indexing. The first nearest is
    the patch itself (distance ~0); the second nearest measures how tightly the
    reference set clusters. Lower = more compact = more reliable.
    """
    flat = np.asarray(ref_patches, dtype=np.float32).reshape(-1, ref_patches.shape[-1])
    faiss.normalize_L2(flat)
    index = faiss.IndexFlatL2(flat.shape[1])
    index.add(flat)
    k = min(2, len(flat))
    distances, _ = index.search(flat, k=k)
    second = distances[:, -1] if distances.shape[1] >= 2 else distances[:, 0]
    return float(np.mean(second))


def select_dynamic_weight(dino_compact: float, clip_compact: float) -> float:
    ratio = dino_compact / max(clip_compact, 1e-8)
    if ratio > RATIO_UPPER:
        return 0.4  # CLIP reference is more compact
    if ratio < RATIO_LOWER:
        return 0.6  # DINO reference is more compact
    return 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    features_root = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
    baseline_root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions"
    out_path = args.output or (
        ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_dynamic_vs_fixed_20260817" / "report.json"
    )
    manifest = json.loads((ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])

    rows = []
    for seed in args.seeds:
        for shot in args.shots:
            dino_dir = features_root / f"features_vitb14_s{seed}_k{shot}" / "anomalydino_visual"
            clip_dir = features_root / f"features_s{seed}_k{shot}" / "anomalyclip_text"
            baseline_dir = baseline_root / f"v2_mpdd_s{seed}_k{shot}_full_v1"
            for cat in categories:
                dino_path = dino_dir / f"{cat}.npz"
                clip_path = clip_dir / f"{cat}.npz"
                if not dino_path.is_file() or not clip_path.is_file():
                    print(f"[SKIP] missing features {seed}/{shot}/{cat}", flush=True)
                    continue
                dino = load_features(dino_path)
                clip = load_features(clip_path)

                dino_compact = ref_compactness(dino["ref_patch_features"])
                clip_compact = ref_compactness(clip["ref_patch_features"])
                dyn_w = select_dynamic_weight(dino_compact, clip_compact)

                fixed_maps = fuse_category(dino, clip, "concat", 0, False, (448, 448), 0.5)
                dyn_maps = fuse_category(dino, clip, "concat", 0, False, (448, 448), dyn_w)
                fixed_metrics = compute_metrics(fixed_maps.astype(np.float64), dino["imgs_masks"])
                dyn_metrics = compute_metrics(dyn_maps.astype(np.float64), dino["imgs_masks"])

                with np.load(baseline_dir / "anomalydino_visual" / f"{cat}.npz", allow_pickle=False) as data:
                    bmaps = np.asarray(data["anomaly_maps"], dtype=np.float32)
                    bmasks = np.asarray(data["imgs_masks"], dtype=np.uint8)
                dino_baseline = compute_metrics(bmaps.astype(np.float64), bmasks)

                rows.append(
                    {
                        "seed": seed,
                        "shot": shot,
                        "category": cat,
                        "dino_compactness": dino_compact,
                        "clip_compactness": clip_compact,
                        "compactness_ratio": round(dino_compact / max(clip_compact, 1e-8), 4),
                        "dynamic_weight": dyn_w,
                        "fixed_weight": 0.5,
                        "fixed_pixel_ap": fixed_metrics["pixel_ap"],
                        "dynamic_pixel_ap": dyn_metrics["pixel_ap"],
                        "dino_baseline_pixel_ap": dino_baseline["pixel_ap"],
                        "fixed_delta_ap": round(fixed_metrics["pixel_ap"] - dino_baseline["pixel_ap"], 6),
                        "dynamic_delta_ap": round(dyn_metrics["pixel_ap"] - dino_baseline["pixel_ap"], 6),
                        "dynamic_minus_fixed_delta": round(dyn_metrics["pixel_ap"] - fixed_metrics["pixel_ap"], 6),
                    }
                )
            print(f"[s{seed}/k{shot}] done", flush=True)

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    def by_weight():
        return {
            str(w): {
                "count": int(sum(1 for r in rows if r["dynamic_weight"] == w)),
                "mean_dyn_minus_fixed": round(
                    float(np.mean([r["dynamic_minus_fixed_delta"] for r in rows if r["dynamic_weight"] == w])), 6
                ),
            }
            for w in WEIGHT_TIERS
        }

    report = {
        "schema_version": 1,
        "run_id": "a1_dynamic_vs_fixed_20260817",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "dynamic_vs_fixed_comparison",
        "dataset": "mpdd",
        "dataset_role": "development",
        "dynamic_rule": {
            "type": "reference_self_knn_compactness",
            "weight_tiers": WEIGHT_TIERS,
            "ratio_upper": RATIO_UPPER,
            "ratio_lower": RATIO_LOWER,
        },
        "leakage_flags": {
            "test_predictions_used": False,
            "test_labels_used": False,
            "test_masks_used": False,
            "test_dataset_statistics_used": False,
            "test_normal_selection_used": False,
        },
        "aggregate": {
            "n_rows": len(rows),
            "fixed_mean_delta_ap": mean("fixed_delta_ap"),
            "dynamic_mean_delta_ap": mean("dynamic_delta_ap"),
            "dynamic_minus_fixed_mean": mean("dynamic_minus_fixed_delta"),
            "dynamic_wins": int(sum(1 for r in rows if r["dynamic_minus_fixed_delta"] > 0)),
            "by_dynamic_weight": by_weight(),
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
