"""G2 visual-anchor smoke: `subspace_style_same_backbone` (CPU, read-only).

V1 candidate (NOT an official SubspaceAD reproduction): reuse the frozen DINO
raw patch cache, fit a PCA normal subspace on the K normal-reference patches of
the current (seed, shot, category) only, and score each test patch by its
reconstruction residual ||x - x_recon||_2. No backbone change, no re-export, no
GPU. PCA energy threshold is restricted to {0.95, 0.99}; augmentation count is
0. Map/evaluator mirror the matched DINO-only protocol (448x448 map, stride=8).

This script does NOT import `evaluate_a1_feature_fusion` (whose top-level
`from src.utils import dists2map` currently has no backing module); it
re-implements the small, self-contained pieces it needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast  # noqa: E402

STRIDE = 8
LEAKAGE_FLAGS = {
    "test_predictions_used_for_parameter_fit": False,
    "test_labels_used_for_parameter_fit": False,
    "test_masks_used_for_parameter_fit": False,
    "test_dataset_statistics_used_for_calibration": False,
    "test_normal_selection_used": False,
}


def load_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "patch_features": np.asarray(data["patch_features"], dtype=np.float32),
            "ref_patch_features": np.asarray(data["ref_patch_features"], dtype=np.float32),
            "sample_ids": np.asarray(data["sample_ids"]),
            "gt_sp": np.asarray(data["gt_sp"], dtype=np.uint8),
            "imgs_masks": np.asarray(data["imgs_masks"], dtype=np.uint8),
            "grid_size": tuple(int(v) for v in data["grid_size"]),
        }


def compute_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    maps_strided = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_strided = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_strided.ravel()
    flat_labels = (masks_strided.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_strided, maps_strided)),
    }


def subspace_maps(feat: np.ndarray, ref: np.ndarray, pca_ev: float, grid: tuple[int, int], map_size: tuple[int, int]) -> np.ndarray:
    """Fit PCA on normal refs, score test patches by reconstruction residual."""
    n = feat.shape[0]
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float64)
    ref_flat = ref.reshape(-1, d).astype(np.float64)

    pca = PCA(n_components=pca_ev, svd_solver="full", random_state=0)
    pca.fit(ref_flat)
    recon = pca.inverse_transform(pca.transform(feat_flat))
    residual = feat_flat - recon
    score = np.linalg.norm(residual, axis=1).astype(np.float32).reshape(n, *grid)

    maps = np.stack([cv2.resize(s, map_size[::-1], interpolation=cv2.INTER_LINEAR) for s in score])
    return maps.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--shot", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--mode", choices=("subspace_style_same_backbone",), default="subspace_style_same_backbone")
    parser.add_argument("--pca-ev", type=float, choices=[0.95, 0.99], default=0.99)
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    map_size = (args.map_size, args.map_size)
    feats_root = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
    dino_dir = feats_root / f"features_vitb14_s{args.seed}_k{args.shot}" / "anomalydino_visual"

    if args.dataset != "mpdd":
        raise SystemExit("G2 smoke is MPDD-only per the pre-registered plan")
    if not dino_dir.is_dir():
        raise SystemExit(f"missing DINO cache dir: {dino_dir}")

    cats = sorted(p.stem for p in dino_dir.glob("*.npz"))
    if not cats:
        raise SystemExit(f"no DINO npz in {dino_dir}")
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", "categories": cats}))
        return 0

    out_dir = args.output_dir or (
        ROOT / "experiments" / "dynamic_fusion" / "v4_vision_text_20260819" / "02_visual_gate" / "s0_k1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat in cats:
        d = load_features(dino_dir / f"{cat}.npz")
        maps = subspace_maps(d["patch_features"], d["ref_patch_features"], args.pca_ev, d["grid_size"], map_size)
        metrics = compute_metrics(maps.astype(np.float64), d["imgs_masks"])
        rows.append({"category": cat, **metrics})
        print(f"  {cat}: P-AP={metrics['pixel_ap']:.4f} P-AUROC={metrics['pixel_auroc']:.4f} "
              f"AUPRO={metrics['pixel_aupro']:.4f}", flush=True)

    mean = {k: round(float(np.mean([r[k] for r in rows])), 6) for k in ("pixel_auroc", "pixel_ap", "pixel_aupro")}
    report = {
        "run_id": f"v4_g2_visual_anchor_smoke_s{args.seed}_k{args.shot}",
        "pipeline": "v4_visual_anchor",
        "mode": args.mode,
        "dataset": args.dataset,
        "dataset_role": "development",
        "seed": args.seed,
        "shot": args.shot,
        "pca_ev": args.pca_ev,
        "augmentation_count": 0,
        "backbone": "dinov2_vitb14 (frozen DINO raw patch cache, no re-export)",
        "anomaly_score": "l2 reconstruction residual in normal PCA subspace",
        "map_size": args.map_size,
        "stride": STRIDE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mean": mean,
        "per_category": rows,
        "leakage_flags": LEAKAGE_FLAGS,
    }
    (out_dir / "metrics_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "marker.json").write_text(json.dumps({
        "run_id": report["run_id"],
        "status": "smoke_complete",
        "gate_passed": None,
        "paper_eligible": False,
        "dataset_role": "development",
        "seed": args.seed,
        "shot": args.shot,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"mean": mean, "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
