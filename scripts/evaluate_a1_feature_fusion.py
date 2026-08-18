"""Direction A (feature-level dynamic fusion) — A1 evaluation script.

Loads the raw patch features exported by export_anomalydino_mpdd_features.py and
export_anomalyclip_mpdd_features.py, spatially aligns the two grids, concatenates
the tokens, fits a PCA/whitening on the normal references only, and scores each
test patch via a shared KNN memory bank (distance = anomaly).

Modes:
  dino   -> DINO features only (PCA + KNN, sanity check vs DINO baseline)
  clip   -> CLIP features only (PCA + KNN)
  concat -> DINO || CLIP concatenation (the actual A1 direction)

Leakage discipline mirrors the V2 gate protocol: PCA/whitening and the memory
bank are fit on normal references only; no test labels are used for fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from evaluate_unified import aupro_fast
from industrial_ad.fusion.alignment import build_alignment_plan
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize
from src.utils import dists2map

STRIDE: int = 8


# ======================================================================
# Feature loading & fusion
# ======================================================================

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


def resize_patches(patches: np.ndarray, target_grid: tuple[int, int]) -> np.ndarray:
    """[N, H, W, D] -> [N, th, tw, D] via bilinear interpolation."""
    h, w = patches.shape[1], patches.shape[2]
    if (h, w) == target_grid:
        return patches
    x = torch.from_numpy(patches).permute(0, 3, 1, 2)  # [N, D, H, W]
    x = torch.nn.functional.interpolate(
        x, size=target_grid, mode="bilinear", align_corners=False
    )
    return x.permute(0, 2, 3, 1).numpy()


def score_via_memory(
    feat_flat: np.ndarray,
    ref_flat: np.ndarray,
    pca_dim: int,
    whiten: bool,
    n_images: int,
    grid: tuple[int, int],
    map_size: tuple[int, int],
) -> np.ndarray:
    """Fit PCA/whitening on refs, build KNN memory, return [N, map_h, map_w] maps.

    pca_dim <= 0 skips PCA entirely (direct L2-normalize + KNN), which mirrors the
    DINO/AnomalyCLIP baselines.
    """
    if pca_dim > 0:
        pca = PCA(n_components=pca_dim, whiten=whiten, svd_solver="randomized", random_state=0)
        pca.fit(ref_flat)
        ref_proj = pca.transform(ref_flat).astype(np.float32)
        feat_proj = pca.transform(feat_flat).astype(np.float32)
    else:
        ref_proj = ref_flat.astype(np.float32)
        feat_proj = feat_flat.astype(np.float32)

    faiss.normalize_L2(ref_proj)
    faiss.normalize_L2(feat_proj)
    index = faiss.IndexFlatL2(ref_proj.shape[1])
    index.add(ref_proj)
    distances, _ = index.search(feat_proj, k=1)
    dists = (distances[:, 0] / 2.0).reshape(n_images, *grid)

    maps = np.stack([dists2map(d, map_size) for d in dists]).astype(np.float32)
    return maps


def fuse_category(
    dino: dict,
    clip: dict | None,
    mode: str,
    pca_dim: int,
    whiten: bool,
    map_size: tuple[int, int],
    dino_weight: float = 0.5,
) -> np.ndarray:
    grid = dino["grid_size"]

    if mode == "dino":
        feat = dino["patch_features"]
        ref = dino["ref_patch_features"]
    elif mode == "clip":
        feat = clip["patch_features"]
        ref = clip["ref_patch_features"]
        grid = clip["grid_size"]
    elif mode == "concat":
        alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])
        clip_feat = clip["patch_features"][alignment.candidate_order]
        clip_ref = clip["ref_patch_features"]
        clip_feat = resize_patches(clip_feat, grid)
        clip_ref = resize_patches(clip_ref, grid)

        dino_feat = dino["patch_features"]
        dino_ref = dino["ref_patch_features"]
        dino_feat = normalize(dino_feat.reshape(-1, dino_feat.shape[-1])).reshape(dino_feat.shape)
        dino_ref = normalize(dino_ref.reshape(-1, dino_ref.shape[-1])).reshape(dino_ref.shape)
        clip_feat = normalize(clip_feat.reshape(-1, clip_feat.shape[-1])).reshape(clip_feat.shape)
        clip_ref = normalize(clip_ref.reshape(-1, clip_ref.shape[-1])).reshape(clip_ref.shape)

        feat = np.concatenate([dino_weight * dino_feat, (1.0 - dino_weight) * clip_feat], axis=-1)
        ref = np.concatenate([dino_weight * dino_ref, (1.0 - dino_weight) * clip_ref], axis=-1)
    else:
        raise ValueError(f"unknown mode: {mode}")

    n = feat.shape[0]
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d)
    ref_flat = ref.reshape(-1, d)
    return score_via_memory(feat_flat, ref_flat, pca_dim, whiten, n, grid, map_size)


# ======================================================================
# Metrics
# ======================================================================

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


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="A1 feature-level fusion evaluation")
    parser.add_argument("--dino-features", type=Path, required=True)
    parser.add_argument("--clip-features", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="v2 prediction cache dir holding anomalydino_visual/anomalyclip_text anomaly_maps")
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--mode", choices=("dino", "clip", "concat"), default="concat")
    parser.add_argument("--pca-dim", type=int, default=-1,
                        help="PCA target dim; <=0 disables PCA (direct normalize+KNN)")
    parser.add_argument("--dino-weight", type=float, default=0.5,
                        help="weight on the DINO branch in concat mode (CLIP gets 1-w)")
    parser.add_argument("--whiten", type=int, default=1, choices=[0, 1])
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    whiten = bool(args.whiten)
    map_size = (args.map_size, args.map_size)

    manifest = json.loads(
        (ROOT / "data" / "splits" / args.dataset / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])

    out_dir = args.output_dir or (
        ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_fusion" / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = args.baseline_dir
    results = []
    print(f"mode={args.mode} pca_dim={args.pca_dim} whiten={whiten}", flush=True)

    for cat in categories:
        dino_feat_path = args.dino_features / f"{cat}.npz"
        clip_feat_path = args.clip_features / f"{cat}.npz"
        if not dino_feat_path.exists() or not clip_feat_path.exists():
            print(f"  [SKIP] missing features for {cat}", flush=True)
            continue

        dino = load_features(dino_feat_path)
        clip = load_features(clip_feat_path)

        fused_maps = fuse_category(dino, clip, args.mode, args.pca_dim, whiten, map_size, args.dino_weight)
        fused_metrics = compute_metrics(fused_maps.astype(np.float64), dino["imgs_masks"])

        # Baselines from the existing v2 scalar-map cache
        baselines = {}
        for bname, rel in (("anomalydino_visual", "anomalydino_visual"),
                           ("anomalyclip_text", "anomalyclip_text")):
            p = baseline_dir / rel / f"{cat}.npz"
            if p.exists():
                with np.load(p, allow_pickle=False) as data:
                    bmaps = np.asarray(data["anomaly_maps"], dtype=np.float32)
                    bmasks = np.asarray(data["imgs_masks"], dtype=np.uint8)
                baselines[bname] = compute_metrics(bmaps.astype(np.float64), bmasks)

        dino_ap = baselines.get("anomalydino_visual", {}).get("pixel_ap", float("nan"))
        row = {
            "category": cat,
            "mode": args.mode,
            "pca_dim": args.pca_dim,
            "whiten": whiten,
            "fused": fused_metrics,
            "baselines": baselines,
            "delta_auroc": round(fused_metrics["pixel_auroc"] - baselines.get("anomalydino_visual", {}).get("pixel_auroc", 0), 6),
            "delta_ap": round(fused_metrics["pixel_ap"] - dino_ap, 6),
            "delta_aupro": round(fused_metrics["pixel_aupro"] - baselines.get("anomalydino_visual", {}).get("pixel_aupro", 0), 6),
        }
        results.append(row)
        print(f"  {cat}: fused AP={fused_metrics['pixel_ap']:.4f} AUROC={fused_metrics['pixel_auroc']:.4f} "
              f"AUPRO={fused_metrics['pixel_aupro']:.4f} | "
              f"DINO AP={baselines.get('anomalydino_visual',{}).get('pixel_ap',float('nan')):.4f} "
              f"CLIP AP={baselines.get('anomalyclip_text',{}).get('pixel_ap',float('nan')):.4f} "
              f"(ΔAP={row['delta_ap']:+.4f})", flush=True)

    mean = {
        key: round(float(np.mean([r["fused"][key] for r in results])), 6)
        for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
    }
    mean_delta_ap = round(float(np.mean([r["delta_ap"] for r in results])), 6)
    mean_dino_ap = round(float(np.mean([r["baselines"]["anomalydino_visual"]["pixel_ap"] for r in results])), 6)

    report = {
        "pipeline": "v3_direction_a_a1",
        "direction": "A_feature_level_fusion",
        "mode": args.mode,
        "pca_dim": args.pca_dim,
        "whiten": whiten,
        "seed": args.seed,
        "stride": STRIDE,
        "dataset": args.dataset,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mean_fused": mean,
        "mean_dino_baseline_ap": mean_dino_ap,
        "mean_delta_ap_vs_dino": mean_delta_ap,
        "per_category": results,
    }
    report_path = out_dir / f"{args.mode}_pca{args.pca_dim}_whiten{int(whiten)}_w{args.dino_weight:g}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Mean fused: {mean}")
    print(f"Mean DINO baseline AP: {mean_dino_ap}, mean ΔAP = {mean_delta_ap:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
