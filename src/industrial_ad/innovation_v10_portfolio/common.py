"""v10 portfolio shared infrastructure.

Replicates the frozen A1 protocol exactly (concat w=0.5, pca_dim=0, whiten=0,
per-patch L2-normalize per branch -> concat -> faiss.normalize_L2 -> IndexFlatL2
k=1 -> distance/2.0), but keeps the memory bank decomposed by reference IMAGE so
that per-reference distances d_r(q) = min over patches of reference image r are
available (needed by CRAM / NORC region calibration).

faiss is imported lazily so modules can be imported in any venv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.nn import functional as F
except Exception:  # pragma: no cover - torch not always present at import time
    torch = None
    F = None

from industrial_ad.fusion.alignment import build_alignment_plan
from sklearn.preprocessing import normalize

STRIDE: int = 8
MAP_SIZE: tuple[int, int] = (448, 448)


# ======================================================================
# Cache loading
# ======================================================================

def load_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "patch_features": np.asarray(data["patch_features"], dtype=np.float32),
            "ref_patch_features": np.asarray(data["ref_patch_features"], dtype=np.float32),
            "sample_ids": np.asarray(data["sample_ids"]),
            "gt_sp": np.asarray(data["gt_sp"], dtype=np.int64),
            "imgs_masks": np.asarray(data["imgs_masks"], dtype=np.uint8),
            "grid_size": tuple(int(v) for v in data["grid_size"]),
        }


def resize_patches(patches: np.ndarray, target_grid: tuple[int, int]) -> np.ndarray:
    """[N, H, W, D] -> [N, th, tw, D] via bilinear interpolation (matches A1)."""
    h, w = patches.shape[1], patches.shape[2]
    if (h, w) == target_grid:
        return patches
    x = torch.from_numpy(patches).permute(0, 3, 1, 2)  # [N, D, H, W]
    x = F.interpolate(x, size=target_grid, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).numpy()


# ======================================================================
# A1-frozen fused feature blocks (per-reference-image preserved)
# ======================================================================

def build_fused_blocks(dino: dict, clip: dict, dino_weight: float = 0.5):
    """Return A1 concat blocks with reference-image axis preserved.

    Returns (feat, ref, sample_ids, masks, grid):
      feat  [N, H, W, D]  test fused patches (per-patch L2 normalized rows)
      ref   [K, H, W, D]  reference fused patches, K == number of reference images
      masks [N, mh, mw]   dino gt masks (448 resolution)
      grid  (H, W)
    Each fused row (last axis) is L2-normalized (faiss.normalize_L2 parity), so
    squared-L2 distance / 2.0 == 1 - cosine.
    """
    import faiss

    grid = dino["grid_size"]
    alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])
    clip_feat = clip["patch_features"][alignment.candidate_order]
    clip_ref = clip["ref_patch_features"]
    clip_feat = resize_patches(clip_feat, grid)
    clip_ref = resize_patches(clip_ref, grid)

    dino_feat = dino["patch_features"]
    dino_ref = dino["ref_patch_features"]

    # exact A1 parity: sklearn L2 row normalize per branch (same calls as evaluate_a1_feature_fusion)
    dino_feat = normalize(dino_feat.reshape(-1, dino_feat.shape[-1])).reshape(dino_feat.shape)
    dino_ref = normalize(dino_ref.reshape(-1, dino_ref.shape[-1])).reshape(dino_ref.shape)
    clip_feat = normalize(clip_feat.reshape(-1, clip_feat.shape[-1])).reshape(clip_feat.shape)
    clip_ref = normalize(clip_ref.reshape(-1, clip_ref.shape[-1])).reshape(clip_ref.shape)

    feat = np.concatenate([dino_weight * dino_feat, (1.0 - dino_weight) * clip_feat], axis=-1)
    ref = np.concatenate([dino_weight * dino_ref, (1.0 - dino_weight) * clip_ref], axis=-1)

    n, d = feat.shape[0], feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float32)
    ref_flat = ref.reshape(-1, d).astype(np.float32)
    faiss.normalize_L2(feat_flat)
    faiss.normalize_L2(ref_flat)
    feat = feat_flat.reshape(n, *grid, d)
    ref = ref_flat.reshape(*ref.shape[:1], *grid, d)  # [K, H, W, D]
    return feat, ref, dino["sample_ids"], dino["imgs_masks"], grid


# ======================================================================
# Per-reference distance decomposition
# ======================================================================

def per_reference_distances(
    feat_flat: np.ndarray,      # [Q, D] fused query rows (already L2-normalized)
    ref: np.ndarray,            # [K, H, W, D] fused refs (already L2-normalized)
) -> np.ndarray:                # [Q, K] d_r = min over patches of ref image r
    """d_r(q) = min_j [ ||q - p_{r,j}||^2 / 2 ]  ==  1 - cos to best patch of image r."""
    import faiss

    k = ref.shape[0]
    grid_area = int(np.prod(ref.shape[1:3]))
    d = ref.shape[-1]
    ref_per_img = ref.reshape(k, grid_area, d).astype(np.float32)
    out = np.empty((feat_flat.shape[0], k), dtype=np.float32)
    for r in range(k):
        index = faiss.IndexFlatL2(d)
        index.add(ref_per_img[r])
        distances, _ = index.search(feat_flat, k=1)
        out[:, r] = distances[:, 0]
    return out / 2.0  # [Q, K]


def ref_loo_calibration(ref: np.ndarray) -> np.ndarray:
    """Leave-one-reference-out normal disagreement stats.

    For each reference image r (pseudo-query, all its patches), compute
    d over the remaining K-1 reference images -> per-patch MAD_r.
    Returns the flattened per-patch normal MAD array (usable for a 95th
    percentile). No test images, no test statistics, no GT.
    """
    import faiss

    k = ref.shape[0]
    if k < 2:
        return np.empty(0, dtype=np.float32)
    grid_area = int(np.prod(ref.shape[1:3]))
    d = ref.shape[-1]
    ref_per_img = ref.reshape(k, grid_area, d).astype(np.float32)
    all_mad: list[np.ndarray] = []
    for q in range(k):
        banks = [r for r in range(k) if r != q]
        dists = np.empty((grid_area, len(banks)), dtype=np.float32)
        for j, r in enumerate(banks):
            index = faiss.IndexFlatL2(d)
            index.add(ref_per_img[r])
            distances, _ = index.search(ref_per_img[q], k=1)
            dists[:, j] = distances[:, 0]
        dists /= 2.0
        med = np.median(dists, axis=1, keepdims=True)
        mad = np.median(np.abs(dists - med), axis=1)
        all_mad.append(mad)
    return np.concatenate(all_mad)


# ======================================================================
# Metrics (STRIDE=8 protocol, aupro_fast parity with A1 scripts)
# ======================================================================

def compute_pixel_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    """pixel_maps [N, 448, 448]; gt_masks [N, 448, 448] uint8; STRIDE=8."""
    from evaluate_unified import aupro_fast
    from sklearn.metrics import average_precision_score, roc_auc_score

    maps_strided = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_strided = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_strided.ravel()
    flat_labels = (masks_strided.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_strided, maps_strided)),
    }


def maps_from_patches(grid_maps: np.ndarray, map_size: tuple[int, int] = MAP_SIZE) -> np.ndarray:
    """grid_maps [N, H, W] -> [N, map_h, map_w] via dists2map (A1 parity)."""
    from src.utils import dists2map

    return np.stack([dists2map(g, map_size) for g in grid_maps]).astype(np.float32)
