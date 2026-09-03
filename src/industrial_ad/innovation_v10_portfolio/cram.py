"""Route A — CRAM: Cross-Reference Agreement Memory (task book 19 §4).

Candidates over per-reference distances d_r(q) (already in fused A1 space):

  A0 identity: score = d_min                       (== pooled A1 KNN distance)
  A1 gap:      score = d_min + 0.5*max(0, d_med - d_min)
  A2 MAD:      score = d_min * (1 + 0.5*clip(mad / mad95_normal, 0, 1))

All scoring is a pure function of (query fused patches, per-reference fused
reference blocks). It never sees gt_masks / gt_labels / test statistics.
"""

from __future__ import annotations

import numpy as np

from industrial_ad.innovation_v10_portfolio import common


def agreement_stats(dr: np.ndarray) -> dict:
    """dr [Q, K] per-reference min distances -> per-patch agreement statistics."""
    d_min = dr.min(axis=-1)
    d_med = np.median(dr, axis=-1)
    med = np.median(dr, axis=-1, keepdims=True)
    mad = np.median(np.abs(dr - med), axis=-1)
    return {
        "d_min": d_min.astype(np.float32),
        "d_med": d_med.astype(np.float32),
        "gap": (d_med - d_min).astype(np.float32),
        "mad": mad.astype(np.float32),
    }


def score_a0(stats: dict) -> np.ndarray:
    return stats["d_min"]


def score_a1(stats: dict) -> np.ndarray:
    return (stats["d_min"] + 0.5 * np.maximum(0.0, stats["gap"])).astype(np.float32)


def score_a2(stats: dict, mad95_normal: float) -> np.ndarray:
    ratio = np.clip(stats["mad"] / max(mad95_normal, 1e-12), 0.0, 1.0)
    return (stats["d_min"] * (1.0 + 0.5 * ratio)).astype(np.float32)


def mad95_from_normal_mad(normal_mad: np.ndarray, q: float = 0.95) -> float:
    """Scalar 95th percentile of the reference LOO per-patch MAD distribution."""
    if normal_mad.size == 0:
        return float("nan")
    return float(np.percentile(normal_mad, 100.0 * q))


def candidate_maps(
    feat: np.ndarray,
    ref: np.ndarray,
    candidates: tuple[str, ...] = ("a0", "a1", "a2"),
    mad95_normal: float = float("nan"),
    map_size: tuple[int, int] = common.MAP_SIZE,
) -> dict:
    """Compute candidate score maps [N, map_h, map_w] for one category.

    feat [N, H, W, D], ref [K, H, W, D] come from common.build_fused_blocks.
    K == 1: only a0 is meaningful (a1/a2 degenerate to a0) and is returned as a0.
    """
    n, grid = feat.shape[0], feat.shape[1:3]
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float32)
    dr = common.per_reference_distances(feat_flat, ref)   # [N*H*W, K]
    stats = agreement_stats(dr)
    grids: dict[str, np.ndarray] = {}
    for name in candidates:
        if name == "a0":
            grids["a0"] = score_a0(stats).reshape(n, *grid)
        elif name == "a1":
            if ref.shape[0] < 2:
                grids["a1"] = score_a0(stats).reshape(n, *grid)  # degenerate identity
            else:
                grids["a1"] = score_a1(stats).reshape(n, *grid)
        elif name == "a2":
            if ref.shape[0] < 4 or not np.isfinite(mad95_normal) or mad95_normal <= 0:
                grids["a2"] = score_a0(stats).reshape(n, *grid)  # degenerate identity
            else:
                grids["a2"] = score_a2(stats, mad95_normal).reshape(n, *grid)
        else:
            raise ValueError(name)
    maps = {name: common.maps_from_patches(grids[name], map_size) for name in grids}
    return maps


def pooled_min_map(feat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Reference pooled-NN (single index over ALL ref patches) — A1 parity check.

    Returns [N, H, W] grid of distances/2 for comparison with a0.
    """
    import faiss

    n, grid = feat.shape[0], feat.shape[1:3]
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float32)
    ref_flat = ref.reshape(-1, d).astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(ref_flat)
    distances, _ = index.search(feat_flat, k=1)
    return (distances[:, 0] / 2.0).reshape(n, *grid).astype(np.float32)
