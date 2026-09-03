"""Route D — NORC: normal-only region conformalization (task book 19 §7).

GT-free gating machinery. Regions = connected components of an anomaly grid above
a reference-only threshold theta; region non-conformity = region max score; an
auxiliary delta map may modify A1 only inside regions with finite-sample
conformal p <= alpha. Everywhere else the output is exactly A1 (identity).
"""

from __future__ import annotations

import numpy as np
from skimage import measure

ALPHA: float = 0.05
Q: float = 0.95


def loo_theta(fused_ref: np.ndarray, q: float = Q) -> float:
    """theta = q-th percentile of LOO per-patch fused d_min on references only.

    fused_ref: [K, H, W, D] fused reference blocks (already L2-normalized).
    """
    import faiss

    k = fused_ref.shape[0]
    if k < 2:
        raise ValueError("NORC calibration requires K >= 2 reference images")
    grid_area = int(np.prod(fused_ref.shape[1:3]))
    d = fused_ref.shape[-1]
    ref_per_img = fused_ref.reshape(k, grid_area, d).astype(np.float32)
    mins: list[np.ndarray] = []
    for qq in range(k):
        banks = [r for r in range(k) if r != qq]
        dists = np.empty((grid_area, len(banks)), dtype=np.float32)
        for j, r in enumerate(banks):
            index = faiss.IndexFlatL2(d)
            index.add(ref_per_img[r])
            distances, _ = index.search(ref_per_img[qq], k=1)
            dists[:, j] = distances[:, 0]
        mins.append(dists.min(axis=1) / 2.0)
    all_min = np.concatenate(mins)
    return float(np.percentile(all_min, 100.0 * q))


def region_max_scores(score_grid: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connected components of {score_grid > theta}; returns region label map,
    per-region max score, and per-region pixel count."""
    mask = (score_grid > theta).astype(np.uint8)
    if mask.max() == 0:
        return np.zeros_like(mask), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    lbl = measure.label(mask, connectivity=2)
    comps = np.unique(lbl[lbl > 0])
    scores = np.array([float(score_grid[lbl == c].max()) for c in comps], dtype=np.float32)
    counts = np.array([int((lbl == c).sum()) for c in comps], dtype=np.int64)
    return lbl, scores, counts


def conformal_p(score: float, calib_scores: np.ndarray) -> float:
    """Finite-sample conformal rank p = (1 + #{calib >= score}) / (n + 1)."""
    n = int(calib_scores.size)
    if n == 0:
        raise ValueError("empty calibration set")
    return float((1.0 + float((calib_scores >= score).sum())) / (n + 1.0))


def significant_region_mask(
    a1_grid: np.ndarray,
    theta: float,
    calib_region_max: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, dict]:
    """Boolean map (same shape as a1_grid) of regions whose conformal p <= alpha.

    calib_region_max: per-UNIT region-max scores aggregated over the calibration
    reference images (units = reference images, not patches).
    """
    lbl, scores, counts = region_max_scores(a1_grid, theta)
    sig = np.zeros_like(a1_grid, dtype=bool)
    n_regions = 0
    n_activated = 0
    for c, s in zip(np.unique(lbl[lbl > 0]), scores):
        n_regions += 1
        p = conformal_p(float(s), calib_region_max)
        if p <= alpha:
            sig |= lbl == c
            n_activated += 1
    return sig, {"n_regions": n_regions, "n_activated": n_activated}


def gate_delta(
    delta_grid: np.ndarray,
    a1_grid: np.ndarray,
    theta: float,
    calib_region_max: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, dict]:
    """Return gated delta map (delta applied only inside p<=alpha regions).

    Identity guarantee: when no region is significant the gated delta is 0, so
    output == A1 exactly.
    """
    sig, stats = significant_region_mask(a1_grid, theta, calib_region_max, alpha)
    gated = np.where(sig, delta_grid, 0.0).astype(np.float32)
    stats["identity_ratio"] = 1.0 - float(sig.mean())
    return gated, stats
