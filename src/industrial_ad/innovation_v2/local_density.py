"""Route A — LNDC: Local Normal-Density Calibrated Dual-Encoder Memory.

Task book section 5. Reuses the frozen A1 fused descriptor (0.5/0.5 concat + L2).
The reference density rho is estimated with strict LOO:
  * shot >= 2: exclude patches of the same reference image;
  * shot == 1: exclude the patch itself and its Chebyshev radius-1 neighbours.
The test score is the median ratio of the A1 distance to the local rho. No test
statistics, labels or masks are ever used.
"""

from __future__ import annotations

import numpy as np

from industrial_ad.fusion import rcec
from industrial_ad.innovation_v2.common import AlignedFeatures, InnovationError


def fused_flat(aligned: AlignedFeatures, dino_weight: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """A1-style fused descriptors (flat) for query and reference."""
    d = aligned.d_feat.reshape(-1, aligned.d_feat.shape[-1])
    c = aligned.c_feat.reshape(-1, aligned.c_feat.shape[-1])
    dr = aligned.d_ref.reshape(-1, aligned.d_ref.shape[-1])
    cr = aligned.c_ref.reshape(-1, aligned.c_ref.shape[-1])
    z_q = rcec._concat_and_l2(d, c, dino_weight)
    z_r = rcec._concat_and_l2(dr, cr, dino_weight)
    return z_q, z_r


def dino_flat(aligned: AlignedFeatures) -> tuple[np.ndarray, np.ndarray]:
    d = np.ascontiguousarray(aligned.d_feat.reshape(-1, aligned.d_feat.shape[-1]), dtype=np.float32)
    dr = np.ascontiguousarray(aligned.d_ref.reshape(-1, aligned.d_ref.shape[-1]), dtype=np.float32)
    return d, dr


def _loo_exclusion_mask(
    n_patches: int, grid: tuple[int, int], n_images: int, shot: int
) -> np.ndarray:
    """Mask[i, j] = True if ref patch j must be excluded when i is the query.

    shot>=2: exclude all patches of the same reference image.
    shot==1: exclude self + Chebyshev radius-1 neighbours (single reference image).
    """
    h, w = grid
    if shot >= 2:
        per_image = h * w
        img_idx = np.arange(n_patches) // per_image
        return (img_idx[:, None] == img_idx[None, :]).astype(bool)
    rows = np.arange(h * w) // w
    cols = np.arange(h * w) % w
    dr = np.abs(rows[:, None] - rows[None, :])
    dc = np.abs(cols[:, None] - cols[None, :])
    return (dr <= 1) & (dc <= 1)


def ref_density_rho(
    z_ref: np.ndarray,
    grid: tuple[int, int],
    n_images: int,
    shot: int,
    k: int,
    epsilon: float,
    chunk: int = 16384,
    search_k_extra: int = 12,
) -> np.ndarray:
    """rho_i = median over the k allowed LOO neighbours of 0.5*||z_i - z_j||^2.

    Searches k + search_k_extra neighbours so that after LOO exclusion at least
    k allowed neighbours remain (worst case shot=1 excludes 9 patches around i).
    """
    n = z_ref.shape[0]
    mask = _loo_exclusion_mask(n, grid, n_images, shot)
    h, w = grid
    if shot >= 2:
        # Guarantee k allowed neighbours: at most h*w same-image patches can
        # occupy the top-k, so search k + h*w.
        k_search = int(k) + h * w
    else:
        k_search = int(k) + int(search_k_extra)  # shot=1 excludes <= 9 patches
    if k_search >= n:
        k_search = n
    index = rcec._faiss_index(np.ascontiguousarray(z_ref, dtype=np.float32))
    dists, indices = rcec._search_chunked(
        index, np.ascontiguousarray(z_ref, dtype=np.float32), k=k_search, chunk=chunk)
    d_half = (dists / 2.0).astype(np.float32)

    # Vectorised: FAISS returns neighbours in increasing distance, so the first
    # k *allowed* entries are the k smallest LOO distances; rho = their median.
    allowed = ~mask[np.arange(n)[:, None], indices]          # (n, k_search)
    counts = np.cumsum(allowed, axis=1, dtype=np.int32)
    is_first_k = allowed & (counts <= int(k))
    if is_first_k.sum(axis=1).min() < k:
        raise InnovationError("LNDC LOO: some reference patch has < k allowed neighbours")
    vals = np.where(is_first_k, d_half, np.nan)
    with np.errstate(invalid="ignore"):
        rho = np.nanmedian(vals, axis=1).astype(np.float32)
    return rho


def lndc_scores(
    z_query: np.ndarray,
    z_ref: np.ndarray,
    rho: np.ndarray | float,
    k: int,
    epsilon: float,
    chunk: int = 16384,
) -> np.ndarray:
    """s(q) = median over the k nearest ref patches of d(q, i) / (rho_i + epsilon)."""
    rho_arr = np.asarray(rho, dtype=np.float32)
    index = rcec._faiss_index(np.ascontiguousarray(z_ref, dtype=np.float32))
    dists, indices = rcec._search_chunked(
        index, np.ascontiguousarray(z_query, dtype=np.float32), k=int(k), chunk=chunk)
    d_half = (dists / 2.0).astype(np.float32)
    k_ = int(k)
    n = z_query.shape[0]
    ratios = np.empty((n, k_), dtype=np.float32)
    for j in range(k_):
        rho_j = rho_arr[indices[:, j]] if rho_arr.ndim == 1 else rho_arr
        ratios[:, j] = d_half[:, j] / (rho_j + float(epsilon))
    return np.median(ratios, axis=1).astype(np.float32)


def score_lndc(
    aligned: AlignedFeatures,
    candidate: dict,
    cfg: dict,
    descriptor: str = "fused",
) -> tuple[np.ndarray, dict]:
    """Compute the LNDC score grid for one (candidate, descriptor) on a category.

    descriptor: "fused" (A1 1536-D) or "dino" (768-D), for mechanism ablations.
    """
    k = int(candidate["k"])
    epsilon = float(cfg.get("lndc", {}).get("epsilon", 1e-6))
    shot = int(cfg.get("_shot"))
    if shot is None:
        raise InnovationError("LNDC requires cfg['_shot'] to apply LOO rules")

    if descriptor == "fused":
        z_q, z_r = fused_flat(aligned)
    elif descriptor == "dino":
        z_q, z_r = dino_flat(aligned)
    else:
        raise InnovationError(f"unknown LNDC descriptor: {descriptor}")

    rho = ref_density_rho(z_r, aligned.grid, shot, shot, k, epsilon)
    s_flat = lndc_scores(z_q, z_r, rho, k, epsilon)
    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    diag = {
        "descriptor": descriptor,
        "k": k,
        "epsilon": epsilon,
        "rho_mean": round(float(np.mean(rho)), 6),
        "rho_std": round(float(np.std(rho)), 6),
        "rho_min": round(float(np.min(rho)), 6),
        "rho_max": round(float(np.max(rho)), 6),
        "n_ref_patches": int(z_r.shape[0]),
    }
    return s_flat.reshape(n, h, w), diag


def score_lndc_global_sham(
    aligned: AlignedFeatures,
    candidate: dict,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Global-density sham control: one scalar rho for the whole category.

    With a constant rho the ratio is a monotone transform of the raw distance,
    so it isolates the *local* density information of LNDC.
    """
    k = int(candidate["k"])
    epsilon = float(cfg.get("lndc", {}).get("epsilon", 1e-6))
    shot = int(cfg.get("_shot"))
    z_q, z_r = fused_flat(aligned)

    rho = ref_density_rho(z_r, aligned.grid, shot, shot, k, epsilon)
    rho_global = float(np.median(rho))
    s_flat = lndc_scores(z_q, z_r, rho_global, k, epsilon)
    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    diag = {
        "descriptor": "fused",
        "k": k,
        "epsilon": epsilon,
        "global_rho": rho_global,
        "sham": True,
    }
    return s_flat.reshape(n, h, w), diag
