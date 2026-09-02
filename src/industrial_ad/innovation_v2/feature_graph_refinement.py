"""Route F — FAGR: Feature-Affinity Graph Refinement.

Task book section 10. Starting from the raw A1 patch score grid (pre-Gaussian),
run a fixed number of Jacobi iterations on the 4-neighbour graph whose edge
weights come from DINO patch feature affinity:

    w_pq = exp((cos(d_p, d_q) - 1) / tau)
    s_p^{t+1} = (s_p^0 + mu * sum_q w_pq s_q^t) / (1 + mu * sum_q w_pq)

Final upsampling is bilinear only; the frozen Gaussian (sigma=4) is NOT applied
again. ``uniform=True`` (all weights = 1) is the uniform-smoothing control.
"""

from __future__ import annotations

import cv2
import numpy as np

from industrial_ad.innovation_v2.common import AlignedFeatures, InnovationError


def fagr_iterate(
    s0: np.ndarray, d_feat: np.ndarray, mu: float, iters: int, tau: float,
    uniform: bool = False,
) -> np.ndarray:
    """Jacobi refinement on the 4-neighbour graph.

    s0: [N, H, W] raw A1 patch scores (pre-Gaussian).
    d_feat: [N, H, W, D] per-patch DINO features (L2 rows).
    """
    n, h, width = s0.shape
    s = s0.copy()
    d = d_feat.astype(np.float32)

    # Per-image 4-neighbour weights [N, H, W, 4] (0,1,2,3 = up,down,left,right).
    if uniform:
        wgt = np.ones((n, h, width, 4), dtype=np.float32)
    else:
        cos_u = np.einsum("nhwd,nhwd->nhw", d, np.roll(d, 1, axis=1))          # with up
        cos_d = np.einsum("nhwd,nhwd->nhw", d, np.roll(d, -1, axis=1))         # with down
        cos_l = np.einsum("nhwd,nhwd->nhw", d, np.roll(d, 1, axis=2))          # with left
        cos_r = np.einsum("nhwd,nhwd->nhw", d, np.roll(d, -1, axis=2))         # with right
        wgt = np.stack([cos_u, cos_d, cos_l, cos_r], axis=-1)
        wgt = np.exp((wgt - 1.0) / float(tau)).astype(np.float32)

    # Boundary masks: rows 0 / h-1, cols 0 / width-1 edges are invalid (weight 0).
    boundary = np.ones((n, h, width, 4), dtype=np.float32)
    boundary[:, 0, :, 0] = 0.0   # up edge at row 0
    boundary[:, h - 1, :, 1] = 0.0
    boundary[:, :, 0, 2] = 0.0   # left edge at col 0
    boundary[:, :, width - 1, 3] = 0.0
    wgt = wgt * boundary

    def neighbor_sum(x: np.ndarray) -> np.ndarray:
        up = np.roll(x, 1, axis=1) * wgt[:, :, :, 0]
        down = np.roll(x, -1, axis=1) * wgt[:, :, :, 1]
        left = np.roll(x, 1, axis=2) * wgt[:, :, :, 2]
        right = np.roll(x, -1, axis=2) * wgt[:, :, :, 3]
        return up + down + left + right

    def neighbor_wsum() -> np.ndarray:
        return wgt.sum(axis=-1)

    denom = 1.0 + float(mu) * neighbor_wsum()
    for _ in range(int(iters)):
        num = s0 + float(mu) * neighbor_sum(s)
        s = num / denom
    return s


def bilinear_map(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling only (no Gaussian) for the FAGR final map."""
    return np.stack([
        cv2.resize(grid[i], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        for i in range(grid.shape[0])
    ]).astype(np.float32)


def score_fagr(
    aligned: AlignedFeatures,
    candidate: dict,
    cfg: dict,
    uniform: bool = False,
) -> tuple[np.ndarray, dict]:
    from industrial_ad.innovation_v2.common import a1_grid

    mu = float(candidate["mu"])
    iters = int(candidate["iters"])
    tau = float(cfg.get("fagr", {}).get("tau", 0.10))
    s0 = a1_grid(aligned)  # raw A1 patch scores [N, H, W] (pre-Gaussian)
    d = aligned.d_feat  # [N, H, W, 768] per-branch L2 DINO
    s = fagr_iterate(s0, d, mu, iters, tau, uniform=uniform)
    diag = {"mu": mu, "iters": iters, "tau": tau,
            "uniform_control": bool(uniform),
            "s0_min": round(float(s0.min()), 6), "s0_max": round(float(s0.max()), 6),
            "s_min": round(float(s.min()), 6), "s_max": round(float(s.max()), 6)}
    return s, diag


def score_fagr_uniform(aligned: AlignedFeatures, candidate: dict, cfg: dict):
    return score_fagr(aligned, candidate, cfg, uniform=True)
