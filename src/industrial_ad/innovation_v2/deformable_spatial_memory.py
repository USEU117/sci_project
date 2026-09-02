"""Route B — DSAM: Deformable Spatially-Aware Dual-Encoder Memory.

Task book section 6. For every query image x reference image pair:
  1. bidirectional nearest neighbours on DINO patch features (mutual matches only);
  2. robust estimator (median translation, or OpenCV RANSAC affine) with fixed seed;
  3. inlier ratio; fallback to global A1 when mutual matches < 16 or ratio < 0.20;
  4. score = min over refs of min fused A1 distance within the L-infinity window
     of radius R around the transformed query position.

Spatial coordinates are DINO patch (row, col). The fused score uses the frozen
A1 descriptor distance 0.5*||z_q - z_i||^2 = 1 - z_q . z_i (unit vectors).
"""

from __future__ import annotations

import cv2
import numpy as np

from industrial_ad.fusion import rcec
from industrial_ad.innovation_v2.common import AlignedFeatures, InnovationError


def _chunked_cosine_sim(a: np.ndarray, b: np.ndarray, chunk: int = 128) -> np.ndarray:
    """cosine-sim matrix [a_n, b_m] for L2 rows, chunked to bound memory."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    for i0 in range(0, a.shape[0], chunk):
        i1 = min(i0 + chunk, a.shape[0])
        out[i0:i1] = a[i0:i1] @ b.T
    return out


def _mutual_matches(
    q_flat: np.ndarray, r_flat: np.ndarray, chunk: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bidirectional nearest neighbours; returns nn_q, nn_r, mutual-mask (n_q,)."""
    sim = _chunked_cosine_sim(q_flat, r_flat, chunk)
    nn_q = np.argmax(sim, axis=1)              # [n_q]
    nn_r = np.argmax(sim, axis=0)              # [n_r]
    mutual = nn_r[nn_q] == np.arange(q_flat.shape[0])
    return nn_q, nn_r, mutual


def _estimate_transform(
    q_pts: np.ndarray, r_pts: np.ndarray,
    alignment: str, seed: int, ransac_threshold: float,
) -> tuple[callable | None, float, int]:
    """Estimate T: query-patch coords -> ref-patch coords from matched points.

    Returns (transform(rows, cols) -> (rows', cols') vectorised callable,
    inlier_ratio, n_matches). transform is None when estimation fails.
    """
    n = q_pts.shape[0]
    if n == 0:
        return None, 0.0, 0

    if alignment == "translation":
        disp = r_pts - q_pts
        med = np.median(disp, axis=0)
        inliers = np.max(np.abs(disp - med), axis=1) <= float(ransac_threshold)
        ratio = float(inliers.mean())
        tx, ty = float(med[0]), float(med[1])

        def T(rows, cols):
            return rows + tx, cols + ty

        return T, ratio, n

    if alignment == "identity":
        def T(rows, cols):
            return rows, cols
        return T, 1.0, n

    if alignment == "affine":
        cv2.setRNGSeed(int(seed))
        M, inl = cv2.estimateAffinePartial2D(
            q_pts, r_pts, method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold),
            maxIters=2000, confidence=0.99)
        if M is None or inl is None:
            return None, 0.0, n
        ratio = float(np.mean(inl.ravel() > 0)) if inl.size else 0.0
        A = M[:, :2].astype(np.float64)
        b = M[:, 2].astype(np.float64)

        def T(rows, cols):
            pts = np.stack([rows, cols], axis=1).astype(np.float64)
            out = pts @ A.T + b
            return out[:, 0], out[:, 1]

        return T, ratio, n

    raise InnovationError(f"unknown DSAM alignment: {alignment}")


def _constrained_min_dist(
    zq: np.ndarray, zr: np.ndarray,
    r_rows: np.ndarray, r_cols: np.ndarray,
    centers: np.ndarray | None, R: float,
) -> np.ndarray:
    """Per-query-patch min fused distance within the L-inf window (or global)."""
    sim = _chunked_cosine_sim(zq, zr, 128)          # [n_q, n_r]
    dist = 1.0 - sim                                # == 0.5*||z_q-z_i||^2
    if centers is None:
        return dist.min(axis=1)
    dr = np.abs(r_rows[:, None] - centers[:, 0][None, :])   # [n_r, n_q]
    dc = np.abs(r_cols[:, None] - centers[:, 1][None, :])
    mask = (dr <= R) & (dc <= R)                            # [n_r, n_q]
    m = mask.T                                              # [n_q, n_r]
    out = np.where(m, dist, np.inf).min(axis=1)
    # query patches whose window contains no ref patch (e.g. centre mapped
    # outside the image) degrade to the global nearest neighbour instead of inf
    empty = ~m.any(axis=1)
    if np.any(empty):
        out = np.where(empty, dist.min(axis=1), out)
    return out


def score_dsam(
    aligned: AlignedFeatures,
    candidate: dict,
    cfg: dict,
    descriptor: str = "fused",
) -> tuple[np.ndarray, dict]:
    """DSAM score grid [N, H, W] for one candidate on one category."""
    alignment = candidate["alignment"]
    R = float(candidate["r"])
    dsam_cfg = cfg.get("dsam", {})
    min_mutual = int(dsam_cfg.get("min_mutual", 16))
    min_inlier = float(dsam_cfg.get("min_inlier_ratio", 0.20))
    ransac_thr = float(dsam_cfg.get("ransac_threshold", 3.0))
    seed = int(dsam_cfg.get("seed", 0))
    dino_weight = float(cfg.get("fixed", {}).get("dino_weight", 0.5))

    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    n_refs = aligned.n_references
    d_feat = aligned.d_feat.reshape(n, h * w, -1)
    d_ref = aligned.d_ref.reshape(n_refs, h * w, -1)

    if descriptor == "fused":
        qz = rcec._concat_and_l2(
            aligned.d_feat.reshape(-1, 768), aligned.c_feat.reshape(-1, 768), dino_weight)
        rz = rcec._concat_and_l2(
            aligned.d_ref.reshape(-1, 768), aligned.c_ref.reshape(-1, 768), dino_weight)
    elif descriptor == "dino":
        qz = np.ascontiguousarray(aligned.d_feat.reshape(-1, 768), dtype=np.float32)
        rz = np.ascontiguousarray(aligned.d_ref.reshape(-1, 768), dtype=np.float32)
    else:
        raise InnovationError(f"unknown DSAM descriptor: {descriptor}")
    qz = qz.reshape(n, h * w, -1)
    rz = rz.reshape(n_refs, h * w, -1)

    r_rows = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)).reshape(-1)
    r_cols = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1)).reshape(-1)

    scores = np.empty((n, h * w), dtype=np.float32)
    diag_mutual = []
    diag_ratio = []
    diag_fallback = 0
    diag_pool = []

    for t in range(n):
        best = np.full(h * w, np.inf, dtype=np.float32)
        q_dino = d_feat[t]
        for r in range(n_refs):
            nn_q, _, mutual = _mutual_matches(q_dino, d_ref[r])
            q_idx = np.nonzero(mutual)[0]
            if q_idx.size == 0:
                fallback = True
                ratio = 0.0
            else:
                r_idx = nn_q[q_idx]
                q_pts = np.stack([r_rows[q_idx], r_cols[q_idx]], axis=1).astype(np.float32)
                r_pts = np.stack([r_rows[r_idx], r_cols[r_idx]], axis=1).astype(np.float32)
                T, ratio, _ = _estimate_transform(q_pts, r_pts, alignment, seed, ransac_thr)
                if T is None:
                    fallback = True
                else:
                    fallback = (q_idx.size < min_mutual) or (ratio < min_inlier)
            diag_mutual.append(int(q_idx.size))
            diag_ratio.append(float(ratio))
            if fallback:
                diag_fallback += 1
                diag_pool.append(h * w)
                constrained = _constrained_min_dist(qz[t], rz[r], r_rows, r_cols, None, R)
            else:
                cy, cx = T(r_rows, r_cols)
                centers = np.stack([cy, cx], axis=1).astype(np.float32)
                pool = (
                    (np.abs(r_rows[:, None] - centers[:, 0][None, :]) <= R) &
                    (np.abs(r_cols[:, None] - centers[:, 1][None, :]) <= R)
                ).sum(axis=0)
                diag_pool.append(int(np.mean(pool)))
                constrained = _constrained_min_dist(
                    qz[t], rz[r], r_rows, r_cols, centers, R)
            best = np.minimum(best, constrained)
        scores[t] = best

    if not np.all(np.isfinite(scores)):
        raise InnovationError("DSAM produced non-finite scores")

    diag = {
        "alignment": alignment,
        "r": R,
        "mean_mutual_matches": round(float(np.mean(diag_mutual)), 2),
        "mean_inlier_ratio": round(float(np.mean(diag_ratio)), 4),
        "fallback_ratio": round(diag_fallback / (n * n_refs), 4),
        "mean_pool_size": round(float(np.mean(diag_pool)), 2),
        "total_pool_patches": int(np.sum(diag_pool)),
    }
    return scores.reshape(n, h, w), diag


def score_dsam_fixed_loc(aligned: AlignedFeatures, candidate: dict, cfg: dict):
    """Small-gate control: same-radius local window but NO alignment (identity)."""
    ctrl = dict(candidate)
    ctrl["alignment"] = "identity"
    return score_dsam(aligned, ctrl, cfg)
