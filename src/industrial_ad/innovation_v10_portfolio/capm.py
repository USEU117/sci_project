"""Route C — CAPM primitives (task book 19 §6).

Alignment is DINO-only (frozen DINO vitb14 patch features, cosine mutual NN +
RANSAC affine); scoring happens in the A1 FUSED space (same rows as frozen A1),
with d_pos restricting each query patch's memory search to reference patches
whose (inverse-transformed) coordinates lie within radius 2 patch units.
"""

from __future__ import annotations

import cv2
import numpy as np

RADIUS: int = 2          # patch units, pre-registered
RELIABLE_INLIER: float = 0.30  # fixed reliability gate (pre-registered)
MAX_ITERS: int = 2000
CONFIDENCE: float = 0.99


def normalize_rows(x: np.ndarray) -> np.ndarray:
    import faiss

    out = x.astype(np.float32).copy()
    faiss.normalize_L2(out)
    return out


def mutual_matches(qf: np.ndarray, rf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    d = qf.shape[-1]
    index_q = faiss.IndexFlatIP(d)
    index_r = faiss.IndexFlatIP(d)
    index_q.add(qf)
    index_r.add(rf)
    _, q_to_r = index_r.search(qf, 1)
    _, r_to_q = index_q.search(rf, 1)
    qi = np.arange(qf.shape[0])
    ri = q_to_r[:, 0]
    mutual = r_to_q[ri, 0] == qi
    return qi[mutual], ri[mutual]


def estimate_affine(qf: np.ndarray, rf: np.ndarray, grid: tuple[int, int]):
    """RANSAC affine ref->query from mutual NN; returns dict with M (2x3) or None."""
    qi, ri = mutual_matches(qf, rf)
    n_mut = int(qi.size)
    if n_mut < 8:
        return {"ok": False, "n_mutual": n_mut, "inlier_ratio": 0.0, "M": None}
    qy, qx = np.unravel_index(qi, grid)
    ry, rx = np.unravel_index(ri, grid)
    src = np.stack([rx.astype(np.float32), ry.astype(np.float32)], axis=1)
    dst = np.stack([qx.astype(np.float32), qy.astype(np.float32)], axis=1)
    M, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=MAX_ITERS, confidence=CONFIDENCE)
    if M is None or inliers is None:
        return {"ok": False, "n_mutual": n_mut, "inlier_ratio": 0.0, "M": None}
    n_in = int(inliers.sum())
    return {"ok": True, "n_mutual": n_mut, "inlier_ratio": float(n_in / n_mut),
            "n_inliers": n_in, "M": M}


def d_pos_grid(
    fused_query: np.ndarray,    # [H, W, D] fused query patch rows (L2-normalized)
    fused_ref: np.ndarray,      # [H, W, D] fused ref patch rows (L2-normalized)
    M: np.ndarray,              # 2x3 affine mapping ref coords -> query coords
    radius: int = RADIUS,
) -> np.ndarray:                # [H, W] d_pos per query patch (1 - cos to best in-radius ref)
    """d_pos(q) = min over ref patches within radius of inverse-mapped q coordinate."""
    h, w, d = fused_query.shape
    inv = cv2.invertAffineTransform(M)
    qy, qx = np.mgrid[0:h, 0:w]
    qcoord = np.stack([qx.ravel().astype(np.float32), qy.ravel().astype(np.float32)], axis=1)
    tf = cv2.transform(qcoord.reshape(-1, 1, 2), inv).reshape(-1, 2)  # query -> ref coords
    ry, rx = np.mgrid[0:h, 0:w]
    rcoord = np.stack([rx.ravel(), ry.ravel()], axis=1).astype(np.float32)  # [H*W, 2]
    fq = fused_query.reshape(-1, d)
    fr = fused_ref.reshape(-1, d)
    out = np.empty(h * w, dtype=np.float32)
    # per query: distance over in-radius refs (vectorised select then dot)
    for qidx in range(h * w):
        delta = rcoord - tf[qidx]
        in_rad = np.flatnonzero((delta ** 2).sum(axis=1) <= radius ** 2)
        if in_rad.size == 0:
            out[qidx] = np.inf
            continue
        dots = fq[qidx] @ fr[in_rad].T
        out[qidx] = 1.0 - dots.max()
    return out.reshape(h, w)


def candidate_grid(
    fused_query: np.ndarray,
    fused_ref: np.ndarray,
    M: np.ndarray,
    inlier_ratio: float,
    d_global: np.ndarray,  # [H, W] A1 d_min grid
    radius: int = RADIUS,
) -> np.ndarray:
    """CAPM score grid == A1 exactly when alignment is unreliable (identity)."""
    if not np.isfinite(inlier_ratio) or inlier_ratio < RELIABLE_INLIER or M is None:
        return d_global
    d_pos = d_pos_grid(fused_query, fused_ref, M, radius)
    # no local evidence (empty in-radius neighbourhood) -> identical to A1 there
    d_pos = np.where(np.isfinite(d_pos), d_pos, d_global)
    bonus = np.maximum(0.0, d_pos - d_global)
    return (d_global + 0.25 * bonus).astype(np.float32)
