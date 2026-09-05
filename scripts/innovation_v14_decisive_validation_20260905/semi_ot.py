"""Semi-relaxed capacity OT (doc28 s4.4).

min_P  <P, C> + eps * sum_ij P_ij (log P_ij - 1) + tau * KL(P^T 1 || rho)
 s.t.  P 1 = a   (query rows fixed; sum a = 1)

C : [Q, A]  cost (query x anchors). a : [Q] row masses (sum 1).
rho : [A] target column-capacity distribution (sum 1, rho>0).
eps : entropic regularisation. tau : capacity (KL) strength.
Q != A allowed; no forced global bijection.

Fixed-point: optimality gives P_ij ∝ exp(-(C_ij + w_j)/eps) with row masses a_i and
w_j = tau * log(m_j / rho_j), m = P^T 1. Iterate {w <- damped update; row-project}.
"""
from __future__ import annotations

import numpy as np


def _row_project(Cw: np.ndarray, a: np.ndarray, eps: float) -> np.ndarray:
    """P_ij = a_i * softmax_j(-Cw_ij/eps)  (rows exactly a)."""
    logits = -Cw / eps
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    e /= e.sum(axis=1, keepdims=True)
    return e * a[:, None]


def solve_semi_ot(C: np.ndarray, a: np.ndarray, rho: np.ndarray,
                  eps: float, tau: float, iters: int = 20000,
                  tol: float = 1e-10, damp: float = 0.5,
                  lb0: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Return (P, stats). stats: row_err, col_mass, converged, iters, lb.

    Stable generalised-Sinkhorn form: P_ij = a_i K_ij b_j / Z_i with
    K = exp(-C/eps), Z_i = sum_j K_ij b_j, and the column-KL fixed point
    b_j = (c_j / rho_j)^(-tau/(eps+tau)),  c_j = sum_i a_i K_ij / Z_i.
    The exponent tau/(eps+tau) in (0,1) keeps the iteration stable even when a
    column is (almost) empty (no log(0) trap; verified analytically for the
    1-row case: P_j propto rho_j^(tau/(eps+tau)) exp(-C_j/(eps+tau))).

    lb0: optional warm-start initial log-b (same column count). Purely an
    initialisation detail: the converged fixed point is unchanged. Used by the
    P2-A probe loop where consecutive cost matrices share anchors/rho, so the
    previous solution's lb is a good starting point (deterministic order).
    """
    C = np.asarray(C, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    a = a / a.sum()
    rho = np.asarray(rho, dtype=np.float64)
    rho = rho / rho.sum()
    Q, A = C.shape
    if rho.shape[0] != A:
        raise ValueError("rho length must equal C columns")
    logK = -C / eps
    lb = (np.zeros(A) if lb0 is None else np.asarray(lb0, dtype=np.float64).copy())
    lc = np.zeros(A)
    alpha = tau / (eps + tau)
    converged = False
    for it in range(iters):
        lb_old = lb.copy()
        M = logK + lb[None, :]                       # log(K_ij b_j)
        mx = M.max(axis=1)
        Z = mx + np.log(np.exp(M - mx[:, None]).sum(axis=1))   # log row-sum (normaliser)
        # log(b_j c_j) = log( sum_i a_i K_ij b_j / Z_i ) ; c_j = col mass per unit b
        log_bc = np.log(np.maximum(a @ np.exp(M - Z[:, None]), 1e-300))
        lc = log_bc - lb - np.log(np.maximum(rho, 1e-300))     # log(c_j / rho_j)
        lb = lb_old + damp * (-alpha * lc - lb_old)            # b <- (c/rho)^(-alpha)
        if it >= 10 and it % 100 == 0:
            if np.abs(lb - lb_old).max() < tol:
                converged = True
                break
    P = np.exp(logK + lb[None, :] - (logK + lb[None, :]).max(axis=1, keepdims=True))
    Zf = P.sum(axis=1, keepdims=True)
    P = P / Zf * a[:, None]
    row_err = float(np.abs(P.sum(axis=1) - a).max())
    stats = {"row_err": row_err, "col_mass": P.sum(axis=0).tolist(),
             "converged": bool(converged), "iters": int(it + 1),
             "Q": int(Q), "A": int(A), "eps": float(eps), "tau": float(tau),
             "lb": lb}
    return P, stats


def row_costs(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-row expected matching cost = sum_j (P_ij/rowmass) C_ij."""
    rs = P.sum(axis=1, keepdims=True)
    return (P / np.maximum(rs, 1e-12) * C).sum(axis=1)


def capacity_premium(C: np.ndarray, a: np.ndarray, rho: np.ndarray,
                     eps: float, tau: float, lb0: np.ndarray | None = None,
                     **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(premium, P, free_cost, stats): premium = semi-OT row cost - free soft-NN cost.

    Free baseline: per-row entropic soft matching with NO capacity (tau->0 in the
    same solver family) = a_i softmax(-C/eps). Capacity premium = extra expected
    cost each query row pays because of soft column-capacity coupling.
    lb0: warm-start log-b for the capacity solve only (init detail, same fixed
    point); the free solve always starts from zeros.
    """
    P, stats = solve_semi_ot(C, a, rho, eps, tau, lb0=lb0, **kw)
    P0, _ = solve_semi_ot(C, a, np.ones_like(rho) / rho.size, eps, tau=0.0, **kw)
    free_cost = row_costs(P0, C)
    cap_cost = row_costs(P, C)
    return cap_cost - free_cost, P, free_cost, stats


def spillover(inside_mask: np.ndarray, premium: np.ndarray,
              near_ring: int = 2, grid=(16, 16)):
    """Mask -> inside mean premium, 1-2 token ring mean, far-background mean.

    grid: per-image spatial grid of the Q rows (default 16x16 probe grid).
    Both premium (1-D ravel of Q rows) and inside_mask (1-D ravel or 2-D grid)
    are reshaped to `grid` before the inside/ring/far split.
    """
    prem = np.asarray(premium)
    if prem.ndim != 2:
        prem = prem.reshape(grid)
    m2 = np.asarray(inside_mask, dtype=bool)
    if m2.ndim != 2:
        m2 = m2.reshape(grid)
    inside = prem[m2]
    if inside.size == 0:
        return {"inside": 0.0, "ring": 0.0, "far": float(prem.mean())}
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(m2, iterations=near_ring) & ~m2
    far = ~binary_dilation(m2, iterations=near_ring)
    return {"inside": float(prem[m2].mean()),
            "ring": float(prem[ring].mean()) if ring.any() else 0.0,
            "far": float(prem[far].mean()) if far.any() else 0.0}
