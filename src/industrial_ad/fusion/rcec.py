"""RCEC v1 — Reference-Conditioned Cross-Encoder Consistency.

Core pure functions for the RCEC innovation candidate:

  * paired normal reference memory built from (DINO patch, CLIP-image patch)
    tuples that originate from the *same* reference image and aligned patch
    position;
  * DINO->CLIP conditional consistency score: first retrieve the Top-k DINO
    normal neighbours, then measure the CLIP distance only within those
    neighbours;
  * optional symmetric term DINO|CLIP;
  * reference-only leave-one-out (LOO) calibration statistics (median / MAD)
    with strict exclusion rules (leave-one-reference-image-out for shot>=2,
    self + Chebyshev radius-1 exclusion for shot=1);
  * robust z-score combination: S = (1-lam)*rz(s_A1) + lam*rz(r_cond);
  * per-image aggregation without touching the pixel map (max / top-q mean).

Leakage discipline: none of the algorithm functions accept ``gt_sp``,
``imgs_masks``, ``labels`` or any test-derived statistic. Labels are only ever
consumed by the evaluator layer outside this module.

Distance semantics: all distances are half squared-L2 on L2-normalised patch
vectors, i.e. ``0.5*||x-y||^2``, which matches the frozen A1 convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RCECError(RuntimeError):
    """Base class for RCEC fatal conditions (must fail loudly)."""


class AlignmentError(RCECError):
    """Sample order / pairing cannot be verified."""


class CalibrationDegenerateError(RCECError):
    """Reference statistics are numerically degenerate (e.g. MAD < eps)."""


class CandidateShortageError(RCECError):
    """Fewer than k candidates remain after LOO exclusion."""


# ---------------------------------------------------------------------------
# Normalization helpers (bit-identical to frozen A1 conventions)
# ---------------------------------------------------------------------------

def _l2_normalize_flat(x: np.ndarray) -> np.ndarray:
    """Per-row L2 normalise a [N, D] float32 array (returns a new array)."""
    out = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))
    if np.any(norms <= 0.0):
        raise RCECError("zero-norm patch vector encountered during L2 normalisation")
    out /= norms[:, None]
    return out


def _concat_and_l2(x_d: np.ndarray, x_c: np.ndarray, dino_weight: float = 0.5) -> np.ndarray:
    """A1-style concat: (w*d, (1-w)*c) then per-row L2, float32."""
    z = np.concatenate([dino_weight * x_d, (1.0 - dino_weight) * x_c], axis=-1).astype(np.float32)
    faiss.normalize_L2(z)
    return z


# ---------------------------------------------------------------------------
# Paired alignment
# ---------------------------------------------------------------------------

def align_and_normalize_paired_features(
    dino_patch: np.ndarray,
    clip_patch: np.ndarray,
    dino_ref: np.ndarray,
    clip_ref: np.ndarray,
    dino_sample_ids: np.ndarray,
    clip_sample_ids: np.ndarray,
    dino_grid: Sequence[int],
    resize_fn: callable,
) -> dict:
    """Reorder CLIP test samples to the DINO order, resize the CLIP grids to the
    DINO grid, then per-patch L2-normalise both branches (test + reference).

    Returns a dict with ``d_feat``, ``c_feat`` (N,H,W,D), ``d_ref``, ``c_ref``
    (S,H,W,D), ``grid`` and ``candidate_order``. Raises AlignmentError on any
    sample-order mismatch.
    """
    from industrial_ad.fusion.alignment import build_alignment_plan

    dino_ids = np.asarray(dino_sample_ids).reshape(-1)
    clip_ids = np.asarray(clip_sample_ids).reshape(-1)
    try:
        plan = build_alignment_plan(dino_ids, clip_ids)
    except ValueError as exc:  # alignment raises ValueError on mismatch
        raise AlignmentError(f"sample alignment failed: {exc}") from exc

    grid = tuple(int(v) for v in dino_grid)
    c_feat_raw = np.asarray(clip_patch, dtype=np.float32)[plan.candidate_order]
    c_ref_raw = np.asarray(clip_ref, dtype=np.float32)
    c_feat = np.asarray(resize_fn(c_feat_raw, grid), dtype=np.float32)
    c_ref = np.asarray(resize_fn(c_ref_raw, grid), dtype=np.float32)
    expected_feat = (c_feat_raw.shape[0],) + grid + (c_feat_raw.shape[-1],)
    if c_feat.shape != expected_feat:
        raise AlignmentError(
            f"CLIP test resize produced unexpected shape {c_feat.shape}, expected {expected_feat}")
    expected_ref = (c_ref_raw.shape[0],) + grid + (c_ref_raw.shape[-1],)
    if c_ref.shape != expected_ref:
        raise AlignmentError(
            f"CLIP ref resize produced unexpected shape {c_ref.shape}, expected {expected_ref}")

    d_feat = np.asarray(dino_patch, dtype=np.float32)
    d_ref = np.asarray(dino_ref, dtype=np.float32)
    if d_feat.shape[1:3] != grid:
        raise AlignmentError(
            f"DINO test grid {tuple(d_feat.shape[1:3])} != manifest grid {grid}")
    if d_ref.shape[1:3] != grid:
        raise AlignmentError(
            f"DINO ref grid {tuple(d_ref.shape[1:3])} != manifest grid {grid}")

    return {
        "d_feat": _l2_normalize_flat(d_feat.reshape(-1, d_feat.shape[-1])).reshape(d_feat.shape),
        "c_feat": _l2_normalize_flat(c_feat.reshape(-1, c_feat.shape[-1])).reshape(c_feat.shape),
        "d_ref": _l2_normalize_flat(d_ref.reshape(-1, d_ref.shape[-1])).reshape(d_ref.shape),
        "c_ref": _l2_normalize_flat(c_ref.reshape(-1, c_ref.shape[-1])).reshape(c_ref.shape),
        "grid": grid,
        "candidate_order": plan.candidate_order,
    }


# ---------------------------------------------------------------------------
# Paired reference memory
# ---------------------------------------------------------------------------

@dataclass
class PairedMemory:
    """Flat paired memory plus the per-patch metadata needed for LOO."""

    d: np.ndarray  # (N, 768) float32 L2-normalised DINO refs
    c: np.ndarray  # (N, 768) float32 L2-normalised CLIP refs
    ref_image: np.ndarray  # (N,) int32 reference image index
    row: np.ndarray  # (N,) int32
    col: np.ndarray  # (N,) int32
    n_images: int  # number of reference images (== shot)
    grid: tuple[int, int]

    @property
    def size(self) -> int:
        return self.d.shape[0]


def build_paired_reference_memory(
    d_ref: np.ndarray, c_ref: np.ndarray, n_reference_images: int
) -> PairedMemory:
    """Build the paired memory from aligned, normalised reference feature maps.

    ``d_ref`` / ``c_ref`` have shape (S, H, W, D); the first axis is the
    reference-image index (manifest list order), so patch ``(s, r, c)`` of DINO
    and CLIP are paired by construction.
    """
    s = int(d_ref.shape[0])
    if c_ref.shape[0] != s:
        raise AlignmentError(
            f"reference image count mismatch: DINO={s} CLIP={c_ref.shape[0]}")
    if n_reference_images != s:
        raise AlignmentError(
            f"manifest reference count {n_reference_images} != cache ref blocks {s}")
    h, w = int(d_ref.shape[1]), int(d_ref.shape[2])
    if tuple(c_ref.shape[1:3]) != (h, w):
        raise AlignmentError(
            f"reference grids differ after alignment: {tuple(d_ref.shape[1:3])} vs {tuple(c_ref.shape[1:3])}")
    d_flat = np.ascontiguousarray(d_ref.reshape(-1, d_ref.shape[-1]), dtype=np.float32)
    c_flat = np.ascontiguousarray(c_ref.reshape(-1, c_ref.shape[-1]), dtype=np.float32)
    image_idx = np.repeat(np.arange(s, dtype=np.int32), h * w)
    rows, cols = np.meshgrid(np.arange(h, dtype=np.int32), np.arange(w, dtype=np.int32), indexing="ij")
    row_flat = np.tile(rows.reshape(-1), s)
    col_flat = np.tile(cols.reshape(-1), s)
    return PairedMemory(
        d=d_flat, c=c_flat, ref_image=image_idx, row=row_flat, col=col_flat,
        n_images=s, grid=(h, w),
    )


def shuffled_paired_memory(memory: PairedMemory, rng: np.random.Generator) -> PairedMemory:
    """Destroy the DINO<->CLIP pairing while keeping both marginal sets intact.

    The CLIP patch collection is permuted; metadata (ref_image/row/col) and the
    DINO side are left untouched so the shuffle only affects the correspondence.
    """
    perm = rng.permutation(memory.size)
    return PairedMemory(
        d=memory.d,
        c=memory.c[perm],
        ref_image=memory.ref_image,
        row=memory.row,
        col=memory.col,
        n_images=memory.n_images,
        grid=memory.grid,
    )


# ---------------------------------------------------------------------------
# Chunked FAISS search
# ---------------------------------------------------------------------------

def _search_chunked(
    index: faiss.Index, query_flat: np.ndarray, k: int, chunk: int = 16384
) -> tuple[np.ndarray, np.ndarray]:
    n = int(query_flat.shape[0])
    dists = np.empty((n, k), dtype=np.float32)
    idx = np.empty((n, k), dtype=np.int64)
    q = np.ascontiguousarray(query_flat, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d, i = index.search(q[start:end], k)
        dists[start:end] = d
        idx[start:end] = i
    return dists, idx


def _faiss_index(feats: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(int(feats.shape[1]))
    index.add(np.ascontiguousarray(feats, dtype=np.float32))
    return index


# ---------------------------------------------------------------------------
# Scores on the full (non-excluded) memory for test queries
# ---------------------------------------------------------------------------

def compute_a1_dists(
    d_feat: np.ndarray, c_feat: np.ndarray, memory: PairedMemory,
    chunk: int = 16384, dino_weight: float = 0.5,
) -> np.ndarray:
    """A1 concat half-L2 distance for every test patch (frozen A1 path)."""
    zq = _concat_and_l2(
        d_feat.reshape(-1, d_feat.shape[-1]), c_feat.reshape(-1, c_feat.shape[-1]), dino_weight
    )
    zr = _concat_and_l2(memory.d, memory.c, dino_weight)
    index = _faiss_index(zr)
    dists, _ = _search_chunked(index, zq, 1, chunk)
    return (dists[:, 0] / 2.0).astype(np.float32)


def compute_dino_dists(
    d_feat: np.ndarray, memory: PairedMemory, chunk: int = 16384
) -> np.ndarray:
    """Matched DINO-only half-L2 distance for every test patch."""
    dq = np.ascontiguousarray(d_feat.reshape(-1, d_feat.shape[-1]), dtype=np.float32)
    index = _faiss_index(memory.d)
    dists, _ = _search_chunked(index, dq, 1, chunk)
    return (dists[:, 0] / 2.0).astype(np.float32)


def compute_dino_duplicate_dists(
    d_feat: np.ndarray, memory: PairedMemory, chunk: int = 16384,
    dino_weight: float = 0.5,
) -> np.ndarray:
    """1536-D dimension control: concat of DINO with itself (w, 1-w = 0.5/0.5).

    Both the query side and the reference side use ``norm([0.5*d; 0.5*d])`` so
    the half squared-L2 distance equals the original DINO distance up to float
    rounding; used as a control experiment (task book 5.1).
    """
    dq = np.ascontiguousarray(d_feat.reshape(-1, d_feat.shape[-1]), dtype=np.float32)
    zq = _concat_and_l2(dq, dq, dino_weight)
    zr = _concat_and_l2(memory.d, memory.d, dino_weight)
    index = _faiss_index(zr)
    dists, _ = _search_chunked(index, zq, 1, chunk)
    return (dists[:, 0] / 2.0).astype(np.float32)


def compute_conditional_scores(
    d_feat: np.ndarray,
    c_feat: np.ndarray,
    memory: PairedMemory,
    direction: str,
    k: int,
    chunk: int = 16384,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Conditional cross-encoder scores on the full memory (test queries).

    Returns ``(r_cd, r_dc)`` where ``r_dc`` is None unless
    ``direction == "symmetric"``. ``r_cd`` is the DINO->CLIP conditional score
    (min CLIP distance over the Top-k DINO neighbours).
    """
    if k < 1:
        raise RCECError(f"k must be >= 1, got {k}")
    dq = np.ascontiguousarray(d_feat.reshape(-1, d_feat.shape[-1]), dtype=np.float32)
    cq = np.ascontiguousarray(c_feat.reshape(-1, c_feat.shape[-1]), dtype=np.float32)

    index_d = _faiss_index(memory.d)
    _, nbr_d = _search_chunked(index_d, dq, k, chunk)
    c_nbr = memory.c[nbr_d]  # (n, k, D)
    sq_c = np.sum((c_nbr - cq[:, None, :]) ** 2, axis=-1) / 2.0  # (n, k)
    r_cd = sq_c.min(axis=1)

    r_dc: Optional[np.ndarray] = None
    if direction == "symmetric":
        index_c = _faiss_index(memory.c)
        _, nbr_c = _search_chunked(index_c, cq, k, chunk)
        d_nbr = memory.d[nbr_c]
        sq_d = np.sum((d_nbr - dq[:, None, :]) ** 2, axis=-1) / 2.0
        r_dc = sq_d.min(axis=1)
    elif direction != "dino_to_clip":
        raise RCECError(f"unknown direction: {direction}")

    return r_cd.astype(np.float32), r_dc.astype(np.float32) if r_dc is not None else None


# ---------------------------------------------------------------------------
# Reference-only LOO statistics
# ---------------------------------------------------------------------------

def _loo_allowed_mask(memory: PairedMemory, shot: int) -> np.ndarray:
    """Per-query (N, N) boolean mask of allowed LOO neighbours.

    shot>=2: exclude the whole reference image of the query patch.
    shot==1: exclude the query patch itself plus Chebyshev radius-1 neighbours.
    """
    n = memory.size
    allowed = np.ones((n, n), dtype=bool)
    image = memory.ref_image
    if shot >= 2:
        for q in range(n):
            allowed[q, image == image[q]] = False
    else:
        rows = memory.row
        cols = memory.col
        for q in range(n):
            same_img = image == image[q]
            within_r = np.abs(rows - rows[q]) <= 1
            within_c = np.abs(cols - cols[q]) <= 1
            allowed[q, same_img & within_r & within_c] = False
    return allowed


def compute_reference_loo_statistics(
    d_ref: np.ndarray,
    c_ref: np.ndarray,
    n_reference_images: int,
    direction: str,
    k: int,
    shot: int,
    chunk: int = 16384,
) -> dict:
    """Reference-only leave-one-out scores for s_A1 and the conditional term.

    Returns a dict::

        {
          "a1_loo": np.ndarray (N_ref,) half-L2 A1 distance under exclusion,
          "cond_loo": np.ndarray (N_ref,) conditional score under exclusion,
          "n_ref_patches": int,
          "exclusion_rule": str,
        }

    shot>=2 uses per-reference-image FAISS indexes (leave-one-reference-image-out);
    shot==1 searches the full index once and filters self + Chebyshev radius-1
    neighbours per query patch.
    """
    memory = build_paired_reference_memory(d_ref, c_ref, n_reference_images)
    n = memory.size

    z_ref = _concat_and_l2(memory.d, memory.c)
    z_flat = np.ascontiguousarray(z_ref, dtype=np.float32)

    a1_loo = np.empty(n, dtype=np.float32)
    cond_loo = np.empty(n, dtype=np.float32)

    if shot >= 2:
        for s_img in range(n_reference_images):
            sel = memory.ref_image == s_img
            other = np.nonzero(~sel)[0]
            if other.size < k:
                raise CandidateShortageError(
                    f"after LOO exclusion only {other.size} candidates remain, need k={k}")
            # ---- A1 LOO in the concat space ----
            z_other = z_flat[other]
            index_z = _faiss_index(z_other)
            d1, _ = _search_chunked(index_z, z_flat[sel], 1, chunk)
            a1_loo[sel] = d1[:, 0] / 2.0

            # ---- conditional term ----
            d_sel = np.ascontiguousarray(memory.d[sel], dtype=np.float32)
            c_sel = np.ascontiguousarray(memory.c[sel], dtype=np.float32)
            index_d = _faiss_index(memory.d[other])
            _, nbr_local = _search_chunked(index_d, d_sel, k, chunk)
            global_idx = other[nbr_local]  # (n_sel, k) global ref indices
            sq_c = np.sum((memory.c[global_idx] - c_sel[:, None, :]) ** 2, axis=-1) / 2.0
            r_cd = sq_c.min(axis=1)

            if direction == "symmetric":
                index_c = _faiss_index(memory.c[other])
                _, nbr_c_local = _search_chunked(index_c, c_sel, k, chunk)
                global_idx_c = other[nbr_c_local]
                sq_d = np.sum((memory.d[global_idx_c] - d_sel[:, None, :]) ** 2, axis=-1) / 2.0
                r_cd = 0.5 * (r_cd + sq_d.min(axis=1))
            elif direction != "dino_to_clip":
                raise RCECError(f"unknown direction: {direction}")
            cond_loo[sel] = r_cd
    else:
        # ---- shot == 1: search the full index once, filter per patch ----
        index_z = _faiss_index(z_flat)
        full_d, full_i = _search_chunked(index_z, z_flat, n, chunk)
        index_d = _faiss_index(memory.d)
        full_d_d, full_i_d = _search_chunked(index_d, memory.d, n, chunk)
        index_c = _faiss_index(memory.c)
        _, full_i_c = _search_chunked(index_c, memory.c, n, chunk)

        rows = memory.row
        cols = memory.col
        for q in range(n):
            within_r = np.abs(rows - rows[q]) <= 1
            within_c = np.abs(cols - cols[q]) <= 1
            blocked = within_r & within_c
            # A1: nearest allowed neighbour in concat space.
            allowed_flag = ~blocked[full_i[q]]
            sel_q = full_i[q][allowed_flag]
            if sel_q.size < k:
                raise CandidateShortageError(
                    f"after LOO exclusion only {sel_q.size} candidates remain, need k={k}")
            a1_loo[q] = full_d[q][allowed_flag][0] / 2.0
            # Conditional: DINO top-k allowed neighbours -> CLIP distances.
            d_allowed = full_i_d[q][~blocked[full_i_d[q]]][:k]
            sq_c = np.sum((memory.c[d_allowed] - memory.c[q : q + 1]) ** 2, axis=-1) / 2.0
            r_cd = float(sq_c.min())
            if direction == "symmetric":
                c_allowed = full_i_c[q][~blocked[full_i_c[q]]][:k]
                sq_d2 = np.sum((memory.d[c_allowed] - memory.d[q : q + 1]) ** 2, axis=-1) / 2.0
                r_cd = 0.5 * (r_cd + float(sq_d2.min()))
            elif direction != "dino_to_clip":
                raise RCECError(f"unknown direction: {direction}")
            cond_loo[q] = r_cd

    exclusion_rule = (
        "leave_one_reference_image_out" if shot >= 2 else "self_and_chebyshev_radius_1"
    )
    return {
        "a1_loo": a1_loo,
        "cond_loo": cond_loo,
        "n_ref_patches": int(n),
        "exclusion_rule": exclusion_rule,
    }


def compute_reference_stats(scores: np.ndarray, epsilon: float = 1e-6) -> dict:
    """Median / MAD statistics of a reference score set."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise RCECError("empty reference score set")
    if not np.all(np.isfinite(scores)):
        raise RCECError("reference scores contain NaN/Inf")
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad < epsilon:
        raise CalibrationDegenerateError(
            f"reference MAD {mad:.3e} < epsilon {epsilon:.1e}; configuration is degenerate")
    return {"median": median, "mad": mad, "n": int(scores.size)}


def robust_z_from_reference(
    x: np.ndarray,
    ref_stats: dict,
    epsilon: float = 1e-6,
    z_clip: tuple[float, float] = (-5.0, 10.0),
) -> np.ndarray:
    """Robust z-score using reference-only median/MAD statistics."""
    mad = float(ref_stats["mad"])
    if mad < epsilon:
        raise CalibrationDegenerateError(
            f"reference MAD {mad:.3e} < epsilon {epsilon:.1e}")
    z = (np.asarray(x, dtype=np.float64) - float(ref_stats["median"])) / (
        1.4826 * mad + epsilon
    )
    return np.clip(z, z_clip[0], z_clip[1]).astype(np.float64)


# ---------------------------------------------------------------------------
# Combination and aggregation
# ---------------------------------------------------------------------------

def combine_rcec_scores(
    s_a1: np.ndarray,
    r_cond: np.ndarray,
    ref_stats_a1: dict,
    ref_stats_cond: dict,
    lam: float,
    epsilon: float = 1e-6,
    z_clip: tuple[float, float] = (-5.0, 10.0),
) -> np.ndarray:
    """S = (1-lam)*rz(s_A1) + lam*rz(r_cond)."""
    if not 0.0 <= lam <= 1.0:
        raise RCECError(f"lambda must be in [0,1], got {lam}")
    z_a1 = robust_z_from_reference(s_a1, ref_stats_a1, epsilon, z_clip)
    z_cond = robust_z_from_reference(r_cond, ref_stats_cond, epsilon, z_clip)
    return ((1.0 - lam) * z_a1 + lam * z_cond).astype(np.float64)


def aggregate_image_score(
    pixel_maps: np.ndarray, mode: str = "max", top_q: Optional[float] = None
) -> np.ndarray:
    """Aggregate full-resolution maps to per-image scores.

    ``max`` is the A1 baseline; ``top_q`` modes use the mean of the top q
    fraction of *per-image* patch scores. Never modifies the pixel map.
    """
    flat = pixel_maps.reshape(pixel_maps.shape[0], -1)
    if mode == "max":
        return flat.max(axis=1).astype(np.float64)
    if mode.startswith("top") and top_q is not None:
        if not 0.0 < top_q <= 1.0:
            raise RCECError(f"top_q must be in (0,1], got {top_q}")
        n_patches = flat.shape[1]
        k = max(1, int(round(n_patches * top_q)))
        part = np.partition(flat, -k, axis=1)[:, -k:]
        return part.mean(axis=1).astype(np.float64)
    raise RCECError(f"unknown aggregation mode: {mode}")


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYS = ("gt_sp", "imgs_masks", "labels", "pixel_gt", "gt_masks")


def validate_no_label_inputs(payload: dict) -> None:
    """Raise if any label/mask field leaks into an algorithm function call."""
    bad = [k for k in payload if k in _FORBIDDEN_KEYS]
    if bad:
        raise RCECError(
            f"label/mask fields must never reach RCEC algorithm functions: {bad}")


__all__ = [
    "RCECError",
    "AlignmentError",
    "CalibrationDegenerateError",
    "CandidateShortageError",
    "PairedMemory",
    "align_and_normalize_paired_features",
    "build_paired_reference_memory",
    "shuffled_paired_memory",
    "compute_a1_dists",
    "compute_dino_dists",
    "compute_dino_duplicate_dists",
    "compute_conditional_scores",
    "compute_reference_loo_statistics",
    "compute_reference_stats",
    "robust_z_from_reference",
    "combine_rcec_scores",
    "aggregate_image_score",
    "validate_no_label_inputs",
]
