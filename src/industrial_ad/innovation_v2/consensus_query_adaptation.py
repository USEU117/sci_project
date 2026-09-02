"""Route C — CE-CQA: Cross-Encoder Consensus-Bounded Query Adaptation.

Task book section 7. Per query image:
  1. rank DINO-only and CLIP-only 1-NN distances *within the image*;
  2. consensus pseudo-normal set P_q = patches normal by both branches (rank <= q);
  3. |P_q| < 16 -> fall back to A1 for that image;
  4. estimate per-branch coordinate-median shifts Delta_D / Delta_C from the A1
     nearest normal patch of each selected patch;
  5. cap the combined shift norm at the 95th percentile of reference-only LOO
     fused-difference norms (no test statistics);
  6. shift the reference memory by eta, re-normalise per branch, concat + KNN.

Query patches are never inserted into the memory (no self-matching).
"""

from __future__ import annotations

import numpy as np

from industrial_ad.fusion import rcec
from industrial_ad.innovation_v2.common import (
    AlignedFeatures, InnovationError, paired_memory,
)

CAP_PERCENTILE = 95.0
MIN_CONSENSUS = 16


def _ref_loo_diff_norms(mem: rcec.PairedMemory, fused_ref: np.ndarray,
                        shot: int, k_search: int | None = None) -> np.ndarray:
    """||z_i - z_loo(i)|| for every ref patch, LOO (exclude same image / self)."""
    if k_search is None:
        h, w = mem.grid
        k_search = (h * w + 1) if shot >= 2 else 2
    index = rcec._faiss_index(np.ascontiguousarray(fused_ref, dtype=np.float32))
    dists, indices = rcec._search_chunked(
        index, np.ascontiguousarray(fused_ref, dtype=np.float32),
        k=k_search, chunk=16384)
    n = fused_ref.shape[0]
    norms = np.empty(n, dtype=np.float32)
    for i in range(n):
        chosen = None
        for j in range(k_search):
            nb = int(indices[i, j])
            if nb == i:
                continue
            if shot >= 2 and mem.ref_image[nb] == mem.ref_image[i]:
                continue
            chosen = nb
            break
        if chosen is None:
            raise InnovationError(f"CE-CQA LOO: no allowed neighbour for ref patch {i}")
        diff = fused_ref[i] - fused_ref[chosen]
        norms[i] = float(np.linalg.norm(diff))
    return norms


def score_cecqa(
    aligned: AlignedFeatures,
    candidate: dict,
    cfg: dict,
    selection: str = "consensus",
) -> tuple[np.ndarray, dict]:
    q_frac = float(candidate["q"])
    eta = float(candidate["eta"])
    dino_weight = float(cfg.get("fixed", {}).get("dino_weight", 0.5))

    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    n_patches = h * w
    mem = paired_memory(aligned)

    d_feat_flat = np.ascontiguousarray(aligned.d_feat.reshape(-1, 768), dtype=np.float32)
    c_feat_flat = np.ascontiguousarray(aligned.c_feat.reshape(-1, 768), dtype=np.float32)
    d_ref_flat = np.ascontiguousarray(aligned.d_ref.reshape(-1, 768), dtype=np.float32)
    c_ref_flat = np.ascontiguousarray(aligned.c_ref.reshape(-1, 768), dtype=np.float32)
    z_q = rcec._concat_and_l2(d_feat_flat, c_feat_flat, dino_weight)
    z_ref = rcec._concat_and_l2(d_ref_flat, c_ref_flat, dino_weight)

    # A1 baseline grid (fallback per image).
    s_a1 = a1_grid_for(aligned, dino_weight)

    # Shift cap from reference-only LOO difference norms (fused space).
    shot = int(cfg["_shot"])
    loo_norms = _ref_loo_diff_norms(mem, z_ref, shot)
    cap = float(np.percentile(loo_norms, CAP_PERCENTILE))

    # A1 nearest normal patch for arbitrary query patches.
    idx_ref = rcec._faiss_index(np.ascontiguousarray(z_ref, dtype=np.float32))

    scores = np.empty((n, h * w), dtype=np.float32)
    n_consensus = []
    n_fallback = 0
    shift_norms = []

    for t in range(n):
        if selection == "consensus":
            idx_d = rcec._faiss_index(np.ascontiguousarray(
                aligned.d_ref.reshape(-1, 768), dtype=np.float32))
            dist_d, _ = rcec._search_chunked(idx_d, d_feat_flat[t * n_patches:(t + 1) * n_patches],
                                             k=1, chunk=16384)
            idx_c = rcec._faiss_index(np.ascontiguousarray(
                aligned.c_ref.reshape(-1, 768), dtype=np.float32))
            dist_c, _ = rcec._search_chunked(idx_c, c_feat_flat[t * n_patches:(t + 1) * n_patches],
                                             k=1, chunk=16384)
            rank_d = np.argsort(np.argsort(dist_d[:, 0]))
            rank_c = np.argsort(np.argsort(dist_c[:, 0]))
            sel = (rank_d < q_frac * n_patches) & (rank_c < q_frac * n_patches)
        elif selection == "a1_rank":
            dist_a1, _ = rcec._search_chunked(
                idx_ref, z_q[t * n_patches:(t + 1) * n_patches], k=1, chunk=16384)
            rank_a1 = np.argsort(np.argsort(dist_a1[:, 0]))
            sel = rank_a1 < q_frac * n_patches
        else:
            raise InnovationError(f"unknown CE-CQA selection: {selection}")
        sel_idx = np.nonzero(sel)[0]
        n_consensus.append(int(sel_idx.size))
        if sel_idx.size < MIN_CONSENSUS:
            scores[t] = s_a1[t].reshape(-1)
            n_fallback += 1
            continue

        q_sel = z_q[t * n_patches + sel_idx]
        _, nn = rcec._search_chunked(idx_ref, np.ascontiguousarray(q_sel, dtype=np.float32),
                                     k=1, chunk=16384)
        nn_flat = nn[:, 0]
        delta_d = d_feat_flat[t * n_patches + sel_idx] - d_ref_flat[nn_flat]
        delta_c = c_feat_flat[t * n_patches + sel_idx] - c_ref_flat[nn_flat]
        Delta_D = np.median(delta_d, axis=0)
        Delta_C = np.median(delta_c, axis=0)
        combined = float(np.sqrt(np.sum(Delta_D ** 2) + np.sum(Delta_C ** 2)))
        if combined > cap:
            scale = cap / combined
            Delta_D = Delta_D * scale
            Delta_C = Delta_C * scale
            combined = float(np.sqrt(np.sum(Delta_D ** 2) + np.sum(Delta_C ** 2)))
        shift_norms.append(combined)

        d_shift = rcec._l2_normalize_flat(d_ref_flat + eta * Delta_D)
        c_shift = rcec._l2_normalize_flat(c_ref_flat + eta * Delta_C)
        z_shift = rcec._concat_and_l2(d_shift, c_shift, dino_weight)
        idx_shift = rcec._faiss_index(np.ascontiguousarray(z_shift, dtype=np.float32))
        dists, _ = rcec._search_chunked(
            idx_shift, z_q[t * n_patches:(t + 1) * n_patches], k=1, chunk=16384)
        scores[t] = (dists[:, 0] / 2.0)

    diag = {
        "q": q_frac,
        "eta": eta,
        "selection": selection,
        "mean_consensus_patches": round(float(np.mean(n_consensus)), 2),
        "fallback_image_ratio": round(n_fallback / n, 4),
        "cap_loo_p95": round(cap, 6),
        "mean_shift_norm": round(float(np.mean(shift_norms)), 6) if shift_norms else None,
        "n_images": n,
    }
    return scores.reshape(n, h, w), diag


def score_cecqa_a1_rank_only(aligned: AlignedFeatures, candidate: dict, cfg: dict):
    """Small-gate control: pseudo-normal selected by A1 fused rank only."""
    return score_cecqa(aligned, candidate, cfg, selection="a1_rank")


def a1_grid_for(aligned: AlignedFeatures, dino_weight: float = 0.5) -> np.ndarray:
    from industrial_ad.innovation_v2.common import a1_grid
    return a1_grid(aligned, dino_weight=dino_weight)
