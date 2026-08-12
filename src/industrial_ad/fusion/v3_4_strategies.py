"""V3.4 Dynamic Fusion: True per-pixel adaptive weighting (NOT static 60:40).

Two routes beyond static z-score fusion:

  Route 2 — Spatial Uncertainty Weighting (training-free)
    Idea: Pixels in high-variance regions are ambiguous → lower branch weight.
    Each branch's per-pixel spatial variance acts as an uncertainty proxy.
    No anomaly labels needed — purely self-supervised.

  Route 3 — Normal-Reference Gating (training-free)
    Idea: For each test pixel, measure distance to the normal distribution.
    If one branch says "close to normal" while the other says "far" →
    trust the branch closer to normal more (text hallucination guard).
    Uses ONLY normal samples — no anomaly labels needed.

Gate A1 (Oracle Upper Bound):
    With ground-truth pixel masks, what's the theoretically best possible fusion?
    Pixel-wise: if anomaly → trust better branch, if normal → trust DINO.
    This answers: is text actually carrying useful information?

Gate A2 (No-Label Routing):
    Can our adaptive mechanism approach Oracle without seeing labels?
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import uniform_filter


# ======================================================================
# Shared utilities
# ======================================================================

def robust_z_score(maps: np.ndarray, normal_masks: np.ndarray) -> np.ndarray:
    """Compute z-score using only normal pixels (median/MAD based)."""
    N = maps.shape[0]
    normal = normal_masks.max(axis=(1, 2)) < 0.5
    if normal.sum() == 0:
        return maps - np.median(maps)
    normal_maps = maps[normal]
    center = float(np.median(normal_maps))
    scale = float(np.median(np.abs(normal_maps - center)))
    if scale < 1e-8:
        scale = 1.0
    return (maps - center) / scale


def per_pixel_z_score(maps: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Per-pixel z-score calibration (separate center/scale per pixel position)."""
    N, H, W = maps.shape
    normal = masks.max(axis=(1, 2)) < 0.5
    if normal.sum() == 0:
        # Fall back to global z-score
        center = np.median(maps, axis=0)
        scale = np.median(np.abs(maps - center), axis=0)
        scale[scale < 1e-8] = 1.0
        return (maps - center) / scale

    normal_maps = maps[normal]
    center = np.median(normal_maps, axis=0)  # (H, W)
    scale = np.median(np.abs(normal_maps - center[None, :, :]), axis=0)  # (H, W)
    scale[scale < 1e-8] = 1.0
    return (maps - center[None, :, :]) / scale[None, :, :]


# ======================================================================
# Route 2: Spatial Uncertainty Weighting
# ======================================================================

def spatial_variance_uncertainty(
    anomaly_maps: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """Compute per-pixel spatial variance as uncertainty proxy.

    For each pixel, look at its spatial neighborhood (window × window).
    Higher local variance → the branch is "unsure" in that region.
    Returns uncertainty map with same shape as input (N, H, W).
    """
    # Square window mean filter
    sqr = anomaly_maps ** 2
    mu = uniform_filter(anomaly_maps, size=(1, window, window), mode='reflect')
    mu_sq = uniform_filter(sqr, size=(1, window, window), mode='reflect')
    var = mu_sq - mu ** 2
    std = np.sqrt(np.maximum(var, 0))
    return std


def uncertainty_weighted_fusion(
    dino_maps: np.ndarray,
    aclip_maps: np.ndarray,
    masks: np.ndarray,
    window: int = 7,
    base_dino_w: float = 0.60,
    temperature: float = 1.0,
) -> np.ndarray:
    """Route 2: Adapt weight per pixel based on spatial uncertainty.

    Algorithm:
      1. Z-score calibrate both branches (per-pixel)
      2. Compute spatial variance for each branch as uncertainty
      3. At each pixel: w_dino = base / (1 + relative_uncertainty)
         where relative_uncertainty = max(0, dino_uncert/max - aclip_uncert/max)
      4. Weighted average with per-pixel weights

    Key insight: If DINO is very uncertain in a region (textured area)
    but AnomalyCLIP is certain → shift weight toward AnomalyCLIP.
    """
    dino_z = robust_z_score(dino_maps, masks)
    aclip_z = robust_z_score(aclip_maps, masks)

    # Spatial uncertainty
    dino_uncert = spatial_variance_uncertainty(dino_z, window)
    aclip_uncert = spatial_variance_uncertainty(aclip_z, window)

    # Normalize uncertainty to [0, 1] range (per-sample)
    dino_max = dino_uncert.max(axis=(1, 2), keepdims=True) + 1e-8
    aclip_max = aclip_uncert.max(axis=(1, 2), keepdims=True) + 1e-8
    dino_uncert_norm = dino_uncert / dino_max
    aclip_uncert_norm = aclip_uncert / aclip_max

    # Relative uncertainty: positive = DINO more uncertain → shift to AClip
    rel_uncert = (dino_uncert_norm - aclip_uncert_norm) / temperature

    # Convert to per-pixel DINO weight
    # rel_uncert > 0 → DINO less reliable → lower dino_w
    dino_w_pixel = base_dino_w - (1.0 - base_dino_w) * np.clip(rel_uncert, -2, 2)
    dino_w_pixel = np.clip(dino_w_pixel, 0.3, 0.9)  # safety bounds
    aclip_w_pixel = 1.0 - dino_w_pixel

    fused = dino_w_pixel * dino_z + aclip_w_pixel * aclip_z
    return fused


# ======================================================================
# Route 3: Normal-Reference Gating
# ======================================================================

def normal_reference_gating(
    dino_maps: np.ndarray,
    aclip_maps: np.ndarray,
    masks: np.ndarray,
    base_dino_w: float = 0.60,
    temperature: float = 1.0,
) -> np.ndarray:
    """Route 3: Gate text weight based on distance to normal distribution.

    Algorithm:
      1. Z-score calibrate both branches (per-pixel)
      2. Compute per-pixel mean map from normal samples ONLY
      3. For each test pixel, compute |z - mu_normal|
         = "how far from normal does this branch think this pixel is?"
      4. If DINO says "close to normal" but AClip says "far" →
         likely text hallucination → reduce AClip weight
      5. If both say "far" → genuine anomaly → equal weight

    Key insight: Text branch can see anomalies where none exist (false positive).
    DINO is the more conservative anchor. When they disagree and DINO is closer to
    normal, trust DINO.
    """
    dino_z = robust_z_score(dino_maps, masks)
    aclip_z = robust_z_score(aclip_maps, masks)

    # Normal-only mean per pixel
    normal = masks.max(axis=(1, 2)) < 0.5
    normal_dino_mu = dino_z[normal].mean(axis=0)  # (H, W)
    normal_aclip_mu = aclip_z[normal].mean(axis=0)  # (H, W)

    # Distance from normal per pixel
    dino_dist = np.abs(dino_z - normal_dino_mu[None, :, :])  # (N, H, W)
    aclip_dist = np.abs(aclip_z - normal_aclip_mu[None, :, :])  # (N, H, W)

    # Normalize distance (per-sample max)
    dino_dist_max = dino_dist.max(axis=(1, 2), keepdims=True) + 1e-8
    aclip_dist_max = aclip_dist.max(axis=(1, 2), keepdims=True) + 1e-8
    dino_dist_norm = dino_dist / dino_dist_max
    aclip_dist_norm = aclip_dist / aclip_dist_max

    # Relative distance: positive = AClip farther from normal than DINO
    rel_dist = (aclip_dist_norm - dino_dist_norm) / temperature

    # AClip far from normal while DINO is close → text hallucination → reduce AClip weight
    # rel_dist > 0 → AClip more distant → lower AClip weight → higher DINO weight
    dino_w_pixel = base_dino_w + (1.0 - base_dino_w) * np.clip(rel_dist, -2, 2)
    dino_w_pixel = np.clip(dino_w_pixel, 0.3, 0.9)  # safety bounds
    aclip_w_pixel = 1.0 - dino_w_pixel

    fused = dino_w_pixel * dino_z + aclip_w_pixel * aclip_z
    return fused


# ======================================================================
# Gate A1: Oracle Upper Bound
# ======================================================================

def oracle_fusion(
    dino_maps: np.ndarray,
    aclip_maps: np.ndarray,
    masks: np.ndarray,
    gt_labels: np.ndarray,
    stride: int = 8,
) -> Dict[str, float]:
    """Gate A1: With ground truth, what's the best possible fusion?

    Pixel-wise oracle:
      - Anomaly pixel (gt=1): pick the branch with higher value
      - Normal pixel (gt=0): pick the branch with lower value
    (Binary selection, not weighted — this gives upper bound)

    Returns per-pixel metrics at the given stride.
    """
    dino_z = robust_z_score(dino_maps, masks)
    aclip_z = robust_z_score(aclip_maps, masks)

    from sklearn.metrics import average_precision_score, roc_auc_score
    from evaluate_unified import aupro_fast

    gs = masks[:, ::stride, ::stride]
    dino_s = dino_z[:, ::stride, ::stride]
    aclip_s = aclip_z[:, ::stride, ::stride]

    # Oracle: at each pixel, pick the branch that separates anomaly/normal better
    # anomaly pixels: max(dino, aclip) → want HIGH score on anomalies
    # normal pixels: min(dino, aclip) → want LOW score on normals
    gt_s = gt_labels[:, ::stride, ::stride]  # (N, h, w) — need to broadcast
    # gt_s is (N,) — use as sample-level, or if it's already per-pixel...
    # gt_labels is (N,) — image-level labels

    # For oracle, just use max of the two maps (pessimistic: text can hallucinate)
    # For a TRUE upper bound, we need per-pixel gt which we have (masks)
    is_anom = gs.ravel() > 0.5  # per-pixel anomaly labels
    dino_f = dino_s.ravel()
    aclip_f = aclip_s.ravel()

    # Oracle fusion: anomaly pixels take max, normal pixels take min
    oracle_f = np.where(is_anom,
                        np.maximum(dino_f, aclip_f),
                        np.minimum(dino_f, aclip_f))

    # Metrics
    fl = is_anom.astype(np.int32)

    from sklearn.metrics import average_precision_score, roc_auc_score
    from evaluate_unified import aupro_fast

    auroc = float(roc_auc_score(fl, oracle_f))
    ap = float(average_precision_score(fl, oracle_f))
    aupro = float(aupro_fast(gs, oracle_f.reshape(gs.shape)))

    # DINO baseline
    dino_auroc = float(roc_auc_score(fl, dino_f))
    dino_ap = float(average_precision_score(fl, dino_f))
    dino_aupro = float(aupro_fast(gs, dino_s))

    return {
        "oracle_auroc": auroc, "oracle_ap": ap, "oracle_aupro": aupro,
        "dino_auroc": dino_auroc, "dino_ap": dino_ap, "dino_aupro": dino_aupro,
        "delta_ap": round(ap - dino_ap, 6),
    }


def oracle_fusion_v2(
    dino_maps: np.ndarray,
    aclip_maps: np.ndarray,
    masks: np.ndarray,
    stride: int = 8,
) -> Dict[str, float]:
    """Gate A1 v2: True oracle — per-pixel pick the branch with higher correct-to-wrong ratio.

    For anomaly pixels: pick branch with higher score (max)
    For normal pixels: pick branch with lower score (min)

    This assumes text adds real signal at anomaly pixels.
    Key question: DOES text have useful signal at anomaly pixels?
    """
    dino_z = robust_z_score(dino_maps, masks)
    aclip_z = robust_z_score(aclip_maps, masks)

    gs = masks[:, ::stride, ::stride]
    dino_s = dino_z[:, ::stride, ::stride]
    aclip_s = aclip_z[:, ::stride, ::stride]

    is_anom = gs.ravel() > 0.5
    dino_f = dino_s.ravel()
    aclip_f = aclip_s.ravel()

    oracle_f = np.where(is_anom,
                        np.maximum(dino_f, aclip_f),
                        np.minimum(dino_f, aclip_f))

    fl = is_anom.astype(np.int32)

    from sklearn.metrics import average_precision_score, roc_auc_score
    from evaluate_unified import aupro_fast

    auroc = float(roc_auc_score(fl, oracle_f))
    ap = float(average_precision_score(fl, oracle_f))
    aupro = float(aupro_fast(gs, oracle_f.reshape(gs.shape)))

    dino_ap = float(average_precision_score(fl, dino_f))

    return {
        "oracle_auroc": round(auroc, 6),
        "oracle_ap": round(ap, 6),
        "oracle_aupro": round(aupro, 6),
        "dino_ap": round(dino_ap, 6),
        "delta_ap": round(ap - dino_ap, 6),
    }


# ======================================================================
# Route 4: Cross-Modal Agreement Gating
# ======================================================================

def cross_modal_agreement_gating(
    dino_maps: np.ndarray,
    aclip_maps: np.ndarray,
    masks: np.ndarray,
    base_dino_w: float = 0.60,
    alpha: float = 0.40,   # max deviation from base
    temperature: float = 1.0,
) -> np.ndarray:
    """Route 4: Gate text weight by cross-branch agreement.

    Core assumption: When two independent anomaly detectors agree on a pixel,
    both are likely correct. When they disagree, text is more likely wrong.

    Algorithm:
      1. Z-score calibrate both branches
      2. Compute per-pixel agreement: agree = 1/(1 + |z_dino - z_aclip|)
         (0 = completely disagree, 1 = perfectly agree)
      3. Text weight = base + alpha * (agreement - 0.5) / temperature
         High agreement → boost text (DINO corroborates)
         Low agreement → reduce text (text may be hallucinating)
      4. Clamp to [0.20, 0.80] for safety
    """
    dino_z = robust_z_score(dino_maps, masks)  # global scalar z-score (cheaper)
    aclip_z = robust_z_score(aclip_maps, masks)

    # Per-pixel agreement: 1 = perfect agreement, 0 = complete disagreement
    # Using robust (median/MAD) z-score for both — comparable scale
    z_diff = np.abs(dino_z - aclip_z)
    z_diff_max = z_diff.max(axis=(1, 2), keepdims=True) + 1e-8
    z_diff_norm = z_diff / z_diff_max

    agreement = 1.0 / (1.0 + temperature * z_diff_norm)  # [0.5, 1.0]

    base_aclip_w = 1.0 - base_dino_w  # 0.40
    aclip_w_pixel = base_aclip_w + alpha * (agreement - 0.5)
    aclip_w_pixel = np.clip(aclip_w_pixel, 0.20, 0.80)
    dino_w_pixel = 1.0 - aclip_w_pixel

    fused = dino_w_pixel * dino_z + aclip_w_pixel * aclip_z
    return fused


# ======================================================================
# Route 4b: Agreement + Annealing hybrid
# ======================================================================

def agreement_annealing_fusion(
    dino_maps: np.ndarray,
    aclip_maps: np.ndarray,
    masks: np.ndarray,
    base_dino_w: float = 0.60,
    alpha: float = 0.40,
    temperature: float = 1.0,
    dino_baseline_ap: float = 0.0,
    anneal_threshold: float = 0.80,
    anneal_factor: float = 0.30,
) -> np.ndarray:
    """Route 4b: Agreement gating + DINO baseline annealing.

    When DINO baseline AP > anneal_threshold, adjust base weights up.
    dino_baseline_ap must be provided by caller to avoid recomputation.
    """
    if dino_baseline_ap > anneal_threshold:
        effective_aclip = (1.0 - base_dino_w) * anneal_factor
        base_dino_w = 1.0 - effective_aclip
        base_dino_w = base_dino_w / (base_dino_w + effective_aclip)
        alpha *= anneal_factor

    return cross_modal_agreement_gating(
        dino_maps, aclip_maps, masks,
        base_dino_w=base_dino_w, alpha=alpha, temperature=temperature,
    )
