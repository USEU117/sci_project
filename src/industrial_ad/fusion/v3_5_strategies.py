"""V3.5 Image-level Hierarchical Fusion (Direction C).

Core idea: Use image-level anomaly scores to decide per-image fusion weights,
keeping pixel-level fusion static (z-score calibrated weighted average).

Why this might work where V3.4 failed:
  - Image-level: 1 decision per image (vs 200K decisions per pixel)
  - DINO image-level scores are reliable: when DINO says "definitely normal"
    or "definitely anomalous", it's usually right
  - Text branch helps most when DINO is uncertain (moderate image scores)

Strategies:
  1. discrete_gate  – 3-bin gating (normal / uncertain / anomalous)
  2. continuous_gate – sigmoid-based continuous weight adjustment
  3. agreement_gate – cross-modal agreement at image level
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# Data structure (compatible with V3.3 BranchData)
# ======================================================================

@dataclass
class BranchData:
    """Single branch prediction data for one category."""
    name: str
    anomaly_maps: np.ndarray   # [N, H, W] float32
    image_scores: np.ndarray   # [N] float32  -- NOW USED (was ignored in V3.3)
    gt_labels: np.ndarray      # [N] uint8
    gt_masks: np.ndarray       # [N, H, W] uint8
    sample_ids: np.ndarray     # [N] str

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)


# ======================================================================
# Calibration utilities (from V3.3)
# ======================================================================

def estimate_robust_stats(
    maps: np.ndarray, masks: np.ndarray, normal_only: bool = True
) -> Tuple[float, float]:
    if normal_only:
        normal_mask = ~masks.astype(bool).any(axis=(1, 2))
        if normal_mask.sum() > 0:
            vals = maps[normal_mask].ravel()
        else:
            vals = maps.ravel()
    else:
        vals = maps.ravel()
    center = float(np.median(vals))
    scale = float(np.subtract(*np.percentile(vals, [75, 25]))) + 1e-8
    return center, scale


def estimate_image_score_stats(
    image_scores: np.ndarray,
    gt_labels: np.ndarray,
) -> Tuple[float, float]:
    """Estimate center and scale of image-level scores from normal samples only."""
    normal_mask = gt_labels == 0
    if normal_mask.sum() > 0:
        normal_scores = image_scores[normal_mask]
    else:
        normal_scores = image_scores
    center = float(np.median(normal_scores))
    scale = float(np.median(np.abs(normal_scores - center)))
    if scale < 1e-8:
        scale = 1.0
    return center, scale


def compute_z_score(maps: np.ndarray, center: float, scale: float) -> np.ndarray:
    scale = max(scale, 1e-8)
    return (maps.astype(np.float64) - center) / scale


# ======================================================================
# Strategy 1: Discrete Gate (3-bin)
# ======================================================================

def discrete_gate_fusion(
    dino_branch: BranchData,
    text_branch: BranchData,
    low_threshold: float = 1.5,
    high_threshold: float = 2.5,
    w_confident: float = 0.85,
    w_uncertain: float = 0.40,
) -> np.ndarray:
    """Per-image discrete gating based on DINO image-level z-score.

    Three regimes:
      |z_dino| > high_threshold  → DINO very confident → dino_weight = w_confident
      |z_dino| < low_threshold   → DINO uncertain      → dino_weight = w_uncertain
      otherwise                   → moderate confidence → dino_weight = 0.60 (default)

    Pixel-level fusion within each image: static z-score calibrated weighted average.

    Parameters
    ----------
    dino_branch : DINO visual branch with valid image_scores
    text_branch : AnomalyCLIP text branch with anomaly_maps
    low_threshold, high_threshold : z-score thresholds for DINO image scores
    w_confident : DINO weight when DINO is confident
    w_uncertain : DINO weight when DINO is uncertain (more text)

    Returns
    -------
    fused_maps : [N, H, W] float64
    """
    dino_maps = dino_branch.anomaly_maps.astype(np.float64)
    text_maps = text_branch.anomaly_maps.astype(np.float64)
    masks = dino_branch.gt_masks
    dino_img_scores = dino_branch.image_scores

    # Image-level z-score for DINO
    img_center, img_scale = estimate_image_score_stats(
        dino_img_scores, dino_branch.gt_labels
    )
    dino_img_z = (dino_img_scores - img_center) / img_scale

    # Pixel-level z-score calibration (global, same as V3.3)
    d_center, d_scale = estimate_robust_stats(dino_maps, masks)
    dino_z = compute_z_score(dino_maps, d_center, d_scale)

    t_center, t_scale = estimate_robust_stats(text_maps, masks)
    text_z = compute_z_score(text_maps, t_center, t_scale)

    N = dino_maps.shape[0]
    fused = np.zeros_like(dino_maps)

    for i in range(N):
        z = dino_img_z[i]
        if abs(z) > high_threshold:
            w_dino = w_confident
        elif abs(z) < low_threshold:
            w_dino = w_uncertain
        else:
            w_dino = 0.60  # moderate confidence → default V3.3 weight

        fused[i] = w_dino * dino_z[i] + (1.0 - w_dino) * text_z[i]

    # Map back to DINO scale
    fused = fused * d_scale + d_center

    return fused


# ======================================================================
# Strategy 2: Continuous Gate (sigmoid)
# ======================================================================

def continuous_gate_fusion(
    dino_branch: BranchData,
    text_branch: BranchData,
    w_min: float = 0.35,
    w_max: float = 0.90,
    steepness: float = 1.0,
) -> np.ndarray:
    """Continuous sigmoid-based weight adjustment.

    w_dino(z) = w_min + (w_max - w_min) * sigmoid(steepness * |z| - 2.0)

    Where z is the DINO image-level z-score.
    - When |z| ≈ 0 (DINO unsure)  → w_dino ≈ w_min (more text)
    - When |z| >> 2 (DINO sure)    → w_dino ≈ w_max (more DINO)

    Pixel-level fusion within each image: static z-score calibrated weighted average.
    """
    dino_maps = dino_branch.anomaly_maps.astype(np.float64)
    text_maps = text_branch.anomaly_maps.astype(np.float64)
    masks = dino_branch.gt_masks
    dino_img_scores = dino_branch.image_scores

    # Image-level z-score for DINO
    img_center, img_scale = estimate_image_score_stats(
        dino_img_scores, dino_branch.gt_labels
    )
    dino_img_z = np.abs((dino_img_scores - img_center) / img_scale)

    # Sigmoid: map |z| from [0, ~4+] to [0, 1]
    sigmoid = 1.0 / (1.0 + np.exp(-steepness * (dino_img_z - 2.0)))
    dino_weights = w_min + (w_max - w_min) * sigmoid  # [N]

    # Pixel-level z-score calibration
    d_center, d_scale = estimate_robust_stats(dino_maps, masks)
    dino_z = compute_z_score(dino_maps, d_center, d_scale)

    t_center, t_scale = estimate_robust_stats(text_maps, masks)
    text_z = compute_z_score(text_maps, t_center, t_scale)

    N = dino_maps.shape[0]
    fused = np.zeros_like(dino_maps)
    for i in range(N):
        w = dino_weights[i]
        fused[i] = w * dino_z[i] + (1.0 - w) * text_z[i]

    fused = fused * d_scale + d_center
    return fused


# ======================================================================
# Strategy 3: Cross-modal Agreement Gate
# ======================================================================

def agreement_gate_fusion(
    dino_branch: BranchData,
    text_branch: BranchData,
    w_default: float = 0.60,
    w_text_boost: float = 0.45,
    agreement_threshold: float = 1.0,
) -> np.ndarray:
    """Per-image gating based on DINO-text image-level agreement.

    When DINO and text agree at image level (both say normal or both say anomalous):
      → DINO is probably right → use default weight (60:40)

    When they disagree (e.g., DINO says normal, text says anomalous):
      → Something interesting might be happening → boost text weight
      → Use lower DINO weight (w_text_boost)

    Agreement is measured by: sign(DINO_z) * sign(text_z) > 0
    """
    dino_maps = dino_branch.anomaly_maps.astype(np.float64)
    text_maps = text_branch.anomaly_maps.astype(np.float64)
    masks = dino_branch.gt_masks

    # Image-level z-scores for both branches
    d_center, d_scale = estimate_image_score_stats(
        dino_branch.image_scores, dino_branch.gt_labels
    )
    dino_img_z = (dino_branch.image_scores - d_center) / d_scale

    # For text branch, use max pixel value as proxy for image score
    # (since original text image_scores may not be reliable)
    text_img_scores = text_maps.max(axis=(1, 2))
    t_center, t_scale = estimate_image_score_stats(
        text_img_scores, dino_branch.gt_labels
    )
    text_img_z = (text_img_scores - t_center) / t_scale

    # Agreement: both above threshold → agree
    agree = (dino_img_z > agreement_threshold) == (text_img_z > agreement_threshold)
    dino_weights = np.where(agree, w_default, w_text_boost)  # [N]

    # Pixel-level fusion
    d_center_px, d_scale_px = estimate_robust_stats(dino_maps, masks)
    dino_z = compute_z_score(dino_maps, d_center_px, d_scale_px)

    t_center_px, t_scale_px = estimate_robust_stats(text_maps, masks)
    text_z = compute_z_score(text_maps, t_center_px, t_scale_px)

    N = dino_maps.shape[0]
    fused = np.zeros_like(dino_maps)
    for i in range(N):
        w = float(dino_weights[i])
        fused[i] = w * dino_z[i] + (1.0 - w) * text_z[i]

    fused = fused * d_scale_px + d_center_px
    return fused


# ======================================================================
# Strategy 4: Oracle Upper Bound for Image-level Gating
# ======================================================================

def oracle_image_gate_fusion(
    dino_branch: BranchData,
    text_branch: BranchData,
) -> Tuple[np.ndarray, dict]:
    """Oracle: use ground-truth image labels to select best per-image weight.

    For each image:
      - If DINO pixel AP > text pixel AP → w_dino = 1.0
      - Else → w_dino = 0.0

    This gives the upper bound for image-level gating.

    Returns (fused_maps, stats).
    """
    from sklearn.metrics import average_precision_score

    dino_maps = dino_branch.anomaly_maps.astype(np.float64)
    text_maps = text_branch.anomaly_maps.astype(np.float64)
    masks = dino_branch.gt_masks

    N = dino_maps.shape[0]
    fused = np.zeros_like(dino_maps)

    dino_choices = 0
    text_choices = 0

    for i in range(N):
        gt = masks[i].ravel() > 0.5
        if gt.sum() == 0:
            # Normal image: trust DINO
            fused[i] = dino_maps[i]
            dino_choices += 1
        else:
            dino_ap = average_precision_score(gt, dino_maps[i].ravel())
            text_ap = average_precision_score(gt, text_maps[i].ravel())
            if dino_ap >= text_ap:
                fused[i] = dino_maps[i]
                dino_choices += 1
            else:
                fused[i] = text_maps[i]
                text_choices += 1

    stats = {
        "dino_choices": dino_choices,
        "text_choices": text_choices,
        "text_ratio": text_choices / N if N > 0 else 0,
    }
    return fused, stats


# ======================================================================
# Hyperparameter grid
# ======================================================================

def build_v3_5_variants() -> List[dict]:
    """Build all V3.5 strategy variants for grid search."""
    variants = []

    # --- Baseline: V3.3 static (60:40) for comparison ---
    variants.append({
        "strategy": "v3_3_static",
        "variant_name": "v3_3_static_dino=0.60",
    })

    # --- Strategy 1: Discrete Gate (key combos) ---
    for low_t in [1.0, 1.5, 2.0]:
        for high_t in [2.0, 2.5, 3.0]:
            if high_t <= low_t:
                continue
            # Representative weights
            variants.append({
                "strategy": "discrete_gate",
                "variant_name": f"lo={low_t}_hi={high_t}_conf=0.85_unc=0.40",
                "low_threshold": low_t,
                "high_threshold": high_t,
                "w_confident": 0.85,
                "w_uncertain": 0.40,
            })

    # --- Strategy 2: Continuous Gate (key combos) ---
    for w_min in [0.30, 0.40]:
        for w_max in [0.85, 0.90]:
            for steep in [1.0, 2.0]:
                variants.append({
                    "strategy": "continuous_gate",
                    "variant_name": f"min={w_min}_max={w_max}_steep={steep}",
                    "w_min": w_min,
                    "w_max": w_max,
                    "steepness": steep,
                })

    # --- Strategy 3: Agreement Gate (key combos) ---
    for w_boost in [0.35, 0.45]:
        for thr in [0.5, 1.0, 1.5]:
            variants.append({
                "strategy": "agreement_gate",
                "variant_name": f"boost={w_boost}_thr={thr}",
                "w_text_boost": w_boost,
                "agreement_threshold": thr,
            })

    return variants


# ======================================================================
# Dispatcher
# ======================================================================

def run_fusion(
    variant: dict,
    dino_branch: BranchData,
    text_branch: BranchData,
) -> np.ndarray:
    """Run a single V3.5 fusion variant."""
    strategy = variant["strategy"]

    if strategy == "v3_3_static":
        # Classic V3.3: z-score calibrate + 60:40 static
        from industrial_ad.fusion.v3_3_strategies import weighted_ensemble_fusion
        branches = {
            "anomalydino_visual": dino_branch,
            "anomalyclip_text": text_branch,
        }
        weights = {"anomalydino_visual": 0.60, "anomalyclip_text": 0.40}
        return weighted_ensemble_fusion(branches, weights, calibrate=True)

    elif strategy == "discrete_gate":
        return discrete_gate_fusion(
            dino_branch, text_branch,
            low_threshold=variant["low_threshold"],
            high_threshold=variant["high_threshold"],
            w_confident=variant["w_confident"],
            w_uncertain=variant["w_uncertain"],
        )

    elif strategy == "continuous_gate":
        return continuous_gate_fusion(
            dino_branch, text_branch,
            w_min=variant["w_min"],
            w_max=variant["w_max"],
            steepness=variant["steepness"],
        )

    elif strategy == "agreement_gate":
        return agreement_gate_fusion(
            dino_branch, text_branch,
            w_text_boost=variant["w_text_boost"],
            agreement_threshold=variant["agreement_threshold"],
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
