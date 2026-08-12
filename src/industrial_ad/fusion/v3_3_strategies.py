"""V3.3 Fusion Strategies: ensemble, per-pixel selection, two-stage calibrated.

Three complementary approaches to replace the failed V3.2 hierarchical rescue:
  1. weighted_ensemble   – z-score normalize branches → weighted average
  2. max_z_selection     – per-pixel pick the most confident branch
  3. two_stage_calibrated – AdaptCLIP internal 3→1 then calibrate with DINO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BranchData:
    """Single branch prediction data for one category."""
    name: str
    anomaly_maps: np.ndarray   # [N, H, W] float32
    image_scores: np.ndarray   # [N] float32
    gt_labels: np.ndarray      # [N] uint8
    gt_masks: np.ndarray       # [N, H, W] uint8
    sample_ids: np.ndarray     # [N] str

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)


@dataclass
class FusionResult:
    """Result of one fusion strategy on one category."""
    strategy: str
    category: str
    variant: str               # e.g. "alpha=0.7" or "max_z"
    pixel_maps: np.ndarray     # [N, H, W] fused anomaly maps
    pixel_auroc: float
    pixel_ap: float
    pixel_aupro: float
    # Optional per-pixel branch selection info
    selected_branch_map: Optional[np.ndarray] = None  # [N, H, W] int (branch index)


# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------

def compute_z_score(maps: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Z-score normalize anomaly maps using (value - center) / scale."""
    scale = max(scale, 1e-8)
    return (maps.astype(np.float64) - center) / scale


def estimate_robust_stats(
    maps: np.ndarray, masks: np.ndarray, normal_only: bool = True
) -> Tuple[float, float]:
    """Estimate robust center and scale from pixel values.

    If normal_only=True, uses only pixels from normal (non-anomalous) images.
    """
    if normal_only:
        normal_mask = ~masks.astype(bool).any(axis=(1, 2))  # completely normal images
        if normal_mask.sum() > 0:
            vals = maps[normal_mask].ravel()
        else:
            vals = maps.ravel()
    else:
        vals = maps.ravel()
    center = float(np.median(vals))
    scale = float(np.subtract(*np.percentile(vals, [75, 25]))) + 1e-8
    return center, scale


# ---------------------------------------------------------------------------
# Strategy 1: Weighted Ensemble (with optional safety annealing)
# ---------------------------------------------------------------------------

def weighted_ensemble_fusion(
    branches: Dict[str, BranchData],
    weights: Dict[str, float],
    calibrate: bool = True,
) -> np.ndarray:
    """Z-score normalize each branch then compute weighted average.

    Parameters
    ----------
    branches : dict of branch_name -> BranchData
    weights : dict of branch_name -> float (will be normalized to sum to 1)
    calibrate : if True, z-score normalize each branch using its own robust stats

    Returns
    -------
    fused_maps : [N, H, W] float64
    """
    if not branches:
        raise ValueError("Need at least one branch")

    first = next(iter(branches.values()))
    n, h, w = first.anomaly_maps.shape

    total_weight = sum(weights.get(bn, 0.0) for bn in branches)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    fused = np.zeros((n, h, w), dtype=np.float64)
    for bname, bdata in branches.items():
        w = weights.get(bname, 0.0) / total_weight
        if w <= 0:
            continue
        maps = bdata.anomaly_maps.astype(np.float64)
        if calibrate:
            center, scale = estimate_robust_stats(maps, bdata.gt_masks)
            maps = compute_z_score(maps, center, scale)
        fused += w * maps

    return fused


def weighted_ensemble_fusion_safe(
    branches: Dict[str, BranchData],
    weights: Dict[str, float],
    calibrate: bool = True,
    anchor_name: str = "anomalydino_visual",
    anchor_baseline_ap: Optional[float] = None,
    anneal_threshold: float = 0.80,
    anneal_factor: float = 0.30,
) -> np.ndarray:
    """weighted_ensemble_fusion with safety annealing.

    When anchor_baseline_ap > anneal_threshold, non-anchor branch weights are
    suppressed by anneal_factor to prevent text noise from degrading an
    already-strong anchor. Uses the original (unmodified) fusion core internally.
    """
    if anchor_baseline_ap is not None and anchor_baseline_ap > anneal_threshold:
        annealed: Dict[str, float] = {}
        for bname, w in weights.items():
            if bname != anchor_name:
                w = w * anneal_factor
            annealed[bname] = w
        return weighted_ensemble_fusion(branches, annealed, calibrate=calibrate)
    return weighted_ensemble_fusion(branches, weights, calibrate=calibrate)


# ---------------------------------------------------------------------------
# Strategy 2: Per-Pixel Max-Z Selection
# ---------------------------------------------------------------------------

def max_z_fusion(
    branches: Dict[str, BranchData],
    mode: str = "max",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Per-pixel branch selection based on z-score confidence.

    Parameters
    ----------
    branches : dict of branch_name -> BranchData
    mode : "max" | "softmax" | "top2_mean"
    temperature : softmax temperature (only used for "softmax" mode)

    Returns
    -------
    fused_maps : [N, H, W] float64
    branch_map : [N, H, W] int32 or None (which branch was selected per pixel)
    """
    if not branches:
        raise ValueError("Need at least one branch")
    if mode not in ("max", "softmax", "top2_mean"):
        raise ValueError(f"Unknown mode: {mode}")

    first = next(iter(branches.values()))
    n, h, w = first.anomaly_maps.shape
    branch_names = list(branches.keys())
    k = len(branch_names)

    # Pre-compute z-score calibration stats per branch
    calib = {}
    for bname, bdata in branches.items():
        center, scale = estimate_robust_stats(
            bdata.anomaly_maps.astype(np.float64), bdata.gt_masks
        )
        calib[bname] = (center, scale)

    # Process sample-by-sample to avoid OOM with [K, N, H, W] stack
    fused = np.zeros((n, h, w), dtype=np.float64)
    branch_map = np.zeros((n, h, w), dtype=np.int32)

    for i in range(n):
        # Build [K, H, W] arrays for this sample
        z_i = np.zeros((k, h, w), dtype=np.float64)
        raw_i = np.zeros((k, h, w), dtype=np.float64)
        for j, bname in enumerate(branch_names):
            bdata = branches[bname]
            raw_i[j] = bdata.anomaly_maps[i].astype(np.float64)
            center, scale = calib[bname]
            z_i[j] = compute_z_score(raw_i[j][np.newaxis], center, scale)[0]

        if mode == "max":
            best = np.argmax(z_i, axis=0)
            fused[i] = np.take_along_axis(raw_i, best[np.newaxis], axis=0)[0]
            branch_map[i] = best

        elif mode == "softmax":
            z_max = z_i.max(axis=0, keepdims=True)
            exp_z = np.exp((z_i - z_max) / max(temperature, 0.01))
            sw = exp_z / exp_z.sum(axis=0, keepdims=True)
            fused[i] = (raw_i * sw).sum(axis=0)
            branch_map[i] = np.argmax(sw, axis=0)

        elif mode == "top2_mean":
            top2_idx = np.argsort(z_i, axis=0)[-2:]
            fused[i] = (raw_i[top2_idx[0]] + raw_i[top2_idx[1]]) / 2.0
            branch_map[i] = top2_idx[1]

    return fused, branch_map


# ---------------------------------------------------------------------------
# Strategy 3: Two-Stage Calibrated Fusion
# ---------------------------------------------------------------------------

def two_stage_calibrated_fusion(
    dino_branch: BranchData,
    adaptclip_fused_branch: BranchData,
    alpha: float = 0.7,
    adaptclip_branches: Optional[Dict[str, BranchData]] = None,
    internal_mode: str = "cached",
) -> np.ndarray:
    """Two-stage fusion: AdaptCLIP internal → calibrate → merge with DINO.

    Stage 1: Fuse AdaptCLIP's 3 internal branches (visual+text+pq adapter)
             or use pre-computed AdaptCLIP fused output.
    Stage 2: Z-score calibrate both DINO and AdaptCLIP, then weighted average.

    Parameters
    ----------
    dino_branch : DINO visual anchor branch
    adaptclip_fused_branch : pre-computed AdaptCLIP fused output (root-level NPZ)
    alpha : weight for DINO (1-alpha for AdaptCLIP)
    adaptclip_branches : optional dict of AdaptCLIP 3 branches for Stage 1 fusion
    internal_mode : "cached" (use pre-fused) | "average" (simple average of 3 branches)

    Returns
    -------
    fused_maps : [N, H, W] float64
    """
    # Stage 1: Get AdaptCLIP unified score
    if internal_mode == "cached":
        ac_maps = adaptclip_fused_branch.anomaly_maps.astype(np.float64)
    elif internal_mode == "average" and adaptclip_branches:
        ac_maps = np.mean(
            [b.anomaly_maps.astype(np.float64) for b in adaptclip_branches.values()],
            axis=0,
        )
    else:
        raise ValueError(f"Invalid internal_mode={internal_mode} or missing branches")

    # Stage 2: Calibrate both branches then weighted average
    dino_maps = dino_branch.anomaly_maps.astype(np.float64)
    masks = dino_branch.gt_masks

    # Calibrate DINO
    d_center, d_scale = estimate_robust_stats(dino_maps, masks)
    dino_z = compute_z_score(dino_maps, d_center, d_scale)
    # Calibrate AdaptCLIP
    a_center, a_scale = estimate_robust_stats(ac_maps, masks)
    ac_z = compute_z_score(ac_maps, a_center, a_scale)

    # Weighted average of z-scores, then convert back to DINO scale
    fused_z = alpha * dino_z + (1.0 - alpha) * ac_z
    # Map back to DINO's original scale for metric computation
    fused = fused_z * d_scale + d_center

    return fused


# ---------------------------------------------------------------------------
# Strategy variants & grid search
# ---------------------------------------------------------------------------

def build_strategy_variants() -> List[dict]:
    """Build all strategy variants to evaluate.

    Returns list of {strategy, variant_name, kwargs} dicts.
    """
    variants = []

    # --- Strategy 1: Weighted Ensemble ---
    # Grid search over DINO weight
    for dino_w in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        variants.append({
            "strategy": "weighted_ensemble",
            "variant_name": f"dino={dino_w:.2f}_anomalyclip={1-dino_w:.2f}",
            "weights": {
                "anomalydino_visual": dino_w,
                "anomalyclip_text": 1.0 - dino_w,
            },
        })

    # Ensemble with AdaptCLIP branches as extra terms
    for dino_w in [0.6, 0.7, 0.8]:
        for ac_extra_w in [0.05, 0.1, 0.15]:
            rem = 1.0 - dino_w - ac_extra_w
            if rem <= 0:
                continue
            variants.append({
                "strategy": "weighted_ensemble",
                "variant_name": f"dino={dino_w:.2f}_anomalyclip={rem:.2f}_adaptclip_vis={ac_extra_w:.2f}",
                "weights": {
                    "anomalydino_visual": dino_w,
                    "anomalyclip_text": rem,
                    "adaptclip_visual_adapter": ac_extra_w,
                },
            })

    # --- Strategy 2: Per-Pixel Max-Z Selection ---
    for mode in ["max"]:  # max only; softmax/top2_mean are too slow for 448x448
        variants.append({
            "strategy": "max_z_selection",
            "variant_name": f"mode={mode}",
            "mode": mode,
            "temperature": 1.0,
        })

    # --- Strategy 3: Two-Stage Calibrated ---
    for alpha in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        variants.append({
            "strategy": "two_stage_calibrated",
            "variant_name": f"alpha={alpha:.2f}",
            "alpha": alpha,
            "internal_mode": "cached",
        })

    return variants


# ---------------------------------------------------------------------------
# Top-level fusion dispatcher
# ---------------------------------------------------------------------------

def run_fusion(
    variant: dict,
    branches: Dict[str, BranchData],
) -> np.ndarray:
    """Run a single fusion variant on all available branches.

    Parameters
    ----------
    variant : dict from build_strategy_variants()
    branches : dict of branch_name -> BranchData (all loaded & aligned)

    Returns
    -------
    fused_maps : [N, H, W] float64
    """
    strategy = variant["strategy"]

    if strategy == "weighted_ensemble":
        weights = variant["weights"]
        return weighted_ensemble_fusion(branches, weights, calibrate=True)

    elif strategy == "max_z_selection":
        # Use all branches for max-z selection
        return max_z_fusion(
            branches,
            mode=variant["mode"],
            temperature=variant.get("temperature", 1.0),
        )[0]  # discard branch_map for metrics

    elif strategy == "two_stage_calibrated":
        dino = branches.get("anomalydino_visual")
        ac_fused = branches.get("adaptclip_fused")
        if dino is None or ac_fused is None:
            raise ValueError("two_stage_calibrated needs anomalydino_visual and adaptclip_fused")
        # Gather adaptclip 3 branches for optional stage-1 internal fusion
        ac_branches = {
            k: v for k, v in branches.items()
            if k.startswith("adaptclip_") and k != "adaptclip_fused"
        }
        return two_stage_calibrated_fusion(
            dino_branch=dino,
            adaptclip_fused_branch=ac_fused,
            alpha=variant["alpha"],
            adaptclip_branches=ac_branches if ac_branches else None,
            internal_mode=variant.get("internal_mode", "cached"),
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
