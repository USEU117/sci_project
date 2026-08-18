"""V3.3 visual-anchored text local rescue (docs 阶段四).

Fixed flow (all CPU, leakage-safe):
    candidate region -> reference out-of-support -> prompt/aug stability
    -> background rejection -> bounded one-directional text residual
    -> visual fallback.

Protocol guarantees:
- Candidate regions come ONLY from the visual anomaly map (reference-derived).
- "Reference out-of-support" uses ONLY the K normal-reference statistics.
- Text may add a BOUNDED, ONE-DIRECTIONAL residual inside visual candidates
  only; it never rewrites the visual score across the whole map.
- Any unreliable condition falls back to the visual anchor (per-pixel).
- Forbidden: test labels, test masks, test-set quantiles, category rules
  selected by test metrics.

Calibration uses POOLED normal-reference statistics (median / IQR / q95), the
same distributional view as v3_3_clean. Per-pixel reference stats are far too
noisy with only K small views and destroyed the ranking (measured, phase-4).

Region reason codes (saved per pixel):
    no_visual_candidate / reference_in_support / prompt_unstable /
    background_rejected / bounded_text_residual / visual_fallback

RouterInput / EvaluationTarget reuse the clean data-boundary from v3_3_clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from industrial_ad.fusion.v3_3_clean import (
    DEFAULT_ANCHOR,
    EvaluationTarget,
    RouterInput,
    compute_z_score,
    estimate_reference_stats,
    evaluate_clean,
    sanitize_finite,
)

EPS: float = 1e-8

# --- region reason codes ---------------------------------------------------
NO_VISUAL_CANDIDATE = 0
REFERENCE_IN_SUPPORT = 1
PROMPT_UNSTABLE = 2
BACKGROUND_REJECTED = 3
BOUNDED_TEXT_RESIDUAL = 4
VISUAL_FALLBACK = 5
REASON_NAMES = {
    NO_VISUAL_CANDIDATE: "no_visual_candidate",
    REFERENCE_IN_SUPPORT: "reference_in_support",
    PROMPT_UNSTABLE: "prompt_unstable",
    BACKGROUND_REJECTED: "background_rejected",
    BOUNDED_TEXT_RESIDUAL: "bounded_text_residual",
    VISUAL_FALLBACK: "visual_fallback",
}


@dataclass(frozen=True)
class LocalRescueConfig:
    """Pre-registered (fixed) hyper-parameters for the rescue router."""

    visual_candidate_quantile: float = 0.95   # visual z > q95 of normal-reference visual z
    text_support_quantile: float = 0.95       # text must exceed q95 of normal-reference text z
    prompt_stability_cv: float = 2.0          # max relative view spread across reference views
    prompt_stability_min_views: int = 2       # need at least this many reference views
    background_reject_margin: int = 4         # border margin (px) never rescued
    background_reject_min_visual_std: float = 1e-3  # flat visual candidates rejected
    residual_cap: float = 2.0                 # max text residual in z-units (bounded)
    fill: float = 0.0                         # sanitize fill for non-finite text


def _pooled_z(maps: np.ndarray, stats: dict) -> np.ndarray:
    return compute_z_score(maps, stats["center"], stats["scale"])


def _pooled_quantile_z(stats: dict, quantile_value: float) -> float:
    """Reference quantile value expressed in z-units: (q_value - center) / scale."""
    return float((quantile_value - stats["center"]) / max(stats["scale"], EPS))


def prompt_stability(ref_maps: np.ndarray, min_views: int = 2, max_cv: float = 2.0) -> bool:
    """Text prompt/augmentation stability from deterministic reference views.

    Uses the pooled normal-reference distribution: each view is z-scored with
    the pooled median/IQR, then the per-view mean magnitude is compared across
    views. Views that disagree strongly (large relative spread) are unstable ->
    visual fallback (reason `prompt_unstable`).
    """
    vals = np.asarray(ref_maps, dtype=np.float64)
    if vals.shape[0] < min_views:
        return False
    stats = estimate_reference_stats(vals)
    z_views = compute_z_score(vals, stats["center"], stats["scale"])
    per_view_mag = np.abs(z_views).mean(axis=(1, 2))
    mean = float(per_view_mag.mean())
    spread = float(per_view_mag.max() - per_view_mag.min())
    cv = spread / max(mean, EPS)
    return bool(np.isfinite(cv) and cv <= max_cv)


def visual_candidate_mask(
    visual_z: np.ndarray, support_z: float, margin: int = 4
) -> np.ndarray:
    """Pixels where the visual branch is beyond its normal-reference support.

    Border margin pixels are always rejected (background/boundary artifacts).
    `visual_z` may be [H,W] or [N,H,W].
    """
    cand = visual_z > support_z
    if margin > 0:
        cand[..., :margin, :] = False
        cand[..., -margin:, :] = False
        cand[..., :, :margin] = False
        cand[..., :, -margin:] = False
    return cand


def background_reject_mask(
    visual_maps: np.ndarray,
    candidate: np.ndarray,
    min_visual_std: float = 1e-3,
) -> np.ndarray:
    """Reject low-texture (flat) candidate pixels using local visual contrast."""
    maps = np.asarray(visual_maps, dtype=np.float64)
    # per-image flatness proxy + per-pixel deviation from the image mean
    std_local = np.std(maps, axis=(-2, -1), keepdims=True)
    local_dev = np.abs(maps - maps.mean(axis=(-2, -1), keepdims=True))
    keep = candidate & (local_dev >= min_visual_std) & (std_local >= min_visual_std)
    return keep


def local_rescue_fusion(
    ri: RouterInput,
    config: LocalRescueConfig = LocalRescueConfig(),
    anchor: str = DEFAULT_ANCHOR,
    text_branch: str = "anomalyclip_text",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Run the visual-anchored text local rescue on one category.

    Returns (fused_maps [N,H,W] float64, diagnostics dict). The fused map is
    the visual z-score everywhere plus a bounded text residual inside accepted
    visual candidate pixels.
    """
    if text_branch not in ri.branches:
        raise ValueError(f"RouterInput lacks text branch {text_branch}")
    if anchor not in ri.branches:
        raise ValueError(f"RouterInput lacks anchor branch {anchor}")

    visual_maps = np.asarray(ri.branches[anchor], dtype=np.float64)
    text_maps = np.asarray(ri.branches[text_branch], dtype=np.float64)
    ref_visual = np.asarray(ri.reference_maps[anchor], dtype=np.float64)
    ref_text = np.asarray(ri.reference_maps[text_branch], dtype=np.float64)
    n, h, w = visual_maps.shape

    # 1) candidate regions from visual only (pooled reference stats)
    vis_stats = estimate_reference_stats(ref_visual)
    visual_z = _pooled_z(visual_maps, vis_stats)
    vis_support_z = _pooled_quantile_z(vis_stats, vis_stats["q" + str(int(config.visual_candidate_quantile * 100))])
    cand = visual_candidate_mask(visual_z, vis_support_z, margin=config.background_reject_margin)

    # 2) background rejection (flat / low-texture candidates)
    cand = background_reject_mask(
        visual_maps, cand, min_visual_std=config.background_reject_min_visual_std
    )

    # 3) prompt/aug stability from reference views
    text_stable = prompt_stability(
        ref_text,
        min_views=config.prompt_stability_min_views,
        max_cv=config.prompt_stability_cv,
    )

    # 4) text z-scores (reference-only stats) + bounded one-directional residual
    text_stats = estimate_reference_stats(ref_text)
    text_z = _pooled_z(sanitize_finite(text_maps, fill=config.fill), text_stats)
    text_support_z = _pooled_quantile_z(text_stats, text_stats["q" + str(int(config.text_support_quantile * 100))])
    text_residual = np.maximum(text_z - text_support_z, 0.0)
    text_residual = np.minimum(text_residual, config.residual_cap)

    reason = np.full((n, h, w), NO_VISUAL_CANDIDATE, dtype=np.uint8)
    fused = visual_z.copy()
    accepted = cand & (text_residual > 0.0)
    if not text_stable:
        # text prompt unstable -> no text anywhere (per-pixel visual fallback)
        reason[cand] = PROMPT_UNSTABLE
    else:
        reason[cand & ~accepted] = REFERENCE_IN_SUPPORT
        reason[accepted] = BOUNDED_TEXT_RESIDUAL
        fused[accepted] = visual_z[accepted] + text_residual[accepted]

    # safety: any non-finite fused pixel falls back to raw visual z
    bad = ~np.isfinite(fused)
    if bad.any():
        fused[bad] = visual_z[bad]
        reason[bad] = VISUAL_FALLBACK

    counts = {REASON_NAMES[k]: int(np.sum(reason == k)) for k in REASON_NAMES}
    diagnostics = {
        "text_stable": bool(text_stable),
        "candidate_pixels": int(np.sum(cand)),
        "accepted_pixels": int(np.sum(accepted)),
        "max_text_residual": float(np.max(text_residual)),
        "visual_support_z": vis_support_z,
        "text_support_z": text_support_z,
        "reason_counts": counts,
        "residual_cap": config.residual_cap,
        "visual_candidate_quantile": config.visual_candidate_quantile,
        "text_support_quantile": config.text_support_quantile,
    }
    return fused, diagnostics


def evaluate_rescue(
    ri: RouterInput,
    target: EvaluationTarget,
    fused_maps: np.ndarray,
    stride: int = 8,
) -> dict:
    """Evaluate rescue output. Only evaluator code reads the target."""
    return evaluate_clean(ri, target, fused_maps, stride=stride)
