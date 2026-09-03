"""Label-free proposal construction and evaluator-only labelling for TCRR R0."""

from __future__ import annotations

import cv2
import numpy as np


def robust01(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Per-map robust normalization; uses only the current score map, never GT."""
    a = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(a, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def component_masks(score01: np.ndarray, quantile: float, min_cells: int = 4) -> list[np.ndarray]:
    """Return 8-connected high-score components; deterministic and label-free."""
    score = np.asarray(score01, dtype=np.float32)
    if score.ndim != 2 or not (0.0 < quantile < 1.0):
        raise ValueError("score must be 2-D and quantile in (0,1)")
    if float(score.max() - score.min()) <= 1e-12:
        return []
    threshold = float(np.quantile(score, quantile))
    # Strict comparison prevents a large zero-valued plateau from becoming one
    # image-sized proposal when the requested quantile itself is zero.
    binary = (score > threshold).astype(np.uint8)
    if not binary.any():
        binary = (score == float(score.max())).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    out = []
    for lab in range(1, n):
        mask = labels == lab
        if int(mask.sum()) >= int(min_cells):
            out.append(mask)
    return out


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(np.floor(len(v) * trim_fraction))
    if cut > 0 and 2 * cut < len(v):
        v = v[cut:-cut]
    return float(v.mean())


def component_features(
    mask: np.ndarray,
    a1_score01: np.ndarray,
    text_score01: np.ndarray,
    *,
    trim_fraction: float = 0.1,
    consistency_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute pre-registered scores for one frozen component; no GT argument exists."""
    m = np.asarray(mask, dtype=bool)
    a = np.asarray(a1_score01, dtype=np.float32)[m]
    t = np.asarray(text_score01, dtype=np.float32)[m]
    if a.size == 0 or t.size != a.size:
        raise ValueError("empty/mismatched component")
    return {
        "area_cells": int(a.size),
        "a1_mean": float(a.mean()),
        "a1_max": float(a.max()),
        "a1_p90": float(np.quantile(a, 0.9)),
        "text_trimmed_mean": _trimmed_mean(t, trim_fraction),
        "text_p90": float(np.quantile(t, 0.9)),
        "text_consistency": float(np.mean(t >= consistency_threshold)),
    }


def proposal_label(mask: np.ndarray, gt_mask: np.ndarray, min_overlap_fraction: float = 0.05) -> dict[str, float]:
    """Evaluator-only label attached after proposal/score construction is frozen."""
    m = np.asarray(mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if m.shape != gt.shape or not m.any():
        raise ValueError("invalid proposal/GT geometry")
    intersection = int(np.logical_and(m, gt).sum())
    union = int(np.logical_or(m, gt).sum())
    overlap_fraction = intersection / int(m.sum())
    return {
        "label": int(overlap_fraction >= min_overlap_fraction),
        "intersection_cells": intersection,
        "overlap_fraction": float(overlap_fraction),
        "iou": float(intersection / union) if union else 0.0,
    }


def region_rerank_map(
    raw_a1: np.ndarray,
    proposal_a1_01: np.ndarray,
    text_01: np.ndarray,
    *,
    quantile: float = 0.95,
    min_cells: int = 4,
    max_factor: float = 1.5,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Boundedly re-rank disjoint A1 components using region-level text P90.

    This is the deployable, label-free TCRR operation.  The text score changes
    only the rank of an A1-proposed component; it cannot create a new region.
    """
    raw = np.asarray(raw_a1, dtype=np.float32)
    prop = np.asarray(proposal_a1_01, dtype=np.float32)
    txt = np.asarray(text_01, dtype=np.float32)
    if raw.shape != prop.shape or raw.shape != txt.shape or raw.ndim != 2:
        raise ValueError("raw/proposal/text maps must share one 2-D shape")
    if max_factor <= 1.0:
        raise ValueError("max_factor must exceed 1")
    out = raw.copy()
    audit = []
    log_max = float(np.log(max_factor))
    for mask in component_masks(prop, quantile, min_cells):
        text_p90 = float(np.quantile(txt[mask], 0.9))
        factor = float(np.exp(log_max * (2.0 * text_p90 - 1.0)))
        out[mask] *= factor
        audit.append({"area_cells": int(mask.sum()), "text_p90": text_p90, "factor": factor})
    return out, audit


def normal_calibrated_region_boost_map(
    raw_a1: np.ndarray,
    proposal_a1_01: np.ndarray,
    raw_text: np.ndarray,
    reference_text_maps: np.ndarray,
    *,
    quantile: float = 0.95,
    min_cells: int = 4,
    z_start: float = 3.0,
    z_full: float = 6.0,
    max_factor: float = 1.5,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
    """Normal-calibrated, boost-only TCRR with an exact identity fallback."""
    raw = np.asarray(raw_a1, dtype=np.float32)
    prop = np.asarray(proposal_a1_01, dtype=np.float32)
    txt = np.asarray(raw_text, dtype=np.float32)
    refs = np.asarray(reference_text_maps, dtype=np.float32)
    if raw.shape != prop.shape or raw.shape != txt.shape or raw.ndim != 2:
        raise ValueError("raw/proposal/text maps must share one 2-D shape")
    if refs.ndim != 3 or refs.shape[1:] != raw.shape or not np.isfinite(refs).all():
        raise ValueError("reference maps must be finite [K,H,W]")
    if not (0 < z_start < z_full) or max_factor <= 1:
        raise ValueError("invalid calibration bounds")
    center = float(np.median(refs))
    mad = float(np.median(np.abs(refs - center)))
    scale = max(1.4826 * mad, 1e-6)
    zmap = (txt - center) / scale
    out = raw.copy()
    audit = []
    for mask in component_masks(prop, quantile, min_cells):
        z_p90 = float(np.quantile(zmap[mask], 0.9))
        strength = float(np.clip((z_p90 - z_start) / (z_full - z_start), 0.0, 1.0))
        factor = float(np.exp(np.log(max_factor) * strength))
        out[mask] *= factor
        audit.append({"area_cells": int(mask.sum()), "text_z_p90": z_p90,
                      "strength": strength, "factor": factor})
    return out, audit, {"center": center, "scale": scale, "reference_pixels": int(refs.size)}
