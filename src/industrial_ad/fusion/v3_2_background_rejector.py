"""V3.2 Background Rejector: identify and reject regions that are likely normal background.

V3.1's main failure mode was selecting normal background regions for text rescue.
This module provides checks to identify and reject such regions before they
enter the reliability pipeline.
"""

from __future__ import annotations

import numpy as np

from .v3_2_contracts import CandidateRegion, V3_2Config


def is_edge_adjacent(region_mask: np.ndarray, margin: int = 4) -> bool:
    """Check if region touches or is near image edge."""
    h, w = region_mask.shape
    rows, cols = np.where(region_mask)
    if len(rows) == 0:
        return False
    return (
        rows.min() < margin
        or rows.max() >= h - margin
        or cols.min() < margin
        or cols.max() >= w - margin
    )


def is_large_background(
    region_mask: np.ndarray, max_area_fraction: float = 0.10
) -> bool:
    """Check if region covers an abnormally large area."""
    area = int(np.sum(region_mask))
    total = int(np.prod(region_mask.shape))
    return area > total * max_area_fraction


def is_scattered_response(
    anomaly_map: np.ndarray, region_mask: np.ndarray, peak_ratio_threshold: float = 0.5
) -> bool:
    """Check if response within region is scattered (no clear concentration).

    A scattered region has peak values not much higher than the mean, indicating
    diffuse background drift rather than a focused anomaly.
    """
    values = anomaly_map[region_mask]
    if len(values) < 4:
        return False
    peak = float(np.max(values))
    if peak <= 0:
        return True
    mean = float(np.mean(values))
    # If peak/mean ratio is low, the response is scattered
    return (peak / max(mean, 1e-8)) < peak_ratio_threshold


def is_boundary_fragmented(
    region_mask: np.ndarray, max_fragmentation: float = 3.0
) -> bool:
    """Check if region boundary is excessively fragmented.

    A perfect circle gives fragmentation ≈ 1.0. Irregular shapes give higher
    values. Default threshold 3.0 allows moderately irregular defect shapes
    while rejecting highly fragmented noise.
    """
    from skimage import measure

    if int(np.sum(region_mask)) < 16:
        return True

    contours = measure.find_contours(region_mask.astype(np.float64), 0.5)
    if not contours:
        return True

    total_perimeter = sum(len(c) for c in contours)
    area = int(np.sum(region_mask))
    expected_circle_perimeter = 2 * np.sqrt(np.pi * max(area, 1))

    if expected_circle_perimeter <= 0:
        return True

    fragmentation = total_perimeter / expected_circle_perimeter
    return fragmentation > max_fragmentation


def check_branch_isolation(
    region: CandidateRegion,
    text_map: np.ndarray,
    pq_map: np.ndarray | None,
    visual_map: np.ndarray,
    text_threshold: float = 1.0,
    pq_threshold: float = 0.8,
) -> dict:
    """Check if a region is supported by only one isolated branch.

    Returns dict with:
      - text_support : bool
      - pq_support : bool
      - visual_support : bool
      - isolated : True only supported by text without PQ or visual backing
    """
    mask = region.mask
    text_support = float(text_map[mask].max()) > text_threshold
    pq_support = (
        float(pq_map[mask].max()) > pq_threshold
        if pq_map is not None and np.isfinite(pq_map).all()
        else False
    )
    visual_support = float(visual_map[mask].max()) > 0.5
    isolated = text_support and not pq_support and not visual_support
    return {
        "text_support": text_support,
        "pq_support": pq_support,
        "visual_support": visual_support,
        "isolated": isolated,
    }


def reject_background_regions(
    candidates: list[CandidateRegion],
    text_map: np.ndarray,
    pq_map: np.ndarray | None,
    visual_map: np.ndarray,
    config: V3_2Config,
) -> tuple[list[CandidateRegion], dict]:
    """Filter candidate regions, removing likely background false positives.

    Returns (kept_regions, rejection_stats).
    """
    kept: list[CandidateRegion] = []
    rejected_edge = 0
    rejected_large = 0
    rejected_scattered = 0
    rejected_fragmented = 0
    rejected_isolated = 0

    for region in candidates:
        mask = region.mask

        if is_edge_adjacent(mask):
            rejected_edge += 1
            continue

        if is_large_background(mask):
            rejected_large += 1
            continue

        if is_scattered_response(text_map, mask):
            rejected_scattered += 1
            continue

        if is_boundary_fragmented(mask):
            rejected_fragmented += 1
            continue

        iso = check_branch_isolation(region, text_map, pq_map, visual_map)
        if iso["isolated"]:
            rejected_isolated += 1
            continue

        kept.append(region)

    stats = {
        "total_candidates": len(candidates),
        "kept": len(kept),
        "rejected_edge": rejected_edge,
        "rejected_large": rejected_large,
        "rejected_scattered": rejected_scattered,
        "rejected_fragmented": rejected_fragmented,
        "rejected_isolated": rejected_isolated,
    }
    return kept, stats
