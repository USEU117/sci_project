"""V3.2 Region Proposal: generate candidate anomalous regions from multi-branch evidence.

Instead of pixel-level comparison (V3.1), V3.2 first generates a small set of
candidate regions from text, PQ, and visual branch anomaly maps, then evaluates
each region with multi-evidence reliability checks.
"""

from __future__ import annotations

import numpy as np
from skimage import measure

from .v3_2_contracts import CandidateRegion, V3_2Config


def _connected_regions(
    anomaly_map: np.ndarray,
    threshold: float,
    min_area: int,
    max_area_frac: float,
    min_compactness: float,
    branch_name: str,
) -> list[dict]:
    """Extract connected regions above threshold from a single anomaly map."""
    binary = anomaly_map > threshold
    if not np.any(binary):
        return []

    components = measure.label(binary, connectivity=2)
    total_pixels = int(np.prod(anomaly_map.shape))
    max_area = max(int(total_pixels * max_area_frac), min_area)
    regions = []
    for region_id in range(1, int(components.max()) + 1):
        region_mask = components == region_id
        area = int(np.sum(region_mask))
        if area < min_area or area > max_area:
            continue

        rows, cols = np.where(region_mask)
        if len(rows) < 2:
            continue

        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        bbox_area = max((r_max - r_min + 1) * (c_max - c_min + 1), 1)
        compactness = area / bbox_area
        if compactness < min_compactness:
            continue

        center_r = float(np.mean(rows))
        center_c = float(np.mean(cols))
        region_values = anomaly_map[region_mask]

        regions.append({
            "mask": region_mask,
            "center_yx": (center_r, center_c),
            "area": area,
            "compactness": compactness,
            "peak_score": float(np.max(region_values)),
            "mean_score": float(np.mean(region_values)),
            "branch": branch_name,
        })

    return regions


def _merge_overlapping_regions(
    regions: list[dict], iou_threshold: float = 0.3
) -> list[dict]:
    """Merge regions from different branches that overlap significantly."""
    if len(regions) <= 1:
        return regions

    masks = [r["mask"].astype(np.float32) for r in regions]
    kept = list(range(len(regions)))
    merged = []

    while kept:
        current = kept.pop(0)
        current_mask = masks[current].astype(bool)
        current_area = regions[current]["area"]
        branches = [regions[current]["branch"]]

        to_merge = []
        for j in kept[:]:
            j_mask = masks[j].astype(bool)
            intersection = np.sum(current_mask & j_mask)
            union = np.sum(current_mask | j_mask)
            iou = intersection / max(union, 1)
            if iou > iou_threshold:
                to_merge.append(j)
                current_mask = current_mask | j_mask
                branches.append(regions[j]["branch"])
                current_area = int(np.sum(current_mask))

        for j in sorted(to_merge, reverse=True):
            kept.remove(j)

        rows, cols = np.where(current_mask)
        merged.append({
            "mask": current_mask,
            "center_yx": (float(np.mean(rows)), float(np.mean(cols))),
            "area": current_area,
            "compactness": regions[current]["compactness"],
            "peak_score": regions[current]["peak_score"],
            "mean_score": regions[current]["mean_score"],
            "branch": ",".join(sorted(set(branches))),
            "num_branches": len(set(branches)),
        })

    return merged


def propose_candidate_regions(
    text_anomaly_map: np.ndarray,
    pq_anomaly_map: np.ndarray | None,
    visual_anomaly_map: np.ndarray,
    config: V3_2Config,
    text_threshold: float | None = None,
    pq_threshold: float | None = None,
    visual_threshold: float | None = None,
) -> list[CandidateRegion]:
    """Generate candidate anomalous regions from multi-branch anomaly maps.

    Parameters
    ----------
    text_anomaly_map : [H, W] text adapter anomaly evidence.
    pq_anomaly_map : [H, W] PQ adapter anomaly evidence, or None.
    visual_anomaly_map : [H, W] AnomalyDINO visual anomaly evidence.
    config : V3_2 routing configuration.
    text_threshold : override for text branch threshold.
    pq_threshold : override for PQ branch threshold.
    visual_threshold : override for visual branch threshold.

    Returns
    -------
    List of CandidateRegion objects, sorted by peak score descending.
    """
    if text_threshold is None:
        text_threshold = config.text_excess_threshold
    if pq_threshold is None:
        pq_threshold = config.pq_excess_threshold
    if visual_threshold is None:
        visual_threshold = (config.visual_ambiguous_low + config.visual_ambiguous_high) / 2

    all_regions: list[dict] = []

    # Text branch regions
    text_regions = _connected_regions(
        text_anomaly_map, text_threshold,
        config.min_region_area, config.max_region_area_fraction,
        config.min_region_compactness, "text"
    )
    all_regions.extend(text_regions)

    # PQ branch regions
    if pq_anomaly_map is not None and np.isfinite(pq_anomaly_map).all():
        pq_regions = _connected_regions(
            pq_anomaly_map, pq_threshold,
            config.min_region_area, config.max_region_area_fraction,
            config.min_region_compactness, "pq"
        )
        all_regions.extend(pq_regions)

    # Visual weak-but-above-normal regions (potential rescue opportunities)
    visual_weak = np.clip(visual_anomaly_map, config.visual_ambiguous_low, visual_threshold)
    visual_regions = _connected_regions(
        visual_weak, config.visual_ambiguous_low,
        config.min_region_area, config.max_region_area_fraction,
        config.min_region_compactness, "visual"
    )
    all_regions.extend(visual_regions)

    if not all_regions:
        return []

    merged = _merge_overlapping_regions(all_regions)
    merged.sort(key=lambda r: (r.get("num_branches", 1), r["peak_score"]), reverse=True)

    candidates = []
    for r in merged:
        branches = r.get("branch", "")
        reg = CandidateRegion(
            mask=r["mask"],
            center_yx=r["center_yx"],
            area=r["area"],
            compactness=r["compactness"],
            text_score_max=float(text_anomaly_map[r["mask"]].max()),
            pq_score_max=(
                float(pq_anomaly_map[r["mask"]].max())
                if pq_anomaly_map is not None and np.isfinite(pq_anomaly_map).all()
                else 0.0
            ),
            visual_score_mean=float(visual_anomaly_map[r["mask"]].mean()),
            visual_score_max=float(visual_anomaly_map[r["mask"]].max()),
            source_branches=branches.split(",") if branches else [r.get("branch", "unknown")],
        )
        candidates.append(reg)

    return candidates
