"""V3.2 Multi-Evidence Reliability Scoring.

For each candidate region, computes five independent reliability signals:
1. Normal range excess -- how far beyond normal reference distribution
2. Spatial stability -- region coherence and internal concentration
3. Augmentation consistency -- stability under semantic-preserving transforms
4. Prompt consistency -- agreement across fixed prompt templates
5. Branch consistency -- text/PQ/visual agreement on the same region

All signals are label-free and use only frozen normal references.
"""

from __future__ import annotations

import numpy as np

from .v3_2_contracts import CandidateRegion, ReliabilityEvidenceV3_2


def score_normal_range_excess(
    region: CandidateRegion,
    text_map: np.ndarray,
    normal_reference_stats: dict[str, float],
) -> float:
    """How far does the text response exceed the normal reference range?

    Parameters
    ----------
    normal_reference_stats : dict with keys 'pixel_center', 'pixel_scale'
        from the grouped reference calibration.

    Returns
    -------
    Score in [0, 1]: 0 = within normal range, 1 = far beyond.
    """
    center = float(normal_reference_stats.get("pixel_center", 0.0))
    scale = max(float(normal_reference_stats.get("pixel_scale", 1.0)), 1e-8)
    region_values = text_map[region.mask]
    if len(region_values) == 0:
        return 0.0
    z_scores = (region_values - center) / scale
    excess_fraction = float(np.mean(z_scores > 1.0))
    magnitude = float(np.clip(np.arcsinh(float(np.median(np.maximum(z_scores, 0)))) / 3.0, 0.0, 1.0))
    return float(np.clip(0.5 * excess_fraction + 0.5 * magnitude, 0.0, 1.0))


def score_spatial_stability(region: CandidateRegion, text_map: np.ndarray) -> float:
    """Score region spatial coherence: concentrated response, smooth boundary.

    Returns
    -------
    Score in [0, 1]: higher = more stable, defect-like structure.
    """
    mask = region.mask
    values = text_map[mask]
    if len(values) < 4:
        return 0.0

    peak = float(np.max(values))
    if peak <= 0:
        return 0.0

    # Internal concentration: peak vs mean ratio
    mean = float(np.mean(values))
    concentration = float(np.clip((peak / max(mean, 1e-8) - 1.0) / 3.0, 0.0, 1.0))

    # Compactness already in region
    compactness_weight = float(np.clip(region.compactness / 0.5, 0.0, 1.0))

    # Surrounding contrast: region mean vs dilated border
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask, iterations=3)
    border_region = dilated & ~mask
    if np.any(border_region):
        exterior_mean = float(np.mean(text_map[border_region]))
        if exterior_mean > 0:
            contrast = float(np.clip(peak / exterior_mean, 0.0, 5.0) / 5.0)
        else:
            contrast = 1.0
    else:
        contrast = 0.5

    return float(np.clip(0.35 * concentration + 0.35 * compactness_weight + 0.30 * contrast, 0.0, 1.0))


def score_augmentation_consistency(
    region: CandidateRegion,
    text_map: np.ndarray,
    augmented_maps: dict[str, np.ndarray],
) -> float:
    """Score stability under semantic-preserving augmentations.

    Parameters
    ----------
    augmented_maps : dict mapping augmentation name to text anomaly map
        (e.g., {'flip_h': ..., 'brightness_0.95': ...}).

    Returns
    -------
    Score in [0, 1]: 1 = highly consistent across augmentations.
    """
    if not augmented_maps:
        return 0.5  # neutral: no augmentation data

    region_mean = float(np.mean(text_map[region.mask]))
    if region_mean <= 0:
        return 0.0

    consistencies = []
    for name, aug_map in augmented_maps.items():
        if aug_map.shape != text_map.shape:
            continue
        aug_mean = float(np.mean(aug_map[region.mask]))
        if region_mean > 0:
            ratio = min(region_mean, aug_mean) / max(region_mean, aug_mean, 1e-8)
            consistencies.append(ratio)

    if not consistencies:
        return 0.5

    return float(np.clip(np.mean(consistencies), 0.0, 1.0))


def score_prompt_consistency(
    region: CandidateRegion,
    text_map: np.ndarray,
    prompt_maps: dict[str, np.ndarray],
) -> float:
    """Score agreement across different prompt templates.

    Parameters
    ----------
    prompt_maps : dict mapping prompt name to text anomaly map
        (e.g., {'normal/anomalous': ..., 'defect-free/defective': ...}).

    Returns
    -------
    Score in [0, 1]: 1 = all prompts agree on region anomaly.
    """
    if len(prompt_maps) < 2:
        return 0.5  # neutral

    region_means = {}
    for name, pmap in prompt_maps.items():
        if pmap.shape != text_map.shape:
            continue
        region_means[name] = float(np.mean(pmap[region.mask]))

    if len(region_means) < 2:
        return 0.5

    values = list(region_means.values())
    mean_val = np.mean(values)
    if mean_val <= 0:
        return 0.0

    # Coefficient of variation: lower = more consistent
    cv = float(np.std(values) / max(mean_val, 1e-8))
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def score_branch_consistency(
    region: CandidateRegion,
    text_map: np.ndarray,
    pq_map: np.ndarray | None,
    visual_map: np.ndarray,
) -> float:
    """Score agreement between text, PQ, and visual branches on this region.

    Returns
    -------
    Score in [0, 1]: 1 = all branches agree region is anomalous.
    """
    text_signal = float(np.mean(text_map[region.mask])) > 0.5
    visual_signal = float(np.mean(visual_map[region.mask])) > 0.5
    if pq_map is not None and np.isfinite(pq_map).all():
        pq_signal = float(np.mean(pq_map[region.mask])) > 0.5
    else:
        pq_signal = None

    # Best: text + PQ agree, visual has weak but present signal
    if text_signal and pq_signal is True:
        return 0.9
    elif text_signal and pq_signal is None:
        return 0.6
    elif text_signal and not pq_signal:
        return 0.3
    elif not text_signal:
        return 0.0
    return 0.5


def score_background_risk(
    region: CandidateRegion,
    text_map: np.ndarray,
    normal_reference_maps: list[np.ndarray] | None = None,
) -> float:
    """Estimate probability that this region is normal background.

    High score = high risk of being background (bad).

    Checks: edge proximity, response pattern similarity to normal ref backgrounds.
    """
    risk = 0.0
    mask = region.mask
    h, w = mask.shape

    # Edge proximity penalty
    rows, cols = np.where(mask)
    if len(rows) > 0:
        edge_dist = min(
            rows.min(), h - 1 - rows.max(),
            cols.min(), w - 1 - cols.max()
        )
        edge_proximity = float(np.clip(1.0 - edge_dist / max(h, w, 1) * 10, 0.0, 1.0))
        risk += edge_proximity * 0.3

    # Large area penalty
    area_frac = region.area / max(int(np.prod(mask.shape)), 1)
    large_area_risk = float(np.clip(area_frac / 0.05, 0.0, 1.0))
    risk += large_area_risk * 0.3

    # Normal reference similarity
    if normal_reference_maps:
        region_values = text_map[mask]
        ref_means = [np.mean(ref) for ref in normal_reference_maps if ref.shape == mask.shape]
        if ref_means:
            ref_avg = float(np.mean(ref_means))
            region_avg = float(np.mean(region_values))
            if ref_avg > 0:
                ratio = region_avg / ref_avg
                ref_similarity = float(np.clip(1.0 - abs(ratio - 1.0), 0.0, 1.0))
            else:
                ref_similarity = 0.0 if region_avg > 0 else 1.0
            risk += ref_similarity * 0.4

    return float(np.clip(risk, 0.0, 1.0))


def compute_reliability_evidence(
    region: CandidateRegion,
    text_map: np.ndarray,
    pq_map: np.ndarray | None,
    visual_map: np.ndarray,
    normal_reference_stats: dict[str, float],
    augmented_maps: dict[str, np.ndarray] | None = None,
    prompt_maps: dict[str, np.ndarray] | None = None,
    normal_reference_maps: list[np.ndarray] | None = None,
) -> ReliabilityEvidenceV3_2:
    """Compute all five reliability evidence signals for a candidate region.

    Returns ReliabilityEvidenceV3_2 with scores in [0, 1].
    """
    return ReliabilityEvidenceV3_2(
        normal_range_excess=score_normal_range_excess(region, text_map, normal_reference_stats),
        spatial_stability=score_spatial_stability(region, text_map),
        augmentation_consistency=score_augmentation_consistency(
            region, text_map, augmented_maps or {}
        ),
        prompt_consistency=score_prompt_consistency(region, text_map, prompt_maps or {}),
        branch_consistency=score_branch_consistency(region, text_map, pq_map, visual_map),
        background_risk=score_background_risk(region, text_map, normal_reference_maps),
    )
