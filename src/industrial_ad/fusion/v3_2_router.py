"""V3.2 Three-Level Hierarchical Selective Rescue Router.

Level 1 -- Opportunity: Are there candidate regions worth checking?
Level 2 -- Reliability: Is the text evidence trustworthy for these regions?
Level 3 -- Modification: How much text rescue to allow (bounded, non-negative)?

V3.2 prioritizes "reject first, rescue rarely" over V3.1's "rescue whenever
text score is higher."
"""

from __future__ import annotations

import numpy as np

from .v3_2_background_rejector import reject_background_regions
from .v3_2_contracts import (
    CandidateRegion,
    RegionRoutingDecision,
    ReliabilityEvidenceV3_2,
    V3_2Config,
    V3_2DecisionLevel,
    V3_2FusionResult,
    V3_2ReasonCode,
)
from .v3_2_region_proposal import propose_candidate_regions
from .v3_2_reliability import compute_reliability_evidence


class HierarchicalSelectiveRescueV3_2:
    """V3.2 three-level hierarchical router for visual-anchored text rescue.

    Level 1: Generate candidate regions and filter backgrounds.
    Level 2: Assess multi-evidence reliability per region.
    Level 3: Apply bounded non-negative residual to visual maps.

    Text assistance is always non-negative and bounded. Image-level scores
    remain unchanged (visual-only) per protocol.
    """

    def __init__(self, config: V3_2Config | None = None) -> None:
        self.config = (config or V3_2Config()).validate()

    def _level_1_opportunity(
        self,
        visual_map: np.ndarray,
        text_map: np.ndarray,
        pq_map: np.ndarray | None,
    ) -> tuple[list[CandidateRegion], dict]:
        """Level 1: Generate candidate regions and apply background rejection."""
        candidates = propose_candidate_regions(
            text_map, pq_map, visual_map, self.config
        )
        kept, bg_stats = reject_background_regions(
            candidates, text_map, pq_map, visual_map, self.config
        )
        return kept, bg_stats

    def _level_2_reliability(
        self,
        candidates: list[CandidateRegion],
        visual_map: np.ndarray,
        text_map: np.ndarray,
        pq_map: np.ndarray | None,
        normal_reference_stats: dict[str, float],
        augmented_maps: dict[str, np.ndarray] | None,
        prompt_maps: dict[str, np.ndarray] | None,
        normal_reference_maps: list[np.ndarray] | None,
    ) -> list[tuple[CandidateRegion, ReliabilityEvidenceV3_2, list[V3_2ReasonCode]]]:
        """Level 2: Compute reliability evidence and filter unacceptable regions.

        Returns list of (region, evidence, rejection_reasons) for regions that
        pass or are rejected.
        """
        results: list[
            tuple[CandidateRegion, ReliabilityEvidenceV3_2, list[V3_2ReasonCode]]
        ] = []

        for region in candidates:
            evidence = compute_reliability_evidence(
                region,
                text_map,
                pq_map,
                visual_map,
                normal_reference_stats,
                augmented_maps,
                prompt_maps,
                normal_reference_maps,
            )
            rejection_reasons: list[V3_2ReasonCode] = []

            # Requires combined reliability above threshold
            combined = evidence.combined_reliability()
            if combined < self.config.min_combined_reliability:
                rejection_reasons.append(V3_2ReasonCode.RELIABILITY_TOO_LOW)

            # Branch consistency check
            if evidence.branch_consistency < self.config.min_branch_consistency:
                rejection_reasons.append(V3_2ReasonCode.PQ_DISAGREEMENT)

            # Background risk too high
            if evidence.background_risk > self.config.max_background_risk:
                rejection_reasons.append(V3_2ReasonCode.BACKGROUND_RISK)

            # Augmentation instability
            if evidence.augmentation_consistency < self.config.min_augmentation_consistency:
                rejection_reasons.append(V3_2ReasonCode.AUGMENTATION_UNSTABLE)

            # Prompt disagreement
            if evidence.prompt_consistency < self.config.min_prompt_consistency:
                rejection_reasons.append(V3_2ReasonCode.PROMPT_DISAGREEMENT)

            # Visual anchor: if visual is confident normal, reject
            if region.visual_score_max < self.config.visual_ambiguous_low:
                rejection_reasons.append(V3_2ReasonCode.VISUAL_CONFIDENT_NORMAL)

            # Visual anchor: if visual is already clearly anomalous, no need for rescue
            if region.visual_score_max > self.config.visual_ambiguous_high:
                rejection_reasons.append(V3_2ReasonCode.VISUAL_CONFIDENT_ANOMALY)

            if not rejection_reasons:
                rejection_reasons = []  # passed

            results.append((region, evidence, rejection_reasons))

        return results

    def _level_3_modification(
        self,
        l2_results: list[
            tuple[CandidateRegion, ReliabilityEvidenceV3_2, list[V3_2ReasonCode]]
        ],
        pixel_gap_map: np.ndarray,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, np.ndarray, list[RegionRoutingDecision]]:
        """Level 3: Apply bounded non-negative residual to allowed regions.

        Returns (residual_map, rescue_mask, routing_decisions).
        """
        residual = np.zeros((h, w), dtype=np.float64)
        rescue_mask = np.zeros((h, w), dtype=bool)
        decisions: list[RegionRoutingDecision] = []

        # Track cumulative rescue area
        total_pixels = h * w
        max_total_rescue = int(total_pixels * self.config.max_rescue_area_fraction)
        cumulative_rescue = 0

        # Sort by reliability (highest first) so most reliable regions rescue first
        scored = sorted(
            l2_results,
            key=lambda x: x[1].combined_reliability() if x[1] else 0,
            reverse=True,
        )

        for region, evidence, rejection_reasons in scored:
            if rejection_reasons:
                # Rejected at L2
                decisions.append(
                    RegionRoutingDecision(
                        region=region,
                        reason=rejection_reasons[0],
                        reliability=evidence,
                        rescue_applied=False,
                        level_reached=V3_2DecisionLevel.L2_RELIABILITY,
                    )
                )
                continue

            # Budget check
            if cumulative_rescue + region.area > max_total_rescue:
                decisions.append(
                    RegionRoutingDecision(
                        region=region,
                        reason=V3_2ReasonCode.RESCUE_BUDGET_CLIPPED,
                        reliability=evidence,
                        rescue_applied=False,
                        max_allowed_residual=self.config.max_pixel_residual,
                        level_reached=V3_2DecisionLevel.L3_MODIFICATION,
                    )
                )
                continue

            # Scale residual by reliability (more reliable → closer to max)
            if self.config.reliability_residual_scale:
                reliability_scale = evidence.combined_reliability()
            else:
                reliability_scale = 1.0

            allowed_residual = self.config.max_pixel_residual * reliability_scale
            gap_values = np.clip(pixel_gap_map[region.mask], 0.0, allowed_residual)
            # V3.2 fix: only rescue pixels where text actually exceeds visual (gap > 0)
            positive_mask = gap_values > 0
            if not np.any(positive_mask):
                decisions.append(
                    RegionRoutingDecision(
                        region=region,
                        reason=V3_2ReasonCode.TEXT_NOT_EXCEEDING,
                        reliability=evidence,
                        rescue_applied=False,
                        max_allowed_residual=allowed_residual,
                        level_reached=V3_2DecisionLevel.L3_MODIFICATION,
                    )
                )
                continue

            region_pixel_indices = np.where(region.mask)
            positive_indices = (
                region_pixel_indices[0][positive_mask],
                region_pixel_indices[1][positive_mask],
            )
            residual[positive_indices] = gap_values[positive_mask]
            rescue_mask[positive_indices] = True
            cumulative_rescue += int(np.sum(positive_mask))

            decisions.append(
                RegionRoutingDecision(
                    region=region,
                    reason=V3_2ReasonCode.RESCUE_APPLIED,
                    reliability=evidence,
                    rescue_applied=True,
                    rescue_residual=float(np.mean(gap_values[positive_mask])),
                    max_allowed_residual=allowed_residual,
                    level_reached=V3_2DecisionLevel.L3_MODIFICATION,
                )
            )

        return residual, rescue_mask, decisions

    def fuse(
        self,
        visual_pixel_maps: np.ndarray,
        text_pixel_maps: np.ndarray,
        visual_image_scores: np.ndarray,
        text_image_scores: np.ndarray,
        sample_ids: np.ndarray,
        normal_reference_stats: dict[str, dict[str, float]] | None = None,
        pq_pixel_maps: np.ndarray | None = None,
        augmented_maps: dict[str, dict[str, np.ndarray]] | None = None,
        prompt_maps: dict[str, dict[str, np.ndarray]] | None = None,
        normal_reference_maps: dict[str, list[np.ndarray]] | None = None,
    ) -> V3_2FusionResult:
        """Fuse visual (anchor) and text (rescue) branch predictions.

        Parameters
        ----------
        visual_pixel_maps : [N, H, W] AnomalyDINO pixel anomaly maps.
        text_pixel_maps : [N, H, W] text adapter pixel anomaly maps.
        visual_image_scores : [N] AnomalyDINO image scores.
        text_image_scores : [N] text adapter image scores.
        sample_ids : [N] sample identifiers.
        normal_reference_stats : per-category calibration stats for normal range.
        pq_pixel_maps : [N, H, W] PQ adapter maps (optional).
        augmented_maps : per-augmentation pixel maps (optional).
        prompt_maps : per-prompt pixel maps (optional).
        normal_reference_maps : per-category normal reference anomaly maps.

        Returns
        -------
        V3_2FusionResult with fused maps, residuals, and full routing audit.
        """
        vis_maps = np.asarray(visual_pixel_maps, dtype=np.float64)
        txt_maps = np.asarray(text_pixel_maps, dtype=np.float64)
        vis_scores = np.asarray(visual_image_scores, dtype=np.float64).reshape(-1)
        txt_scores = np.asarray(text_image_scores, dtype=np.float64).reshape(-1)
        ids = np.asarray(sample_ids)
        n = len(ids)

        if vis_maps.shape != txt_maps.shape:
            raise ValueError("visual and text pixel maps shape mismatch")
        if len(vis_scores) != n or len(txt_scores) != n:
            raise ValueError("image scores length mismatch")

        # V3.2 fix: normalize branch scores to comparable ranges using robust statistics
        # Without this, text_adapter scores (~0.01) are orders of magnitude lower
        # than DINO visual scores (~0.1), making "text exceeds visual" check useless.
        if self.config.normalize_branches:
            vis_median = np.median(vis_maps)
            vis_iqr = np.subtract(*np.percentile(vis_maps, [75, 25])) + 1e-8
            txt_median = np.median(txt_maps)
            txt_iqr = np.subtract(*np.percentile(txt_maps, [75, 25])) + 1e-8
            # Scale text to have same IQR as visual, preserving relative ordering
            txt_maps = (txt_maps - txt_median) * (vis_iqr / txt_iqr) + vis_median
            # Also normalize PQ maps if available
            if pq_pixel_maps is not None:
                pq_maps_raw = np.asarray(pq_pixel_maps, dtype=np.float64)
                pq_median = np.median(pq_maps_raw)
                pq_iqr = np.subtract(*np.percentile(pq_maps_raw, [75, 25])) + 1e-8
                pq_maps_raw = (pq_maps_raw - pq_median) * (vis_iqr / pq_iqr) + vis_median
                pq_pixel_maps = pq_maps_raw

        fused_maps = vis_maps.copy()
        all_residual = np.zeros_like(vis_maps)
        all_rescue_mask = np.zeros_like(vis_maps, dtype=bool)
        all_fallback = np.zeros(n, dtype=bool)
        all_decisions: list[list[RegionRoutingDecision]] = []
        stats_agg = {
            "total_samples": n,
            "samples_with_candidates": 0,
            "samples_with_rescue": 0,
            "total_regions_proposed": 0,
            "total_regions_rescued": 0,
            "total_regions_rejected": 0,
            "total_pixels_rescued": 0,
        }

        use_pq = pq_pixel_maps is not None

        for idx in range(n):
            vmap = vis_maps[idx]
            tmap = txt_maps[idx]
            h, w = vmap.shape

            pq_map = pq_pixel_maps[idx] if use_pq else None

            # Sample-level augmented and prompt maps
            sample_aug = {}
            if augmented_maps and ids[idx] in augmented_maps:
                sample_aug = {
                    k: v[idx] if v.ndim == 3 else v
                    for k, v in augmented_maps.get(ids[idx], {}).items()
                }
            sample_prompt = {}
            if prompt_maps and ids[idx] in prompt_maps:
                sample_prompt = {
                    k: v[idx] if v.ndim == 3 else v
                    for k, v in prompt_maps.get(ids[idx], {}).items()
                }
            sample_normal_refs = (
                normal_reference_maps.get(str(idx), [])
                if normal_reference_maps
                else None
            )

            ref_stats = (
                normal_reference_stats.get(str(idx), {"pixel_center": 0.0, "pixel_scale": 1.0})
                if normal_reference_stats
                else {"pixel_center": 0.0, "pixel_scale": 1.0}
            )

            # Level 1: Region proposal + background rejection
            candidates, bg_stats = self._level_1_opportunity(vmap, tmap, pq_map)
            stats_agg["total_regions_proposed"] += bg_stats["total_candidates"]

            if not candidates:
                all_fallback[idx] = True
                all_decisions.append([])
                continue

            stats_agg["samples_with_candidates"] += 1

            # Level 2: Reliability assessment
            l2_results = self._level_2_reliability(
                candidates, vmap, tmap, pq_map, ref_stats,
                sample_aug, sample_prompt, sample_normal_refs,
            )

            # Level 3: Apply rescue
            pixel_gap = txt_maps[idx] - vis_maps[idx]
            residual_i, rescue_i, sample_decisions = self._level_3_modification(
                l2_results, pixel_gap, h, w
            )

            if np.any(rescue_i):
                fused_maps[idx] += residual_i
                all_residual[idx] = residual_i
                all_rescue_mask[idx] = rescue_i
                stats_agg["samples_with_rescue"] += 1
                stats_agg["total_pixels_rescued"] += int(np.sum(rescue_i))
            else:
                all_fallback[idx] = True

            # Count decisions
            rescued_count = sum(1 for d in sample_decisions if d.rescue_applied)
            rejected_count = sum(1 for d in sample_decisions if not d.rescue_applied)
            stats_agg["total_regions_rescued"] += rescued_count
            stats_agg["total_regions_rejected"] += rejected_count

            all_decisions.append(sample_decisions)

        stats_agg["fallback_rate"] = float(np.mean(all_fallback))
        stats_agg["rescue_rate"] = stats_agg["samples_with_rescue"] / max(n, 1)

        result = V3_2FusionResult(
            sample_ids=ids,
            pixel_maps=fused_maps.astype(np.float32),
            image_scores=vis_scores.astype(np.float32),  # visual unchanged
            pixel_residual=all_residual.astype(np.float32),
            rescue_mask=all_rescue_mask,
            visual_fallback=all_fallback,
            decisions=all_decisions,
            stats=stats_agg,
        )
        return result.validate()
