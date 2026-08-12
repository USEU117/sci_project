from __future__ import annotations

from dataclasses import replace

from .baselines import FixedWeightFusion, single_branch_fusion
from .contracts import BranchPrediction, FusionResult
from .v2_calibration import BranchV2Calibration
from .v2_router import SafeRouterV2Config, SafeVisualDefaultRouterV2


def run_v2_minimum_ablation(
    visual: BranchPrediction,
    text: BranchPrediction,
    visual_calibration: BranchV2Calibration,
    text_calibration: BranchV2Calibration,
    config: SafeRouterV2Config | None = None,
) -> dict[str, FusionResult]:
    """Run the cache-level V2 controls that require no labels or masks.

    V1 and development-selected fixed-weight controls remain separate historical
    runs.  This function covers the V2-specific comparison without exposing
    evaluation truth to any router.
    """

    selected = (config or SafeRouterV2Config()).validate()
    calibrated_visual = visual_calibration.apply(visual)
    calibrated_text = text_calibration.apply(text)
    return {
        "raw_visual": single_branch_fusion(visual, "visual"),
        "raw_text": single_branch_fusion(text, "text"),
        "rank_calibrated_visual": single_branch_fusion(
            calibrated_visual, "visual"
        ),
        "rank_calibrated_fixed_050": FixedWeightFusion(0.5).fuse(
            calibrated_visual, calibrated_text
        ),
        "v2_image_only": SafeVisualDefaultRouterV2(
            visual_calibration,
            text_calibration,
            replace(selected, max_pixel_text_weight=0.0),
        ).fuse(visual, text),
        "v2_pixel_only": SafeVisualDefaultRouterV2(
            visual_calibration,
            text_calibration,
            replace(selected, max_image_text_weight=0.0),
        ).fuse(visual, text),
        "v2_complete_split": SafeVisualDefaultRouterV2(
            visual_calibration, text_calibration, selected
        ).fuse(visual, text),
    }
