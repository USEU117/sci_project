"""Dynamic fusion interfaces that do not accept ground-truth inputs."""

from .alignment import AlignmentPlan, build_alignment_plan, canonical_sample_id
from .baselines import FixedWeightFusion, single_branch_fusion
from .calibration import (
    PIXEL_REFERENCE_QUANTILE,
    BranchCalibration,
    RobustNormalCalibrator,
    load_category_calibrations,
)
from .contracts import BranchPrediction, FusionResult
from .features import (
    augmentation_consistency,
    shot_sensitivity,
    spatial_response_concentration,
)
from .reference import NormalReferencePrediction
from .router import ConfidenceRouter
from .v2_calibration import (
    BranchV2Calibration,
    RankPreservingCalibrator,
    load_v2_category_calibrations,
)
from .v2_router import SafeRouterV2Config, SafeVisualDefaultRouterV2
from .v2_diagnostics import calibration_diagnostics, spearman_correlation
from .v2_ablation import run_v2_minimum_ablation

__all__ = [
    "AlignmentPlan",
    "BranchCalibration",
    "BranchPrediction",
    "ConfidenceRouter",
    "FixedWeightFusion",
    "FusionResult",
    "NormalReferencePrediction",
    "PIXEL_REFERENCE_QUANTILE",
    "RobustNormalCalibrator",
    "BranchV2Calibration",
    "RankPreservingCalibrator",
    "SafeRouterV2Config",
    "SafeVisualDefaultRouterV2",
    "calibration_diagnostics",
    "spearman_correlation",
    "run_v2_minimum_ablation",
    "build_alignment_plan",
    "canonical_sample_id",
    "augmentation_consistency",
    "load_category_calibrations",
    "load_v2_category_calibrations",
    "shot_sensitivity",
    "single_branch_fusion",
    "spatial_response_concentration",
]
