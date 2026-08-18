"""V3.3-clean: leakage-safe fusion with reference-only calibration.

Protocol (docs/DYNAMIC_FUSION_NEXT_STEPS.md, 阶段二):

- RouterInput carries ONLY predictions, K normal-reference maps, unlabeled
  features, sample IDs and metadata. It never carries test labels or masks.
- EvaluationTarget carries ONLY test labels/masks and is passed exclusively to
  evaluators, never into any fusion/calibration/routing code path.
- Calibration statistics come ONLY from the K normal-reference maps of the
  same seed/shot: center = median, scale = IQR, plus MAD and q95/q99.
- All five leakage flags must be false for every report:
  test_predictions_used / test_labels_used / test_masks_used /
  test_dataset_statistics_used / test_normal_selection_used.
- Visual branch is the default safe output; text may only contribute a bounded
  residual and any unreliable branch must fall back to the visual anchor.

This module intentionally does NOT import or accept gt labels/masks anywhere
except EvaluationTarget, which is only consumed by evaluator code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

EPS: float = 1e-8
DEFAULT_ANCHOR: str = "anomalydino_visual"


# ---------------------------------------------------------------------------
# Data boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouterInput:
    """Leakage-safe fusion input. No test labels or masks allowed."""

    branches: Dict[str, np.ndarray]        # branch name -> test maps [N,H,W] float32
    reference_maps: Dict[str, np.ndarray]  # branch name -> K normal-reference maps [K,H,W]
    sample_ids: np.ndarray                 # [N]
    category: str
    seed: int
    shot: int
    metadata: Dict[str, object] = field(default_factory=dict)
    # Intentionally NO gt_labels / gt_masks / gt_* attributes.


@dataclass(frozen=True)
class EvaluationTarget:
    """Test ground truth. Only evaluators may touch this."""

    gt_labels: np.ndarray                 # [N] 0/1
    gt_masks: np.ndarray                  # [N,H,W] uint8
    sample_ids: np.ndarray                # [N]


# ---------------------------------------------------------------------------
# Reference-only calibration
# ---------------------------------------------------------------------------

def estimate_reference_stats(reference_maps: np.ndarray) -> dict:
    """Estimate center(median)/scale(IQR), MAD, q95, q99 from K normal-reference maps.

    Raises ValueError on NaN/Inf or empty input; scale is floored at EPS.
    """
    maps = np.asarray(reference_maps, dtype=np.float64)
    if maps.size == 0:
        raise ValueError("reference_maps is empty")
    vals = maps.ravel()
    if not np.isfinite(vals).all():
        raise ValueError("reference_maps contains NaN or infinity")
    center = float(np.median(vals))
    scale = float(np.subtract(*np.percentile(vals, [75, 25])))
    mad = float(np.median(np.abs(vals - center)) * 1.4826)
    q95, q99 = (float(v) for v in np.percentile(vals, [95, 99]))
    return {
        "center": center,
        "scale": max(scale, EPS),
        "mad": max(mad, EPS),
        "q95": q95,
        "q99": q99,
    }


def compute_z_score(maps: np.ndarray, center: float, scale: float) -> np.ndarray:
    """(value - center) / scale with a positive floor on scale."""
    return (np.asarray(maps, dtype=np.float64) - center) / max(scale, EPS)


def branch_is_reliable(
    reference_stats: dict,
    test_maps: np.ndarray,
    nan_tolerance: float = 0.0,
    degenerate_scale_eps: float = EPS,
) -> bool:
    """A branch is unreliable (fall back to visual) when its reference
    distribution is degenerate or its test maps are mostly non-finite."""
    if reference_stats["scale"] <= degenerate_scale_eps and reference_stats["mad"] <= degenerate_scale_eps:
        return False
    maps = np.asarray(test_maps, dtype=np.float64)
    if not np.isfinite(maps).all():
        bad = float(np.count_nonzero(~np.isfinite(maps))) / maps.size
        if bad > nan_tolerance:
            return False
    return True


# ---------------------------------------------------------------------------
# Alignment & validation (sample IDs, shapes, finiteness)
# ---------------------------------------------------------------------------

def validate_router_input(ri: RouterInput) -> None:
    """Reject misaligned / missing / duplicated sample IDs and bad shapes.

    Raises ValueError describing the first problem found.
    """
    ids = np.asarray(ri.sample_ids)
    if ids.ndim != 1:
        raise ValueError(f"sample_ids must be 1-D, got {ids.ndim}-D")
    if ids.size == 0:
        raise ValueError("sample_ids is empty")
    unique = set(str(v) for v in ids)
    if len(unique) != ids.size:
        raise ValueError("sample_ids contains duplicates")
    if not ri.branches:
        raise ValueError("RouterInput has no branches")
    n = ids.size
    for bname, maps in ri.branches.items():
        arr = np.asarray(maps)
        if arr.ndim != 3 or arr.shape[0] != n:
            raise ValueError(
                f"branch {bname}: expected [N={n},H,W], got {arr.shape}"
            )
        if arr.shape[1:] == (0, 0):
            raise ValueError(f"branch {bname}: zero spatial extent")
    for bname, refs in ri.reference_maps.items():
        arr = np.asarray(refs)
        if arr.ndim != 3:
            raise ValueError(
                f"reference {bname}: expected [K,H,W], got shape {arr.shape}"
            )
        if arr.shape[1:] != ri.branches[bname].shape[1:]:
            raise ValueError(
                f"reference {bname}: spatial shape {arr.shape[1:]} != branch {ri.branches[bname].shape[1:]}"
            )
    missing = set(ri.branches) - set(ri.reference_maps)
    if missing:
        raise ValueError(f"missing reference maps for branches: {sorted(missing)}")


def sanitize_finite(maps: np.ndarray, fill: float | None = None) -> np.ndarray:
    """NaN/Inf-safe handling: replace non-finite values with a finite floor so
    an unreliable branch cannot fabricate a high anomaly score.

    The default floor is the finite minimum of the maps themselves, so the
    z-scored replacement stays an extreme *low* anomaly score. The output is
    always fully finite.
    """
    arr = np.asarray(maps, dtype=np.float64).copy()
    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    if fill is None:
        finite = arr[np.isfinite(arr)]
        floor = float(finite.min()) if finite.size else 0.0
    else:
        floor = float(fill)
    arr[bad] = floor
    return arr


# ---------------------------------------------------------------------------
# Clean fusion strategies
# ---------------------------------------------------------------------------

def weighted_ensemble_clean(
    ri: RouterInput,
    weights: Dict[str, float],
    anchor: str = DEFAULT_ANCHOR,
    nan_tolerance: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Z-score calibrate each branch using ONLY its K normal-reference maps,
    then weighted-average. Unreliable branches fall back to the visual anchor.

    Returns (fused_maps [N,H,W] float64, diagnostics dict).
    """
    validate_router_input(ri)
    n = ri.sample_ids.size
    h, w = ri.branches[anchor].shape[1:]

    stats: Dict[str, dict] = {}
    reliable: Dict[str, bool] = {}
    for bname, maps in ri.branches.items():
        ref_stats = estimate_reference_stats(ri.reference_maps[bname])
        stats[bname] = ref_stats
        reliable[bname] = branch_is_reliable(ref_stats, maps, nan_tolerance=nan_tolerance)

    # Normalise weights over reliable branches; anchor always kept as fallback.
    total = sum(weights.get(bname, 0.0) for bname in ri.branches if reliable.get(bname, False))
    if total <= 0:
        # nothing reliable -> pure visual anchor, uncalibrated raw maps
        fused = np.asarray(ri.branches[anchor], dtype=np.float64)
        return fused, {"reliable": reliable, "fallback": "all", "stats": stats}

    fused = np.zeros((n, h, w), dtype=np.float64)
    fallback_flags: List[str] = []
    for bname, maps in ri.branches.items():
        w_i = weights.get(bname, 0.0) / total
        if w_i <= 0:
            continue
        if not reliable.get(bname, False):
            fallback_flags.append(bname)
            if bname == anchor:
                # Anchor itself is unreliable: fall back to raw anchor maps.
                fused = np.asarray(maps, dtype=np.float64)
            continue
        safe = sanitize_finite(maps)
        z = compute_z_score(safe, stats[bname]["center"], stats[bname]["scale"])
        fused += w_i * z

    return fused, {
        "reliable": reliable,
        "fallback_branches": fallback_flags,
        "stats": stats,
    }


def evaluate_clean(
    ri: RouterInput,
    target: EvaluationTarget,
    fused_maps: np.ndarray,
    stride: int = 8,
) -> dict:
    """Evaluate fused maps against an EvaluationTarget.

    This is the ONLY function allowed to read test ground truth.
    """
    if set(target.sample_ids.tolist()) != set(ri.sample_ids.tolist()):
        raise ValueError("EvaluationTarget sample_ids do not match RouterInput")
    if not np.isfinite(fused_maps).all():
        raise ValueError("fused_maps contains NaN/Inf")
    from evaluate_unified import aupro_fast  # local import avoids cycle at module load

    ms = fused_maps[:, ::stride, ::stride]
    mk = np.asarray(target.gt_masks, dtype=np.uint8)[:, ::stride, ::stride]
    flat_maps = ms.ravel()
    flat_labels = (mk.ravel() > 0.5).astype(np.int32)
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(mk, ms)),
    }
