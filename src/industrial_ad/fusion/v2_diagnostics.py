from __future__ import annotations

import numpy as np


def _finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks without requiring SciPy."""

    array = _finite_1d(values, "values")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if len(left_ranks) != len(right_ranks):
        raise ValueError("left and right must have equal N")
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = np.sqrt(
        np.sum(left_centered**2) * np.sum(right_centered**2)
    )
    if denominator == 0:
        return 1.0 if np.array_equal(left_ranks, right_ranks) else 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def calibration_diagnostics(
    raw: np.ndarray,
    calibrated: np.ndarray,
    *,
    boundary_epsilon: float = 0.001,
) -> dict[str, float | int]:
    raw_array = _finite_1d(raw, "raw")
    calibrated_array = _finite_1d(calibrated, "calibrated")
    if len(raw_array) != len(calibrated_array):
        raise ValueError("raw and calibrated must have equal N")
    if not 0 < boundary_epsilon < 0.5:
        raise ValueError("boundary_epsilon must be in (0, 0.5)")
    _, counts = np.unique(calibrated_array, return_counts=True)
    return {
        "sample_count": int(len(raw_array)),
        "spearman_raw_vs_calibrated": spearman_correlation(
            raw_array, calibrated_array
        ),
        "unique_value_ratio": float(len(counts) / len(calibrated_array)),
        "largest_tie_group": int(np.max(counts)),
        "lower_boundary_rate": float(
            np.mean(calibrated_array <= boundary_epsilon)
        ),
        "upper_boundary_rate": float(
            np.mean(calibrated_array >= 1.0 - boundary_epsilon)
        ),
        "calibrated_min": float(np.min(calibrated_array)),
        "calibrated_max": float(np.max(calibrated_array)),
    }
