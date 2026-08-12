from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def canonical_sample_id(value: str) -> str:
    """Normalize the sample identifiers emitted by different baselines.

    Unified caches currently use either dataset paths such as
    ``candle/test/bad/000.JPG`` or compact identifiers such as
    ``candle-bad-000``.  This function maps both forms to the compact form.
    """

    normalized = str(value).strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3:
        category = parts[0]
        defect = parts[-2].casefold()
        defect = {"normal": "good", "anomaly": "bad"}.get(defect, defect)
        stem = Path(parts[-1]).stem
        return f"{category}-{defect}-{stem}".casefold()
    return Path(normalized).stem.casefold()


def canonicalize_sample_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray([canonical_sample_id(str(value)) for value in values])


def _duplicates(values: np.ndarray) -> list[str]:
    unique, counts = np.unique(values, return_counts=True)
    return unique[counts > 1].tolist()


@dataclass(frozen=True)
class AlignmentPlan:
    """How to reorder a candidate cache to match a reference cache."""

    reference_ids: np.ndarray
    candidate_ids: np.ndarray
    candidate_order: np.ndarray
    order_already_equal: bool


def build_alignment_plan(
    reference_ids: np.ndarray, candidate_ids: np.ndarray
) -> AlignmentPlan:
    reference = canonicalize_sample_ids(np.asarray(reference_ids).reshape(-1))
    candidate = canonicalize_sample_ids(np.asarray(candidate_ids).reshape(-1))

    reference_duplicates = _duplicates(reference)
    candidate_duplicates = _duplicates(candidate)
    if reference_duplicates:
        raise ValueError(
            f"reference sample_ids are not unique after normalization: "
            f"{reference_duplicates[:5]}"
        )
    if candidate_duplicates:
        raise ValueError(
            f"candidate sample_ids are not unique after normalization: "
            f"{candidate_duplicates[:5]}"
        )

    reference_set = set(reference.tolist())
    candidate_set = set(candidate.tolist())
    if reference_set != candidate_set:
        missing = sorted(reference_set - candidate_set)
        extra = sorted(candidate_set - reference_set)
        raise ValueError(
            "sample sets differ after normalization: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )

    candidate_index = {sample_id: index for index, sample_id in enumerate(candidate)}
    order = np.asarray(
        [candidate_index[sample_id] for sample_id in reference], dtype=np.int64
    )
    return AlignmentPlan(
        reference_ids=reference,
        candidate_ids=candidate,
        candidate_order=order,
        order_already_equal=bool(np.array_equal(reference, candidate)),
    )
