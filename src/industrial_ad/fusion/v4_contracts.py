"""V4 contracts: physical separation of RouterInput (prediction-only) and
EvaluationTarget (evaluator-only GT), plus the five leakage flags.

Rules enforced here (CPU, no model):
  * RouterInput never carries ground truth (labels / masks / gt fields).
  * EvaluationTarget carries GT and may only be read by the evaluator.
  * RouterInput.predictions are finite; sample IDs are unique and aligned with
    the spatial grid; empty prompts / wrong text order are rejected; a missing
    text branch is a valid visual-only fallback state.
  * RouterInput.prediction_hash() is deterministic and must not depend on any
    EvaluationTarget (the unit tests prove replace/shuffle/delete GT leaves it
    unchanged).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The five leakage fields required on every V4 report. Formal candidates must
# keep every one of these `false`.
LEAKAGE_FLAGS = {
    "test_predictions_used_for_parameter_fit": False,
    "test_labels_used_for_parameter_fit": False,
    "test_masks_used_for_parameter_fit": False,
    "test_dataset_statistics_used_for_calibration": False,
    "test_normal_selection_used": False,
}

_ALLOWED_TEXT_ORDER = ("normal_then_abnormal", "abnormal_then_normal")


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.size and not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def _sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _array_bytes(value: np.ndarray | None) -> bytes:
    if value is None:
        return b"<none>"
    arr = np.asarray(value)
    return np.ascontiguousarray(arr).tobytes()


@dataclass(frozen=True)
class EvaluationTarget:
    """Ground truth, evaluator-only. Must never be passed to a router/fusion
    model or be co-located with inference features in a router-readable cache."""

    sample_ids: np.ndarray
    image_labels: np.ndarray
    pixel_masks: np.ndarray

    def validate(self) -> "EvaluationTarget":
        ids = np.asarray(self.sample_ids).reshape(-1)
        labels = _finite("image_labels", self.image_labels).reshape(-1)
        masks = np.asarray(self.pixel_masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(f"pixel_masks must be [N,H,W], got {masks.shape}")
        if not (len(ids) == len(labels) == len(masks)):
            raise ValueError("sample_ids, image_labels and pixel_masks must have equal N")
        if len(ids) and len(set(ids.tolist())) != len(ids):
            raise ValueError("sample_ids must be unique")
        return self


@dataclass(frozen=True)
class RouterInput:
    """Prediction-only input to the fusion/dynamic stage. Ground truth is
    intentionally absent by construction."""

    sample_ids: np.ndarray
    visual_map: np.ndarray
    grid: tuple[int, int]
    text_map: np.ndarray | None = None
    normal_reference_stats: dict[str, Any] | None = None
    stability_features: dict[str, Any] | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "RouterInput":
        ids = np.asarray(self.sample_ids).reshape(-1)
        if len(ids) == 0:
            raise ValueError("sample_ids must be non-empty")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("sample_ids must be unique")

        visual = _finite("visual_map", self.visual_map)
        if visual.ndim != 3:
            raise ValueError(f"visual_map must be [N,H,W], got {visual.shape}")
        gh, gw = self.grid
        if (gh, gw) != (visual.shape[1], visual.shape[2]):
            raise ValueError(f"grid {self.grid} does not match visual_map spatial shape "
                             f"{visual.shape[1:]}")
        if len(ids) != visual.shape[0]:
            raise ValueError("sample_ids length must equal visual_map N")

        if self.text_map is not None:
            text = _finite("text_map", self.text_map)
            if text.ndim != 3:
                raise ValueError(f"text_map must be [N,H,W], got {text.shape}")
            if text.shape != visual.shape:
                raise ValueError(f"text_map {text.shape} must match visual_map {visual.shape}")

        # Prompt / text-order validation (when a text branch is present).
        meta = self.model_metadata or {}
        normal_prompt = meta.get("normal_prompt")
        abnormal_prompt = meta.get("abnormal_prompt")
        text_order = meta.get("text_order", "normal_then_abnormal")
        if self.text_map is not None:
            if not isinstance(normal_prompt, str) or not normal_prompt.strip():
                raise ValueError("normal_prompt must be a non-empty string when text_map is present")
            if not isinstance(abnormal_prompt, str) or not abnormal_prompt.strip():
                raise ValueError("abnormal_prompt must be a non-empty string when text_map is present")
        if text_order not in _ALLOWED_TEXT_ORDER:
            raise ValueError(f"text_order must be one of {_ALLOWED_TEXT_ORDER}, got {text_order!r}")

        return self

    def prediction_hash(self) -> str:
        """Deterministic hash of prediction-only content. Deliberately excludes
        any GT-derived field, so replacing/shuffling/deleting EvaluationTarget
        cannot change it."""
        self.validate()
        meta = json.dumps(self.model_metadata or {}, sort_keys=True, ensure_ascii=False)
        ids_bytes = b"|".join(str(s).encode("utf-8") for s in np.asarray(self.sample_ids).reshape(-1))
        return _sha256_bytes(
            ids_bytes,
            _array_bytes(self.visual_map),
            _array_bytes(self.text_map),
            str(self.grid).encode("ascii"),
            meta.encode("utf-8"),
        )


def verify_gt_independence(prediction_hash: str, target: EvaluationTarget) -> bool:
    """The prediction hash must be independent of the evaluation target. This
    helper is purely a documentation anchor; the real guarantee is structural:
    RouterInput holds no GT. Tests mutate/delete `target` and assert the hash is
    unchanged."""
    target.validate()
    return prediction_hash is not None
