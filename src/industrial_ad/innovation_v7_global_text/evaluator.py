"""Evaluator-side helpers (task book 17: labels/masks load ONLY here).

Exporters / cache builders must never import this module.
"""

from __future__ import annotations

import numpy as np

from . import is_normal_id


def labels_from_ids(sample_ids) -> np.ndarray:
    """1 = defective (no '/good/'), 0 = normal. Derived from the frozen ids."""
    return np.asarray([1 if not is_normal_id(s) else 0 for s in sample_ids],
                      dtype=np.int64)


def image_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Image-AP / Image-AUROC (sklearn); None if a class is missing."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return {"image_ap": None, "image_auroc": None}
    return {"image_ap": float(average_precision_score(y, s)),
            "image_auroc": float(roc_auc_score(y, s))}


def top1_mean(maps448: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """Mean of the top `frac` pixels per 448 map (image score)."""
    flat = np.sort(maps448.reshape(len(maps448), -1), axis=1)
    k = max(1, int(round(frac * flat.shape[1])))
    return flat[:, -k:].mean(axis=1)
