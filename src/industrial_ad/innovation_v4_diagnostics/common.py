"""innovation_v4_diagnostics.common — shared guards, loaders and evaluator-only metrics.

The label-free algorithmic view and the evaluator-only ground-truth view are kept
apart exactly like innovation_v2: methods never see gt_sp / imgs_masks; tier
stratification and metric computation happen after scores are written.

Only MPDD (development) may be touched by the diagnostics (task book 14 section
10). BTAD / MVTec / VisA are protected by ``assert_frozen_validation_dataset``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # <repo>
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "anomalydino")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v2 import common as v2c  # noqa: E402
from industrial_ad.innovation_v2.common import (  # noqa: E402
    AlignedFeatures,
    DATASETS,
    InnovationError,
    ValidationDatasetAccessError,
    align_features,
    dirs_for,
    load_features,
    manifest_for,
    reference_ids_for,
    sha256_bytes,
    sha256_file,
)

EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_v4_diagnostics"
MPDD_DATA_ROOT = ROOT / "data" / "mpdd_raw" / "MPDD"

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def assert_development_only() -> None:
    """Diagnostics may only run on MPDD (development role)."""
    v2c.assert_development_only("mpdd")


def development_dataset() -> str:
    return "mpdd"


# ---------------------------------------------------------------------------
# Loading: label-free view + separate evaluator-only ground truth
# ---------------------------------------------------------------------------

def aligned_category(dataset: str, seed: int, shot: int, category: str) -> AlignedFeatures:
    """Aligned DINO+CLIP features for one MPDD category (label-free view).

    ``aligned`` carries d_feat/c_feat/d_ref/c_ref/grid/sample_ids/ref_ids only —
    no gt_sp / imgs_masks.
    """
    dino, clip = v2c.load_category_features(dataset, seed, shot, category)
    ref_ids = reference_ids_for(manifest_for(dataset), category, seed, shot)
    return align_features(dino, clip, ref_ids)


def evaluator_gt(dataset: str, seed: int, shot: int, category: str) -> dict:
    """Evaluator-only ground truth view (never handed to any algorithm)."""
    dino, _ = v2c.load_category_features(dataset, seed, shot, category)
    return {
        "gt_sp": np.asarray(dino["gt_sp"], dtype=np.int64),      # (N,) 0=good 1=bad
        "imgs_masks": np.asarray(dino["imgs_masks"], dtype=np.uint8),  # (N,448,448)
        "sample_ids": np.asarray(dino["sample_ids"]),
        "grid": tuple(int(v) for v in dino["grid_size"]),
    }


def image_path_for(dataset: str, relative: str) -> Path:
    """Map a manifest/cache relative id (e.g. bracket_black/test/good/000.png)
    onto the MPDD image file used by the frozen feature pipeline."""
    return MPDD_DATA_ROOT / relative


# ---------------------------------------------------------------------------
# Evaluator-only pixel metrics on tier subsets
# ---------------------------------------------------------------------------

def tier_of_area(area_frac: float, small_max: float = 0.005,
                 mid_max: float = 0.05) -> str:
    """Pre-registered GT defect-area tiers (fraction of image pixels)."""
    if area_frac <= small_max:
        return "small"
    if area_frac <= mid_max:
        return "mid"
    return "large"


def pixel_ap_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray):
    """Pixel-level AP / AUROC for one pooled subset (evaluator-only)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None, None
    y = np.concatenate([np.ones(pos_scores.size, dtype=np.int64),
                        np.zeros(neg_scores.size, dtype=np.int64)])
    s = np.concatenate([pos_scores.astype(np.float64), neg_scores.astype(np.float64)])
    if np.unique(y).size < 2 or np.all(s == s[0]):
        return None, None
    return (float(average_precision_score(y, s)),
            float(roc_auc_score(y, s)))


def tier_pooled_map_scores(
    maps: np.ndarray,  # (N, 448, 448) scores (higher = more anomalous)
    gt_masks: np.ndarray,  # (N, 448, 448) uint8
    bad_idx: np.ndarray,
    good_idx: np.ndarray,
    tier: str,
    small_max: float = 0.005,
    mid_max: float = 0.05,
) -> dict:
    """Pooled metrics for bad images whose mask area fraction falls in `tier`.

    Returns pos/neg pixel scores together with per-image AUROC (used by the
    oracle headroom computation).
    """
    sel = []
    per_image = []
    total = maps.shape[1] * maps.shape[2]
    for i in bad_idx:
        m = gt_masks[i]
        frac = float(m.sum()) / float(total)
        if tier_of_area(frac, small_max, mid_max) != tier:
            continue
        sel.append(i)
        pm = maps[i][m > 0]
        nm = maps[i][m == 0]
        per_image.append({"sample": int(i), "area_frac": round(frac, 6),
                          "n_pos_px": int((m > 0).sum()),
                          "image_auroc": pixel_ap_auroc(pm, nm)[1]})
    if not sel:
        return {"tier": tier, "n_bad_images": 0, "n_pos_px": 0, "n_neg_px": 0,
                "pos_scores": np.empty(0, dtype=np.float32),
                "neg_scores": np.empty(0, dtype=np.float32),
                "ap": None, "auroc": None, "per_image": per_image}
    sel = np.asarray(sel)
    pos = maps[sel][gt_masks[sel] > 0]
    # negatives: bad-image background plus every good image (whole image)
    good_maps = maps[good_idx]
    neg_parts = [maps[sel][gt_masks[sel] == 0]]
    if good_maps.size:
        neg_parts.append(good_maps.ravel())
    neg = np.concatenate(neg_parts)
    ap, auroc = pixel_ap_auroc(pos, neg)
    return {"tier": tier, "n_bad_images": int(len(sel)), "n_pos_px": int(pos.size),
            "n_neg_px": int(neg.size), "pos_scores": np.asarray(pos, dtype=np.float32),
            "neg_scores": np.asarray(neg, dtype=np.float32),
            "ap": ap, "auroc": auroc, "per_image": per_image}


def oracle_pooled_map(
    maps_a1: np.ndarray, maps_freq: np.ndarray,
    gt_masks: np.ndarray, bad_idx: np.ndarray,
) -> np.ndarray:
    """Instance-level oracle: per bad image pick the map that scores its GT best
    (evaluator-only, uses GT). Returns an oracle [N,H,W] contribution array used
    purely to estimate complementarity headroom — never to train a method."""
    oracle = maps_a1.copy()
    for i in bad_idx:
        m = gt_masks[i] > 0
        if not m.any():
            continue
        _, a1_auc = pixel_ap_auroc(maps_a1[i][m], maps_a1[i][~m])
        _, fr_auc = pixel_ap_auroc(maps_freq[i][m], maps_freq[i][~m])
        if a1_auc is None or fr_auc is None:
            continue
        if fr_auc > a1_auc:
            oracle[i] = maps_freq[i]
    return oracle
