"""innovation_v6_dgsafe — S0 DG-SAFE (task book 16).

Dual-geometry safeguarded evidence fusion: A1 nearest-neighbour maps + official
SubspaceAD reconstruction-residual maps, calibrated and gated by normal-only
reliability. This package holds export/load/align/metric tooling and the
Wave 0 (identity replay) / Wave 1 (complementarity) diagnostics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

A1_MAPS_ROOT = ROOT / "submission_repro_20260827" / "predictions_compact" / "maps" / "mpdd"
MPDD_DATA_ROOT = ROOT / "data" / "mpdd_raw" / "MPDD"
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_v6_dgsafe"
MAP_SIZE = (448, 448)
STRIDE = 8
PROTOCOL = json.loads(
    (ROOT / "configs" / "innovation_v6_dgsafe" / "wave0_protocol.json").read_text(encoding="utf-8")
)


def assert_development_only() -> None:
    """S0 runs on MPDD (development) only; validation datasets stay frozen."""
    from industrial_ad.innovation_v2.common import assert_development_only as _g
    _g("mpdd")


# ---------------------------------------------------------------------------
# A1 maps (frozen compact predictions; label-free)
# ---------------------------------------------------------------------------

def load_a1_patch_map(category: str, seed: int, shot: int) -> dict:
    """Load the frozen A1 concat patch map (N,32,32) float16 + sample_ids/refs."""
    path = A1_MAPS_ROOT / f"s{seed}_k{shot}" / f"{category}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=False)
    return {
        "sample_ids": np.asarray(d["sample_ids"]),
        "patch_map": np.asarray(d["concat_patch_map"], dtype=np.float32),  # (N,32,32)
        "ref_ids": np.asarray(d["ref_ids"]),
        "dataset": str(d["dataset"]),
        "seed": int(d["seed"]),
        "shot": int(d["shot"]),
    }


def grid_to_map448(grid2d: np.ndarray) -> np.ndarray:
    """Deterministic grid -> 448 map: INTER_LINEAR upsample + Gaussian sigma=4
    (same construction as the frozen A1 dists2map path)."""
    from src.utils import dists2map
    return np.asarray(dists2map(grid2d, MAP_SIZE), dtype=np.float32)


def a1_maps448(patch_map: np.ndarray) -> np.ndarray:
    """(N,32,32) -> (N,448,448) float32 via the frozen map path."""
    return np.stack([grid_to_map448(p) for p in patch_map]).astype(np.float32)


# ---------------------------------------------------------------------------
# SubspaceAD exported raw residual grids (produced by --export-maps)
# ---------------------------------------------------------------------------

def load_sub_raw(export_root: Path, seed: int, shot: int, category: str) -> dict:
    path = Path(export_root) / f"{category}_s{seed}_k{shot}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=False)
    return {
        "sample_ids": np.asarray(d["sample_ids"]),
        "amap_raw": np.asarray(d["amap_raw"], dtype=np.float32),   # (N,h_p,w_p)
        "ref_ids": np.asarray(d["ref_ids"]),
        "image_res": int(d["image_res"]),
        "pca_ev": float(d["pca_ev"]),
        "aug_count": int(d["aug_count"]),
        "official_commit": str(d["official_commit"]),
    }


def sub_maps448(amap_raw: np.ndarray) -> np.ndarray:
    """Raw 48x48 residual grid -> (N,448,448) via the same deterministic
    INTER_LINEAR + Gaussian sigma=4 path used for A1."""
    return a1_maps448(amap_raw)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_perm(sample_ids_b: np.ndarray, sample_ids_a: np.ndarray) -> np.ndarray:
    """Permutation that reorders b to match a; raises on mismatch/duplicates."""
    if len(sample_ids_b) != len(sample_ids_a):
        raise ValueError(f"length mismatch {len(sample_ids_b)} vs {len(sample_ids_a)}")
    order = {sid: i for i, sid in enumerate(sample_ids_b)}
    if len(order) != len(sample_ids_b):
        raise ValueError("duplicate sample_ids in b")
    perm = np.asarray([order[sid] for sid in sample_ids_a], dtype=np.int64)
    if not np.array_equal(sample_ids_b[perm], sample_ids_a):
        raise ValueError("sample set mismatch after alignment")
    return perm


# ---------------------------------------------------------------------------
# Evaluator-only ground truth + metrics (never handed to a method)
# ---------------------------------------------------------------------------

def gt_masks_for(sample_ids, data_root=MPDD_DATA_ROOT, res=MAP_SIZE):
    """Load 448 masks per sample from MPDD ground-truth dirs (evaluator side)."""
    import cv2
    out = np.zeros((len(sample_ids), res[0], res[1]), dtype=np.uint8)
    for i, sid in enumerate(sample_ids):
        rel = Path(sid)
        parts = rel.parts  # e.g. bracket_black/test/hole/129.png
        if len(parts) < 4 or parts[1] != "test":
            raise ValueError(f"unexpected sample_id: {sid}")
        cat, defect, stem = parts[0], parts[2], Path(parts[3]).stem
        if defect == "good":
            continue
        mask_path = Path(data_root) / cat / "ground_truth" / defect / f"{stem}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"cannot read {mask_path}")
        m = cv2.resize(m, (res[1], res[0]), interpolation=cv2.INTER_NEAREST)
        out[i] = (m > 0).astype(np.uint8)
    return out


def pixel_metrics_448(maps448: np.ndarray, gtm: np.ndarray, stride: int = STRIDE) -> dict:
    """Pooled Pixel-AP / Pixel-AUROC over the flattened[::stride] protocol
    (mirrors the frozen A1 evaluation convention at 448)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = gtm.ravel()[::stride].astype(np.int64)
    s = maps448.ravel()[::stride].astype(np.float64)
    has_pos, has_neg = bool((y == 1).any()), bool((y == 0).any())
    if not (has_pos and has_neg):
        return {"pixel_ap": None, "pixel_auroc": None}
    return {
        "pixel_ap": float(average_precision_score(y, s)),
        "pixel_auroc": float(roc_auc_score(y, s)),
    }


def image_ap_448(maps448: np.ndarray, gt_sp_like: np.ndarray) -> dict:
    """Image-level AP/AUROC from per-image max-pooled scores (evaluator side)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    img_s = np.asarray(maps448.reshape(len(maps448), -1).max(axis=1), dtype=np.float64)
    y = np.asarray(gt_sp_like, dtype=np.int64)
    has_pos, has_neg = bool((y == 1).any()), bool((y == 0).any())
    if not (has_pos and has_neg):
        return {"image_ap": None, "image_auroc": None}
    return {
        "image_ap": float(average_precision_score(y, img_s)),
        "image_auroc": float(roc_auc_score(y, img_s)),
    }
