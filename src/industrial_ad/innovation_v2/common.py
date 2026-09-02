"""Shared framework for the A2 innovation_v2 multi-route program.

Responsibilities (task book ``12_MULTI_ROUTE_...`` section 4):
  * route registry and validation-dataset guard (development vs frozen);
  * aligned per-branch features (CLIP resized to DINO grid, per-branch L2);
  * frozen A1 / DINO / CLIP baseline scores reusing the RCEC primitives;
  * six-metric evaluator (labels/masks only enter here);
  * config + input hashing and report schema v1;
  * generic per-category evaluation driving a route ``score_fn``.

The algorithm layer never receives ``gt_sp`` / ``imgs_masks``; those are loaded
separately by the evaluator only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # <repo>/src/industrial_ad/innovation_v2
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from evaluate_a1_feature_fusion import STRIDE, compute_metrics, load_features, resize_patches  # noqa: E402
from evaluate_a1_complete_metrics import compute_image_metrics  # noqa: E402
from industrial_ad.fusion import rcec  # noqa: E402
from src.utils import dists2map  # noqa: E402

FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_v2"
DEV_ROOT = EXPERIMENT_ROOT / "01_small_gates"

# Feature cache layout identical to the frozen A1 / RCEC runners.
DATASETS = {
    "mpdd": {"role": "development",
             "dino_fmt": "features_vitb14_s{seed}_k{shot}",
             "clip_fmt": "features_s{seed}_k{shot}"},
    "btad": {"role": "external_frozen_validation",
             "dino_fmt": "features_vitb14_btad_s{seed}_k{shot}",
             "clip_fmt": "features_btad_s{seed}_k{shot}"},
    "visa": {"role": "in_domain_frozen_validation",
             "dino_fmt": "visa_features_vitb14/s{seed}_k{shot}",
             "clip_fmt": "visa_features/s{seed}_k{shot}"},
    "mvtec": {"role": "external_frozen_validation",
              "dino_fmt": "mvtec_features_vitb14/s{seed}_k{shot}",
              "clip_fmt": "mvtec_features/s{seed}_k{shot}"},
}

# Pre-registered route registry (id -> label) — fixed by the task book.
ROUTE_LABELS = {
    "A_LNDC": "Local Normal-Density Calibrated Dual-Encoder Memory",
    "B_DSAM": "Deformable Spatially-Aware Dual-Encoder Memory",
    "C_CEQA": "Cross-Encoder Consensus-Bounded Query Adaptation",
    "D_DEVA": "Dual-Encoder Equivariance-Validated Normal Memory Augmentation",
    "E_NCPRA": "Normal-only Cross-Encoder Predictive Residual Adapter",
    "F_FAGR": "Feature-Affinity Graph Refinement",
}


class InnovationError(RuntimeError):
    pass


class ValidationDatasetAccessError(InnovationError):
    pass


def validate_algorithm_inputs(payload: dict) -> None:
    """Reject payloads carrying test labels/masks before any algorithm runs."""
    banned = {"gt_sp", "imgs_masks", "test_scores", "test_labels", "test_masks"}
    present = sorted(banned & set(payload))
    if present:
        raise InnovationError(
            f"algorithm input must not contain label/mask fields: {present}")


# ---------------------------------------------------------------------------
# Dataset / config helpers
# ---------------------------------------------------------------------------

def dataset_role(dataset: str) -> str:
    return DATASETS[dataset]["role"]


def assert_development_only(dataset: str) -> None:
    """Guard: candidates may only be developed/tuned on MPDD."""
    if dataset != "mpdd":
        raise ValidationDatasetAccessError(
            f"{dataset} is {DATASETS[dataset]['role']}; candidates cannot be "
            "developed or tuned on validation datasets")


def assert_frozen_validation_dataset(dataset: str) -> None:
    if dataset not in ("btad", "mvtec", "visa"):
        raise ValidationDatasetAccessError(
            f"{dataset} is not an allowed frozen-validation dataset")


def dirs_for(dataset: str, seed: int, shot: int) -> tuple[Path, Path]:
    meta = DATASETS[dataset]
    dino = FEATURES_ROOT / meta["dino_fmt"].format(seed=seed, shot=shot) / "anomalydino_visual"
    clip = FEATURES_ROOT / meta["clip_fmt"].format(seed=seed, shot=shot) / "anomalyclip_text"
    return dino, clip


def manifest_for(dataset: str) -> dict:
    return json.loads(
        (ROOT / "data" / "splits" / dataset / "manifest.json").read_text(encoding="utf-8"))


def reference_ids_for(manifest: dict, category: str, seed: int, shot: int) -> list[str]:
    return list(manifest["categories"][category][str(seed)][str(shot)])


def load_config(path: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_sha256"] = sha256_file(path)
    return cfg


def candidates_from_config(cfg: dict) -> list[dict]:
    raise NotImplementedError("overridden per route in each config/runner")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def hash_npz(path: Path) -> str:
    return sha256_file(path)


# ---------------------------------------------------------------------------
# Aligned features
# ---------------------------------------------------------------------------

@dataclass
class AlignedFeatures:
    d_feat: np.ndarray  # (N,H,W,768) L2 per-branch
    c_feat: np.ndarray  # (N,H,W,768)
    d_ref: np.ndarray   # (S,H,W,768)
    c_ref: np.ndarray   # (S,H,W,768)
    grid: tuple[int, int]
    sample_ids: np.ndarray
    ref_ids: list[str]
    category: str = ""

    @property
    def n_images(self) -> int:
        return int(self.d_feat.shape[0])

    @property
    def n_references(self) -> int:
        return int(self.d_ref.shape[0])


def align_features(dino: dict, clip: dict, ref_ids: list[str]) -> AlignedFeatures:
    """Reorder CLIP to DINO order, resize CLIP grids to DINO grid, L2 both branches."""
    if dino["ref_patch_features"].shape[0] != len(ref_ids):
        raise rcec.AlignmentError(
            f"DINO ref blocks {dino['ref_patch_features'].shape[0]} != manifest refs {len(ref_ids)}")
    if clip["ref_patch_features"].shape[0] != len(ref_ids):
        raise rcec.AlignmentError(
            f"CLIP ref blocks {clip['ref_patch_features'].shape[0]} != manifest refs {len(ref_ids)}")
    aligned = rcec.align_and_normalize_paired_features(
        dino_patch=dino["patch_features"],
        clip_patch=clip["patch_features"],
        dino_ref=dino["ref_patch_features"],
        clip_ref=clip["ref_patch_features"],
        dino_sample_ids=dino["sample_ids"],
        clip_sample_ids=clip["sample_ids"],
        dino_grid=dino["grid_size"],
        resize_fn=resize_patches,
    )
    return AlignedFeatures(
        d_feat=aligned["d_feat"],
        c_feat=aligned["c_feat"],
        d_ref=aligned["d_ref"],
        c_ref=aligned["c_ref"],
        grid=aligned["grid"],
        sample_ids=np.asarray(dino["sample_ids"]),
        ref_ids=list(ref_ids),
    )


def paired_memory(aligned: AlignedFeatures) -> rcec.PairedMemory:
    return rcec.build_paired_reference_memory(
        aligned.d_ref, aligned.c_ref, aligned.n_references)


# ---------------------------------------------------------------------------
# Baseline scores (frozen A1 / DINO / CLIP), all returned as [N, H, W] grids
# ---------------------------------------------------------------------------

def a1_grid(aligned: AlignedFeatures, dino_weight: float = 0.5) -> np.ndarray:
    mem = paired_memory(aligned)
    flat = rcec.compute_a1_dists(aligned.d_feat, aligned.c_feat, mem, dino_weight=dino_weight)
    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    return flat.reshape(n, h, w)


def dino_grid(aligned: AlignedFeatures) -> np.ndarray:
    mem = paired_memory(aligned)
    flat = rcec.compute_dino_dists(aligned.d_feat, mem)
    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    return flat.reshape(n, h, w)


def clip_grid(aligned: AlignedFeatures) -> np.ndarray:
    """CLIP-only 1-NN distance grid (route C consensus selection)."""
    mem = paired_memory(aligned)
    ref = mem.c
    query = aligned.c_feat.reshape(-1, aligned.c_feat.shape[-1])
    query = np.ascontiguousarray(query, dtype=np.float32)
    index = rcec._faiss_index(np.ascontiguousarray(ref, dtype=np.float32))
    dists, _ = rcec._search_chunked(index, query, k=1, chunk=16384)
    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    return (dists[:, 0] / 2.0).reshape(n, h, w)


# ---------------------------------------------------------------------------
# Six-metric evaluator (labels/masks only here)
# ---------------------------------------------------------------------------

def full_metrics(maps: np.ndarray, gt_masks: np.ndarray, gt_sp: np.ndarray) -> dict:
    pixel = None
    try:
        pixel = compute_metrics(maps.astype(np.float64), gt_masks)
    except ValueError:
        pixel = None
    if len(np.unique(gt_sp)) < 2:
        image = {"image_auroc": None, "image_ap": None, "image_f1_max": None}
    else:
        image = compute_image_metrics(gt_sp, maps)
    return {"pixel": pixel, "image": image}


def grids_to_maps(score_grid: np.ndarray, map_size: tuple[int, int]) -> np.ndarray:
    return np.stack(
        [dists2map(score_grid[i], map_size) for i in range(score_grid.shape[0])]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Generic per-category evaluation driving a route score_fn
# ---------------------------------------------------------------------------

def evaluate_category_generic(
    aligned: AlignedFeatures,
    ref_ids: list[str],
    seed: int,
    shot: int,
    route_id: str,
    candidate: dict,
    cfg: dict,
    category: str,
    score_fn: Callable,
    dino_weight: float = 0.5,
    map_fn: Callable = None,
) -> dict:
    """Evaluate one (category, candidate) for any route.

    ``score_fn(aligned, candidate, cfg) -> (scores_grid [N,H,W], diagnostics dict)``.
    The score function never receives labels/masks. Returns the report dict with
    route/A1/DINO metrics and delta vs A1.

    ``map_fn`` overrides the default Gaussian upsampling (used by route F FAGR,
    which must not apply the second sigma=4 Gaussian).
    """
    maps_size = tuple(int(v) for v in cfg.get("postprocess", {}).get("map_size", [448, 448]))
    if map_fn is None:
        map_fn = grids_to_maps

    s_new, diagnostics = score_fn(aligned, candidate, cfg)
    if s_new.shape != (aligned.n_images, *aligned.grid):
        raise InnovationError(
            f"{route_id} score shape {s_new.shape} != {(aligned.n_images, *aligned.grid)}")
    if not np.all(np.isfinite(s_new)):
        raise InnovationError(f"{route_id} produced non-finite scores")

    s_a1 = a1_grid(aligned, dino_weight=dino_weight)
    s_dino = dino_grid(aligned)

    maps_new = map_fn(s_new, maps_size)
    # A1/DINO baselines always use the frozen Gaussian pipeline, so
    # delta_vs_a1 measures against the paper's frozen A1.
    maps_a1 = grids_to_maps(s_a1, maps_size)
    maps_dino = grids_to_maps(s_dino, maps_size)

    report = {
        "schema_version": 1,
        "program": "innovation_v2",
        "route": route_id,
        "candidate_id": candidate.get("id", json.dumps(candidate, sort_keys=True)),
        "dataset": "mpdd",
        "dataset_role": "development",
        "seed": seed,
        "shot": shot,
        "category": category,
        "n_test_images": aligned.n_images,
        "grid": list(aligned.grid),
        "candidate": candidate,
        "diagnostics": diagnostics,
        "leakage_flags": {
            "test_labels_used_by_method": False,
            "test_masks_used_by_method": False,
            "test_distribution_used_for_calibration": False,
            "validation_dataset_used_for_tuning": False,
            "category_specific_test_rules_used": False,
        },
        "checks": {"no_nan_inf_scores": bool(np.all(np.isfinite(s_new)))},
    }
    # Metrics are attached by the runner after loading ground truth, so the
    # algorithm report above stays label-free; keep a placeholder for structure.
    report["metrics"] = None
    report["_maps"] = {"new": maps_new, "a1": maps_a1, "dino": maps_dino}
    return report


def attach_metrics(report: dict, gt_masks: np.ndarray, gt_sp: np.ndarray) -> None:
    """Compute six metrics for a report produced by evaluate_category_generic."""
    m_new = full_metrics(report["_maps"]["new"], gt_masks, gt_sp)
    m_a1 = full_metrics(report["_maps"]["a1"], gt_masks, gt_sp)
    m_dino = full_metrics(report["_maps"]["dino"], gt_masks, gt_sp)

    delta = {}
    for group in ("pixel", "image"):
        delta[group] = {}
        src = m_new[group]
        keys = src.keys() if isinstance(src, dict) else []
        for key in keys:
            a, b = m_new[group][key], m_a1[group][key]
            delta[group][key] = None if (a is None or b is None) else float(a - b)

    report["metrics"] = {"new": m_new, "a1": m_a1, "dino": m_dino, "delta_vs_a1": delta}
    report.pop("_maps", None)


def load_category_features(dataset: str, seed: int, shot: int, category: str) -> tuple[dict, dict]:
    dino_dir, clip_dir = dirs_for(dataset, seed, shot)
    clip_path = clip_dir / f"{category}.npz"
    if not clip_path.is_file():
        raise InnovationError(f"missing clip features: {clip_path}")
    return load_features(dino_dir / f"{category}.npz"), load_features(clip_path)


__all__ = [
    "ROOT", "FEATURES_ROOT", "EXPERIMENT_ROOT", "DEV_ROOT", "DATASETS", "ROUTE_LABELS",
    "InnovationError", "ValidationDatasetAccessError",
    "validate_algorithm_inputs", "dataset_role", "assert_development_only",
    "assert_frozen_validation_dataset", "dirs_for", "manifest_for", "reference_ids_for",
    "load_config", "sha256_file", "sha256_bytes", "hash_npz",
    "AlignedFeatures", "align_features", "paired_memory",
    "a1_grid", "dino_grid", "clip_grid",
    "full_metrics", "grids_to_maps", "evaluate_category_generic", "attach_metrics",
    "load_category_features",
]
