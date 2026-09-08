"""Support-only image-level facility-coverage selection probe.

The unit selected here is an entire normal support image.  For each k=4
category, one support image is held out as the query and the other three clean
support images are the only candidate pool.  A facility objective selects one
or two images whose complete 16 x 16 fused-cell bank covers the candidate pool
with the smallest mean nearest-neighbour ``1 - cosine`` distance.

The script never uses the held-out feature or mask in the selector.  Synthetic
queries and masks are used only after selection, and are read from the frozen
v14 support cache.  The hard UTC deadline is checked at fold boundaries and a
partial result is written after every completed fold.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep numerical libraries on the declared CPU budget before importing numpy.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V14_COMMON = ROOT / "scripts" / "innovation_v14_decisive_validation_20260905"
sys.path.insert(0, str(V14_COMMON))

from v14_common import assert_fit_ids_are_support  # noqa: E402


SCRIPT_VERSION = "support_selection_v1"
SEED = 0
SHOT = 4
GRID = 16
RAW_SIZE = 1024
BRANCH_DIM = 768
FUSED_DIM = 1536
CELL_COUNT = GRID * GRID
CATEGORIES = (
    "bracket_black", "bracket_brown", "bracket_white", "connector",
    "metal_plate", "tubes",
)
FAMILIES = ("cutpaste", "local_erasure", "thin_scratch")
SYN_SEEDS = (0, 1, 2)
NUISANCE_CELL_STEP = 4
DEADLINE = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)

CACHE = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support"
OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "support_selection"

STRATEGIES = (
    "facility_k1", "facility_k2",
    "uniform_combo_mean_k1", "uniform_combo_mean_k2",
    "fixed_first_k1", "fixed_first_k2", "all3",
)


class DeadlineReached(RuntimeError):
    """The pre-registered overnight deadline was reached."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_deadline(stage: str) -> None:
    if datetime.now(timezone.utc) >= DEADLINE:
        raise DeadlineReached(stage)


def _json_value(value: Any) -> Any:
    """Convert numpy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norm, np.float32(1e-12))


def _to_grid16(x: np.ndarray) -> np.ndarray:
    """Resize a feature tensor to 32 grid if needed, then average pool to 16."""
    arr = np.asarray(x, dtype=np.float32)
    lead = arr.shape[:-3]
    h, w, d = arr.shape[-3:]
    flat = arr.reshape(-1, h, w, d)
    t = torch.from_numpy(flat).permute(0, 3, 1, 2)
    with torch.inference_mode():
        if (h, w) != (32, 32):
            t = torch.nn.functional.interpolate(t, size=(32, 32), mode="bilinear",
                                                 align_corners=False)
        t = torch.nn.functional.avg_pool2d(t, kernel_size=2, stride=2)
    out = t.permute(0, 2, 3, 1).numpy()
    return out.reshape(*lead, GRID, GRID, d).astype(np.float32, copy=False)


def _fuse(dino: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """A1 branch-normalize, weighted concatenate and row-normalize."""
    d = _to_grid16(dino)
    c = _to_grid16(clip)
    d_t = torch.from_numpy(d).reshape(-1, BRANCH_DIM)
    c_t = torch.from_numpy(c).reshape(-1, BRANCH_DIM)
    with torch.inference_mode():
        d_t = torch.nn.functional.normalize(d_t, dim=-1)
        c_t = torch.nn.functional.normalize(c_t, dim=-1)
        z = torch.cat([0.5 * d_t, 0.5 * c_t], dim=-1)
        z = torch.nn.functional.normalize(z, dim=-1)
    return z.numpy().reshape(*d.shape[:-1], FUSED_DIM).astype(np.float32, copy=False)


def _load_category(cat: str) -> dict[str, Any]:
    dp = CACHE / f"v14_p1_support_dino_s0_k{SHOT}" / f"{cat}.npz"
    cp = CACHE / f"v14_p1_support_clip_s0_k{SHOT}" / f"{cat}.npz"
    with np.load(dp, allow_pickle=False) as d, np.load(cp, allow_pickle=False) as c:
        refs = np.asarray(d["ref_rel"]).astype(str).tolist()
        clip_refs = np.asarray(c["ref_rel"]).astype(str).tolist()
        if refs != clip_refs:
            raise ValueError(f"{cat}: DINO/CLIP support order mismatch")
        if len(refs) != SHOT:
            raise ValueError(f"{cat}: expected k{SHOT}, got {len(refs)} refs")
        assert_fit_ids_are_support(refs, cat, SHOT, "0")
        normalized_refs = [r.replace("\\", "/") for r in refs]
        if any("/test/" in f"/{r}" for r in normalized_refs):
            raise ValueError(f"{cat}: test path in support refs")
        syn_kinds = np.asarray(d["syn_kinds"]).astype(str).tolist()
        clip_kinds = np.asarray(c["syn_kinds"]).astype(str).tolist()
        if syn_kinds != list(FAMILIES) or clip_kinds != list(FAMILIES):
            raise ValueError(f"{cat}: unexpected synthetic family order")
        syn_count = int(np.asarray(d["syn_seeds"]))
        if syn_count < max(SYN_SEEDS) + 1 or int(np.asarray(c["syn_seeds"])) != syn_count:
            raise ValueError(f"{cat}: unexpected synthetic seed count")
        nui_keys = np.asarray(d["nui_keys"]).astype(str).tolist()
        if nui_keys != np.asarray(c["nui_keys"]).astype(str).tolist():
            raise ValueError(f"{cat}: nuisance key order mismatch")
        clean_d = np.asarray(d["clean_feat"], dtype=np.float32)
        clean_c = np.asarray(c["clean_feat"], dtype=np.float32)
        syn_d = np.asarray(d["syn_feat"], dtype=np.float32)
        syn_c = np.asarray(c["syn_feat"], dtype=np.float32)
        nui_d = np.asarray(d["nui_feat"], dtype=np.float32)
        nui_c = np.asarray(c["nui_feat"], dtype=np.float32)
        masks = np.asarray(d["syn_masks"], dtype=np.uint8)
    clean = _fuse(clean_d, clean_c).reshape(SHOT, CELL_COUNT, FUSED_DIM)
    syn = _fuse(syn_d, syn_c).reshape(SHOT, len(FAMILIES) * syn_count,
                                       CELL_COUNT, FUSED_DIM)
    nui = _fuse(nui_d, nui_c).reshape(SHOT, len(nui_keys), CELL_COUNT, FUSED_DIM)
    return {
        "category": cat,
        "refs": refs,
        "clean": clean,
        "syn": syn,
        "nui": nui,
        "masks": masks,
        "nui_keys": nui_keys,
        "syn_count": syn_count,
    }


def _combo_key(combo: tuple[int, ...]) -> str:
    return "".join(str(i) for i in combo)


def _combos(size: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(SHOT - 1), size))


def _matmul_by_image(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return [query_cell, candidate_image, candidate_cell] cosine blocks."""
    q = _unit_rows(np.asarray(query, dtype=np.float32).reshape(-1, FUSED_DIM))
    blocks = []
    for image_cells in candidate:
        _check_deadline("candidate_matmul")
        b = _unit_rows(image_cells.reshape(-1, FUSED_DIM))
        blocks.append(np.matmul(q, b.T).astype(np.float32, copy=False))
    return np.stack(blocks, axis=1)


def _subset_score(sim: np.ndarray, combo: tuple[int, ...]) -> np.ndarray:
    idx = list(combo)
    selected = sim[:, idx, :].reshape(sim.shape[0], -1)
    return (1.0 - np.max(selected, axis=1)).astype(np.float32, copy=False)


def _facility_coverages(candidate: np.ndarray) -> dict[str, float]:
    """Coverage over the candidate pool, with no held-out input."""
    pool = candidate.reshape(-1, FUSED_DIM)
    sim = _matmul_by_image(pool, candidate)
    out: dict[str, float] = {}
    for size in (1, 2, 3):
        for combo in _combos(size):
            score = _subset_score(sim, combo)
            out[_combo_key(combo)] = float(np.mean(score))
    return out


def _map_native(score: np.ndarray) -> np.ndarray:
    return cv2.resize(np.asarray(score, dtype=np.float32).reshape(GRID, GRID),
                      (RAW_SIZE, RAW_SIZE), interpolation=cv2.INTER_LINEAR)


def _pixel_ap(score: np.ndarray, mask: np.ndarray) -> tuple[float | None, int, float]:
    y = (np.asarray(mask, dtype=np.uint8).reshape(-1) > 0).astype(np.uint8)
    positive = int(y.sum())
    prevalence = float(y.mean())
    if positive == 0 or positive == y.size:
        return None, positive, prevalence
    return float(average_precision_score(y, np.asarray(score, dtype=np.float32).reshape(-1))), positive, prevalence


def _normal_summary(sim: np.ndarray, strategy_combos: dict[str, tuple[int, ...]],
                    n_clean: int, n_nui: int) -> dict[str, dict[str, float | None]]:
    """Patch-level AUC on fixed 4-stride clean/nuisance cells."""
    values: dict[str, dict[str, float | None]] = {}
    labels = np.concatenate([
        np.zeros(n_clean, dtype=np.uint8),
        np.ones(n_nui, dtype=np.uint8),
    ])
    for name, combo in strategy_combos.items():
        scores = _subset_score(sim, combo)
        try:
            auc = float(roc_auc_score(labels, scores))
        except ValueError:
            auc = None
        clean_scores = scores[:n_clean]
        nui_scores = scores[n_clean:]
        values[name] = {
            "nuisance_auc": auc,
            "clean_mean_score": float(np.mean(clean_scores)) if clean_scores.size else None,
            "nuisance_mean_score": float(np.mean(nui_scores)) if nui_scores.size else None,
            "nuisance_minus_clean": (float(np.mean(nui_scores) - np.mean(clean_scores))
                                      if clean_scores.size and nui_scores.size else None),
        }
    return values


def _strategy_specs(facility: dict[str, float]) -> tuple[dict[str, tuple[int, ...]], dict[str, list[tuple[int, ...]]]]:
    """Return deterministic selected combos and all combos for random means."""
    specs: dict[str, tuple[int, ...]] = {
        "facility_k1": min(_combos(1), key=lambda c: (facility[_combo_key(c)], c)),
        "facility_k2": min(_combos(2), key=lambda c: (facility[_combo_key(c)], c)),
        "fixed_first_k1": (0,),
        "fixed_first_k2": (0, 1),
        "all3": (0, 1, 2),
    }
    random_specs = {
        "uniform_combo_mean_k1": _combos(1),
        "uniform_combo_mean_k2": _combos(2),
    }
    return specs, random_specs


def _strategy_coverage(name: str, combo: tuple[int, ...], facility: dict[str, float]) -> float | None:
    key = _combo_key(combo)
    if key in facility:
        return float(facility[key])
    return None


def _evaluate_fold(data: dict[str, Any], held_out: int) -> dict[str, Any]:
    _check_deadline(f"before_fold_{data['category']}_{held_out}")
    refs = data["refs"]
    candidate_indices = [i for i in range(SHOT) if i != held_out]
    # The selector receives only these three arrays.  The held-out index is
    # never passed to _facility_coverages or _strategy_specs.
    candidate = data["clean"][candidate_indices]
    candidate_refs = [refs[i] for i in candidate_indices]
    if held_out in candidate_indices:
        raise AssertionError("held-out image entered candidate pool")
    select_start = time.perf_counter()
    facility = _facility_coverages(candidate)
    selected_specs, random_specs = _strategy_specs(facility)
    selection_ms = (time.perf_counter() - select_start) * 1000.0

    # For normal stability, use all 15 nuisance keys but a fixed 4-stride cell
    # lattice (16 cells per image) to keep the CPU probe within the time budget.
    clean_grid = data["clean"][held_out].reshape(GRID, GRID, FUSED_DIM)
    nui_grid = data["nui"][held_out].reshape(-1, GRID, GRID, FUSED_DIM)
    normal_cells = np.concatenate([
        clean_grid[::NUISANCE_CELL_STEP, ::NUISANCE_CELL_STEP].reshape(-1, FUSED_DIM),
        nui_grid[:, ::NUISANCE_CELL_STEP, ::NUISANCE_CELL_STEP].reshape(-1, FUSED_DIM),
    ], axis=0)
    normal_sim = _matmul_by_image(normal_cells, candidate)
    n_clean = (GRID // NUISANCE_CELL_STEP) ** 2
    n_nui = len(data["nui_keys"]) * n_clean
    normal_combo_specs = dict(selected_specs)
    normal_combo_specs.update({name: combos[0] for name, combos in random_specs.items()})
    # Compute exact normal nuisance metrics for every combination, then take the
    # exact uniform mean for the two random-combination controls.
    normal_by_combo: dict[str, dict[str, float | None]] = {}
    all_normal_combos = [(0, 1, 2)] + _combos(1) + _combos(2)
    for combo in all_normal_combos:
        normal_by_combo[_combo_key(combo)] = _normal_summary(
            normal_sim, {"combo": combo}, n_clean, n_nui
        )["combo"]

    strategy_normal: dict[str, dict[str, Any]] = {}
    for name, combo in selected_specs.items():
        strategy_normal[name] = {"selected_indices": list(combo),
                                 "selected_source_indices": [candidate_indices[i] for i in combo],
                                 **normal_by_combo[_combo_key(combo)]}
    for name, combos in random_specs.items():
        rows = [normal_by_combo[_combo_key(c)] for c in combos]
        strategy_normal[name] = {
            "combination_count": len(combos),
            "selected_indices": None,
            "selected_source_indices": None,
            "nuisance_auc": float(np.mean([r["nuisance_auc"] for r in rows
                                            if r["nuisance_auc"] is not None])) if rows else None,
            "clean_mean_score": float(np.mean([r["clean_mean_score"] for r in rows
                                                if r["clean_mean_score"] is not None])) if rows else None,
            "nuisance_mean_score": float(np.mean([r["nuisance_mean_score"] for r in rows
                                                   if r["nuisance_mean_score"] is not None])) if rows else None,
            "nuisance_minus_clean": float(np.mean([r["nuisance_minus_clean"] for r in rows
                                                    if r["nuisance_minus_clean"] is not None])) if rows else None,
        }

    fold = {
        "category": data["category"],
        "shot": SHOT,
        "seed": SEED,
        "held_out_index": held_out,
        "held_out_rel": refs[held_out],
        "candidate_indices": candidate_indices,
        "candidate_rel": candidate_refs,
        "grid": [GRID, GRID],
        "fused_feature_dim": FUSED_DIM,
        "cells_per_selected_image": CELL_COUNT,
        "selection": {
            "objective": "mean over candidate-pool cells of min(1 - cosine) to selected image cells",
            "candidate_pool_cells": int(candidate.shape[0] * CELL_COUNT),
            "coverage_by_combo": facility,
            "facility_k1_indices": list(selected_specs["facility_k1"]),
            "facility_k1_source_indices": [candidate_indices[i] for i in selected_specs["facility_k1"]],
            "facility_k1_rel": [candidate_refs[i] for i in selected_specs["facility_k1"]],
            "facility_k2_indices": list(selected_specs["facility_k2"]),
            "facility_k2_source_indices": [candidate_indices[i] for i in selected_specs["facility_k2"]],
            "facility_k2_rel": [candidate_refs[i] for i in selected_specs["facility_k2"]],
            "selection_ms": float(selection_ms),
        },
        "strategies": {},
        "normal_stability": strategy_normal,
    }

    # Evaluate all three fixed synthetic variants.  Candidate selection and
    # facility coverage are already complete before the held-out mask is read.
    for family_index, family in enumerate(FAMILIES):
        family_start = family_index * data["syn_count"]
        for seed in SYN_SEEDS:
            _check_deadline(f"before_synthetic_{data['category']}_{held_out}_{family}_{seed}")
            query = data["syn"][held_out, family_start + seed]
            sim = _matmul_by_image(query, candidate)
            mask = data["masks"][held_out, family_start + seed]
            if np.asarray(mask).shape != (RAW_SIZE, RAW_SIZE):
                raise ValueError(f"{data['category']}: expected native mask, got {mask.shape}")
            per_combo: dict[str, dict[str, Any]] = {}
            combos = [(0, 1, 2)] + _combos(1) + _combos(2)
            for combo in combos:
                score_grid = _subset_score(sim, combo).reshape(GRID, GRID)
                score_map = _map_native(score_grid)
                ap, positive, prevalence = _pixel_ap(score_map, mask)
                per_combo[_combo_key(combo)] = {
                    "indices": list(combo),
                    "candidate_rel": [candidate_refs[i] for i in combo],
                    "memory_images": len(combo),
                    "memory_cells": len(combo) * CELL_COUNT,
                    "pixel_ap_native": ap,
                    "positive_pixels": positive,
                    "mask_prevalence": prevalence,
                    "valid_ap": ap is not None,
                }
            # Attach one family/seed row to every strategy.  For uniform means,
            # use the exact mean across all equal-budget combinations.
            for name, combo in selected_specs.items():
                key = _combo_key(combo)
                strategy = fold["strategies"].setdefault(name, {
                    "memory_images": len(combo), "memory_cells": len(combo) * CELL_COUNT,
                    "selected_indices": list(combo),
                    "selected_source_indices": [candidate_indices[i] for i in combo],
                    "selected_rel": [candidate_refs[i] for i in combo],
                    "synthetic": {},
                })
                strategy["synthetic"].setdefault(family, {})[f"seed_{seed}"] = per_combo[key]
            for name, size in (("uniform_combo_mean_k1", 1), ("uniform_combo_mean_k2", 2)):
                choices = _combos(size)
                rows = [per_combo[_combo_key(c)] for c in choices]
                key = f"seed_{seed}"
                strategy = fold["strategies"].setdefault(name, {
                    "memory_images": size, "memory_cells": size * CELL_COUNT,
                    "selected_indices": None, "selected_rel": None,
                    "selected_source_indices": None,
                    "combination_count": len(choices), "combination_rel": [
                        [candidate_refs[i] for i in c] for c in choices],
                    "synthetic": {},
                })
                aps = [r["pixel_ap_native"] for r in rows if r["pixel_ap_native"] is not None]
                positives = [r["positive_pixels"] for r in rows]
                strategy["synthetic"].setdefault(family, {})[key] = {
                    "pixel_ap_native": float(np.mean(aps)) if aps else None,
                    "positive_pixels_min": int(min(positives)) if positives else None,
                    "positive_pixels_max": int(max(positives)) if positives else None,
                    "valid_combination_count": len(aps),
                    "combination_count": len(rows),
                }
            # all3 is included in selected_specs and therefore already attached.

    # Add family means and fold-level macro values after all fixed variants.
    for name, strategy in fold["strategies"].items():
        family_means: dict[str, float | None] = {}
        for family in FAMILIES:
            vals = [row["pixel_ap_native"] for row in strategy["synthetic"].get(family, {}).values()
                    if row.get("pixel_ap_native") is not None]
            family_means[family] = float(np.mean(vals)) if vals else None
        strategy["family_mean_ap_native"] = family_means
        valid = [v for v in family_means.values() if v is not None]
        strategy["macro_ap_native"] = float(np.mean(valid)) if valid else None
        strategy["effective_budget"] = {
            "selected_images": strategy["memory_images"],
            "memory_cells": strategy["memory_cells"],
            "grid": [GRID, GRID],
        }
    return fold


def _aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"completed_folds": len(folds), "strategies": {}}
    for name in STRATEGIES:
        rows = [f["strategies"][name] for f in folds if name in f.get("strategies", {})]
        fam = {}
        for family in FAMILIES:
            vals = [r["family_mean_ap_native"].get(family) for r in rows
                    if r["family_mean_ap_native"].get(family) is not None]
            fam[family] = float(np.mean(vals)) if vals else None
        macros = [r["macro_ap_native"] for r in rows if r.get("macro_ap_native") is not None]
        aucs = [r.get("nuisance_auc") for f in folds
                for rname, r in f.get("normal_stability", {}).items()
                if rname == name and r.get("nuisance_auc") is not None]
        covers = []
        selection_times = [float(f["selection"]["selection_ms"]) for f in folds
                           if f.get("selection", {}).get("selection_ms") is not None]
        for f in folds:
            s = f["strategies"].get(name)
            if not s:
                continue
            combo = s.get("selected_indices")
            if combo is not None:
                covers.append(f["selection"]["coverage_by_combo"].get(_combo_key(tuple(combo))))
            elif name.startswith("uniform_combo_mean"):
                size = 1 if name.endswith("k1") else 2
                keys = [_combo_key(c) for c in _combos(size)]
                covers.extend([f["selection"]["coverage_by_combo"][k] for k in keys])
        out["strategies"][name] = {
            "fold_count": len(rows),
            "macro_ap_native": float(np.mean(macros)) if macros else None,
            "family_mean_ap_native": fam,
            "normal_nuisance_auc": float(np.mean(aucs)) if aucs else None,
            "mean_coverage_1_minus_cosine": float(np.mean(covers)) if covers else None,
            "selection_cost_ms_mean": float(np.mean(selection_times)) if selection_times else None,
            "selection_cost_ms_p95": float(np.percentile(selection_times, 95)) if selection_times else None,
            "memory_images": rows[0].get("memory_images") if rows else None,
            "memory_cells": rows[0].get("memory_cells") if rows else None,
        }
    # Descriptive deltas only; there is no paper gate in this exploratory probe.
    for budget in (1, 2):
        f = out["strategies"].get(f"facility_k{budget}", {})
        u = out["strategies"].get(f"uniform_combo_mean_k{budget}", {})
        fixed = out["strategies"].get(f"fixed_first_k{budget}", {})
        f["delta_macro_vs_uniform"] = (f.get("macro_ap_native") - u.get("macro_ap_native")
                                        if f.get("macro_ap_native") is not None and u.get("macro_ap_native") is not None else None)
        f["delta_macro_vs_fixed_first"] = (f.get("macro_ap_native") - fixed.get("macro_ap_native")
                                            if f.get("macro_ap_native") is not None and fixed.get("macro_ap_native") is not None else None)
    return out


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "status": "pre_registered_support_only_probe",
        "seed": SEED,
        "shot": SHOT,
        "categories": list(args.categories),
        "grid": [GRID, GRID],
        "fused_feature_dim": FUSED_DIM,
        "cache": str(CACHE),
        "support_rule": "v14 manifest seed 0 k4 ref_rel; each fold holds one ref out",
        "candidate_rule": "the three remaining clean support images only",
        "facility_objective": "candidate-pool all 3*256 cells -> selected image subset mean min(1-cosine)",
        "selection_budgets": [1, 2],
        "controls": ["uniform_combo_mean", "fixed_first", "all3"],
        "synthetic_families": list(FAMILIES),
        "synthetic_seeds": list(SYN_SEEDS),
        "synthetic_mask": "native 1024 x 1024; no grid threshold or mask downsample",
        "normal_nuisance_keys": "all 15 cache keys; fixed 4-stride 16-cell lattice per image",
        "selector_inputs": ["candidate clean fused features", "candidate ref_rel order"],
        "selector_forbidden_inputs": ["held-out feature", "held-out mask", "synthetic query", "any test path or label"],
        "memory_rule": "all 256 cells of each selected image; no patch deletion",
        "cpu": {"cuda_visible_devices": "", "torch_threads": int(args.torch_threads), "opencv_threads": 1},
        "hard_deadline_utc": DEADLINE.isoformat(),
        "claim_boundary": "exploratory support proxy only; no paper confirmation",
        "started_utc": _utc_now(),
    }


def _write_partial(protocol: dict[str, Any], folds: list[dict[str, Any]], status: str,
                   error: str | None = None) -> None:
    _write_json(OUT / "PARTIAL_RESULTS.json", {
        "protocol": protocol,
        "status": status,
        "error": error,
        "updated_utc": _utc_now(),
        "completed_folds": len(folds),
        "folds": folds,
        "summary": _aggregate(folds),
    })


def _write_final(protocol: dict[str, Any], folds: list[dict[str, Any]], status: str,
                 error: str | None, elapsed_s: float) -> None:
    summary = _aggregate(folds)
    if status != "complete":
        decision = "PARTIAL_STOP; no support-selection claim"
    else:
        f1 = summary["strategies"].get("facility_k1", {}).get("delta_macro_vs_uniform")
        f2 = summary["strategies"].get("facility_k2", {}).get("delta_macro_vs_uniform")
        if f1 is not None and f2 is not None and f1 > 0 and f2 > 0:
            decision = "CANDIDATE_FACILITY_SIGNAL_ONLY; support proxy, no paper confirmation"
        else:
            decision = "NO_ROBUST_FACILITY_SIGNAL_IN_THIS_PROBE"
    result = {
        "protocol": protocol,
        "status": status,
        "decision": decision,
        "error": error,
        "finished_utc": _utc_now(),
        "elapsed_s": float(elapsed_s),
        "summary": summary,
        "folds": folds,
    }
    _write_json(OUT / "RESULTS.json", result)
    _write_json(OUT / "RUN_MANIFEST.json", {
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "input_cache": str(CACHE.resolve()),
        "status": status,
        "completed_folds": len(folds),
        "categories": list(protocol["categories"]),
        "shot": SHOT,
        "seed": SEED,
        "hard_deadline_utc": DEADLINE.isoformat(),
        "claim_boundary": protocol["claim_boundary"],
    })
    lines = [
        "# Support 图级 facility coverage 选择：首轮结果",
        "",
        f"状态：`{status}`；决定：`{decision}`。",
        "",
        "本轮只使用 v14 k4、seed 0 的 normal support cache。每折留出一张 support 图，选择器只读取其余三张 clean 图的全部 16 × 16 fused cells；held-out feature 和 mask 只在选择完成后进入评价。",
        "",
        "facility 的单位是整张图，目标是候选池全部 patch 到被选图 patch bank 的平均 `1 − cosine` 距离。T4 C2 的 farthest-point 是 patch 行级 coreset，本轮不删除选中图内的 patch。",
        "",
        "AP 是 16 × 16 score grid 双线性回到原始 1024 × 1024 mask 的 Pixel-AP。thin_scratch 若有原像素正像素则保留；没有正像素只记 invalid，不因 grid 可见性人为填分。normal nuisance AUC 使用固定 4-stride cell lattice，只作稳定性诊断。",
        "",
        f"完成 folds：`{len(folds)}`；墙钟时间：`{elapsed_s:.3f} s`。逐折选择路径、预算和 per-family/per-seed 结果见 `RESULTS.json`。",
        "",
        "总体摘要（原像素 mask AP；每张图 256 cells）：",
        "",
        f"- facility k 1：macro `{summary['strategies'].get('facility_k1', {}).get('macro_ap_native')}`；相对等预算组合均值 `Δ = {summary['strategies'].get('facility_k1', {}).get('delta_macro_vs_uniform')}`；normal nuisance AUC `{summary['strategies'].get('facility_k1', {}).get('normal_nuisance_auc')}`；coverage 选择耗时均值 / p 95 `{summary['strategies'].get('facility_k1', {}).get('selection_cost_ms_mean')} / {summary['strategies'].get('facility_k1', {}).get('selection_cost_ms_p95')} ms`。",
        f"- facility k 2：macro `{summary['strategies'].get('facility_k2', {}).get('macro_ap_native')}`；相对等预算组合均值 `Δ = {summary['strategies'].get('facility_k2', {}).get('delta_macro_vs_uniform')}`；normal nuisance AUC `{summary['strategies'].get('facility_k2', {}).get('normal_nuisance_auc')}`；coverage 选择耗时均值 / p 95 `{summary['strategies'].get('facility_k2', {}).get('selection_cost_ms_mean')} / {summary['strategies'].get('facility_k2', {}).get('selection_cost_ms_p95')} ms`。",
        f"- 对照 macro：uniform k 1 `{summary['strategies'].get('uniform_combo_mean_k1', {}).get('macro_ap_native')}`、uniform k 2 `{summary['strategies'].get('uniform_combo_mean_k2', {}).get('macro_ap_native')}`、all 3 `{summary['strategies'].get('all3', {}).get('macro_ap_native')}`。",
        "",
        "逐类别 facility 选择使用 `selected_source_indices`（原始四张 support 的 index；不是 held-out index）：",
        "",
        "| category | k 1 facility − uniform | k 2 facility − uniform | k 1 选图次数 | k 2 选图次数 |",
        "|---|---:|---:|---|---|",
    ]
    import collections
    for category in protocol["categories"]:
        cat_folds = [f for f in folds if f["category"] == category]
        cells = []
        for budget in (1, 2):
            fac = [f["strategies"][f"facility_k{budget}"]["macro_ap_native"] for f in cat_folds]
            uni = [f["strategies"][f"uniform_combo_mean_k{budget}"]["macro_ap_native"] for f in cat_folds]
            delta = float(np.mean(np.asarray(fac) - np.asarray(uni))) if fac else None
            cells.append(delta)
            paths = collections.Counter(tuple(f["strategies"][f"facility_k{budget}"]["selected_source_indices"])
                                         for f in cat_folds)
            cells.append(", ".join(f"{''.join(map(str, k))} × {v}" for k, v in sorted(paths.items())))
        lines.append(f"| {category} | `{cells[0]}` | `{cells[2]}` | {cells[1]} | {cells[3]} |")
    lines += [
        "",
        "边界：即使 facility 在本轮 proxy 中优于对照，也只能作为候选信号；不能写成论文创新已证实，也不能外推到真实 MPDD test。",
    ]
    (OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(args)
    _write_json(OUT / "PROTOCOL.json", protocol)
    folds: list[dict[str, Any]] = []
    started = time.perf_counter()
    status = "complete"
    error = None
    try:
        for cat in args.categories:
            _check_deadline(f"before_category_{cat}")
            data = _load_category(cat)
            for held_out in range(SHOT):
                _check_deadline(f"before_category_fold_{cat}_{held_out}")
                fold = _evaluate_fold(data, held_out)
                folds.append(fold)
                _write_partial(protocol, folds, "running")
                print(json.dumps({"category": cat, "held_out": held_out,
                                  "folds": len(folds),
                                  "facility_k1": fold["selection"]["facility_k1_rel"],
                                  "facility_k2": fold["selection"]["facility_k2_rel"],
                                  "selection_ms": fold["selection"]["selection_ms"]},
                                 ensure_ascii=False), flush=True)
            del data
    except DeadlineReached as exc:
        status = "partial_deadline"
        error = str(exc)
    except Exception as exc:  # preserve partial output, then return an error code
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - started
        _write_partial(protocol, folds, status, error)
        _write_final(protocol, folds, status, error, elapsed)
    print(json.dumps({"status": status, "error": error, "completed_folds": len(folds),
                      "elapsed_s": elapsed}, ensure_ascii=False), flush=True)
    return 0 if status in ("complete", "partial_deadline") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default=",".join(CATEGORIES),
                        help="comma-separated categories; defaults to all six")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    args.categories = tuple(x.strip() for x in args.categories.split(",") if x.strip())
    unknown = [x for x in args.categories if x not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    cv2.setNumThreads(1)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
