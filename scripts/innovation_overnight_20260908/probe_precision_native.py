"""Support-only native-resolution precision probe for the frozen A1 bank.

This is the k4 confirmation arm for the earlier precision probe.  It keeps the
native DINO 32 x 32 grid (and resizes CLIP only from 37 x 37 to 32 x 32), so no
16-cell pooling is used.  Each held-out support image is queried against the
other three support images' 1024 clean cells.  Only memory is cast to FP16 or
symmetrically per-channel INT8 and decoded back to float32 before retrieval;
queries stay float32.  Synthetic AP uses the original 1024 x 1024 masks,
including thin_scratch.  No test image, test label, fitting, or runtime
quantized matmul is read or run.

The fixed UTC deadline is checked at safe category, fold, method, and episode
boundaries.  Completed evidence is written before a deadline or error stop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Constrain numerical libraries before importing numpy/torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[2]
OLD_LOADER_DIR = ROOT / "scripts" / "innovation_t3_efficiency_20260905"
sys.path.insert(0, str(OLD_LOADER_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import probe_e1_compress as E1  # noqa: E402  (reuse resize_patches and A1)

try:  # noqa: E402
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "precision_native"
CACHE = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support"
CATEGORIES = ("bracket_black", "bracket_brown", "bracket_white", "connector",
              "metal_plate", "tubes")
SYN_FAMILIES = ("cutpaste", "local_erasure", "thin_scratch")
NUI_FAMILIES = ("exposure", "gamma", "white_balance", "lr_brightness_gradient", "specular_blob")
NUI_INDICES = (0, 3, 6, 9, 12)  # one pre-registered strength-0 variant per family
METHODS = ("control_f32", "fp16_roundtrip", "int8_pc_roundtrip")
SHOT = 4
K_EXPECTED = 4
GRID = 32
CELLS = GRID * GRID
DIM = 1536
MASK_SIZE = 1024
CELL_SIZE = MASK_SIZE // GRID
EPS = np.float32(1e-12)
HARD_DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)

GATES = {
    "fp16": {"ap_delta": -0.005, "worst_ap_delta": -0.02,
             "distance_p99": 0.002, "nn_agreement": 0.995},
    "int8": {"ap_delta": -0.02, "worst_ap_delta": -0.05,
             "distance_p99": 0.02, "nn_agreement": 0.95},
}


class DeadlineReached(RuntimeError):
    """The pre-registered UTC stop was reached at a safe loop boundary."""


def _check_deadline(stage: str) -> None:
    # Keep this literal comparison in the new script; it does not depend on a
    # heartbeat or scheduler firing at exactly 07:00 Beijing time.
    if datetime.now(timezone.utc) >= datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc):
        raise DeadlineReached(stage)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _json_value(value: Any) -> Any:
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
        return _finite(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norm, EPS)


def _load_branch(category: str, branch: str) -> dict[str, Any]:
    path = CACHE / f"v14_p1_support_{branch}_s0_k{SHOT}" / f"{category}.npz"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        nui_keys = tuple(np.asarray(z["nui_keys"]).astype(str).tolist())
        if len(nui_keys) != 15:
            raise ValueError(f"{category}/{branch}: expected 15 nuisance keys, got {len(nui_keys)}")
        return {
            "branch": str(np.asarray(z["branch"]).item()),
            "ref_rel": np.asarray(z["ref_rel"]).astype(str).copy(),
            "clean_feat": np.asarray(z["clean_feat"], dtype=np.float32).copy(),
            "syn_feat": np.asarray(z["syn_feat"], dtype=np.float32).copy(),
            "syn_masks": np.asarray(z["syn_masks"], dtype=np.uint8).copy(),
            # Only the five pre-registered strength-0 nuisance queries are
            # loaded; all 15 keys remain checked and recorded in metadata.
            "nui_feat": np.asarray(z["nui_feat"][:, list(NUI_INDICES)], dtype=np.float32).copy(),
            "syn_kinds": tuple(np.asarray(z["syn_kinds"]).astype(str).tolist()),
            "nui_keys": nui_keys,
        }


def load_cells(category: str) -> dict[str, Any]:
    """Fuse native 32 x 32 DINO/CLIP cells from the k4 support-only cache."""
    dino = _load_branch(category, "dino")
    clip = _load_branch(category, "clip")
    try:
        if dino["branch"] != "dino" or clip["branch"] != "clip":
            raise ValueError(f"{category}: branch metadata mismatch")
        if not np.array_equal(dino["ref_rel"], clip["ref_rel"]):
            raise ValueError(f"{category}: DINO/CLIP reference order mismatch")
        if dino["syn_kinds"] != tuple(SYN_FAMILIES) or clip["syn_kinds"] != tuple(SYN_FAMILIES):
            raise ValueError(f"{category}: unexpected synthetic order")
        expected_nui = tuple(f"{family}_0" for family in NUI_FAMILIES)
        if tuple(dino["nui_keys"][idx] for idx in NUI_INDICES) != expected_nui:
            raise ValueError(f"{category}: unexpected nuisance order {dino['nui_keys']}")
        if dino["nui_keys"] != clip["nui_keys"]:
            raise ValueError(f"{category}: DINO/CLIP nuisance key mismatch")
        k = int(dino["clean_feat"].shape[0])
        if k != K_EXPECTED or dino["clean_feat"].shape[1:3] != (32, 32):
            raise ValueError(f"{category}: expected k4 native DINO clean, got {dino['clean_feat'].shape}")
        if clip["clean_feat"].shape[1:3] != (37, 37):
            raise ValueError(f"{category}: expected native CLIP 37 x 37, got {clip['clean_feat'].shape}")
        if dino["syn_masks"].shape != (k, 9, MASK_SIZE, MASK_SIZE):
            raise ValueError(f"{category}: expected original masks {(k, 9, MASK_SIZE, MASK_SIZE)}, got {dino['syn_masks'].shape}")

        # Native path: DINO stays 32 x 32.  CLIP is only resized from its
        # native 37 x 37 patch lattice to the common 32 x 32 lattice.
        c_clean = E1.resize_patches(clip["clean_feat"], (GRID, GRID))
        c_syn = E1.resize_patches(
            clip["syn_feat"].reshape(-1, *clip["syn_feat"].shape[2:]), (GRID, GRID)
        ).reshape(k, 9, GRID, GRID, 768)
        c_nui = E1.resize_patches(
            clip["nui_feat"].reshape(-1, *clip["nui_feat"].shape[2:]), (GRID, GRID)
        ).reshape(k, len(NUI_INDICES), GRID, GRID, 768)
        d_clean = dino["clean_feat"]
        d_syn = dino["syn_feat"]
        d_nui = dino["nui_feat"]

        clean = np.asarray(E1.a1(d_clean, c_clean), dtype=np.float32).reshape(k, CELLS, DIM)
        syn = np.asarray(E1.a1(
            d_syn.reshape(-1, GRID, GRID, 768), c_syn.reshape(-1, GRID, GRID, 768)
        ), dtype=np.float32).reshape(k, 9, CELLS, DIM)
        nui = np.asarray(E1.a1(
            d_nui.reshape(-1, GRID, GRID, 768), c_nui.reshape(-1, GRID, GRID, 768)
        ), dtype=np.float32).reshape(k, len(NUI_INDICES), CELLS, DIM)
        return {
            "K": k, "clean": clean, "syn": syn, "nui": nui,
            "masks": dino["syn_masks"], "ref_rel": dino["ref_rel"].tolist(),
            "nui_keys": dino["nui_keys"], "selected_nui_keys": expected_nui,
            "syn_kinds": dino["syn_kinds"],
        }
    finally:
        # Do not retain both branch cache copies while a category is evaluated.
        del dino, clip


def _symmetric_pc_int8(bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen symmetric per-channel memory quantization."""
    scale = np.max(np.abs(np.asarray(bank, dtype=np.float32)), axis=0).astype(np.float32) / np.float32(127.0)
    scale = np.maximum(scale, EPS).astype(np.float32)
    quant = np.clip(np.rint(np.asarray(bank, dtype=np.float32) / scale), -127, 127).astype(np.int8)
    return quant, scale


def _prepare(method: str, bank: np.ndarray) -> tuple[dict[str, Any], float]:
    """Prepare memory-only roundtrip representation and record actual bytes."""
    start = time.perf_counter()
    base = _unit_rows(bank)
    rows, dim = base.shape
    control_bytes = rows * dim * 4
    if method == "control_f32":
        stored = base.copy()
        prepared = {"runtime": stored, "storage_bytes": stored.nbytes,
                    "resident_bank_bytes": stored.nbytes, "decoded_bank_bytes": stored.nbytes,
                    "scale_bytes": 0, "rows": rows, "dim": dim, "dtype": "float32"}
    elif method == "fp16_roundtrip":
        stored = base.astype(np.float16, copy=True)
        decoded = _unit_rows(stored.astype(np.float32))
        prepared = {"runtime": decoded, "storage_bytes": stored.nbytes,
                    "resident_bank_bytes": decoded.nbytes, "decoded_bank_bytes": decoded.nbytes,
                    "scale_bytes": 0, "rows": rows, "dim": dim, "dtype": "float16_storage"}
    elif method == "int8_pc_roundtrip":
        quant, scale = _symmetric_pc_int8(base)
        decoded = _unit_rows(quant.astype(np.float32) * scale[None, :])
        prepared = {"runtime": decoded, "storage_bytes": quant.nbytes + scale.nbytes,
                    "resident_bank_bytes": decoded.nbytes, "decoded_bank_bytes": decoded.nbytes,
                    "scale_bytes": scale.nbytes, "rows": rows, "dim": dim,
                    "dtype": "int8_storage+float32_scale"}
    else:  # pragma: no cover - METHODS is fixed
        raise ValueError(f"unknown method: {method}")
    prepared["control_storage_bytes"] = control_bytes
    prepared["storage_ratio_vs_control"] = prepared["storage_bytes"] / control_bytes
    prepared["resident_ratio_vs_control"] = prepared["resident_bank_bytes"] / control_bytes
    return prepared, time.perf_counter() - start


def _score(query: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float32 query against a decoded float32 memory bank."""
    q = _unit_rows(np.asarray(query, dtype=np.float32).reshape(-1, DIM))
    b = np.asarray(bank, dtype=np.float32)
    dots = q @ b.T
    nn = np.argmax(dots, axis=1).astype(np.int32)
    return (1.0 - dots[np.arange(dots.shape[0]), nn]).astype(np.float32), nn


def _score_to_original(score_grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(score_grid, dtype=np.float32), CELL_SIZE, axis=0), CELL_SIZE, axis=1)


def _ap(mask: np.ndarray, score_grid: np.ndarray) -> float | None:
    y = (np.asarray(mask, dtype=np.uint8).reshape(-1) > 0).astype(np.uint8)
    if y.sum() == 0 or y.sum() == y.size:
        return None
    score = _score_to_original(score_grid)
    return float(average_precision_score(y, score.reshape(-1).astype(np.float64)))


def _distance_error(base: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    err = np.abs(np.asarray(candidate, dtype=np.float64) - np.asarray(base, dtype=np.float64))
    if err.size == 0:
        return {"mean": None, "p95": None, "p99": None, "max": None}
    return {"mean": float(np.mean(err)), "p95": float(np.percentile(err, 95)),
            "p99": float(np.percentile(err, 99)), "max": float(np.max(err))}


def _nn_agreement(base_nn: np.ndarray, candidate_nn: np.ndarray) -> float:
    return float(np.mean(np.asarray(base_nn).reshape(-1) == np.asarray(candidate_nn).reshape(-1)))


def _episodes(cells: dict[str, Any], held_out: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for e in range(9):
        episodes.append({
            "id": f"syn_{SYN_FAMILIES[e // 3]}_s{e % 3}", "kind": "synthetic",
            "family": SYN_FAMILIES[e // 3], "query": cells["syn"][held_out, e],
            "mask": cells["masks"][held_out, e], "episode": e,
        })
    episodes.append({"id": "normal_clean", "kind": "normal_clean", "family": "normal",
                     "query": cells["clean"][held_out], "mask": None, "episode": -1})
    for idx, family in enumerate(NUI_FAMILIES):
        episodes.append({"id": f"normal_nui_{family}_0", "kind": "normal_nui",
                         "family": family, "nuisance_key": f"{family}_0",
                         "query": cells["nui"][held_out, idx], "mask": None, "episode": idx})
    return episodes


def _evaluate_rows(category: str, held_out: int, method: str,
                   episodes: list[dict[str, Any]],
                   control_scores: dict[str, tuple[np.ndarray, np.ndarray]],
                   prepared: dict[str, Any], bank_idx: list[int], refs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        _check_deadline(f"before_{category}_h{held_out}_{method}_{episode['id']}")
        candidate_score, candidate_nn = _score(episode["query"], prepared["runtime"])
        base_score, base_nn = control_scores[episode["id"]]
        ap = _ap(episode["mask"], candidate_score.reshape(GRID, GRID)) if episode["mask"] is not None else None
        control_ap = _ap(episode["mask"], base_score.reshape(GRID, GRID)) if episode["mask"] is not None else None
        dist = _distance_error(base_score, candidate_score)
        rows.append({
            "category": category, "shot": SHOT, "held_out": held_out,
            "heldout_ref": refs[held_out], "bank_source_indices": bank_idx,
            "memory_ref_indices": bank_idx, "method": method,
            "episode": int(episode["episode"]), "episode_id": episode["id"],
            "kind": episode["kind"], "family": episode["family"],
            "nuisance_key": episode.get("nuisance_key"),
            "query_cells": int(candidate_score.size), "memory_rows": int(prepared["rows"]),
            "memory_storage_bytes": int(prepared["storage_bytes"]),
            "memory_resident_bank_bytes": int(prepared["resident_bank_bytes"]),
            "memory_decoded_bank_bytes": int(prepared["decoded_bank_bytes"]),
            "memory_scale_bytes": int(prepared["scale_bytes"]),
            "memory_storage_ratio_vs_control": float(prepared["storage_ratio_vs_control"]),
            "memory_resident_ratio_vs_control": float(prepared["resident_ratio_vs_control"]),
            "ap": ap, "control_ap": control_ap,
            "ap_delta": float(ap - control_ap) if ap is not None and control_ap is not None else None,
            "nn_identity_agreement": _nn_agreement(base_nn, candidate_nn),
            "distance_error": dist,
            "mask_source": "v14_support_syn_masks_original_renderer" if episode["mask"] is not None else None,
            "mask_resolution": [MASK_SIZE, MASK_SIZE] if episode["mask"] is not None else None,
            "score_grid": [GRID, GRID], "support_only": True,
        })
    return rows


def _metric(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _error_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [float(r["distance_error"][key]) for r in rows
              for key in ("mean",) if r["distance_error"].get(key) is not None]
    p99 = [float(r["distance_error"]["p99"]) for r in rows if r["distance_error"].get("p99") is not None]
    return {"mean": _metric(values), "p99_mean": _metric(p99),
            "p99_worst": float(np.max(p99)) if p99 else None,
            "max_worst": float(np.max([r["distance_error"]["max"] for r in rows
                                        if r["distance_error"].get("max") is not None])) if any(
                                            r["distance_error"].get("max") is not None for r in rows) else None}


def _summarize_method(rows: list[dict[str, Any]], prep_records: list[dict[str, Any]],
                      score_seconds: list[float]) -> dict[str, Any]:
    syn_rows = [r for r in rows if r["kind"] == "synthetic"]
    normal_rows = [r for r in rows if r["kind"].startswith("normal_")]
    family_metrics: dict[str, Any] = {}
    for family in SYN_FAMILIES:
        fam = [r for r in syn_rows if r["family"] == family and r["ap"] is not None]
        deltas = [float(r["ap_delta"]) for r in fam if r["ap_delta"] is not None]
        family_metrics[family] = {
            "valid_episodes": len(fam), "candidate_ap": _metric([float(r["ap"]) for r in fam]),
            "control_ap": _metric([float(r["control_ap"]) for r in fam if r["control_ap"] is not None]),
            "ap_delta": _metric(deltas),
            "worst_episode_ap_delta": float(np.min(deltas)) if deltas else None,
            "nn_identity_agreement": _metric([float(r["nn_identity_agreement"]) for r in fam]),
            "distance_error": _error_summary(fam),
            "memory_storage_ratio_vs_control": _metric([float(r["memory_storage_ratio_vs_control"]) for r in fam]),
        }
    all_deltas = [float(r["ap_delta"]) for r in syn_rows if r["ap_delta"] is not None]
    return {
        "family_metrics": family_metrics,
        "valid_synthetic_episodes": sum(v["valid_episodes"] for v in family_metrics.values()),
        "macro_ap": _metric([float(r["ap"]) for r in syn_rows if r["ap"] is not None]),
        "control_macro_ap": _metric([float(r["control_ap"]) for r in syn_rows if r["control_ap"] is not None]),
        "macro_ap_delta": _metric(all_deltas),
        "worst_episode_ap_delta": float(np.min(all_deltas)) if all_deltas else None,
        "nn_identity_agreement": _metric([float(r["nn_identity_agreement"]) for r in rows]),
        "distance_error": _error_summary(rows),
        "normal_score_error": _error_summary(normal_rows),
        "normal_query_count": len(normal_rows),
        "memory": {
            "rows": int(np.max([r["memory_rows"] for r in rows])) if rows else None,
            "storage_bytes": int(np.mean([r["memory_storage_bytes"] for r in rows])) if rows else None,
            "resident_bank_bytes": int(np.mean([r["memory_resident_bank_bytes"] for r in rows])) if rows else None,
            "decoded_bank_bytes": int(np.mean([r["memory_decoded_bank_bytes"] for r in rows])) if rows else None,
            "scale_bytes": int(np.mean([r["memory_scale_bytes"] for r in rows])) if rows else None,
            "storage_ratio_vs_control": _metric([float(r["memory_storage_ratio_vs_control"]) for r in rows]),
            "resident_ratio_vs_control": _metric([float(r["memory_resident_ratio_vs_control"]) for r in rows]),
        },
        "score_seconds": float(np.sum(score_seconds)) if score_seconds else None,
        "episode_rows": len(rows),
    }


def _category_result(category: str, cells: dict[str, Any], rows: list[dict[str, Any]],
                     folds: list[dict[str, Any]], prep_records: dict[str, list[dict[str, Any]]],
                     score_records: dict[str, list[float]], complete: bool) -> dict[str, Any]:
    methods = {method: _summarize_method(
        [r for r in rows if r["method"] == method], prep_records[method], score_records[method]
    ) for method in METHODS}
    return {
        "category": category, "shot": SHOT, "seed": 0, "k": int(cells["K"]),
        "grid": [GRID, GRID], "dimension": DIM, "heldout_folds_completed": len(folds),
        "complete": bool(complete), "selected_nui_keys": list(cells["selected_nui_keys"]),
        "methods": methods, "folds": folds, "rows": rows,
    }


def run_category(category: str) -> dict[str, Any]:
    cells: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    prep_records: dict[str, list[dict[str, Any]]] = {m: [] for m in METHODS}
    score_records: dict[str, list[float]] = {m: [] for m in METHODS}
    try:
        _check_deadline(f"before_load_{category}")
        cells = load_cells(category)
        _check_deadline(f"after_load_{category}")
        for held_out in range(int(cells["K"])):
            _check_deadline(f"before_fold_{category}_h{held_out}")
            bank_idx = [idx for idx in range(int(cells["K"])) if idx != held_out]
            bank = cells["clean"][bank_idx].reshape(-1, DIM)
            episodes = _episodes(cells, held_out)
            prepared: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                _check_deadline(f"before_prepare_{category}_h{held_out}_{method}")
                obj, prep_seconds = _prepare(method, bank)
                prepared[method] = obj
                prep_records[method].append({
                    "held_out": held_out, "bank_source_indices": bank_idx,
                    "rows": obj["rows"], "storage_bytes": obj["storage_bytes"],
                    "resident_bank_bytes": obj["resident_bank_bytes"],
                    "decoded_bank_bytes": obj["decoded_bank_bytes"], "scale_bytes": obj["scale_bytes"],
                    "storage_ratio_vs_control": obj["storage_ratio_vs_control"],
                    "resident_ratio_vs_control": obj["resident_ratio_vs_control"],
                    "prepare_seconds": prep_seconds,
                })
            control_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            t_control = time.perf_counter()
            for episode in episodes:
                _check_deadline(f"before_score_{category}_h{held_out}_control_{episode['id']}")
                control_scores[episode["id"]] = _score(episode["query"], prepared["control_f32"]["runtime"])
            score_records["control_f32"].append(time.perf_counter() - t_control)
            fold_rows: list[dict[str, Any]] = []
            for method in METHODS:
                _check_deadline(f"before_eval_{category}_h{held_out}_{method}")
                if method == "control_f32":
                    method_rows = _evaluate_rows(category, held_out, method, episodes,
                                                 control_scores, prepared[method], bank_idx, cells["ref_rel"])
                else:
                    t_score = time.perf_counter()
                    method_rows = _evaluate_rows(category, held_out, method, episodes,
                                                 control_scores, prepared[method], bank_idx, cells["ref_rel"])
                    score_records[method].append(time.perf_counter() - t_score)
                fold_rows.extend(method_rows)
            # The control score loop is also the control evaluation time; control
            # rows are generated from those same measured scores.
            rows.extend(fold_rows)
            fold_summary = {
                "category": category, "shot": SHOT, "held_out": held_out,
                "bank_source_indices": bank_idx, "heldout_ref": cells["ref_rel"][held_out],
                "methods": {method: _summarize_method(
                    [r for r in fold_rows if r["method"] == method],
                    [prep_records[method][-1]],
                    [score_records[method][-1]] if score_records[method] else [],
                ) for method in METHODS},
            }
            folds.append(fold_summary)
            del bank, prepared, episodes, fold_rows, control_scores
        result = _category_result(category, cells, rows, folds, prep_records, score_records, True)
        return result
    except DeadlineReached as exc:
        if cells is not None:
            exc.partial = _category_result(category, cells, rows, folds, prep_records, score_records, False)
        raise
    except Exception as exc:
        if cells is not None:
            exc.partial = _category_result(category, cells, rows, folds, prep_records, score_records, False)
        raise
    finally:
        if cells is not None:
            del cells


def _aggregate(category_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"category_count": len(category_results),
                           "categories": [r["category"] for r in category_results], "methods": {}}
    for method in METHODS:
        records = [r["methods"][method] for r in category_results if method in r.get("methods", {})]
        family_metrics: dict[str, Any] = {}
        for family in SYN_FAMILIES:
            fams = [r["family_metrics"][family] for r in records
                    if r["family_metrics"][family].get("ap_delta") is not None]
            family_metrics[family] = {
                "valid_episodes": int(np.sum([x["valid_episodes"] for x in fams])) if fams else 0,
                "candidate_ap": _metric([float(x["candidate_ap"]) for x in fams if x["candidate_ap"] is not None]),
                "control_ap": _metric([float(x["control_ap"]) for x in fams if x["control_ap"] is not None]),
                "ap_delta": _metric([float(x["ap_delta"]) for x in fams]),
                "worst_episode_ap_delta": float(np.min([x["worst_episode_ap_delta"] for x in fams
                                                         if x["worst_episode_ap_delta"] is not None])) if any(
                                                             x["worst_episode_ap_delta"] is not None for x in fams) else None,
                "nn_identity_agreement": _metric([float(x["nn_identity_agreement"]) for x in fams
                                                  if x["nn_identity_agreement"] is not None]),
                "distance_error_p99_worst": float(np.max([x["distance_error"]["p99_worst"] for x in fams
                                                           if x["distance_error"].get("p99_worst") is not None])) if any(
                                                               x["distance_error"].get("p99_worst") is not None for x in fams) else None,
                "memory_storage_ratio_vs_control": _metric([float(x["memory_storage_ratio_vs_control"]) for x in fams
                                                             if x["memory_storage_ratio_vs_control"] is not None]),
            }
        out["methods"][method] = {
            "family_metrics": family_metrics,
            "macro_ap": _metric([float(r["macro_ap"]) for r in records if r["macro_ap"] is not None]),
            "control_macro_ap": _metric([float(r["control_macro_ap"]) for r in records if r["control_macro_ap"] is not None]),
            "macro_ap_delta": _metric([float(r["macro_ap_delta"]) for r in records if r["macro_ap_delta"] is not None]),
            "worst_episode_ap_delta": float(np.min([r["worst_episode_ap_delta"] for r in records
                                                     if r["worst_episode_ap_delta"] is not None])) if any(
                                                         r["worst_episode_ap_delta"] is not None for r in records) else None,
            "nn_identity_agreement": _metric([float(r["nn_identity_agreement"]) for r in records
                                              if r["nn_identity_agreement"] is not None]),
            "distance_error_p99_worst_category": float(np.max([r["distance_error"]["p99_worst"] for r in records
                                                                if r["distance_error"].get("p99_worst") is not None])) if any(
                                                                    r["distance_error"].get("p99_worst") is not None for r in records) else None,
            "normal_score_error": {
                "mean": _metric([float(r["normal_score_error"]["mean"]) for r in records
                                  if r["normal_score_error"].get("mean") is not None]),
                "p99_worst_category": float(np.max([r["normal_score_error"]["p99_worst"] for r in records
                                                     if r["normal_score_error"].get("p99_worst") is not None])) if any(
                                                         r["normal_score_error"].get("p99_worst") is not None for r in records) else None,
            },
            "memory": {
                "rows": _metric([float(r["memory"]["rows"]) for r in records if r["memory"]["rows"] is not None]),
                "storage_bytes": _metric([float(r["memory"]["storage_bytes"]) for r in records if r["memory"]["storage_bytes"] is not None]),
                "resident_bank_bytes": _metric([float(r["memory"]["resident_bank_bytes"]) for r in records if r["memory"]["resident_bank_bytes"] is not None]),
                "scale_bytes": _metric([float(r["memory"]["scale_bytes"]) for r in records if r["memory"]["scale_bytes"] is not None]),
                "storage_ratio_vs_control": _metric([float(r["memory"]["storage_ratio_vs_control"]) for r in records
                                                     if r["memory"].get("storage_ratio_vs_control") is not None]),
                "resident_ratio_vs_control": _metric([float(r["memory"]["resident_ratio_vs_control"]) for r in records
                                                      if r["memory"].get("resident_ratio_vs_control") is not None]),
            },
            "score_seconds": float(np.sum([r["score_seconds"] for r in records if r["score_seconds"] is not None])) if records else None,
            "category_count": len(records),
        }
    return out


def _gate_decisions(aggregate: dict[str, Any], status: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "complete_status": status == "complete",
        "six_categories": aggregate.get("category_count") == len(CATEGORIES),
        "no_failures": not failures,
        "native32": True,
        "no_runtime_quantized_method": tuple(METHODS) == ("control_f32", "fp16_roundtrip", "int8_pc_roundtrip"),
    }
    for method in ("fp16_roundtrip", "int8_pc_roundtrip"):
        rec = aggregate.get("methods", {}).get(method, {})
        spec = GATES["fp16" if method.startswith("fp16") else "int8"]
        quality = {
            "macro_ap": rec.get("macro_ap_delta") is not None and rec["macro_ap_delta"] >= spec["ap_delta"],
            "worst_episode_ap": rec.get("worst_episode_ap_delta") is not None and rec["worst_episode_ap_delta"] >= spec["worst_ap_delta"],
            "distance_p99": rec.get("distance_error_p99_worst_category") is not None and rec["distance_error_p99_worst_category"] <= spec["distance_p99"],
            "nn_identity": rec.get("nn_identity_agreement") is not None and rec["nn_identity_agreement"] >= spec["nn_agreement"],
        }
        storage_ratio = rec.get("memory", {}).get("storage_ratio_vs_control")
        storage_ok = storage_ratio is not None and storage_ratio <= (0.51 if method.startswith("fp16") else 0.30)
        checks[method] = {"quality_checks": quality, "quality_gate": bool(all(quality.values())),
                          "storage_ratio": storage_ratio, "storage_gate": bool(storage_ok),
                          "storage_claim": bool(all(quality.values()) and storage_ok)}
    checks["quality_or_storage_claim"] = any(checks[m]["storage_claim"] for m in ("fp16_roundtrip", "int8_pc_roundtrip"))
    return checks


def _decision(status: str, failures: list[dict[str, Any]], gates: dict[str, Any]) -> str:
    if status == "partial_deadline":
        return "PRECISION_NATIVE_PARTIAL_NO_CLAIM"
    if status != "complete" or failures:
        return "PRECISION_NATIVE_FAIL_ARCHIVE"
    if not gates.get("six_categories", False):
        return "PRECISION_NATIVE_PARTIAL_NO_CLAIM"
    return "PRECISION_NATIVE_PROBE_PASS" if gates.get("quality_or_storage_claim", False) else "PRECISION_NATIVE_FAIL_QUALITY_ARCHIVE"


def _positive_signal(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Expose raw positive/negative AP deltas without turning noise into a gate."""
    out: dict[str, Any] = {}
    for method in ("fp16_roundtrip", "int8_pc_roundtrip"):
        record = aggregate.get("methods", {}).get(method, {})
        family = record.get("family_metrics", {})
        out[method] = {
            "macro_ap_delta": record.get("macro_ap_delta"),
            "macro_ap_delta_positive": bool(record.get("macro_ap_delta") is not None and record["macro_ap_delta"] > 0.0),
            "family_ap_delta": {name: family.get(name, {}).get("ap_delta") for name in SYN_FAMILIES},
            "family_ap_delta_positive": {name: bool(family.get(name, {}).get("ap_delta") is not None and family[name]["ap_delta"] > 0.0)
                                         for name in SYN_FAMILIES},
        }
    return out


def _fmt(value: Any) -> str:
    v = _finite(value)
    return "NA" if v is None else f"{v:.6f}"


def _write_partial(protocol: dict[str, Any], status: str, categories: list[dict[str, Any]],
                   rows: list[dict[str, Any]], failures: list[dict[str, Any]], error: str | None = None) -> None:
    _write_json(OUT / "PARTIAL_RESULTS.json", {
        "schema_version": 1, "protocol": protocol, "status": status, "error": error,
        "updated_utc": _utc_now(), "categories_completed": [c["category"] for c in categories],
        "failures": failures, "rows": rows,
        "aggregate": _aggregate(categories) if categories else {"category_count": 0, "categories": [], "methods": {}},
    })
    with (OUT / "PER_EPISODE.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_value(row), ensure_ascii=False) + "\n")


def _write_decision_cn(result: dict[str, Any]) -> None:
    aggregate = result.get("aggregate", {})
    lines = ["# Precision native32 探针结论", "",
             f"状态：`{result.get('status')}`；决策：`{result.get('decision')}`。", "",
             "范围：v14 support-only k4，原生 DINO 32 × 32 网格；每折 heldout 1 张、memory 使用其余 3 张 clean 图。未读取 MPDD test、未运行 FP16/INT8 runtime。",
             "FP16 与 INT8 均只量化 memory 并 roundtrip 解码为 float32；query 保持 float32。每族 nuisance 只取固定第 0 强度，作为 normal score-error 查询。", ""]
    if aggregate.get("methods"):
        lines += ["每类汇总的 family AP Δ、NN 一致率、距离 p99 与 bank storage 比例见 `RESULTS.json`；六类宏平均如下：", "",
                  "| 候选 | macro AP Δ | 最差 episode ΔAP | NN 一致率 | 距离 p99 最差类 | normal score error p99 最差类 | bank storage 比 | scale bytes |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            rec = aggregate["methods"].get(method, {})
            lines.append("| {m} | {ap} | {worst} | {nn} | {p99} | {np99} | {ratio} | {scale} |".format(
                m=method, ap=_fmt(rec.get("macro_ap_delta")), worst=_fmt(rec.get("worst_episode_ap_delta")),
                nn=_fmt(rec.get("nn_identity_agreement")), p99=_fmt(rec.get("distance_error_p99_worst_category")),
                np99=_fmt(rec.get("normal_score_error", {}).get("p99_worst_category")),
                ratio=_fmt(rec.get("memory", {}).get("storage_ratio_vs_control")),
                scale=rec.get("memory", {}).get("scale_bytes", "NA")))
        lines += ["", "原始 mask AP 使用固定 32 × 32 score 的 32 × 32 nearest repeat 回到 1024 × 1024；thin_scratch 不因 mask 下采样被删掉。`family_metrics` 在每个 category 和全局 aggregate 中给出三族 AP 差、NN 一致率、距离误差与 bank bytes 比例。", ""]
        signal = result.get("positive_signal", {})
        lines += ["小正信号（仅描述原始 ΔAP，不作为筛选门）：", ""]
        for method in ("fp16_roundtrip", "int8_pc_roundtrip"):
            item = signal.get(method, {})
            fam = item.get("family_ap_delta", {})
            lines.append(f"- `{method}` 宏平均 ΔAP = `{_fmt(item.get('macro_ap_delta'))}`；family ΔAP：" +
                         "，".join(f"{name} {_fmt(fam.get(name))}" for name in SYN_FAMILIES) + "。")
        lines.append("")
    if result.get("error"):
        lines += [f"错误/截止原因：`{result['error']}`", ""]
    lines += ["复现命令：", "", "```powershell", "$env:CUDA_VISIBLE_DEVICES=\"\"", "$env:OMP_NUM_THREADS=\"2\"", "$env:MKL_NUM_THREADS=\"2\"", "$env:OPENBLAS_NUM_THREADS=\"2\"", "python scripts/innovation_overnight_20260908/probe_precision_native.py", "```", "", "`PARTIAL_RESULTS.json` 和 `PER_EPISODE.jsonl` 保留截止或异常前已完成证据。"]
    (OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _protocol(categories: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1, "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": _sha256_file(Path(__file__)), "created_utc": _utc_now(),
        "git_commit": _git_commit(), "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__, "torch_threads_requested": 2,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "seed": 0,
        "shot": SHOT, "expected_k": K_EXPECTED, "categories": categories, "methods": list(METHODS),
        "cache": "outputs/dynamic_fusion/v14_p1_support", "cache_variant": "s0_k4",
        "loader": "scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py::resize_patches,a1",
        "grid": [GRID, GRID], "dino_pooling": "none", "clip_resize": "37_to_32",
        "dimension": DIM, "memory_rule": "heldout one support image excluded; remaining three clean support images only",
        "query_rule": "heldout clean, nine synthetic episodes, and five fixed nuisance strength-0 queries",
        "nuisance_indices": list(NUI_INDICES), "nuisance_selection": "one fixed strength-0 variant per family; no result-dependent choice",
        "synthetic_families": list(SYN_FAMILIES), "synthetic_mask_resolution": [MASK_SIZE, MASK_SIZE],
        "score_to_mask": "32 x 32 nearest repeat", "support_only": True,
        "test_images_read": False, "test_labels_read": False, "runtime_quantized_methods_run": False,
        "quantization": "memory-only; symmetric per-channel INT8 scale max(abs(memory[:,c]))/127; round-to-nearest; query remains float32",
        "hard_deadline_utc": HARD_DEADLINE_UTC.isoformat(),
    }


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cats", default=None, help="comma-separated support categories; default is all six")
    parser.add_argument("--output", default=str(OUT), help="output directory")
    args = parser.parse_args()
    OUT = Path(args.output).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    categories = [x.strip() for x in args.cats.split(",")] if args.cats else list(CATEGORIES)
    unknown = [x for x in categories if x not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")
    protocol = _protocol(categories)
    _write_json(OUT / "PROTOCOL.json", protocol)
    started = time.perf_counter()
    status = "complete"
    error: str | None = None
    category_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    manifest = {"schema_version": 1, "status": "running", "protocol": protocol,
                "started_utc": _utc_now()}
    _write_json(OUT / "RUN_MANIFEST.json", manifest)
    context = threadpool_limits(limits=2) if threadpool_limits is not None else None
    if context is None:
        class _NullContext:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
        context = _NullContext()
    try:
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is visible to torch despite CUDA_VISIBLE_DEVICES empty")
        with context:
            for category in categories:
                try:
                    _check_deadline(f"before_category_{category}")
                    category_result = run_category(category)
                    category_results.append(category_result)
                    all_rows.extend(category_result["rows"])
                    _write_partial(protocol, "running", category_results, all_rows, failures)
                    print(f"done {category}: {category_result['heldout_folds_completed']} folds", flush=True)
                except DeadlineReached as exc:
                    partial = getattr(exc, "partial", None)
                    if partial is not None:
                        category_results.append(partial)
                        all_rows.extend(partial["rows"])
                    status = "partial_deadline"
                    error = str(exc)
                    _write_partial(protocol, status, category_results, all_rows, failures, error)
                    break
                except Exception as exc:
                    partial = getattr(exc, "partial", None)
                    if partial is not None:
                        category_results.append(partial)
                        all_rows.extend(partial["rows"])
                    failures.append({"category": category, "error": repr(exc),
                                     "traceback": traceback.format_exc()})
                    status = "error_partial"
                    error = f"{type(exc).__name__}: {exc}"
                    _write_partial(protocol, status, category_results, all_rows, failures, error)
                    print(f"failed {category}: {exc}", flush=True)
    except DeadlineReached as exc:
        status = "partial_deadline"
        error = str(exc)
        _write_partial(protocol, status, category_results, all_rows, failures, error)
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
        failures.append({"category": "global", "error": repr(exc), "traceback": traceback.format_exc()})
        _write_partial(protocol, status, category_results, all_rows, failures, error)

    elapsed = time.perf_counter() - started
    aggregate = _aggregate(category_results) if category_results else {"category_count": 0, "categories": [], "methods": {}}
    gates = _gate_decisions(aggregate, status, failures)
    decision = _decision(status, failures, gates)
    positive_signal = _positive_signal(aggregate)
    result = {
        "schema_version": 1, "protocol": protocol, "status": status, "decision": decision,
        "error": error, "failures": failures, "elapsed_seconds": elapsed,
        "categories_requested": categories, "categories_completed": [c["category"] for c in category_results if c.get("complete")],
        "categories": [{k: v for k, v in c.items() if k != "rows"} for c in category_results],
        "aggregate": aggregate, "gates": gates, "positive_signal": positive_signal, "rows": all_rows,
        "torch_threads": int(torch.get_num_threads()), "cuda_available": bool(torch.cuda.is_available()),
    }
    _write_json(OUT / "RESULTS.json", result)
    protocol["finished_utc"] = _utc_now()
    protocol["status"] = status
    protocol["elapsed_seconds"] = elapsed
    protocol["rows"] = len(all_rows)
    _write_json(OUT / "PROTOCOL.json", protocol)
    manifest.update({"status": status, "finished_utc": protocol["finished_utc"],
                     "elapsed_seconds": elapsed, "categories_requested": categories,
                     "categories_completed": result["categories_completed"], "rows": len(all_rows),
                     "failures": failures, "decision": decision,
                     "script_sha256": protocol["script_sha256"], "git_commit": protocol["git_commit"],
                     "torch_threads": int(torch.get_num_threads()), "cuda_available": bool(torch.cuda.is_available())})
    _write_json(OUT / "RUN_MANIFEST.json", manifest)
    _write_partial(protocol, status, category_results, all_rows, failures, error)
    _write_decision_cn(result)
    print(json.dumps({"status": status, "decision": decision, "categories": result["categories_completed"],
                      "rows": len(all_rows), "elapsed_seconds": elapsed}, ensure_ascii=False), flush=True)
    return 0 if decision == "PRECISION_NATIVE_PROBE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
