"""Support-only resolution/storage probe for the frozen full448/full896 banks.

The probe pairs the existing full448 and full896 DINO feature caches on the
same k2 clean/synthetic episodes.  It compares three fixed arms: full448
float32, full896 float32, and a full896 memory-only symmetric per-channel INT8
roundtrip.  The query remains float32 in every arm.  Synthetic masks are the
original 1024 x 1024 renderer masks and the score map uses ``src.utils``'s
existing ``dists2map`` implementation.  No test image, label, online encoder,
or quantized runtime path is read or run.

This file deliberately measures persistent bank bytes separately from the
decoded resident/peak bank-array bytes.  It therefore cannot be read as an
end-to-end latency or total-memory result.
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

# Set CPU/GPU policy before importing numerical libraries.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.utils import dists2map  # noqa: E402

try:  # noqa: E402
    import torch
except Exception:  # pragma: no cover - this probe itself only needs numpy
    torch = None

try:  # noqa: E402
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


SCRIPT_VERSION = "resolution_storage_v1"
OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "resolution_storage"
LEGACY_896_DIR = ROOT / "outputs" / "dynamic_fusion" / "overnight_20260908_full896" / "features"
EXTENSION_DIR = ROOT / "outputs" / "dynamic_fusion" / "overnight_20260908_full896_extension" / "features"
V14_448_DIR = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support" / "v14_p1_support_dino_s0_k2"

CATEGORIES = (
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
)
EXTENSION_CATEGORIES = {"bracket_brown", "bracket_white", "connector", "tubes"}
LEGACY_CATEGORIES = {"bracket_black", "metal_plate"}
SYN_FAMILIES = ("thin_scratch", "cutpaste")
SYN_SEED = 0
SHOT = 2
GRID_448 = 32
GRID_896 = 64
DIM = 768
MASK_SIZE = 1024
EPS = np.float32(1e-12)
METHODS = ("full448_f32", "full896_f32", "full896_int8_roundtrip")
HARD_DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)
NO_NEW_CATEGORY_AFTER_UTC = datetime(2026, 9, 7, 22, 30, tzinfo=timezone.utc)


class DeadlineReached(RuntimeError):
    """The explicit 07:00 Beijing hard deadline was reached."""


class NoNewCategoryAfter(RuntimeError):
    """The 06:30 Beijing boundary forbids starting another category."""


class RuntimeBudgetReached(DeadlineReached):
    """The relative 30 minute first-round budget was reached."""


MAX_RUNTIME_SECONDS = 30 * 60
_RUN_STARTED_PERF: float | None = None


def _check_limits(stage: str, *, category_boundary: bool = False) -> None:
    """Check the literal UTC deadlines at safe loop boundaries."""
    if _RUN_STARTED_PERF is not None and time.perf_counter() - _RUN_STARTED_PERF >= MAX_RUNTIME_SECONDS:
        raise RuntimeBudgetReached(stage)
    now = datetime.now(timezone.utc)
    # Keep this direct comparison in the script; it must not depend on a
    # scheduler/heartbeat firing at exactly 07:00 Beijing time.
    if now >= datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc):
        raise DeadlineReached(stage)
    if category_boundary and now >= datetime(2026, 9, 7, 22, 30, tzinfo=timezone.utc):
        raise NoNewCategoryAfter(stage)


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
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _as_strings(value: Any) -> tuple[str, ...]:
    return tuple(np.asarray(value).astype(str).reshape(-1).tolist())


def _family_indices(syn_kinds: Any, syn_seeds: Any, family: str) -> int:
    kinds = _as_strings(syn_kinds)
    if family not in kinds:
        raise ValueError(f"missing synthetic family {family}; got {kinds}")
    seed_array = np.asarray(syn_seeds)
    if seed_array.ndim == 0:
        seed_count = int(seed_array.item())
        seeds = list(range(seed_count))
    else:
        seeds = [int(x) for x in seed_array.reshape(-1).tolist()]
    if SYN_SEED not in seeds:
        raise ValueError(f"missing frozen synthetic seed {SYN_SEED}; got {seeds}")
    return kinds.index(family) * len(seeds) + seeds.index(SYN_SEED)


def _select_synthetic(z: Any, feature_key: str, family: str) -> np.ndarray:
    idx = _family_indices(z["syn_kinds"], z["syn_seeds"], family)
    arr = np.asarray(z[feature_key], dtype=np.float32)
    if arr.ndim != 5 or idx >= arr.shape[1]:
        raise ValueError(f"{feature_key}: unexpected synthetic shape/index {arr.shape}/{idx}")
    return arr[:, idx].copy()


def _select_masks(z: Any, family: str) -> np.ndarray:
    idx = _family_indices(z["syn_kinds"], z["syn_seeds"], family)
    arr = np.asarray(z["syn_masks"], dtype=np.uint8)
    if arr.ndim != 4 or idx >= arr.shape[1]:
        raise ValueError(f"syn_masks: unexpected shape/index {arr.shape}/{idx}")
    return arr[:, idx].copy()


def _load_category(category: str) -> dict[str, Any]:
    """Load one paired category without re-encoding cached images."""
    if category in EXTENSION_CATEGORIES:
        path = EXTENSION_DIR / f"{category}.npz"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        with np.load(path, allow_pickle=False) as z:
            refs = _as_strings(z["ref_rel"])
            clean448 = np.asarray(z["clean_feat_448"], dtype=np.float32)
            clean896 = np.asarray(z["clean_feat_896"], dtype=np.float32)
            syn448 = np.stack(
                [_select_synthetic(z, "syn_feat_448", family) for family in SYN_FAMILIES],
                axis=1,
            )
            syn896 = np.stack(
                [_select_synthetic(z, "syn_feat_896", family) for family in SYN_FAMILIES],
                axis=1,
            )
            masks = np.stack([_select_masks(z, family) for family in SYN_FAMILIES], axis=1)
            if not np.all(np.asarray(z["clean_valid"]) != 0):
                raise ValueError(f"{category}: invalid clean cache row")
            selected_valid = np.asarray(z["syn_valid"], dtype=np.uint8)
            # The source valid matrix follows the source kind/seed order.
            selected_indices = [_family_indices(z["syn_kinds"], z["syn_seeds"], family)
                                for family in SYN_FAMILIES]
            if not np.all(selected_valid[:, selected_indices] != 0):
                raise ValueError(f"{category}: invalid selected synthetic cache row")
        source = "full896_extension_dual_cache"
        mask_source = str(path.relative_to(ROOT))
    elif category in LEGACY_CATEGORIES:
        path896 = LEGACY_896_DIR / f"{category}.npz"
        path448 = V14_448_DIR / f"{category}.npz"
        if not path896.is_file():
            raise FileNotFoundError(str(path896))
        if not path448.is_file():
            raise FileNotFoundError(str(path448))

        # Load and select the 64 x 64 cache first, keeping only seed 0 for this
        # probe.  The old cache contains three seeds per family.
        with np.load(path896, allow_pickle=False) as z896:
            refs896 = _as_strings(z896["ref_rel"])
            clean896 = np.asarray(z896["clean_feat"], dtype=np.float32)
            syn896 = np.stack(
                [_select_synthetic(z896, "syn_feat", family) for family in SYN_FAMILIES],
                axis=1,
            )
            masks = np.stack([_select_masks(z896, family) for family in SYN_FAMILIES], axis=1)
            if not np.all(np.asarray(z896["clean_valid"]) != 0):
                raise ValueError(f"{category}: invalid full896 clean cache row")
            selected_valid = np.asarray(z896["syn_valid"], dtype=np.uint8)
            selected_indices = [_family_indices(z896["syn_kinds"], z896["syn_seeds"], family)
                                for family in SYN_FAMILIES]
            if not np.all(selected_valid[:, selected_indices] != 0):
                raise ValueError(f"{category}: invalid full896 selected synthetic cache row")

        # The legacy full896 cache predates the paired 448/896 extension.  Its
        # existing v14 k2 DINO cache is the registered full448 fallback.  The
        # references and selected renderer masks are checked before use.
        with np.load(path448, allow_pickle=False) as z448:
            refs448 = _as_strings(z448["ref_rel"])
            clean448 = np.asarray(z448["clean_feat"], dtype=np.float32)
            syn448 = np.stack(
                [_select_synthetic(z448, "syn_feat", family) for family in SYN_FAMILIES],
                axis=1,
            )
            masks448 = np.stack([_select_masks(z448, family) for family in SYN_FAMILIES], axis=1)
        if refs896 != refs448:
            raise ValueError(f"{category}: full448/full896 ref_rel mismatch")
        if not np.array_equal(masks, masks448):
            raise ValueError(f"{category}: full448 fallback masks do not match full896 masks")
        refs = refs896
        source = "full896_cache_plus_v14_k2_full448_fallback"
        mask_source = str(path896.relative_to(ROOT))
    else:
        raise ValueError(f"unknown category {category}")

    refs = tuple(refs)
    if len(refs) != SHOT:
        raise ValueError(f"{category}: expected k2 references, got {len(refs)}")
    expected = {
        "clean448": (SHOT, GRID_448, GRID_448, DIM),
        "clean896": (SHOT, GRID_896, GRID_896, DIM),
        "syn448": (SHOT, len(SYN_FAMILIES), GRID_448, GRID_448, DIM),
        "syn896": (SHOT, len(SYN_FAMILIES), GRID_896, GRID_896, DIM),
        "masks": (SHOT, len(SYN_FAMILIES), MASK_SIZE, MASK_SIZE),
    }
    arrays = {
        "clean448": clean448,
        "clean896": clean896,
        "syn448": syn448,
        "syn896": syn896,
        "masks": masks,
    }
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(f"{category}/{name}: expected {shape}, got {arrays[name].shape}")
        if not np.all(np.isfinite(arrays[name])) and name != "masks":
            raise ValueError(f"{category}/{name}: non-finite feature")
    # Renderer masks are kept byte-for-byte as cached (the current caches use
    # 0/255 rather than 0/1); AP converts every positive pixel to a label.
    if not np.all(np.isfinite(masks)):
        raise ValueError(f"{category}: masks contain non-finite values")
    return {
        "category": category,
        "ref_rel": list(refs),
        "clean448": np.ascontiguousarray(clean448, dtype=np.float32),
        "clean896": np.ascontiguousarray(clean896, dtype=np.float32),
        "syn448": np.ascontiguousarray(syn448, dtype=np.float32),
        "syn896": np.ascontiguousarray(syn896, dtype=np.float32),
        "masks": np.ascontiguousarray(masks, dtype=np.uint8),
        "cache_source": source,
        "mask_source": mask_source,
        "synthetic_families": list(SYN_FAMILIES),
        "synthetic_seed": SYN_SEED,
    }


def _unit_rows(value: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32).reshape(-1, DIM)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, EPS)


def _symmetric_pc_int8(bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen symmetric per-channel memory quantization."""
    base = np.asarray(bank, dtype=np.float32)
    scale = np.max(np.abs(base), axis=0).astype(np.float32) / np.float32(127.0)
    scale = np.maximum(scale, EPS).astype(np.float32)
    quant = np.clip(np.rint(base / scale[None, :]), -127, 127).astype(np.int8)
    return quant, scale


def _prepare(method: str, bank: np.ndarray) -> tuple[dict[str, Any], float]:
    """Prepare a fixed memory representation and measure preparation only."""
    started = time.perf_counter()
    base = _unit_rows(bank)
    rows, dim = base.shape
    full32_bytes = int(rows * dim * np.dtype(np.float32).itemsize)
    if method == "full448_f32" or method == "full896_f32":
        stored = np.ascontiguousarray(base, dtype=np.float32)
        prepared = {
            "runtime": stored,
            "persistent_bank_bytes": int(stored.nbytes),
            "persistent_scale_bytes": 0,
            "decoded_resident_bank_bytes": int(stored.nbytes),
            "decode_peak_bank_array_bytes": int(stored.nbytes),
            "storage_dtype": "float32",
        }
    elif method == "full896_int8_roundtrip":
        quant, scale = _symmetric_pc_int8(base)
        decoded = _unit_rows(quant.astype(np.float32) * scale[None, :])
        # Peak is the stored quantized bank + scales plus the decoded runtime
        # bank-array.  It is deliberately not a whole-process RSS measurement.
        prepared = {
            "runtime": np.ascontiguousarray(decoded, dtype=np.float32),
            "persistent_bank_bytes": int(quant.nbytes + scale.nbytes),
            "persistent_scale_bytes": int(scale.nbytes),
            "decoded_resident_bank_bytes": int(decoded.nbytes),
            "decode_peak_bank_array_bytes": int(quant.nbytes + scale.nbytes + decoded.nbytes),
            "storage_dtype": "int8_per_channel+float32_scale_roundtrip",
        }
    else:  # pragma: no cover - METHODS is fixed
        raise ValueError(f"unknown method {method}")
    prepared.update(
        {
            "rows": int(rows),
            "dim": int(dim),
            "full32_bank_bytes": full32_bytes,
            "prepare_seconds": float(time.perf_counter() - started),
        }
    )
    return prepared, float(prepared["prepare_seconds"])


def _nearest_dist(query: np.ndarray, bank: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return sqrt cosine distance and NN row for each float32 query cell."""
    q = _unit_rows(query)
    b = np.asarray(bank, dtype=np.float32)
    out = np.empty(q.shape[0], dtype=np.float32)
    nn = np.empty(q.shape[0], dtype=np.int32)
    for start in range(0, q.shape[0], chunk):
        stop = min(start + chunk, q.shape[0])
        sim = q[start:stop] @ b.T
        best = np.argmax(sim, axis=1)
        nn[start:stop] = best.astype(np.int32)
        out[start:stop] = np.sqrt(
            np.maximum(2.0 - 2.0 * sim[np.arange(stop - start), best], 0.0)
        ).astype(np.float32)
    return out, nn


def _score_map(query: np.ndarray, bank: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native, nn = _nearest_dist(query, bank)
    native_grid = native.reshape(grid, grid)
    score_map = np.asarray(dists2map(native_grid, (MASK_SIZE, MASK_SIZE)), dtype=np.float32)
    return native, nn, score_map


def _ap(mask: np.ndarray, score_map: np.ndarray) -> float | None:
    labels = (np.asarray(mask, dtype=np.uint8).reshape(-1) > 0).astype(np.uint8)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return None
    return float(average_precision_score(labels, np.asarray(score_map, dtype=np.float64).reshape(-1)))


def _stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _metric(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summary_error(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values: list[float] = []
    for row in rows:
        error = row.get(key)
        if isinstance(error, dict) and error.get("p99") is not None:
            values.append(float(error["p99"]))
    return {
        "p99_mean": _metric(values),
        "p99_worst": float(np.max(values)) if values else None,
    }


def _memory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    fields = (
        "native_memory_rows",
        "persistent_bank_bytes",
        "persistent_scale_bytes",
        "decoded_resident_bank_bytes",
        "decode_peak_bank_array_bytes",
        "storage_ratio_vs_full448_f32",
        "storage_ratio_vs_full896_f32",
    )
    return {
        field: _metric([float(row[field]) for row in rows if row.get(field) is not None])
        for field in fields
    }


def _category_summary(category: str, rows: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["arm"] == method]
        synth_rows = [row for row in method_rows if row["query_kind"] == "synthetic"]
        methods[method] = {
            "synthetic_episode_count": len(synth_rows),
            "valid_ap_count": sum(row["ap"] is not None for row in synth_rows),
            "macro_ap": _metric([float(row["ap"]) for row in synth_rows if row["ap"] is not None]),
            "delta_vs_full448_macro_ap": _metric(
                [float(row["ap_delta_vs_full448_f32"]) for row in synth_rows
                 if row["ap_delta_vs_full448_f32"] is not None]
            ),
            "delta_vs_full896_macro_ap": _metric(
                [float(row["ap_delta_vs_full896_f32"]) for row in synth_rows
                 if row["ap_delta_vs_full896_f32"] is not None]
            ),
            "family_metrics": {
                family: {
                    "episode_count": sum(row["family"] == family for row in synth_rows),
                    "macro_ap": _metric([float(row["ap"]) for row in synth_rows
                                          if row["family"] == family and row["ap"] is not None]),
                    "delta_vs_full448_macro_ap": _metric(
                        [float(row["ap_delta_vs_full448_f32"]) for row in synth_rows
                         if row["family"] == family and row["ap_delta_vs_full448_f32"] is not None]
                    ),
                    "delta_vs_full896_macro_ap": _metric(
                        [float(row["ap_delta_vs_full896_f32"]) for row in synth_rows
                         if row["family"] == family and row["ap_delta_vs_full896_f32"] is not None]
                    ),
                    "nn_identity_vs_full896_f32": _metric(
                        [float(row["nn_identity_vs_full896_f32"]) for row in synth_rows
                         if row["family"] == family and row["nn_identity_vs_full896_f32"] is not None]
                    ),
                    "distance_error_vs_full896_f32_p99": _summary_error(
                        [row for row in synth_rows if row["family"] == family],
                        "distance_error_vs_full896_f32",
                    ),
                }
                for family in SYN_FAMILIES
            },
            "nn_identity_vs_full896_f32": _metric(
                [float(row["nn_identity_vs_full896_f32"]) for row in synth_rows
                 if row["nn_identity_vs_full896_f32"] is not None]
            ),
            "distance_error_vs_full896_f32_p99": _summary_error(
                synth_rows, "distance_error_vs_full896_f32"
            ),
            "memory": _memory_summary(method_rows),
            "retrieval_seconds": float(np.sum([row["retrieval_seconds"] for row in method_rows])),
            "encoding_latency_measured": False,
            "runtime_quantized": False,
        }
    return {
        "category": category,
        "shot": SHOT,
        "synthetic_seed": SYN_SEED,
        "synthetic_families": list(SYN_FAMILIES),
        "heldout_folds": 2,
        "synthetic_unique_episode_count": sum(row["arm"] == METHODS[0] for row in rows),
        "complete": bool(complete),
        "methods": methods,
        "rows": rows,
    }


def _aggregate(category_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired synthetic rows with each episode weighted equally."""
    rows = [row for result in category_results for row in result.get("rows", [])]
    out: dict[str, Any] = {
        "category_count": len(category_results),
        "categories": [result["category"] for result in category_results],
        "synthetic_unique_episode_count": sum(
            int(result.get("synthetic_unique_episode_count", 0)) for result in category_results
        ),
        "methods": {},
    }
    for method in METHODS:
        method_rows = [row for row in rows if row["arm"] == method and row["query_kind"] == "synthetic"]
        out["methods"][method] = {
            "synthetic_episode_count": len(method_rows),
            "valid_ap_count": sum(row["ap"] is not None for row in method_rows),
            "macro_ap": _metric([float(row["ap"]) for row in method_rows if row["ap"] is not None]),
            "delta_vs_full448_macro_ap": _metric(
                [float(row["ap_delta_vs_full448_f32"]) for row in method_rows
                 if row["ap_delta_vs_full448_f32"] is not None]
            ),
            "delta_vs_full896_macro_ap": _metric(
                [float(row["ap_delta_vs_full896_f32"]) for row in method_rows
                 if row["ap_delta_vs_full896_f32"] is not None]
            ),
            "family_metrics": {
                family: {
                    "episode_count": sum(row["family"] == family for row in method_rows),
                    "macro_ap": _metric([float(row["ap"]) for row in method_rows
                                          if row["family"] == family and row["ap"] is not None]),
                    "delta_vs_full448_macro_ap": _metric(
                        [float(row["ap_delta_vs_full448_f32"]) for row in method_rows
                         if row["family"] == family and row["ap_delta_vs_full448_f32"] is not None]
                    ),
                    "delta_vs_full896_macro_ap": _metric(
                        [float(row["ap_delta_vs_full896_f32"]) for row in method_rows
                         if row["family"] == family and row["ap_delta_vs_full896_f32"] is not None]
                    ),
                    "nn_identity_vs_full896_f32": _metric(
                        [float(row["nn_identity_vs_full896_f32"]) for row in method_rows
                         if row["family"] == family and row["nn_identity_vs_full896_f32"] is not None]
                    ),
                    "distance_error_vs_full896_f32_p99": _summary_error(
                        [row for row in method_rows if row["family"] == family],
                        "distance_error_vs_full896_f32",
                    ),
                }
                for family in SYN_FAMILIES
            },
            "nn_identity_vs_full896_f32": _metric(
                [float(row["nn_identity_vs_full896_f32"]) for row in method_rows
                 if row["nn_identity_vs_full896_f32"] is not None]
            ),
            "distance_error_vs_full896_f32_p99": _summary_error(
                method_rows, "distance_error_vs_full896_f32"
            ),
            "memory": _memory_summary([row for row in rows if row["arm"] == method]),
            "retrieval_seconds": float(np.sum([
                row["retrieval_seconds"] for row in rows if row["arm"] == method
            ])),
            "encoding_latency_measured": False,
            "runtime_quantized": False,
        }

    low = out["methods"].get("full448_f32", {})
    high = out["methods"].get("full896_f32", {})
    quant = out["methods"].get("full896_int8_roundtrip", {})
    out["paired_question"] = {
        "full896_f32_gain_vs_full448_f32": (
            float(high["macro_ap"] - low["macro_ap"])
            if high.get("macro_ap") is not None and low.get("macro_ap") is not None else None
        ),
        "full896_int8_delta_vs_full448_f32": quant.get("delta_vs_full448_macro_ap"),
        "full896_int8_delta_vs_full896_f32": quant.get("delta_vs_full896_macro_ap"),
        "int8_persistent_bytes_vs_full448_ratio": quant.get("memory", {}).get(
            "storage_ratio_vs_full448_f32"
        ),
        "int8_persistent_bytes_vs_full896_ratio": quant.get("memory", {}).get(
            "storage_ratio_vs_full896_f32"
        ),
        "int8_decoded_resident_equals_full896_f32": (
            quant.get("memory", {}).get("decoded_resident_bank_bytes")
            == high.get("memory", {}).get("decoded_resident_bank_bytes")
            if quant.get("memory") and high.get("memory") else None
        ),
    }
    return out


def _protocol(categories: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": _sha256_file(Path(__file__)),
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__ if torch is not None else None,
        "torch_threads_requested": 2,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "shot": SHOT,
        "categories": categories,
        "methods": list(METHODS),
        "synthetic_families": list(SYN_FAMILIES),
        "synthetic_seed": SYN_SEED,
        "synthetic_unique_episode_count": len(categories) * SHOT * len(SYN_FAMILIES),
        "cache_sources": {
            "legacy_full896": str(LEGACY_896_DIR.relative_to(ROOT)),
            "full896_extension_dual": str(EXTENSION_DIR.relative_to(ROOT)),
            "legacy_full448_fallback": str(V14_448_DIR.relative_to(ROOT)),
        },
        "cache_loader": "numpy npz cached features; no image re-encoding",
        "grids": {"full448_f32": [GRID_448, GRID_448], "full896_f32": [GRID_896, GRID_896],
                  "full896_int8_roundtrip": [GRID_896, GRID_896]},
        "feature_dim": DIM,
        "memory_rule": "heldout one clean image excluded; bank is the other one clean image",
        "query_rule": "same heldout image, synthetic thin_scratch/cutpaste seed 0 only",
        "mask_source": "original renderer syn_masks at 1024 x 1024",
        "map_postprocess": "src.utils.dists2map(native nearest sqrt cosine distance, sigma 4)",
        "int8_rule": "memory-only symmetric per-channel scale=max(abs(memory[:,c]))/127, round-to-nearest, clip [-127,127], decode float32 and normalize",
        "query_dtype": "float32 for all arms",
        "runtime_quantized_methods_run": False,
        "encoding_latency_measured": False,
        "persistent_bytes_include_int8_scale": True,
        "decoded_memory_is_not_saved_as_persistent_bytes": True,
        "test_images_read": False,
        "test_labels_read": False,
        "cpu_only": True,
        "hard_deadline_utc": HARD_DEADLINE_UTC.isoformat(),
        "no_new_category_after_utc": NO_NEW_CATEGORY_AFTER_UTC.isoformat(),
        "max_runtime_minutes": 30,
    }


def _run_category(category: str) -> dict[str, Any]:
    cells: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    try:
        _check_limits(f"before_load_{category}")
        cells = _load_category(category)
        _check_limits(f"after_load_{category}")
        refs = cells["ref_rel"]
        for held_out in range(SHOT):
            _check_limits(f"before_fold_{category}_h{held_out}")
            bank_index = 1 - held_out
            bank448 = cells["clean448"][bank_index].reshape(-1, DIM)
            bank896 = cells["clean896"][bank_index].reshape(-1, DIM)
            prepared: dict[str, dict[str, Any]] = {}
            for method, bank in (
                ("full448_f32", bank448),
                ("full896_f32", bank896),
                ("full896_int8_roundtrip", bank896),
            ):
                _check_limits(f"before_prepare_{category}_h{held_out}_{method}")
                prepared[method], _ = _prepare(method, bank)

            for family_index, family in enumerate(SYN_FAMILIES):
                _check_limits(f"before_episode_{category}_h{held_out}_{family}")
                query448 = cells["syn448"][held_out, family_index]
                query896 = cells["syn896"][held_out, family_index]
                mask = cells["masks"][held_out, family_index]
                episode_id = f"{category}_h{held_out}_{family}_s{SYN_SEED}"
                query_by_method = {
                    "full448_f32": (query448, GRID_448),
                    "full896_f32": (query896, GRID_896),
                    "full896_int8_roundtrip": (query896, GRID_896),
                }
                scored: dict[str, dict[str, Any]] = {}
                for method in METHODS:
                    _check_limits(f"before_score_{episode_id}_{method}")
                    query, grid = query_by_method[method]
                    started = time.perf_counter()
                    native, nn, score_map = _score_map(
                        query, prepared[method]["runtime"], grid
                    )
                    elapsed = float(time.perf_counter() - started)
                    scored[method] = {
                        "native": native,
                        "nn": nn,
                        "score_map": score_map,
                        "ap": _ap(mask, score_map),
                        "retrieval_seconds": elapsed,
                    }

                ap448 = scored["full448_f32"]["ap"]
                ap896 = scored["full896_f32"]["ap"]
                base_native = scored["full896_f32"]["native"]
                base_nn = scored["full896_f32"]["nn"]
                for method in METHODS:
                    _check_limits(f"before_row_{episode_id}_{method}")
                    result = scored[method]
                    is_highres = method != "full448_f32"
                    distance_error = (
                        _stats(np.abs(result["native"] - base_native))
                        if method in {"full896_f32", "full896_int8_roundtrip"} else None
                    )
                    nn_identity = (
                        float(np.mean(result["nn"] == base_nn))
                        if is_highres else None
                    )
                    ap_value = result["ap"]
                    rows.append({
                        "category": category,
                        "shot": SHOT,
                        "held_out": held_out,
                        "heldout_ref": refs[held_out],
                        "memory_source_index": bank_index,
                        "memory_ref": refs[bank_index],
                        "query_kind": "synthetic",
                        "family": family,
                        "synthetic_seed": SYN_SEED,
                        "unique_synthetic_episode": True,
                        "episode_id": episode_id,
                        "arm": method,
                        "grid": [GRID_448, GRID_448] if method == "full448_f32" else [GRID_896, GRID_896],
                        "feature_dim": DIM,
                        "native_memory_rows": prepared[method]["rows"],
                        "persistent_bank_bytes": prepared[method]["persistent_bank_bytes"],
                        "persistent_scale_bytes": prepared[method]["persistent_scale_bytes"],
                        "decoded_resident_bank_bytes": prepared[method]["decoded_resident_bank_bytes"],
                        "decode_peak_bank_array_bytes": prepared[method]["decode_peak_bank_array_bytes"],
                        "storage_ratio_vs_full448_f32": (
                            prepared[method]["persistent_bank_bytes"] / 3145728.0
                        ),
                        "storage_ratio_vs_full896_f32": (
                            prepared[method]["persistent_bank_bytes"] / 12582912.0
                        ),
                        "storage_dtype": prepared[method]["storage_dtype"],
                        "ap": ap_value,
                        "ap_delta_vs_full448_f32": (
                            float(ap_value - ap448)
                            if ap_value is not None and ap448 is not None else None
                        ),
                        "ap_delta_vs_full896_f32": (
                            float(ap_value - ap896)
                            if ap_value is not None and ap896 is not None else None
                        ),
                        "nn_identity_vs_full896_f32": nn_identity,
                        "distance_error_vs_full896_f32": distance_error,
                        "score_native_stats": _stats(result["native"]),
                        "score_map_stats": _stats(result["score_map"]),
                        "mask_source": cells["mask_source"],
                        "mask_resolution": [MASK_SIZE, MASK_SIZE],
                        "map_postprocess": "src.utils.dists2map",
                        "query_dtype": "float32",
                        "support_only": True,
                        "runtime_quantized": False,
                        "encoding_latency_measured": False,
                        "retrieval_seconds": result["retrieval_seconds"],
                        "encode_seconds": None,
                    })
                _check_limits(f"after_episode_{episode_id}")
                del scored
            del prepared, bank448, bank896
        _check_limits(f"after_category_{category}")
        result = _category_summary(category, rows, True)
        result["cache_source"] = cells["cache_source"]
        result["mask_source"] = cells["mask_source"]
        result["references"] = refs
        return result
    except DeadlineReached as exc:
        if cells is not None:
            partial = _category_summary(category, rows, False)
            partial["cache_source"] = cells["cache_source"]
            partial["mask_source"] = cells["mask_source"]
            partial["references"] = cells["ref_rel"]
            exc.partial = partial
        raise
    except Exception as exc:
        if cells is not None:
            partial = _category_summary(category, rows, False)
            partial["cache_source"] = cells["cache_source"]
            partial["mask_source"] = cells["mask_source"]
            partial["references"] = cells["ref_rel"]
            exc.partial = partial
        raise
    finally:
        if cells is not None:
            del cells


def _write_partial(
    protocol: dict[str, Any],
    status: str,
    category_results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    error: str | None = None,
) -> None:
    rows = [row for result in category_results for row in result.get("rows", [])]
    _write_json(
        OUT / "PARTIAL_RESULTS.json",
        {
            "schema_version": 1,
            "protocol": protocol,
            "status": status,
            "error": error,
            "updated_utc": _utc_now(),
            "categories_completed_or_partial": [result["category"] for result in category_results],
            "failures": failures,
            "rows": rows,
            "aggregate": _aggregate(category_results),
        },
    )
    with (OUT / "PER_EPISODE.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_value(row), ensure_ascii=False) + "\n")


def _fmt(value: Any) -> str:
    v = _finite(value)
    return "NA" if v is None else f"{v:.8f}"


def _write_decision_cn(result: dict[str, Any]) -> None:
    aggregate = result.get("aggregate", {})
    methods = aggregate.get("methods", {})
    q = aggregate.get("paired_question", {})
    lines = [
        "# Resolution-storage 组合探针结论",
        "",
        f"状态：`{result.get('status')}`；决策：`{result.get('decision')}`。",
        "",
        "本轮是 support-only 诊断：六类 k2、每个 heldout 图各取 thin_scratch 与 cutpaste 的 seed 0，实际 unique synthetic episodes 为 24；三臂在同一 episode 上配对。未读取 test 图像/标签。",
        "",
        "| arm | macro AP | ΔAP vs full448 FP32 | ΔAP vs full896 FP32 | native memory rows | persistent bank bytes | scale bytes | decoded resident bytes | decode peak bank-array bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        rec = methods.get(method, {})
        mem = rec.get("memory", {})
        lines.append(
            "| {m} | {ap} | {d448} | {d896} | {rows} | {bytes} | {scale} | {resident} | {peak} |".format(
                m=method,
                ap=_fmt(rec.get("macro_ap")),
                d448=_fmt(rec.get("delta_vs_full448_macro_ap")),
                d896=_fmt(rec.get("delta_vs_full896_macro_ap")),
                rows=_fmt(mem.get("native_memory_rows")),
                bytes=_fmt(mem.get("persistent_bank_bytes")),
                scale=_fmt(mem.get("persistent_scale_bytes")),
                resident=_fmt(mem.get("decoded_resident_bank_bytes")),
                peak=_fmt(mem.get("decode_peak_bank_array_bytes")),
            )
        )
    lines += [
        "",
        f"full896 FP32 相对 full448 FP32 的配对宏平均 AP 增益：`{_fmt(q.get('full896_f32_gain_vs_full448_f32'))}`。",
        f"full896 INT8 roundtrip 相对 full448 FP32：`{_fmt(q.get('full896_int8_delta_vs_full448_f32'))}`；相对 full896 FP32：`{_fmt(q.get('full896_int8_delta_vs_full896_f32'))}`。",
        f"INT8 持久化 bank bytes / full448 FP32：`{_fmt(q.get('int8_persistent_bytes_vs_full448_ratio'))}`；/ full896 FP32：`{_fmt(q.get('int8_persistent_bytes_vs_full896_ratio'))}`。INT8 scale 已计入持久化 bytes。",
        f"INT8 相对 full896 FP32 的 NN identity：`{_fmt(methods.get('full896_int8_roundtrip', {}).get('nn_identity_vs_full896_f32'))}`；distance error p99（episode 均值/最差）：`{_fmt(methods.get('full896_int8_roundtrip', {}).get('distance_error_vs_full896_f32_p99', {}).get('p99_mean'))}` / `{_fmt(methods.get('full896_int8_roundtrip', {}).get('distance_error_vs_full896_f32_p99', {}).get('p99_worst'))}`。",
        "",
        "full896 INT8 只在持久化表示上压缩；解码后的 resident bank bytes 与 full896 FP32 相同，decode 峰值还包含 quantized bank、scale 和 decoded bank，因此不能据此声称运行内存降低。编码延迟未测量（缓存复用），也不能外推端到端推理速度、总内存或等成本，更不能把本探针称为新颖算法。",
        "",
        "每类/每族 AP、NN 一致率、距离误差和 memory 字段见 `RESULTS.json` 与 `PER_EPISODE.jsonl`。",
        "",
        "复现命令：",
        "",
        "```powershell",
        "$env:CUDA_VISIBLE_DEVICES=\"\"",
        "$env:OMP_NUM_THREADS=\"2\"",
        "$env:MKL_NUM_THREADS=\"2\"",
        "$env:OPENBLAS_NUM_THREADS=\"2\"",
        "$env:NUMEXPR_NUM_THREADS=\"2\"",
        "python scripts/innovation_overnight_20260908/probe_resolution_storage.py",
        "```",
        "",
        "截止或异常时仍保留 `PARTIAL_RESULTS.json`、`PER_EPISODE.jsonl` 和最终状态字段。",
    ]
    (OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global OUT, _RUN_STARTED_PERF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cats", default=None, help="comma-separated support categories; default is all six")
    parser.add_argument("--output", default=str(OUT), help="output directory")
    args = parser.parse_args()
    OUT = Path(args.output).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    categories = [x.strip() for x in args.cats.split(",")] if args.cats else list(CATEGORIES)
    unknown = [category for category in categories if category not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")
    if len(set(categories)) != len(categories):
        raise SystemExit("duplicate categories are not allowed")

    protocol = _protocol(categories)
    _write_json(OUT / "PROTOCOL.json", protocol)
    started = time.perf_counter()
    _RUN_STARTED_PERF = started
    category_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    status = "complete"
    error: str | None = None
    _write_json(
        OUT / "RUN_MANIFEST.json",
        {"schema_version": 1, "status": "running", "protocol": protocol, "started_utc": _utc_now()},
    )

    try:
        if torch is not None:
            try:
                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            if torch.cuda.is_available():
                raise RuntimeError("CUDA is visible despite CUDA_VISIBLE_DEVICES empty")
        cv2.setNumThreads(1)
        context = threadpool_limits(limits=2) if threadpool_limits is not None else None
        if context is None:
            for category in categories:
                _check_limits(f"before_category_{category}", category_boundary=True)
                category_results.append(_run_category(category))
                _write_partial(protocol, "running", category_results, failures)
                print(f"done {category}", flush=True)
        else:
            with context:
                for category in categories:
                    _check_limits(f"before_category_{category}", category_boundary=True)
                    category_results.append(_run_category(category))
                    _write_partial(protocol, "running", category_results, failures)
                    print(f"done {category}", flush=True)
    except NoNewCategoryAfter as exc:
        status = "partial_no_new_category"
        error = str(exc)
        _write_partial(protocol, status, category_results, failures, error)
    except RuntimeBudgetReached as exc:
        status = "partial_runtime_budget"
        error = str(exc)
        partial = getattr(exc, "partial", None)
        if partial is not None:
            category_results.append(partial)
        _write_partial(protocol, status, category_results, failures, error)
    except DeadlineReached as exc:
        status = "partial_deadline"
        error = str(exc)
        partial = getattr(exc, "partial", None)
        if partial is not None:
            category_results.append(partial)
        _write_partial(protocol, status, category_results, failures, error)
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
        partial = getattr(exc, "partial", None)
        if partial is not None:
            category_results.append(partial)
        failures.append({"category": "global", "error": repr(exc), "traceback": traceback.format_exc()})
        _write_partial(protocol, status, category_results, failures, error)

    elapsed = float(time.perf_counter() - started)
    aggregate = _aggregate(category_results)
    if status == "complete" and aggregate["category_count"] != len(categories):
        status = "error_partial"
        error = error or "category count mismatch"
    if any(not bool(result.get("complete")) for result in category_results):
        status = "partial_deadline" if status == "complete" else status
    if status == "complete" and failures:
        status = "error_partial"
    decision = (
        "RESOLUTION_STORAGE_SUPPORT_ONLY_COMPLETE"
        if status == "complete" and not failures and aggregate["category_count"] == len(categories)
        else "RESOLUTION_STORAGE_PARTIAL_NO_CLAIM"
        if status in {"partial_deadline", "partial_no_new_category", "partial_runtime_budget"}
        else "RESOLUTION_STORAGE_FAIL_ARCHIVE"
    )
    result = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "protocol": protocol,
        "status": status,
        "decision": decision,
        "error": error,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "categories_requested": categories,
        "categories_completed_or_partial": [item["category"] for item in category_results],
        "synthetic_unique_episode_count_requested": len(categories) * SHOT * len(SYN_FAMILIES),
        "synthetic_unique_episode_count_observed": aggregate.get("synthetic_unique_episode_count", 0),
        "aggregate": aggregate,
        "category_results": category_results,
        "updated_utc": _utc_now(),
    }
    _write_json(OUT / "RESULTS.json", result)
    _write_partial(protocol, status, category_results, failures, error)
    _write_decision_cn(result)
    _write_json(
        OUT / "RUN_MANIFEST.json",
        {
            "schema_version": 1,
            "status": status,
            "decision": decision,
            "started_utc": protocol["created_utc"],
            "finished_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "categories_requested": categories,
            "categories_completed_or_partial": [item["category"] for item in category_results],
            "rows": sum(len(item.get("rows", [])) for item in category_results),
            "synthetic_unique_episode_count_observed": aggregate.get("synthetic_unique_episode_count", 0),
            "encoding_latency_measured": False,
            "runtime_quantized_methods_run": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": _git_commit(),
        },
    )
    print(f"status={status} decision={decision} categories={aggregate['category_count']} elapsed={elapsed:.1f}s", flush=True)
    return 0 if status == "complete" and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
