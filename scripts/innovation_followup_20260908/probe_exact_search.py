"""Measure an exact triangle-bound nearest-neighbour search on cached DINO features.

This is a bounded engineering probe.  It uses two preselected classes from the
existing full896 cache, keeps every support patch in a normal-only bank, and
compares a full float32 Euclidean 1-NN scan with a pivot lower-bound scan that
reranks every retained candidate exactly.  No model is loaded and no labels or
masks are read.

The pivot count, class list, query subsample and all numerical tolerances are
constants in this file.  They are intentionally not command-line parameters:
this run is a single pre-registered measurement rather than a search over
retrieval settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Keep BLAS/OpenMP bounded before importing NumPy.  This probe is CPU-only.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np  # noqa: E402

try:  # noqa: E402
    import torch
except Exception:  # pragma: no cover - torch is optional for this read-only probe
    torch = None


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_VERSION = "exact_search_v1"
INPUTS = {
    "bracket_black": ROOT
    / "outputs"
    / "dynamic_fusion"
    / "overnight_20260908_full896"
    / "features"
    / "bracket_black.npz",
    "connector": ROOT
    / "outputs"
    / "dynamic_fusion"
    / "overnight_20260908_full896_extension"
    / "features"
    / "connector.npz",
}
CATEGORIES = ["bracket_black", "connector"]
PIVOT_COUNT = 32
QUERY_COUNT_PER_CLASS = 512
WARMUP_REPEATS = 2
FORMAL_REPEATS = 3
TIE_TOL = np.float32(1.0e-6)
MAX_RUNTIME_S = 15 * 60
QUERY_CHECK_INTERVAL = 16


class BudgetExceeded(RuntimeError):
    """Raised when the bounded probe has reached its wall-clock budget."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")
    tmp.replace(path)


def _stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _check_budget(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise BudgetExceeded(stage)


def _normalize_rows(array: np.ndarray, label: str) -> np.ndarray:
    rows = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
    norms = np.linalg.norm(rows, axis=1, keepdims=True).astype(np.float32, copy=False)
    if not np.all(np.isfinite(norms)) or np.any(norms <= np.float32(1.0e-12)):
        raise ValueError(f"{label} contains non-finite or zero rows")
    rows = rows / norms
    return np.asarray(rows, dtype=np.float32, order="C")


def _cache_keys(z: Any) -> tuple[str, str]:
    clean_key = "clean_feat_896" if "clean_feat_896" in z.files else "clean_feat"
    query_key = "syn_feat_896" if "syn_feat_896" in z.files else "syn_feat"
    return clean_key, query_key


def _load_category(path: Path, category: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        clean_key, query_key = _cache_keys(z)
        clean = np.array(z[clean_key], dtype=np.float32, copy=True)
        query = np.array(z[query_key], dtype=np.float32, copy=True)
        clean_valid = np.array(z["clean_valid"], copy=True) if "clean_valid" in z.files else None
        grid_key = "grid_896" if "grid_896" in z.files else "grid_size"
        grid_value = np.array(z[grid_key], copy=True).reshape(-1).tolist() if grid_key in z.files else []
        feature_dim = int(np.asarray(z["feature_dim"]).reshape(-1)[0]) if "feature_dim" in z.files else int(clean.shape[-1])
    if clean.ndim != 4 or query.ndim != 5:
        raise ValueError(f"{category}: unexpected clean/query ranks {clean.shape} / {query.shape}")
    if clean.shape[0] != 2 or query.shape[0] != 2 or clean.shape[-1] != query.shape[-1]:
        raise ValueError(f"{category}: expected two support images and matching dimensions")
    if clean.shape[1:3] != (64, 64) or query.shape[2:4] != (64, 64):
        raise ValueError(f"{category}: high-resolution cache is not 64 x 64")
    if feature_dim != 768:
        raise ValueError(f"{category}: expected DINO dimension 768, got {feature_dim}")
    if clean_valid is not None and not np.all(np.asarray(clean_valid) > 0):
        raise ValueError(f"{category}: cache marks a clean support feature invalid")
    clean_unit = _normalize_rows(clean, f"{category} clean support")
    all_query_unit = _normalize_rows(query, f"{category} synthetic query")
    n_all = all_query_unit.shape[0]
    query_indices = np.linspace(0, n_all - 1, QUERY_COUNT_PER_CLASS, dtype=np.int64)
    if np.unique(query_indices).size != QUERY_COUNT_PER_CLASS:
        raise ValueError(f"{category}: query subsample is not unique")
    query_unit = np.ascontiguousarray(all_query_unit[query_indices])
    meta = {
        "category": category,
        "input": str(path.relative_to(ROOT)),
        "input_sha256": _sha256(path),
        "clean_key": clean_key,
        "query_key": query_key,
        "clean_shape": list(clean.shape),
        "query_shape": list(query.shape),
        "feature_dim": feature_dim,
        "grid_value": grid_value,
        "support_images": int(clean.shape[0]),
        "support_rows": int(clean_unit.shape[0]),
        "query_rows_before_subsample": int(n_all),
        "query_rows_used": int(query_unit.shape[0]),
        "query_selection": "np.linspace(0, n-1, 512, dtype=int64) over flattened synthetic cache; feature-only",
        "query_index_sha256": hashlib.sha256(query_indices.tobytes()).hexdigest(),
        "query_first_indices": query_indices[:8].tolist(),
        "query_last_indices": query_indices[-8:].tolist(),
    }
    return clean_unit, query_unit, meta


def _euclidean_to_rows(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Direct Euclidean distance for one unit F32 row and many unit rows.

    The subtraction, square and reduction are deliberately written as the
    Euclidean norm.  No cosine distance is used by the pruning rule.  The
    unit-vector identity is only used later for the reported d^2 / 2 value.
    """
    diff = np.empty_like(rows, dtype=np.float32)
    np.subtract(rows, query, out=diff)
    np.square(diff, out=diff)
    squared = np.sum(diff, axis=1, dtype=np.float32)
    return np.sqrt(squared).astype(np.float32, copy=False)


def _select_pivots(bank: np.ndarray, count: int, deadline: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic farthest-point selection using only the normal bank."""
    if count <= 0 or count > bank.shape[0]:
        raise ValueError("invalid pivot count")
    start = time.perf_counter()
    selected = [0]
    min_dist = _euclidean_to_rows(bank[0], bank)
    for _ in range(1, count):
        _check_budget(deadline, "pivot selection")
        next_index = int(np.argmax(min_dist))
        selected.append(next_index)
        min_dist = np.minimum(min_dist, _euclidean_to_rows(bank[next_index], bank))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    indices = np.asarray(selected, dtype=np.int64)
    return np.ascontiguousarray(bank[indices]), indices, float(elapsed_ms)


def _precompute_bank_pivot_distances(
    bank: np.ndarray, pivots: np.ndarray, deadline: float
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = np.empty((bank.shape[0], pivots.shape[0]), dtype=np.float32)
    for p in range(pivots.shape[0]):
        _check_budget(deadline, "bank-pivot distance precompute")
        out[:, p] = _euclidean_to_rows(pivots[p], bank)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return out, float(elapsed_ms)


def _exact_search(
    queries: np.ndarray,
    bank: np.ndarray,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    start = time.perf_counter()
    indices = np.empty(queries.shape[0], dtype=np.int64)
    distances = np.empty(queries.shape[0], dtype=np.float32)
    tie_counts = np.empty(queries.shape[0], dtype=np.int32)
    for i, query in enumerate(queries):
        if deadline is not None and i % QUERY_CHECK_INTERVAL == 0:
            _check_budget(deadline, "full exact search")
        d = _euclidean_to_rows(query, bank)
        j = int(np.argmin(d))
        best = np.float32(d[j])
        indices[i] = j
        distances[i] = best
        tie_counts[i] = int(np.count_nonzero(d <= best + TIE_TOL))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return indices, distances, float(elapsed_ms), tie_counts


def _bound_exact_search(
    queries: np.ndarray,
    bank: np.ndarray,
    pivots: np.ndarray,
    bank_pivot_distances: np.ndarray,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Exact 1-NN with a Euclidean triangle lower bound and exact reranking."""
    start_total = time.perf_counter()
    q_pivot_ms = 0.0
    bound_ms = 0.0
    rerank_ms = 0.0
    indices = np.empty(queries.shape[0], dtype=np.int64)
    distances = np.empty(queries.shape[0], dtype=np.float32)
    candidate_counts = np.empty(queries.shape[0], dtype=np.int32)
    upper_bounds = np.empty(queries.shape[0], dtype=np.float32)
    for i, query in enumerate(queries):
        if deadline is not None and i % QUERY_CHECK_INTERVAL == 0:
            _check_budget(deadline, "pivot bound search")

        t0 = time.perf_counter()
        query_pivot_distances = _euclidean_to_rows(query, pivots)
        q_pivot_ms += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        # Reverse triangle inequality: d(q, x) >= |d(q, p) - d(x, p)|.
        lower_bound = np.max(
            np.abs(bank_pivot_distances - query_pivot_distances[None, :]), axis=1
        )
        upper = np.min(query_pivot_distances)
        # Strictly impossible rows are pruned.  The tolerance retains numeric
        # ties and protects exactness against F32 round-off at the boundary.
        keep = lower_bound <= upper + TIE_TOL
        candidate_indices = np.flatnonzero(keep)
        bound_ms += (time.perf_counter() - t0) * 1000.0
        if candidate_indices.size == 0:
            raise AssertionError("triangle bound removed every candidate")

        t0 = time.perf_counter()
        candidate_distances = _euclidean_to_rows(query, bank[candidate_indices])
        local = int(np.argmin(candidate_distances))
        indices[i] = int(candidate_indices[local])
        distances[i] = np.float32(candidate_distances[local])
        candidate_counts[i] = int(candidate_indices.size)
        upper_bounds[i] = np.float32(upper)
        rerank_ms += (time.perf_counter() - t0) * 1000.0

    total_ms = (time.perf_counter() - start_total) * 1000.0
    return {
        "indices": indices,
        "distances": distances,
        "candidate_counts": candidate_counts,
        "upper_bounds": upper_bounds,
        "timing_ms": {
            "total": float(total_ms),
            "query_to_pivot": float(q_pivot_ms),
            "lower_bound": float(bound_ms),
            "candidate_rerank": float(rerank_ms),
            "unattributed_loop": float(max(0.0, total_ms - q_pivot_ms - bound_ms - rerank_ms)),
        },
    }


def _run_warmups(
    queries: np.ndarray,
    bank: np.ndarray,
    pivots: np.ndarray,
    bank_pivot_distances: np.ndarray,
    deadline: float,
) -> list[dict[str, Any]]:
    # A fixed prefix warms both kernels without spending the formal budget on
    # an additional hidden query selection.
    warm_queries = queries[: min(32, queries.shape[0])]
    rows: list[dict[str, Any]] = []
    for repeat in range(WARMUP_REPEATS):
        _check_budget(deadline, "warmup")
        _, _, exact_ms, _ = _exact_search(warm_queries, bank, deadline)
        bound = _bound_exact_search(
            warm_queries, bank, pivots, bank_pivot_distances, deadline
        )
        rows.append(
            {
                "repeat": repeat + 1,
                "query_count": int(warm_queries.shape[0]),
                "full_exact_ms": float(exact_ms),
                "pivot_bound_ms": float(bound["timing_ms"]["total"]),
            }
        )
    return rows


def _formal_runs(
    queries: np.ndarray,
    bank: np.ndarray,
    pivots: np.ndarray,
    bank_pivot_distances: np.ndarray,
    deadline: float,
) -> dict[str, Any]:
    exact_runs: list[dict[str, Any]] = []
    bound_runs: list[dict[str, Any]] = []
    reference_exact: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    reference_bound: dict[str, Any] | None = None
    for repeat in range(FORMAL_REPEATS):
        _check_budget(deadline, "formal search")
        exact_i, exact_d, exact_ms, exact_ties = _exact_search(queries, bank, deadline)
        bound = _bound_exact_search(
            queries, bank, pivots, bank_pivot_distances, deadline
        )
        if reference_exact is None:
            reference_exact = (exact_i.copy(), exact_d.copy(), exact_ties.copy())
            reference_bound = bound
        else:
            if not np.array_equal(exact_i, reference_exact[0]):
                raise AssertionError("full exact NN changed between formal repeats")
            if not np.array_equal(bound["indices"], reference_bound["indices"]):
                raise AssertionError("bound exact NN changed between formal repeats")
        exact_runs.append(
            {
                "repeat": repeat + 1,
                "query_count": int(queries.shape[0]),
                "elapsed_ms": float(exact_ms),
            }
        )
        bound_runs.append(
            {
                "repeat": repeat + 1,
                "query_count": int(queries.shape[0]),
                "elapsed_ms": float(bound["timing_ms"]["total"]),
                "query_to_pivot_ms": float(bound["timing_ms"]["query_to_pivot"]),
                "lower_bound_ms": float(bound["timing_ms"]["lower_bound"]),
                "candidate_rerank_ms": float(bound["timing_ms"]["candidate_rerank"]),
                "unattributed_loop_ms": float(bound["timing_ms"]["unattributed_loop"]),
            }
        )
    assert reference_exact is not None and reference_bound is not None
    exact_i, exact_d, exact_ties = reference_exact
    bound_i = np.asarray(reference_bound["indices"], dtype=np.int64)
    bound_d = np.asarray(reference_bound["distances"], dtype=np.float32)
    abs_diff = np.abs(bound_d.astype(np.float64) - exact_d.astype(np.float64))
    tie_aware = abs_diff <= float(TIE_TOL)
    exact_index_match = bound_i == exact_i
    # A distinct index with the same distance is valid under the stated tie
    # tolerance.  The direct full scan supplies the tie multiplicity report.
    strict_index_match = int(np.count_nonzero(exact_index_match))
    tie_aware_match = int(np.count_nonzero(tie_aware))
    candidates = np.asarray(reference_bound["candidate_counts"], dtype=np.int64)
    bank_n = int(bank.shape[0])
    d2_over_2 = np.square(bound_d.astype(np.float64)) / 2.0
    exact_d2_over_2 = np.square(exact_d.astype(np.float64)) / 2.0
    return {
        "exact_reference": {
            "nn_index": exact_i,
            "distance_euclidean": exact_d,
            "distance_d2_over_2": exact_d2_over_2,
            "tie_counts_at_1e-6": exact_ties,
        },
        "bound_reference": {
            "nn_index": bound_i,
            "distance_euclidean": bound_d,
            "distance_d2_over_2": d2_over_2,
            "candidate_counts": candidates,
            "upper_bounds": np.asarray(reference_bound["upper_bounds"], dtype=np.float32),
        },
        "formal_exact_runs": exact_runs,
        "formal_bound_runs": bound_runs,
        "correctness": {
            "query_count": int(queries.shape[0]),
            "bank_count": bank_n,
            "max_abs_euclidean_distance_diff": float(np.max(abs_diff)),
            "mean_abs_euclidean_distance_diff": float(np.mean(abs_diff)),
            "max_abs_d2_over_2_diff": float(
                np.max(np.abs(d2_over_2 - exact_d2_over_2))
            ),
            "strict_nn_index_match_count": strict_index_match,
            "strict_nn_index_match_rate": float(strict_index_match / queries.shape[0]),
            "tie_aware_match_count": tie_aware_match,
            "tie_aware_match_rate": float(tie_aware_match / queries.shape[0]),
            "exact_tie_query_count": int(np.count_nonzero(exact_ties > 1)),
            "max_exact_tie_count": int(np.max(exact_ties)),
            "tie_tolerance_euclidean": float(TIE_TOL),
        },
        "candidate_retention": {
            "n_candidates": _stats(candidates),
            "retention_rate": _stats(candidates.astype(np.float64) / bank_n),
            "pruned_count": _stats((bank_n - candidates).astype(np.float64)),
            "all_queries_have_candidate": bool(np.all(candidates > 0)),
        },
    }


def _strip_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _strip_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_arrays(v) for v in value]
    return value


def _summarize_timing(category_result: dict[str, Any]) -> dict[str, Any]:
    exact = category_result["formal_exact_runs"]
    bound = category_result["formal_bound_runs"]
    exact_ms = np.asarray([x["elapsed_ms"] for x in exact], dtype=np.float64)
    bound_ms = np.asarray([x["elapsed_ms"] for x in bound], dtype=np.float64)
    precompute = float(category_result["pivot_cost_ms"]["bank_pivot_distance_precompute"])
    pivot_select = float(category_result["pivot_cost_ms"]["pivot_selection"])
    qn = int(category_result["query_count"])
    amortized_bound = bound_ms + (pivot_select + precompute) / max(qn, 1)
    return {
        "full_exact_formal_ms": _stats(exact_ms),
        "pivot_bound_formal_ms_including_bound_and_rerank": _stats(bound_ms),
        "pivot_bound_amortized_ms_including_pivot_setup": _stats(amortized_bound),
        "pivot_setup_ms": {
            "pivot_selection": pivot_select,
            "bank_pivot_distance_precompute": precompute,
            "total": pivot_select + precompute,
        },
        "speed_ratio_bound_over_exact": _stats(bound_ms / exact_ms),
        "speed_ratio_amortized_bound_over_exact": _stats(amortized_bound / exact_ms),
        "query_count": qn,
    }


def _run_category(category: str, deadline: float) -> dict[str, Any]:
    path = INPUTS[category]
    bank, queries, input_meta = _load_category(path, category)
    pivots, pivot_indices, pivot_selection_ms = _select_pivots(
        bank, PIVOT_COUNT, deadline
    )
    bank_pivot_distances, bank_pivot_ms = _precompute_bank_pivot_distances(
        bank, pivots, deadline
    )
    warmups = _run_warmups(
        queries, bank, pivots, bank_pivot_distances, deadline
    )
    formal = _formal_runs(
        queries, bank, pivots, bank_pivot_distances, deadline
    )
    result = {
        "category": category,
        "input": input_meta,
        "bank": {
            "rows": int(bank.shape[0]),
            "dimension": int(bank.shape[1]),
            "support_only": True,
            "unit_norm_max_abs_error": float(
                np.max(np.abs(np.linalg.norm(bank, axis=1) - 1.0))
            ),
        },
        "query_count": int(queries.shape[0]),
        "pivot": {
            "count": PIVOT_COUNT,
            "selection": "deterministic farthest-point greedy; first bank row 0; normal-only",
            "indices": pivot_indices,
            "indices_sha256": hashlib.sha256(pivot_indices.tobytes()).hexdigest(),
            "unit_norm_max_abs_error": float(
                np.max(np.abs(np.linalg.norm(pivots, axis=1) - 1.0))
            ),
        },
        "pivot_cost_ms": {
            "pivot_selection": pivot_selection_ms,
            "bank_pivot_distance_precompute": bank_pivot_ms,
        },
        "warmups": warmups,
        "formal_exact_runs": formal["formal_exact_runs"],
        "formal_bound_runs": formal["formal_bound_runs"],
        "correctness": formal["correctness"],
        "candidate_retention": formal["candidate_retention"],
    }
    result["timing"] = _summarize_timing(result)
    # Keep per-query arrays because this is a small fixed query sample and the
    # exactness audit is more useful when it can be independently rechecked.
    result["per_query"] = {
        "exact_nn_index": formal["exact_reference"]["nn_index"],
        "bound_nn_index": formal["bound_reference"]["nn_index"],
        "exact_distance_euclidean": formal["exact_reference"]["distance_euclidean"],
        "bound_distance_euclidean": formal["bound_reference"]["distance_euclidean"],
        "exact_distance_d2_over_2": formal["exact_reference"]["distance_d2_over_2"],
        "bound_distance_d2_over_2": formal["bound_reference"]["distance_d2_over_2"],
        "candidate_count": formal["bound_reference"]["candidate_counts"],
        "exact_tie_count": formal["exact_reference"]["tie_counts_at_1e-6"],
    }
    return _strip_arrays(result)


def _aggregate(categories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not categories:
        return {"n_categories": 0}
    correctness = [v["correctness"] for v in categories.values()]
    retention = [v["candidate_retention"] for v in categories.values()]
    exact = np.asarray(
        [r["timing"]["full_exact_formal_ms"]["mean"] for r in categories.values()],
        dtype=np.float64,
    )
    bound = np.asarray(
        [
            r["timing"]["pivot_bound_formal_ms_including_bound_and_rerank"]["mean"]
            for r in categories.values()
        ],
        dtype=np.float64,
    )
    total_queries = int(sum(r["query_count"] for r in categories.values()))
    strict = int(sum(r["correctness"]["strict_nn_index_match_count"] for r in categories.values()))
    tie_aware = int(sum(r["correctness"]["tie_aware_match_count"] for r in categories.values()))
    total_bank_rows = int(sum(r["bank"]["rows"] for r in categories.values()))
    all_candidates = np.concatenate(
        [np.asarray(r["per_query"]["candidate_count"], dtype=np.float64) for r in categories.values()]
    )
    all_abs_diff = np.concatenate(
        [
            np.abs(
                np.asarray(r["per_query"]["bound_distance_euclidean"], dtype=np.float64)
                - np.asarray(r["per_query"]["exact_distance_euclidean"], dtype=np.float64)
            )
            for r in categories.values()
        ]
    )
    return {
        "n_categories": len(categories),
        "categories": list(categories),
        "query_count": total_queries,
        "strict_nn_index_match_rate": float(strict / total_queries),
        "tie_aware_match_rate": float(tie_aware / total_queries),
        "max_abs_euclidean_distance_diff": float(np.max(all_abs_diff)),
        "candidate_count": _stats(all_candidates),
        "candidate_retention_rate": _stats(all_candidates / np.asarray([r["bank"]["rows"] for r in categories.values() for _ in range(r["query_count"])], dtype=np.float64)),
        "formal_mean_exact_ms_by_category": {k: float(v) for k, v in zip(categories, exact)},
        "formal_mean_bound_ms_by_category": {k: float(v) for k, v in zip(categories, bound)},
        "macro_speed_ratio_bound_over_exact": float(np.mean(bound / exact)),
        "macro_speed_ratio_bound_over_exact_weighted_by_queries": float(
            np.sum(bound * np.asarray([r["query_count"] for r in categories.values()]))
            / np.sum(exact * np.asarray([r["query_count"] for r in categories.values()]))
        ),
        "total_bank_rows_across_categories": total_bank_rows,
        "note": "Categories are independent banks; aggregate timing is descriptive, not one shared index.",
    }


def _decision_markdown(result: dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    aggregate = result.get("aggregate", {})
    lines = [
        "# 精确三角下界检索探针结论",
        "",
        f"状态：`{status}`。输入为已有 full896 高分辨率缓存，未重新提取、未初始化 GPU、未读取 mask 或 test 标签。",
        "",
        "本轮固定 `bracket_black`、`connector` 两类；每类两张 clean support 图的全部 `8192 × 768` unit F32 行作为 memory，固定 `32` 个 normal-only farthest-point pivot，并从 synthetic cache 展平行中固定抽取 `512` 个 query。pivot 臂先用欧氏三角下界筛选，再对保留行做完整欧氏重排。",
        "",
    ]
    if aggregate:
        lines.extend(
            [
                f"两类合计 query 数为 {aggregate.get('query_count')}；严格 NN 索引一致率 {aggregate.get('strict_nn_index_match_rate', 0.0):.6f}，并列容忍后一致率 {aggregate.get('tie_aware_match_rate', 0.0):.6f}，最大欧氏距离差 {aggregate.get('max_abs_euclidean_distance_diff', 0.0):.3e}。",
                f"候选数统计为 n={aggregate.get('candidate_count', {}).get('n')}、均值 {aggregate.get('candidate_count', {}).get('mean', 0.0):.2f}；候选保留率均值 {aggregate.get('candidate_retention_rate', {}).get('mean', 0.0):.4f}。",
                f"pivot 臂相对 full exact 的宏平均时间比为 {aggregate.get('macro_speed_ratio_bound_over_exact', 0.0):.3f}；该比值已包含 query-pivot、下界和候选重排，另有 pivot setup 成本，不能从理论直接称为加速。",
                "",
            ]
        )
    category_rows = []
    for category, item in result.get("categories", {}).items():
        c = item["correctness"]
        r = item["candidate_retention"]
        t = item["timing"]
        category_rows.append(
            f"| {category} | {item['query_count']} | {c['strict_nn_index_match_rate']:.6f} | {c['tie_aware_match_rate']:.6f} | {r['retention_rate']['mean']:.4f} | {t['full_exact_formal_ms']['mean']:.1f} | {t['pivot_bound_formal_ms_including_bound_and_rerank']['mean']:.1f} |"
        )
    if category_rows:
        lines.extend(
            [
                "| 类别 | query n | 严格 NN 一致率 | 并列容忍一致率 | 候选保留率均值 | full exact ms | pivot bound ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *category_rows,
                "",
            ]
        )
    lines.extend(
        [
            "解释边界：若候选保留率接近 1 或 pivot 臂时间更高，说明本机与此高维 memory 上下界没有带来实用收益；若保留率较低且时间比小于 1，也只代表本次两类、固定 query 子样本和实现。该结果验证精确性与实测成本，不构成新算法、普遍加速或因果结论。",
            "",
            "审计与复现：`PROTOCOL_CN.md`、`PROTOCOL.json`、`AUDIT_CN.md`、`COMMAND.txt`；原始 full896 cache 保持不变。",
            "",
        ]
    )
    return "\n".join(lines)


def _protocol(output_dir: Path, started_utc: str) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "role": "bounded engineering probe; known exact triangle-bound retrieval, not an original algorithm",
        "input_mode": "read-only existing full896 feature cache; no model extraction",
        "categories": CATEGORIES,
        "fixed_pivot_count": PIVOT_COUNT,
        "pivot_selection": "normal-only deterministic farthest-point greedy, first bank row 0",
        "fixed_query_count_per_category": QUERY_COUNT_PER_CLASS,
        "query_selection": "np.linspace over flattened synthetic feature rows; feature-only; no masks/labels",
        "bank": "two clean support images, all 64 x 64 cells, unit float32 rows, 768 dimensions",
        "distance": "direct Euclidean norm on unit float32 vectors for both full scan and triangle bound",
        "triangle_bound": "max_p abs(d(q,p)-d(x,p)); prune only if lower_bound > pivot upper_bound + 1e-6; exact rerank retained rows",
        "return_conversion": "after NN only: d^2 / 2 recorded for comparison with 1 - cosine",
        "tie_tolerance_euclidean": float(TIE_TOL),
        "warmup_repeats": WARMUP_REPEATS,
        "formal_repeats": FORMAL_REPEATS,
        "resources": {
            "cpu_only": True,
            "gpu": False,
            "torch_threads": 2,
            "blas_threads": 2,
            "max_runtime_s": MAX_RUNTIME_S,
            "platform": platform.platform(),
            "python": sys.version,
        },
        "no_test_selection": True,
        "no_parameter_search": True,
        "output_dir": str(output_dir),
        "started_utc": started_utc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "dynamic_fusion" / "innovation_followup_20260908" / "exact_search",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch is not None:
        try:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

    started_utc = _utc_now()
    started_mono = time.monotonic()
    deadline = started_mono + MAX_RUNTIME_S
    protocol = _protocol(output_dir, started_utc)
    _write_json(output_dir / "PROTOCOL.json", protocol)
    (output_dir / "AUDIT_CN.md").write_text(
        "# 旧代码审计\n\n"
        "在 `scripts/`、`src/`、`tests/` 中检索 `triangle`、`pivot`、`lower bound`、`exact NN`、`IndexFlatL2` 等关键词。旧路径包含完整 FAISS / NumPy 近邻、coreset 和普通 pandas `pivot` 表操作；未见 pivot + triangle-bound 的精确剪枝实现。本脚本因此是新实验文件，不覆盖旧实现。\n\n"
        "本轮使用既有 full896 cache 的 clean support 特征和 synthetic query 特征，未重新提取、未初始化 CUDA，也不读取标签或 mask。\n",
        encoding="utf-8",
        newline="\n",
    )

    categories: dict[str, dict[str, Any]] = {}
    status = "complete"
    error: str | None = None
    try:
        for category in CATEGORIES:
            _check_budget(deadline, f"before category {category}")
            item = _run_category(category, deadline)
            categories[category] = item
            partial = {
                "protocol": protocol,
                "status": "partial" if len(categories) < len(CATEGORIES) else "complete",
                "categories": categories,
                "completed_categories": list(categories),
                "elapsed_s": time.monotonic() - started_mono,
            }
            _write_json(output_dir / "PARTIAL_RESULTS.json", partial)
    except BudgetExceeded as exc:
        status = "partial"
        error = f"budget exceeded at {exc}"
    except Exception as exc:  # preserve diagnostic context in the archive
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed_s = time.monotonic() - started_mono
        result: dict[str, Any] = {
            "protocol": protocol,
            "status": status,
            "error": error,
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            "elapsed_s": float(elapsed_s),
            "completed_categories": list(categories),
            "categories": categories,
            "aggregate": _aggregate(categories),
        }
        _write_json(output_dir / "PARTIAL_RESULTS.json", result)
        if status == "complete":
            _write_json(output_dir / "RESULTS.json", result)
            (output_dir / "DECISION_CN.md").write_text(
                _decision_markdown(result), encoding="utf-8", newline="\n"
            )
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

