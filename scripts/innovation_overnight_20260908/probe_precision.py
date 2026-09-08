"""Support-only precision/efficiency probe for the frozen A1 memory bank.

The probe keeps every normal memory cell and evaluates storage-only and resident
INT8/FP16 retrieval variants on the v14 support cache.  It deliberately imports
the existing v14 cell loader instead of changing an older experiment script.
No test images, test masks, GPU calls, fitting, or result-dependent selection are
used here.
"""
from __future__ import annotations

import argparse
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

# Keep all numerical libraries on the declared CPU budget before importing them.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[2]
OLD_SCRIPT_DIR = ROOT / "scripts" / "innovation_t3_efficiency_20260905"
sys.path.insert(0, str(OLD_SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import probe_e1_compress as E1  # noqa: E402  (the existing support-only loader)

try:  # noqa: E402
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency in older environments
    threadpool_limits = None


OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "precision"
CATEGORIES = ("bracket_black", "bracket_brown", "bracket_white", "connector",
              "metal_plate", "tubes")
FAMILIES = ("cutpaste", "local_erasure", "thin_scratch")
METHODS = ("control_f32", "fp16_roundtrip", "fp16_runtime",
           "int8_pc_roundtrip", "int8_pc_runtime")
STORAGE_METHODS = ("fp16_roundtrip", "int8_pc_roundtrip")
INT8_METHODS = ("int8_pc_roundtrip", "int8_pc_runtime")
GRID = 16
DIM = 1536
INT8_CHUNK = 256
TOP_K = 10
EPS = np.float32(1e-12)

GATES = {
    "fp16": {"ap_delta": -0.005, "worst_ap_delta": -0.02,
             "distance_p99": 0.002, "nn_agreement": 0.995},
    "int8": {"ap_delta": -0.02, "worst_ap_delta": -0.05,
             "distance_p99": 0.02, "nn_agreement": 0.95},
}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _json_value(value: Any) -> Any:
    """Convert numpy scalars/arrays without emitting non-standard NaN JSON."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norm, EPS)


def _fuse_cells(cells: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the A1 support tensors using the existing loader's cell layout."""
    k = int(cells["K"])
    clean = E1.a1(cells["d"], cells["c"]).reshape(k, GRID * GRID, DIM)
    syn = E1.a1(
        cells["ds"].reshape(-1, GRID, GRID, 768),
        cells["cs"].reshape(-1, GRID, GRID, 768),
    ).reshape(k, 9, GRID * GRID, DIM)
    nui = E1.a1(
        cells["dn"].reshape(-1, GRID, GRID, 768),
        cells["cn"].reshape(-1, GRID, GRID, 768),
    ).reshape(k, 15, GRID * GRID, DIM)
    return (np.asarray(clean, dtype=np.float32),
            np.asarray(syn, dtype=np.float32),
            np.asarray(nui, dtype=np.float32))


def _symmetric_pc_int8(bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-channel memory quantization with frozen round-to-nearest."""
    scale = np.max(np.abs(bank), axis=0).astype(np.float32) / np.float32(127.0)
    scale = np.maximum(scale, EPS).astype(np.float32)
    quant = np.clip(np.rint(bank / scale), -127, 127).astype(np.int8)
    return quant, scale


def _quantize_query(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row dynamic symmetric query quantization for the runtime INT8 path."""
    q = _unit_rows(query)
    scale = np.max(np.abs(q), axis=1, keepdims=True).astype(np.float32) / np.float32(127.0)
    scale = np.maximum(scale, EPS).astype(np.float32)
    quant = np.clip(np.rint(q / scale), -127, 127).astype(np.int8)
    return quant, scale


def _prepare(method: str, bank: np.ndarray) -> tuple[dict[str, Any], float]:
    """Prepare one candidate and return metadata plus measured preparation time."""
    start = time.perf_counter()
    rows, dim = bank.shape
    if method == "control_f32":
        runtime = _unit_rows(bank.copy())
        prepared = {"runtime": runtime, "rows": rows, "dim": dim,
                    "storage_bytes": rows * dim * 4,
                    "resident_bank_bytes": runtime.nbytes,
                    "workspace_bytes": 0, "scale_bytes": 0}
    elif method == "fp16_roundtrip":
        stored = bank.astype(np.float16, copy=True)
        decoded = _unit_rows(stored.astype(np.float32))
        prepared = {"runtime": decoded, "rows": rows, "dim": dim,
                    "storage_bytes": stored.nbytes,
                    "resident_bank_bytes": decoded.nbytes,
                    "workspace_bytes": 0, "scale_bytes": 0}
    elif method == "fp16_runtime":
        stored = bank.astype(np.float16, copy=True)
        prepared = {"runtime": stored, "rows": rows, "dim": dim,
                    "storage_bytes": stored.nbytes,
                    "resident_bank_bytes": stored.nbytes,
                    "workspace_bytes": 0, "scale_bytes": 0}
    elif method in INT8_METHODS:
        quant, scale = _symmetric_pc_int8(bank)
        if method == "int8_pc_roundtrip":
            decoded = _unit_rows(quant.astype(np.float32) * scale[None, :])
            runtime = decoded
            resident = decoded.nbytes + scale.nbytes
            workspace = 0
        else:
            runtime = quant
            resident = quant.nbytes + scale.nbytes
            # A single weighted float32 channel chunk is materialized per dot call.
            workspace = min(dim, INT8_CHUNK) * rows * 4
        prepared = {"runtime": runtime, "scale": scale, "rows": rows, "dim": dim,
                    "storage_bytes": quant.nbytes + scale.nbytes,
                    "resident_bank_bytes": resident,
                    "workspace_bytes": workspace, "scale_bytes": scale.nbytes}
    else:  # pragma: no cover - guarded by METHODS
        raise ValueError(f"unknown method: {method}")
    return prepared, time.perf_counter() - start


def _score_float32(query: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = _unit_rows(query)
    dots = q @ bank.T
    nn = np.argmax(dots, axis=1).astype(np.int32)
    return (1.0 - dots[np.arange(dots.shape[0]), nn]).astype(np.float32), nn


def _score_float16(query: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float16)
    dots = np.matmul(q, bank.T)
    nn = np.argmax(dots, axis=1).astype(np.int32)
    return (1.0 - dots[np.arange(dots.shape[0]), nn]).astype(np.float32), nn


def _score_int8_runtime(query: np.ndarray, quant: np.ndarray,
                        scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate cosine with INT8 bank resident; only one float channel chunk exists."""
    q_quant, q_scale = _quantize_query(query)
    n_query, dim = q_quant.shape
    n_bank = quant.shape[0]
    dots = np.zeros((n_query, n_bank), dtype=np.float32)
    q_i32 = q_quant.astype(np.int32)
    for start in range(0, dim, INT8_CHUNK):
        stop = min(dim, start + INT8_CHUNK)
        # q_i32 avoids int8 accumulation overflow.  The bank remains int8 resident;
        # this temporary only holds one weighted channel chunk.
        bank_chunk = quant[:, start:stop].astype(np.float32) * scale[start:stop][None, :]
        dots += q_i32[:, start:stop].astype(np.float32) @ bank_chunk.T
    dots *= q_scale
    nn = np.argmax(dots, axis=1).astype(np.int32)
    return (1.0 - dots[np.arange(n_query), nn]).astype(np.float32), nn


def _score(method: str, query: np.ndarray, prepared: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if method in ("control_f32", "fp16_roundtrip", "int8_pc_roundtrip"):
        return _score_float32(query, prepared["runtime"])
    if method == "fp16_runtime":
        return _score_float16(query, prepared["runtime"])
    if method == "int8_pc_runtime":
        return _score_int8_runtime(query, prepared["runtime"], prepared["scale"])
    raise ValueError(method)


def _ap(y: np.ndarray, score: np.ndarray) -> float | None:
    labels = np.asarray(y, dtype=np.uint8)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return None
    return float(average_precision_score(labels, np.asarray(score, dtype=np.float64)))


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    labels = np.asarray(y, dtype=np.uint8)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return None
    try:
        return float(roc_auc_score(labels, np.asarray(score, dtype=np.float64)))
    except ValueError:
        return None


def _rank_stats(base_score: np.ndarray, cand_score: np.ndarray,
                base_nn: np.ndarray, cand_nn: np.ndarray) -> dict[str, float | None]:
    a = np.asarray(base_score, dtype=np.float64).reshape(-1)
    b = np.asarray(cand_score, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"spearman": None, "top10_overlap": None,
                "pairwise_agreement": None, "nn_identity_agreement": None}
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    if np.std(ra) <= 0 or np.std(rb) <= 0:
        spearman = 1.0 if np.array_equal(a, b) else 0.0
    else:
        spearman = float(np.corrcoef(ra, rb)[0, 1])
    k = min(TOP_K, a.size)
    top_a = set(np.argsort(-a, kind="mergesort")[:k].tolist())
    top_b = set(np.argsort(-b, kind="mergesort")[:k].tolist())
    overlap = float(len(top_a & top_b) / k) if k else None
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    upper = np.triu(np.ones_like(da, dtype=bool), k=1)
    valid = upper & (da != 0) & (db != 0)
    pairwise = float(np.mean(np.sign(da[valid]) == np.sign(db[valid]))) if valid.any() else 1.0
    nn_agree = float(np.mean(np.asarray(base_nn).reshape(-1) == np.asarray(cand_nn).reshape(-1)))
    return {"spearman": spearman, "top10_overlap": overlap,
            "pairwise_agreement": pairwise, "nn_identity_agreement": nn_agree}


def _distance_stats(base_score: np.ndarray, cand_score: np.ndarray) -> dict[str, float | None]:
    err = np.abs(np.asarray(cand_score, dtype=np.float64) - np.asarray(base_score, dtype=np.float64))
    if err.size == 0:
        return {"mean": None, "p95": None, "p99": None, "max": None}
    return {"mean": float(np.mean(err)), "p95": float(np.percentile(err, 95)),
            "p99": float(np.percentile(err, 99)), "max": float(np.max(err))}


def _episodes(clean: np.ndarray, syn: np.ndarray, nui: np.ndarray,
              masks: np.ndarray, held_out: int) -> tuple[list[dict[str, Any]], list[str]]:
    episodes: list[dict[str, Any]] = []
    invalid: list[str] = []
    for e in range(9):
        family = FAMILIES[e // 3]
        y = np.asarray(masks[held_out, e], dtype=np.uint8).reshape(-1)
        episode_id = f"syn_{family}_e{e}"
        if y.sum() == 0 or y.sum() == y.size:
            invalid.append(episode_id)
            continue
        episodes.append({"id": episode_id, "kind": "synthetic", "family": family,
                         "query": syn[held_out, e], "labels": y,
                         "positive_cells": int(y.sum())})
    episodes.append({"id": "normal_clean", "kind": "normal_clean", "family": "normal",
                     "query": clean[held_out], "labels": None,
                     "positive_cells": 0})
    for e in range(nui.shape[1]):
        episodes.append({"id": f"normal_nui_e{e}", "kind": "normal_nui", "family": "normal",
                         "query": nui[held_out, e], "labels": None,
                         "positive_cells": 0})
    return episodes, invalid


def _method_episode_metrics(method: str, episodes: list[dict[str, Any]],
                            control_scores: dict[str, tuple[np.ndarray, np.ndarray]],
                            scores: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[np.ndarray] = []
    for episode in episodes:
        eid = episode["id"]
        base_s, base_nn = control_scores[eid]
        cand_s, cand_nn = scores[eid]
        rank = _rank_stats(base_s, cand_s, base_nn, cand_nn)
        dist = _distance_stats(base_s, cand_s)
        errors.append(np.abs(cand_s.astype(np.float64) - base_s.astype(np.float64)))
        row: dict[str, Any] = {"episode": eid, "kind": episode["kind"],
                               "family": episode["family"],
                               "query_cells": int(cand_s.size),
                               "ap": _ap(episode["labels"], cand_s) if episode["labels"] is not None else None,
                               "control_ap": _ap(episode["labels"], base_s) if episode["labels"] is not None else None,
                               "ap_delta": None, "distance_error": dist, "rank": rank}
        if row["ap"] is not None and row["control_ap"] is not None:
            row["ap_delta"] = float(row["ap"] - row["control_ap"])
        rows.append(row)

    syn_rows = [r for r in rows if r["kind"] == "synthetic" and r["ap"] is not None]
    family_ap = {}
    for family in FAMILIES:
        vals = [r["ap"] for r in syn_rows if r["family"] == family and r["ap"] is not None]
        family_ap[family] = float(np.mean(vals)) if vals else None
    ap_vals = [r["ap"] for r in syn_rows if r["ap"] is not None]
    base_ap_vals = [r["control_ap"] for r in syn_rows if r["control_ap"] is not None]
    deltas = [r["ap_delta"] for r in syn_rows if r["ap_delta"] is not None]
    clean_s = scores["normal_clean"][0]
    nui_s = [scores[e["id"]][0] for e in episodes if e["kind"] == "normal_nui"]
    normal_score = np.concatenate([clean_s] + nui_s) if nui_s else clean_s
    normal_y = np.concatenate([np.zeros(clean_s.size, dtype=np.uint8)] +
                              [np.ones(s.size, dtype=np.uint8) for s in nui_s]) if nui_s else np.zeros(clean_s.size, dtype=np.uint8)
    rank_rows = [r["rank"] for r in rows]
    dist_values = np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
    summary = {
        "family_ap": family_ap,
        "macro_ap": float(np.mean(ap_vals)) if ap_vals else None,
        "control_macro_ap": float(np.mean(base_ap_vals)) if base_ap_vals else None,
        "macro_ap_delta": float(np.mean(deltas)) if deltas else None,
        "worst_episode_ap_delta": float(np.min(deltas)) if deltas else None,
        "normal_nuisance_auc": _auc(normal_y, normal_score),
        "rank": {
            "spearman_mean": float(np.mean([r["spearman"] for r in rank_rows if r["spearman"] is not None])) if rank_rows else None,
            "top10_overlap_mean": float(np.mean([r["top10_overlap"] for r in rank_rows if r["top10_overlap"] is not None])) if rank_rows else None,
            "pairwise_agreement_mean": float(np.mean([r["pairwise_agreement"] for r in rank_rows if r["pairwise_agreement"] is not None])) if rank_rows else None,
            "nn_identity_agreement_mean": float(np.mean([r["nn_identity_agreement"] for r in rank_rows if r["nn_identity_agreement"] is not None])) if rank_rows else None,
        },
        "distance_error": {
            "mean": float(np.mean(dist_values)) if dist_values.size else None,
            "p95": float(np.percentile(dist_values, 95)) if dist_values.size else None,
            "p99": float(np.percentile(dist_values, 99)) if dist_values.size else None,
            "max": float(np.max(dist_values)) if dist_values.size else None,
        },
        "episode_count": len(rows),
        "valid_synthetic_episode_count": len(syn_rows),
    }
    return rows, summary


def run_category(category: str) -> dict[str, Any]:
    cat_start = time.perf_counter()
    cells = E1.load_cells(category)
    clean, syn, nui = _fuse_cells(cells)
    masks = np.asarray(cells["masks"], dtype=np.uint8)
    k = int(clean.shape[0])
    method_records: dict[str, list[dict[str, Any]]] = {m: [] for m in METHODS}
    invalid_by_h: dict[str, list[str]] = {}
    episode_records: dict[str, list[dict[str, Any]]] = {m: [] for m in METHODS}
    for held_out in range(k):
        bank = clean[[idx for idx in range(k) if idx != held_out]].reshape(-1, DIM)
        episodes, invalid = _episodes(clean, syn, nui, masks, held_out)
        invalid_by_h[str(held_out)] = invalid
        prepared_by_method: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            prepared, prep_seconds = _prepare(method, bank)
            prepared_by_method[method] = prepared
            method_records[method].append({"held_out": held_out,
                                           "rows": prepared["rows"],
                                           "storage_bytes": prepared["storage_bytes"],
                                           "resident_bank_bytes": prepared["resident_bank_bytes"],
                                           "workspace_bytes": prepared["workspace_bytes"],
                                           "scale_bytes": prepared["scale_bytes"],
                                           "prepare_seconds": prep_seconds})

        control_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        control_prep = prepared_by_method["control_f32"]
        # One warmup call per method is excluded from the measured wall time.
        for method in METHODS:
            _score(method, episodes[0]["query"][:1], prepared_by_method[method])
        for method in METHODS:
            scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            score_start = time.perf_counter()
            for episode in episodes:
                scores[episode["id"]] = _score(method, episode["query"], prepared_by_method[method])
            score_seconds = time.perf_counter() - score_start
            if method == "control_f32":
                control_scores = scores
            rows, summary = _method_episode_metrics(method, episodes, control_scores if control_scores else scores, scores)
            episode_records[method].append({"held_out": held_out, "metrics": summary,
                                            "episodes": rows,
                                            "score_seconds": score_seconds})

    methods_out: dict[str, Any] = {}
    for method in METHODS:
        folds = method_records[method]
        per_fold = episode_records[method]
        ap_values = [f["metrics"]["macro_ap"] for f in per_fold if f["metrics"]["macro_ap"] is not None]
        control_ap = [f["metrics"]["control_macro_ap"] for f in per_fold if f["metrics"]["control_macro_ap"] is not None]
        deltas = [f["metrics"]["macro_ap_delta"] for f in per_fold if f["metrics"]["macro_ap_delta"] is not None]
        worst_episode = [r["ap_delta"] for f in per_fold for r in f["episodes"]
                         if r["kind"] == "synthetic" and r["ap_delta"] is not None]
        p99s = [f["metrics"]["distance_error"]["p99"] for f in per_fold
                if f["metrics"]["distance_error"]["p99"] is not None]
        maxes = [f["metrics"]["distance_error"]["max"] for f in per_fold
                 if f["metrics"]["distance_error"]["max"] is not None]
        rank_keys = ("spearman_mean", "top10_overlap_mean", "pairwise_agreement_mean",
                     "nn_identity_agreement_mean")
        rank_out = {key: float(np.mean([f["metrics"]["rank"][key] for f in per_fold
                                        if f["metrics"]["rank"][key] is not None]))
                    if any(f["metrics"]["rank"][key] is not None for f in per_fold) else None
                    for key in rank_keys}
        method_out = {
            "macro_ap": float(np.mean(ap_values)) if ap_values else None,
            "control_macro_ap": float(np.mean(control_ap)) if control_ap else None,
            "macro_ap_delta": float(np.mean(deltas)) if deltas else None,
            "worst_episode_ap_delta": float(np.min(worst_episode)) if worst_episode else None,
            "normal_nuisance_auc": float(np.mean([f["metrics"]["normal_nuisance_auc"] for f in per_fold
                                                   if f["metrics"]["normal_nuisance_auc"] is not None]))
            if any(f["metrics"]["normal_nuisance_auc"] is not None for f in per_fold) else None,
            "rank": rank_out,
            "distance_error": {
                "p99_worst_fold": float(np.max(p99s)) if p99s else None,
                "max_worst_fold": float(np.max(maxes)) if maxes else None,
                "mean_fold_p99": float(np.mean(p99s)) if p99s else None,
            },
            "memory": {
                "rows": int(max(f["rows"] for f in folds)),
                "rows_min": int(min(f["rows"] for f in folds)),
                "storage_bytes": int(round(np.mean([f["storage_bytes"] for f in folds]))),
                "resident_bank_bytes": int(round(np.mean([f["resident_bank_bytes"] for f in folds]))),
                "workspace_bytes": int(round(np.mean([f["workspace_bytes"] for f in folds]))),
                "scale_bytes": int(round(np.mean([f["scale_bytes"] for f in folds]))),
            },
            "timing": {
                "prepare_seconds": float(np.sum([f["prepare_seconds"] for f in folds])),
                "score_seconds": float(np.sum([f["score_seconds"] for f in per_fold])),
                "total_seconds": float(np.sum([f["prepare_seconds"] for f in folds]) +
                                        np.sum([f["score_seconds"] for f in per_fold])),
            },
            "folds": per_fold,
        }
        methods_out[method] = method_out

    base_storage = methods_out["control_f32"]["memory"]["storage_bytes"]
    base_resident = methods_out["control_f32"]["memory"]["resident_bank_bytes"]
    for method, record in methods_out.items():
        record["memory"]["storage_ratio_vs_control"] = record["memory"]["storage_bytes"] / base_storage
        record["memory"]["resident_ratio_vs_control"] = record["memory"]["resident_bank_bytes"] / base_resident
        record["timing"]["score_ratio_vs_control"] = (
            record["timing"]["score_seconds"] / methods_out["control_f32"]["timing"]["score_seconds"]
            if methods_out["control_f32"]["timing"]["score_seconds"] > 0 else None)
    invalid_all = [f"h{h}:{eid}" for h, vals in invalid_by_h.items() for eid in vals]
    return {"category": category, "shot": 2, "seed": 0, "k": k, "grid": [GRID, GRID],
            "dimension": DIM, "valid_synthetic_episode_count": sum(
                f["metrics"]["valid_synthetic_episode_count"] for f in episode_records["control_f32"]),
            "invalid_synthetic_episodes": invalid_all,
            "methods": methods_out, "wall_seconds": time.perf_counter() - cat_start}


def _aggregate(categories: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"category_count": len(categories), "methods": {}}
    for method in METHODS:
        recs = [c["methods"][method] for c in categories]
        def mean_field(path: tuple[str, ...]) -> float | None:
            vals: list[float] = []
            for rec in recs:
                value: Any = rec
                for key in path:
                    value = value[key]
                if value is not None:
                    vals.append(float(value))
            return float(np.mean(vals)) if vals else None
        def min_field(path: tuple[str, ...]) -> float | None:
            vals: list[float] = []
            for rec in recs:
                value: Any = rec
                for key in path:
                    value = value[key]
                if value is not None:
                    vals.append(float(value))
            return float(np.min(vals)) if vals else None
        memory = recs[0]["memory"] if recs else {}
        out["methods"][method] = {
            "macro_ap": mean_field(("macro_ap",)),
            "control_macro_ap": mean_field(("control_macro_ap",)),
            "macro_ap_delta": mean_field(("macro_ap_delta",)),
            "worst_category_macro_ap_delta": min_field(("macro_ap_delta",)),
            "worst_episode_ap_delta": min_field(("worst_episode_ap_delta",)),
            "normal_nuisance_auc": mean_field(("normal_nuisance_auc",)),
            "rank": {key: mean_field(("rank", key)) for key in (
                "spearman_mean", "top10_overlap_mean", "pairwise_agreement_mean",
                "nn_identity_agreement_mean")},
            "distance_error": {
                "mean_fold_p99": mean_field(("distance_error", "mean_fold_p99")),
                "p99_worst_category": max(float(rec["distance_error"]["p99_worst_fold"])
                                            for rec in recs if rec["distance_error"]["p99_worst_fold"] is not None)
                if any(rec["distance_error"]["p99_worst_fold"] is not None for rec in recs) else None,
                "max_worst_category": max(float(rec["distance_error"]["max_worst_fold"])
                                           for rec in recs if rec["distance_error"]["max_worst_fold"] is not None)
                if any(rec["distance_error"]["max_worst_fold"] is not None for rec in recs) else None,
            },
            "memory": {
                "rows_min": min(int(rec["memory"]["rows_min"]) for rec in recs) if recs else None,
                "rows_max": max(int(rec["memory"]["rows"]) for rec in recs) if recs else None,
                "storage_bytes_mean": int(round(np.mean([rec["memory"]["storage_bytes"] for rec in recs]))) if recs else None,
                "resident_bank_bytes_mean": int(round(np.mean([rec["memory"]["resident_bank_bytes"] for rec in recs]))) if recs else None,
                "storage_ratio_vs_control": mean_field(("memory", "storage_ratio_vs_control")),
                "resident_ratio_vs_control": mean_field(("memory", "resident_ratio_vs_control")),
            },
            "timing": {
                "prepare_seconds_sum": float(np.sum([rec["timing"]["prepare_seconds"] for rec in recs])) if recs else None,
                "score_seconds_sum": float(np.sum([rec["timing"]["score_seconds"] for rec in recs])) if recs else None,
                "total_seconds_sum": float(np.sum([rec["timing"]["total_seconds"] for rec in recs])) if recs else None,
                "score_ratio_vs_control": (
                    float(np.sum([rec["timing"]["score_seconds"] for rec in recs]) /
                          np.sum([categories[i]["methods"]["control_f32"]["timing"]["score_seconds"]
                                  for i in range(len(categories))]))
                    if recs and np.sum([categories[i]["methods"]["control_f32"]["timing"]["score_seconds"]
                                        for i in range(len(categories))]) > 0 else None),
            },
        }
        # Every fold in this probe has one held-out support image per row; rows are
        # therefore the direct check that no candidate deleted a memory unit.
        out["methods"][method]["rows_all_equal_control"] = all(
            c["methods"][method]["memory"]["rows"] == c["methods"]["control_f32"]["memory"]["rows"]
            and c["methods"][method]["memory"]["rows_min"] == c["methods"]["control_f32"]["memory"]["rows_min"]
            for c in categories)
    return out


def _gate_decisions(aggregate: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for method in METHODS:
        rec = aggregate["methods"][method]
        spec = GATES["fp16"] if method.startswith("fp16") else GATES["int8"]
        quality = {
            "rows": bool(rec["rows_all_equal_control"]),
            "macro_ap": rec["macro_ap_delta"] is not None and rec["macro_ap_delta"] >= spec["ap_delta"],
            "worst_episode_ap": rec["worst_episode_ap_delta"] is not None and rec["worst_episode_ap_delta"] >= spec["worst_ap_delta"],
            "distance_p99": rec["distance_error"]["p99_worst_category"] is not None and rec["distance_error"]["p99_worst_category"] <= spec["distance_p99"],
            "nn_identity": rec["rank"]["nn_identity_agreement_mean"] is not None and rec["rank"]["nn_identity_agreement_mean"] >= spec["nn_agreement"],
        }
        storage_ratio = rec["memory"]["storage_ratio_vs_control"]
        if method.startswith("fp16"):
            storage_ok = storage_ratio is not None and storage_ratio <= 0.51
        elif method.startswith("int8"):
            storage_ok = storage_ratio is not None and storage_ratio < 0.30
        else:
            storage_ok = False
        checks[method] = {
            "quality_gate": bool(all(quality.values())), "quality_checks": quality,
            "storage_gate": bool(storage_ok), "storage_ratio": storage_ratio,
            "storage_claim": bool(all(quality.values()) and storage_ok),
        }
    runtime = aggregate["methods"].get("int8_pc_runtime", {})
    runtime_mem = runtime.get("memory", {}).get("resident_ratio_vs_control")
    runtime_time = runtime.get("timing", {}).get("score_ratio_vs_control")
    checks["G-P1_rows"] = all(v["quality_checks"]["rows"] for v in checks.values() if isinstance(v, dict) and "quality_checks" in v)
    checks["G-P4_storage"] = any(v.get("storage_claim", False) for v in checks.values() if isinstance(v, dict))
    checks["G-P5_runtime"] = bool(runtime_mem is not None and runtime_mem < 0.50 and
                                  runtime_time is not None and runtime_time < 1.0 and
                                  checks["int8_pc_runtime"]["quality_gate"])
    checks["runtime_measurement"] = {"resident_ratio": runtime_mem, "score_ratio": runtime_time,
                                     "claim": checks["G-P5_runtime"]}
    return checks


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _write_decision(result: dict[str, Any], failure: str | None = None) -> None:
    path = OUT / "DECISION_CN.md"
    if failure is not None:
        body = ("# Precision 探针结论\n\n"
                "本轮执行失败，已归档失败信息，不能据此宣称量化候选通过。\n\n"
                f"- 错误：`{failure}`\n"
                "- 产物：`RESULTS.json`、`RUN_MANIFEST.json`、`DECISION_CN.md`\n")
        path.write_text(body, encoding="utf-8")
        return
    aggregate = result["aggregate"]
    gates = result["gates"]
    fp16 = aggregate["methods"]["fp16_roundtrip"]
    i8 = aggregate["methods"]["int8_pc_roundtrip"]
    i8rt = aggregate["methods"]["int8_pc_runtime"]
    claims: list[str] = []
    if gates["fp16_roundtrip"]["storage_claim"]:
        claims.append("FP16 roundtrip 达到预注册质量门并提供约 0.5 倍持久化 memory")
    if gates["int8_pc_roundtrip"]["storage_claim"]:
        claims.append("对称 per-channel INT8 解码检索达到预注册质量门并提供低于 0.3 倍持久化 memory")
    if gates["G-P5_runtime"]:
        claims.append("INT8 常驻检索同时满足预注册质量、常驻内存和实测时间门")
    decision = result["decision"]
    completed = "、".join(result.get("categories_completed", []))
    lines = ["# Precision 探针结论", "", f"决策：`{decision}`", "",
             f"范围：support-only，完成类别 {completed}，seed0/k2、每类轮换留出 support 图；未读取 MPDD test。",
             f"候选：FP16 roundtrip、FP16 runtime、对称 per-channel INT8 roundtrip、INT8 runtime；所有候选均保留全部 {result['aggregate']['methods']['control_f32']['memory']['rows_min']} 行 memory。", ""]
    if claims:
        lines += ["可支持的效率结论："] + [f"- {claim}" for claim in claims] + [""]
    else:
        lines += ["本轮没有候选同时通过预注册的质量与存储门；结果作为失败/边界探针归档。", ""]
    lines += ["关键聚合（相对 control_f32）：", "",
              "| 候选 | AP Δ | 最差 episode AP Δ | 距离 p99 最差类 | NN 身份一致率 | 存储比 | 常驻比 | 检索时间比 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        rec = aggregate["methods"][method]
        lines.append("| {m} | {ap} | {worst} | {p99} | {nn} | {sr} | {rr} | {tr} |".format(
            m=method, ap=_fmt(rec["macro_ap_delta"]), worst=_fmt(rec["worst_episode_ap_delta"]),
            p99=_fmt(rec["distance_error"]["p99_worst_category"]),
            nn=_fmt(rec["rank"]["nn_identity_agreement_mean"]),
            sr=_fmt(rec["memory"]["storage_ratio_vs_control"]),
            rr=_fmt(rec["memory"]["resident_ratio_vs_control"]),
            tr=_fmt(rec["timing"]["score_ratio_vs_control"])))
    lines += ["", "解释：roundtrip 候选的常驻 bank 先解码到 float32，因此只支持存储压缩；INT8 runtime 的常驻 bank 保留 int8 和 scale，分块临时 workspace 为固定 256 channels。实际 CPU wall time 以 perf_counter 记录，不能将存储比直接解释为延迟比。", "", "复现命令：", "", "```powershell", "$env:CUDA_VISIBLE_DEVICES=\"\"", "$env:OMP_NUM_THREADS=\"2\"", "$env:MKL_NUM_THREADS=\"2\"", "python scripts/innovation_overnight_20260908/probe_precision.py --cats bracket_black,bracket_brown,bracket_white,connector,metal_plate,tubes", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    val = _finite(value)
    return "NA" if val is None else f"{val:.6f}"


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description="Support-only A1 FP16/INT8 precision probe")
    parser.add_argument("--cats", default=None,
                        help="comma-separated support categories; default is all six MPDD categories")
    parser.add_argument("--output", default=str(OUT), help="output directory")
    args = parser.parse_args()
    OUT = Path(args.output).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [item.strip() for item in args.cats.split(",")] if args.cats else list(CATEGORIES)
    unknown = [cat for cat in cats if cat not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")
    start = time.perf_counter()
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "git_commit": _git_commit(), "python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_threads": int(torch.get_num_threads()), "requested_cpu_threads": 2,
        "seed": 0, "shot": 2, "categories": cats, "methods": list(METHODS),
        "support_cache": "outputs/dynamic_fusion/v14_p1_support",
        "loader": "scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py::load_cells",
        "status": "running",
    }
    _write_json(OUT / "RUN_MANIFEST.json", manifest)
    try:
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is visible to torch despite CUDA_VISIBLE_DEVICES empty")
        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        context = threadpool_limits(limits=2) if threadpool_limits is not None else None
        if context is None:
            context = _NullContext()
        with context:
            for category in cats:
                try:
                    row = run_category(category)
                    results.append(row)
                    print(f"done {category}: {row['wall_seconds']:.2f}s", flush=True)
                except Exception as exc:  # keep partial evidence when one category fails
                    failures.append({"category": category, "error": repr(exc),
                                     "traceback": traceback.format_exc()})
                    print(f"failed {category}: {exc}", flush=True)
        aggregate = _aggregate(results) if results else {"category_count": 0, "methods": {}}
        gates = _gate_decisions(aggregate) if results else {}
        quality_or_storage = any(gates.get(m, {}).get("storage_claim", False)
                                 for m in STORAGE_METHODS)
        decision = "PRECISION_PROBE_PASS" if results and not failures and quality_or_storage else "PRECISION_PROBE_FAIL_ARCHIVE"
        result = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "experiments/dynamic_fusion/innovation_overnight_20260908/precision/PROTOCOL_CN.md",
            "data_role": "support_only; no test masks/images; quantizer fit per held-out normal memory",
            "seed": 0, "shot": 2, "categories_requested": cats, "categories_completed": [r["category"] for r in results],
            "failures": failures, "aggregate": aggregate, "gates": gates,
            "decision": decision, "elapsed_seconds": time.perf_counter() - start,
            "torch_threads": int(torch.get_num_threads()), "cuda_available": bool(torch.cuda.is_available()),
            "results": results,
        }
        _write_json(OUT / "RESULTS.json", result)
        manifest["status"] = "completed" if not failures else "completed_with_failures"
        manifest["elapsed_seconds"] = result["elapsed_seconds"]
        _write_json(OUT / "RUN_MANIFEST.json", manifest)
        _write_decision(result)
        print(f"DECISION={decision}", flush=True)
        return 0 if decision == "PRECISION_PROBE_PASS" else 1
    except Exception as exc:
        error = repr(exc)
        failure_result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                          "decision": "PRECISION_PROBE_FAIL_ARCHIVE", "error": error,
                          "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - start}
        _write_json(OUT / "RESULTS.json", failure_result)
        manifest["status"] = "failed"
        manifest["error"] = error
        manifest["elapsed_seconds"] = failure_result["elapsed_seconds"]
        _write_json(OUT / "RUN_MANIFEST.json", manifest)
        _write_decision(failure_result, failure=error)
        print(f"PRECISION_PROBE_FAIL_ARCHIVE: {error}", flush=True)
        return 1


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
