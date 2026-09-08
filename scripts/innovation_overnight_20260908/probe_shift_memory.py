"""Support-only deployment photometric-shift memory probe.

This script tests whether normal photometric variants from *other* support
images make a held-out support image's nuisance scores look more like its clean
scores.  It is intentionally separate from the old D-DEVA image augmentation
route: it consumes the existing v14 feature cache, uses a leave-image-out
protocol, and includes equal-memory duplicate-clean controls.

The hard deadline is a UTC datetime.  Every category, fold, episode, and arm
checks it; the current rows are written before a deadline stop or an error.
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

# Set process-level CPU/GPU constraints before importing numpy/torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[2]
OLD_LOADER_DIR = ROOT / "scripts" / "innovation_t3_efficiency_20260905"
sys.path.insert(0, str(OLD_LOADER_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import probe_e1_compress as E1  # noqa: E402  (existing v14 support loader pieces)

try:  # noqa: E402
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "shift_memory"
CACHE = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support"
CATEGORIES = ("bracket_black", "bracket_brown", "bracket_white", "connector",
              "metal_plate", "tubes")
METHODS = ("clean_only", "augmented_all5", "duplicate_clean_equal_all5",
           "augmented_loo_family", "duplicate_clean_equal_loo")
SYN_KINDS = ("cutpaste", "local_erasure", "thin_scratch")
NUI_FAMILIES = ("exposure", "gamma", "white_balance", "lr_brightness_gradient", "specular_blob")
NUI_INDICES = (0, 3, 6, 9, 12)  # one pre-registered strength-0 variant per family
GRID = 16
DIM = 1536
CELL_SIZE = 64  # 1024 / 16; fixed nearest repeat for original-mask AP
HARD_DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)


class DeadlineReached(RuntimeError):
    """The fixed UTC stop was reached at a safe loop boundary."""


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _check_deadline(stage: str) -> None:
    # Keep the comparison explicit so the stop cannot depend on heartbeat timing.
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
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _row_norm(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norm, np.float32(1e-12))


def _load_v14(cat: str, branch: str, shot: int) -> dict[str, np.ndarray]:
    path = CACHE / f"v14_p1_support_{branch}_s0_k{shot}" / f"{cat}.npz"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        keys = ("ref_rel", "clean_feat", "syn_feat", "syn_masks", "nui_feat",
                "syn_kinds", "nui_keys", "syn_seeds", "grid_size", "branch")
        return {key: np.asarray(z[key]).copy() for key in keys}


def _pool16(x: np.ndarray) -> np.ndarray:
    # Reuse the established v14 32 -> 16 cell pooling implementation.
    return np.asarray(E1._pool16_unit(x), dtype=np.float32)


def load_cells(cat: str, shot: int) -> dict[str, Any]:
    """Load and align the v14 support cache without reading test data."""
    dino = _load_v14(cat, "dino", shot)
    clip = _load_v14(cat, "clip", shot)
    if dino["branch"].item() != "dino" or clip["branch"].item() != "clip":
        raise ValueError(f"branch metadata mismatch for {cat} k{shot}")
    if not np.array_equal(dino["ref_rel"].astype(str), clip["ref_rel"].astype(str)):
        raise ValueError(f"reference order mismatch for {cat} k{shot}")
    if not np.array_equal(dino["nui_keys"].astype(str), clip["nui_keys"].astype(str)):
        raise ValueError(f"nuisance key mismatch for {cat} k{shot}")
    if not np.array_equal(dino["syn_kinds"].astype(str), clip["syn_kinds"].astype(str)):
        raise ValueError(f"synthetic kind mismatch for {cat} k{shot}")

    d_clean = _pool16(dino["clean_feat"])
    d_syn = _pool16(dino["syn_feat"].reshape(-1, 32, 32, 768))
    d_nui = _pool16(dino["nui_feat"].reshape(-1, 32, 32, 768))

    c_clean_raw = dino["clean_feat"]  # only for shape clarity below
    del c_clean_raw
    c_clean32 = E1.resize_patches(clip["clean_feat"].reshape(-1, *clip["clean_feat"].shape[1:]), (32, 32))
    c_syn32 = E1.resize_patches(clip["syn_feat"].reshape(-1, *clip["syn_feat"].shape[2:]), (32, 32))
    c_nui32 = E1.resize_patches(clip["nui_feat"].reshape(-1, *clip["nui_feat"].shape[2:]), (32, 32))
    k = int(d_clean.shape[0])
    c_clean = _pool16(c_clean32).reshape(k, GRID * GRID, 768)
    c_syn = _pool16(c_syn32).reshape(k, 9, GRID * GRID, 768)
    c_nui = _pool16(c_nui32).reshape(k, 15, GRID * GRID, 768)

    # E1.a1 is the frozen 0.5/0.5 concat + row L2 path used by the prior probe.
    clean = np.asarray(E1.a1(d_clean, c_clean.reshape(k, GRID, GRID, 768)), dtype=np.float32).reshape(k, -1, DIM)
    syn = np.asarray(E1.a1(d_syn.reshape(-1, GRID, GRID, 768), c_syn.reshape(-1, GRID, GRID, 768)), dtype=np.float32).reshape(k, 9, -1, DIM)
    nui = np.asarray(E1.a1(d_nui.reshape(-1, GRID, GRID, 768), c_nui.reshape(-1, GRID, GRID, 768)), dtype=np.float32).reshape(k, 15, -1, DIM)
    masks = np.asarray(dino["syn_masks"], dtype=np.uint8)
    syn_kinds = tuple(dino["syn_kinds"].astype(str).tolist())
    nui_keys = tuple(dino["nui_keys"].astype(str).tolist())
    if masks.shape != (k, len(SYN_KINDS) * 3, 1024, 1024):
        raise ValueError(f"unexpected original synthetic mask shape for {cat}: {masks.shape}")
    if syn_kinds != SYN_KINDS:
        raise ValueError(f"unexpected synthetic order for {cat}: {syn_kinds}")
    if tuple(nui_keys[i] for i in NUI_INDICES) != tuple(f"{family}_0" for family in NUI_FAMILIES):
        raise ValueError(f"unexpected nuisance order for {cat}: {nui_keys}")
    return {"K": k, "clean": clean, "syn": syn, "nui": nui, "masks": masks,
            "ref_rel": dino["ref_rel"].astype(str).tolist(),
            "syn_kinds": syn_kinds, "nui_keys": nui_keys}


def _prepare_banks(clean: np.ndarray, nui: np.ndarray, bank_idx: list[int]) -> dict[str, Any]:
    clean_bank = _row_norm(clean[bank_idx].reshape(-1, DIM))
    nui_bank = _row_norm(nui[bank_idx].reshape(-1, 15, DIM))
    # One fixed strength per nuisance family; all five families are available to
    # the practical all5 candidate, regardless of the query nuisance key.
    selected = nui_bank[:, list(NUI_INDICES), :].reshape(-1, DIM)
    all5 = np.concatenate([clean_bank, selected], axis=0)
    dup_all5 = np.tile(clean_bank, (6, 1))
    loo_aug: dict[int, np.ndarray] = {}
    loo_dup: dict[int, np.ndarray] = {}
    for family_idx in range(5):
        others = [idx for idx in range(5) if idx != family_idx]
        chosen = nui_bank[:, [NUI_INDICES[idx] for idx in others], :].reshape(-1, DIM)
        loo_aug[family_idx] = np.concatenate([clean_bank, chosen], axis=0)
        loo_dup[family_idx] = np.tile(clean_bank, (5, 1))
    return {"clean_only": clean_bank, "augmented_all5": all5,
            "duplicate_clean_equal_all5": dup_all5,
            "augmented_loo_family": loo_aug,
            "duplicate_clean_equal_loo": loo_dup}


def _score16(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = _row_norm(np.asarray(query, dtype=np.float32).reshape(-1, DIM))
    b = np.asarray(bank, dtype=np.float32)
    sim = q @ b.T
    return (1.0 - np.max(sim, axis=1)).astype(np.float32)


def _score_original_grid(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    return _score16(query, bank).reshape(GRID, GRID)


def _score_to_original(score16: np.ndarray) -> np.ndarray:
    # Exact 16 -> 1024 nearest repeat makes the original thin-scratch mask
    # observable without inventing sub-cell feature scores.
    return np.repeat(np.repeat(np.asarray(score16, dtype=np.float32), CELL_SIZE, axis=0),
                     CELL_SIZE, axis=1)


def _ap(mask: np.ndarray | None, score_map: np.ndarray | None) -> float | None:
    if mask is None or score_map is None:
        return None
    y = (np.asarray(mask).reshape(-1) > 0).astype(np.uint8)
    if y.sum() == 0 or y.sum() == y.size:
        return None
    return float(average_precision_score(y, np.asarray(score_map, dtype=np.float64).reshape(-1)))


def _auc(clean_score: np.ndarray, nuisance_scores: list[np.ndarray]) -> float | None:
    if not nuisance_scores:
        return None
    n = np.concatenate(nuisance_scores).astype(np.float64)
    c = np.asarray(clean_score, dtype=np.float64).reshape(-1)
    y = np.concatenate([np.zeros(c.size, dtype=np.uint8), np.ones(n.size, dtype=np.uint8)])
    try:
        return float(roc_auc_score(y, np.concatenate([c, n])))
    except ValueError:
        return None


def _normal_distribution(clean_score: np.ndarray, nuisance_by_family: dict[int, list[np.ndarray]]) -> dict[str, Any]:
    c = np.asarray(clean_score, dtype=np.float64).reshape(-1)
    n_all = [x for values in nuisance_by_family.values() for x in values]
    n = np.concatenate(n_all).astype(np.float64) if n_all else np.empty(0, dtype=np.float64)
    if not n.size:
        return {"auc": None, "clean_mean": None, "clean_p95": None,
                "nuisance_mean": None, "nuisance_p95": None,
                "mean_abs_shift": None, "p95_abs_shift": None,
                "nuisance_p99_exceed_rate": None, "by_family": {}}
    clean_mean = float(np.mean(c))
    clean_p95 = float(np.percentile(c, 95))
    nui_mean = float(np.mean(n))
    nui_p95 = float(np.percentile(n, 95))
    clean_p99 = float(np.percentile(c, 99))
    by_family: dict[str, Any] = {}
    for family_idx, values in nuisance_by_family.items():
        if not values:
            continue
        x = np.concatenate(values).astype(np.float64)
        by_family[NUI_FAMILIES[family_idx]] = {
            "n": int(x.size), "mean": float(np.mean(x)), "p95": float(np.percentile(x, 95)),
            "mean_abs_shift": float(abs(np.mean(x) - clean_mean)),
            "p95_abs_shift": float(abs(np.percentile(x, 95) - clean_p95)),
            "p99_exceed_rate": float(np.mean(x > clean_p99)),
        }
    return {
        "auc": _auc(c, n_all), "clean_mean": clean_mean, "clean_p95": clean_p95,
        "nuisance_mean": nui_mean, "nuisance_p95": nui_p95,
        "mean_abs_shift": float(abs(nui_mean - clean_mean)),
        "p95_abs_shift": float(abs(nui_p95 - clean_p95)),
        "nuisance_p99_exceed_rate": float(np.mean(n > clean_p99)),
        "by_family": by_family,
    }


def _method_bank(method: str, banks: dict[str, Any], kind: str,
                 family_idx: int | None) -> tuple[np.ndarray, bool]:
    if method in ("augmented_loo_family", "duplicate_clean_equal_loo") and kind == "nuisance" and family_idx is not None:
        # This is explicitly an evaluation-only leave-family-out diagnostic.
        return banks[method][family_idx], True
    if method == "augmented_loo_family":
        # Synthetic and clean queries do not have a nuisance family to exclude;
        # use the deployable all-five-family bank for those episodes.
        return banks["augmented_all5"], False
    if method == "duplicate_clean_equal_loo":
        return banks["duplicate_clean_equal_all5"], False
    return banks[method], False


def _episode_rows(category: str, shot: int, held_out: int, clean: np.ndarray,
                  syn: np.ndarray, nui: np.ndarray, masks: np.ndarray,
                  banks: dict[str, Any], methods: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores_by_method: dict[str, dict[str, Any]] = {
        method: {"clean": None, "nuisance": {idx: [] for idx in range(5)}} for method in methods
    }
    # Synthetic episodes remain support-only and keep the original renderer mask.
    episodes: list[dict[str, Any]] = [{"kind": "clean", "episode": -1, "family_idx": None,
                                      "query": clean[held_out], "mask": None, "key": "clean"}]
    for kind_idx, kind in enumerate(SYN_KINDS):
        for seed in range(3):
            e = kind_idx * 3 + seed
            episodes.append({"kind": "synthetic", "episode": e, "family_idx": None,
                             "synthetic_kind": kind, "seed": seed,
                             "query": syn[held_out, e], "mask": masks[held_out, e],
                             "key": f"syn_{kind}_s{seed}"})
    for e in range(15):
        family_idx = e // 3
        episodes.append({"kind": "nuisance", "episode": e, "family_idx": family_idx,
                         "nuisance_key": NUI_FAMILIES[family_idx] + f"_{e % 3}",
                         "query": nui[held_out, e], "mask": None,
                         "key": f"nui_{e}"})

    for episode in episodes:
        _check_deadline(f"before_{category}_h{held_out}_{episode['key']}")
        for method in methods:
            _check_deadline(f"before_{category}_h{held_out}_{episode['key']}_{method}")
            bank, family_excluded = _method_bank(method, banks, episode["kind"], episode["family_idx"])
            t0 = time.perf_counter()
            score16 = _score_original_grid(episode["query"], bank)
            score_ms = (time.perf_counter() - t0) * 1000.0
            score_flat = score16.reshape(-1)
            score_map = _score_to_original(score16) if episode["kind"] == "synthetic" else None
            ap = _ap(episode["mask"], score_map)
            row = {
                "category": category, "shot": shot, "seed": 0, "held_out": held_out,
                "kind": episode["kind"], "episode": int(episode["episode"]),
                "synthetic_kind": episode.get("synthetic_kind"), "nuisance_key": episode.get("nuisance_key"),
                "family_excluded_for_diagnostic": bool(family_excluded), "method": method,
                "query_cells": int(score_flat.size), "memory_rows": int(bank.shape[0]),
                "memory_images": int(bank.shape[0] // 256), "memory_bytes_float32": int(bank.nbytes),
                "score_mean": float(np.mean(score_flat)), "score_p95": float(np.percentile(score_flat, 95)),
                "score_p99": float(np.percentile(score_flat, 99)), "ap": ap,
                "mask_area": float(np.mean(episode["mask"])) if episode["mask"] is not None else None,
                "random_area_ap": float(np.mean(episode["mask"])) if episode["mask"] is not None else None,
                "mask_source": "v14_support_syn_masks_original_renderer" if episode["mask"] is not None else None,
                "mask_resolution": [1024, 1024] if episode["mask"] is not None else None,
                "score_grid": [GRID, GRID],
                "score_ms": float(score_ms), "support_only": True,
            }
            rows.append(row)
            if episode["kind"] == "clean":
                scores_by_method[method]["clean"] = score_flat.copy()
            elif episode["kind"] == "nuisance":
                scores_by_method[method]["nuisance"][episode["family_idx"]].append(score_flat.copy())
    fold_summary: dict[str, Any] = {"category": category, "shot": shot, "held_out": held_out,
                                    "methods": {}}
    for method in methods:
        dist = _normal_distribution(scores_by_method[method]["clean"], scores_by_method[method]["nuisance"])
        fold_summary["methods"][method] = {"normal_distribution": dist}
    return rows, fold_summary


def _prepare_memory_metadata(shot: int) -> dict[str, Any]:
    base_rows = (shot - 1) * GRID * GRID
    base_bytes = base_rows * DIM * 4
    factors = {"clean_only": 1, "augmented_all5": 6, "duplicate_clean_equal_all5": 6,
               "augmented_loo_family": 5, "duplicate_clean_equal_loo": 5}
    return {method: {"memory_rows": base_rows * factor,
                     "memory_bytes_float32": base_bytes * factor,
                     "memory_factor_vs_clean": factor}
            for method, factor in factors.items()}


def _aggregate(rows: list[dict[str, Any]], folds: list[dict[str, Any]], shot: int) -> dict[str, Any]:
    out: dict[str, Any] = {"shot": shot, "categories_completed": sorted({r["category"] for r in rows}),
                           "synthetic_valid_episodes": 0, "methods": {}}
    keys = {(r["category"], r["held_out"], r["episode"]) for r in rows if r["kind"] == "synthetic" and r["ap"] is not None}
    out["synthetic_valid_episodes"] = len(keys)
    mem = _prepare_memory_metadata(shot)
    for method in METHODS:
        syn = [r for r in rows if r["kind"] == "synthetic" and r["ap"] is not None and r["method"] == method]
        clean_by_episode = {(r["category"], r["held_out"], r["episode"]): r["ap"]
                            for r in rows if r["kind"] == "synthetic" and r["ap"] is not None and r["method"] == "clean_only"}
        deltas = [float(r["ap"] - clean_by_episode[(r["category"], r["held_out"], r["episode"])])
                  for r in syn if (r["category"], r["held_out"], r["episode"]) in clean_by_episode]
        fam_ap = {}
        for kind in SYN_KINDS:
            vals = [r["ap"] for r in syn if r.get("synthetic_kind") == kind and r["ap"] is not None]
            fam_ap[kind] = float(np.mean(vals)) if vals else None
        fold_method = [f["methods"][method]["normal_distribution"] for f in folds if method in f["methods"]]
        def mean_dist(key: str) -> float | None:
            vals = [d[key] for d in fold_method if d.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        normal = {key: mean_dist(key) for key in ("auc", "mean_abs_shift", "p95_abs_shift", "nuisance_p99_exceed_rate")}
        clean_normal = []
        for f in folds:
            if "clean_only" in f["methods"] and method != "clean_only":
                clean_normal.append(f["methods"]["clean_only"]["normal_distribution"])
        normal_clean = {
            key: float(np.mean([d[key] for d in clean_normal if d.get(key) is not None]))
            if any(d.get(key) is not None for d in clean_normal) else None
            for key in ("auc", "mean_abs_shift", "p95_abs_shift", "nuisance_p99_exceed_rate")
        }
        out["methods"][method] = {
            "synthetic_macro_ap": float(np.mean([r["ap"] for r in syn])) if syn else None,
            "synthetic_family_ap": fam_ap,
            "synthetic_macro_delta_vs_clean": float(np.mean(deltas)) if deltas else None,
            "synthetic_worst_episode_delta_vs_clean": float(np.min(deltas)) if deltas else None,
            "normal_distribution": normal,
            "normal_delta_vs_clean": {
                key: (normal[key] - normal_clean[key]) if normal[key] is not None and normal_clean[key] is not None else None
                for key in normal
            },
            "memory": mem[method],
            "timing": {
                "score_ms_sum": float(np.sum([r["score_ms"] for r in rows if r["method"] == method])),
                "score_ms_mean": float(np.mean([r["score_ms"] for r in rows if r["method"] == method]))
                if any(r["method"] == method for r in rows) else None,
            },
        }
    # Equal-budget controls are direct paired checks, independent of absolute AP.
    for aug, dup in (("augmented_all5", "duplicate_clean_equal_all5"),
                     ("augmented_loo_family", "duplicate_clean_equal_loo")):
        a = out["methods"][aug]
        d = out["methods"][dup]
        a["synthetic_delta_vs_equal_budget_control"] = (
            a["synthetic_macro_ap"] - d["synthetic_macro_ap"]
            if a["synthetic_macro_ap"] is not None and d["synthetic_macro_ap"] is not None else None)
        a["normal_p95_abs_shift_delta_vs_equal_budget_control"] = (
            a["normal_distribution"]["p95_abs_shift"] - d["normal_distribution"]["p95_abs_shift"]
            if a["normal_distribution"]["p95_abs_shift"] is not None and d["normal_distribution"]["p95_abs_shift"] is not None else None)
    return out


def _decision(aggregate: dict[str, Any], status: str) -> tuple[str, dict[str, Any]]:
    if status != "complete":
        return "PARTIAL_STOP_NO_CLAIM", {"status": status}
    a = aggregate["methods"].get("augmented_all5", {})
    syn_ok = (a.get("synthetic_macro_delta_vs_clean") is not None and
              a["synthetic_macro_delta_vs_clean"] >= -0.01 and
              a.get("synthetic_worst_episode_delta_vs_clean") is not None and
              a["synthetic_worst_episode_delta_vs_clean"] >= -0.05)
    nd = a.get("normal_delta_vs_clean", {})
    normal_cleaner = (nd.get("mean_abs_shift") is not None and nd["mean_abs_shift"] <= 0.0 and
                      nd.get("p95_abs_shift") is not None and nd["p95_abs_shift"] <= 0.0 and
                      (nd["mean_abs_shift"] < 0.0 or nd["p95_abs_shift"] < 0.0))
    eq = a.get("normal_p95_abs_shift_delta_vs_equal_budget_control")
    equal_budget_ok = eq is not None and eq <= 0.0
    checks = {"G-SYN": syn_ok, "G-SHIFT_clean": normal_cleaner,
              "G-SHIFT_equal_budget": equal_budget_ok}
    if all(checks.values()):
        return "SHIFT_MEMORY_SIGNAL_SUPPORT_ONLY", checks
    return "NO_POSITIVE_SHIFT_MEMORY_SIGNAL", checks


def _write_partial(rows: list[dict[str, Any]], folds: list[dict[str, Any]], protocol: dict[str, Any],
                   status: str, error: str | None = None) -> None:
    payload = {"protocol": protocol, "status": status, "error": error,
               "updated_utc": _utc_now(), "rows": rows, "fold_summaries": folds}
    _write_json(OUT / "PARTIAL_RESULTS.json", payload)
    with (OUT / "PER_EPISODE.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_value(row), ensure_ascii=False) + "\n")


def _write_decision(protocol: dict[str, Any], rows: list[dict[str, Any]], folds: list[dict[str, Any]],
                    status: str, elapsed_s: float, error: str | None) -> dict[str, Any]:
    aggregate = _aggregate(rows, folds, int(protocol["shot"])) if rows else {"shot": protocol["shot"], "categories_completed": [], "synthetic_valid_episodes": 0, "methods": {}}
    decision, checks = _decision(aggregate, status)
    result = {"schema_version": 1, "protocol": protocol, "status": status,
              "decision": decision, "checks": checks, "error": error,
              "elapsed_seconds": float(elapsed_s), "aggregate": aggregate,
              "fold_summaries": folds, "rows": rows}
    _write_json(OUT / "RESULTS.json", result)
    lines = ["# Shift-memory 探针结论", "", f"状态：`{status}`；决策：`{decision}`。", "",
             "本轮只使用 V14 support-only cache，seed0，未读取 MPDD test；每折 memory 只来自非 heldout support 图。部署候选 `augmented_all5` 不读取 query nuisance 族；`augmented_loo_family` 仅是留族交叉诊断。", ""]
    if aggregate.get("methods"):
        lines += ["关键结果（相对 clean-only）：", "", "| 候选 | synthetic AP Δ | 最差 episode ΔAP | nuisance mean abs shift Δ | nuisance p95 abs shift Δ | equal-budget p95 Δ | memory 行数 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            m = aggregate["methods"][method]
            nd = m.get("normal_delta_vs_clean", {})
            lines.append(f"| {method} | {_fmt(m.get('synthetic_macro_delta_vs_clean'))} | {_fmt(m.get('synthetic_worst_episode_delta_vs_clean'))} | {_fmt(nd.get('mean_abs_shift'))} | {_fmt(nd.get('p95_abs_shift'))} | {_fmt(m.get('normal_p95_abs_shift_delta_vs_equal_budget_control'))} | {m.get('memory', {}).get('memory_rows', 'NA')} |")
        lines += ["", "解释：原始 1024 × 1024 mask 用固定 16 × 16 score 的 64 × 64 nearest repeat 评价，因此 thin_scratch 不因 16-grid 阈值为空而被删掉。equal-budget duplicate-clean 只控制 memory 行数增加，不增加光度内容。", ""]
    if error:
        lines += [f"错误/截止原因：`{error}`", ""]
    lines += ["复现命令：", "", "```powershell", "$env:CUDA_VISIBLE_DEVICES=\"\"", "$env:OMP_NUM_THREADS=\"2\"", "$env:MKL_NUM_THREADS=\"2\"", "$env:OPENBLAS_NUM_THREADS=\"2\"", f"python scripts/innovation_overnight_20260908/probe_shift_memory.py --shot {protocol['shot']} --cats {','.join(protocol['categories'])}", "```", "", "详细逐 episode 记录：`PER_EPISODE.jsonl`；运行中断时保留 `PARTIAL_RESULTS.json`。"]
    (OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _fmt(value: Any) -> str:
    v = _finite(value)
    return "NA" if v is None else f"{v:.6f}"


def _protocol(cats: list[str], shot: int) -> dict[str, Any]:
    return {
        "schema_version": 1, "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": _sha256_file(Path(__file__)), "created_utc": _utc_now(),
        "git_commit": _git_commit(), "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "torch_threads_requested": 2, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": 0, "shot": shot, "categories": cats, "methods": list(METHODS),
        "cache": "outputs/dynamic_fusion/v14_p1_support",
        "loader": "scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py components: _pool16_unit, resize_patches, a1",
        "support_only": True, "test_images_read": False, "test_labels_read": False,
        "nuisance_families": list(NUI_FAMILIES), "nuisance_indices": list(NUI_INDICES),
        "nuisance_selection": "one fixed strength-0 variant per family; no result-dependent choice",
        "synthetic_order": list(SYN_KINDS), "synthetic_mask_resolution": [1024, 1024],
        "score_grid": [GRID, GRID], "score_to_mask": "64 x 64 nearest repeat",
        "hard_deadline_utc": HARD_DEADLINE_UTC.isoformat(),
        "memory_rule": "heldout support image excluded; only other support images' clean/nuisance cells",
        "old_route_audit": "D-DEVA photometric augmentation was a separate image re-encode + equivariance filter; see AUDIT_CN.md",
    }


def run(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [x.strip() for x in args.cats.split(",")] if args.cats else list(CATEGORIES)
    unknown = [x for x in cats if x not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown categories: {unknown}")
    protocol = _protocol(cats, int(args.shot))
    protocol["started_utc"] = _utc_now()
    _write_json(OUT / "PROTOCOL.json", protocol)
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    status = "complete"
    error = None
    started = time.perf_counter()
    context = threadpool_limits(limits=2) if threadpool_limits is not None else _NullContext()
    try:
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if torch.cuda.is_available():
            raise RuntimeError("CUDA available despite CUDA_VISIBLE_DEVICES empty")
        with context:
            for category in cats:
                _check_deadline(f"before_category_{category}")
                cells = load_cells(category, int(args.shot))
                for held_out in range(int(cells["K"])):
                    _check_deadline(f"before_fold_{category}_h{held_out}")
                    bank_idx = [idx for idx in range(int(cells["K"])) if idx != held_out]
                    t_bank = time.perf_counter()
                    banks = _prepare_banks(cells["clean"], cells["nui"], bank_idx)
                    bank_build_ms = (time.perf_counter() - t_bank) * 1000.0
                    fold_rows, fold_summary = _episode_rows(
                        category, int(args.shot), held_out, cells["clean"], cells["syn"],
                        cells["nui"], cells["masks"], banks, METHODS)
                    for row in fold_rows:
                        row["memory_build_ms"] = float(bank_build_ms)
                        row["bank_source_indices"] = bank_idx
                        row["heldout_ref"] = cells["ref_rel"][held_out]
                        row["memory_ref_indices"] = bank_idx
                    rows.extend(fold_rows)
                    fold_summary["memory_build_ms"] = float(bank_build_ms)
                    fold_summary["memory_rows"] = {method: int(
                        banks[method][0].shape[0] if isinstance(banks[method], dict) else banks[method].shape[0]
                    ) for method in METHODS}
                    folds.append(fold_summary)
                    _write_partial(rows, folds, protocol, "running")
                    del banks, fold_rows, fold_summary
                del cells
    except DeadlineReached as exc:
        status = "partial_deadline"
        error = str(exc)
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    result = _write_decision(protocol, rows, folds, status, elapsed, error)
    protocol["finished_utc"] = _utc_now()
    protocol["status"] = status
    protocol["elapsed_seconds"] = float(elapsed)
    protocol["rows"] = len(rows)
    _write_json(OUT / "PROTOCOL.json", protocol)
    _write_json(OUT / "RUN_MANIFEST.json", {
        "schema_version": 1, "status": status, "started_utc": protocol["started_utc"],
        "finished_utc": protocol["finished_utc"], "elapsed_seconds": float(elapsed),
        "categories_requested": cats, "categories_completed": sorted({r["category"] for r in rows}),
        "shot": int(args.shot), "seed": 0, "rows": len(rows), "error": error,
        "script_sha256": protocol["script_sha256"], "git_commit": protocol["git_commit"],
        "torch_threads": int(torch.get_num_threads()), "cuda_available": bool(torch.cuda.is_available()),
        "decision": result["decision"],
    })
    _write_partial(rows, folds, protocol, status, error)
    print(json.dumps({"status": status, "decision": result["decision"], "rows": len(rows),
                      "elapsed_seconds": elapsed}, ensure_ascii=False), flush=True)
    return 0 if status in ("complete", "partial_deadline") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, choices=(2, 4), default=2)
    parser.add_argument("--cats", default=None)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
