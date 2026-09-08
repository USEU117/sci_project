"""Bounded native-image high-frequency probe for the 2026-09-08 run.

The old v10 STR route already tested a two-level Haar residual on 448-pixel
maps, with gray and color-opponent channels, against real A1-missed regions.
That route is archived negative.  This file deliberately audits a different,
small engineering question rather than presenting a new frequency method:

* construct a fixed 64 x 64 descriptor directly from each 1024 x 1024 support
  image's grayscale pixels;
* use six local high-frequency / gradient statistics after subtracting the
  image-wide grayscale mean;
* compare against a two-statistic centered raw-gray control and the cached full
  DINO 32 x 32 descriptor;
* fit per-dimension median/MAD scale from the other clean support image only,
  then use a global nearest-neighbour memory.

The default run is fixed to bracket_black and metal_plate, seed 0, shot 2,
and the 24 masks shared with the tiling probe: two families (thin_scratch and
cutpaste), three v14 renderer seeds, and two leave-one-out query images per
class.  Every synthetic and nuisance query computes its own descriptor from
its own rendered image; no clean counterpart is used to score synthetic data.
No MPDD test image or test label is read, no parameters are selected from AP,
and no main table is written.

Reproduce from the repository root with:

    .venv-anomalyclip\\Scripts\\python.exe \\
        scripts\\innovation_overnight_20260908\\probe_highfreq.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep all numeric work on the requested small CPU budget.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[2]
for p in (
    ROOT,
    ROOT / "scripts",
    ROOT / "scripts" / "innovation_v12_new_observables",
    ROOT / "scripts" / "innovation_v14_decisive_validation_20260905",
):
    sys.path.insert(0, str(p))

import ntof_render as R  # noqa: E402
from export_p1_support_variants import _variant_seed  # noqa: E402
from v14_common import (  # noqa: E402
    DATA_ROOT,
    assert_fit_ids_are_support,
    load_manifest,
    support_paths,
)


EXP_OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "highfreq"
CACHE = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support"
CATEGORIES = ("bracket_black", "metal_plate")
SHOT = 2
SEED = "0"
RAW_SIZE = 1024
GRID = 64
CELL = RAW_SIZE // GRID
DINO_GRID = 32
SYN_KINDS = ("thin_scratch", "cutpaste")
SYN_SEEDS = (0, 1, 2)
NUI_KEYS = tuple(R.REF_KEYS)
THREADS = 2
RAW_SIGMA = 1.0
MEMORY_MAD_FLOOR = 1.0e-6
MASK_INTERPOLATION = "linear"
DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)
METHODS = ("raw_hf_grad64", "raw_only64", "dino_full32")
SCRIPT_VERSION = "highfreq_probe_v1"
PROTOCOL_VERSION = "highfreq_native64_support_probe_v1"


class DeadlineReached(RuntimeError):
    """Run boundary deadline reached; partial artifacts remain inspectable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_deadline(stage: str) -> None:
    if datetime.now(timezone.utc) >= DEADLINE_UTC:
        raise DeadlineReached(stage)


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")


def _load_rgb(rel: str) -> np.ndarray:
    rel = str(rel).replace("\\", "/")
    if "/test/" in f"/{rel}":
        raise ValueError(f"test image is outside this support-only probe: {rel}")
    path = DATA_ROOT / rel
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (RAW_SIZE, RAW_SIZE):
        raise ValueError(f"expected {RAW_SIZE} x {RAW_SIZE}, got {rgb.shape} for {rel}")
    return rgb


def _protocol() -> dict[str, Any]:
    """Return frozen choices; this is written before loading any query result."""
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": _utc_now(),
        "role": "support-only low-level descriptor probe; no test image or test label",
        "scope": {
            "dataset": "MPDD",
            "categories": list(CATEGORIES),
            "seed": int(SEED),
            "shot": SHOT,
            "support_selection": "manifest seed0 k2; each query support image held out from memory",
            "synthetic_families": list(SYN_KINDS),
            "synthetic_seeds": list(SYN_SEEDS),
            "unique_synthetic_masks": 24,
            "nuisance_variants_per_query": len(NUI_KEYS),
        },
        "old_route_audit": {
            "old_script": "scripts/innovation_v10_portfolio/run_r0_str.py",
            "old_module": "src/industrial_ad/innovation_v10_portfolio/spectral.py",
            "old_result": "experiments/dynamic_fusion/innovation_v10_portfolio/str/R0_RESULT.json",
            "old_mechanism": "2-level Haar robust residual on 448 map; gray + R-G + B-Y; real A1-missed-region diagnostic",
            "old_decision": "FAIL_ARCHIVED; mean delta STR-A1 -0.027368; positive 1/6 categories",
            "overlap": "both are raw image high-frequency/gradient evidence",
            "difference": "this probe uses 1024-native 64-grid gray statistics, k2 support-derived 24 masks, global NN, memory-only scaling, raw/DINO controls",
            "novelty_boundary": "engineering validation only; high-frequency/gradient representation itself is not claimed novel",
        },
        "input": {
            "raw_shape": [RAW_SIZE, RAW_SIZE, 3],
            "raw_source": "MPDD train/good support original images only",
            "cache": "outputs/dynamic_fusion/v14_p1_support (DINO clean/synthetic/nuisance features and mask parity)",
            "mask_rule": "R.render_synthetic with existing v14 _variant_seed; exact mask parity asserted when cache exists",
        },
        "descriptor": {
            "raw_grid": [GRID, GRID],
            "cell_pixels": [CELL, CELL],
            "gray": "RGB luminance 0.299 R + 0.587 G + 0.114 B, divided by 255",
            "photometric_mean_removal": "subtract one image-wide grayscale mean before local statistics",
            "high_frequency": "gray_centered - GaussianBlur(gray_centered, sigma=1.0)",
            "gradient": "Sobel ksize=3 on gray_centered",
            "raw_hf_grad64_components": [
                "cell RMS highpass",
                "cell mean absolute highpass",
                "cell mean absolute gx",
                "cell mean absolute gy",
                "cell mean gradient magnitude",
                "cell gradient-magnitude standard deviation",
            ],
            "raw_hf_grad64_dim": 6,
            "raw_only64_components": ["cell mean centered gray", "cell gray standard deviation"],
            "raw_only64_dim": 2,
            "dino_full32": "cached frozen DINO 32 x 32 x 768 descriptor; no raw re-encoding",
        },
        "normalization_and_score": {
            "memory_scale": "per-dimension median/MAD estimated from the other clean support image cells only",
            "mad_multiplier": 1.4826,
            "mad_floor": MEMORY_MAD_FLOOR,
            "row_normalization": "L2 after memory-only scale normalization",
            "score": "global NN anomaly score = 1 - maximum cosine over the other clean support image",
            "map": f"bilinear {GRID}-grid or native {DINO_GRID}-grid score map to 1024 x 1024",
        },
        "methods": {
            "raw_hf_grad64": "candidate six-statistic native raw descriptor",
            "raw_only64": "single raw-grayscale two-statistic control",
            "dino_full32": "single frozen DINO full-image descriptor control",
        },
        "evaluation": {
            "synthetic": "original 1024 x 1024 renderer mask; Pixel-AP; thin_scratch and cutpaste only",
            "normal": "cell-level ROC-AUC of clean query versus 15 photometric nuisance queries; 0.5 is stable",
            "normal_auxiliary": "mean nuisance score minus mean clean score",
            "parameter_selection": "none; all descriptor and normalization constants frozen before AP",
            "main_table": False,
        },
        "resources": {"device": "cpu", "threads": THREADS, "gpu_used": False},
        "hard_deadline_beijing": "2026-09-08T07:00:00+08:00",
        "hard_deadline_utc": DEADLINE_UTC.isoformat(),
    }


def _freeze_protocol(path: Path) -> dict[str, Any]:
    p = _protocol()
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old_cmp = dict(old)
        new_cmp = dict(p)
        old_cmp.pop("created_utc", None)
        new_cmp.pop("created_utc", None)
        if old_cmp != new_cmp:
            raise RuntimeError(f"Frozen high-frequency protocol mismatch: {path}")
        return old
    _write_json(path, p)
    return p


def _gray(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return g - float(g.mean())


def _cell_stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed raw and high-frequency/gradient 64 x 64 descriptors."""
    if a.shape != (RAW_SIZE, RAW_SIZE):
        raise ValueError(f"expected grayscale {RAW_SIZE} x {RAW_SIZE}, got {a.shape}")
    hp = a - cv2.GaussianBlur(a, ksize=(0, 0), sigmaX=RAW_SIGMA, sigmaY=RAW_SIGMA)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))
    shape = (GRID, CELL, GRID, CELL)

    def mean_cell(x: np.ndarray) -> np.ndarray:
        return x.reshape(*shape).mean(axis=(1, 3))

    def std_cell(x: np.ndarray) -> np.ndarray:
        return x.reshape(*shape).std(axis=(1, 3))

    raw = np.stack([mean_cell(a), std_cell(a)], axis=-1).astype(np.float32)
    hf = np.stack(
        [
            np.sqrt(mean_cell(hp * hp)),
            mean_cell(np.abs(hp)),
            mean_cell(np.abs(gx)),
            mean_cell(np.abs(gy)),
            mean_cell(gm),
            std_cell(gm),
        ],
        axis=-1,
    ).astype(np.float32)
    return raw, hf


def _raw_descriptors(rgb: np.ndarray) -> dict[str, np.ndarray]:
    raw, hf = _cell_stats(_gray(rgb))
    return {"raw_only64": raw, "raw_hf_grad64": hf}


def _dino_desc_from_cache(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[:2] != (DINO_GRID, DINO_GRID):
        raise ValueError(f"expected DINO 32 x 32, got {x.shape}")
    return x


def _memory_normalize(bank: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit scales on bank only, then apply exactly the same transform to query."""
    b = np.asarray(bank, dtype=np.float32).reshape(-1, bank.shape[-1])
    q = np.asarray(query, dtype=np.float32).reshape(-1, query.shape[-1])
    med = np.median(b, axis=0)
    mad = 1.4826 * np.median(np.abs(b - med), axis=0)
    scale = np.maximum(mad, MEMORY_MAD_FLOOR).astype(np.float32)

    def transform(x: np.ndarray) -> np.ndarray:
        y = (x - med) / scale
        n = np.linalg.norm(y, axis=1, keepdims=True)
        return (y / np.maximum(n, 1e-12)).astype(np.float32)

    return transform(b), transform(q), {
        "dim": int(b.shape[-1]),
        "bank_cells": int(b.shape[0]),
        "median_mean": float(med.mean()),
        "scale_min": float(scale.min()),
        "scale_median": float(np.median(scale)),
        "scale_max": float(scale.max()),
    }


def _global_nn_score(query: np.ndarray, bank: np.ndarray, chunk: int = 256) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32)
    b = np.asarray(bank, dtype=np.float32)
    out = np.empty(q.shape[0], dtype=np.float32)
    for start in range(0, q.shape[0], chunk):
        stop = min(start + chunk, q.shape[0])
        sim = q[start:stop] @ b.T
        out[start:stop] = 1.0 - sim.max(axis=1)
    return out


def _score_map(score: np.ndarray, grid: int) -> np.ndarray:
    if score.size != grid * grid:
        raise ValueError(f"score size {score.size} != {grid} x {grid}")
    return cv2.resize(score.reshape(grid, grid), (RAW_SIZE, RAW_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _ap(mask: np.ndarray, score_map: np.ndarray) -> float:
    y = (mask.reshape(-1) > 0).astype(np.int32)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, score_map.reshape(-1).astype(np.float64)))


def _normal_stats(clean_map: np.ndarray, nuisance_maps: list[np.ndarray]) -> dict[str, float]:
    clean = clean_map.reshape(-1).astype(np.float64)
    nui = np.concatenate([m.reshape(-1) for m in nuisance_maps]).astype(np.float64)
    labels = np.concatenate([np.zeros(clean.size, dtype=np.int32), np.ones(nui.size, dtype=np.int32)])
    vals = np.concatenate([clean, nui])
    return {
        "auc_clean_vs_nuisance": float(roc_auc_score(labels, vals)),
        "mean_clean_score": float(clean.mean()),
        "mean_nuisance_score": float(nui.mean()),
        "nuisance_minus_clean": float(nui.mean() - clean.mean()),
        "p95_clean_score": float(np.percentile(clean, 95)),
        "p95_nuisance_score": float(np.percentile(nui, 95)),
    }


def _load_category_cache(cat: str) -> dict[str, Any]:
    path = CACHE / f"v14_p1_support_dino_s0_k{SHOT}" / f"{cat}.npz"
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        d = {
            "clean": np.asarray(z["clean_feat"], dtype=np.float32),
            "syn": np.asarray(z["syn_feat"], dtype=np.float32),
            "masks": np.asarray(z["syn_masks"], dtype=np.uint8),
            "nui": np.asarray(z["nui_feat"], dtype=np.float32),
            "rels": z["ref_rel"].astype(str).tolist(),
            "syn_kinds": z["syn_kinds"].astype(str).tolist(),
            "syn_seeds": int(np.asarray(z["syn_seeds"])),
            "nui_keys": z["nui_keys"].astype(str).tolist(),
        }
    if d["clean"].shape[:3] != (SHOT, DINO_GRID, DINO_GRID):
        raise ValueError(f"unexpected clean cache shape for {cat}: {d['clean'].shape}")
    if d["syn"].shape[:3] != (SHOT, 9, DINO_GRID) or d["nui"].shape[:3] != (SHOT, 15, DINO_GRID):
        raise ValueError(f"unexpected episode cache shape for {cat}: syn={d['syn'].shape} nui={d['nui'].shape}")
    if d["syn_kinds"] != list(R.SYNTHETIC_KINDS) or d["syn_seeds"] != 3:
        raise ValueError(f"unexpected v14 synthetic ordering for {cat}: {d['syn_kinds']} / {d['syn_seeds']}")
    if d["nui_keys"] != list(NUI_KEYS):
        raise ValueError(f"unexpected v14 nuisance ordering for {cat}")
    return d


def _cache_mask_check(cat: str, rel: str, kind: str, seed: int, mask: np.ndarray) -> bool:
    d = _load_category_cache(cat)
    if rel not in d["rels"]:
        raise ValueError(f"support ID {rel} missing from cache for {cat}")
    h = d["rels"].index(rel)
    idx = d["syn_kinds"].index(kind) * d["syn_seeds"] + seed
    return bool(np.array_equal(mask, d["masks"][h, idx]))


def _query_dino(cache: dict[str, Any], h: int, kind: str | None, seed_or_nui: int | None) -> np.ndarray:
    if kind is None:
        if seed_or_nui is None:
            return _dino_desc_from_cache(cache["clean"][h])
        return _dino_desc_from_cache(cache["nui"][h, seed_or_nui])
    idx = cache["syn_kinds"].index(kind) * cache["syn_seeds"] + int(seed_or_nui)
    return _dino_desc_from_cache(cache["syn"][h, idx])


def _score_from_descriptors(cache: dict[str, Any], bank_rgb: np.ndarray, query_rgb: np.ndarray, bank_index: int, query_dino: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Score query from raw rendered pixels plus its cached DINO feature."""
    braw = _raw_descriptors(bank_rgb)
    qraw = _raw_descriptors(query_rgb)
    bank_desc = {
        "raw_hf_grad64": braw["raw_hf_grad64"],
        "raw_only64": braw["raw_only64"],
        "dino_full32": _dino_desc_from_cache(cache["clean"][bank_index]),
    }
    query_desc = {
        "raw_hf_grad64": qraw["raw_hf_grad64"],
        "raw_only64": qraw["raw_only64"],
        "dino_full32": _dino_desc_from_cache(query_dino),
    }
    maps: dict[str, np.ndarray] = {}
    norm_meta: dict[str, Any] = {}
    for method in METHODS:
        b, q, stats = _memory_normalize(bank_desc[method], query_desc[method])
        grid = DINO_GRID if method == "dino_full32" else GRID
        maps[method] = _score_map(_global_nn_score(q, b), grid)
        norm_meta[method] = stats
    return maps, {"bank_index": bank_index, "normalization": norm_meta}


def run_category(cat: str, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache = _load_category_cache(cat)
    rels = support_paths(cat, SHOT, SEED, manifest)
    assert_fit_ids_are_support(rels, cat, SHOT, SEED)
    if rels != cache["rels"]:
        raise ValueError(f"manifest/cache support ordering mismatch for {cat}: {rels} != {cache['rels']}")
    if len(rels) != SHOT:
        raise ValueError(f"expected {SHOT} support images for {cat}, got {len(rels)}")
    bank_rgbs = [_load_rgb(rels[i]) for i in range(SHOT)]
    query_rgbs = [_load_rgb(rels[i]) for i in range(SHOT)]
    ap_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    for h in range(SHOT):
        _check_deadline(f"before_{cat}_h{h}")
        bank_index = 1 - h
        bank_rgb = bank_rgbs[bank_index]
        query_rgb = query_rgbs[h]
        for kind in SYN_KINDS:
            for seed in SYN_SEEDS:
                _check_deadline(f"before_synthetic_{cat}_h{h}_{kind}_{seed}")
                rendered, mask = R.render_synthetic(query_rgb, kind, _variant_seed(cat, rels[h], kind, seed))
                cache_ok = _cache_mask_check(cat, rels[h], kind, seed, mask)
                if not cache_ok:
                    raise AssertionError(f"renderer mask mismatch against v14 cache: {cat} {rels[h]} {kind} {seed}")
                dino_q = _query_dino(cache, h, kind, seed)
                maps, norm_meta = _score_from_descriptors(cache, bank_rgb, rendered, bank_index, dino_q)
                for method, score_map in maps.items():
                    ap_rows.append({
                        "category": cat,
                        "query_index": h,
                        "query_rel": rels[h],
                        "bank_index": bank_index,
                        "bank_rel": rels[bank_index],
                        "strict_leave_image_memory": True,
                        "kind": kind,
                        "seed": seed,
                        "episode": cache["syn_kinds"].index(kind) * cache["syn_seeds"] + seed,
                        "method": method,
                        "mask_sha256": _sha256_array(mask),
                        "mask_positive_pixels": int((mask > 0).sum()),
                        "mask_area": float((mask > 0).mean()),
                        "mask_cache_verified": cache_ok,
                        "pixel_ap": _ap(mask, score_map),
                        "raw_source": "support-rendered query image",
                        "normalization_memory_only": True,
                        "normalization": norm_meta["normalization"][method],
                    })
                mask_rows.append({
                    "category": cat,
                    "query_index": h,
                    "query_rel": rels[h],
                    "kind": kind,
                    "seed": seed,
                    "episode": cache["syn_kinds"].index(kind) * cache["syn_seeds"] + seed,
                    "mask_sha256": _sha256_array(mask),
                    "mask_cache_verified": cache_ok,
                })
        # Clean and nuisance are only the normal-control route.  The clean
        # descriptor is never used to align or score synthetic episodes.
        clean_dino = _query_dino(cache, h, None, None)
        clean_maps, _clean_meta = _score_from_descriptors(cache, bank_rgb, query_rgb, bank_index, clean_dino)
        nuisance_maps: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
        for e, key in enumerate(NUI_KEYS):
            _check_deadline(f"before_nuisance_{cat}_h{h}_{e}")
            rendered, _ = R.render_by_key(query_rgb, key, cat, rels[h])
            dino_q = _query_dino(cache, h, None, e)
            maps, _ = _score_from_descriptors(cache, bank_rgb, rendered, bank_index, dino_q)
            for method in METHODS:
                nuisance_maps[method].append(maps[method])
        for method in METHODS:
            stats = _normal_stats(clean_maps[method], nuisance_maps[method])
            normal_rows.append({
                "category": cat,
                "query_index": h,
                "query_rel": rels[h],
                "bank_index": bank_index,
                "bank_rel": rels[bank_index],
                "strict_leave_image_memory": True,
                "method": method,
                **stats,
            })
    summary = {
        "category": cat,
        "shot": SHOT,
        "support_rels": rels,
        "unique_synthetic_masks": len({r["mask_sha256"] for r in mask_rows}),
        "synthetic_mask_rows": len(mask_rows),
        "mask_cache_all_verified": all(bool(r["mask_cache_verified"]) for r in mask_rows),
    }
    return ap_rows, normal_rows, summary


def _finite_mean(values: list[float]) -> float:
    vals = [float(x) for x in values if np.isfinite(x)]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(ap_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], summaries: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "complete",
        "elapsed_s": elapsed_s,
        "n_ap_rows": len(ap_rows),
        "n_normal_rows": len(normal_rows),
        "methods": {},
        "per_category": {},
        "data_role": {
            "support_only": True,
            "real_test_read": False,
            "strict_leave_image_memory": True,
            "query_clean_counterpart_used_for_synthetic": False,
            "mask_role": "support-derived evaluation labels only",
        },
    }
    for method in METHODS:
        rows = [r for r in ap_rows if r["method"] == method]
        nrows = [r for r in normal_rows if r["method"] == method]
        out["methods"][method] = {
            "thin_scratch_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rows if r["kind"] == "thin_scratch"]),
            "cutpaste_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rows if r["kind"] == "cutpaste"]),
            "synthetic_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rows]),
            "normal_auc_clean_vs_nuisance_macro": _finite_mean([r["auc_clean_vs_nuisance"] for r in nrows]),
            "normal_nuisance_minus_clean_macro": _finite_mean([r["nuisance_minus_clean"] for r in nrows]),
            "n_synthetic_rows": len(rows),
            "n_normal_rows": len(nrows),
        }
    for cat in CATEGORIES:
        out["per_category"][cat] = {}
        for method in METHODS:
            rr = [r for r in ap_rows if r["category"] == cat and r["method"] == method]
            nn = [r for r in normal_rows if r["category"] == cat and r["method"] == method]
            out["per_category"][cat][method] = {
                "thin_scratch_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "thin_scratch"]),
                "cutpaste_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "cutpaste"]),
                "normal_auc_clean_vs_nuisance": _finite_mean([r["auc_clean_vs_nuisance"] for r in nn]),
                "normal_nuisance_minus_clean": _finite_mean([r["nuisance_minus_clean"] for r in nn]),
            }
    out["deltas_vs_dino_full32"] = {}
    for method in ("raw_hf_grad64", "raw_only64"):
        out["deltas_vs_dino_full32"][method] = {
            "thin_scratch": out["methods"][method]["thin_scratch_pixel_ap_macro"] - out["methods"]["dino_full32"]["thin_scratch_pixel_ap_macro"],
            "cutpaste": out["methods"][method]["cutpaste_pixel_ap_macro"] - out["methods"]["dino_full32"]["cutpaste_pixel_ap_macro"],
            "synthetic": out["methods"][method]["synthetic_pixel_ap_macro"] - out["methods"]["dino_full32"]["synthetic_pixel_ap_macro"],
            "normal_auc": out["methods"][method]["normal_auc_clean_vs_nuisance_macro"] - out["methods"]["dino_full32"]["normal_auc_clean_vs_nuisance_macro"],
        }
    out["mask_audit"] = {
        "category_summaries": summaries,
        "total_unique_masks": len({r["mask_sha256"] for r in ap_rows}),
        "total_unique_masks_from_one_method": len({r["mask_sha256"] for r in ap_rows if r["method"] == METHODS[0]}),
        "all_cache_verified": all(s["mask_cache_all_verified"] for s in summaries),
    }
    out["decision"] = "LOW_LEVEL_SIGNAL_NEGATIVE_OR_NON_INDEPENDENT"
    out["decision_note"] = (
        "Probe-only result. Any raw high-frequency delta is support-derived evidence; "
        "the already archived STR route establishes that this representation family is not a new method by itself."
    )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _report_cn(protocol: dict[str, Any], result: dict[str, Any], command: str) -> str:
    m = result["methods"]
    lines = [
        "# 原图低层高频 / 梯度描述子探针记录（2026-09-08）",
        "",
        "## 旧路线审计与口径边界",
        "",
        "旧 `innovation_v10_portfolio` 已经实现并实测了 2-level Haar 高频残差（gray、R-G、B-Y）以及 Sobel gradient control：`scripts/innovation_v10_portfolio/run_r0_str.py` + `src/industrial_ad/innovation_v10_portfolio/spectral.py`。其 k4、真实 A1-missed-region 结果为 mean Δ(STR−A1) = −0.027368、仅 1/6 类为正，已 `FAIL_ARCHIVED`。因此本次结果不能把“高频/梯度描述子”本身称作新颖算法。",
        "",
        "本 probe 的差异只用于有限口径审计：1024 原图直接构造 64×64 灰度小维统计，k2 的 bracket_black / metal_plate，24 个与 tiling 相同的 support synthetic masks；全局 NN、memory-only median/MAD 尺度归一化，并列 raw-only 与 DINO full controls。",
        "",
        "## 数据角色与冻结协议",
        "",
        f"- 协议：`{protocol['protocol_version']}`；seed0 × k2；CPU threads={THREADS}；无 GPU。",
        "- 每个 query 只用同类另一张 `/train/good/` support 图建 memory；缓存 ID 已调用 `assert_fit_ids_are_support`，没有读取 `/test/`。",
        "- synthetic / nuisance query 都从其自身渲染图直接构造 raw 描述子；clean query 只进入正常光度敏感性控制，没有 clean counterpart oracle。",
        "- 24 个 unique mask 与 v14 cache 逐字节核对；mask 只作评价标签，未参与描述子、memory 或归一化。",
        "- 所有描述子维度、Gaussian sigma、cell 大小、median/MAD、NN 和映射参数在读取 AP 前冻结，没有扫描。",
        "",
        "## 首轮结果（support-derived）",
        "",
        "| 方法 | thin_scratch AP | cutpaste AP | 两族等权 AP | 正常 clean/光度 AUC | 光度均分−clean均分 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        x = m[method]
        lines.append(
            f"| {method} | {x['thin_scratch_pixel_ap_macro']:.4f} | {x['cutpaste_pixel_ap_macro']:.4f} | "
            f"{x['synthetic_pixel_ap_macro']:.4f} | {x['normal_auc_clean_vs_nuisance_macro']:.4f} | "
            f"{x['normal_nuisance_minus_clean_macro']:+.6f} |"
        )
    lines.extend([
        "",
        "相对 DINO full control 的两族等权 AP：",
        "",
        f"- raw_hf_grad64：`{result['deltas_vs_dino_full32']['raw_hf_grad64']['synthetic']:+.4f}`（thin_scratch `{result['deltas_vs_dino_full32']['raw_hf_grad64']['thin_scratch']:+.4f}`，cutpaste `{result['deltas_vs_dino_full32']['raw_hf_grad64']['cutpaste']:+.4f}`）。",
        f"- raw_only64：`{result['deltas_vs_dino_full32']['raw_only64']['synthetic']:+.4f}`（thin_scratch `{result['deltas_vs_dino_full32']['raw_only64']['thin_scratch']:+.4f}`，cutpaste `{result['deltas_vs_dino_full32']['raw_only64']['cutpaste']:+.4f}`）。",
        "",
        f"首轮结论：`{result['decision']}`。该小样本只说明本口径下的低层信号是否可观察，不能改变旧 STR 的归档结论，也不能写成论文主方法。",
        "",
        f"mask audit：unique masks={result['mask_audit']['total_unique_masks_from_one_method']}，cache parity={result['mask_audit']['all_cache_verified']}；AP rows={result['n_ap_rows']}。",
        "",
        f"复现命令：`{command}`",
        "",
    ])
    return "\n".join(lines)


def run() -> int:
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    protocol = _freeze_protocol(EXP_OUT / "PROTOCOL.json")
    (EXP_OUT / "PROTOCOL_CN.md").write_text(
        "# 原图低层高频 / 梯度探针冻结协议\n\n"
        "具体字段、旧路线审计、固定参数与数据角色见 `PROTOCOL.json`。\n\n"
        "本 probe 不是高频算法创新主张；它只复核 1024 原图 64-grid 小维统计在两类 k2 support-derived masks 上是否提供独立低层信号。\n",
        encoding="utf-8",
    )
    command = ".venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_highfreq.py"
    manifest = load_manifest()
    started_utc = _utc_now()
    started = time.perf_counter()
    all_ap: list[dict[str, Any]] = []
    all_normal: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    status = "complete"
    error: str | None = None
    try:
        for cat in CATEGORIES:
            _check_deadline(f"before_category_{cat}")
            ap_rows, normal_rows, summary = run_category(cat, manifest)
            all_ap.extend(ap_rows)
            all_normal.extend(normal_rows)
            summaries.append(summary)
            print(f"  done {cat}: AP rows={len(ap_rows)} normal rows={len(normal_rows)}", flush=True)
    except DeadlineReached as exc:
        status = "partial_deadline"
        error = str(exc)
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
        raise
    elapsed = time.perf_counter() - started
    result = aggregate(all_ap, all_normal, summaries, elapsed)
    result["status"] = status
    result["error"] = error
    result["started_utc"] = started_utc
    result["finished_utc"] = _utc_now()
    result["protocol_sha256"] = _sha256_file(EXP_OUT / "PROTOCOL.json")
    result["code_sha256"] = _sha256_file(Path(__file__))
    result["reproduction"] = {"command": command, "script": "scripts/innovation_overnight_20260908/probe_highfreq.py"}
    _write_json(EXP_OUT / "RESULTS.json", {"protocol": protocol, "result": result})
    _write_csv(EXP_OUT / "PER_EPISODE.csv", all_ap)
    _write_csv(EXP_OUT / "NORMAL_PER_IMAGE.csv", all_normal)
    _write_csv(EXP_OUT / "MASK_AUDIT.csv", [
        {
            "category": r["category"],
            "query_index": r["query_index"],
            "query_rel": r["query_rel"],
            "kind": r["kind"],
            "seed": r["seed"],
            "episode": r["episode"],
            "mask_sha256": r["mask_sha256"],
            "mask_cache_verified": r["mask_cache_verified"],
        }
        for r in all_ap
        if r["method"] == METHODS[0]
    ])
    (EXP_OUT / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    (EXP_OUT / "REPORT_CN.md").write_text(_report_cn(protocol, result, command), encoding="utf-8")
    print("\n==== HIGH-FREQUENCY PROBE ====")
    for method in METHODS:
        x = result["methods"][method]
        print(
            f"{method}: thin={x['thin_scratch_pixel_ap_macro']:.4f} "
            f"cut={x['cutpaste_pixel_ap_macro']:.4f} "
            f"normal_auc={x['normal_auc_clean_vs_nuisance_macro']:.4f}",
            flush=True,
        )
    print(f"status={status} elapsed_s={elapsed:.3f} outputs={EXP_OUT}", flush=True)
    return 0 if status in ("complete", "partial_deadline") else 1


if __name__ == "__main__":
    raise SystemExit(run())
