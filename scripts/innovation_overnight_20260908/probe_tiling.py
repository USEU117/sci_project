"""Support-only DINO 2 x 2 tiling probe for the 2026-09-08 overnight run.

This is a deliberately small, fixed probe of an image-domain input mechanism:
each 1024 x 1024 support image is either encoded as one full image (DINO
32 x 32) or split into four 512 x 512 tiles.  The four independently encoded
32 x 32 grids are stitched into a 64 x 64 grid.  The backbone is frozen and
the only memory is the other support image of the same category.

Arms
----
``full_dino32``     full-image frozen DINO, KNN on 32 x 32, standard dists2map.
``full_map_interp`` full-image KNN on 32 x 32, then pure score-map resize.
``full_feat_interp64`` full-image feature grid bilinear-resized to 64 x 64,
                       then KNN (feature-interpolation control).
``tile_reencode64`` four true tile re-encodings stitched to 64 x 64.

The protocol intentionally evaluates thin_scratch first, then cutpaste, and
then the pre-registered photometric nuisance ladder.  No test image or test
label is read.  ``random_area_ap`` is the mask positive-pixel prevalence.  It
is a large-sample/random-ranking reference; finite-sample AP is nonlinear, so
this number is not claimed to be the exact finite-sample AP expectation.
``normal_fp_rate`` is measured against a p99 threshold derived from the
held-out image's leave-one-image memory only.  For synthetic images that rate
is reported on the mask outside the synthetic region, which is the normal
background.

The hard stop is encoded as a UTC datetime rather than a copied Unix number.
The loop writes a partial checkpoint after every episode so a stop at the
deadline remains inspectable and resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Match the import order used by the existing v14 support exporter.  In
# particular, this gives us its frozen DINO ``make_extractor('dino')`` helper
# and the established renderer/seed implementation without downloading or
# constructing a new model.
for _p in (
    ROOT,
    ROOT / "scripts",
    ROOT / "scripts" / "innovation_v12_new_observables",
    ROOT / "scripts" / "innovation_v14_decisive_validation_20260905",
    ROOT / "methods" / "anomalydino",
    ROOT / "methods" / "AnomalyCLIP-main",
):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import ntof_render as R  # noqa: E402
from export_p1_support_variants import _variant_seed, make_extractor  # noqa: E402
from v14_common import (  # noqa: E402
    CATEGORIES as _V14_CATEGORIES,
    DATA_ROOT,
    assert_fit_ids_are_support,
    load_manifest,
    support_paths,
)
from src.utils import dists2map  # noqa: E402


SCRIPT_VERSION = "tiling_probe_v1"
CATEGORIES = ["bracket_black", "metal_plate"]
SHOT = 2
SEED = 0
SYN_KINDS = ["thin_scratch", "cutpaste"]
SYN_SEEDS = [0, 1, 2]
NUI_KEYS = list(R.REF_KEYS)  # all 5 families x 3 pre-registered strengths
FULL_GRID = 32
TILE_GRID = 64
TILE_ROWS = 2
TILE_COLS = 2
RAW_SIZE = 1024
TARGET_MAP = (RAW_SIZE, RAW_SIZE)
DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()
ARMS = ("full_dino32", "full_map_interp", "full_feat_interp64", "tile_reencode64")

EXP_OUT = ROOT / "experiments/dynamic_fusion/innovation_overnight_20260908/tiling"
OUT = ROOT / "outputs/dynamic_fusion/overnight_20260908_tiling"
V14_CACHE = ROOT / "outputs/dynamic_fusion/v14_p1_support"


class DeadlineReached(RuntimeError):
    """Raised only at loop boundaries after the partial result is saved."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _check_deadline(stage: str) -> None:
    # Called before and after each model extraction and before each episode.
    # ``time.time`` is only the clock reading; the deadline itself is the
    # explicit UTC datetime above, avoiding an unverified magic timestamp.
    if time.time() >= DEADLINE_UTC:
        raise DeadlineReached(stage)


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_rgb(rel: str) -> np.ndarray:
    rel_norm = rel.replace("\\", "/")
    if "/test/" in f"/{rel_norm}":
        raise ValueError(f"test image is outside this support-only probe: {rel}")
    p = DATA_ROOT / rel
    x = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if x is None:
        raise FileNotFoundError(str(p))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if x.shape[:2] != (RAW_SIZE, RAW_SIZE):
        raise ValueError(f"expected {RAW_SIZE} x {RAW_SIZE}, got {x.shape} for {rel}")
    return x


def _tile_images(rgb: np.ndarray) -> list[np.ndarray]:
    h, w = rgb.shape[:2]
    if (h % TILE_ROWS) or (w % TILE_COLS):
        raise ValueError(f"image shape {rgb.shape} is not divisible by 2 x 2")
    th, tw = h // TILE_ROWS, w // TILE_COLS
    return [
        rgb[y * th : (y + 1) * th, x * tw : (x + 1) * tw]
        for y in range(TILE_ROWS)
        for x in range(TILE_COLS)
    ]


def _stitch_tiles(tile_features: list[np.ndarray]) -> np.ndarray:
    if len(tile_features) != 4:
        raise ValueError("2 x 2 tiling requires four tile features")
    shapes = {tuple(x.shape) for x in tile_features}
    if len(shapes) != 1:
        raise ValueError(f"tile grid shape mismatch: {sorted(shapes)}")
    h, w, d = tile_features[0].shape
    if (h, w) != (FULL_GRID, FULL_GRID):
        raise ValueError(f"expected each tile to encode to 32 x 32, got {(h, w)}")
    top = np.concatenate(tile_features[:2], axis=1)
    bottom = np.concatenate(tile_features[2:], axis=1)
    return np.concatenate([top, bottom], axis=0).astype(np.float32, copy=False)


def _resize_features(feat: np.ndarray, grid: int) -> np.ndarray:
    f = np.ascontiguousarray(feat, dtype=np.float32)
    if f.shape[:2] == (grid, grid):
        return f.copy()
    h, w, d = f.shape
    t = torch.from_numpy(f).permute(2, 0, 1)[None]
    with torch.inference_mode():
        y = torch.nn.functional.interpolate(
            t, size=(grid, grid), mode="bilinear", align_corners=False
        )
    return y[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)


def _l2_rows(feat: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(feat, dtype=np.float32).reshape(-1, feat.shape[-1])
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def _nearest_dist(q: np.ndarray, bank: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Cosine-equivalent L2 distance in bounded query chunks."""
    qn = _l2_rows(q)
    bn = _l2_rows(bank)
    out = np.empty(qn.shape[0], dtype=np.float32)
    for i in range(0, qn.shape[0], chunk):
        j = min(i + chunk, qn.shape[0])
        sim = qn[i:j] @ bn.T
        out[i:j] = np.sqrt(np.maximum(2.0 - 2.0 * sim.max(axis=1), 0.0)).astype(np.float32)
    return out


def _loo_dist(bank: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Nearest distance for every memory cell while excluding itself."""
    bn = _l2_rows(bank)
    n = bn.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        _check_deadline("memory_loo")
        j = min(i + chunk, n)
        sim = bn[i:j] @ bn.T
        sim[np.arange(j - i), np.arange(i, j)] = -np.inf
        out[i:j] = np.sqrt(np.maximum(2.0 - 2.0 * sim.max(axis=1), 0.0)).astype(np.float32)
    return out


def _score_grid(query: np.ndarray, bank: np.ndarray, grid: int) -> np.ndarray:
    return _nearest_dist(query, bank).reshape(grid, grid)


def _map_standard(score_grid: np.ndarray) -> np.ndarray:
    return np.asarray(dists2map(score_grid, TARGET_MAP), dtype=np.float32)


def _map_linear(score_grid: np.ndarray) -> np.ndarray:
    return cv2.resize(score_grid.astype(np.float32), TARGET_MAP[::-1], interpolation=cv2.INTER_LINEAR)


def _build_arm_features(extract, rgb: np.ndarray, device: str) -> dict[str, np.ndarray]:
    """Extract full image and four tiles; no feature map is learned or fitted."""
    _check_deadline("before_full_extract")
    _synchronize(device)
    t0 = time.perf_counter()
    full = np.asarray(extract(rgb), dtype=np.float32)
    _synchronize(device)
    full_ms = (time.perf_counter() - t0) * 1000.0
    _check_deadline("after_full_extract")
    if full.shape[:2] != (FULL_GRID, FULL_GRID):
        raise ValueError(f"full DINO grid must be 32 x 32, got {full.shape}")

    tiles = []
    tile_ms = 0.0
    for idx, tile in enumerate(_tile_images(rgb)):
        _check_deadline(f"before_tile_extract_{idx}")
        _synchronize(device)
        t1 = time.perf_counter()
        ft = np.asarray(extract(tile), dtype=np.float32)
        _synchronize(device)
        tile_ms += (time.perf_counter() - t1) * 1000.0
        _check_deadline(f"after_tile_extract_{idx}")
        tiles.append(ft)
    tile64 = _stitch_tiles(tiles)
    interp64 = _resize_features(full, TILE_GRID)
    return {
        "full": full,
        "interp64": interp64,
        "tile64": tile64,
        "extract_ms": float(full_ms + tile_ms),
        "full_extract_ms": float(full_ms),
        "tile_extract_ms": float(tile_ms),
    }


def _prepare_bank(features: dict[str, np.ndarray], device: str) -> dict[str, dict[str, Any]]:
    """Build per-arm bank and p99 LOO threshold from one support image only."""
    specs = {
        "full_dino32": (features["full"], FULL_GRID, _map_standard),
        "full_map_interp": (features["full"], FULL_GRID, _map_linear),
        "full_feat_interp64": (features["interp64"], TILE_GRID, _map_standard),
        "tile_reencode64": (features["tile64"], TILE_GRID, _map_standard),
    }
    out: dict[str, dict[str, Any]] = {}
    for arm, (feat, grid, map_fn) in specs.items():
        _check_deadline(f"before_bank_{arm}")
        bank = _l2_rows(feat)
        loo_grid = _loo_dist(feat).reshape(grid, grid)
        threshold_map = map_fn(loo_grid)
        out[arm] = {
            "bank": bank,
            "grid": grid,
            "map_fn": map_fn,
            "threshold": float(np.percentile(threshold_map, 99.0)),
            "memory_cells": int(bank.shape[0]),
            "memory_feature_dim": int(bank.shape[1]),
        }
    return out


def _query_maps(
    features: dict[str, np.ndarray],
    bank: dict[str, dict[str, Any]],
    only: str | None = None,
) -> dict[str, np.ndarray]:
    specs = {
        "full_dino32": (features["full"], FULL_GRID),
        "full_map_interp": (features["full"], FULL_GRID),
        "full_feat_interp64": (features["interp64"], TILE_GRID),
        "tile_reencode64": (features["tile64"], TILE_GRID),
    }
    maps = {}
    names = (ARMS if only is None else (only,))
    for arm in names:
        feat, grid = specs[arm]
        _check_deadline(f"before_query_{arm}")
        grid_score = _score_grid(feat, bank[arm]["bank"], grid)
        maps[arm] = bank[arm]["map_fn"](grid_score)
    return maps


def _fp_rate(score_map: np.ndarray, threshold: float, mask: np.ndarray | None) -> tuple[float, int]:
    if mask is None:
        keep = np.ones(score_map.shape, dtype=bool)
        scope = "all_cells"
    else:
        keep = mask == 0
        scope = "outside_mask"
    del scope
    return (float(np.mean(score_map[keep] > threshold)) if keep.any() else float("nan"), int(keep.sum()))


def _episode_metrics(
    score_map: np.ndarray,
    threshold: float,
    mask: np.ndarray | None,
) -> tuple[float | None, float | None, float, int, str]:
    if mask is None:
        return None, None, *_fp_rate(score_map, threshold, None), "all_cells"
    y = (mask.reshape(-1) > 0).astype(np.int32)
    if y.sum() == 0 or y.sum() == y.size:
        ap = None
    else:
        ap = float(average_precision_score(y, score_map.reshape(-1)))
    random_area_ap = float(y.mean())
    fp, n_scope = _fp_rate(score_map, threshold, mask)
    return ap, random_area_ap, fp, n_scope, "outside_mask"


def _cache_mask_check(cat: str, rel: str, syn_kind: str, seed: int, mask: np.ndarray) -> bool | None:
    """Verify our renderer seed against the existing v14 export when present."""
    p = V14_CACHE / f"v14_p1_support_dino_s0_k{SHOT}" / f"{cat}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        rels = z["ref_rel"].astype(str).tolist()
        h = rels.index(rel)
        kinds = z["syn_kinds"].astype(str).tolist()
        seeds = int(np.asarray(z["syn_seeds"]))
        if syn_kind not in kinds or seed >= seeds:
            return None
        idx = kinds.index(syn_kind) * seeds + int(seed)
        return bool(np.array_equal(mask, z["syn_masks"][h, idx]))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def _write_partial(rows: list[dict[str, Any]], meta: dict[str, Any], status: str, error: str | None = None) -> None:
    payload = {
        "protocol": meta,
        "status": status,
        "error": error,
        "updated_utc": _utc_now(),
        "rows": rows,
    }
    _write_json(OUT / "PARTIAL_RESULTS.json", payload)
    # JSONL is convenient for tailing a run while it is still active and keeps
    # each episode inspectable if a process is interrupted between checkpoints.
    with (OUT / "PER_EPISODE.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def _write_protocol() -> dict[str, Any]:
    protocol = {
        "script_version": SCRIPT_VERSION,
        "role": "support-only input-resolution probe; no test image or test label",
        "categories": CATEGORIES,
        "shot": SHOT,
        "seed": SEED,
        "support_selection": "manifest seed 0, k2; each query image held out from memory",
        "synthetic_order": SYN_KINDS,
        "synthetic_seeds": SYN_SEEDS,
        "synthetic_seed_rule": "existing v14 _variant_seed(cat, support_rel, family, seed)",
        "nuisance_keys": NUI_KEYS,
        "input": {
            "raw_shape": [RAW_SIZE, RAW_SIZE],
            "tiles": "2 x 2 non-overlapping 512 x 512 crops",
            "tile_stitch": "each tile independently DINO-encoded at 448 edge to 32 x 32; concatenate to 64 x 64",
            "backbone": "existing v14 export make_extractor('dino'), frozen, no new download",
        },
        "arms": list(ARMS),
        "map_target": list(TARGET_MAP),
        "evaluation": {
            "mask": "same 1024 x 1024 renderer mask for all arms",
            "ap": "pixel average precision on the original mask",
            "random_area_ap": "mask positive-pixel prevalence; large-sample/random-ranking reference, not exact finite-sample AP expectation",
            "normal_fp": "p99 of memory-image LOO score map; clean/nuisance all pixels, synthetic outside mask",
            "memory": "only the other support image in the same category; no query image cells",
        },
        "resources": {"device": "cuda:0", "batch": 1, "torch_threads": 2, "gpu": "RTX 3060 6 GB"},
        "hard_deadline_utc": datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).isoformat(),
        "hard_deadline_epoch": DEADLINE_UTC,
    }
    _write_json(EXP_OUT / "PROTOCOL.json", protocol)
    return protocol


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["category"], row["query_index"], row["kind"], row["episode"], row["arm"])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_rows": len(rows), "arms": {}}
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        syn = [r for r in rr if r["kind"] in SYN_KINDS and r.get("ap") is not None]
        clean = [r for r in rr if r["kind"] == "clean"]
        nui = [r for r in rr if r["kind"] == "nuisance"]
        fam = {}
        for kind in SYN_KINDS:
            x = [r for r in syn if r["kind"] == kind]
            fam[kind] = {
                "n": len(x),
                "mean_ap": float(np.mean([r["ap"] for r in x])) if x else None,
                "mean_random_area_ap": float(np.mean([r["random_area_ap"] for r in x])) if x else None,
                "mean_normal_fp_outside_mask": float(np.mean([r["normal_fp_rate"] for r in x])) if x else None,
            }
        out["arms"][arm] = {
            "n": len(rr),
            "synthetic": fam,
            "synthetic_mean_ap": float(np.mean([r["ap"] for r in syn])) if syn else None,
            "clean_normal_fp_rate": float(np.mean([r["normal_fp_rate"] for r in clean])) if clean else None,
            "nuisance_normal_fp_rate": float(np.mean([r["normal_fp_rate"] for r in nui])) if nui else None,
            "mean_extract_ms": float(np.mean([r["extract_ms"] for r in rr])) if rr else None,
            "mean_score_ms": float(np.mean([r["score_ms"] for r in rr])) if rr else None,
        }
    return out


def _decision(summary: dict[str, Any], rows: list[dict[str, Any]], status: str) -> str:
    if status != "complete":
        return "PARTIAL_STOP; no innovation claim"
    t = summary["arms"].get("tile_reencode64", {})
    f = summary["arms"].get("full_dino32", {})
    # This is descriptive only.  The probe has two categories and is not a
    # paper-level confirmation gate.
    if t.get("synthetic_mean_ap") is None or f.get("synthetic_mean_ap") is None:
        return "NO_DECISION_INSUFFICIENT_ROWS"
    delta = float(t["synthetic_mean_ap"] - f["synthetic_mean_ap"])
    if delta > 0.0:
        return "CANDIDATE_TILING_SIGNAL; two-category support-synthetic probe only"
    return "NO_POSITIVE_TILING_SIGNAL_IN_THIS_PROBE"


def _write_decision(protocol: dict[str, Any], rows: list[dict[str, Any]], status: str, elapsed_s: float, error: str | None) -> None:
    summary = _summary(rows)
    decision = _decision(summary, rows, status)
    result = {
        "protocol": protocol,
        "status": status,
        "decision": decision,
        "error": error,
        "started_utc": protocol.get("started_utc"),
        "finished_utc": _utc_now(),
        "elapsed_s": float(elapsed_s),
        "summary": summary,
        "rows": rows,
    }
    _write_json(OUT / "RESULTS.json", result)
    _write_json(EXP_OUT / "SUMMARY.json", {k: result[k] for k in ("status", "decision", "error", "elapsed_s", "summary")})
    lines = [
        "# 2 × 2 原图裁块重编码探针决策",
        "",
        f"状态：`{status}`；决策：`{decision}`。",
        "",
        "该 probe 只使用 MPDD manifest 的 seed 0、k2 support 原图，在每个 query support 图上留出该图并只用另一张图建 memory。thin_scratch、cutpaste 和光度 nuisance 的 renderer seed 固定，未读取 test 图或 test 标签。",
        "",
        "评价使用相同的 1024 × 1024 原图 mask。`random_area_ap` 是 mask 正像素比例，作为大样本/随机排序参考；有限样本 AP 是非线性的，因此不把它称作有限样本 AP 的精确期望。正常 FP 阈值是留出 memory 图的 LOO p99，合成缺陷的 FP 只统计 mask 外正常背景。",
        "",
        "`tile_reencode64` 是四个 512 × 512 crop 的冻结 DINO 独立前向并拼接 64 × 64；`full_feat_interp64` 仅作特征插值控制，`full_map_interp` 仅作距离图插值控制。",
        "",
        f"实际墙钟时间：`{elapsed_s:.3f} s`。详细每 episode 记录见 `outputs/dynamic_fusion/overnight_20260908_tiling/RESULTS.json` 与 `PER_EPISODE.jsonl`。",
        "",
        "边界：本轮只有 2 类、k2、support 合成 episode；即使出现正向差值，也只能作为候选信号，不能写成论文创新已证实。",
    ]
    (EXP_OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_resume(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    p = OUT / "PARTIAL_RESULTS.json"
    if not p.exists():
        return [], set()
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        old = obj.get("protocol", {})
        for key in ("categories", "shot", "synthetic_order", "synthetic_seeds", "arms"):
            if old.get(key) != protocol.get(key):
                return [], set()
        rows = list(obj.get("rows", []))
        return rows, {_row_key(r) for r in rows}
    except Exception:
        return [], set()


def run(args: argparse.Namespace) -> int:
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = _write_protocol()
    protocol["started_utc"] = _utc_now()
    _write_json(EXP_OUT / "PROTOCOL.json", protocol)
    rows, done = _load_resume(protocol)
    started = time.perf_counter()
    status = "complete"
    error = None
    extract = None
    try:
        _check_deadline("before_model_load")
        _synchronize(args.device)
        t_model = time.perf_counter()
        extract = make_extractor("dino", args.device)
        _synchronize(args.device)
        protocol["model_load_ms"] = (time.perf_counter() - t_model) * 1000.0
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)
        manifest = load_manifest()
        for cat in CATEGORIES:
            rels = support_paths(cat, SHOT, str(SEED), manifest)
            assert_fit_ids_are_support(rels, cat, SHOT, str(SEED))
            if len(rels) != SHOT:
                raise ValueError(f"expected k2 support for {cat}, got {len(rels)}")
            for h, query_rel in enumerate(rels):
                _check_deadline(f"before_category_{cat}_query_{h}")
                bank_h = 1 - h
                bank_rel = rels[bank_h]
                # Only the other support image is encoded as the memory image.
                bank_rgb = _load_rgb(bank_rel)
                t_bank = time.perf_counter()
                bank_features = _build_arm_features(extract, bank_rgb, args.device)
                bank = _prepare_bank(bank_features, args.device)
                bank_extract_ms = float(bank_features["extract_ms"])
                bank_build_ms = float((time.perf_counter() - t_bank) * 1000.0)
                del bank_features, bank_rgb
                _check_deadline(f"after_bank_{cat}_query_{h}")

                query_rgb = _load_rgb(query_rel)
                episodes: list[tuple[str, int, np.ndarray, np.ndarray | None, int | None, str, bool | None]] = [
                    ("clean", -1, query_rgb, None, None, "clean", None)
                ]
                for kind in SYN_KINDS:
                    for seed in SYN_SEEDS:
                        _check_deadline(f"before_render_{cat}_{h}_{kind}_{seed}")
                        img, mask = R.render_synthetic(
                            query_rgb, kind, _variant_seed(cat, query_rel, kind, seed)
                        )
                        mask_ok = _cache_mask_check(cat, query_rel, kind, seed, mask)
                        if mask_ok is False:
                            raise AssertionError(f"renderer mismatch against v14 cache: {cat} {query_rel} {kind} {seed}")
                        episodes.append((kind, seed, img, mask, seed, "synthetic", mask_ok))
                for e, key in enumerate(NUI_KEYS):
                    _check_deadline(f"before_render_nuisance_{cat}_{h}_{e}")
                    img, _ = R.render_by_key(query_rgb, key, cat, query_rel)
                    episodes.append(("nuisance", e, img, None, None, key, None))

                for kind, episode, img, mask, seed, label, mask_verified in episodes:
                    _check_deadline(f"before_episode_{cat}_{h}_{kind}_{episode}")
                    t_extract = time.perf_counter()
                    q_features = _build_arm_features(extract, img, args.device)
                    q_extract_ms = float(q_features["extract_ms"])
                    _check_deadline(f"after_episode_extract_{cat}_{h}_{kind}_{episode}")
                    maps: dict[str, np.ndarray] = {}
                    for arm in ARMS:
                        _check_deadline(f"before_episode_score_{cat}_{h}_{kind}_{episode}_{arm}")
                        t_score = time.perf_counter()
                        maps[arm] = _query_maps(q_features, bank, only=arm)[arm]
                        score_ms = (time.perf_counter() - t_score) * 1000.0
                        ap, random_ap, fp, n_scope, fp_scope = _episode_metrics(
                            maps[arm], bank[arm]["threshold"], mask
                        )
                        row = {
                            "category": cat,
                            "query_index": int(h),
                            "query_rel": query_rel,
                            "bank_index": int(bank_h),
                            "bank_rel": bank_rel,
                            "strict_leave_image_memory": True,
                            "kind": kind,
                            "episode": int(episode),
                            "episode_label": label,
                            "seed": int(seed) if seed is not None else None,
                            "arm": arm,
                            "grid": int(bank[arm]["grid"]),
                            "mask_sha256": _sha256(mask) if mask is not None else None,
                            "mask_area": float(mask.mean()) if mask is not None else 0.0,
                            "ap": ap,
                            "random_area_ap": random_ap,
                            "normal_fp_rate": fp,
                            "normal_fp_scope": fp_scope,
                            "normal_fp_scope_cells": n_scope,
                            "memory_p99_threshold": float(bank[arm]["threshold"]),
                            "memory_cells": int(bank[arm]["memory_cells"]),
                            "memory_feature_dim": int(bank[arm]["memory_feature_dim"]),
                            "extract_ms": q_extract_ms,
                            "score_ms": float(score_ms),
                            "episode_elapsed_ms": float(q_extract_ms + score_ms),
                            "bank_extract_ms": bank_extract_ms,
                            "memory_build_ms": bank_build_ms,
                            "renderer_seed_rule": "v14_variant_seed" if kind in SYN_KINDS else None,
                            "renderer_cache_mask_verified": mask_verified,
                        }
                        if _row_key(row) not in done:
                            rows.append(row)
                            done.add(_row_key(row))
                        # Keep a real partial artifact at episode granularity.
                        _write_partial(rows, protocol, "running")
                        _check_deadline(f"after_episode_{cat}_{h}_{kind}_{episode}_{arm}")
                    del q_features, maps, img
                del query_rgb, bank
    except DeadlineReached as exc:
        status = "partial_deadline"
        error = str(exc)
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        _write_partial(rows, protocol, status, error)
        _write_decision(protocol, rows, status, elapsed, error)
    print(json.dumps({"status": status, "error": error, "rows": len(rows), "elapsed_s": elapsed}, ensure_ascii=False), flush=True)
    return 0 if status in ("complete", "partial_deadline") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
