"""Frozen DINO full-image 896-edge / native 64-grid probe.

The probe isolates input resolution from the preceding 2 x 2 crop probe.  The
same 1024 x 1024 support images, deterministic synthetic renderer seeds, masks,
and leave-one-image memory are used.  A full image is resized by the existing
DINOv2 preprocessing to 896 edge, producing a native 64 x 64 patch grid.  The
old full 448-edge and 2 x 2 tile 64-grid AP values are read from the completed
tiling result as controls; the old result is never overwritten.

Only clean and synthetic full896 features are cached.  Nuisance images are
encoded and scored online so the cache remains bounded.  No p99 threshold is
carried over from the k2 single-image memory audit: normal rows report score
distribution statistics only.
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
from export_p1_support_variants import _variant_seed  # noqa: E402
from v14_common import DATA_ROOT, assert_fit_ids_are_support, load_manifest, support_paths  # noqa: E402
from src.backbones import get_model  # noqa: E402
from src.utils import dists2map  # noqa: E402


SCRIPT_VERSION = "full896_probe_v1"
CATEGORIES = ["bracket_black", "metal_plate"]
SHOT = 2
SEED = 0
SYN_KINDS = ["thin_scratch", "cutpaste"]
SYN_SEEDS = [0, 1, 2]
NUI_KEYS = list(R.REF_KEYS)
N_SYN = len(SYN_KINDS) * len(SYN_SEEDS)
RAW_SIZE = 1024
FULL896_EDGE = 896
GRID = 64
FEATURE_DIM = 768
TARGET_MAP = (RAW_SIZE, RAW_SIZE)
DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()
MAX_RUNTIME_S = 45 * 60
TOKENS_896 = GRID * GRID
TOKENS_TILE_TOTAL = 4 * 32 * 32
ATTN_PROXY_896 = TOKENS_896 * TOKENS_896
ATTN_PROXY_TILE = 4 * (32 * 32) * (32 * 32)

EXP_OUT = ROOT / "experiments/dynamic_fusion/innovation_overnight_20260908/full896"
OUT = ROOT / "outputs/dynamic_fusion/overnight_20260908_full896"
TILING_RESULTS = ROOT / "outputs/dynamic_fusion/overnight_20260908_tiling/RESULTS.json"
V14_CACHE = ROOT / "outputs/dynamic_fusion/v14_p1_support"


class DeadlineReached(RuntimeError):
    pass


class RuntimeBudgetReached(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _check_limits(stage: str, started_perf: float) -> None:
    if time.time() >= DEADLINE_UTC:
        raise DeadlineReached(stage)
    if time.perf_counter() - started_perf >= MAX_RUNTIME_S:
        raise RuntimeBudgetReached(stage)


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_stats(device: str) -> dict[str, float | None]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
            "free_mb": None,
            "total_mb": None,
        }
    try:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_mb": float(torch.cuda.memory_allocated() / 1e6),
            "reserved_mb": float(torch.cuda.memory_reserved() / 1e6),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1e6),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 1e6),
            "free_mb": float(free / 1e6),
            "total_mb": float(total / 1e6),
        }
    except Exception:
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
            "free_mb": None,
            "total_mb": None,
        }


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _load_rgb(rel: str) -> np.ndarray:
    rel_norm = rel.replace("\\", "/")
    if "/test/" in f"/{rel_norm}":
        raise ValueError(f"test image outside support-only probe: {rel}")
    p = DATA_ROOT / rel
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(p))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (RAW_SIZE, RAW_SIZE):
        raise ValueError(f"expected 1024 x 1024 for {rel}, got {rgb.shape}")
    return rgb


def make_full896_extractor(device: str):
    """Use the existing DINOv2 ViT-B/14 checkpoint at a fixed 896 edge."""
    model = get_model("dinov2_vitb14", device, smaller_edge_size=FULL896_EDGE)
    # ``get_model`` returns the existing wrapper; its ``load_model`` already
    # sets the underlying torch module to eval mode (the wrapper itself has no
    # ``eval`` method).
    model.model.eval()

    def extract(rgb: np.ndarray) -> np.ndarray:
        tensor, grid = model.prepare_image(rgb)
        with torch.inference_mode():
            tokens = model.extract_features(tensor).astype(np.float32)
        got = tuple(int(x) for x in grid)
        if got != (GRID, GRID):
            raise ValueError(f"full896 expected 64 x 64 grid, got {got}")
        return tokens.reshape(GRID, GRID, FEATURE_DIM)

    return extract, model


def _encode(extract, rgb: np.ndarray, device: str, started_perf: float, stage: str) -> tuple[np.ndarray, float, dict[str, float | None]]:
    _check_limits(f"before_{stage}", started_perf)
    _sync(device)
    t0 = time.perf_counter()
    feat = np.asarray(extract(rgb), dtype=np.float32)
    _sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = _gpu_stats(device)
    _check_limits(f"after_{stage}", started_perf)
    return feat, float(elapsed_ms), stats


def _l2_rows(feat: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(feat, dtype=np.float32).reshape(-1, feat.shape[-1])
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def _nearest_dist(query: np.ndarray, bank: np.ndarray, chunk: int = 512) -> np.ndarray:
    q = _l2_rows(query)
    b = _l2_rows(bank)
    out = np.empty(q.shape[0], dtype=np.float32)
    for i in range(0, q.shape[0], chunk):
        j = min(i + chunk, q.shape[0])
        sim = q[i:j] @ b.T
        out[i:j] = np.sqrt(np.maximum(2.0 - 2.0 * sim.max(axis=1), 0.0)).astype(np.float32)
    return out


def _map_standard(score_grid: np.ndarray) -> np.ndarray:
    return np.asarray(dists2map(score_grid, TARGET_MAP), dtype=np.float32)


def _score_map(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    return _map_standard(_nearest_dist(query, bank).reshape(GRID, GRID))


def _score_stats(score_map: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    if mask is None:
        values = np.asarray(score_map, dtype=np.float32).reshape(-1)
        scope = "all_pixels"
    else:
        keep = np.asarray(mask) == 0
        values = np.asarray(score_map, dtype=np.float32)[keep]
        scope = "mask_outside_normal_pixels"
    if values.size == 0:
        return {
            "scope": scope,
            "pixels": 0,
            "score_mean": None,
            "score_std": None,
            "score_p50": None,
            "score_p95": None,
            "score_p99": None,
            "score_max": None,
        }
    return {
        "scope": scope,
        "pixels": int(values.size),
        "score_mean": float(values.mean()),
        "score_std": float(values.std()),
        "score_p50": float(np.percentile(values, 50)),
        "score_p95": float(np.percentile(values, 95)),
        "score_p99": float(np.percentile(values, 99)),
        "score_max": float(values.max()),
    }


def _ap_and_prevalence(score_map: np.ndarray, mask: np.ndarray) -> tuple[float | None, float]:
    y = (np.asarray(mask).reshape(-1) > 0).astype(np.int32)
    if y.sum() == 0 or y.sum() == y.size:
        return None, float(y.mean())
    return float(average_precision_score(y, score_map.reshape(-1))), float(y.mean())


def _load_controls() -> dict[tuple[str, int, str, int], dict[str, Any]]:
    if not TILING_RESULTS.exists():
        raise FileNotFoundError(f"required completed tiling controls missing: {TILING_RESULTS}")
    z = json.loads(TILING_RESULTS.read_text(encoding="utf-8"))
    controls = {}
    for r in z.get("rows", []):
        key = (r["category"], int(r["query_index"]), r["kind"], int(r["episode"]))
        controls.setdefault(key, {})[r["arm"]] = r
    required = 0
    for cat in CATEGORIES:
        for qi in range(SHOT):
            for kind, episode in [("clean", -1), *[(k, s) for k in SYN_KINDS for s in SYN_SEEDS], *[("nuisance", i) for i in range(len(NUI_KEYS))]]:
                key = (cat, qi, kind, episode)
                if key not in controls or any(a not in controls[key] for a in ("full_dino32", "tile_reencode64")):
                    raise ValueError(f"missing tiling control rows for {key}")
                required += 1
    if required != len(CATEGORIES) * SHOT * (1 + N_SYN + len(NUI_KEYS)):
        raise AssertionError("control episode count")
    return controls


def _cache_mask_check(cat: str, rel: str, kind: str, seed: int, mask: np.ndarray) -> bool | None:
    p = V14_CACHE / f"v14_p1_support_dino_s0_k{SHOT}" / f"{cat}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        rels = z["ref_rel"].astype(str).tolist()
        if rel not in rels:
            return None
        kinds = z["syn_kinds"].astype(str).tolist()
        seeds = int(np.asarray(z["syn_seeds"]))
        if kind not in kinds or seed >= seeds:
            return None
        idx = kinds.index(kind) * seeds + int(seed)
        return bool(np.array_equal(mask, z["syn_masks"][rels.index(rel), idx]))


def _save_feature_cache(
    cat: str,
    rels: list[str],
    clean: np.ndarray,
    syn: np.ndarray,
    masks: np.ndarray,
    clean_valid: np.ndarray,
    syn_valid: np.ndarray,
) -> None:
    path = OUT / "features" / f"{cat}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ref_rel=np.asarray(rels),
        clean_feat=clean,
        syn_feat=syn,
        syn_masks=masks,
        syn_kinds=np.asarray(SYN_KINDS),
        syn_seeds=np.asarray(SYN_SEEDS),
        clean_valid=clean_valid.astype(np.uint8),
        syn_valid=syn_valid.astype(np.uint8),
        grid_size=np.asarray([GRID, GRID], dtype=np.int64),
        full_smaller_edge=np.asarray(FULL896_EDGE, dtype=np.int64),
        feature_dim=np.asarray(FEATURE_DIM, dtype=np.int64),
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def _write_partial(rows: list[dict[str, Any]], protocol: dict[str, Any], status: str, error: str | None) -> None:
    payload = {
        "protocol": protocol,
        "status": status,
        "error": error,
        "updated_utc": _utc_now(),
        "rows": rows,
    }
    _write_json(OUT / "PARTIAL_RESULTS.json", payload)
    with (OUT / "PER_EPISODE.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def _protocol() -> dict[str, Any]:
    p = {
        "script_version": SCRIPT_VERSION,
        "role": "support-only full-image native 64-grid input-resolution probe",
        "categories": CATEGORIES,
        "shot": SHOT,
        "seed": SEED,
        "synthetic_order": SYN_KINDS,
        "synthetic_seeds": SYN_SEEDS,
        "nuisance_keys": NUI_KEYS,
        "input": {
            "raw_shape": [RAW_SIZE, RAW_SIZE],
            "full_smaller_edge_size": FULL896_EDGE,
            "native_grid": [GRID, GRID],
            "feature_dim": FEATURE_DIM,
            "backbone": "existing DINOv2 ViT-B/14 checkpoint, frozen, no download",
        },
        "controls": "read-only full_dino32 and tile_reencode64 rows from overnight_20260908_tiling/RESULTS.json",
        "memory": "one other k2 support image only; query image excluded",
        "mask": "same v14 renderer masks and original 1024 x 1024 mask for all episodes",
        "metrics": {
            "synthetic": "Pixel-AP on original mask",
            "random_area_ap": "mask positive-pixel prevalence; large-sample/random-ranking reference, not exact finite-sample AP expectation",
            "normal": "score distribution only; no old p99 threshold or calibrated FP",
        },
        "cache": "clean and synthetic full896 feature arrays; nuisance features online only",
        "resources": {
            "device": "cuda:0",
            "batch": 1,
            "torch_threads": 2,
            "gpu": "RTX 3060 6 GB",
            "max_runtime_s": MAX_RUNTIME_S,
        },
        "compute": {
            "full896_tokens": TOKENS_896,
            "tile_total_tokens": TOKENS_TILE_TOTAL,
            "full896_attention_proxy": ATTN_PROXY_896,
            "tile_attention_proxy": ATTN_PROXY_TILE,
            "full896_to_tile_attention_proxy_ratio": ATTN_PROXY_896 / ATTN_PROXY_TILE,
            "note": "same stitched token count does not mean equal compute: full-image attention is quadratic over 4096 tokens; four 1024-token crop attentions sum to about one quarter proxy",
        },
        "hard_deadline_utc": datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).isoformat(),
        "oom_fallback": "same 896 edge on CPU in explicit rerun; never lower resolution",
    }
    _write_json(EXP_OUT / "PROTOCOL.json", p)
    return p


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["category"], int(row["query_index"]), row["kind"], int(row["episode"]))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_rows": len(rows), "full896": {}}
    syn = [r for r in rows if r["kind"] in SYN_KINDS and r.get("ap") is not None]
    per_cat_family: dict[str, Any] = {}
    for cat in CATEGORIES:
        per_cat_family[cat] = {}
        for kind in SYN_KINDS:
            x = [r for r in syn if r["category"] == cat and r["kind"] == kind]
            per_cat_family[cat][kind] = {
                "n": len(x),
                "mean_ap": float(np.mean([r["ap"] for r in x])) if x else None,
                "mean_delta_vs_full_dino32": float(np.mean([r["delta_vs_full_dino32"] for r in x])) if x else None,
                "mean_delta_vs_tile_reencode64": float(np.mean([r["delta_vs_tile_reencode64"] for r in x])) if x else None,
            }
    out["per_category_family"] = per_cat_family
    for kind in SYN_KINDS:
        x = [r for r in syn if r["kind"] == kind]
        out["full896"][kind] = {
            "n": len(x),
            "mean_ap": float(np.mean([r["ap"] for r in x])) if x else None,
            "mean_random_area_prevalence": float(np.mean([r["random_area_ap"] for r in x])) if x else None,
            "mean_score_p95_outside_mask": float(np.mean([r["score_p95"] for r in x])) if x else None,
        }
    out["full896"]["synthetic_mean_ap"] = float(np.mean([r["ap"] for r in syn])) if syn else None
    out["full896"]["clean_score_p95"] = float(np.mean([r["score_p95"] for r in rows if r["kind"] == "clean"])) if any(r["kind"] == "clean" for r in rows) else None
    out["full896"]["nuisance_score_p95"] = float(np.mean([r["score_p95"] for r in rows if r["kind"] == "nuisance"])) if any(r["kind"] == "nuisance" for r in rows) else None
    out["full896"]["mean_encode_ms"] = float(np.mean([r["encode_ms"] for r in rows])) if rows else None
    out["full896"]["mean_score_ms"] = float(np.mean([r["score_ms"] for r in rows])) if rows else None
    for control in ("full_dino32", "tile_reencode64"):
        c = [r[f"control_{control}_ap"] for r in syn if r.get(f"control_{control}_ap") is not None]
        out["full896"][f"mean_delta_vs_{control}"] = (
            float(np.mean([r["ap"] - r[f"control_{control}_ap"] for r in syn])) if c else None
        )
    return out


def _write_decision(protocol: dict[str, Any], rows: list[dict[str, Any]], status: str, elapsed_s: float, error: str | None) -> None:
    summary = _summary(rows)
    f896 = summary["full896"]
    if status != "complete":
        decision = "PARTIAL_STOP; no resolution/context claim"
    elif f896.get("mean_delta_vs_tile_reencode64") is None:
        decision = "NO_DECISION_INSUFFICIENT_ROWS"
    elif f896["mean_delta_vs_tile_reencode64"] > 0.0:
        decision = "CANDIDATE_NATIVE64_CONTEXT_RESOLUTION_SIGNAL; support-synthetic probe only"
    else:
        decision = "NO_NATIVE64_GAIN_OVER_TILE_IN_THIS_PROBE"
    result = {
        "protocol": protocol,
        "status": status,
        "decision": decision,
        "error": error,
        "finished_utc": _utc_now(),
        "elapsed_s": float(elapsed_s),
        "summary": summary,
        "rows": rows,
    }
    _write_json(OUT / "RESULTS.json", result)
    _write_json(EXP_OUT / "SUMMARY.json", {k: result[k] for k in ("status", "decision", "error", "elapsed_s", "summary")})

    lines = [
        "# Full 896 原生 64 × 64 图像域探针决策",
        "",
        f"状态：`{status}`；决策：`{decision}`。",
        "",
        "本 probe 使用同一 MPDD seed 0、k2 support、同一 24 个 synthetic masks、同一 15 项 nuisance 和留一图 memory。full896 是完整 1024 × 1024 图像以 896 edge 通过冻结 DINO 得到的原生 64 × 64；full44832 与 tile64 的 AP 只读自已完成的 tiling 结果。",
        "",
        "`random_area_ap` 记录的是 mask 正像素比例，作为大样本 / 随机排序 prevalence 参考；有限样本 AP 非线性，因此不是有限样本 AP 精确期望。正常项不沿用旧单图 LOO p99，只报 score distribution。",
        "",
        "full896 与四 crop 的总 token 数都为 4096，但 full-image attention-work proxy 约为四 crop 总和的 4 倍；结果不能解读为等算力比较。",
        "",
        "每类每族 full896 delta（完整数值见 SUMMARY.json）：",
        json.dumps(summary.get("per_category_family", {}), ensure_ascii=False, indent=1),
        "",
        f"实际墙钟时间：`{elapsed_s:.3f} s`。clean / synthetic 原生 feature cache 见 `outputs/dynamic_fusion/overnight_20260908_full896/features/`；逐 episode 结果见 `RESULTS.json` 和 `PER_EPISODE.jsonl`。",
        "",
        "边界：本轮只有 2 类、k2、support synthetic episodes；任意正差值都只是候选输入分辨率 / 上下文信号，不能写成论文创新已证实。",
    ]
    (EXP_OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    protocol["started_utc"] = _utc_now()
    _write_json(EXP_OUT / "PROTOCOL.json", protocol)
    controls = _load_controls()
    rows: list[dict[str, Any]] = []
    started_perf = time.perf_counter()
    status = "complete"
    error: str | None = None
    model = None
    extract = None
    max_gpu: dict[str, float | None] = _gpu_stats(args.device)
    current_cache: tuple[str, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    try:
        _check_limits("before_model_load", started_perf)
        _sync(args.device)
        t_model = time.perf_counter()
        extract, model = make_full896_extractor(args.device)
        _sync(args.device)
        protocol["model_load_ms"] = float((time.perf_counter() - t_model) * 1000.0)
        max_gpu = _gpu_stats(args.device)
        protocol["model_load_gpu"] = max_gpu
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)

        manifest = load_manifest()
        for cat in CATEGORIES:
            rels = support_paths(cat, SHOT, str(SEED), manifest)
            assert_fit_ids_are_support(rels, cat, SHOT, str(SEED))
            if len(rels) != SHOT:
                raise ValueError(f"expected k2 support for {cat}, got {len(rels)}")
            clean_cache = np.zeros((SHOT, GRID, GRID, FEATURE_DIM), dtype=np.float32)
            syn_cache = np.zeros((SHOT, N_SYN, GRID, GRID, FEATURE_DIM), dtype=np.float32)
            syn_masks = np.zeros((SHOT, N_SYN, RAW_SIZE, RAW_SIZE), dtype=np.uint8)
            clean_valid = np.zeros(SHOT, dtype=bool)
            syn_valid = np.zeros((SHOT, N_SYN), dtype=bool)
            current_cache = (cat, rels, clean_cache, syn_cache, syn_masks, clean_valid, syn_valid)
            for h, query_rel in enumerate(rels):
                _check_limits(f"before_{cat}_query_{h}", started_perf)
                bank_h = 1 - h
                bank_rel = rels[bank_h]
                # Build this query's memory from exactly one other support image.
                if not clean_valid[bank_h]:
                    bank_rgb = _load_rgb(bank_rel)
                    bf, bms, gst = _encode(extract, bank_rgb, args.device, started_perf, f"bank_{cat}_{bank_h}")
                    clean_cache[bank_h] = bf
                    clean_valid[bank_h] = True
                    max_gpu = gst
                    del bf, bank_rgb
                bank_feat = clean_cache[bank_h]
                query_rgb = _load_rgb(query_rel)

                episodes: list[tuple[str, int, np.ndarray, np.ndarray | None, int | None, str, bool | None, float]] = []
                if not clean_valid[h]:
                    qf, qms, gst = _encode(extract, query_rgb, args.device, started_perf, f"clean_{cat}_{h}")
                    clean_cache[h] = qf
                    clean_valid[h] = True
                    max_gpu = gst
                    del qf
                    clean_ms = qms
                else:
                    clean_ms = 0.0
                episodes.append(("clean", -1, query_rgb, None, None, "clean", None, clean_ms))

                for kind in SYN_KINDS:
                    kind_i = SYN_KINDS.index(kind)
                    for seed in SYN_SEEDS:
                        _check_limits(f"before_render_{cat}_{h}_{kind}_{seed}", started_perf)
                        img, mask = R.render_synthetic(query_rgb, kind, _variant_seed(cat, query_rel, kind, seed))
                        mask_ok = _cache_mask_check(cat, query_rel, kind, seed, mask)
                        if mask_ok is False:
                            raise AssertionError(f"renderer mismatch against v14 cache: {cat} {query_rel} {kind} {seed}")
                        e = kind_i * len(SYN_SEEDS) + int(seed)
                        syn_masks[h, e] = mask
                        sms = 0.0
                        if not syn_valid[h, e]:
                            sf, sms, gst = _encode(extract, img, args.device, started_perf, f"syn_{cat}_{h}_{kind}_{seed}")
                            syn_cache[h, e] = sf
                            syn_valid[h, e] = True
                            max_gpu = gst
                            del sf
                        episodes.append((kind, int(seed), img, mask, int(seed), "synthetic", mask_ok, sms))

                for e, key in enumerate(NUI_KEYS):
                    _check_limits(f"before_render_nuisance_{cat}_{h}_{e}", started_perf)
                    img, _ = R.render_by_key(query_rgb, key, cat, query_rel)
                    episodes.append(("nuisance", int(e), img, None, None, key, None, 0.0))

                for kind, episode, img, mask, seed, label, mask_ok, encode_ms in episodes:
                    _check_limits(f"before_episode_{cat}_{h}_{kind}_{episode}", started_perf)
                    if kind == "clean":
                        query_feat = clean_cache[h]
                    elif kind in SYN_KINDS:
                        query_feat = syn_cache[h, SYN_KINDS.index(kind) * len(SYN_SEEDS) + int(episode)]
                    else:
                        query_feat, encode_ms, gst = _encode(extract, img, args.device, started_perf, f"nuisance_{cat}_{h}_{episode}")
                        max_gpu = gst
                    _check_limits(f"after_episode_encode_{cat}_{h}_{kind}_{episode}", started_perf)
                    t_score = time.perf_counter()
                    score_map = _score_map(query_feat, bank_feat)
                    score_ms = (time.perf_counter() - t_score) * 1000.0
                    if mask is None:
                        ap = None
                        random_area_ap = None
                        stats = _score_stats(score_map, None)
                    else:
                        ap, random_area_ap = _ap_and_prevalence(score_map, mask)
                        stats = _score_stats(score_map, mask)
                    key_controls = controls[(cat, h, kind, episode)]
                    ctrl_full = key_controls["full_dino32"]
                    ctrl_tile = key_controls["tile_reencode64"]
                    if ctrl_full["query_rel"] != query_rel or ctrl_full["bank_rel"] != bank_rel:
                        raise AssertionError(f"control path mismatch for {cat} h{h} {kind} {episode}")
                    if mask is not None:
                        if ctrl_full["mask_sha256"] != _sha256(mask):
                            raise AssertionError(f"control mask mismatch for {cat} h{h} {kind} {episode}")
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
                        "mask_sha256": _sha256(mask) if mask is not None else None,
                        "mask_area": float(mask.mean()) if mask is not None else 0.0,
                        "mask_cache_verified": mask_ok,
                        "arm": "full896_native64",
                        "grid": GRID,
                        "ap": ap,
                        "random_area_ap": random_area_ap,
                        **stats,
                        "normal_fp_rate": None,
                        "control_full_dino32_ap": ctrl_full.get("ap"),
                        "control_tile_reencode64_ap": ctrl_tile.get("ap"),
                        "delta_vs_full_dino32": (float(ap - ctrl_full["ap"]) if ap is not None and ctrl_full.get("ap") is not None else None),
                        "delta_vs_tile_reencode64": (float(ap - ctrl_tile["ap"]) if ap is not None and ctrl_tile.get("ap") is not None else None),
                        "feature_cached": bool(kind == "clean" or kind in SYN_KINDS),
                        "encode_ms": float(encode_ms),
                        "score_ms": float(score_ms),
                        "episode_elapsed_ms": float(encode_ms + score_ms),
                        "gpu": _gpu_stats(args.device),
                        "tokens": TOKENS_896,
                        "attention_work_proxy": ATTN_PROXY_896,
                    }
                    rows.append(row)
                    _write_partial(rows, protocol, "running", None)
                    _check_limits(f"after_episode_{cat}_{h}_{kind}_{episode}", started_perf)
                    del query_feat, score_map, img
                # Persist a usable clean/synthetic cache after each held-out
                # support image so a deadline or OOM during the next image
                # does not discard all already encoded features.
                _save_feature_cache(cat, rels, clean_cache, syn_cache, syn_masks, clean_valid, syn_valid)
                del query_rgb, bank_feat
            _save_feature_cache(cat, rels, clean_cache, syn_cache, syn_masks, clean_valid, syn_valid)
            protocol.setdefault("feature_cache", {})[cat] = {
                "path": str((OUT / "features" / f"{cat}.npz").relative_to(ROOT)),
                "clean_valid": clean_valid.astype(int).tolist(),
                "syn_valid": syn_valid.astype(int).tolist(),
                "bytes_uncompressed_float32": int(clean_cache.nbytes + syn_cache.nbytes),
            }
            _write_json(EXP_OUT / "PROTOCOL.json", protocol)
    except (DeadlineReached, RuntimeBudgetReached) as exc:
        status = "partial_deadline" if isinstance(exc, DeadlineReached) else "partial_budget"
        error = str(exc)
    except RuntimeError as exc:
        if _is_oom(exc):
            status = "partial_oom"
            error = f"{type(exc).__name__}: {exc}"
            protocol["oom_fallback_status"] = "same 896 edge on CPU explicit rerun; no resolution reduction"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            status = "error_partial"
            error = f"{type(exc).__name__}: {exc}"
            raise
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started_perf
        protocol["finished_utc"] = _utc_now()
        protocol["elapsed_s"] = float(elapsed)
        protocol["final_gpu"] = _gpu_stats(args.device)
        if current_cache is not None:
            c_cat, c_rels, c_clean, c_syn, c_masks, c_clean_valid, c_syn_valid = current_cache
            _save_feature_cache(c_cat, c_rels, c_clean, c_syn, c_masks, c_clean_valid, c_syn_valid)
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)
        _write_partial(rows, protocol, status, error)
        _write_decision(protocol, rows, status, elapsed, error)
        if model is not None and status != "complete" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"status": status, "error": error, "rows": len(rows), "elapsed_s": elapsed}, ensure_ascii=False), flush=True)
    return 0 if status in ("complete", "partial_deadline", "partial_budget", "partial_oom") else 1


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
