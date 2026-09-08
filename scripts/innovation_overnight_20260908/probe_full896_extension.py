"""Four-category extension of the full-image DINO 896-edge probe.

The extension keeps the same support-only, k2, seed-0 protocol as the first
full896 probe.  It adds bracket_brown, bracket_white, connector, and tubes.
For every episode the same rendered 1024 x 1024 image is encoded twice by one
frozen DINOv2 ViT-B/14 model: full44832 (448 edge, native 32 x 32) and
full896 (896 edge, native 64 x 64).  The two AP values therefore share the
same input and original pixel mask.  No tile arm is rerun here.

Clean and synthetic features are checkpointed at both resolutions.  The
small fixed nuisance subset is encoded online and only contributes score
distribution rows.  No old p99/FP threshold is reused.  The loop checks the
explicit UTC deadline before and after every extraction and episode and saves
partial results/cache in finally.
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
from PIL import Image

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
from torchvision import transforms  # noqa: E402

import ntof_render as R  # noqa: E402
from export_p1_support_variants import _variant_seed  # noqa: E402
from src.backbones import get_model  # noqa: E402
from src.utils import dists2map  # noqa: E402
from v14_common import DATA_ROOT, assert_fit_ids_are_support, load_manifest, support_paths  # noqa: E402


SCRIPT_VERSION = "full896_extension_v1"
CATEGORIES = ["bracket_brown", "bracket_white", "connector", "tubes"]
LEGACY_CATEGORIES = ["bracket_black", "metal_plate"]
SHOT = 2
SEED = 0
SYN_KINDS = ["thin_scratch", "cutpaste"]
SYN_SEEDS = [0, 1, 2]
# One fixed, pre-registered strength (index 1) from each existing family.
NUI_KEYS = [
    "exposure_1",
    "gamma_1",
    "white_balance_1",
    "lr_brightness_gradient_1",
    "specular_blob_1",
]
RAW_SIZE = 1024
GRID_448 = 32
GRID_896 = 64
FEATURE_DIM = 768
TARGET_MAP = (RAW_SIZE, RAW_SIZE)
FULL448_EDGE = 448
FULL896_EDGE = 896
ARM_448 = "full44832"
ARM_896 = "full896_native64"
DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()
MAX_RUNTIME_S = 45 * 60
TOKENS_448 = GRID_448 * GRID_448
TOKENS_896 = GRID_896 * GRID_896
ATTN_PROXY_448 = TOKENS_448 * TOKENS_448
ATTN_PROXY_896 = TOKENS_896 * TOKENS_896

EXP_OUT = ROOT / "experiments/dynamic_fusion/innovation_overnight_20260908/full896_extension"
OUT = ROOT / "outputs/dynamic_fusion/overnight_20260908_full896_extension"
LEGACY_RESULTS = ROOT / "outputs/dynamic_fusion/overnight_20260908_full896/RESULTS.json"
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
    """Keep torch allocator counters separate from whole-device counters."""
    empty = {
        "torch_allocated_mb": None,
        "torch_reserved_mb": None,
        "torch_peak_allocated_mb": None,
        "torch_peak_reserved_mb": None,
        "device_free_mb": None,
        "device_total_mb": None,
    }
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return empty
    try:
        free, total = torch.cuda.mem_get_info()
        return {
            "torch_allocated_mb": float(torch.cuda.memory_allocated() / 1e6),
            "torch_reserved_mb": float(torch.cuda.memory_reserved() / 1e6),
            "torch_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1e6),
            "torch_peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 1e6),
            "device_free_mb": float(free / 1e6),
            "device_total_mb": float(total / 1e6),
        }
    except Exception:
        return empty


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


def _make_extractor(device: str):
    """Load one existing DINO checkpoint and expose fixed 448/896 transforms."""
    model = get_model("dinov2_vitb14", device, smaller_edge_size=FULL896_EDGE)
    model.model.eval()
    patch = int(model.model.patch_size)
    preprocess = {
        edge: transforms.Compose(
            [
                transforms.Resize(
                    size=edge,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        for edge in (FULL448_EDGE, FULL896_EDGE)
    }

    def extract(rgb: np.ndarray, edge: int) -> np.ndarray:
        if edge not in preprocess:
            raise ValueError(f"unsupported edge {edge}")
        tensor = preprocess[edge](Image.fromarray(rgb))
        height, width = tensor.shape[1:]
        height -= height % patch
        width -= width % patch
        tensor = tensor[:, :height, :width]
        grid = (height // patch, width // patch)
        with torch.inference_mode():
            tokens = model.model.get_intermediate_layers(
                tensor.unsqueeze(0).to(device)
            )[0].squeeze(0)
        arr = tokens.float().cpu().numpy()
        expected = (edge // patch, edge // patch)
        if tuple(grid) != expected or arr.shape != (expected[0] * expected[1], FEATURE_DIM):
            raise ValueError(f"edge {edge} expected {expected} x {FEATURE_DIM}, got {grid} / {arr.shape}")
        return arr.reshape(expected[0], expected[1], FEATURE_DIM).astype(np.float32, copy=False)

    return extract, model


def _encode(extract, rgb: np.ndarray, edge: int, device: str, started_perf: float, stage: str):
    _check_limits(f"before_{stage}_{edge}", started_perf)
    _sync(device)
    t0 = time.perf_counter()
    feat = np.asarray(extract(rgb, edge), dtype=np.float32)
    _sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = _gpu_stats(device)
    _check_limits(f"after_{stage}_{edge}", started_perf)
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


def _score_map(query: np.ndarray, bank: np.ndarray, grid: int) -> np.ndarray:
    native = _nearest_dist(query, bank).reshape(grid, grid)
    return np.asarray(dists2map(native, TARGET_MAP), dtype=np.float32)


def _score_stats(score_map: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    if mask is None:
        values = np.asarray(score_map, dtype=np.float32).reshape(-1)
        scope = "all_pixels"
    else:
        values = np.asarray(score_map, dtype=np.float32)[np.asarray(mask) == 0]
        scope = "mask_outside_normal_pixels"
    if values.size == 0:
        return {
            "score_scope": scope,
            "score_pixels": 0,
            "score_mean": None,
            "score_std": None,
            "score_p50": None,
            "score_p95": None,
            "score_p99": None,
            "score_max": None,
        }
    return {
        "score_scope": scope,
        "score_pixels": int(values.size),
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


def _cache_mask_check(cat: str, rel: str, kind: str, seed: int, mask: np.ndarray) -> bool | None:
    path = V14_CACHE / f"v14_p1_support_dino_s0_k{SHOT}" / f"{cat}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        rels = z["ref_rel"].astype(str).tolist()
        if rel not in rels:
            return None
        kinds = z["syn_kinds"].astype(str).tolist()
        n_seeds = int(np.asarray(z["syn_seeds"]))
        if kind not in kinds or seed >= n_seeds:
            return None
        idx = kinds.index(kind) * n_seeds + int(seed)
        return bool(np.array_equal(mask, z["syn_masks"][rels.index(rel), idx]))


def _save_feature_cache(
    cat: str,
    rels: list[str],
    clean448: np.ndarray,
    clean896: np.ndarray,
    syn448: np.ndarray,
    syn896: np.ndarray,
    masks: np.ndarray,
    clean_valid: np.ndarray,
    syn_valid: np.ndarray,
) -> None:
    path = OUT / "features" / f"{cat}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ref_rel=np.asarray(rels),
        clean_feat_448=clean448,
        clean_feat_896=clean896,
        syn_feat_448=syn448,
        syn_feat_896=syn896,
        syn_masks=masks,
        syn_kinds=np.asarray(SYN_KINDS),
        syn_seeds=np.asarray(SYN_SEEDS),
        clean_valid=clean_valid.astype(np.uint8),
        syn_valid=syn_valid.astype(np.uint8),
        grid_448=np.asarray([GRID_448, GRID_448], dtype=np.int64),
        grid_896=np.asarray([GRID_896, GRID_896], dtype=np.int64),
        full448_smaller_edge=np.asarray(FULL448_EDGE, dtype=np.int64),
        full896_smaller_edge=np.asarray(FULL896_EDGE, dtype=np.int64),
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


def _initial_protocol() -> dict[str, Any]:
    p = {
        "script_version": SCRIPT_VERSION,
        "role": "support-only four-category full44832 versus full896 native-grid extension",
        "categories": CATEGORIES,
        "legacy_categories": LEGACY_CATEGORIES,
        "shot": SHOT,
        "seed": SEED,
        "synthetic_order": SYN_KINDS,
        "synthetic_seeds": SYN_SEEDS,
        "new_unique_synthetic_masks": len(CATEGORIES) * SHOT * len(SYN_KINDS) * len(SYN_SEEDS),
        "legacy_unique_synthetic_masks": 24,
        "combined_unique_synthetic_masks": 24 + len(CATEGORIES) * SHOT * len(SYN_KINDS) * len(SYN_SEEDS),
        "nuisance_keys": NUI_KEYS,
        "input": {
            "raw_shape": [RAW_SIZE, RAW_SIZE],
            "full44832": {"smaller_edge_size": FULL448_EDGE, "native_grid": [GRID_448, GRID_448]},
            "full896": {"smaller_edge_size": FULL896_EDGE, "native_grid": [GRID_896, GRID_896]},
            "feature_dim": FEATURE_DIM,
            "backbone": "existing DINOv2 ViT-B/14 checkpoint, one frozen model, no download",
        },
        "comparison": "same rendered image and original pixel mask are processed by full44832 and full896 in each new episode; no tile arm",
        "legacy_aggregate": "read-only old two-category full896 rows and their recorded full_dino32 control for six-category context",
        "memory": "one other k2 support image only; query image excluded",
        "mask": "same renderer mask and original 1024 x 1024 pixel map for both arms",
        "metrics": {
            "synthetic": "Pixel-AP on original mask",
            "delta": "full896 AP minus same-episode full44832 AP",
            "random_area_ap": "mask positive-pixel prevalence; large-sample/random-ranking reference, not exact finite-sample AP expectation",
            "normal": "score distribution only; no old p99 threshold or calibrated FP",
        },
        "cache": "clean and synthetic features at both 44832 and full896; nuisance online only",
        "resources": {
            "device": "cuda:0",
            "batch": 1,
            "torch_threads": 2,
            "gpu": "RTX 3060 6 GB",
            "max_runtime_s": MAX_RUNTIME_S,
        },
        "timing": "warmup is separate and excluded from formal episode timing; per-extraction torch allocator and whole-device memory are recorded",
        "hard_deadline_utc": datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).isoformat(),
        "oom_fallback": "same 448 / 896 resolutions on CPU in explicit rerun; never lower resolution",
        "attention_proxy": {
            "full44832_tokens": TOKENS_448,
            "full896_tokens": TOKENS_896,
            "full44832_attention_proxy": ATTN_PROXY_448,
            "full896_attention_proxy": ATTN_PROXY_896,
            "full896_to_full44832_attention_proxy_ratio": ATTN_PROXY_896 / ATTN_PROXY_448,
            "note": "full896 and full44832 are both full-image attention; quadratic token proxy is 16 x at 4096 versus 1024 tokens",
        },
    }
    _write_json(EXP_OUT / "PROTOCOL.json", p)
    return p


def _row(
    *,
    cat: str,
    query_index: int,
    query_rel: str,
    bank_index: int,
    bank_rel: str,
    kind: str,
    episode: int,
    label: str,
    seed: int | None,
    mask: np.ndarray | None,
    mask_ok: bool | None,
    arm: str,
    grid: int,
    score_map: np.ndarray,
    encode_ms: float,
    score_ms: float,
    bank_encode_ms: float,
    gpu: dict[str, float | None],
    bank_gpu: dict[str, float | None],
    feature_cached: bool,
) -> dict[str, Any]:
    if mask is None:
        ap = None
        random_area_ap = None
        stats = _score_stats(score_map, None)
    else:
        ap, random_area_ap = _ap_and_prevalence(score_map, mask)
        stats = _score_stats(score_map, mask)
    return {
        "category": cat,
        "query_index": int(query_index),
        "query_rel": query_rel,
        "bank_index": int(bank_index),
        "bank_rel": bank_rel,
        "strict_leave_image_memory": True,
        "kind": kind,
        "episode": int(episode),
        "episode_label": label,
        "seed": int(seed) if seed is not None else None,
        "paired_episode_id": f"{cat}|q{query_index}|{kind}|{episode}",
        "mask_sha256": _sha256(mask) if mask is not None else None,
        "mask_area": float(mask.mean()) if mask is not None else 0.0,
        "mask_cache_verified": mask_ok,
        "arm": arm,
        "grid": int(grid),
        "ap": ap,
        "random_area_ap": random_area_ap,
        **stats,
        "normal_fp_rate": None,
        "feature_cached": bool(feature_cached),
        "encode_ms": float(encode_ms),
        "score_ms": float(score_ms),
        "episode_elapsed_ms": float(encode_ms + score_ms),
        "bank_encode_ms": float(bank_encode_ms),
        "gpu": gpu,
        "bank_gpu": bank_gpu,
        "tokens": int(TOKENS_448 if arm == ARM_448 else TOKENS_896),
        "attention_work_proxy": int(ATTN_PROXY_448 if arm == ARM_448 else ATTN_PROXY_896),
    }


def _legacy_records() -> list[dict[str, Any]]:
    if not LEGACY_RESULTS.exists():
        raise FileNotFoundError(f"required old full896 result missing: {LEGACY_RESULTS}")
    obj = json.loads(LEGACY_RESULTS.read_text(encoding="utf-8"))
    records = []
    for r in obj.get("rows", []):
        if r.get("kind") not in SYN_KINDS or r.get("ap") is None:
            continue
        control = r.get("control_full_dino32_ap")
        if control is None:
            raise ValueError(f"old full896 row has no full44832 control: {r}")
        records.append(
            {
                "category": r["category"],
                "kind": r["kind"],
                "ap_448": float(control),
                "ap_896": float(r["ap"]),
                "delta": float(r["ap"] - control),
                "source": "legacy_full896_result_with_tiling_control",
            }
        )
    expected = len(LEGACY_CATEGORIES) * SHOT * len(SYN_KINDS) * len(SYN_SEEDS)
    if len(records) != expected:
        raise ValueError(f"expected {expected} legacy synthetic rows, got {len(records)}")
    if {r["category"] for r in records} != set(LEGACY_CATEGORIES):
        raise ValueError("legacy aggregate category mismatch")
    return records


def _pair_rows(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("kind") in SYN_KINDS and r.get("ap") is not None:
            by_id[(r["paired_episode_id"], r["arm"])] = r
    pairs = []
    for key in sorted({k[0] for k in by_id}):
        a = by_id.get((key, ARM_448))
        b = by_id.get((key, ARM_896))
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_unique_masks": len(records), "overall": {}, "per_category_family": {}}
    for key in ("ap_448", "ap_896", "delta"):
        vals = [float(r[key]) for r in records]
        out["overall"][key] = float(np.mean(vals)) if vals else None
    for cat in sorted({r["category"] for r in records}):
        out["per_category_family"][cat] = {}
        for kind in SYN_KINDS:
            x = [r for r in records if r["category"] == cat and r["kind"] == kind]
            out["per_category_family"][cat][kind] = {
                "n": len(x),
                "mean_ap_448": float(np.mean([r["ap_448"] for r in x])) if x else None,
                "mean_ap_896": float(np.mean([r["ap_896"] for r in x])) if x else None,
                "mean_delta_896_minus_448": float(np.mean([r["delta"] for r in x])) if x else None,
            }
    for kind in SYN_KINDS:
        x = [r for r in records if r["kind"] == kind]
        out.setdefault("per_family", {})[kind] = {
            "n": len(x),
            "mean_ap_448": float(np.mean([r["ap_448"] for r in x])) if x else None,
            "mean_ap_896": float(np.mean([r["ap_896"] for r in x])) if x else None,
            "mean_delta_896_minus_448": float(np.mean([r["delta"] for r in x])) if x else None,
        }
    return out


def _normal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in (ARM_448, ARM_896):
        rr = [r for r in rows if r["arm"] == arm]
        out[arm] = {
            "clean_score_p95_mean": float(np.mean([r["score_p95"] for r in rr if r["kind"] == "clean"])) if any(r["kind"] == "clean" for r in rr) else None,
            "nuisance_score_p95_mean": float(np.mean([r["score_p95"] for r in rr if r["kind"] == "nuisance"])) if any(r["kind"] == "nuisance" for r in rr) else None,
            "n_rows": len(rr),
        }
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _pair_rows(rows)
    new_records = [
        {
            "category": r896["category"],
            "kind": r896["kind"],
            "ap_448": float(r448["ap"]),
            "ap_896": float(r896["ap"]),
            "delta": float(r896["ap"] - r448["ap"]),
            "source": "same_episode_extension",
        }
        for r448, r896 in pairs
    ]
    legacy = _legacy_records()
    merged = legacy + new_records
    new_summary = _group_summary(new_records)
    merged_summary = _group_summary(merged)
    return {
        "n_rows": len(rows),
        "new_four_class": new_summary,
        "six_class_merged": merged_summary,
        "normal_score_distribution": _normal_summary(rows),
        "paired_synthetic_rows": len(new_records),
        "warmup": None,
        "formal_timing": {
            ARM_448: {
                "mean_encode_ms": float(np.mean([r["encode_ms"] for r in rows if r["arm"] == ARM_448])) if any(r["arm"] == ARM_448 for r in rows) else None,
                "mean_score_ms": float(np.mean([r["score_ms"] for r in rows if r["arm"] == ARM_448])) if any(r["arm"] == ARM_448 for r in rows) else None,
            },
            ARM_896: {
                "mean_encode_ms": float(np.mean([r["encode_ms"] for r in rows if r["arm"] == ARM_896])) if any(r["arm"] == ARM_896 for r in rows) else None,
                "mean_score_ms": float(np.mean([r["score_ms"] for r in rows if r["arm"] == ARM_896])) if any(r["arm"] == ARM_896 for r in rows) else None,
            },
        },
    }


def _write_decision(protocol: dict[str, Any], rows: list[dict[str, Any]], status: str, elapsed_s: float, error: str | None) -> None:
    summary = _summary(rows)
    new_overall = summary["new_four_class"]["overall"]
    if status != "complete":
        decision = "PARTIAL_STOP; no extension claim"
    elif new_overall.get("delta") is None:
        decision = "NO_DECISION_INSUFFICIENT_ROWS"
    elif new_overall["delta"] > 0.0:
        decision = "CANDIDATE_NATIVE64_EXTENSION_SIGNAL; support-synthetic probe only"
    else:
        decision = "NO_NATIVE64_GAIN_IN_NEW_FOUR_CATEGORIES"
    summary["decision"] = decision
    summary["warmup"] = protocol.get("warmup")
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
    _write_json(
        EXP_OUT / "SUMMARY.json",
        {k: result[k] for k in ("status", "decision", "error", "elapsed_s", "summary")},
    )

    lines = [
        "# Full 896 扩类探针决策",
        "",
        f"状态：`{status}`；决策：`{decision}`。",
        "",
        "本轮只新增 bracket_brown、bracket_white、connector、tubes。每个 episode 同时计算 full44832 与 full896，使用同一 rendered image、同一 k2 留图 memory、同一 1024 × 1024 原图 mask；没有 tile 臂，也没有重跑已完成两类。",
        "",
        "新增四类的 synthetic unique mask 计数为 48；六类合并把既有两类 full896 结果与本轮四类结果合并，共 72 个 mask。六类合并中的旧两类 full44832 数值来自既有 tiling 记录，仅用于上下文；新四类的 delta 均来自本脚本同 episode 两臂。",
        "",
        "random_area_ap 是 mask 正像素 prevalence 的大样本 / 随机排序参考，不是有限样本 AP 的精确期望。正常项只报告分数分布，不复用旧单图 LOO p99 或 FP 阈值。",
        "",
        "新四类逐类别逐族 `full896 - full44832`（完整数值见 SUMMARY.json）：",
        json.dumps(summary["new_four_class"]["per_category_family"], ensure_ascii=False, indent=1),
        "",
        "六类合并摘要（完整数值见 SUMMARY.json）：",
        json.dumps(summary["six_class_merged"]["overall"], ensure_ascii=False, indent=1),
        "",
        f"实际墙钟时间：`{elapsed_s:.3f} s`。clean / synthetic 双分辨率 feature cache 见 `outputs/dynamic_fusion/overnight_20260908_full896_extension/features/`；逐 episode 结果见 `RESULTS.json` 和 `PER_EPISODE.jsonl`。",
        "",
        "边界：这是四类、k2、support-synthetic 短探针。任何正向差值都只能作为候选输入分辨率 / 上下文信号；不要把旧两类 cutpaste 相对 448 的轻微下降写成改善，也不能据此宣称论文创新已证实。",
    ]
    (EXP_OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_episode_rows(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    status: str,
    cat: str,
    h: int,
    query_rel: str,
    bank_h: int,
    bank_rel: str,
    kind: str,
    episode: int,
    label: str,
    seed: int | None,
    mask: np.ndarray | None,
    mask_ok: bool | None,
    feat448: np.ndarray,
    feat896: np.ndarray,
    bank448: np.ndarray,
    bank896: np.ndarray,
    enc448_ms: float,
    enc896_ms: float,
    bank448_ms: float,
    bank896_ms: float,
    gpu448: dict[str, float | None],
    gpu896: dict[str, float | None],
    bank_gpu448: dict[str, float | None],
    bank_gpu896: dict[str, float | None],
    cached: bool,
    device: str,
    started_perf: float,
) -> None:
    _check_limits(f"before_episode_score_{cat}_{h}_{kind}_{episode}", started_perf)
    t0 = time.perf_counter()
    score448 = _score_map(feat448, bank448, GRID_448)
    score448_ms = (time.perf_counter() - t0) * 1000.0
    row448 = _row(
        cat=cat, query_index=h, query_rel=query_rel, bank_index=bank_h, bank_rel=bank_rel,
        kind=kind, episode=episode, label=label, seed=seed, mask=mask, mask_ok=mask_ok,
        arm=ARM_448, grid=GRID_448, score_map=score448, encode_ms=enc448_ms,
        score_ms=score448_ms, bank_encode_ms=bank448_ms, gpu=gpu448, bank_gpu=bank_gpu448,
        feature_cached=cached,
    )
    rows.append(row448)
    _write_partial(rows, protocol, status, None)
    _check_limits(f"after_episode_score_{cat}_{h}_{kind}_{episode}_{ARM_448}", started_perf)
    del score448

    t1 = time.perf_counter()
    score896 = _score_map(feat896, bank896, GRID_896)
    score896_ms = (time.perf_counter() - t1) * 1000.0
    row896 = _row(
        cat=cat, query_index=h, query_rel=query_rel, bank_index=bank_h, bank_rel=bank_rel,
        kind=kind, episode=episode, label=label, seed=seed, mask=mask, mask_ok=mask_ok,
        arm=ARM_896, grid=GRID_896, score_map=score896, encode_ms=enc896_ms,
        score_ms=score896_ms, bank_encode_ms=bank896_ms, gpu=gpu896, bank_gpu=bank_gpu896,
        feature_cached=cached,
    )
    rows.append(row896)
    _write_partial(rows, protocol, status, None)
    _check_limits(f"after_episode_score_{cat}_{h}_{kind}_{episode}_{ARM_896}", started_perf)
    del score896


def run(args: argparse.Namespace) -> int:
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = _initial_protocol()
    protocol["started_utc"] = _utc_now()
    _write_json(EXP_OUT / "PROTOCOL.json", protocol)
    rows: list[dict[str, Any]] = []
    started_perf = time.perf_counter()
    status = "complete"
    error: str | None = None
    model = None
    current_cache: tuple[Any, ...] | None = None
    try:
        _check_limits("before_model_load", started_perf)
        _sync(args.device)
        t_model = time.perf_counter()
        extract, model = _make_extractor(args.device)
        _sync(args.device)
        protocol["model_load_ms"] = float((time.perf_counter() - t_model) * 1000.0)
        protocol["model_load_gpu"] = _gpu_stats(args.device)
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)

        manifest = load_manifest()
        warm_rel = support_paths(CATEGORIES[0], SHOT, str(SEED), manifest)[0]
        warm_rgb = _load_rgb(warm_rel)
        warmup: dict[str, Any] = {}
        for edge in (FULL448_EDGE, FULL896_EDGE):
            _check_limits(f"before_warmup_{edge}", started_perf)
            _, ms, gst = _encode(extract, warm_rgb, edge, args.device, started_perf, "warmup")
            warmup[str(edge)] = {"encode_ms": ms, "gpu": gst, "rel": warm_rel}
            _check_limits(f"after_warmup_{edge}", started_perf)
        protocol["warmup"] = warmup
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        protocol["formal_gpu_baseline"] = _gpu_stats(args.device)
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)
        del warm_rgb

        for cat in CATEGORIES:
            rels = support_paths(cat, SHOT, str(SEED), manifest)
            assert_fit_ids_are_support(rels, cat, SHOT, str(SEED))
            if len(rels) != SHOT:
                raise ValueError(f"expected k2 support for {cat}, got {len(rels)}")
            clean448 = np.zeros((SHOT, GRID_448, GRID_448, FEATURE_DIM), dtype=np.float32)
            clean896 = np.zeros((SHOT, GRID_896, GRID_896, FEATURE_DIM), dtype=np.float32)
            syn448 = np.zeros((SHOT, len(SYN_KINDS) * len(SYN_SEEDS), GRID_448, GRID_448, FEATURE_DIM), dtype=np.float32)
            syn896 = np.zeros((SHOT, len(SYN_KINDS) * len(SYN_SEEDS), GRID_896, GRID_896, FEATURE_DIM), dtype=np.float32)
            syn_masks = np.zeros((SHOT, len(SYN_KINDS) * len(SYN_SEEDS), RAW_SIZE, RAW_SIZE), dtype=np.uint8)
            clean_valid = np.zeros(SHOT, dtype=bool)
            syn_valid = np.zeros((SHOT, len(SYN_KINDS) * len(SYN_SEEDS)), dtype=bool)
            current_cache = (cat, rels, clean448, clean896, syn448, syn896, syn_masks, clean_valid, syn_valid)

            for h, query_rel in enumerate(rels):
                _check_limits(f"before_{cat}_query_{h}", started_perf)
                bank_h = 1 - h
                bank_rel = rels[bank_h]
                if not clean_valid[bank_h]:
                    bank_rgb = _load_rgb(bank_rel)
                    bank448, bank448_ms, bank_gpu448 = _encode(extract, bank_rgb, FULL448_EDGE, args.device, started_perf, f"bank_{cat}_{bank_h}")
                    bank896, bank896_ms, bank_gpu896 = _encode(extract, bank_rgb, FULL896_EDGE, args.device, started_perf, f"bank_{cat}_{bank_h}")
                    clean448[bank_h] = bank448
                    clean896[bank_h] = bank896
                    clean_valid[bank_h] = True
                    del bank_rgb, bank448, bank896
                else:
                    bank448_ms = 0.0
                    bank896_ms = 0.0
                    bank_gpu448 = _gpu_stats(args.device)
                    bank_gpu896 = _gpu_stats(args.device)
                bank448 = clean448[bank_h]
                bank896 = clean896[bank_h]

                query_rgb = _load_rgb(query_rel)
                if not clean_valid[h]:
                    q448, clean448_ms, clean_gpu448 = _encode(extract, query_rgb, FULL448_EDGE, args.device, started_perf, f"clean_{cat}_{h}")
                    q896, clean896_ms, clean_gpu896 = _encode(extract, query_rgb, FULL896_EDGE, args.device, started_perf, f"clean_{cat}_{h}")
                    clean448[h] = q448
                    clean896[h] = q896
                    clean_valid[h] = True
                    del q448, q896
                else:
                    clean448_ms = 0.0
                    clean896_ms = 0.0
                    clean_gpu448 = _gpu_stats(args.device)
                    clean_gpu896 = _gpu_stats(args.device)
                _append_episode_rows(
                    rows, protocol, "running", cat, h, query_rel, bank_h, bank_rel,
                    "clean", -1, "clean", None, None, None, clean448[h], clean896[h],
                    bank448, bank896, clean448_ms, clean896_ms, bank448_ms, bank896_ms,
                    clean_gpu448, clean_gpu896, bank_gpu448, bank_gpu896, True,
                    args.device, started_perf,
                )

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
                        if not syn_valid[h, e]:
                            sf448, syn448_ms, syn_gpu448 = _encode(extract, img, FULL448_EDGE, args.device, started_perf, f"syn_{cat}_{h}_{kind}_{seed}")
                            sf896, syn896_ms, syn_gpu896 = _encode(extract, img, FULL896_EDGE, args.device, started_perf, f"syn_{cat}_{h}_{kind}_{seed}")
                            syn448[h, e] = sf448
                            syn896[h, e] = sf896
                            syn_valid[h, e] = True
                            del sf448, sf896
                        else:
                            syn448_ms = 0.0
                            syn896_ms = 0.0
                            syn_gpu448 = _gpu_stats(args.device)
                            syn_gpu896 = _gpu_stats(args.device)
                        _append_episode_rows(
                            rows, protocol, "running", cat, h, query_rel, bank_h, bank_rel,
                            kind, int(seed), "synthetic", int(seed), mask, mask_ok,
                            syn448[h, e], syn896[h, e], bank448, bank896, syn448_ms, syn896_ms,
                            bank448_ms, bank896_ms, syn_gpu448, syn_gpu896, bank_gpu448,
                            bank_gpu896, True, args.device, started_perf,
                        )
                        del img, mask

                for e, key in enumerate(NUI_KEYS):
                    _check_limits(f"before_render_nuisance_{cat}_{h}_{e}", started_perf)
                    img, _ = R.render_by_key(query_rgb, key, cat, query_rel)
                    nf448, nui448_ms, nui_gpu448 = _encode(extract, img, FULL448_EDGE, args.device, started_perf, f"nuisance_{cat}_{h}_{e}")
                    nf896, nui896_ms, nui_gpu896 = _encode(extract, img, FULL896_EDGE, args.device, started_perf, f"nuisance_{cat}_{h}_{e}")
                    _append_episode_rows(
                        rows, protocol, "running", cat, h, query_rel, bank_h, bank_rel,
                        "nuisance", int(e), key, None, None, None, nf448, nf896,
                        bank448, bank896, nui448_ms, nui896_ms, bank448_ms, bank896_ms,
                        nui_gpu448, nui_gpu896, bank_gpu448, bank_gpu896, False,
                        args.device, started_perf,
                    )
                    del img, nf448, nf896

                _save_feature_cache(cat, rels, clean448, clean896, syn448, syn896, syn_masks, clean_valid, syn_valid)
                protocol.setdefault("feature_cache", {})[cat] = {
                    "path": str((OUT / "features" / f"{cat}.npz").relative_to(ROOT)),
                    "clean_valid": clean_valid.astype(int).tolist(),
                    "syn_valid": syn_valid.astype(int).tolist(),
                }
                _write_json(EXP_OUT / "PROTOCOL.json", protocol)
                del query_rgb, bank448, bank896
            _save_feature_cache(cat, rels, clean448, clean896, syn448, syn896, syn_masks, clean_valid, syn_valid)
            protocol.setdefault("feature_cache", {})[cat] = {
                "path": str((OUT / "features" / f"{cat}.npz").relative_to(ROOT)),
                "clean_valid": clean_valid.astype(int).tolist(),
                "syn_valid": syn_valid.astype(int).tolist(),
                "bytes_uncompressed_float32": int(clean448.nbytes + clean896.nbytes + syn448.nbytes + syn896.nbytes),
            }
            _write_json(EXP_OUT / "PROTOCOL.json", protocol)
    except (DeadlineReached, RuntimeBudgetReached) as exc:
        status = "partial_deadline" if isinstance(exc, DeadlineReached) else "partial_budget"
        error = str(exc)
    except RuntimeError as exc:
        if _is_oom(exc):
            status = "partial_oom"
            error = f"{type(exc).__name__}: {exc}"
            protocol["oom_fallback_status"] = "same 448 / 896 resolution on CPU explicit rerun; no resolution reduction"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            status = "error_partial"
            error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - started_perf
        protocol["finished_utc"] = _utc_now()
        protocol["elapsed_s"] = float(elapsed)
        protocol["final_gpu"] = _gpu_stats(args.device)
        if current_cache is not None:
            c_cat, c_rels, c_clean448, c_clean896, c_syn448, c_syn896, c_masks, c_clean_valid, c_syn_valid = current_cache
            _save_feature_cache(c_cat, c_rels, c_clean448, c_clean896, c_syn448, c_syn896, c_masks, c_clean_valid, c_syn_valid)
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
