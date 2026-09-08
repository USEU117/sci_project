"""Real MPDD full44832 versus full89664 resolution diagnostic.

All six MPDD categories use the complete cached seed-0/k2 test sample list and
448 x 448 masks.  The two arms share one frozen DINOv2 ViT-B/14 checkpoint,
the same two-image normal memory, the same normalized nearest-neighbor
distance, and the same dists2map post-processing; only input edge/resolution
changes.  Test labels and masks are read for evaluation only.

The script reports full 448 x 448 pixel AP/AUROC as the primary diagnostic,
an A1-compatible stride-8 AP/AUROC diagnostic without calling AUPRO, image
AUROC/AP from map maxima, per-category and per-defect-type results, and the
fraction of fine masks whose positive pixels disappear under stride 8.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

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
from torchvision import transforms  # noqa: E402

import ntof_render as _unused_renderer  # noqa: E402,F401
from evaluate_a1_feature_fusion import STRIDE as A1_STRIDE  # noqa: E402
from src.backbones import get_model  # noqa: E402
from src.utils import dists2map  # noqa: E402
from v14_common import DATA_ROOT, assert_fit_ids_are_support, load_manifest, support_paths  # noqa: E402


SCRIPT_VERSION = "real_resolution_v1"
CATEGORIES = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]
SHOT = 2
SEED = 0
RAW_SIZE = 1024
MAP_SIZE = 448
GRID_448 = 32
GRID_896 = 64
EDGE_448 = 448
EDGE_896 = 896
FEATURE_DIM = 768
ARM_448 = "full44832"
ARM_896 = "full89664"
NN_BLOCK = 512
MAX_RUNTIME_S = 60 * 60
EARLY_STOP_UTC = datetime(2026, 9, 7, 22, 30, tzinfo=timezone.utc).timestamp()
DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()
TEST_CACHE = ROOT / "outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k2/anomalydino_visual"
EXP_OUT = ROOT / "experiments/dynamic_fusion/innovation_overnight_20260908/real_resolution"
OUT = ROOT / "outputs/dynamic_fusion/overnight_20260908_real_resolution"

EXPECTED_COUNTS = {
    "bracket_black": 79,
    "bracket_brown": 77,
    "bracket_white": 60,
    "connector": 44,
    "metal_plate": 97,
    "tubes": 101,
}


class EarlyStopReached(RuntimeError):
    pass


class DeadlineReached(RuntimeError):
    pass


class RuntimeBudgetReached(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_limits(stage: str, started_perf: float) -> None:
    now = time.time()
    if now >= DEADLINE_UTC:
        raise DeadlineReached(stage)
    if now >= EARLY_STOP_UTC:
        raise EarlyStopReached(stage)
    if time.perf_counter() - started_perf >= MAX_RUNTIME_S:
        raise RuntimeBudgetReached(stage)


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_stats(device: str) -> dict[str, float | None]:
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


def _sha256_array(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _load_rgb(rel: str) -> np.ndarray:
    rel_norm = rel.replace("\\", "/")
    if "/train/" not in f"/{rel_norm}" and "/test/" not in f"/{rel_norm}":
        raise ValueError(f"unexpected image role/path: {rel}")
    p = DATA_ROOT / rel
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(p))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (RAW_SIZE, RAW_SIZE):
        raise ValueError(f"expected 1024 x 1024 for {rel}, got {rgb.shape}")
    return rgb


def _load_test_cache(cat: str) -> dict[str, Any]:
    path = TEST_CACHE / f"{cat}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"cached test/mask file missing: {path}")
    with np.load(path, allow_pickle=False) as z:
        sample_ids = z["sample_ids"].astype(str)
        masks = np.asarray(z["imgs_masks"], dtype=np.uint8)
        gt_sp = np.asarray(z["gt_sp"], dtype=np.uint8)
        grid = tuple(int(x) for x in z["grid_size"])
        dataset = str(np.asarray(z["dataset"]).item())
        role = str(np.asarray(z["dataset_role"]).item())
        branch = str(np.asarray(z["branch"]).item())
        seed = int(np.asarray(z["seed"]))
        shot = int(np.asarray(z["shot"]))
    if len(sample_ids) != EXPECTED_COUNTS[cat]:
        raise ValueError(f"{cat}: expected {EXPECTED_COUNTS[cat]} cached sample_ids, got {len(sample_ids)}")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"{cat}: cached sample_ids are not unique")
    if masks.shape != (len(sample_ids), MAP_SIZE, MAP_SIZE):
        raise ValueError(f"{cat}: expected masks [{len(sample_ids)},448,448], got {masks.shape}")
    if gt_sp.shape != (len(sample_ids),):
        raise ValueError(f"{cat}: gt_sp shape mismatch {gt_sp.shape}")
    if grid != (GRID_448, GRID_448) or dataset != "mpdd" or seed != SEED or shot != SHOT:
        raise ValueError(f"{cat}: cache metadata mismatch grid={grid} dataset={dataset} seed={seed} shot={shot}")
    if branch != "anomalydino_visual":
        raise ValueError(f"{cat}: unexpected cache branch {branch}")
    for rel, label, mask in zip(sample_ids.tolist(), gt_sp.tolist(), masks):
        parts = rel.replace("\\", "/").split("/")
        if len(parts) != 4 or parts[0] != cat or parts[1] != "test":
            raise ValueError(f"{cat}: unexpected cached sample_id {rel}")
        if not (DATA_ROOT / rel).is_file():
            raise FileNotFoundError(f"cached sample image missing: {DATA_ROOT / rel}")
        if int(label) == 0 and np.any(mask > 0):
            raise ValueError(f"{cat}: normal sample has nonzero cached mask {rel}")
        if int(label) == 1 and not np.any(mask > 0):
            raise ValueError(f"{cat}: anomaly sample has empty cached mask {rel}")
    return {
        "sample_ids": sample_ids,
        "masks": masks,
        "gt_sp": gt_sp,
        "grid": grid,
        "cache_path": str(path.relative_to(ROOT)),
        "cache_sha256": _sha256_array(masks),
        "dataset_role": role,
    }


def _make_extractor(device: str):
    """Use one existing DINO weight with the two fixed image transforms."""
    model = get_model("dinov2_vitb14", device, smaller_edge_size=EDGE_896)
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
        for edge in (EDGE_448, EDGE_896)
    }

    def extract(rgb: np.ndarray, edge: int) -> np.ndarray:
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
            raise ValueError(f"edge {edge}: expected {expected} x {FEATURE_DIM}, got {grid} / {arr.shape}")
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


def _score_map(
    query: np.ndarray,
    bank: np.ndarray,
    grid: int,
    device: str,
    started_perf: float,
    stage: str,
) -> np.ndarray:
    """A1-compatible normalized L2 squared / 2 NN score using torch blocks."""
    q = torch.from_numpy(_l2_rows(query)).to(device)
    b = torch.from_numpy(_l2_rows(bank)).to(device)
    out = np.empty(q.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, q.shape[0], NN_BLOCK):
            _check_limits(f"{stage}_nn_block_{start}", started_perf)
            end = min(start + NN_BLOCK, q.shape[0])
            sim = q[start:end] @ b.T
            # IndexFlatL2 on unit vectors followed by distance / 2 gives
            # 1 - cosine, which is the frozen A1 DINO-only convention.
            dist = 1.0 - sim.max(dim=1).values
            out[start:end] = dist.detach().cpu().numpy().astype(np.float32, copy=False)
            del sim, dist
    del q, b
    _sync(device)
    _check_limits(f"after_{stage}_nn", started_perf)
    return np.asarray(dists2map(out.reshape(grid, grid), (MAP_SIZE, MAP_SIZE)), dtype=np.float32)


def _safe_pixel_metrics(maps: np.ndarray, masks: np.ndarray) -> dict[str, float | None]:
    labels = (np.asarray(masks).reshape(-1) > 0).astype(np.int32)
    scores = np.asarray(maps, dtype=np.float32).reshape(-1)
    if np.unique(labels).size < 2:
        return {"pixel_ap": None, "pixel_auroc": None, "valid": False}
    return {
        "pixel_ap": float(average_precision_score(labels, scores)),
        "pixel_auroc": float(roc_auc_score(labels, scores)),
        "valid": True,
    }


def _a1_stride_metrics(maps: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    # Mirrors evaluate_a1_feature_fusion.compute_metrics' [::STRIDE, ::STRIDE]
    # sampling but deliberately omits its expensive AUPRO calculation.
    maps_s = np.asarray(maps)[:, ::A1_STRIDE, ::A1_STRIDE]
    masks_s = np.asarray(masks)[:, ::A1_STRIDE, ::A1_STRIDE]
    out = _safe_pixel_metrics(maps_s, masks_s)
    out.update({
        "stride": int(A1_STRIDE),
        "map_shape": list(maps_s.shape[1:]),
        "positive_pixels": int((masks_s > 0).sum()),
        "total_pixels": int(masks_s.size),
    })
    return out


def _image_metrics(gt_sp: np.ndarray, maps: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(gt_sp, dtype=np.int32)
    scores = np.asarray(maps, dtype=np.float32).reshape(len(labels), -1).max(axis=1)
    if np.unique(labels).size < 2:
        return {"image_auroc": None, "image_ap": None}
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
    }


def _single_ap(score_map: np.ndarray, mask: np.ndarray) -> float | None:
    labels = (np.asarray(mask).reshape(-1) > 0).astype(np.int32)
    if np.unique(labels).size < 2:
        return None
    return float(average_precision_score(labels, np.asarray(score_map).reshape(-1)))


def _defect_type(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    if len(parts) != 4 or parts[1] != "test":
        raise ValueError(f"unexpected test sample id: {rel}")
    return parts[2]


def _mean_defined(values: list[float | None]) -> float | None:
    x = [float(v) for v in values if v is not None]
    return float(np.mean(x)) if x else None


def _fine_summary(masks: np.ndarray) -> dict[str, Any]:
    anomaly_masks = np.asarray(masks)[np.asarray(masks).reshape(len(masks), -1).sum(axis=1) > 0]
    full_counts = anomaly_masks.reshape(len(anomaly_masks), -1).sum(axis=1) if len(anomaly_masks) else np.asarray([], dtype=np.int64)
    stride_masks = anomaly_masks[:, ::A1_STRIDE, ::A1_STRIDE] if len(anomaly_masks) else anomaly_masks
    stride_counts = stride_masks.reshape(len(stride_masks), -1).sum(axis=1) if len(stride_masks) else np.asarray([], dtype=np.int64)
    disappeared = stride_counts == 0
    ratios = stride_counts / np.maximum(full_counts, 1) if len(full_counts) else np.asarray([], dtype=np.float64)
    return {
        "anomaly_images": int(len(anomaly_masks)),
        "full_positive_pixels_mean": float(full_counts.mean()) if len(full_counts) else None,
        "stride8_positive_pixels_mean": float(stride_counts.mean()) if len(stride_counts) else None,
        "stride8_zero_mask_count": int(disappeared.sum()) if len(disappeared) else 0,
        "stride8_zero_mask_rate": float(disappeared.mean()) if len(disappeared) else None,
        "stride8_positive_fraction_of_full_mean": float(ratios.mean()) if len(ratios) else None,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def _write_partial(rows: list[dict[str, Any]], category_summaries: dict[str, Any], protocol: dict[str, Any], status: str, error: str | None) -> None:
    _write_json(
        OUT / "PARTIAL_RESULTS.json",
        {
            "protocol": protocol,
            "status": status,
            "error": error,
            "updated_utc": _utc_now(),
            "rows": rows,
            "category_summaries": category_summaries,
        },
    )
    with (OUT / "PER_IMAGE.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def _initial_protocol() -> dict[str, Any]:
    p = {
        "script_version": SCRIPT_VERSION,
        "role": "support-only real MPDD resolution diagnostic; no independent confirmation",
        "dataset": "MPDD",
        "categories": CATEGORIES,
        "seed": SEED,
        "shot": SHOT,
        "memory": "all two manifest train/good support images per category; K=2; no LOO; query test image excluded",
        "test_cache": str(TEST_CACHE.relative_to(ROOT)),
        "expected_test_counts": {**EXPECTED_COUNTS, "total": int(sum(EXPECTED_COUNTS.values())), "normal": 176, "anomaly": 282},
        "input": {
            "raw_shape": [RAW_SIZE, RAW_SIZE],
            "full44832": {"smaller_edge_size": EDGE_448, "native_grid": [GRID_448, GRID_448]},
            "full89664": {"smaller_edge_size": EDGE_896, "native_grid": [GRID_896, GRID_896]},
            "map_size": [MAP_SIZE, MAP_SIZE],
            "feature_dim": FEATURE_DIM,
            "backbone": "existing DINOv2 ViT-B/14 checkpoint, one frozen model, no download",
        },
        "distance": "normalized nearest-neighbor 1 - cosine; equivalent to normalized L2 squared distance / 2",
        "map_postprocess": "same dists2map to 448 x 448 for both arms",
        "metrics": {
            "full_pixel": ["pixel_ap", "pixel_auroc"],
            "a1_stride": int(A1_STRIDE),
            "a1_stride_metrics": ["pixel_ap", "pixel_auroc"],
            "image": ["image_auroc", "image_ap"],
            "image_score": "max over 448 x 448 map",
            "per_defect_type": "each defect type against all category good normals",
            "fine_anomaly": "full and stride8 positive mask pixels plus stride8 disappearance count/rate",
            "aupro": "not computed",
        },
        "no_test_selection": "all cached sample_ids and masks are read; test labels are evaluation-only and never used to fit/select",
        "resources": {
            "device": "cuda:0",
            "batch": 1,
            "torch_threads": 2,
            "nn_block": NN_BLOCK,
            "gpu": "RTX 3060 6 GB",
            "max_runtime_s": MAX_RUNTIME_S,
        },
        "timing": "warmup separate; formal per-image encode and score latency; torch allocator and whole-device memory separate",
        "early_stop_utc": datetime(2026, 9, 7, 22, 30, tzinfo=timezone.utc).isoformat(),
        "hard_deadline_utc": datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).isoformat(),
        "oom_fallback": "same 448 / 896 configuration on CPU in explicit rerun; never lower resolution",
        "comparison_boundary": "DINO full44832 low-resolution control only; no A1 fusion comparison",
    }
    _write_json(EXP_OUT / "PROTOCOL.json", p)
    return p


def _category_summary(
    cat: str,
    sample_ids: np.ndarray,
    gt_sp: np.ndarray,
    masks: np.ndarray,
    maps448: np.ndarray,
    maps896: np.ndarray,
) -> dict[str, Any]:
    types = np.asarray([_defect_type(x) for x in sample_ids.tolist()])
    out: dict[str, Any] = {
        "category": cat,
        "n_samples": int(len(sample_ids)),
        "n_normal": int((gt_sp == 0).sum()),
        "n_anomaly": int((gt_sp == 1).sum()),
        "arms": {},
        "defect_types": {},
        "fine_anomaly": _fine_summary(masks),
    }
    for arm, maps in ((ARM_448, maps448), (ARM_896, maps896)):
        out["arms"][arm] = {
            "full_pixel": _safe_pixel_metrics(maps, masks),
            "stride8": _a1_stride_metrics(maps, masks),
            "image": _image_metrics(gt_sp, maps),
        }
    for dtype in sorted(x for x in set(types.tolist()) if x != "good"):
        keep = np.logical_or(types == "good", types == dtype)
        drow: dict[str, Any] = {
            "n_normal": int(np.logical_and(keep, gt_sp == 0).sum()),
            "n_anomaly": int(np.logical_and(keep, gt_sp == 1).sum()),
            "fine_anomaly": _fine_summary(masks[keep]),
        }
        for arm, maps in ((ARM_448, maps448), (ARM_896, maps896)):
            drow[arm] = {
                "full_pixel": _safe_pixel_metrics(maps[keep], masks[keep]),
                "stride8": _a1_stride_metrics(maps[keep], masks[keep]),
                "image": _image_metrics(gt_sp[keep], maps[keep]),
            }
        out["defect_types"][dtype] = drow
    return out


def _macro_category(category_summaries: dict[str, Any]) -> dict[str, Any]:
    out = {"n_categories": len(category_summaries), "arms": {}, "fine_anomaly": {}}
    for arm in (ARM_448, ARM_896):
        out["arms"][arm] = {
            "full_pixel": {
                key: _mean_defined([x["arms"][arm]["full_pixel"].get(key) for x in category_summaries.values()])
                for key in ("pixel_ap", "pixel_auroc")
            },
            "stride8": {
                key: _mean_defined([x["arms"][arm]["stride8"].get(key) for x in category_summaries.values()])
                for key in ("pixel_ap", "pixel_auroc")
            },
            "image": {
                key: _mean_defined([x["arms"][arm]["image"].get(key) for x in category_summaries.values()])
                for key in ("image_ap", "image_auroc")
            },
        }
    fine = [x["fine_anomaly"] for x in category_summaries.values()]
    out["fine_anomaly"] = {
        "mean_stride8_zero_mask_rate": _mean_defined([x.get("stride8_zero_mask_rate") for x in fine]),
        "mean_stride8_positive_fraction_of_full": _mean_defined([x.get("stride8_positive_fraction_of_full_mean") for x in fine]),
        "total_anomaly_images": int(sum(x.get("anomaly_images", 0) for x in fine)),
        "total_stride8_zero_mask_count": int(sum(x.get("stride8_zero_mask_count", 0) for x in fine)),
    }
    return out


def _macro_defect_type(category_summaries: dict[str, Any]) -> dict[str, Any]:
    groups = []
    for c in category_summaries.values():
        for dtype, d in c["defect_types"].items():
            groups.append((c["category"], dtype, d))
    out = {"n_defect_type_groups": len(groups), "arms": {}, "groups": {}}
    for cat, dtype, d in groups:
        out["groups"][f"{cat}/{dtype}"] = d
    for arm in (ARM_448, ARM_896):
        out["arms"][arm] = {
            "full_pixel": {
                key: _mean_defined([d[arm]["full_pixel"].get(key) for _, _, d in groups])
                for key in ("pixel_ap", "pixel_auroc")
            },
            "stride8": {
                key: _mean_defined([d[arm]["stride8"].get(key) for _, _, d in groups])
                for key in ("pixel_ap", "pixel_auroc")
            },
            "image": {
                key: _mean_defined([d[arm]["image"].get(key) for _, _, d in groups])
                for key in ("image_ap", "image_auroc")
            },
        }
    return out


def _write_final(
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    category_summaries: dict[str, Any],
    status: str,
    elapsed_s: float,
    error: str | None,
) -> None:
    expected_rows = int(sum(EXPECTED_COUNTS.values()) * 2)
    complete_coverage = (
        status == "complete"
        and len(rows) == expected_rows
        and set(category_summaries) == set(CATEGORIES)
        and all(x["n_samples"] == EXPECTED_COUNTS[c] for c, x in category_summaries.items())
    )
    if status == "complete" and not complete_coverage:
        status = "error_partial"
        error = error or f"coverage incomplete: rows={len(rows)} expected={expected_rows}"
    summary = {
        "n_rows": len(rows),
        "expected_rows": expected_rows,
        "n_categories": len(category_summaries),
        "categories": {c: category_summaries[c]["n_samples"] for c in category_summaries},
        "n_test_images": int(sum(x["n_samples"] for x in category_summaries.values())),
        "n_normal": int(sum(x["n_normal"] for x in category_summaries.values())),
        "n_anomaly": int(sum(x["n_anomaly"] for x in category_summaries.values())),
        "per_category": category_summaries,
        "category_macro": _macro_category(category_summaries) if category_summaries else None,
        "defect_type_macro": _macro_defect_type(category_summaries) if category_summaries else None,
        "aupro": "not computed",
    }
    decision = (
        "REAL_DIAGNOSTIC_COMPLETE; DINO_RESOLUTION_ONLY; no A1 fusion comparison"
        if status == "complete"
        else "PARTIAL_STOP; no real-resolution claim"
    )
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
    macro = summary["category_macro"]
    lines = [
        "# 真实 MPDD Full 448 / Full 896 诊断结论",
        "",
        f"状态：`{status}`；决策：`{decision}`。",
        "",
        "本轮完整读取 6 类 seed 0、k2 的现有 cached sample_ids 与 448 × 448 masks；每类两张 normal support 全部建 K=2 memory，不做 LOO。两臂使用同一冻结 DINOv2 权重、距离和 dists2map，只有输入 edge 不同。",
        "",
        "full 448 × 448 pixel AP/AUROC 是主指标；stride8 仅为 A1-compatible 诊断，未计算 AUPRO。image score 固定为 448 × 448 map 的 max。逐类、defect type 与 macro 结果完整保存在 SUMMARY.json。",
        "",
        "category macro：",
        json.dumps(macro, ensure_ascii=False, indent=1),
        "",
        "stride8 细小异常消失计数/比例在每个 category、defect type 和 per-image row 中记录；任何 stride8 指标无效都保留为 null，不用 full-resolution 标签补值。",
        "",
        f"实际墙钟时间：`{elapsed_s:.3f} s`。逐图结果见 `outputs/dynamic_fusion/overnight_20260908_real_resolution/PER_IMAGE.jsonl`；partial 也保留。",
        "",
        "边界：这是真实 MPDD 的 DINO resolution-only 诊断，不是独立确认；不与 A1 fusion 比较，不用 test 结果改配置，也不据此宣称论文创新已证实。",
    ]
    (EXP_OUT / "DECISION_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = _initial_protocol()
    protocol["started_utc"] = _utc_now()
    _write_json(EXP_OUT / "PROTOCOL.json", protocol)
    rows: list[dict[str, Any]] = []
    category_summaries: dict[str, Any] = {}
    started_perf = time.perf_counter()
    status = "complete"
    error: str | None = None
    model = None
    try:
        _check_limits("before_cache_validation", started_perf)
        cached = {cat: _load_test_cache(cat) for cat in CATEGORIES}
        protocol["loaded_cache_counts"] = {cat: int(len(cached[cat]["sample_ids"])) for cat in CATEGORIES}
        protocol["loaded_cache_sha256"] = {cat: cached[cat]["cache_sha256"] for cat in CATEGORIES}
        protocol["loaded_total"] = int(sum(protocol["loaded_cache_counts"].values()))
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)

        manifest = load_manifest()
        support = {}
        for cat in CATEGORIES:
            rels = support_paths(cat, SHOT, str(SEED), manifest)
            assert_fit_ids_are_support(rels, cat, SHOT, str(SEED))
            if len(rels) != SHOT:
                raise ValueError(f"expected k2 support for {cat}, got {len(rels)}")
            support[cat] = rels

        _check_limits("before_model_load", started_perf)
        _sync(args.device)
        t_model = time.perf_counter()
        extract, model = _make_extractor(args.device)
        _sync(args.device)
        protocol["model_load_ms"] = float((time.perf_counter() - t_model) * 1000.0)
        protocol["model_load_gpu"] = _gpu_stats(args.device)
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)

        warm_cat = CATEGORIES[0]
        warm_rgb = _load_rgb(support[warm_cat][0])
        warmup = {}
        for edge in (EDGE_448, EDGE_896):
            _, ms, gst = _encode(extract, warm_rgb, edge, args.device, started_perf, "warmup")
            warmup[str(edge)] = {"encode_ms": ms, "gpu": gst, "rel": support[warm_cat][0]}
        protocol["warmup"] = warmup
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        protocol["formal_gpu_baseline"] = _gpu_stats(args.device)
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)
        del warm_rgb

        for cat in CATEGORIES:
            _check_limits(f"before_category_{cat}", started_perf)
            info = cached[cat]
            sample_ids = info["sample_ids"]
            masks = info["masks"]
            gt_sp = info["gt_sp"]
            rels = support[cat]

            bank448_parts = []
            bank896_parts = []
            bank_encode = {}
            bank_gpu = {}
            for i, rel in enumerate(rels):
                support_rgb = _load_rgb(rel)
                bf448, b448_ms, b448_gpu = _encode(extract, support_rgb, EDGE_448, args.device, started_perf, f"bank_{cat}_{i}")
                bf896, b896_ms, b896_gpu = _encode(extract, support_rgb, EDGE_896, args.device, started_perf, f"bank_{cat}_{i}")
                bank448_parts.append(bf448.reshape(-1, FEATURE_DIM))
                bank896_parts.append(bf896.reshape(-1, FEATURE_DIM))
                bank_encode.setdefault(ARM_448, 0.0)
                bank_encode.setdefault(ARM_896, 0.0)
                bank_encode[ARM_448] += b448_ms
                bank_encode[ARM_896] += b896_ms
                bank_gpu[ARM_448] = b448_gpu
                bank_gpu[ARM_896] = b896_gpu
                del support_rgb, bf448, bf896
            bank448 = np.concatenate(bank448_parts, axis=0).astype(np.float32, copy=False)
            bank896 = np.concatenate(bank896_parts, axis=0).astype(np.float32, copy=False)

            maps448 = []
            maps896 = []
            image_rows = []
            for i, (rel, mask, label) in enumerate(zip(sample_ids.tolist(), masks, gt_sp.tolist())):
                _check_limits(f"before_image_{cat}_{i}", started_perf)
                rgb = _load_rgb(rel)
                q448, enc448_ms, gpu448 = _encode(extract, rgb, EDGE_448, args.device, started_perf, f"query_{cat}_{i}")
                q896, enc896_ms, gpu896 = _encode(extract, rgb, EDGE_896, args.device, started_perf, f"query_{cat}_{i}")
                t0 = time.perf_counter()
                map448 = _score_map(q448, bank448, GRID_448, args.device, started_perf, f"score_{cat}_{i}_{ARM_448}")
                _sync(args.device)
                score448_ms = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                map896 = _score_map(q896, bank896, GRID_896, args.device, started_perf, f"score_{cat}_{i}_{ARM_896}")
                _sync(args.device)
                score896_ms = (time.perf_counter() - t1) * 1000.0
                _check_limits(f"after_image_scores_{cat}_{i}", started_perf)
                maps448.append(map448)
                maps896.append(map896)

                stride_mask = mask[::A1_STRIDE, ::A1_STRIDE]
                full_pixels = int((mask > 0).sum())
                stride_pixels = int((stride_mask > 0).sum())
                common = {
                    "category": cat,
                    "sample_id": rel,
                    "anomaly_type": _defect_type(rel),
                    "gt_sp": int(label),
                    "gt_area_pixels_full448": full_pixels,
                    "gt_area_fraction_full448": float(full_pixels / (MAP_SIZE * MAP_SIZE)),
                    "gt_area_pixels_stride8": stride_pixels,
                    "gt_area_fraction_stride8": float(stride_pixels / (MAP_SIZE // A1_STRIDE) ** 2),
                    "stride8_mask_disappears": bool(int(label) == 1 and stride_pixels == 0),
                    "mask_sha256": _sha256_array(mask),
                    "memory_k": SHOT,
                    "memory_leave_one_out": False,
                    "query_in_memory": False,
                }
                for arm, score_map, enc_ms, score_ms, gpu, grid in (
                    (ARM_448, map448, enc448_ms, score448_ms, gpu448, GRID_448),
                    (ARM_896, map896, enc896_ms, score896_ms, gpu896, GRID_896),
                ):
                    stride_map = score_map[::A1_STRIDE, ::A1_STRIDE]
                    image_row = {
                        **common,
                        "arm": arm,
                        "grid": grid,
                        "image_score_max": float(score_map.max()),
                        "pixel_ap_full448_per_image": _single_ap(score_map, mask),
                        "pixel_ap_stride8_per_image": _single_ap(stride_map, stride_mask),
                        "encode_ms": float(enc_ms),
                        "score_ms": float(score_ms),
                        "episode_elapsed_ms": float(enc_ms + score_ms),
                        "bank_encode_ms": float(bank_encode[arm]),
                        "gpu": gpu,
                        "bank_gpu": bank_gpu[arm],
                        "tokens": int((GRID_448 * GRID_448) if arm == ARM_448 else (GRID_896 * GRID_896)),
                        "a1_stride": int(A1_STRIDE),
                        "aupro": None,
                    }
                    rows.append(image_row)
                    image_rows.append(image_row)
                _write_partial(rows, category_summaries, protocol, "running", None)
                _check_limits(f"after_image_{cat}_{i}", started_perf)
                del rgb, q448, q896, map448, map896

            maps448_arr = np.stack(maps448).astype(np.float32, copy=False)
            maps896_arr = np.stack(maps896).astype(np.float32, copy=False)
            category_summaries[cat] = _category_summary(cat, sample_ids, gt_sp, masks, maps448_arr, maps896_arr)
            category_summaries[cat]["bank"] = {
                "k": SHOT,
                "memory_images": rels,
                "memory_cells": {ARM_448: int(bank448.shape[0]), ARM_896: int(bank896.shape[0])},
                "encode_ms": bank_encode,
                "gpu_last": bank_gpu,
            }
            _write_json(OUT / f"CATEGORY_{cat}.json", category_summaries[cat])
            protocol.setdefault("completed_categories", []).append(cat)
            _write_json(EXP_OUT / "PROTOCOL.json", protocol)
            _write_partial(rows, category_summaries, protocol, "running", None)
            del bank448, bank896, maps448, maps896, maps448_arr, maps896_arr, image_rows

    except (EarlyStopReached, DeadlineReached, RuntimeBudgetReached) as exc:
        if isinstance(exc, EarlyStopReached):
            status = "partial_early_stop"
        elif isinstance(exc, DeadlineReached):
            status = "partial_deadline"
        else:
            status = "partial_budget"
        error = str(exc)
    except RuntimeError as exc:
        if _is_oom(exc):
            status = "partial_oom"
            error = f"{type(exc).__name__}: {exc}"
            protocol["oom_fallback_status"] = "same 448 / 896 configuration on CPU explicit rerun; no resolution reduction"
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
        _write_json(EXP_OUT / "PROTOCOL.json", protocol)
        _write_partial(rows, category_summaries, protocol, status, error)
        _write_final(protocol, rows, category_summaries, status, elapsed, error)
        if model is not None and status != "complete" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"status": status, "error": error, "rows": len(rows), "categories": len(category_summaries), "elapsed_s": elapsed}, ensure_ascii=False), flush=True)
    return 0 if status == "complete" else 1


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
