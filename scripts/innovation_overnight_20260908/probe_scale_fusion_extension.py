"""Four-category extension of the cached multi-scale matching control.

The two-category ``scale_fusion`` result is left untouched.  This extension
uses only the completed dual-resolution cache from
``overnight_20260908_full896_extension/features`` for bracket_brown,
bracket_white, connector, and tubes.  Each cache contains the same support
clean/synthetic episode at a 448-edge 32 x 32 grid and an 896-edge native
64 x 64 grid.  No model is loaded and no new feature is extracted.

The four arms and all post-processing are exactly those of the original
scale-fusion control:

* ``full896_native64``;
* ``dino448_bilinear64`` (32 x 32 feature bilinear-resized to 64 x 64);
* ``multiscale_concat64`` (row-L2 each scale, fixed equal weights, concat,
  row-L2 nearest-neighbour);
* ``late_mean64`` (fixed arithmetic mean of the two single-arm score grids).

The new run has 48 unique support-derived masks.  A six-category summary is
also reported by reusing the completed two-category result read-only; the old
files are never overwritten.  Clean support queries are recorded as score
distributions only.  Synthetic queries use their own cached variant and no
clean counterpart oracle is used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (
    ROOT,
    ROOT / "scripts",
    ROOT / "scripts" / "innovation_v14_decisive_validation_20260905",
    ROOT / "methods" / "anomalydino",
):
    sys.path.insert(0, str(p))

from src.utils import dists2map  # noqa: E402
from v14_common import assert_fit_ids_are_support, load_manifest, support_paths  # noqa: E402


EXP_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908"
EXP_OUT = EXP_ROOT / "scale_fusion_extension"
FEATURE_DIR = ROOT / "outputs" / "dynamic_fusion" / "overnight_20260908_full896_extension" / "features"
V14_CACHE = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support" / "v14_p1_support_dino_s0_k2"
LEGACY_RESULT = EXP_ROOT / "scale_fusion" / "RESULTS.json"
FULL896_EXTENSION_PROTOCOL = EXP_ROOT / "full896_extension" / "PROTOCOL.json"

CATEGORIES = ("bracket_brown", "bracket_white", "connector", "tubes")
LEGACY_CATEGORIES = ("bracket_black", "metal_plate")
ALL_CATEGORIES = LEGACY_CATEGORIES + CATEGORIES
SHOT = 2
SEED = "0"
RAW_SIZE = 1024
HIGH_GRID = 64
LOW_GRID = 32
FEATURE_DIM = 768
SYN_KINDS = ("thin_scratch", "cutpaste")
SYN_SEEDS = (0, 1, 2)
N_SYN = len(SYN_KINDS) * len(SYN_SEEDS)
ARMS = (
    "full896_native64",
    "dino448_bilinear64",
    "multiscale_concat64",
    "late_mean64",
)
THREADS = 2
EQUAL_WEIGHT = 1.0 / np.sqrt(2.0)
MAP_GAUSSIAN_SIGMA = 4.0
DEADLINE_UTC = datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)
SCRIPT_VERSION = "scale_fusion_extension_probe_v1"
PROTOCOL_VERSION = "scale_fusion_extension_cached64_support_probe_v1"


class DeadlineReached(RuntimeError):
    """Hard Beijing deadline reached; completed rows remain inspectable."""


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


def _protocol() -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": _utc_now(),
        "role": "support-only cached multi-scale matching extension; no test image or test label",
        "scope": {
            "dataset": "MPDD",
            "new_categories": list(CATEGORIES),
            "legacy_categories_read_only": list(LEGACY_CATEGORIES),
            "seed": int(SEED),
            "shot": SHOT,
            "synthetic_families": list(SYN_KINDS),
            "synthetic_seeds": list(SYN_SEEDS),
            "new_unique_synthetic_masks": 48,
            "six_category_summary": True,
        },
        "cache_index_audit": {
            "full896_extension_protocol": "experiments/dynamic_fusion/innovation_overnight_20260908/full896_extension/PROTOCOL.json",
            "dual_resolution_feature_dir": "outputs/dynamic_fusion/overnight_20260908_full896_extension/features",
            "cache_fields": "clean_feat_448 / clean_feat_896 / syn_feat_448 / syn_feat_896 / syn_masks",
            "cache_role": "existing DINO 448-edge 32 x 32 and 896-edge native 64 x 64 support features",
            "v14_mask_audit_dir": "outputs/dynamic_fusion/v14_p1_support/v14_p1_support_dino_s0_k2",
            "index_rule": "match manifest ref_rel; extension cache kind-major thin_scratch/cutpaste and seed-major episode index",
            "mask_rule": "extension mask must equal the corresponding existing v14 mask byte-for-byte",
            "legacy_result_read_only": "experiments/dynamic_fusion/innovation_overnight_20260908/scale_fusion/RESULTS.json",
        },
        "feature_arms": {
            "full896_native64": "cached 896-edge native 64 x 64 feature; row-L2 then global L2 NN",
            "dino448_bilinear64": "cached 448-edge 32 x 32 feature; fixed bilinear feature interpolation to 64 x 64, row-L2 then global L2 NN",
            "multiscale_concat64": "row-L2 each 64-grid scale, equal weights 1/sqrt(2), concatenate, final row-L2, global L2 NN",
            "late_mean64": "arithmetic mean of the two single-scale 64-grid L2-NN score grids before common map post-processing",
        },
        "fixed_parameters": {
            "high_grid": [HIGH_GRID, HIGH_GRID],
            "low_grid_source": [LOW_GRID, LOW_GRID],
            "feature_dim": FEATURE_DIM,
            "bilinear_mode": "torch.nn.functional.interpolate(mode=bilinear, align_corners=False, size=64 x 64) on CPU",
            "concat_equal_weight": "each row-L2 scale multiplied by 1/sqrt(2); no weight scan",
            "row_normalization": "L2 with floor 1e-8",
            "nearest_neighbor": "global over all 4096 memory cells; Euclidean distance of row-L2 descriptors via sqrt(2 - 2 max cosine)",
            "score_map_postprocess": "src.utils.dists2map: cv2 INTER_LINEAR resize to 1024 x 1024 followed by Gaussian sigma=4",
        },
        "memory_and_query_roles": {
            "memory": "only the other support image in the same category; query image cells excluded",
            "synthetic_query": "own cached 448 and 896 feature variant; no clean counterpart or alignment oracle",
            "normal_control": "own cached clean query feature; clean score distribution only",
            "test_read": False,
            "new_feature_extraction": False,
            "parameter_fit_from_ap": False,
            "type_routing": False,
        },
        "evaluation": {
            "synthetic": "Pixel-AP on the original 1024 x 1024 renderer mask",
            "normal": "clean support score map distribution after common post-processing; no nuisance extraction",
            "comparison": "all four arms shown together for new four categories; six-category summary reuses legacy two-category arm summaries read-only",
            "main_table": False,
        },
        "resources": {"device": "cpu", "threads": THREADS, "gpu_used": False, "new_feature_extraction": False},
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
            raise RuntimeError(f"Frozen scale-fusion extension protocol mismatch: {path}")
        return old
    _write_json(path, p)
    return p


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def _load_v14_mask_index(cat: str) -> dict[str, Any]:
    path = V14_CACHE / f"{cat}.npz"
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        return {
            "rels": z["ref_rel"].astype(str).tolist(),
            "masks": np.asarray(z["syn_masks"]).copy(),
            "kinds": z["syn_kinds"].astype(str).tolist(),
            "seeds": int(np.asarray(z["syn_seeds"])),
        }


def _load_category_cache(cat: str, manifest: dict[str, Any]) -> dict[str, Any]:
    d = _load_npz(FEATURE_DIR / f"{cat}.npz")
    rels = support_paths(cat, SHOT, SEED, manifest)
    assert_fit_ids_are_support(rels, cat, SHOT, SEED)
    if rels != d["ref_rel"].astype(str).tolist():
        raise ValueError(f"support/cache ref_rel mismatch for {cat}: {rels}")
    expected = {
        "clean_feat_448": (SHOT, LOW_GRID, LOW_GRID, FEATURE_DIM),
        "clean_feat_896": (SHOT, HIGH_GRID, HIGH_GRID, FEATURE_DIM),
        "syn_feat_448": (SHOT, N_SYN, LOW_GRID, LOW_GRID, FEATURE_DIM),
        "syn_feat_896": (SHOT, N_SYN, HIGH_GRID, HIGH_GRID, FEATURE_DIM),
        "syn_masks": (SHOT, N_SYN, RAW_SIZE, RAW_SIZE),
    }
    for key, shape in expected.items():
        if tuple(d[key].shape) != shape:
            raise ValueError(f"unexpected {key} shape for {cat}: {d[key].shape}")
    if d["syn_kinds"].astype(str).tolist() != list(SYN_KINDS):
        raise ValueError(f"unexpected extension synthetic ordering for {cat}: {d['syn_kinds']}")
    if d["syn_seeds"].astype(int).tolist() != list(SYN_SEEDS):
        raise ValueError(f"unexpected extension synthetic seeds for {cat}: {d['syn_seeds']}")
    if not np.all(d["clean_valid"] == 1) or not np.all(d["syn_valid"] == 1):
        raise ValueError(f"extension cache has invalid entries for {cat}")
    return {"data": d, "rels": rels, "v14": _load_v14_mask_index(cat)}


def _episode_index(kind: str, seed: int) -> int:
    return SYN_KINDS.index(kind) * len(SYN_SEEDS) + int(seed)


def _audit_masks(cache: dict[str, Any], cat: str) -> list[dict[str, Any]]:
    d = cache["data"]
    v14 = cache["v14"]
    if v14["rels"] != cache["rels"] or v14["seeds"] != len(SYN_SEEDS):
        raise ValueError(f"v14 support/mask index mismatch for {cat}")
    rows: list[dict[str, Any]] = []
    for h, rel in enumerate(cache["rels"]):
        for kind in SYN_KINDS:
            for seed in SYN_SEEDS:
                i = _episode_index(kind, seed)
                vi = v14["kinds"].index(kind) * v14["seeds"] + int(seed)
                mask = d["syn_masks"][h, i]
                equal = bool(np.array_equal(mask, v14["masks"][h, vi]))
                if not equal:
                    raise AssertionError(f"extension/v14 mask mismatch: {cat} {rel} {kind} {seed}")
                rows.append(
                    {
                        "category": cat,
                        "query_index": h,
                        "query_rel": rel,
                        "kind": kind,
                        "seed": seed,
                        "episode": i,
                        "v14_episode": vi,
                        "mask_sha256": _sha256_array(mask),
                        "mask_positive_pixels": int((mask > 0).sum()),
                        "mask_area": float((mask > 0).mean()),
                        "extension_v14_mask_equal": equal,
                    }
                )
    return rows


def _bilinear_to64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (LOW_GRID, LOW_GRID, FEATURE_DIM):
        raise ValueError(f"expected 32 x 32 x 768 feature, got {x.shape}")
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        y = torch.nn.functional.interpolate(t, size=(HIGH_GRID, HIGH_GRID), mode="bilinear", align_corners=False)
    return y[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32, copy=False)


def _l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1.0e-8)


def _build_bank(high: np.ndarray, low32: np.ndarray) -> dict[str, np.ndarray]:
    high_n = _l2_rows(high)
    low_n = _l2_rows(_bilinear_to64(low32))
    concat_n = _l2_rows(np.concatenate((EQUAL_WEIGHT * high_n, EQUAL_WEIGHT * low_n), axis=1))
    return {"high": high_n, "low": low_n, "concat": concat_n}


def _nearest_l2_grid(query: np.ndarray, bank_n: np.ndarray) -> np.ndarray:
    query_n = _l2_rows(query)
    sim = query_n @ bank_n.T
    best = sim.max(axis=1)
    return np.sqrt(np.maximum(2.0 - 2.0 * best, 0.0)).astype(np.float32).reshape(HIGH_GRID, HIGH_GRID)


def _postprocess(score_grid: np.ndarray) -> np.ndarray:
    if score_grid.shape != (HIGH_GRID, HIGH_GRID):
        raise ValueError(f"expected 64 x 64 score grid, got {score_grid.shape}")
    return np.asarray(dists2map(score_grid.astype(np.float32), (RAW_SIZE, RAW_SIZE)), dtype=np.float32)


def _score_maps(query_high: np.ndarray, query_low32: np.ndarray, bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    query_low = _bilinear_to64(query_low32)
    high_grid = _nearest_l2_grid(query_high, bank["high"])
    low_grid = _nearest_l2_grid(query_low, bank["low"])
    query_high_n = _l2_rows(query_high)
    query_low_n = _l2_rows(query_low)
    query_concat = np.concatenate((EQUAL_WEIGHT * query_high_n, EQUAL_WEIGHT * query_low_n), axis=1)
    concat_grid = _nearest_l2_grid(query_concat, bank["concat"])
    late_grid = (high_grid + low_grid) * np.float32(0.5)
    return {
        "full896_native64": _postprocess(high_grid),
        "dino448_bilinear64": _postprocess(low_grid),
        "multiscale_concat64": _postprocess(concat_grid),
        "late_mean64": _postprocess(late_grid),
    }


def _ap(mask: np.ndarray, score_map: np.ndarray) -> float:
    y = (np.asarray(mask).reshape(-1) > 0).astype(np.int32)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, score_map.reshape(-1).astype(np.float64)))


def _score_stats(score_map: np.ndarray) -> dict[str, float]:
    x = np.asarray(score_map, dtype=np.float64).reshape(-1)
    return {
        "score_mean": float(x.mean()),
        "score_std": float(x.std()),
        "score_p50": float(np.percentile(x, 50)),
        "score_p95": float(np.percentile(x, 95)),
        "score_p99": float(np.percentile(x, 99)),
        "score_max": float(x.max()),
    }


def run_category(cat: str, cache: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    d = cache["data"]
    ap_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    mask_rows = _audit_masks(cache, cat)
    for h, query_rel in enumerate(cache["rels"]):
        _check_deadline(f"before_{cat}_h{h}")
        bank_index = 1 - h
        bank = _build_bank(d["clean_feat_896"][bank_index], d["clean_feat_448"][bank_index])
        for kind in SYN_KINDS:
            for seed in SYN_SEEDS:
                _check_deadline(f"before_synthetic_{cat}_h{h}_{kind}_{seed}")
                i = _episode_index(kind, seed)
                maps = _score_maps(d["syn_feat_896"][h, i], d["syn_feat_448"][h, i], bank)
                mask = d["syn_masks"][h, i]
                for arm in ARMS:
                    ap_rows.append(
                        {
                            "category": cat,
                            "query_index": h,
                            "query_rel": query_rel,
                            "bank_index": bank_index,
                            "bank_rel": cache["rels"][bank_index],
                            "strict_leave_image_memory": True,
                            "query_clean_counterpart_used": False,
                            "kind": kind,
                            "seed": seed,
                            "episode": i,
                            "arm": arm,
                            "grid": HIGH_GRID,
                            "mask_sha256": _sha256_array(mask),
                            "mask_positive_pixels": int((mask > 0).sum()),
                            "mask_area": float((mask > 0).mean()),
                            "mask_cache_verified": True,
                            "pixel_ap": _ap(mask, maps[arm]),
                            "feature_source": "cached full896_extension dual-resolution feature",
                            "normalization": "row-L2 only; fixed equal concat weight where applicable",
                            "postprocess": "dists2map bilinear + Gaussian sigma4",
                        }
                    )
        _check_deadline(f"before_clean_{cat}_h{h}")
        clean_maps = _score_maps(d["clean_feat_896"][h], d["clean_feat_448"][h], bank)
        for arm in ARMS:
            normal_rows.append(
                {
                    "category": cat,
                    "query_index": h,
                    "query_rel": query_rel,
                    "bank_index": bank_index,
                    "bank_rel": cache["rels"][bank_index],
                    "strict_leave_image_memory": True,
                    "arm": arm,
                    "grid": HIGH_GRID,
                    "feature_source": "cached full896_extension dual-resolution clean feature",
                    **_score_stats(clean_maps[arm]),
                }
            )
    return ap_rows, normal_rows, mask_rows


def _finite_mean(values: list[float]) -> float:
    x = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def _method_summary(ap_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rr = [r for r in ap_rows if r["arm"] == arm]
    nn = [r for r in normal_rows if r["arm"] == arm]
    return {
        "thin_scratch_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "thin_scratch"]),
        "cutpaste_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "cutpaste"]),
        "synthetic_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rr]),
        "normal_clean_score_mean_macro": _finite_mean([r["score_mean"] for r in nn]),
        "normal_clean_score_p95_macro": _finite_mean([r["score_p95"] for r in nn]),
        "normal_clean_score_p99_macro": _finite_mean([r["score_p99"] for r in nn]),
        "n_synthetic_rows": len(rr),
        "n_normal_rows": len(nn),
    }


def _category_summary(ap_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], cat: str, arm: str) -> dict[str, float]:
    rr = [r for r in ap_rows if r["category"] == cat and r["arm"] == arm]
    nn = [r for r in normal_rows if r["category"] == cat and r["arm"] == arm]
    return {
        "thin_scratch_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "thin_scratch"]),
        "cutpaste_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "cutpaste"]),
        "synthetic_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr]),
        "normal_clean_score_mean": _finite_mean([r["score_mean"] for r in nn]),
        "normal_clean_score_p95": _finite_mean([r["score_p95"] for r in nn]),
        "normal_clean_score_p99": _finite_mean([r["score_p99"] for r in nn]),
    }


def _six_summary(new_per_category: dict[str, Any], legacy_result: dict[str, Any]) -> dict[str, Any]:
    cats = {}
    for cat in LEGACY_CATEGORIES:
        cats[cat] = legacy_result["per_category"][cat]
    for cat in CATEGORIES:
        cats[cat] = new_per_category[cat]
    methods: dict[str, Any] = {}
    for arm in ARMS:
        vals = [cats[cat][arm] for cat in ALL_CATEGORIES]
        methods[arm] = {
            "thin_scratch_pixel_ap_macro": _finite_mean([v["thin_scratch_pixel_ap"] for v in vals]),
            "cutpaste_pixel_ap_macro": _finite_mean([v["cutpaste_pixel_ap"] for v in vals]),
            "synthetic_pixel_ap_macro": _finite_mean([v["synthetic_pixel_ap"] for v in vals]),
            "normal_clean_score_mean_macro": _finite_mean([v["normal_clean_score_mean"] for v in vals]),
            "normal_clean_score_p95_macro": _finite_mean([v["normal_clean_score_p95"] for v in vals]),
            "normal_clean_score_p99_macro": _finite_mean([v["normal_clean_score_p99"] for v in vals]),
            "category_count": len(vals),
        }
    return {"methods": methods, "per_category": cats, "categories": list(ALL_CATEGORIES)}


def _fusion_deltas(methods: dict[str, Any]) -> dict[str, Any]:
    best_single = max(methods["full896_native64"]["synthetic_pixel_ap_macro"], methods["dino448_bilinear64"]["synthetic_pixel_ap_macro"])
    concat = methods["multiscale_concat64"]["synthetic_pixel_ap_macro"]
    late = methods["late_mean64"]["synthetic_pixel_ap_macro"]
    return {
        "multiscale_concat64_vs_full896": concat - methods["full896_native64"]["synthetic_pixel_ap_macro"],
        "multiscale_concat64_vs_dino448": concat - methods["dino448_bilinear64"]["synthetic_pixel_ap_macro"],
        "late_mean64_vs_full896": late - methods["full896_native64"]["synthetic_pixel_ap_macro"],
        "late_mean64_vs_dino448": late - methods["dino448_bilinear64"]["synthetic_pixel_ap_macro"],
        "multiscale_concat64_vs_late_mean64": concat - late,
        "multiscale_concat64_vs_best_single": concat - best_single,
        "late_mean64_vs_best_single": late - best_single,
    }


def aggregate(ap_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]], legacy_result: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    methods = {arm: _method_summary(ap_rows, normal_rows, arm) for arm in ARMS}
    per_category = {cat: {arm: _category_summary(ap_rows, normal_rows, cat, arm) for arm in ARMS} for cat in CATEGORIES}
    six = _six_summary(per_category, legacy_result)
    return {
        "status": "complete",
        "elapsed_s": float(elapsed_s),
        "n_ap_rows": len(ap_rows),
        "n_normal_rows": len(normal_rows),
        "new_four_category_methods": methods,
        "new_four_category_per_category": per_category,
        "six_category_summary": six,
        "new_four_deltas": _fusion_deltas(methods),
        "six_category_deltas": _fusion_deltas(six["methods"]),
        "legacy_reuse": {
            "result_path": str(LEGACY_RESULT.relative_to(ROOT)),
            "status": legacy_result.get("status"),
            "categories": list(LEGACY_CATEGORIES),
            "read_only": True,
            "old_two_category_results_modified": False,
        },
        "data_role": {
            "support_only": True,
            "real_test_read": False,
            "strict_leave_image_memory": True,
            "query_clean_counterpart_used_for_synthetic": False,
            "new_feature_extraction": False,
            "mask_role": "cached support-derived evaluation labels only",
        },
        "mask_audit": {
            "total_rows": len(mask_rows),
            "total_unique_masks": len({r["mask_sha256"] for r in mask_rows}),
            "all_extension_v14_equal": all(bool(r["extension_v14_mask_equal"]) for r in mask_rows),
            "categories": {cat: len({r["mask_sha256"] for r in mask_rows if r["category"] == cat}) for cat in CATEGORIES},
        },
        "decision": "EXTENSION_COMPLETE_CONCAT_VS_LATE_REPORTED_NO_NOVELTY_CLAIM",
        "decision_note": "The extension reports the direct fixed concat-minus-late difference for four new categories and the six-category reuse summary; neither fixed control is a paper novelty claim.",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _report_cn(protocol: dict[str, Any], result: dict[str, Any], command: str) -> str:
    new = result["new_four_category_methods"]
    six = result["six_category_summary"]["methods"]
    lines = [
        "# 缓存多尺度匹配扩类控制记录（2026-09-08）",
        "",
        "## 范围与数据角色",
        "",
        "本轮只新增 bracket_brown、bracket_white、connector、tubes 四类，复用 full896_extension/features 的 448-edge 32-grid 与 896-edge 64-grid 缓存；没有 GPU、没有新 feature 提取，不改两类原 scale_fusion 结果。",
        "",
        "每个 query 只用同类另一张 k2 support 图建 memory；synthetic query 使用自身双分辨率 cached variant，没有 clean counterpart oracle。48 个 mask 与既有 v14 mask 逐字节核对。normal 只报告 clean score 分布，不当作 FPR。",
        "",
        "## 四类新结果",
        "",
        "| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        x = new[arm]
        lines.append(
            f"| {arm} | {x['thin_scratch_pixel_ap_macro']:.4f} | {x['cutpaste_pixel_ap_macro']:.4f} | "
            f"{x['synthetic_pixel_ap_macro']:.4f} | {x['normal_clean_score_mean_macro']:.6f} | "
            f"{x['normal_clean_score_p95_macro']:.6f} | {x['normal_clean_score_p99_macro']:.6f} |"
        )
    nd = result["new_four_deltas"]
    lines.extend(
        [
            "",
            f"四类直接差值：concat − late = `{nd['multiscale_concat64_vs_late_mean64']:+.6f}`；concat − 最佳单支 = `{nd['multiscale_concat64_vs_best_single']:+.6f}`；late − 最佳单支 = `{nd['late_mean64_vs_best_single']:+.6f}`。",
            "",
            "## 六类汇总",
            "",
            "六类汇总将本轮四类按类别等权纳入，并只读复用原 scale_fusion 的 bracket_black / metal_plate 类别摘要；没有跨脚本重算旧两类。",
            "",
            "| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        x = six[arm]
        lines.append(
            f"| {arm} | {x['thin_scratch_pixel_ap_macro']:.4f} | {x['cutpaste_pixel_ap_macro']:.4f} | "
            f"{x['synthetic_pixel_ap_macro']:.4f} | {x['normal_clean_score_mean_macro']:.6f} | "
            f"{x['normal_clean_score_p95_macro']:.6f} | {x['normal_clean_score_p99_macro']:.6f} |"
        )
    sd = result["six_category_deltas"]
    lines.extend(
        [
            "",
            f"六类直接差值：concat − late = `{sd['multiscale_concat64_vs_late_mean64']:+.6f}`；concat − 最佳单支 = `{sd['multiscale_concat64_vs_best_single']:+.6f}`；late − 最佳单支 = `{sd['late_mean64_vs_best_single']:+.6f}`。",
            "",
            f"结论：`{result['decision']}`。concat 与 late 的差值已直接呈现；结果只用于本轮 support-derived 控制审计，不能写成论文创新。",
            "",
            f"mask audit：rows={result['mask_audit']['total_rows']}，unique masks={result['mask_audit']['total_unique_masks']}，extension/v14 parity={result['mask_audit']['all_extension_v14_equal']}；new AP rows={result['n_ap_rows']}。",
            "",
            f"复现命令：`{command}`",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> int:
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    protocol = _freeze_protocol(EXP_OUT / "PROTOCOL.json")
    (EXP_OUT / "PROTOCOL_CN.md").write_text(
        "# 缓存多尺度匹配扩类冻结协议\n\n"
        "双分辨率 cache index、固定四臂、48 masks、normal clean 分布与六类只读复用规则见 `PROTOCOL.json`。\n\n"
        "本轮不新提取 feature，不改两类原 scale_fusion 结果。\n",
        encoding="utf-8",
    )
    command = ".venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_scale_fusion_extension.py"
    started_utc = _utc_now()
    started = time.perf_counter()
    all_ap: list[dict[str, Any]] = []
    all_normal: list[dict[str, Any]] = []
    all_masks: list[dict[str, Any]] = []
    status = "complete"
    error: str | None = None
    try:
        torch.set_num_threads(THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        legacy_obj = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
        legacy_result = legacy_obj["result"]
        if legacy_result.get("status") != "complete":
            raise RuntimeError(f"legacy scale_fusion result is not complete: {legacy_result.get('status')}")
        if set(legacy_result.get("methods", {})) != set(ARMS):
            raise RuntimeError("legacy scale_fusion arm set mismatch")
        manifest = load_manifest()
        for cat in CATEGORIES:
            _check_deadline(f"before_category_{cat}")
            cache = _load_category_cache(cat, manifest)
            ap_rows, normal_rows, mask_rows = run_category(cat, cache)
            all_ap.extend(ap_rows)
            all_normal.extend(normal_rows)
            all_masks.extend(mask_rows)
            print(f"  done {cat}: AP rows={len(ap_rows)} normal rows={len(normal_rows)}", flush=True)
    except DeadlineReached as exc:
        status = "partial_deadline"
        error = str(exc)
    except Exception as exc:
        status = "error_partial"
        error = f"{type(exc).__name__}: {exc}"
        raise
    elapsed = time.perf_counter() - started
    result = aggregate(all_ap, all_normal, all_masks, legacy_result, elapsed)
    result["status"] = status
    result["error"] = error
    result["started_utc"] = started_utc
    result["finished_utc"] = _utc_now()
    result["protocol_sha256"] = _sha256_file(EXP_OUT / "PROTOCOL.json")
    result["code_sha256"] = _sha256_file(Path(__file__))
    result["reproduction"] = {"command": command, "script": "scripts/innovation_overnight_20260908/probe_scale_fusion_extension.py"}
    _write_json(EXP_OUT / "RESULTS.json", {"protocol": protocol, "result": result})
    _write_csv(EXP_OUT / "PER_EPISODE.csv", all_ap)
    _write_csv(EXP_OUT / "NORMAL_CLEAN.csv", all_normal)
    _write_csv(EXP_OUT / "MASK_AUDIT.csv", all_masks)
    (EXP_OUT / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    (EXP_OUT / "REPORT_CN.md").write_text(_report_cn(protocol, result, command), encoding="utf-8")
    print("\n==== SCALE-FUSION EXTENSION PROBE ====", flush=True)
    for arm in ARMS:
        x = result["new_four_category_methods"][arm]
        print(
            f"{arm}: thin={x['thin_scratch_pixel_ap_macro']:.4f} cut={x['cutpaste_pixel_ap_macro']:.4f} "
            f"mean={x['synthetic_pixel_ap_macro']:.4f} clean_p95={x['normal_clean_score_p95_macro']:.6f}",
            flush=True,
        )
    print(f"concat_minus_late_new={result['new_four_deltas']['multiscale_concat64_vs_late_mean64']:+.6f} "
          f"six={result['six_category_deltas']['multiscale_concat64_vs_late_mean64']:+.6f}", flush=True)
    print(f"status={status} elapsed_s={elapsed:.3f} outputs={EXP_OUT}", flush=True)
    return 0 if status in ("complete", "partial_deadline") else 1


if __name__ == "__main__":
    raise SystemExit(run())
