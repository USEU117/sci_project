"""Bounded cross-input-resolution residual probe.

The overnight resolution probes compared full 896 and full 448 features as
independent nearest-neighbour arms and compared fixed concat/late fusion.  No
history audit found a direct cross-resolution feature residual or a
normal-memory-only low-to-high predictor.  This follow-up therefore tests
that narrow control on the two pre-registered categories bracket_black and
metal_plate.

All feature arrays are read from completed caches.  The high branch is the
existing full896 native 64 x 64 cache and the low branch is the existing v14
DINO 448-edge 32 x 32 cache, bilinearly resized to 64 x 64.  No model is
loaded and no new feature is extracted.

Arms are fixed before reading AP:

* high-only and low-only global L2 nearest-neighbour maps;
* the original fixed late mean of those two maps;
* direct residual-only, ``||L2(high) - L2(low)||``;
* fixed equal-weight high/residual concatenation;
* normal-memory-only ridge low-to-high prediction residual;
* a wrong-pair residual control using the next synthetic seed from the same
  query support image.

The wrong-pair arm is an explicit synthetic control and is not used to choose
any parameter.  Clean rows report normal score distributions only.  Every
synthetic query uses its own high and low cached variant for the main arms;
no clean counterpart oracle is used.
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


EXP_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_followup_20260908"
EXP_OUT = EXP_ROOT / "scale_residual"
HIGH_CACHE_DIR = ROOT / "outputs" / "dynamic_fusion" / "overnight_20260908_full896" / "features"
LOW_CACHE_DIR = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support" / "v14_p1_support_dino_s0_k2"
CATEGORIES = ("bracket_black", "metal_plate")
SHOT = 2
SEED = "0"
RAW_SIZE = 1024
HIGH_GRID = 64
LOW_GRID = 32
FEATURE_DIM = 768
SYN_KINDS = ("thin_scratch", "cutpaste")
SYN_SEEDS = (0, 1, 2)
N_SYN = len(SYN_KINDS) * len(SYN_SEEDS)
RIDGE_ALPHA = 1.0
EQUAL_WEIGHT = 1.0 / np.sqrt(2.0)
ARMS = (
    "high_only64",
    "low_only64",
    "late_mean64",
    "direct_residual64",
    "high_residual_concat64",
    "ridge_residual64",
    "wrong_pair_residual64",
)
MAIN_ARMS = ARMS[:-1]
THREADS = 2
MAP_GAUSSIAN_SIGMA = 4.0
DEADLINE_UTC = datetime(2026, 9, 8, 23, 0, tzinfo=timezone.utc)
SCRIPT_VERSION = "scale_residual_followup_v1"
PROTOCOL_VERSION = "scale_residual_cached64_support_probe_v1"


class DeadlineReached(RuntimeError):
    """The explicit 07:00 Beijing hard deadline was reached."""


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
        "role": "support-only cross-input-resolution residual control; no test image or test label",
        "historical_audit": {
            "resolution_storage_script": "scripts/innovation_overnight_20260908/probe_resolution_storage.py",
            "resolution_storage_finding": "full448/full896 were evaluated as independent NN arms and INT8 roundtrip; no high-low feature residual",
            "scale_fusion_script": "scripts/innovation_overnight_20260908/probe_scale_fusion.py",
            "scale_fusion_finding": "fixed full896/low bilinear/concat/late arms; no direct cross-resolution residual or low-to-high predictor",
            "full896_extension_script": "scripts/innovation_overnight_20260908/probe_full896_extension.py",
            "full896_extension_finding": "dual-resolution clean/synthetic cache export only; no residual arm",
            "related_prior_art": "https://arxiv.org/abs/2602.09524",
            "related_prior_art_finding": "HLGFA (2026, v3) already describes frozen-backbone high-low resolution consistency, structure/detail decomposition, and gated residual",
            "novelty_boundary": "the residual concept is not original; this probe only measures a few-shot normal-only closed-form residual and support-only cost/quality control",
            "audit_conclusion": "direct cross-input-resolution residual route is not a completed historical experiment; this run is a bounded control, not a novelty claim",
        },
        "scope": {
            "dataset": "MPDD",
            "categories": list(CATEGORIES),
            "seed": int(SEED),
            "shot": SHOT,
            "synthetic_families": list(SYN_KINDS),
            "synthetic_seeds": list(SYN_SEEDS),
            "unique_synthetic_masks": 24,
        },
        "cache_index": {
            "high": "outputs/dynamic_fusion/overnight_20260908_full896/features/<category>.npz: clean_feat/syn_feat 64 x 64 x 768",
            "low": "outputs/dynamic_fusion/v14_p1_support/v14_p1_support_dino_s0_k2/<category>.npz: clean_feat/syn_feat 32 x 32 x 768",
            "index_rule": "match manifest ref_rel; high full896 kind-major thin_scratch/cutpaste; low v14 kind-major cutpaste/local_erasure/thin_scratch",
            "mask_rule": "high full896 synthetic mask must equal the corresponding low v14 mask byte-for-byte",
        },
        "fixed_arms": {
            "high_only64": "high feature row-L2 then global L2 NN",
            "low_only64": "low feature bilinear-to-64, row-L2 then global L2 NN",
            "late_mean64": "arithmetic mean of high-only and low-only 64-grid distance maps, then common map post-processing",
            "direct_residual64": "row-L2 high minus row-L2 low; score is residual row L2 norm",
            "high_residual_concat64": "row-L2 high and row-L2 direct residual, each multiplied by 1/sqrt(2), concat, row-L2, global L2 NN",
            "ridge_residual64": "fit low-to-high ridge on the other clean support image only; query score is row L2 of own high minus predicted high",
            "wrong_pair_residual64": "synthetic-only control: current query high minus same-query next-seed low, row-L2 residual norm",
        },
        "fixed_parameters": {
            "high_grid": [HIGH_GRID, HIGH_GRID],
            "low_grid_source": [LOW_GRID, LOW_GRID],
            "feature_dim": FEATURE_DIM,
            "bilinear_mode": "torch.nn.functional.interpolate(mode=bilinear, align_corners=False, size=64 x 64) on CPU",
            "row_l2_floor": 1.0e-8,
            "concat_equal_weight": "1/sqrt(2) per normalized high/residual branch; no weight scan",
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_fit": "fit_intercept=False; X=other-support row-L2 low; Y=other-support row-L2 high; normal memory only",
            "wrong_pair_rule": "same query support image, same family, seed (s + 1) mod 3; fixed before AP",
            "nearest_neighbor": "global over all 4096 memory cells; Euclidean distance of row-L2 descriptors via sqrt(2 - 2 max cosine)",
            "score_map_postprocess": "src.utils.dists2map: cv2 INTER_LINEAR resize to 1024 x 1024 followed by Gaussian sigma=4",
        },
        "data_roles": {
            "memory": "only the other clean support image in the same category for NN and ridge fitting; query image cells excluded",
            "synthetic_query": "main arms use the query's own high and low cached variant; no clean counterpart oracle",
            "normal_control": "own clean query score distribution; not FPR",
            "wrong_pair_exception": "explicit synthetic control only; low branch comes from the same query support image's fixed next-seed variant",
            "test_read": False,
            "new_feature_extraction": False,
            "parameter_fit_from_ap": False,
            "type_routing": False,
        },
        "evaluation": {
            "synthetic": "Pixel-AP on original 1024 x 1024 mask",
            "normal": "mean / p95 / p99 score distribution for clean query after common map post-processing; no clean-vs-nuisance FPR",
            "main_table": False,
            "novelty_boundary": "cross-resolution residual and ridge are bounded controls; no algorithm novelty claim",
        },
        "resources": {"device": "cpu", "threads": THREADS, "gpu_used": False, "new_feature_extraction": False},
        "hard_deadline_beijing": "2026-09-09T07:00:00+08:00",
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
            raise RuntimeError(f"Frozen scale-residual protocol mismatch: {path}")
        return old
    _write_json(path, p)
    return p


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def _episode_index(kinds: list[str], seed_count: int, kind: str, seed: int) -> int:
    return kinds.index(kind) * seed_count + int(seed)


def _load_category_cache(cat: str, manifest: dict[str, Any]) -> dict[str, Any]:
    high = _load_npz(HIGH_CACHE_DIR / f"{cat}.npz")
    low = _load_npz(LOW_CACHE_DIR / f"{cat}.npz")
    rels = support_paths(cat, SHOT, SEED, manifest)
    assert_fit_ids_are_support(rels, cat, SHOT, SEED)
    if rels != high["ref_rel"].astype(str).tolist() or rels != low["ref_rel"].astype(str).tolist():
        raise ValueError(f"support/cache ref_rel mismatch for {cat}: {rels}")
    expected_high = {
        "clean_feat": (SHOT, HIGH_GRID, HIGH_GRID, FEATURE_DIM),
        "syn_feat": (SHOT, N_SYN, HIGH_GRID, HIGH_GRID, FEATURE_DIM),
        "syn_masks": (SHOT, N_SYN, RAW_SIZE, RAW_SIZE),
    }
    expected_low = {
        "clean_feat": (SHOT, LOW_GRID, LOW_GRID, FEATURE_DIM),
        "syn_feat": (SHOT, 9, LOW_GRID, LOW_GRID, FEATURE_DIM),
        "syn_masks": (SHOT, 9, RAW_SIZE, RAW_SIZE),
    }
    for key, shape in expected_high.items():
        if tuple(high[key].shape) != shape:
            raise ValueError(f"unexpected high {key} shape for {cat}: {high[key].shape}")
    for key, shape in expected_low.items():
        if tuple(low[key].shape) != shape:
            raise ValueError(f"unexpected low {key} shape for {cat}: {low[key].shape}")
    high_kinds = high["syn_kinds"].astype(str).tolist()
    high_seeds = high["syn_seeds"].astype(int).tolist()
    low_kinds = low["syn_kinds"].astype(str).tolist()
    low_seed_count = int(np.asarray(low["syn_seeds"]))
    if high_kinds != list(SYN_KINDS) or high_seeds != list(SYN_SEEDS):
        raise ValueError(f"unexpected high synthetic index for {cat}: {high_kinds} / {high_seeds}")
    if low_seed_count != len(SYN_SEEDS) or not all(k in low_kinds for k in SYN_KINDS):
        raise ValueError(f"unexpected low synthetic index for {cat}: {low_kinds} / {low_seed_count}")
    if not np.all(high["clean_valid"] == 1) or not np.all(high["syn_valid"] == 1):
        raise ValueError(f"invalid high cache entries for {cat}")
    return {
        "high": high,
        "low": low,
        "rels": rels,
        "high_kinds": high_kinds,
        "low_kinds": low_kinds,
        "low_seed_count": low_seed_count,
    }


def _audit_masks(cache: dict[str, Any], cat: str) -> list[dict[str, Any]]:
    high = cache["high"]
    low = cache["low"]
    rows: list[dict[str, Any]] = []
    for h, rel in enumerate(cache["rels"]):
        for kind in SYN_KINDS:
            for seed in SYN_SEEDS:
                hi = _episode_index(cache["high_kinds"], len(SYN_SEEDS), kind, seed)
                li = _episode_index(cache["low_kinds"], cache["low_seed_count"], kind, seed)
                mask = high["syn_masks"][h, hi]
                equal = bool(np.array_equal(mask, low["syn_masks"][h, li]))
                if not equal:
                    raise AssertionError(f"high/low mask mismatch: {cat} {rel} {kind} {seed}")
                rows.append(
                    {
                        "category": cat,
                        "query_index": h,
                        "query_rel": rel,
                        "kind": kind,
                        "seed": seed,
                        "high_episode": hi,
                        "low_episode": li,
                        "mask_sha256": _sha256_array(mask),
                        "mask_positive_pixels": int((mask > 0).sum()),
                        "mask_area": float((mask > 0).mean()),
                        "high_low_mask_equal": equal,
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


def _postprocess(score_grid: np.ndarray) -> np.ndarray:
    if score_grid.shape != (HIGH_GRID, HIGH_GRID):
        raise ValueError(f"expected 64 x 64 score grid, got {score_grid.shape}")
    return np.asarray(dists2map(score_grid.astype(np.float32), (RAW_SIZE, RAW_SIZE)), dtype=np.float32)


def _nearest_l2_grid(query_n: np.ndarray, bank_n: np.ndarray) -> np.ndarray:
    sim = query_n @ bank_n.T
    best = sim.max(axis=1)
    return np.sqrt(np.maximum(2.0 - 2.0 * best, 0.0)).astype(np.float32).reshape(HIGH_GRID, HIGH_GRID)


def _fit_ridge(low_bank_n: np.ndarray, high_bank_n: np.ndarray) -> np.ndarray:
    x = np.asarray(low_bank_n, dtype=np.float64)
    y = np.asarray(high_bank_n, dtype=np.float64)
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += RIDGE_ALPHA
    rhs = x.T @ y
    return np.linalg.solve(gram, rhs).astype(np.float32)


def _build_bank(high: np.ndarray, low32: np.ndarray) -> dict[str, np.ndarray]:
    high_n = _l2_rows(high)
    low_n = _l2_rows(_bilinear_to64(low32))
    direct_residual_n = _l2_rows(high_n - low_n)
    concat_n = _l2_rows(np.concatenate((EQUAL_WEIGHT * high_n, EQUAL_WEIGHT * direct_residual_n), axis=1))
    ridge_w = _fit_ridge(low_n, high_n)
    return {
        "high": high_n,
        "low": low_n,
        "direct_residual": direct_residual_n,
        "concat": concat_n,
        "ridge_w": ridge_w,
    }


def _score_maps(query_high: np.ndarray, query_low32: np.ndarray, bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    high_n = _l2_rows(query_high)
    low_n = _l2_rows(_bilinear_to64(query_low32))
    high_grid = _nearest_l2_grid(high_n, bank["high"])
    low_grid = _nearest_l2_grid(low_n, bank["low"])
    direct = high_n - low_n
    direct_grid = np.linalg.norm(direct, axis=1).astype(np.float32).reshape(HIGH_GRID, HIGH_GRID)
    direct_n = _l2_rows(direct)
    concat = np.concatenate((EQUAL_WEIGHT * high_n, EQUAL_WEIGHT * direct_n), axis=1)
    concat_grid = _nearest_l2_grid(_l2_rows(concat), bank["concat"])
    pred_high = low_n @ bank["ridge_w"]
    ridge_grid = np.linalg.norm(high_n - pred_high, axis=1).astype(np.float32).reshape(HIGH_GRID, HIGH_GRID)
    late_grid = (high_grid + low_grid) * np.float32(0.5)
    return {
        "high_only64": _postprocess(high_grid),
        "low_only64": _postprocess(low_grid),
        "late_mean64": _postprocess(late_grid),
        "direct_residual64": _postprocess(direct_grid),
        "high_residual_concat64": _postprocess(concat_grid),
        "ridge_residual64": _postprocess(ridge_grid),
    }


def _wrong_pair_map(query_high: np.ndarray, query_low_wrong32: np.ndarray) -> np.ndarray:
    high_n = _l2_rows(query_high)
    low_n = _l2_rows(_bilinear_to64(query_low_wrong32))
    score = np.linalg.norm(high_n - low_n, axis=1).astype(np.float32).reshape(HIGH_GRID, HIGH_GRID)
    return _postprocess(score)


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
    high = cache["high"]
    low = cache["low"]
    ap_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    mask_rows = _audit_masks(cache, cat)
    for h, query_rel in enumerate(cache["rels"]):
        _check_deadline(f"before_{cat}_h{h}")
        bank_index = 1 - h
        bank = _build_bank(high["clean_feat"][bank_index], low["clean_feat"][bank_index])
        for kind in SYN_KINDS:
            for seed in SYN_SEEDS:
                _check_deadline(f"before_synthetic_{cat}_h{h}_{kind}_{seed}")
                hi = _episode_index(cache["high_kinds"], len(SYN_SEEDS), kind, seed)
                li = _episode_index(cache["low_kinds"], cache["low_seed_count"], kind, seed)
                maps = _score_maps(high["syn_feat"][h, hi], low["syn_feat"][h, li], bank)
                wrong_seed = (seed + 1) % len(SYN_SEEDS)
                wrong_li = _episode_index(cache["low_kinds"], cache["low_seed_count"], kind, wrong_seed)
                maps["wrong_pair_residual64"] = _wrong_pair_map(high["syn_feat"][h, hi], low["syn_feat"][h, wrong_li])
                mask = high["syn_masks"][h, hi]
                for arm in ARMS:
                    ap_rows.append(
                        {
                            "category": cat,
                            "query_index": h,
                            "query_rel": query_rel,
                            "bank_index": bank_index,
                            "bank_rel": cache["rels"][bank_index],
                            "strict_leave_image_memory": True,
                            "uses_other_support_memory": arm != "direct_residual64" and arm != "wrong_pair_residual64",
                            "query_clean_counterpart_used": False,
                            "kind": kind,
                            "seed": seed,
                            "wrong_pair_seed": wrong_seed if arm == "wrong_pair_residual64" else None,
                            "high_episode": hi,
                            "low_episode": li,
                            "arm": arm,
                            "grid": HIGH_GRID,
                            "mask_sha256": _sha256_array(mask),
                            "mask_positive_pixels": int((mask > 0).sum()),
                            "mask_area": float((mask > 0).mean()),
                            "mask_cache_verified": True,
                            "pixel_ap": _ap(mask, maps[arm]),
                            "feature_source": "cached full896 high + cached v14 DINO448 low",
                            "normalization": "row-L2 feature branches; direct/ridge residual scored by row L2 norm",
                            "ridge_alpha": RIDGE_ALPHA if arm == "ridge_residual64" else None,
                            "postprocess": "dists2map bilinear + Gaussian sigma4",
                        }
                    )
        _check_deadline(f"before_clean_{cat}_h{h}")
        clean_maps = _score_maps(high["clean_feat"][h], low["clean_feat"][h], bank)
        for arm in MAIN_ARMS:
            normal_rows.append(
                {
                    "category": cat,
                    "query_index": h,
                    "query_rel": query_rel,
                    "bank_index": bank_index,
                    "bank_rel": cache["rels"][bank_index],
                    "strict_leave_image_memory": True,
                    "uses_other_support_memory": arm != "direct_residual64",
                    "query_clean_counterpart_used": False,
                    "arm": arm,
                    "grid": HIGH_GRID,
                    "feature_source": "cached full896 high + cached v14 DINO448 low",
                    "normal_role": "clean score distribution, not FPR",
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


def _deltas(methods: dict[str, Any]) -> dict[str, float]:
    return {
        "direct_residual64_vs_high_only64": methods["direct_residual64"]["synthetic_pixel_ap_macro"] - methods["high_only64"]["synthetic_pixel_ap_macro"],
        "direct_residual64_vs_late_mean64": methods["direct_residual64"]["synthetic_pixel_ap_macro"] - methods["late_mean64"]["synthetic_pixel_ap_macro"],
        "ridge_residual64_vs_high_only64": methods["ridge_residual64"]["synthetic_pixel_ap_macro"] - methods["high_only64"]["synthetic_pixel_ap_macro"],
        "ridge_residual64_vs_late_mean64": methods["ridge_residual64"]["synthetic_pixel_ap_macro"] - methods["late_mean64"]["synthetic_pixel_ap_macro"],
        "high_residual_concat64_vs_high_only64": methods["high_residual_concat64"]["synthetic_pixel_ap_macro"] - methods["high_only64"]["synthetic_pixel_ap_macro"],
        "high_residual_concat64_vs_late_mean64": methods["high_residual_concat64"]["synthetic_pixel_ap_macro"] - methods["late_mean64"]["synthetic_pixel_ap_macro"],
        "high_residual_concat64_vs_direct_residual64": methods["high_residual_concat64"]["synthetic_pixel_ap_macro"] - methods["direct_residual64"]["synthetic_pixel_ap_macro"],
        "wrong_pair_residual64_vs_direct_residual64": methods["wrong_pair_residual64"]["synthetic_pixel_ap_macro"] - methods["direct_residual64"]["synthetic_pixel_ap_macro"],
    }


def aggregate(ap_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    methods = {arm: _method_summary(ap_rows, normal_rows, arm) for arm in ARMS}
    per_category = {cat: {arm: _category_summary(ap_rows, normal_rows, cat, arm) for arm in ARMS} for cat in CATEGORIES}
    deltas = _deltas(methods)
    best_baseline = max(methods["high_only64"]["synthetic_pixel_ap_macro"], methods["low_only64"]["synthetic_pixel_ap_macro"], methods["late_mean64"]["synthetic_pixel_ap_macro"])
    residual_best = max(methods["direct_residual64"]["synthetic_pixel_ap_macro"], methods["ridge_residual64"]["synthetic_pixel_ap_macro"], methods["high_residual_concat64"]["synthetic_pixel_ap_macro"])
    decision = "RESIDUAL_ARM_SUPPORT_SIGNAL_ONLY" if residual_best > best_baseline else "NO_RESIDUAL_GAIN_OVER_FIXED_BASELINES"
    return {
        "status": "complete",
        "elapsed_s": float(elapsed_s),
        "n_ap_rows": len(ap_rows),
        "n_normal_rows": len(normal_rows),
        "methods": methods,
        "per_category": per_category,
        "deltas": deltas,
        "baseline_best_synthetic_ap": best_baseline,
        "residual_best_synthetic_ap": residual_best,
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
            "all_high_low_equal": all(bool(r["high_low_mask_equal"]) for r in mask_rows),
            "categories": {cat: len({r["mask_sha256"] for r in mask_rows if r["category"] == cat}) for cat in CATEGORIES},
        },
        "decision": decision,
        "decision_note": "Direct and ridge residuals are bounded controls. Any positive result is support-derived and cannot be called a new algorithm; the wrong-pair arm is a fixed correspondence sanity control.",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _report_cn(protocol: dict[str, Any], result: dict[str, Any], command: str) -> str:
    m = result["methods"]
    d = result["deltas"]
    lines = [
        "# 跨输入分辨率残差控制探针记录（2026-09-08）",
        "",
        "## 历史审计与路线",
        "",
        "历史 `probe_resolution_storage.py` 只将 full448 / full896 作为独立 NN 臂比较；既有 `scale_fusion` 只做 high、low、固定 concat 与 late mean。没有完成跨输入分辨率 feature residual 或 low→high normal-memory ridge。故本轮锁定为有界控制，不声称方法新颖。",
        "相关先例：HLGFA（2026, v3，https://arxiv.org/abs/2602.09524）已涉及冻结 backbone 的高低分辨率一致性、structure/detail 分解与 gated residual；本 probe 只检验 few-shot normal-only 闭式 residual 的支持集质量/成本，不主张 residual 概念原创。",
        "",
        "## 协议与数据角色",
        "",
        f"- 协议：`{protocol['protocol_version']}`；bracket_black / metal_plate，seed0 × k2，CPU threads={THREADS}，无 GPU。",
        "- 复用 full896 64-grid 与 v14 DINO448 32-grid cache；低分辨率固定 bilinear 到 64-grid；未下载、未提取新 feature。",
        "- 每个主 synthetic query 的 high / low 都来自自身 cached variant；memory 仅为同类另一张 clean support 图，未使用 clean counterpart oracle。",
        "- 24 个原始 1024 mask 全部 high/low cache 逐字节一致；normal 只报 clean score 分布，不当 FPR。",
        "- ridge 固定 `alpha=1.0`、无 intercept，仅用另一张 clean support 图拟合；wrong-pair 固定使用同一 query 图的下一个 seed，未按 AP 调整。",
        "",
        "## 首轮结果（support-derived）",
        "",
        "| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        x = m[arm]
        lines.append(
            f"| {arm} | {x['thin_scratch_pixel_ap_macro']:.4f} | {x['cutpaste_pixel_ap_macro']:.4f} | "
            f"{x['synthetic_pixel_ap_macro']:.4f} | {x['normal_clean_score_mean_macro']:.6f} | "
            f"{x['normal_clean_score_p95_macro']:.6f} | {x['normal_clean_score_p99_macro']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"关键差值：direct residual − late = `{d['direct_residual64_vs_late_mean64']:+.6f}`；ridge residual − late = `{d['ridge_residual64_vs_late_mean64']:+.6f}`；high/residual concat − late = `{d['high_residual_concat64_vs_late_mean64']:+.6f}`；wrong-pair residual − direct residual = `{d['wrong_pair_residual64_vs_direct_residual64']:+.6f}`。",
            "",
            f"结论：`{result['decision']}`。{result['decision_note']}",
            "",
            f"mask audit：rows={result['mask_audit']['total_rows']}，unique masks={result['mask_audit']['total_unique_masks']}，high/low parity={result['mask_audit']['all_high_low_equal']}；AP rows={result['n_ap_rows']}。",
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
        "# 跨输入分辨率残差控制冻结协议\n\n"
        "历史审计、cache index、固定残差臂、ridge alpha、wrong-pair 规则与同一 map 后处理见 `PROTOCOL.json`。\n\n"
        "本轮只做 support-derived 控制，不做新颖性主张。\n",
        encoding="utf-8",
    )
    command = ".venv-anomalyclip/Scripts/python.exe scripts/innovation_followup_20260908/probe_scale_residual.py"
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
    result = aggregate(all_ap, all_normal, all_masks, elapsed)
    result["status"] = status
    result["error"] = error
    result["started_utc"] = started_utc
    result["finished_utc"] = _utc_now()
    result["protocol_sha256"] = _sha256_file(EXP_OUT / "PROTOCOL.json")
    result["code_sha256"] = _sha256_file(Path(__file__))
    result["reproduction"] = {"command": command, "script": "scripts/innovation_followup_20260908/probe_scale_residual.py"}
    _write_json(EXP_OUT / "RESULTS.json", {"protocol": protocol, "result": result})
    _write_csv(EXP_OUT / "PER_EPISODE.csv", all_ap)
    _write_csv(EXP_OUT / "NORMAL_PER_IMAGE.csv", all_normal)
    _write_csv(EXP_OUT / "MASK_AUDIT.csv", all_masks)
    (EXP_OUT / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    (EXP_OUT / "REPORT_CN.md").write_text(_report_cn(protocol, result, command), encoding="utf-8")
    print("\n==== SCALE-RESIDUAL PROBE ====", flush=True)
    for arm in ARMS:
        x = result["methods"][arm]
        print(
            f"{arm}: thin={x['thin_scratch_pixel_ap_macro']:.4f} cut={x['cutpaste_pixel_ap_macro']:.4f} "
            f"mean={x['synthetic_pixel_ap_macro']:.4f} clean_p95={x['normal_clean_score_p95_macro']:.6f}",
            flush=True,
        )
    print(
        f"direct_minus_late={result['deltas']['direct_residual64_vs_late_mean64']:+.6f} "
        f"ridge_minus_late={result['deltas']['ridge_residual64_vs_late_mean64']:+.6f} "
        f"concat_minus_late={result['deltas']['high_residual_concat64_vs_late_mean64']:+.6f}",
        flush=True,
    )
    print(f"decision={result['decision']} status={status} elapsed_s={elapsed:.3f} outputs={EXP_OUT}", flush=True)
    return 0 if status in ("complete", "partial_deadline") else 1


if __name__ == "__main__":
    raise SystemExit(run())
