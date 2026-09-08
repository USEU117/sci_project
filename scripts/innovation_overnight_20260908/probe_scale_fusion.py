"""Bounded cached multi-scale matching control for the 2026-09-08 run.

This probe reuses two completed feature exports and does not download a model
or extract a new feature.  ``full896`` is the native 64 x 64 feature grid
from a full 1024-pixel image at the existing DINO 896-edge export.  The v14
support cache is the corresponding frozen DINO 448-edge 32 x 32 grid.  The
only scale operation here is a fixed bilinear feature resize from 32 x 32 to
64 x 64.

Four arms are reported together on the same 24 support-derived masks:

* ``full896_native64``: high-resolution single-scale descriptor;
* ``dino448_bilinear64``: low-resolution descriptor resized to 64 x 64;
* ``multiscale_concat64``: row-L2 each scale, equal-weight concatenation,
  then row-L2 nearest-neighbour distance;
* ``late_mean64``: arithmetic mean of the two single-scale 64-grid score
  maps, as a fixed late-fusion control.

Each query uses the other support image in its category as memory.  Synthetic
queries use their own cached variant at each scale; a clean counterpart is
never used to align, fit, or score a synthetic query.  Clean support queries
are recorded only as normal score-distribution controls.  No test image or
test label is read, and no parameter is selected from AP.

Run from the repository root with::

    .venv-anomalyclip\\Scripts\\python.exe \\
        scripts\\innovation_overnight_20260908\\probe_scale_fusion.py
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

# Keep interpolation and matrix products on the requested small CPU budget.
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


EXP_OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "scale_fusion"
FULL896_FEATURES = ROOT / "outputs" / "dynamic_fusion" / "overnight_20260908_full896" / "features"
V14_CACHE = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support" / "v14_p1_support_dino_s0_k2"
FULL896_PROTOCOL = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "full896" / "PROTOCOL.json"
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
SCRIPT_VERSION = "scale_fusion_probe_v1"
PROTOCOL_VERSION = "scale_fusion_cached64_support_probe_v1"


class DeadlineReached(RuntimeError):
    """Hard Beijing deadline reached; partial rows remain inspectable."""


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
    """Return all choices fixed before any AP or score summary is read."""
    return {
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": _utc_now(),
        "role": "support-only cached multi-scale matching control; no test image or test label",
        "scope": {
            "dataset": "MPDD",
            "categories": list(CATEGORIES),
            "seed": int(SEED),
            "shot": SHOT,
            "support_selection": "manifest seed0 k2; each query support image held out from memory",
            "synthetic_families": list(SYN_KINDS),
            "synthetic_seeds": list(SYN_SEEDS),
            "unique_synthetic_masks": 24,
        },
        "cache_index_audit": {
            "full896_protocol": "experiments/dynamic_fusion/innovation_overnight_20260908/full896/PROTOCOL.json",
            "full896_feature_dir": "outputs/dynamic_fusion/overnight_20260908_full896/features",
            "full896_role": "existing DINO 896-edge full image; clean/synthetic native 64 x 64 x 768",
            "v14_cache_dir": "outputs/dynamic_fusion/v14_p1_support/v14_p1_support_dino_s0_k2",
            "v14_role": "existing DINO 448-edge support cache; clean/synthetic 32 x 32 x 768",
            "index_rule": "match ref_rel; full896 kind-major episode index; v14 kind-major episode index; no positional assumption across exports",
            "mask_rule": "full896 1024 mask is canonical and must equal the corresponding v14 mask byte-for-byte",
        },
        "feature_arms": {
            "full896_native64": "full896 cached native 64 x 64 feature; row-L2 then global L2 NN",
            "dino448_bilinear64": "v14 cached 32 x 32 feature; fixed bilinear feature interpolation to 64 x 64, row-L2 then global L2 NN",
            "multiscale_concat64": "row-L2 each 64-grid scale, equal weights 1/sqrt(2), concatenate, final row-L2, global L2 NN",
            "late_mean64": "arithmetic mean of the two single-scale 64-grid L2-NN score grids before shared map post-processing",
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
            "synthetic_query": "own cached full896 and v14 feature variant; no clean counterpart or alignment oracle",
            "normal_control": "own cached clean query feature; clean score distribution only",
            "test_read": False,
            "parameter_fit_from_ap": False,
            "type_routing": False,
        },
        "evaluation": {
            "synthetic": "Pixel-AP on the original 1024 x 1024 renderer mask",
            "normal": "clean support score map distribution after common post-processing; no nuisance extraction",
            "comparison": "all four arms shown in one probe output; concat and late are fixed controls, not a paper method",
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
            raise RuntimeError(f"Frozen scale-fusion protocol mismatch: {path}")
        return old
    _write_json(path, p)
    return p


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def _load_category_cache(cat: str, manifest: dict[str, Any]) -> dict[str, Any]:
    full = _load_npz(FULL896_FEATURES / f"{cat}.npz")
    low = _load_npz(V14_CACHE / f"{cat}.npz")
    rels = support_paths(cat, SHOT, SEED, manifest)
    assert_fit_ids_are_support(rels, cat, SHOT, SEED)
    if rels != full["ref_rel"].astype(str).tolist() or rels != low["ref_rel"].astype(str).tolist():
        raise ValueError(f"support/cache index mismatch for {cat}: {rels}")
    if tuple(full["clean_feat"].shape) != (SHOT, HIGH_GRID, HIGH_GRID, FEATURE_DIM):
        raise ValueError(f"unexpected full896 clean shape for {cat}: {full['clean_feat'].shape}")
    if tuple(full["syn_feat"].shape) != (SHOT, N_SYN, HIGH_GRID, HIGH_GRID, FEATURE_DIM):
        raise ValueError(f"unexpected full896 synthetic shape for {cat}: {full['syn_feat'].shape}")
    if tuple(full["syn_masks"].shape) != (SHOT, N_SYN, RAW_SIZE, RAW_SIZE):
        raise ValueError(f"unexpected full896 mask shape for {cat}: {full['syn_masks'].shape}")
    if tuple(low["clean_feat"].shape) != (SHOT, LOW_GRID, LOW_GRID, FEATURE_DIM):
        raise ValueError(f"unexpected v14 clean shape for {cat}: {low['clean_feat'].shape}")
    if tuple(low["syn_feat"].shape)[:2] != (SHOT, 9) or tuple(low["syn_feat"].shape[2:]) != (LOW_GRID, LOW_GRID, FEATURE_DIM):
        raise ValueError(f"unexpected v14 synthetic shape for {cat}: {low['syn_feat'].shape}")
    full_kinds = full["syn_kinds"].astype(str).tolist()
    full_seeds = full["syn_seeds"].astype(int).tolist()
    low_kinds = low["syn_kinds"].astype(str).tolist()
    low_seeds = int(np.asarray(low["syn_seeds"]))
    if full_kinds != list(SYN_KINDS) or full_seeds != list(SYN_SEEDS):
        raise ValueError(f"unexpected full896 episode index for {cat}: {full_kinds} / {full_seeds}")
    if low_seeds != len(SYN_SEEDS) or not all(k in low_kinds for k in SYN_KINDS):
        raise ValueError(f"unexpected v14 episode index for {cat}: {low_kinds} / {low_seeds}")
    if not np.all(full["clean_valid"] == 1) or not np.all(full["syn_valid"] == 1):
        raise ValueError(f"full896 cache has invalid entries for {cat}")
    return {"full": full, "low": low, "rels": rels, "full_kinds": full_kinds, "low_kinds": low_kinds, "low_seeds": low_seeds}


def _episode_index(kinds: list[str], seed_count: int, kind: str, seed: int) -> int:
    return kinds.index(kind) * seed_count + int(seed)


def _audit_masks(cache: dict[str, Any], cat: str) -> list[dict[str, Any]]:
    full = cache["full"]
    low = cache["low"]
    rows: list[dict[str, Any]] = []
    for h, rel in enumerate(cache["rels"]):
        for kind in SYN_KINDS:
            for seed in SYN_SEEDS:
                fi = _episode_index(cache["full_kinds"], len(SYN_SEEDS), kind, seed)
                li = _episode_index(cache["low_kinds"], cache["low_seeds"], kind, seed)
                mask = full["syn_masks"][h, fi]
                equal = bool(np.array_equal(mask, low["syn_masks"][h, li]))
                if not equal:
                    raise AssertionError(f"full896/v14 mask mismatch: {cat} {rel} {kind} {seed}")
                rows.append(
                    {
                        "category": cat,
                        "query_index": h,
                        "query_rel": rel,
                        "kind": kind,
                        "seed": seed,
                        "full896_episode": fi,
                        "v14_episode": li,
                        "mask_sha256": _sha256_array(mask),
                        "mask_positive_pixels": int((mask > 0).sum()),
                        "mask_area": float((mask > 0).mean()),
                        "full896_v14_mask_equal": equal,
                    }
                )
    return rows


def _bilinear_to64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[:2] != (LOW_GRID, LOW_GRID) or x.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected 32 x 32 x 768 feature, got {x.shape}")
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        y = torch.nn.functional.interpolate(t, size=(HIGH_GRID, HIGH_GRID), mode="bilinear", align_corners=False)
    return y[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32, copy=False)


def _l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1.0e-8)


def _build_bank(high: np.ndarray, low: np.ndarray) -> dict[str, np.ndarray]:
    high_n = _l2_rows(high)
    low_n = _l2_rows(low)
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


def _score_maps(query_high: np.ndarray, query_low: np.ndarray, bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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
    full = cache["full"]
    low = cache["low"]
    ap_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    mask_rows = _audit_masks(cache, cat)
    for h, query_rel in enumerate(cache["rels"]):
        _check_deadline(f"before_{cat}_h{h}")
        bank_index = 1 - h
        bank_high = full["clean_feat"][bank_index]
        bank_low = _bilinear_to64(low["clean_feat"][bank_index])
        bank = _build_bank(bank_high, bank_low)

        # Synthetic scores use the variant's own cached feature at both scales.
        for kind in SYN_KINDS:
            for seed in SYN_SEEDS:
                _check_deadline(f"before_synthetic_{cat}_h{h}_{kind}_{seed}")
                fi = _episode_index(cache["full_kinds"], len(SYN_SEEDS), kind, seed)
                li = _episode_index(cache["low_kinds"], cache["low_seeds"], kind, seed)
                query_high = full["syn_feat"][h, fi]
                query_low = _bilinear_to64(low["syn_feat"][h, li])
                maps = _score_maps(query_high, query_low, bank)
                mask = full["syn_masks"][h, fi]
                mask_hash = _sha256_array(mask)
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
                            "full896_episode": fi,
                            "v14_episode": li,
                            "arm": arm,
                            "grid": HIGH_GRID,
                            "mask_sha256": mask_hash,
                            "mask_positive_pixels": int((mask > 0).sum()),
                            "mask_area": float((mask > 0).mean()),
                            "mask_cache_verified": True,
                            "pixel_ap": _ap(mask, maps[arm]),
                            "feature_source": "cached full896 + cached v14 DINO448",
                            "normalization": "row-L2 only; fixed equal concat weight where applicable",
                            "postprocess": "dists2map bilinear + Gaussian sigma4",
                        }
                    )

        # Clean is a normal score-distribution control only.
        _check_deadline(f"before_clean_{cat}_h{h}")
        clean_high = full["clean_feat"][h]
        clean_low = _bilinear_to64(low["clean_feat"][h])
        clean_maps = _score_maps(clean_high, clean_low, bank)
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
                    "feature_source": "cached full896 clean + cached v14 DINO448 clean",
                    **_score_stats(clean_maps[arm]),
                }
            )
    return ap_rows, normal_rows, mask_rows


def _finite_mean(values: list[float]) -> float:
    x = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def aggregate(ap_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "complete",
        "elapsed_s": float(elapsed_s),
        "n_ap_rows": len(ap_rows),
        "n_normal_rows": len(normal_rows),
        "methods": {},
        "per_category": {},
        "data_role": {
            "support_only": True,
            "real_test_read": False,
            "strict_leave_image_memory": True,
            "query_clean_counterpart_used_for_synthetic": False,
            "new_feature_extraction": False,
            "mask_role": "cached support-derived evaluation labels only",
        },
    }
    for arm in ARMS:
        rr = [r for r in ap_rows if r["arm"] == arm]
        nn = [r for r in normal_rows if r["arm"] == arm]
        out["methods"][arm] = {
            "thin_scratch_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "thin_scratch"]),
            "cutpaste_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "cutpaste"]),
            "synthetic_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rr]),
            "normal_clean_score_mean_macro": _finite_mean([r["score_mean"] for r in nn]),
            "normal_clean_score_p95_macro": _finite_mean([r["score_p95"] for r in nn]),
            "normal_clean_score_p99_macro": _finite_mean([r["score_p99"] for r in nn]),
            "n_synthetic_rows": len(rr),
            "n_normal_rows": len(nn),
        }
    for cat in CATEGORIES:
        out["per_category"][cat] = {}
        for arm in ARMS:
            rr = [r for r in ap_rows if r["category"] == cat and r["arm"] == arm]
            nn = [r for r in normal_rows if r["category"] == cat and r["arm"] == arm]
            out["per_category"][cat][arm] = {
                "thin_scratch_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "thin_scratch"]),
                "cutpaste_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr if r["kind"] == "cutpaste"]),
                "synthetic_pixel_ap": _finite_mean([r["pixel_ap"] for r in rr]),
                "normal_clean_score_mean": _finite_mean([r["score_mean"] for r in nn]),
                "normal_clean_score_p95": _finite_mean([r["score_p95"] for r in nn]),
                "normal_clean_score_p99": _finite_mean([r["score_p99"] for r in nn]),
            }
    best_single = max(out["methods"]["full896_native64"]["synthetic_pixel_ap_macro"], out["methods"]["dino448_bilinear64"]["synthetic_pixel_ap_macro"])
    concat_ap = out["methods"]["multiscale_concat64"]["synthetic_pixel_ap_macro"]
    late_ap = out["methods"]["late_mean64"]["synthetic_pixel_ap_macro"]
    out["deltas_vs_single"] = {
        "multiscale_concat64": {
            "vs_full896_native64": concat_ap - out["methods"]["full896_native64"]["synthetic_pixel_ap_macro"],
            "vs_dino448_bilinear64": concat_ap - out["methods"]["dino448_bilinear64"]["synthetic_pixel_ap_macro"],
            "vs_best_single": concat_ap - best_single,
        },
        "late_mean64": {
            "vs_full896_native64": late_ap - out["methods"]["full896_native64"]["synthetic_pixel_ap_macro"],
            "vs_dino448_bilinear64": late_ap - out["methods"]["dino448_bilinear64"]["synthetic_pixel_ap_macro"],
            "vs_best_single": late_ap - best_single,
        },
    }
    out["mask_audit"] = {
        "total_rows": len(mask_rows),
        "total_unique_masks": len({r["mask_sha256"] for r in mask_rows}),
        "all_full896_v14_equal": all(bool(r["full896_v14_mask_equal"]) for r in mask_rows),
        "categories": {cat: len({r["mask_sha256"] for r in mask_rows if r["category"] == cat}) for cat in CATEGORIES},
    }
    if concat_ap > best_single or late_ap > best_single:
        out["decision"] = "FIXED_MULTISCALE_CANDIDATE_ONLY"
        out["decision_note"] = "One fixed two-scale control exceeds the best single arm in this support-synthetic probe; this is not a routing or paper novelty claim."
    else:
        out["decision"] = "NO_FIXED_MULTISCALE_GAIN_IN_THIS_PROBE"
        out["decision_note"] = "Neither fixed two-scale control exceeds the best single arm in this support-synthetic probe; simple multi-scale matching is retained only as a negative control."
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
    methods = result["methods"]
    lines = [
        "# 缓存多尺度匹配控制探针记录（2026-09-08）",
        "",
        "## 目的与边界",
        "",
        "本 probe 只复用既有 full896 clean/synthetic feature 与 v14 DINO448 clean/synthetic feature；没有下载模型、没有新特征提取。它检验 context/detail 两尺度联合匹配能否在固定配置下减少单尺度取舍。多尺度拼接与 late mean 都是已知的工程控制，不能称论文新颖性。",
        "",
        "## 冻结协议与数据角色",
        "",
        f"- 协议：`{protocol['protocol_version']}`；seed0 × k2；CPU threads={THREADS}；GPU=False。",
        "- full896 是缓存的 64 × 64 × 768 高分支；v14 DINO448 是缓存的 32 × 32 × 768 低分支，低分支只做固定 bilinear → 64 × 64。",
        "- 每个 query 只用同类另一张 support 图建 memory；synthetic query 使用自身 cached variant，未使用 clean 对应图或对齐 oracle。",
        "- 24 个 mask 由 full896 cache 作为标签，并与 v14 cache 逐字节核对；没有读取 `/test/`。",
        "- single、concat、late 四个 arm 同时计算；concat 固定两支等权后 L2，late 固定两张 64-grid score map 均值，没有参数扫描、type routing 或按既有 tiling 结果选参。",
        "- normal 只报告 clean score 分布（mean / p95 / p99），不额外提取 nuisance feature。",
        "",
        "## 首轮结果（support-derived）",
        "",
        "| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        x = methods[arm]
        lines.append(
            f"| {arm} | {x['thin_scratch_pixel_ap_macro']:.4f} | {x['cutpaste_pixel_ap_macro']:.4f} | "
            f"{x['synthetic_pixel_ap_macro']:.4f} | {x['normal_clean_score_mean_macro']:.6f} | "
            f"{x['normal_clean_score_p95_macro']:.6f} | {x['normal_clean_score_p99_macro']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"结论：`{result['decision']}`。{result['decision_note']}",
            "",
            f"mask audit：rows={result['mask_audit']['total_rows']}，unique masks={result['mask_audit']['total_unique_masks']}，full896/v14 parity={result['mask_audit']['all_full896_v14_equal']}；AP rows={result['n_ap_rows']}。",
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
        "# 缓存多尺度匹配控制冻结协议\n\n"
        "固定字段、cache index、四个 arm、数据角色和后处理见 `PROTOCOL.json`。\n\n"
        "本 probe 不提取新 feature，不做论文创新主张。\n",
        encoding="utf-8",
    )
    command = ".venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_scale_fusion.py"
    started_utc = _utc_now()
    started = time.perf_counter()
    all_ap: list[dict[str, Any]] = []
    all_normal: list[dict[str, Any]] = []
    all_masks: list[dict[str, Any]] = []
    status = "complete"
    error: str | None = None
    try:
        # Set these in the worker process even when this script is imported by a caller.
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
    result["reproduction"] = {"command": command, "script": "scripts/innovation_overnight_20260908/probe_scale_fusion.py"}
    _write_json(EXP_OUT / "RESULTS.json", {"protocol": protocol, "result": result})
    _write_csv(EXP_OUT / "PER_EPISODE.csv", all_ap)
    _write_csv(EXP_OUT / "NORMAL_CLEAN.csv", all_normal)
    _write_csv(EXP_OUT / "MASK_AUDIT.csv", all_masks)
    (EXP_OUT / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    (EXP_OUT / "REPORT_CN.md").write_text(_report_cn(protocol, result, command), encoding="utf-8")
    print("\n==== SCALE-FUSION PROBE ====", flush=True)
    for arm in ARMS:
        x = result["methods"][arm]
        print(
            f"{arm}: thin={x['thin_scratch_pixel_ap_macro']:.4f} "
            f"cut={x['cutpaste_pixel_ap_macro']:.4f} "
            f"mean={x['synthetic_pixel_ap_macro']:.4f} "
            f"clean_p95={x['normal_clean_score_p95_macro']:.6f}",
            flush=True,
        )
    print(f"decision={result['decision']} status={status} elapsed_s={elapsed:.3f} outputs={EXP_OUT}", flush=True)
    return 0 if status in ("complete", "partial_deadline") else 1


if __name__ == "__main__":
    raise SystemExit(run())
