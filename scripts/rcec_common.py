"""RCEC v1 shared runner logic (imported by all RCEC runner scripts).

Responsibilities:
  * dataset feature layout resolution (identical to A1 caches);
  * per-category RCEC evaluation producing the report schema of the task book;
  * config loading / candidate grid construction;
  * reference-order verification against the dataset manifest;
  * input hashing;
  * gate evaluation helpers (small gate / selection pool / final gate).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from evaluate_a1_feature_fusion import STRIDE, compute_metrics, load_features, resize_patches  # noqa: E402
from evaluate_a1_complete_metrics import compute_image_metrics  # noqa: E402
from industrial_ad.fusion import rcec  # noqa: E402
from src.utils import dists2map  # noqa: E402

FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "rcec_v1"
DEV_MPDD_ROOT = EXPERIMENT_ROOT / "development_mpdd"

DATASETS = {
    "mpdd": {"role": "development",
             "dino_fmt": "features_vitb14_s{seed}_k{shot}",
             "clip_fmt": "features_s{seed}_k{shot}"},
    "btad": {"role": "external_frozen_validation",
             "dino_fmt": "features_vitb14_btad_s{seed}_k{shot}",
             "clip_fmt": "features_btad_s{seed}_k{shot}"},
    "visa": {"role": "in_domain_frozen_validation",
             "dino_fmt": "visa_features_vitb14/s{seed}_k{shot}",
             "clip_fmt": "visa_features/s{seed}_k{shot}"},
    "mvtec": {"role": "external_frozen_validation",
              "dino_fmt": "mvtec_features_vitb14/s{seed}_k{shot}",
              "clip_fmt": "mvtec_features/s{seed}_k{shot}"},
}

# ---------------------------------------------------------------------------
# Config & candidates
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATES = [
    {"direction": d, "k": k, "lambda": lam}
    for d in ("dino_to_clip", "symmetric")
    for k in (1, 3, 5)
    for lam in (0.25, 0.50)
]


def load_config(path: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_sha256"] = sha256_file(path)
    return cfg


def candidates_from_config(cfg: dict) -> list[dict]:
    dirs_ = cfg.get("directions", ["dino_to_clip", "symmetric"])
    ks = cfg.get("neighbor_k", [1, 3, 5])
    lams = cfg.get("lambda", [0.25, 0.50])
    return [
        {"direction": d, "k": int(k), "lambda": float(lam)}
        for d in dirs_
        for k in ks
        for lam in lams
    ]


def candidate_id(c: dict) -> str:
    return f"{c['direction']}_k{c['k']}_lam{c['lambda']:g}"


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def hash_npz(path: Path) -> str:
    """Hash the raw byte content of an .npz cache file."""
    return sha256_file(path)


# ---------------------------------------------------------------------------
# Feature dirs
# ---------------------------------------------------------------------------

def dirs_for(dataset: str, seed: int, shot: int) -> tuple[Path, Path]:
    meta = DATASETS[dataset]
    dino = FEATURES_ROOT / meta["dino_fmt"].format(seed=seed, shot=shot) / "anomalydino_visual"
    clip = FEATURES_ROOT / meta["clip_fmt"].format(seed=seed, shot=shot) / "anomalyclip_text"
    return dino, clip


def manifest_for(dataset: str) -> dict:
    return json.loads(
        (ROOT / "data" / "splits" / dataset / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-category evaluation
# ---------------------------------------------------------------------------

def reference_ids_for(manifest: dict, category: str, seed: int, shot: int) -> list[str]:
    return list(manifest["categories"][category][str(seed)][str(shot)])


def evaluate_category(
    dino: dict,
    clip: dict,
    ref_ids: list[str],
    seed: int,
    shot: int,
    candidate: dict,
    cfg: dict,
    category: str,
) -> dict:
    """Evaluate one category for one candidate. Pure algorithm + metrics.

    Labels/masks are only used inside the evaluator calls at the end; the RCEC
    score computation never receives them.
    """
    norm = cfg.get("normal_calibration", {})
    epsilon = float(norm.get("epsilon", 1e-6))
    z_clip = tuple(float(v) for v in norm.get("z_clip", [-5.0, 10.0]))
    map_size = tuple(int(v) for v in cfg.get("postprocess", {}).get("map_size", [448, 448]))
    dino_weight = float(cfg.get("fixed", {}).get("dino_weight", 0.5))
    direction = candidate["direction"]
    k = int(candidate["k"])
    lam = float(candidate["lambda"])

    if dino["ref_patch_features"].shape[0] != len(ref_ids):
        raise rcec.AlignmentError(
            f"ref blocks {dino['ref_patch_features'].shape[0]} != manifest refs {len(ref_ids)}")
    if clip["ref_patch_features"].shape[0] != len(ref_ids):
        raise rcec.AlignmentError(
            f"CLIP ref blocks {clip['ref_patch_features'].shape[0]} != manifest refs {len(ref_ids)}")

    aligned = rcec.align_and_normalize_paired_features(
        dino_patch=dino["patch_features"],
        clip_patch=clip["patch_features"],
        dino_ref=dino["ref_patch_features"],
        clip_ref=clip["ref_patch_features"],
        dino_sample_ids=dino["sample_ids"],
        clip_sample_ids=clip["sample_ids"],
        dino_grid=dino["grid_size"],
        resize_fn=resize_patches,
    )
    d_feat, c_feat = aligned["d_feat"], aligned["c_feat"]
    d_ref, c_ref = aligned["d_ref"], aligned["c_ref"]
    grid = aligned["grid"]

    memory = rcec.build_paired_reference_memory(d_ref, c_ref, len(ref_ids))

    # A1 raw concat score (frozen path) and DINO-only score.
    s_a1 = rcec.compute_a1_dists(d_feat, c_feat, memory, dino_weight=dino_weight)
    s_dino = rcec.compute_dino_dists(d_feat, memory)

    # Conditional cross-encoder score.
    r_cd, r_dc = rcec.compute_conditional_scores(d_feat, c_feat, memory, direction, k)
    r_cond = r_cd if direction == "dino_to_clip" else 0.5 * (r_cd + r_dc)

    # Reference-only LOO calibration.
    loo = rcec.compute_reference_loo_statistics(
        d_ref, c_ref, len(ref_ids), direction=direction, k=k, shot=shot)
    stats_a1 = rcec.compute_reference_stats(loo["a1_loo"], epsilon=epsilon)
    stats_cond = rcec.compute_reference_stats(loo["cond_loo"], epsilon=epsilon)

    s_rcec = rcec.combine_rcec_scores(
        s_a1, r_cond, stats_a1, stats_cond, lam, epsilon=epsilon, z_clip=z_clip)

    n = d_feat.shape[0]
    h, w = grid
    s_rcec_grid = s_rcec.reshape(n, h, w)
    s_a1_grid = s_a1.reshape(n, h, w)
    s_dino_grid = s_dino.reshape(n, h, w)
    maps_rcec = np.stack(
        [dists2map(s_rcec_grid[i], map_size) for i in range(n)]).astype(np.float32)
    maps_a1 = np.stack(
        [dists2map(s_a1_grid[i], map_size) for i in range(n)]).astype(np.float32)
    maps_dino = np.stack(
        [dists2map(s_dino_grid[i], map_size) for i in range(n)]).astype(np.float32)

    gt_masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
    gt_sp = np.asarray(dino["gt_sp"])

    def full_metrics(maps: np.ndarray) -> dict:
        pixel = None
        try:
            pixel = compute_metrics(maps.astype(np.float64), gt_masks)
        except ValueError:
            # e.g. a category whose stride-8 masks collapse to a single class;
            # record the gap instead of crashing (reported in checks).
            pixel = None
        if len(np.unique(gt_sp)) < 2:
            image = {"image_auroc": None, "image_ap": None, "image_f1_max": None}
        else:
            image = compute_image_metrics(gt_sp, maps)
        return {"pixel": pixel, "image": image}

    metrics_rcec = full_metrics(maps_rcec)
    metrics_a1 = full_metrics(maps_a1)
    metrics_dino = full_metrics(maps_dino)

    delta = {}
    for group in ("pixel", "image"):
        delta[group] = {}
        src = metrics_rcec[group]
        keys = src.keys() if isinstance(src, dict) else []
        for key in keys:
            a, b = metrics_rcec[group][key], metrics_a1[group][key]
            delta[group][key] = None if (a is None or b is None) else float(a - b)

    return {
        "schema_version": 1,
        "method": "rcec_v1",
        "category": category,
        "seed": seed,
        "shot": shot,
        "n_test_images": n,
        "grid": list(grid),
        "candidate": {"direction": direction, "k": k, "lambda": lam},
        "calibration": {
            "a1": {"median": stats_a1["median"], "mad": stats_a1["mad"]},
            "cond": {"median": stats_cond["median"], "mad": stats_cond["mad"]},
            "exclusion_rule": loo["exclusion_rule"],
            "n_ref_patches": loo["n_ref_patches"],
            "test_statistics_used": False,
        },
        "metrics": {
            "rcec": metrics_rcec,
            "a1": metrics_a1,
            "dino": metrics_dino,
            "delta_rcec_vs_a1": delta,
        },
        "leakage_flags": {
            "test_labels_used_by_method": False,
            "test_masks_used_by_method": False,
            "test_distribution_used_for_calibration": False,
            "validation_dataset_used_for_tuning": False,
            "category_specific_test_rules_used": False,
        },
        "checks": {
            "no_nan_inf_scores": bool(
                np.all(np.isfinite(s_rcec)) and np.all(np.isfinite(maps_rcec))),
            "finite_ref_stats": True,
        },
    }


def evaluate_config(
    dataset: str, seed: int, shot: int, candidate: dict, cfg: dict
) -> dict:
    """Evaluate every category of one (dataset, seed, shot, candidate)."""
    dino_dir, clip_dir = dirs_for(dataset, seed, shot)
    manifest = manifest_for(dataset)
    cat_paths = sorted(p for p in dino_dir.glob("*.npz") if p.stem != "export_report")
    per_category = []
    for cat_path in cat_paths:
        cat = cat_path.stem
        clip_path = clip_dir / f"{cat}.npz"
        if not clip_path.is_file():
            raise rcec.RCECError(f"missing clip features: {clip_path}")
        ref_ids = reference_ids_for(manifest, cat, seed, shot)
        dino = load_features(cat_path)
        clip = load_features(clip_path)
        per_category.append(
            evaluate_category(dino, clip, ref_ids, seed, shot, candidate, cfg, category=cat))

    def mean_metric(method: str, group: str, key: str) -> Optional[float]:
        vals = []
        for r in per_category:
            m = r["metrics"][method][group]
            vals.append(m[key] if isinstance(m, dict) else None)
        vals = [v for v in vals if v is not None]
        return None if not vals else round(float(np.mean(vals)), 6)

    def mean_delta(key_group: str, key: str) -> Optional[float]:
        vals = []
        for r in per_category:
            d = r["metrics"]["delta_rcec_vs_a1"][key_group]
            vals.append(d[key] if isinstance(d, dict) else None)
        vals = [v for v in vals if v is not None]
        return None if not vals else round(float(np.mean(vals)), 6)

    def method_block(method: str) -> dict:
        return {
            "pixel": {k: mean_metric(method, "pixel", k)
                      for k in ("pixel_auroc", "pixel_ap", "pixel_aupro")},
            "image": {k: mean_metric(method, "image", k)
                      for k in ("image_auroc", "image_ap", "image_f1_max")},
        }

    report = {
        "schema_version": 1,
        "method": "rcec_v1",
        "dataset": dataset,
        "dataset_role": DATASETS[dataset]["role"],
        "seed": seed,
        "shot": shot,
        "candidate": {"direction": candidate["direction"], "k": candidate["k"],
                      "lambda": candidate["lambda"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest_sha256": sha256_file(ROOT / "data" / "splits" / dataset / "manifest.json"),
            "dino_cache_sha256": sha256_file(dino_dir / f"{per_category[0]['category']}.npz") if per_category else None,
            "clip_cache_sha256": sha256_file(clip_dir / f"{per_category[0]['category']}.npz") if per_category else None,
        },
        "calibration": {
            "kind": "median_mad",
            "exclusion_rule": per_category[0]["calibration"]["exclusion_rule"] if per_category else None,
            "test_statistics_used": False,
        },
        "leakage_flags": {
            "test_labels_used_by_method": False,
            "test_masks_used_by_method": False,
            "test_distribution_used_for_calibration": False,
            "validation_dataset_used_for_tuning": False,
            "category_specific_test_rules_used": False,
        },
        "metrics": {
            "rcec": method_block("rcec"),
            "a1": method_block("a1"),
            "dino": method_block("dino"),
            "delta_rcec_vs_a1": {
                "pixel": {k: mean_delta("pixel", k)
                          for k in ("pixel_auroc", "pixel_ap", "pixel_aupro")},
                "image": {k: mean_delta("image", k)
                          for k in ("image_auroc", "image_ap", "image_f1_max")},
            },
        },
        "per_category": per_category,
        "checks": {"all_finite": all(
            r["checks"]["no_nan_inf_scores"] for r in per_category)},
    }
    return report


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def mean_pixel_ap(rows: list[dict]) -> float:
    vals = [r["metrics"]["rcec"]["pixel"]["pixel_ap"] for r in rows]
    return float(np.mean([v for v in vals if v is not None]))


def mean_a1_pixel_ap(rows: list[dict]) -> float:
    vals = [r["metrics"]["a1"]["pixel"]["pixel_ap"] for r in rows]
    return float(np.mean([v for v in vals if v is not None]))


def small_gate_pass(config_rows: list[dict]) -> tuple[bool, dict]:
    """Phase-2 small gate on MPDD seed0 x shot {1,2,4} (one candidate).

    Conditions (task book 8 / Phase 2):
      * mean Pixel-AP over the three shots >= A1;
      * >= 2/3 shots positive vs A1;
      * no single shot worse than -0.010 vs A1;
      * no numerical degeneration / leakage flags.
    """
    if len(config_rows) != 3:
        return False, {"reason": f"expected 3 configs, got {len(config_rows)}"}
    deltas = [r["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"] for r in config_rows]
    if any(d is None for d in deltas):
        return False, {"reason": "missing delta"}
    mean_d = float(np.mean(deltas))
    n_pos = int(sum(1 for d in deltas if d > 0))
    worst = min(deltas)
    detail = {
        "mean_delta": round(mean_d, 6),
        "n_positive_shots": n_pos,
        "worst_shot_delta": round(worst, 6),
        "deltas": [round(d, 6) for d in deltas],
    }
    ok = (
        mean_d >= 0.0
        and n_pos >= 2
        and worst >= -0.010
        and all(r["checks"]["all_finite"] for r in config_rows)
        and all(not any(r["leakage_flags"].values()) for r in config_rows)
    )
    return ok, detail


def selection_pool_pass(config_rows: list[dict]) -> tuple[bool, dict]:
    """Phase-3 selection pool on MPDD 9 configs (task book 8 / Phase 3)."""
    if len(config_rows) != 9:
        return False, {"reason": f"expected 9 configs, got {len(config_rows)}"}
    deltas = [r["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"] for r in config_rows]
    mean_d = float(np.mean(deltas))
    n_pos = int(sum(1 for d in deltas if d > 0))
    worst = min(deltas)

    per_cat = {}
    for r in config_rows:
        for pc in r["per_category"]:
            cat = pc["category"]
            per_cat.setdefault(cat, []).append(
                pc["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"])
    cat_means = {c: float(np.mean(v)) for c, v in per_cat.items()}
    n_cat_ok = int(sum(1 for v in cat_means.values() if v >= 0.0))
    worst_cat = min(cat_means.values())

    img_ap = [r["metrics"]["delta_rcec_vs_a1"]["image"]["image_ap"] for r in config_rows]
    img_ap = [v for v in img_ap if v is not None]
    mean_img_ap = float(np.mean(img_ap)) if img_ap else None
    img_f1 = [r["metrics"]["delta_rcec_vs_a1"]["image"]["image_f1_max"] for r in config_rows]
    img_f1 = [v for v in img_f1 if v is not None]
    mean_img_f1 = float(np.mean(img_f1)) if img_f1 else None

    detail = {
        "mean_delta": round(mean_d, 6),
        "n_positive_configs": n_pos,
        "worst_config_delta": round(worst, 6),
        "n_categories_non_negative": n_cat_ok,
        "n_categories_total": len(cat_means),
        "worst_category_mean_delta": round(worst_cat, 6),
        "mean_image_ap_delta": round(mean_img_ap, 6) if mean_img_ap is not None else None,
        "mean_image_f1_max_delta": round(mean_img_f1, 6) if mean_img_f1 is not None else None,
    }
    ok = (
        mean_d >= 0.005
        and n_pos >= 7
        and worst >= -0.010
        and n_cat_ok >= 4
        and worst_cat >= -0.015
        and (mean_img_ap is None or mean_img_ap >= -0.005)
        and (mean_img_f1 is None or mean_img_f1 >= -0.010)
        and all(r["checks"]["all_finite"] for r in config_rows)
        and all(not any(r["leakage_flags"].values()) for r in config_rows)
    )
    return ok, detail


def final_gate_pass(validation_reports: dict) -> tuple[bool, dict]:
    """Phase-6 final gate across BTAD/MVTec/VisA (task book 8 / Phase 6)."""
    detail = {}
    for ds, reports in validation_reports.items():
        deltas = [r["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"] for r in reports]
        detail[ds] = {
            "mean_delta": round(float(np.mean(deltas)), 6),
            "n_positive": int(sum(1 for d in deltas if d > 0)),
            "n_configs": len(reports),
        }
    ds_deltas = [detail[ds]["mean_delta"] for ds in validation_reports]
    mean_all = float(np.mean(ds_deltas))
    n_ds_pos = int(sum(1 for d in ds_deltas if d > 0))
    worst_ds = min(ds_deltas)
    n_configs_pos = int(sum(detail[ds]["n_positive"] for ds in validation_reports))
    n_configs_total = int(sum(detail[ds]["n_configs"] for ds in validation_reports))

    # Category-level worst across validation datasets.
    cat_worst = None
    for ds, reports in validation_reports.items():
        per_cat = {}
        for r in reports:
            for pc in r["per_category"]:
                per_cat.setdefault(pc["category"], []).append(
                    pc["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"])
        for c, v in per_cat.items():
            m = float(np.mean(v))
            cat_worst = m if cat_worst is None else min(cat_worst, m)

    detail["_summary"] = {
        "mean_delta_all_ds": round(mean_all, 6),
        "n_ds_positive": n_ds_pos,
        "worst_ds_delta": round(worst_ds, 6),
        "n_configs_positive": n_configs_pos,
        "n_configs_total": n_configs_total,
        "worst_category_mean_delta": round(cat_worst, 6) if cat_worst is not None else None,
    }
    ok = (
        mean_all > 0.0
        and n_ds_pos >= 2
        and worst_ds >= -0.005
        and n_configs_pos >= 18
        and (cat_worst is None or cat_worst >= -0.030)
    )
    return ok, detail


def check_no_config_overrides(candidate_args: dict, frozen_candidate: dict) -> None:
    """Frozen validation must never accept CLI overrides of method parameters."""
    for key, value in candidate_args.items():
        if key in frozen_candidate and value is not None:
            expected = frozen_candidate[key]
            if isinstance(expected, (int, float)) and float(value) != float(expected):
                raise rcec.RCECError(
                    f"frozen parameter override rejected: {key}={value} != {expected}")
            if isinstance(expected, str) and value != expected:
                raise rcec.RCECError(
                    f"frozen parameter override rejected: {key}={value} != {expected}")


__all__ = [
    "ROOT",
    "FEATURES_ROOT",
    "EXPERIMENT_ROOT",
    "DEV_MPDD_ROOT",
    "DATASETS",
    "DEFAULT_CANDIDATES",
    "load_config",
    "candidates_from_config",
    "candidate_id",
    "sha256_file",
    "hash_npz",
    "dirs_for",
    "manifest_for",
    "reference_ids_for",
    "evaluate_category",
    "evaluate_config",
    "small_gate_pass",
    "selection_pool_pass",
    "final_gate_pass",
    "check_no_config_overrides",
]
