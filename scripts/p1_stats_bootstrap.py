"""P1-A/P1-B: bootstrap CI and failure boundaries for A1 vs matched feature-DINO-only.

Reads the 324 compact per-image patch maps plus the user's legal GT masks, and:

P1-A  - for every (dataset, seed, shot) config:
          - full-sample Delta-Pixel-AP (pooled per-category, mean over categories;
            sanity-checked against the p0_3 report at 5e-4)
          - CATEGORY bootstrap CI (B draws of the category multiset, resampling
            the precomputed per-category pooled Pixel-AP) -- matches the paper-table
            statistic (mean over categories)
          - IMAGE bootstrap CI (paired per-image Delta-AP over anomalous test
            images; unit = image) -- image-level robustness companion
        - per (dataset, shot): 3-seed mean +/- std of the full-sample Delta-AP
P1-B  - worst category, negative-gain categories, and per-image failure samples
        (anomalous test images where concat per-image Pixel-AP < dino-only).

Metric definition is identical to the frozen A1 evaluator: for a category, all
test-image pixels are pooled at stride=8 and Pixel-AP is computed on that pooled
vector (average_precision_score). Per-image Pixel-AP is defined only for images
whose GT mask contains at least one anomalous pixel. No test labels/masks are
used for fitting anything. All outputs are written under evidence/p1/.

Statistical-level note: the category bootstrap resamples the 3-15 category means
of the *pooled* pixel metric (the paper-table unit); the image bootstrap
resamples anomalous images on the *per-image* pixel metric. The two are different
units and are reported separately, never pooled into one sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

cv2.setLogLevel(0)  # silence libpng iCCP noise from per-mask imread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_repro_20260827"))
sys.path.insert(0, str(ROOT / "src"))

from recompute_tables import (  # noqa: E402
    MAP_SIZE,
    STRIDE,
    build_visa_mask_map,
    load_mask_for_sample,
    maps_to_448,
)

MAPS_ROOT = ROOT / "submission_repro_20260827" / "predictions_compact" / "maps"
REPORT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "p0_rebuild_20260826"
OUT_ROOT = ROOT / "submission_repro_20260827" / "evidence" / "p1"
DATA_ROOTS = {
    "mpdd": ROOT / "data" / "mpdd_raw" / "MPDD",
    "btad": ROOT / "data" / "btad_raw" / "BTech_Dataset_transformed",
    "visa": ROOT / "data" / "visa_raw",
    "mvtec": ROOT / "data" / "mvtec",
}
DATASETS = ("mpdd", "btad", "visa", "mvtec")
SEEDS = (0, 1, 2)
SHOTS = (1, 2, 4)
B_BOOTSTRAP = 2000
CI_LEVEL = 0.95
ALPHA = (1.0 - CI_LEVEL) / 2.0
SANITY_TOLERANCE = 5e-4
RNG_SEED = 20260827


def ap_score(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, bit-identical to sklearn.average_precision_score."""
    order = np.argsort(s, kind="mergesort")[::-1]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    pos = np.flatnonzero(y_sorted)
    if pos.size == 0:
        return float("nan")
    prec = tp[pos] / (np.arange(1, len(y_sorted) + 1)[pos])
    return float(prec.mean())


def mask_for(dataset: str, sample_id: str, data_root: Path,
             visa_mask_map, cache: dict) -> np.ndarray:
    key = f"{dataset}:{sample_id}"
    if key not in cache:
        cache[key] = load_mask_for_sample(dataset, sample_id, data_root, visa_mask_map, MAP_SIZE)
    return cache[key]


def process_config(dataset: str, seed: int, shot: int,
                   mask_cache: dict, visa_mask_map) -> dict:
    cfg_dir = MAPS_ROOT / dataset / f"s{seed}_k{shot}"
    data_root = DATA_ROOTS[dataset]
    per_cat = []
    per_image_rows = []  # (category, sample_id, concat_ap, dino_ap, delta)
    for cat_path in sorted(cfg_dir.glob("*.npz")):
        with np.load(cat_path, allow_pickle=False) as data:
            concat_f16 = np.asarray(data["concat_patch_map"])
            dino_f16 = np.asarray(data["dino_patch_map"])
            ids = [str(s) for s in data["sample_ids"]]
        concat56 = maps_to_448(concat_f16)[:, ::STRIDE, ::STRIDE].astype(np.float32)
        dino56 = maps_to_448(dino_f16)[:, ::STRIDE, ::STRIDE].astype(np.float32)
        labels = np.stack([mask_for(dataset, sid, data_root, visa_mask_map, mask_cache)
                           for sid in ids])[:, ::STRIDE, ::STRIDE].astype(np.int8)
        name = cat_path.stem
        # pooled per-category Pixel-AP (paper-table statistic)
        sc = concat56.ravel()
        sd = dino56.ravel()
        lab = labels.ravel()
        ap_c = ap_score(lab, sc)
        ap_d = ap_score(lab, sd)
        per_cat.append({
            "category": name, "test_images": len(ids),
            "concat_pixel_ap": round(ap_c, 6), "dino_pixel_ap": round(ap_d, 6),
            "delta_ap": round(ap_c - ap_d, 6),
        })
        # per-image Pixel-AP on anomalous images only
        for j, sid in enumerate(ids):
            lab_j = labels[j].ravel()
            if lab_j.sum() == 0:
                continue
            apc = ap_score(lab_j, concat56[j].ravel())
            apd = ap_score(lab_j, dino56[j].ravel())
            per_image_rows.append((name, sid, apc, apd, apc - apd))

    full_delta = float(np.mean([r["delta_ap"] for r in per_cat]))
    ref = json.loads((REPORT_ROOT / f"{dataset}_s{seed}_k{shot}.json").read_text(encoding="utf-8"))
    sanity = abs(full_delta - ref["mean_delta_ap_vs_feature_dino"]) <= SANITY_TOLERANCE

    # category bootstrap (resample category means of pooled AP)
    cat_deltas = np.asarray([r["delta_ap"] for r in per_cat], dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED + seed * 31 + shot * 7)
    cat_rep = [float(np.mean(cat_deltas[rng.integers(0, len(cat_deltas), size=len(cat_deltas))]))
               for _ in range(B_BOOTSTRAP)]

    # paired image bootstrap over anomalous images (per-image Delta-AP)
    img_deltas = np.asarray([r[4] for r in per_image_rows], dtype=np.float64)
    if img_deltas.size:
        img_rep = [float(np.mean(img_deltas[rng.integers(0, img_deltas.size, size=img_deltas.size)]))
                   for _ in range(B_BOOTSTRAP)]
        img_ci = ci(img_rep)
    else:
        img_ci = {"mean": None, "std": None, "lo": None, "hi": None,
                  "n_bootstrap": B_BOOTSTRAP, "level": CI_LEVEL, "note": "no anomalous images"}

    return {
        "dataset": dataset, "seed": seed, "shot": shot,
        "n_categories": len(per_cat),
        "n_test_images": int(sum(r["test_images"] for r in per_cat)),
        "n_anomalous_images": int(len(per_image_rows)),
        "full_sample_delta_ap": round(full_delta, 6),
        "sanity_ref_delta_ap": round(ref["mean_delta_ap_vs_feature_dino"], 6),
        "sanity_within_tolerance": bool(sanity),
        "category_bootstrap": ci(cat_rep),
        "image_bootstrap": img_ci,
        "per_category": per_cat,
        "per_image_failures_top5": sorted(per_image_rows, key=lambda t: t[4])[:5],
    }


def ci(replicates: list[float]) -> dict:
    arr = np.asarray(replicates)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=1)), 6),
        "lo": round(float(np.percentile(arr, 100 * ALPHA)), 6),
        "hi": round(float(np.percentile(arr, 100 * (1 - ALPHA))), 6),
        "n_bootstrap": int(len(arr)),
        "level": CI_LEVEL,
    }


def worker(dataset: str) -> dict:
    visa_mask_map = build_visa_mask_map(DATA_ROOTS["visa"]) if dataset == "visa" else None
    mask_cache: dict[str, np.ndarray] = {}
    rows = []
    for seed in SEEDS:
        for shot in SHOTS:
            row = process_config(dataset, seed, shot, mask_cache, visa_mask_map)
            rows.append(row)
            print(f"[{dataset} s{seed}/k{shot}] delta={row['full_sample_delta_ap']:.6f} "
                  f"(ref {row['sanity_ref_delta_ap']:.6f}, ok={row['sanity_within_tolerance']}) "
                  f"catCI=[{row['category_bootstrap']['lo']},{row['category_bootstrap']['hi']}] "
                  f"imgCI=[{row['image_bootstrap']['lo']},{row['image_bootstrap']['hi']}]",
                  flush=True)
    return {"dataset": dataset, "configs": rows}


def main() -> int:
    global B_BOOTSTRAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-b", type=int, default=B_BOOTSTRAP)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    B_BOOTSTRAP = args.bootstrap_b

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.workers > 1 and len(args.datasets) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            dataset_results = list(pool.map(worker, args.datasets))
    else:
        dataset_results = [worker(ds) for ds in args.datasets]

    config_rows = [r for dr in dataset_results for r in dr["configs"]]
    failure_rows = []
    for r in config_rows:
        for name, sid, apc, apd, d in r.pop("per_image_failures_top5"):
            failure_rows.append({"dataset": r["dataset"], "seed": r["seed"], "shot": r["shot"],
                                 "category": name, "sample_id": sid,
                                 "concat_pixel_ap": round(apc, 6), "dino_pixel_ap": round(apd, 6),
                                 "delta_ap": round(d, 6)})

    # shot-wise / dataset-wise
    shot_rows = []
    for dataset in args.datasets:
        for shot in SHOTS:
            deltas = [r["full_sample_delta_ap"] for r in config_rows
                      if r["dataset"] == dataset and r["shot"] == shot]
            arr = np.asarray(deltas)
            shot_rows.append({"dataset": dataset, "shot": shot, "n_seeds": len(deltas),
                              "mean_delta_ap": round(float(arr.mean()), 6),
                              "std_delta_ap": round(float(arr.std(ddof=1)), 6) if len(deltas) > 1 else 0.0,
                              "seeds": [round(d, 6) for d in deltas]})
    dataset_rows = []
    for dataset in args.datasets:
        darr = np.asarray([r["full_sample_delta_ap"] for r in config_rows if r["dataset"] == dataset])
        dataset_rows.append({"dataset": dataset, "n_configs": len(darr),
                             "mean_delta_ap": round(float(darr.mean()), 6),
                             "std_delta_ap": round(float(darr.std(ddof=1)), 6)})

    neg_cats = [cat for r in config_rows for cat in r["per_category"] if cat["delta_ap"] < 0]
    worst_rows = []
    for dataset in args.datasets:
        cats = sorted({r["category"] for row in config_rows if row["dataset"] == dataset
                       for r in row["per_category"]})
        for cat in cats:
            ds = [r for row in config_rows if row["dataset"] == dataset for r in row["per_category"]
                  if r["category"] == cat]
            worst_rows.append({"dataset": dataset, "category": cat, "n_configs": len(ds),
                               "mean_delta_ap": round(float(np.mean([r["delta_ap"] for r in ds])), 6),
                               "negative_configs": sum(1 for r in ds if r["delta_ap"] < 0)})
    worst_rows.sort(key=lambda r: r["mean_delta_ap"])

    payload = {
        "schema_version": 1,
        "kind": "p1_stats_bootstrap_ci",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistics_note": (
            "full_sample_delta_ap and category_bootstrap use the paper-table statistic "
            "(mean over categories of per-category pooled Pixel-AP, stride=8). "
            "image_bootstrap uses per-image Pixel-AP over anomalous test images only. "
            "The two units are reported separately and never pooled."),
        "bootstrap": {"B": B_BOOTSTRAP, "level": CI_LEVEL, "rng_seed": RNG_SEED,
                      "sanity_tolerance": SANITY_TOLERANCE},
        "sanity_all_within_tolerance": bool(all(r["sanity_within_tolerance"] for r in config_rows)),
        "configs": config_rows,
        "shot_wise": shot_rows,
        "dataset_wise": dataset_rows,
        "negative_categories": {
            "count_across_configs": len(neg_cats),
            "unique_categories": sorted({row["dataset"] + "@" + cat["category"]
                                         for row in config_rows
                                         for cat in row["per_category"] if cat["delta_ap"] < 0}),
        },
        "worst_categories_by_dataset": worst_rows,
        "failure_samples_top5_per_config": failure_rows,
    }
    (out_root / "p1_a_bootstrap_ci.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "n_configs": len(config_rows),
                      "sanity_all_within_tolerance": payload["sanity_all_within_tolerance"],
                      "output": str(out_root / "p1_a_bootstrap_ci.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
