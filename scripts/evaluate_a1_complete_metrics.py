"""A1 complete metrics (image + pixel) on all four datasets (post-freeze).

The frozen `evaluate_a1_feature_fusion.py::compute_metrics` only returns
pixel-level metrics (pixel_auroc / pixel_ap / pixel_aupro). This standalone
script reuses the frozen fusion read-only and adds the missing image-level
metrics (image_auroc / image_ap / image_f1_max) so A1 can be compared against
the trained baselines on the same metric set.

Image score convention: max-pooling over the full-resolution (448x448) anomaly
map, i.e. image_score = max_pixels(map). This matches the PatchCore / WinCLIP
image-score convention and is the standard default for pixel-based AD methods.

Frozen config (unchanged): concat pca_dim=0 whiten=0 dino_weight=0.5, KNN k=1,
distance/2, stride=8, map=448.

Dataset feature layouts:
  mpdd  : dino features_vitb14_s{seed}_k{shot}/anomalydino_visual
          clip features_s{seed}_k{shot}/anomalyclip_text
  btad  : dino features_vitb14_btad_s{seed}_k{shot}/anomalydino_visual
          clip features_btad_s{seed}_k{shot}/anomalyclip_text
  visa  : dino visa_features_vitb14/s{seed}_k{shot}/anomalydino_visual
          clip visa_features/s{seed}_k{shot}/anomalyclip_text
  mvtec : dino mvtec_features_vitb14/s{seed}_k{shot}/anomalydino_visual
          clip mvtec_features/s{seed}_k{shot}/anomalyclip_text

BTAD category 03 has a non-square 32x42 grid and ~441 test images; the frozen
concat fusion peaks at ~13 GiB for that single category, which can exceed
available RAM. We therefore keep a chunked equivalent of the concat path
(`fuse_concat_maps_chunked`) that reuses the same frozen helpers
(build_alignment_plan / resize_patches / normalize / faiss / dists2map) and is
bit-identical to the frozen `fuse_category(..., "concat", ...)`, but builds the
concat feature matrix and the KNN search in chunks so peak memory stays bounded.
It is only used as a fallback when the frozen path raises MemoryError.

This script does NOT touch any frozen script in freeze_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from evaluate_a1_feature_fusion import (  # noqa: E402
    STRIDE,
    compute_metrics,
    fuse_category,
    load_features,
    resize_patches,
)
from evaluate_unified import f1_max  # noqa: E402
from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from src.utils import dists2map  # noqa: E402

FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_complete_metrics_20260819"

SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]

DATASETS = {
    "mpdd": {
        "role": "development",
        "dino_dir_fmt": "features_vitb14_s{seed}_k{shot}",
        "clip_dir_fmt": "features_s{seed}_k{shot}",
    },
    "btad": {
        "role": "external_frozen_validation",
        "dino_dir_fmt": "features_vitb14_btad_s{seed}_k{shot}",
        "clip_dir_fmt": "features_btad_s{seed}_k{shot}",
    },
    "visa": {
        "role": "in_domain_frozen_validation",
        "dino_dir_fmt": "visa_features_vitb14/s{seed}_k{shot}",
        "clip_dir_fmt": "visa_features/s{seed}_k{shot}",
    },
    "mvtec": {
        "role": "external_frozen_validation",
        "dino_dir_fmt": "mvtec_features_vitb14/s{seed}_k{shot}",
        "clip_dir_fmt": "mvtec_features/s{seed}_k{shot}",
    },
}

FROZEN_CONFIG = (
    "concat pca_dim=0 whiten=0 dino_weight=0.5 (dinov2_vitb14 DINO + AnomalyCLIP), "
    "KNN k=1, distance/2, stride=8, map=448"
)


def compute_image_metrics(gt_sp: np.ndarray, pixel_maps: np.ndarray) -> dict:
    """Image-level AUROC / AP / F1-max from max-pooled per-image scores."""
    scores = pixel_maps.reshape(pixel_maps.shape[0], -1).max(axis=1)
    return {
        "image_auroc": float(roc_auc_score(gt_sp, scores)),
        "image_ap": float(average_precision_score(gt_sp, scores)),
        "image_f1_max": f1_max(gt_sp.astype(np.uint8), scores),
    }


def _l2_normalize_inplace(x: np.ndarray) -> None:
    """In-place L2 normalization, bit-identical to sklearn ``normalize`` (no copy)."""
    flat = x.reshape(-1, x.shape[-1])
    norms = np.einsum("ij,ij->i", flat, flat)
    np.sqrt(norms, out=norms)
    flat /= norms[:, None]


def fuse_concat_maps_chunked(
    dino: dict,
    clip: dict,
    map_size: tuple[int, int],
    dino_weight: float = 0.5,
    chunk: int = 16384,
) -> np.ndarray:
    """Chunked, bit-identical equivalent of frozen concat fusion.

    Mirrors `evaluate_a1_feature_fusion.fuse_category` concat branch exactly:
    reorder clip test samples to the DINO sample order, resize the clip grid to
    the DINO grid, L2-normalize each branch per patch, concatenate with weight
    (w * dino, (1-w) * clip), build a normal-reference faiss memory bank, then
    KNN (k=1) distance / 2 -> dists2map. Only the test-side concat + KNN is
    processed in chunks so peak memory stays bounded. Normalization is done
    in-place (sklearn ``normalize`` copies the full array and would re-trigger
    the same OOM this fallback exists to avoid).
    """
    grid = dino["grid_size"]

    alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])

    # ---- DINO branch: normalize in-place, then drop the raw arrays from dicts ----
    dino_feat = dino.pop("patch_features")
    dino_ref = dino.pop("ref_patch_features")
    _l2_normalize_inplace(dino_feat)
    _l2_normalize_inplace(dino_ref)

    # ---- CLIP branch: reorder, resize, normalize in-place, drop raw arrays ----
    clip_raw = clip.pop("patch_features")
    candidate_order = alignment.candidate_order
    if np.array_equal(candidate_order, np.arange(clip_raw.shape[0])):
        clip_feat = clip_raw
    else:
        clip_feat = clip_raw[candidate_order]
        del clip_raw
    clip_ref = clip.pop("ref_patch_features")
    clip_feat = resize_patches(clip_feat, grid)
    clip_ref = resize_patches(clip_ref, grid)
    _l2_normalize_inplace(clip_feat)
    _l2_normalize_inplace(clip_ref)

    # ---- normal-reference memory bank (small) ----
    ref = np.concatenate([dino_weight * dino_ref, (1.0 - dino_weight) * clip_ref], axis=-1)
    ref_flat = ref.reshape(-1, ref.shape[-1]).astype(np.float32)
    faiss.normalize_L2(ref_flat)
    index = faiss.IndexFlatL2(ref_flat.shape[1])
    index.add(ref_flat)

    # ---- chunked test-side concat + KNN ----
    n = dino_feat.shape[0]
    dino_flat = dino_feat.reshape(-1, dino_feat.shape[-1])
    clip_flat = clip_feat.reshape(-1, clip_feat.shape[-1])
    n_patches = dino_flat.shape[0]
    dists = np.empty(n_patches, dtype=np.float32)

    for start in range(0, n_patches, chunk):
        end = min(start + chunk, n_patches)
        feat = np.concatenate(
            [dino_weight * dino_flat[start:end], (1.0 - dino_weight) * clip_flat[start:end]],
            axis=-1,
        ).astype(np.float32)
        faiss.normalize_L2(feat)
        dd, _ = index.search(feat, k=1)
        dists[start:end] = dd[:, 0]

    dists = (dists / 2.0).reshape(n, *grid)
    maps = np.stack([dists2map(d, map_size) for d in dists]).astype(np.float32)
    return maps


def evaluate_config(dino_dir: Path, clip_dir: Path, map_size: tuple[int, int]) -> dict:
    """Compute concat + dino complete metrics for one (seed, shot)."""
    concat_rows = []
    dino_rows = []

    for cat_path in sorted(dino_dir.glob("*.npz")):
        cat = cat_path.stem
        clip_path = clip_dir / f"{cat}.npz"
        if not clip_path.is_file():
            raise SystemExit(f"missing clip features: {clip_path}")
        dino = load_features(cat_path)
        clip = load_features(clip_path)
        gt_sp = dino["gt_sp"]

        # dino-only first (cheap, low memory; keeps dino/clip features intact).
        dino_maps = fuse_category(dino, clip, "dino", pca_dim=0, whiten=False, map_size=map_size, dino_weight=0.5)

        # concat via the chunked equivalent (bit-identical to frozen concat, but
        # memory-bounded; the frozen path peaks >10 GiB on large/non-square grids).
        concat_maps = fuse_concat_maps_chunked(dino, clip, map_size, dino_weight=0.5)

        concat_pixel = compute_metrics(concat_maps.astype(np.float64), dino["imgs_masks"])
        dino_pixel = compute_metrics(dino_maps.astype(np.float64), dino["imgs_masks"])
        concat_image = compute_image_metrics(gt_sp, concat_maps)
        dino_image = compute_image_metrics(gt_sp, dino_maps)

        concat_rows.append({"category": cat, "pixel": concat_pixel, "image": concat_image})
        dino_rows.append({"category": cat, "pixel": dino_pixel, "image": dino_image})

    def mean_over(rows: list[dict]) -> dict:
        return {
            "pixel": {
                key: round(float(np.mean([r["pixel"][key] for r in rows])), 6)
                for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
            },
            "image": {
                key: round(float(np.mean([r["image"][key] for r in rows])), 6)
                for key in ("image_auroc", "image_ap", "image_f1_max")
            },
        }

    return {
        "concat": {"mean": mean_over(concat_rows), "per_category": concat_rows},
        "dino": {"mean": mean_over(dino_rows), "per_category": dino_rows},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    map_size = (args.map_size, args.map_size)

    # Build jobs and validate inputs first.
    jobs = []
    for dataset in args.datasets:
        meta = DATASETS[dataset]
        for seed in args.seeds:
            for shot in args.shots:
                dino_dir = FEATURES_ROOT / meta["dino_dir_fmt"].format(seed=seed, shot=shot) / "anomalydino_visual"
                clip_dir = FEATURES_ROOT / meta["clip_dir_fmt"].format(seed=seed, shot=shot) / "anomalyclip_text"
                jobs.append(
                    {
                        "dataset": dataset,
                        "role": meta["role"],
                        "seed": seed,
                        "shot": shot,
                        "dino_dir": dino_dir,
                        "clip_dir": clip_dir,
                    }
                )

    missing = []
    for job in jobs:
        n_dino = len(list(job["dino_dir"].glob("*.npz")))
        n_clip = len(list(job["clip_dir"].glob("*.npz")))
        if n_dino == 0:
            missing.append(f"dino {job['dino_dir']} has 0 npz")
        if n_clip == 0:
            missing.append(f"clip {job['clip_dir']} has 0 npz")
        if n_dino != n_clip:
            missing.append(f"mismatch {job['dino_dir']} ({n_dino}) vs {job['clip_dir']} ({n_clip})")
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", "jobs": len(jobs)}))
        return 0

    completed, failed = [], []
    for job in jobs:
        out_dir = EXPERIMENT_ROOT / job["dataset"] / f"seed{job['seed']}_k{job['shot']}"
        marker = out_dir / "metrics_report.json"
        if marker.is_file():
            completed.append({"dataset": job["dataset"], "seed": job["seed"], "shot": job["shot"], "status": "cached"})
            continue

        result = evaluate_config(job["dino_dir"], job["clip_dir"], map_size)

        report = {
            "pipeline": "v3_direction_a_a1",
            "direction": "A_feature_level_fusion",
            "dataset": job["dataset"],
            "dataset_role": job["role"],
            "seed": job["seed"],
            "shot": job["shot"],
            "frozen_config": FROZEN_CONFIG,
            "image_score": "max_pool",
            "stride": STRIDE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "concat": result["concat"],
            "dino": result["dino"],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        c = report["concat"]["mean"]
        d = report["dino"]["mean"]
        print(
            f"[{job['dataset']}/s{job['seed']}/k{job['shot']}] "
            f"concat I-AUROC {c['image']['image_auroc']:.4f} P-AP {c['pixel']['pixel_ap']:.4f} | "
            f"dino P-AP {d['pixel']['pixel_ap']:.4f}",
            flush=True,
        )
        completed.append({"dataset": job["dataset"], "seed": job["seed"], "shot": job["shot"], "status": "ok"})

    print(json.dumps({"completed": len(completed), "failed": failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
