"""Complete CLIP-image-tower-only KNN controls for the frozen A1 study.

This is a post-freeze control. It does not modify A1 or select any parameter.
It reuses the exact AnomalyCLIP image-tower patch caches and normal references
already used by A1, then applies the same L2 -> FAISS IndexFlatL2(k=1) ->
distance/2 -> dists2map(448) pipeline. No text feature is used.

The implementation aligns CLIP test sample IDs to the DINO/evaluator order and
performs FAISS search in chunks to bound RAM. It reports the same six metrics as
the final A1 table: image AUROC/AP/F1-max and pixel AUROC/AP/AUPRO.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from evaluate_a1_complete_metrics import (  # noqa: E402
    DATASETS,
    FEATURES_ROOT,
    compute_image_metrics,
)
from evaluate_a1_feature_fusion import STRIDE, compute_metrics, load_features  # noqa: E402
from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from src.utils import dists2map  # noqa: E402

SEEDS = (0, 1, 2)
SHOTS = (1, 2, 4)
DEFAULT_DATASETS = ("btad", "mvtec")
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "dynamic_fusion"
    / "v3_direction_a"
    / "clip_only_controls_20260830"
)

METHOD_DEFINITION = (
    "AnomalyCLIP ViT-L/14@336 image tower only; no text; frozen patch features; "
    "native CLIP grid; branch L2; FAISS IndexFlatL2 k=1; distance/2; "
    "dists2map 448; stride=8; image score=max-pool"
)


def clip_maps_chunked(
    dino: dict,
    clip: dict,
    map_size: tuple[int, int],
    chunk: int,
) -> tuple[np.ndarray, int]:
    """Return CLIP-only maps in DINO/evaluator sample order and reorder count."""
    alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])
    order = np.asarray(alignment.candidate_order, dtype=np.int64)
    reorder_count = int(np.count_nonzero(order != np.arange(order.size)))

    raw_feat = clip["patch_features"]
    feat = raw_feat if reorder_count == 0 else raw_feat[order]
    ref = np.asarray(clip["ref_patch_features"], dtype=np.float32)
    grid = tuple(int(v) for v in clip["grid_size"])

    ref_flat = np.ascontiguousarray(ref.reshape(-1, ref.shape[-1]), dtype=np.float32)
    faiss.normalize_L2(ref_flat)
    index = faiss.IndexFlatL2(ref_flat.shape[1])
    index.add(ref_flat)

    feat_flat = feat.reshape(-1, feat.shape[-1])
    distances = np.empty(feat_flat.shape[0], dtype=np.float32)
    for start in range(0, feat_flat.shape[0], chunk):
        end = min(start + chunk, feat_flat.shape[0])
        block = np.ascontiguousarray(feat_flat[start:end], dtype=np.float32)
        faiss.normalize_L2(block)
        dd, _ = index.search(block, k=1)
        distances[start:end] = dd[:, 0]

    patch_maps = (distances / 2.0).reshape(feat.shape[0], *grid)
    maps = np.stack([dists2map(d, map_size) for d in patch_maps]).astype(np.float32)
    return maps, reorder_count


def mean_metrics(rows: list[dict]) -> dict:
    return {
        "image": {
            key: round(float(np.mean([row["image"][key] for row in rows])), 6)
            for key in ("image_auroc", "image_ap", "image_f1_max")
        },
        "pixel": {
            key: round(float(np.mean([row["pixel"][key] for row in rows])), 6)
            for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
        },
    }


def evaluate_config(
    dino_dir: Path,
    clip_dir: Path,
    map_size: tuple[int, int],
    chunk: int,
) -> dict:
    rows = []
    for dino_path in sorted(dino_dir.glob("*.npz")):
        category = dino_path.stem
        clip_path = clip_dir / f"{category}.npz"
        if not clip_path.is_file():
            raise FileNotFoundError(f"Missing CLIP features: {clip_path}")

        dino = load_features(dino_path)
        clip = load_features(clip_path)
        if set(dino["sample_ids"].tolist()) != set(clip["sample_ids"].tolist()):
            raise ValueError(f"Sample-ID set mismatch for {category}")

        maps, reorder_count = clip_maps_chunked(dino, clip, map_size, chunk)
        rows.append(
            {
                "category": category,
                "n_images": int(maps.shape[0]),
                "clip_grid": list(clip["grid_size"]),
                "sample_reorder_count": reorder_count,
                "image": compute_image_metrics(dino["gt_sp"], maps),
                "pixel": compute_metrics(maps.astype(np.float64), dino["imgs_masks"]),
            }
        )
        print(
            f"  {category}: P-AP={rows[-1]['pixel']['pixel_ap']:.4f} "
            f"I-AUROC={rows[-1]['image']['image_auroc']:.4f}",
            flush=True,
        )

    return {"mean": mean_metrics(rows), "per_category": rows}


def existing_mvtec_pixel_crosscheck(seed: int, shot: int, report: dict) -> dict | None:
    path = (
        ROOT
        / "experiments"
        / "dynamic_fusion"
        / "v3_direction_a"
        / "a1_mvtec_20260818"
        / f"seed{seed}_k{shot}"
        / "clip_pca0_whiten0_w0.5_report.json"
    )
    if not path.is_file():
        return None
    old = json.loads(path.read_text(encoding="utf-8"))
    current = report["clip_only"]["mean"]["pixel"]
    previous = old["mean_fused"]
    deltas = {
        key: round(current[key] - previous[key], 9)
        for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
    }
    return {
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "deltas": deltas,
        "within_5e-6": all(abs(v) <= 5e-6 for v in deltas.values()),
    }


def build_jobs(datasets: list[str], seeds: list[int], shots: list[int]) -> list[dict]:
    jobs = []
    for dataset in datasets:
        meta = DATASETS[dataset]
        for seed in seeds:
            for shot in shots:
                jobs.append(
                    {
                        "dataset": dataset,
                        "role": meta["role"],
                        "seed": seed,
                        "shot": shot,
                        "dino_dir": FEATURES_ROOT
                        / meta["dino_dir_fmt"].format(seed=seed, shot=shot)
                        / "anomalydino_visual",
                        "clip_dir": FEATURES_ROOT
                        / meta["clip_dir_fmt"].format(seed=seed, shot=shot)
                        / "anomalyclip_text",
                    }
                )
    return jobs


def validate_jobs(jobs: list[dict]) -> None:
    problems = []
    for job in jobs:
        dino_files = sorted(p.stem for p in job["dino_dir"].glob("*.npz"))
        clip_files = sorted(p.stem for p in job["clip_dir"].glob("*.npz"))
        if not dino_files or dino_files != clip_files:
            problems.append(
                f"{job['dataset']} s{job['seed']} k{job['shot']}: "
                f"DINO={dino_files}, CLIP={clip_files}"
            )
    if problems:
        raise SystemExit("Input validation failed:\n  " + "\n  ".join(problems))


def write_summary(output_root: Path, jobs: list[dict]) -> None:
    config_rows = []
    metric_names = (
        "image_auroc",
        "image_ap",
        "image_f1_max",
        "pixel_auroc",
        "pixel_ap",
        "pixel_aupro",
    )
    for job in jobs:
        path = output_root / job["dataset"] / f"seed{job['seed']}_k{job['shot']}" / "report.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        mean = report["clip_only"]["mean"]
        config_rows.append(
            {
                "dataset": job["dataset"],
                "role": job["role"],
                "seed": job["seed"],
                "shot": job["shot"],
                **mean["image"],
                **mean["pixel"],
            }
        )

    dataset_rows = []
    for dataset in sorted({row["dataset"] for row in config_rows}):
        rows = [row for row in config_rows if row["dataset"] == dataset]
        aggregate = {
            "dataset": dataset,
            "role": rows[0]["role"],
            "n_configs": len(rows),
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            aggregate[f"{metric}_mean"] = round(float(values.mean()), 6)
            aggregate[f"{metric}_std"] = round(float(values.std()), 6)
        dataset_rows.append(aggregate)

    payload = {
        "schema_version": 1,
        "kind": "clip_image_tower_only_complete_metrics",
        "method_definition": METHOD_DEFINITION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_rows": config_rows,
        "dataset_rows": dataset_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if config_rows:
        with (output_root / "per_config.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(config_rows[0]))
            writer.writeheader()
            writer.writerows(config_rows)
    if dataset_rows:
        with (output_root / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0]))
            writer.writeheader()
            writer.writerows(dataset_rows)

    lines = [
        "# CLIP image-tower-only complete-metric controls",
        "",
        f"- Method: {METHOD_DEFINITION}",
        "- This is a post-freeze control and does not change A1 selection.",
        "",
        "| dataset | configs | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        cells = [row["dataset"], str(row["n_configs"])]
        for metric in metric_names:
            cells.append(f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Note: 3 seeds x 3 shots are reference-sampling configurations on shared test sets, not independent datasets.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.datasets, args.seeds, args.shots)
    validate_jobs(jobs)
    if args.validate_only:
        print(json.dumps({"status": "passed", "jobs": len(jobs)}))
        return 0

    for job in jobs:
        out_dir = args.output_root / job["dataset"] / f"seed{job['seed']}_k{job['shot']}"
        marker = out_dir / "report.json"
        if marker.is_file():
            print(f"[cached] {job['dataset']} s{job['seed']} k{job['shot']}", flush=True)
            continue

        print(f"[{job['dataset']} s{job['seed']} k{job['shot']}]", flush=True)
        result = evaluate_config(
            job["dino_dir"],
            job["clip_dir"],
            (args.map_size, args.map_size),
            args.chunk,
        )
        report = {
            "schema_version": 1,
            "kind": "clip_image_tower_only_complete_metrics",
            "dataset": job["dataset"],
            "dataset_role": job["role"],
            "seed": job["seed"],
            "shot": job["shot"],
            "method_definition": METHOD_DEFINITION,
            "stride": STRIDE,
            "map_size": args.map_size,
            "chunk": args.chunk,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "clip_only": result,
        }
        if job["dataset"] == "mvtec":
            report["existing_pixel_report_crosscheck"] = existing_mvtec_pixel_crosscheck(
                job["seed"], job["shot"], report
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        mean = result["mean"]
        print(
            f"  mean: P-AP={mean['pixel']['pixel_ap']:.4f}, "
            f"I-AUROC={mean['image']['image_auroc']:.4f}",
            flush=True,
        )

    write_summary(args.output_root, jobs)
    print(json.dumps({"status": "complete", "jobs": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
