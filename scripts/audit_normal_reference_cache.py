"""Audit normal-reference prediction caches against the frozen manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import NormalReferencePrediction
from industrial_ad.fusion.alignment import canonicalize_sample_ids


REQUIRED_KEYS = {
    "sample_ids",
    "source_ids",
    "augmentation_ids",
    "image_scores",
    "pixel_maps",
    "dataset",
    "branch",
    "category",
    "seed",
    "shot",
    "score_direction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar")
    return str(value.reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--min-views-per-source", type=int, default=4)
    parser.add_argument("--expected-categories", type=int, default=12)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_categories = manifest["categories"]
    rows: list[dict[str, object]] = []
    for category in sorted(expected_categories):
        path = args.cache_dir / f"{category}.npz"
        row: dict[str, object] = {
            "category": category,
            "cache": str(path.resolve()),
            "status": "failed",
            "errors": [],
        }
        try:
            if not path.is_file():
                raise ValueError(f"cache missing: {path}")
            with np.load(path, allow_pickle=False) as data:
                missing = REQUIRED_KEYS.difference(data.files)
                if missing:
                    raise ValueError(f"missing arrays: {sorted(missing)}")
                if scalar(data, "dataset") != args.dataset:
                    raise ValueError("dataset metadata differs")
                if scalar(data, "branch") != args.branch:
                    raise ValueError("branch metadata differs")
                if scalar(data, "category") != category:
                    raise ValueError("category metadata differs")
                if int(scalar(data, "seed")) != args.seed:
                    raise ValueError("seed metadata differs")
                if int(scalar(data, "shot")) != args.shot:
                    raise ValueError("shot metadata differs")
                if scalar(data, "score_direction") != "higher_is_more_anomalous":
                    raise ValueError("score direction must be higher_is_more_anomalous")
                cache = NormalReferencePrediction(
                    sample_ids=data["sample_ids"],
                    source_ids=data["source_ids"],
                    augmentation_ids=data["augmentation_ids"],
                    image_scores=data["image_scores"],
                    pixel_maps=data["pixel_maps"],
                )
                cache.validate(args.min_views_per_source)

            expected_paths = expected_categories[category][str(args.seed)][str(args.shot)]
            expected_sources = set(
                canonicalize_sample_ids(np.asarray(expected_paths)).tolist()
            )
            actual_sources = set(
                canonicalize_sample_ids(np.asarray(cache.source_ids)).tolist()
            )
            if actual_sources != expected_sources:
                raise ValueError(
                    "source_ids differ from manifest: "
                    f"missing={sorted(expected_sources - actual_sources)}, "
                    f"extra={sorted(actual_sources - expected_sources)}"
                )
            maps = np.asarray(cache.pixel_maps)
            if maps.ndim == 4 and maps.shape[1] == 1:
                maps = maps[:, 0]
            row.update(
                {
                    "status": "passed",
                    "sources": cache.source_count,
                    "views": len(cache.sample_ids),
                    "height": int(maps.shape[1]),
                    "width": int(maps.shape[2]),
                    "image_score_min": float(np.min(cache.image_scores)),
                    "image_score_max": float(np.max(cache.image_scores)),
                    "cache_sha256": sha256(path),
                }
            )
        except Exception as exc:
            row["errors"] = [str(exc)]
        rows.append(row)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "branch": args.branch,
        "seed": args.seed,
        "shot": args.shot,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "min_views_per_source": args.min_views_per_source,
        "expected_categories": args.expected_categories,
        "categories_found": len(rows),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] != "passed" for row in rows),
        "all_passed": (
            len(rows) == args.expected_categories
            and all(row["status"] == "passed" for row in rows)
        ),
        "test_predictions_used": False,
        "test_labels_used": False,
        "categories": rows,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "category",
        "status",
        "sources",
        "views",
        "height",
        "width",
        "image_score_min",
        "image_score_max",
        "cache_sha256",
        "errors",
        "cache",
    ]
    with args.report_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "errors": " | ".join(row.get("errors", []))})
    print(
        json.dumps(
            {
                "categories_found": report["categories_found"],
                "passed": report["passed"],
                "failed": report["failed"],
                "all_passed": report["all_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
