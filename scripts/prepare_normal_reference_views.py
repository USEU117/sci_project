"""Prepare deterministic normal-reference views from the frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion.alignment import canonical_sample_id


VIEW_SPECS = [
    ("identity", "identity", 1.0),
    ("brightness_090", "brightness", 0.90),
    ("brightness_110", "brightness", 1.10),
    ("contrast_090", "contrast", 0.90),
    ("contrast_110", "contrast", 1.10),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(image: Image.Image, operation: str, factor: float) -> Image.Image:
    if operation == "identity":
        return image.copy()
    if operation == "brightness":
        return ImageEnhance.Brightness(image).enhance(factor)
    if operation == "contrast":
        return ImageEnhance.Contrast(image).enhance(factor)
    raise ValueError(f"unsupported operation: {operation}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["dataset"] != args.dataset:
        raise SystemExit("dataset differs from manifest")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for category, category_data in manifest["categories"].items():
        selected = category_data[str(args.seed)][str(args.shot)]
        for relative_path in selected:
            source_path = args.data_root / relative_path
            source_id = canonical_sample_id(relative_path)
            if not source_path.is_file():
                failures.append(f"missing source: {source_path}")
                continue
            source_hash = sha256(source_path)
            with Image.open(source_path) as opened:
                image = opened.convert("RGB")
                source_width, source_height = image.size
                for view_name, operation, factor in VIEW_SPECS:
                    sample_id = f"{source_id}::{view_name}"
                    output_path = (
                        args.output_dir
                        / "images"
                        / category
                        / source_id
                        / f"{view_name}.png"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    view = transform(image, operation, factor)
                    view.save(output_path, format="PNG", optimize=True)
                    with Image.open(output_path) as check:
                        if check.size != (source_width, source_height):
                            failures.append(f"shape changed: {output_path}")
                    rows.append(
                        {
                            "category": category,
                            "sample_id": sample_id,
                            "source_id": source_id,
                            "augmentation_id": view_name,
                            "operation": operation,
                            "factor": factor,
                            "source_path": str(source_path.resolve()),
                            "source_sha256": source_hash,
                            "view_path": str(output_path.resolve()),
                            "view_sha256": sha256(output_path),
                            "width": source_width,
                            "height": source_height,
                        }
                    )

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        failures.append("generated sample_ids are not unique")
    expected_views = len(manifest["categories"]) * args.shot * len(VIEW_SPECS)
    if len(rows) != expected_views:
        failures.append(f"generated {len(rows)} views, expected {expected_views}")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "shot": args.shot,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "data_root": str(args.data_root.resolve()),
        "view_specs": [
            {"augmentation_id": name, "operation": operation, "factor": factor}
            for name, operation, factor in VIEW_SPECS
        ],
        "normal_sources": len(set(str(row["source_id"]) for row in rows)),
        "views": len(rows),
        "test_images_used": False,
        "test_labels_used": False,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "items": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_views.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "normal_sources": report["normal_sources"],
                "views": report["views"],
                "failures": len(failures),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
