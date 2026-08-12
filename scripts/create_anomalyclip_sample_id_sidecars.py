"""Create auditable sample-ID sidecars for legacy AnomalyCLIP caches.

The legacy cache was produced by ``Dataset.data_all`` with a DataLoader using
``shuffle=False``.  Therefore each per-category cache follows the order stored
in ``meta.json``.  This script verifies labels and, optionally, every resized
mask before persisting the derived identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion.alignment import canonicalize_sample_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_mask(item: dict[str, object], data_root: Path, image_size: int) -> np.ndarray:
    if int(item["anomaly"]) == 0:
        return np.zeros((image_size, image_size), dtype=bool)
    mask_path = data_root / str(item["mask_path"])
    if not mask_path.is_file():
        raise ValueError(f"mask file missing: {mask_path}")
    with Image.open(mask_path) as image:
        # Match dataset.py exactly: convert every positive source pixel to 255
        # before torchvision's PIL resize and the later >0.5 tensor threshold.
        binary = (np.asarray(image.convert("L")) > 0).astype(np.uint8) * 255
        resized = Image.fromarray(binary, mode="L").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.uint8) > 127


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--verify-masks", action="store_true")
    parser.add_argument("--dataset-code", type=Path, required=True)
    parser.add_argument("--inference-code", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.meta_json.read_text(encoding="utf-8"))
    test_metadata = metadata["test"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for category, items in test_metadata.items():
        cache_path = args.cache_dir / f"{category}.npz"
        row: dict[str, object] = {
            "category": category,
            "cache": str(cache_path.resolve()),
            "status": "failed",
            "errors": [],
        }
        try:
            if not cache_path.is_file():
                raise ValueError(f"cache missing: {cache_path}")
            with np.load(cache_path, allow_pickle=False) as cache:
                required = {"gt_sp", "imgs_masks"}
                missing = required.difference(cache.files)
                if missing:
                    raise ValueError(f"cache missing keys: {sorted(missing)}")
                labels = np.asarray(cache["gt_sp"]).reshape(-1)
                masks = np.asarray(cache["imgs_masks"])

            expected_labels = np.asarray(
                [int(item["anomaly"]) for item in items], dtype=labels.dtype
            )
            if len(labels) != len(items):
                raise ValueError(
                    f"sample count differs: cache={len(labels)}, metadata={len(items)}"
                )
            if not np.array_equal(labels, expected_labels):
                mismatch = int(np.count_nonzero(labels != expected_labels))
                raise ValueError(f"metadata label order differs at {mismatch} samples")

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.shape != (len(items), args.image_size, args.image_size):
                raise ValueError(f"unexpected cached mask shape: {masks.shape}")

            mask_mismatches = 0
            if args.verify_masks:
                for index, item in enumerate(items):
                    expected = expected_mask(item, args.data_root, args.image_size)
                    if not np.array_equal(masks[index] > 0.5, expected):
                        mask_mismatches += 1
                if mask_mismatches:
                    raise ValueError(f"resized masks differ at {mask_mismatches} samples")

            sample_ids = canonicalize_sample_ids(
                np.asarray([str(item["img_path"]) for item in items])
            )
            if len(set(sample_ids.tolist())) != len(sample_ids):
                raise ValueError("derived sample_ids are not unique")

            sidecar_path = args.output_dir / f"{category}.sample_ids.npz"
            np.savez_compressed(sidecar_path, sample_ids=sample_ids)
            row.update(
                {
                    "samples": len(sample_ids),
                    "labels_equal": True,
                    "masks_verified": args.verify_masks,
                    "mask_mismatches": mask_mismatches,
                    "cache_sha256": sha256(cache_path),
                    "sidecar": str(sidecar_path.resolve()),
                    "sidecar_sha256": sha256(sidecar_path),
                    "sample_id_sha256": hashlib.sha256(
                        "\n".join(sample_ids.tolist()).encode("utf-8")
                    ).hexdigest(),
                    "status": "passed",
                }
            )
        except Exception as exc:
            row["errors"] = [str(exc)]
        rows.append(row)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(args.cache_dir.resolve()),
        "meta_json": str(args.meta_json.resolve()),
        "meta_sha256": sha256(args.meta_json),
        "dataset_code": str(args.dataset_code.resolve()),
        "dataset_code_sha256": sha256(args.dataset_code),
        "inference_code": str(args.inference_code.resolve()),
        "inference_code_sha256": sha256(args.inference_code),
        "ordering_evidence": [
            "dataset.py extends data_all in meta.json category/list order",
            "test.py DataLoader uses shuffle=False",
            "test.py appends per-category results in DataLoader order",
        ],
        "categories_found": len(rows),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] != "passed" for row in rows),
        "all_passed": bool(rows) and all(row["status"] == "passed" for row in rows),
        "categories": rows,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
