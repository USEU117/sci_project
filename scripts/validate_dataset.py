"""Validate MVTec-style image/ground-truth layouts without third-party packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def stem_set(path: Path) -> set[str]:
    return {p.stem for p in files(path)}


def first_directory(*paths: Path) -> Path:
    return next((path for path in paths if path.is_dir()), paths[0])


def validate_category(category_dir: Path) -> dict:
    train_good = first_directory(
        category_dir / "train" / "good", category_dir / "train" / "ok"
    )
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"
    errors: list[str] = []
    if not train_good.is_dir():
        errors.append("missing train/good")
    if not test_dir.is_dir():
        errors.append("missing test")
    train_count = len(files(train_good)) if train_good.is_dir() else 0
    test_images = files(test_dir) if test_dir.is_dir() else []
    normal_names = {"good", "ok"}
    abnormal_images = [p for p in test_images if p.parent.name.lower() not in normal_names]
    missing_masks: list[str] = []
    if gt_dir.is_dir():
        masks = stem_set(gt_dir)
        for image in abnormal_images:
            # MVTec masks commonly use <image_stem>_mask.png.
            if image.stem not in masks and f"{image.stem}_mask" not in masks:
                missing_masks.append(image.relative_to(category_dir).as_posix())
    elif abnormal_images:
        errors.append("missing ground_truth")
    if missing_masks:
        errors.append(f"missing masks for {len(missing_masks)} abnormal images")
    return {
        "train_good": train_count,
        "test_images": len(test_images),
        "abnormal_test_images": len(abnormal_images),
        "mask_files": len(files(gt_dir)) if gt_dir.is_dir() else 0,
        "errors": errors,
        "missing_masks_examples": missing_masks[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", required=True, choices=["mvtec", "visa", "mpdd", "btad"]
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/logs"))
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    categories = sorted(p for p in root.iterdir() if p.is_dir())
    report = {
        "dataset": args.dataset,
        "root": str(root),
        "category_count": len(categories),
        "categories": {p.name: validate_category(p) for p in categories},
    }
    expected_categories = {"mvtec": 15, "visa": 12, "mpdd": 6, "btad": 3}
    report["expected_category_count"] = expected_categories[args.dataset]
    report["category_count_matches"] = (
        len(categories) == expected_categories[args.dataset]
    )
    report["error_count"] = sum(
        len(info["errors"]) for info in report["categories"].values()
    )
    if not report["category_count_matches"]:
        report["error_count"] += 1
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.dataset}_validation.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {destination}")
    if report["error_count"]:
        raise SystemExit(report["error_count"])


if __name__ == "__main__":
    main()
