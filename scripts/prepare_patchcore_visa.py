"""Convert official VisA metadata into a minimal PatchCore/MVTec layout.

This is an adapter for a VisA engineering baseline. It never modifies the
official VisA tree; image and mask files are copied into a separate ignored
directory understood by PatchCore's MVTec loader.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

def copy_entries(source: Path, target: Path, entries: list[dict], split: str) -> int:
    copied = 0
    for entry in entries:
        image = source / entry["img_path"]
        category = (
            "good"
            if entry["anomaly"] == 0
            else (entry["specie_name"] or "anomaly")
        )
        destination = target / split / category / image.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
        if split == "test" and entry["anomaly"] == 1:
            mask = source / entry["mask_path"]
            mask_destination = (
                target / "ground_truth" / category / f"{image.stem}.png"
            )
            mask_destination.parent.mkdir(parents=True, exist_ok=True)
            mask_array = np.asarray(Image.open(mask).convert("L"))
            Image.fromarray((mask_array > 0).astype(np.uint8) * 255).save(
                mask_destination
            )
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    args = parser.parse_args()

    meta = json.loads((args.source / "meta.json").read_text(encoding="utf-8"))
    total = 0
    for category in args.categories:
        if category not in meta["train"] or category not in meta["test"]:
            raise KeyError(f"missing category in meta.json: {category}")
        class_root = args.target / category
        total += copy_entries(args.source, class_root, meta["train"][category], "train")
        total += copy_entries(args.source, class_root, meta["test"][category], "test")
    print(f"prepared {len(args.categories)} categories and {total} images at {args.target}")


if __name__ == "__main__":
    main()
