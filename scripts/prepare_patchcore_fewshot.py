"""Build a PatchCore/MVTec view from a unified few-shot manifest.

Files are hard-linked when possible, so creating many shot/seed views does not
duplicate the VisA image data. Existing matching files are kept, making the
operation resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path, PurePosixPath


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"target exists but differs: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file():
            link_or_copy(path, destination / path.relative_to(source))
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--shot", type=int, choices=[1, 2, 4], required=True)
    ap.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    category_map = manifest["categories"].get(args.category)
    if category_map is None:
        raise SystemExit(f"category not found in manifest: {args.category}")
    selected = category_map[str(args.seed)][str(args.shot)]

    source_class = args.source.resolve() / args.category
    target_class = args.target.resolve() / args.category
    if not source_class.is_dir():
        raise SystemExit(f"source class does not exist: {source_class}")

    train_count = 0
    selected_names: list[str] = []
    for relative in selected:
        posix = PurePosixPath(relative)
        if posix.parts[0] != args.category:
            raise SystemExit(f"manifest path belongs to another category: {relative}")
        filename = posix.name
        source_image = source_class / "train" / "good" / filename
        if not source_image.is_file():
            raise SystemExit(f"mapped PatchCore training image is missing: {source_image}")
        link_or_copy(source_image, target_class / "train" / "good" / filename)
        selected_names.append(filename)
        train_count += 1

    test_count = link_tree(source_class / "test", target_class / "test")
    mask_count = link_tree(
        source_class / "ground_truth", target_class / "ground_truth"
    )
    metadata = {
        "dataset": manifest["dataset"],
        "category": args.category,
        "shot": args.shot,
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": __import__("hashlib").sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "selected_reference_images": selected,
        "patchcore_train_filenames": selected_names,
        "train_count": train_count,
        "test_count": test_count,
        "mask_count": mask_count,
    }
    (target_class / "fewshot_selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
