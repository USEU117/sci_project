"""Verify and safely extract MPDD/BTAD archives with provenance reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_CATEGORIES = {"mpdd": 6, "btad": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {member.filename}")
        if path.parts and ":" in path.parts[0]:
            raise ValueError(f"unsafe archive drive member: {member.filename}")
    return members


def looks_like_category(path: Path) -> bool:
    train = path / "train"
    test = path / "test"
    normal = (train / "good").is_dir() or (train / "ok").is_dir()
    return train.is_dir() and test.is_dir() and normal


def discover_dataset_root(destination: Path, expected_categories: int) -> Path:
    candidates = []
    for path in [destination, *destination.rglob("*")]:
        if not path.is_dir():
            continue
        category_count = sum(
            1 for child in path.iterdir() if child.is_dir() and looks_like_category(child)
        )
        if category_count == expected_categories:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one dataset root with {expected_categories} categories, "
            f"found {len(candidates)}: {[str(path) for path in candidates[:5]]}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mpdd", "btad"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-kind", choices=("official", "mirror"), required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"archive does not exist: {args.archive}")
    actual_hash = sha256(args.archive)
    if args.expected_sha256 and actual_hash.lower() != args.expected_sha256.lower():
        raise SystemExit(
            f"archive SHA256 mismatch: expected {args.expected_sha256}, actual {actual_hash}"
        )
    if args.destination.exists() and any(args.destination.iterdir()):
        raise SystemExit(f"destination must be absent or empty: {args.destination}")

    args.destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        members = safe_members(archive)
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise SystemExit(f"zip integrity failed at: {corrupt_member}")
        archive.extractall(args.destination, members=members)
    dataset_root = discover_dataset_root(
        args.destination, EXPECTED_CATEGORIES[args.dataset]
    )
    category_names = sorted(
        child.name
        for child in dataset_root.iterdir()
        if child.is_dir() and looks_like_category(child)
    )
    image_count = sum(
        1
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "archive": str(args.archive.resolve()),
        "archive_size_bytes": args.archive.stat().st_size,
        "archive_sha256": actual_hash,
        "expected_sha256": args.expected_sha256 or None,
        "source_url": args.source_url,
        "source_kind": args.source_kind,
        "zip_member_count": len(members),
        "destination": str(args.destination.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "category_count": len(category_names),
        "categories": category_names,
        "image_file_count": image_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
