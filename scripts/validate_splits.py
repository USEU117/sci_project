"""Validate deterministic nested few-shot manifests.

The validator uses only the Python standard library so it can run before any
method-specific environment is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"ERROR: {message}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--checksum", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    manifest_path = args.manifest.resolve()
    checksum_path = (
        args.checksum.resolve()
        if args.checksum
        else manifest_path.with_name("manifest.sha256")
    )
    errors: list[str] = []

    if not manifest_path.is_file():
        raise SystemExit(f"Manifest does not exist: {manifest_path}")
    raw = manifest_path.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    if not checksum_path.is_file():
        add_error(errors, f"missing checksum file: {checksum_path}")
        expected_digest = None
    else:
        fields = checksum_path.read_text(encoding="ascii").split()
        expected_digest = fields[0].lower() if fields else ""
        if expected_digest != actual_digest:
            add_error(
                errors,
                f"checksum mismatch: expected {expected_digest}, actual {actual_digest}",
            )

    manifest = json.loads(raw.decode("utf-8"))
    root = Path(manifest["root"])
    shots = sorted(int(value) for value in manifest["shots"])
    seeds = sorted(int(value) for value in manifest["seeds"])
    categories = manifest.get("categories", {})
    selected_hashes = manifest.get("selected_file_sha256", {})
    if not root.is_dir():
        add_error(errors, f"dataset root does not exist: {root}")
    if not categories:
        add_error(errors, "manifest has no categories")
    if shots != [1, 2, 4]:
        add_error(errors, f"expected shots [1, 2, 4], got {shots}")
    if seeds != [0, 1, 2]:
        add_error(errors, f"expected seeds [0, 1, 2], got {seeds}")
    if manifest.get("nested") is not True:
        add_error(errors, "nested must be true")

    checked_files = 0
    for category, seed_map in sorted(categories.items()):
        for seed in seeds:
            key = str(seed)
            if key not in seed_map:
                add_error(errors, f"{category}: missing seed {seed}")
                continue
            selections = seed_map[key]
            previous: set[str] = set()
            for shot in shots:
                shot_key = str(shot)
                paths = selections.get(shot_key)
                if not isinstance(paths, list):
                    add_error(errors, f"{category}/seed={seed}: missing {shot}-shot list")
                    continue
                if len(paths) != shot:
                    add_error(
                        errors,
                        f"{category}/seed={seed}/{shot}-shot: "
                        f"expected {shot} paths, got {len(paths)}",
                    )
                if len(set(paths)) != len(paths):
                    add_error(
                        errors,
                        f"{category}/seed={seed}/{shot}-shot: duplicate paths",
                    )
                current = set(paths)
                if not previous.issubset(current):
                    add_error(
                        errors,
                        f"{category}/seed={seed}: {shot}-shot is not nested",
                    )
                previous = current
                for relative in paths:
                    posix = PurePosixPath(relative)
                    if posix.is_absolute() or ".." in posix.parts:
                        add_error(errors, f"unsafe path: {relative}")
                        continue
                    if posix.parts and posix.parts[0] != category:
                        add_error(
                            errors,
                            f"{category}/seed={seed}: path belongs to another category: "
                            f"{relative}",
                        )
                    path = root.joinpath(*posix.parts)
                    if not path.is_file():
                        add_error(errors, f"missing reference image: {relative}")
                    elif manifest.get("dataset") in {"mpdd", "btad"}:
                        expected_file_hash = selected_hashes.get(relative)
                        if not expected_file_hash:
                            add_error(errors, f"missing selected file SHA256: {relative}")
                        else:
                            actual_file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                            if actual_file_hash != expected_file_hash:
                                add_error(errors, f"selected file SHA256 mismatch: {relative}")
                    checked_files += 1

    report = {
        "manifest": str(manifest_path),
        "dataset": manifest.get("dataset"),
        "root": str(root),
        "sha256": actual_digest,
        "checksum_match": expected_digest == actual_digest,
        "category_count": len(categories),
        "shots": shots,
        "seeds": seeds,
        "checked_path_entries": checked_files,
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
