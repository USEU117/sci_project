"""Audit two frozen prediction directories without running either baseline.

This is a development-time data-integrity tool.  Ground truth labels may be
checked here, but they are never returned to or accepted by the fusion router.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion.alignment import build_alignment_plan


REQUIRED_KEYS = {"gt_sp", "pr_sp", "imgs_masks", "anomaly_maps", "sample_ids"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npz_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}


def npz_array_shape(path: Path, key: str) -> tuple[int, ...]:
    """Read a .npy member header without materializing the full array."""

    with zipfile.ZipFile(path) as archive, archive.open(f"{key}.npy") as member:
        version = npy_format.read_magic(member)
        if version == (1, 0):
            shape, _, _ = npy_format.read_array_header_1_0(member)
        else:
            shape, _, _ = npy_format.read_array_header_2_0(member)
    return tuple(int(value) for value in shape)


def normalized_map_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 4 and shape[1] == 1:
        return shape[0], shape[2], shape[3]
    if len(shape) == 3:
        return shape
    raise ValueError(f"expected [N,H,W] or [N,1,H,W], got {shape}")


def load_small_arrays(
    path: Path, sidecar: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["gt_sp"]).reshape(-1)
        sample_ids = (
            np.asarray(data["sample_ids"]) if "sample_ids" in data.files else None
        )
    if sample_ids is None:
        if sidecar is None:
            raise ValueError(f"{path}: sample_ids missing and no sidecar supplied")
        with np.load(sidecar, allow_pickle=False) as data:
            if set(data.files) != {"sample_ids"}:
                raise ValueError(f"{sidecar}: expected only sample_ids")
            sample_ids = np.asarray(data["sample_ids"])
    if len(sample_ids) != len(labels):
        raise ValueError(f"{path}: sidecar sample count differs from gt_sp")
    return sample_ids, labels


def audit_pair(
    category: str,
    visual_path: Path,
    text_path: Path,
    include_sha256: bool,
    visual_sidecar: Path | None = None,
    text_sidecar: Path | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "category": category,
        "visual_cache": str(visual_path.resolve()),
        "text_cache": str(text_path.resolve()),
        "visual_bytes": visual_path.stat().st_size,
        "text_bytes": text_path.stat().st_size,
        "status": "failed",
        "errors": [],
    }
    if visual_sidecar is not None:
        result["visual_sidecar"] = str(visual_sidecar.resolve())
    if text_sidecar is not None:
        result["text_sidecar"] = str(text_sidecar.resolve())
    try:
        visual_keys = npz_members(visual_path)
        text_keys = npz_members(text_path)
        visual_required = REQUIRED_KEYS - ({"sample_ids"} if visual_sidecar else set())
        text_required = REQUIRED_KEYS - ({"sample_ids"} if text_sidecar else set())
        visual_missing = sorted(visual_required - visual_keys)
        text_missing = sorted(text_required - text_keys)
        if visual_missing or text_missing:
            raise ValueError(
                f"missing keys: visual={visual_missing}, text={text_missing}"
            )

        visual_ids, visual_labels = load_small_arrays(visual_path, visual_sidecar)
        text_ids, text_labels = load_small_arrays(text_path, text_sidecar)
        plan = build_alignment_plan(visual_ids, text_ids)
        text_labels = text_labels[plan.candidate_order]

        visual_map_shape = normalized_map_shape(
            npz_array_shape(visual_path, "anomaly_maps")
        )
        text_map_shape = normalized_map_shape(npz_array_shape(text_path, "anomaly_maps"))
        if visual_map_shape[0] != len(plan.reference_ids):
            raise ValueError("visual anomaly-map N does not match sample_ids")
        if text_map_shape[0] != len(plan.reference_ids):
            raise ValueError("text anomaly-map N does not match sample_ids")
        if not np.array_equal(visual_labels, text_labels):
            mismatch_count = int(np.count_nonzero(visual_labels != text_labels))
            raise ValueError(f"labels differ after alignment: {mismatch_count} samples")

        result.update(
            {
                "samples": len(plan.reference_ids),
                "order_already_equal": plan.order_already_equal,
                "labels_equal": True,
                "visual_map_shape": list(visual_map_shape),
                "text_map_shape": list(text_map_shape),
                "resize_required": visual_map_shape[1:] != text_map_shape[1:],
                "status": "passed",
            }
        )
        if include_sha256:
            result["visual_sha256"] = sha256(visual_path)
            result["text_sha256"] = sha256(text_path)
            if visual_sidecar is not None:
                result["visual_sidecar_sha256"] = sha256(visual_sidecar)
            if text_sidecar is not None:
                result["text_sidecar_sha256"] = sha256(text_sidecar)
    except Exception as exc:  # Report every category before returning a failure code.
        result["errors"] = [str(exc)]
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "category",
        "status",
        "samples",
        "order_already_equal",
        "labels_equal",
        "visual_map_shape",
        "text_map_shape",
        "resize_required",
        "visual_bytes",
        "text_bytes",
        "visual_sha256",
        "text_sha256",
        "visual_sidecar",
        "text_sidecar",
        "visual_sidecar_sha256",
        "text_sidecar_sha256",
        "errors",
        "visual_cache",
        "text_cache",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "visual_map_shape": json.dumps(
                        row.get("visual_map_shape"), ensure_ascii=False
                    ),
                    "text_map_shape": json.dumps(
                        row.get("text_map_shape"), ensure_ascii=False
                    ),
                    "errors": " | ".join(row.get("errors", [])),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--visual-sidecar-dir", type=Path)
    parser.add_argument("--text-sidecar-dir", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, required=True)
    parser.add_argument("--expected-categories", type=int, default=12)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()

    visual = {path.stem: path for path in args.visual_dir.glob("*.npz")}
    text = {path.stem: path for path in args.text_dir.glob("*.npz")}
    categories = sorted(set(visual) | set(text))
    rows: list[dict[str, object]] = []
    for category in categories:
        if category not in visual or category not in text:
            rows.append(
                {
                    "category": category,
                    "status": "failed",
                    "errors": [
                        f"cache missing: visual={category in visual}, "
                        f"text={category in text}"
                    ],
                }
            )
            continue
        visual_sidecar = (
            args.visual_sidecar_dir / f"{category}.sample_ids.npz"
            if args.visual_sidecar_dir
            else None
        )
        text_sidecar = (
            args.text_sidecar_dir / f"{category}.sample_ids.npz"
            if args.text_sidecar_dir
            else None
        )
        rows.append(
            audit_pair(
                category,
                visual[category],
                text[category],
                args.sha256,
                visual_sidecar,
                text_sidecar,
            )
        )

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visual_dir": str(args.visual_dir.resolve()),
        "text_dir": str(args.text_dir.resolve()),
        "expected_categories": args.expected_categories,
        "categories_found": len(categories),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] != "passed" for row in rows),
        "all_passed": (
            len(categories) == args.expected_categories
            and all(row["status"] == "passed" for row in rows)
        ),
        "categories": rows,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.report_csv, rows)
    print(json.dumps({key: summary[key] for key in (
        "categories_found", "passed", "failed", "all_passed"
    )}, ensure_ascii=False))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
