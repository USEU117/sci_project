"""Verify that frozen test sample IDs do not change across shot settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion.alignment import build_alignment_plan


def parse_shot_dirs(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected SHOT=PATH, got {value!r}")
        shot_text, path_text = value.split("=", 1)
        shot = int(shot_text)
        if shot in result:
            raise ValueError(f"duplicate shot {shot}")
        result[shot] = Path(path_text)
    return result


def load_ids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "sample_ids" not in data.files:
            raise ValueError(f"{path}: sample_ids missing")
        return np.asarray(data["sample_ids"])


def digest(values: np.ndarray) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="append", required=True, metavar="SHOT=PATH")
    parser.add_argument("--text", action="append", required=True, metavar="SHOT=PATH")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--expected-categories", type=int, default=12)
    args = parser.parse_args()

    visual_dirs = parse_shot_dirs(args.visual)
    text_dirs = parse_shot_dirs(args.text)
    expected_shots = sorted(args.expected_shots)
    errors: list[str] = []
    if sorted(visual_dirs) != expected_shots:
        errors.append(f"visual shots are {sorted(visual_dirs)}, expected {expected_shots}")
    if sorted(text_dirs) != expected_shots:
        errors.append(f"text shots are {sorted(text_dirs)}, expected {expected_shots}")

    per_shot_files: dict[int, tuple[dict[str, Path], dict[str, Path]]] = {}
    categories: set[str] = set()
    for shot in expected_shots:
        if shot not in visual_dirs or shot not in text_dirs:
            per_shot_files[shot] = ({}, {})
            continue
        visual_files = {
            path.stem: path for path in visual_dirs[shot].glob("*.npz")
        }
        text_files = {
            path.stem: path for path in text_dirs[shot].glob("*.npz")
        }
        per_shot_files[shot] = (visual_files, text_files)
        categories.update(visual_files)
        categories.update(text_files)

    rows: list[dict[str, object]] = []
    for category in sorted(categories):
        row: dict[str, object] = {"category": category, "shots": {}, "status": "failed"}
        category_errors: list[str] = []
        reference_ids: np.ndarray | None = None
        for shot in expected_shots:
            visual_files, text_files = per_shot_files[shot]
            if category not in visual_files or category not in text_files:
                category_errors.append(
                    f"shot {shot} missing visual={category in visual_files}, "
                    f"text={category in text_files}"
                )
                continue
            try:
                visual_ids = load_ids(visual_files[category])
                text_ids = load_ids(text_files[category])
                pair_plan = build_alignment_plan(visual_ids, text_ids)
                if reference_ids is None:
                    reference_ids = pair_plan.reference_ids
                elif not np.array_equal(reference_ids, pair_plan.reference_ids):
                    category_errors.append(
                        f"shot {shot} test order differs from shot {expected_shots[0]}"
                    )
                row["shots"][str(shot)] = {
                    "samples": len(pair_plan.reference_ids),
                    "visual_text_order_equal": pair_plan.order_already_equal,
                    "sample_id_sha256": digest(pair_plan.reference_ids),
                }
            except Exception as exc:
                category_errors.append(f"shot {shot}: {exc}")
        row["errors"] = category_errors
        row["status"] = "passed" if not category_errors else "failed"
        rows.append(row)

    all_passed = (
        not errors
        and len(categories) == args.expected_categories
        and all(row["status"] == "passed" for row in rows)
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_shots": expected_shots,
        "expected_categories": args.expected_categories,
        "categories_found": len(categories),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] != "passed" for row in rows),
        "global_errors": errors,
        "all_passed": all_passed,
        "categories": rows,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
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
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
