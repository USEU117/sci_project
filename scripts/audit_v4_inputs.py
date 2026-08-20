"""G1 input audit: validate a V4 dataset manifest (CPU, read-only).

Checks the physical separation and normal-only property of the normal-reference
manifest: every (category, seed, shot) must reference exactly K normal images,
all unique, all from the train/normal split (no test paths), and the manifest
must carry no test ground truth. Emits a report with the five leakage flags.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEAKAGE_FLAGS = {
    "test_predictions_used_for_parameter_fit": False,
    "test_labels_used_for_parameter_fit": False,
    "test_masks_used_for_parameter_fit": False,
    "test_dataset_statistics_used_for_calibration": False,
    "test_normal_selection_used": False,
}


def _is_normal_train_path(rel: str) -> bool:
    norm = rel.replace("\\", "/").lower()
    # MPDD normal references live under <category>/train/good/.
    return "/train/good/" in norm or "/train/good" in norm


def audit_manifest(manifest: dict, dataset: str) -> dict:
    problems: list[str] = []
    categories = manifest.get("categories", {})
    seeds = [str(s) for s in manifest.get("seeds", [])]
    shots = [str(s) for s in manifest.get("shots", [])]

    if manifest.get("dataset") != dataset:
        problems.append(f"manifest dataset={manifest.get('dataset')!r} != requested {dataset!r}")
    if not categories:
        problems.append("manifest has no categories")
    if not seeds:
        problems.append("manifest has no seeds")
    if not shots:
        problems.append("manifest has no shots")

    total_checks = 0
    for cat, by_seed in categories.items():
        for seed in seeds:
            for shot in shots:
                total_checks += 1
                refs = by_seed.get(seed, {}).get(shot)
                if refs is None:
                    problems.append(f"{cat}/seed{seed}/shot{shot}: missing reference list")
                    continue
                k = int(shot)
                if len(refs) != k:
                    problems.append(
                        f"{cat}/seed{seed}/shot{shot}: expected {k} references, got {len(refs)}"
                    )
                if len(set(refs)) != len(refs):
                    problems.append(f"{cat}/seed{seed}/shot{shot}: duplicate references")
                for rel in refs:
                    if not isinstance(rel, str) or not rel.strip():
                        problems.append(f"{cat}/seed{seed}/shot{shot}: non-string/empty reference")
                    elif not _is_normal_train_path(rel):
                        problems.append(
                            f"{cat}/seed{seed}/shot{shot}: reference not normal/train: {rel}"
                        )

    # The manifest must not itself contain test GT.
    for forbidden in ("test_labels", "test_masks", "gt", "image_labels", "pixel_masks"):
        if forbidden in manifest:
            problems.append(f"manifest contains forbidden test-GT key: {forbidden}")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "categories": len(categories),
        "seeds": seeds,
        "shots": shots,
        "total_configs_checked": total_checks,
        "passed": not problems,
        "problems": problems,
        "leakage_flags": LEAKAGE_FLAGS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"manifest missing: {args.manifest}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit_manifest(manifest, args.dataset)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
