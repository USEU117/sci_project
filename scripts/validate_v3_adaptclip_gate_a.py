"""Validate-only preflight for the bounded AdaptCLIP MPDD seed0/K1 Gate A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_CHECKPOINT_SHA256 = "777821da141eb57d159acef46868440faf773a2dd0acf5c276ec3f258c27edee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "schema_version": 1,
        "run_id": "v3_adaptclip_mpdd_s0_k1_gate_a_v1",
        "mode": "validate_only",
        "dataset": "mpdd",
        "dataset_role": "development",
        "seed": 0,
        "shot": 1,
        "gpu_started": False,
        "btad_accessed": False,
    }
    if not args.metadata_audit.is_file():
        result.update(status="blocked", blocker="metadata_audit_missing")
    else:
        audit = json.loads(args.metadata_audit.read_text(encoding="utf-8"))
        required = {
            "status": "passed", "dataset": "mpdd", "dataset_role": "development",
            "seed": 0, "shot": 1, "categories": 6, "test_images": 458,
            "anomalous_test_images": 282, "test_labels_used_by_router": False,
            "test_masks_used_by_router": False, "test_set_statistics_used_by_router": False,
            "btad_accessed": False,
        }
        mismatches = {key: {"expected": value, "actual": audit.get(key)} for key, value in required.items() if audit.get(key) != value}
        if mismatches:
            result.update(status="blocked", blocker="metadata_audit_mismatch", mismatches=mismatches)
        elif not args.checkpoint.is_file():
            result.update(
                status="blocked", blocker="official_checkpoint_missing",
                expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            )
        else:
            actual = sha256(args.checkpoint)
            if actual != EXPECTED_CHECKPOINT_SHA256:
                result.update(status="blocked", blocker="checkpoint_sha256_mismatch",
                              expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
                              actual_checkpoint_sha256=actual)
            else:
                result.update(
                    status="passed", checkpoint_sha256=actual,
                    next_safe_action="run exactly one MPDD seed0/K1 Gate A with batch_size=1 and metric computation disabled",
                    output_dir=str(args.output_dir.resolve()),
                )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
