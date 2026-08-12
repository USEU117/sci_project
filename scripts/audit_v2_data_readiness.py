"""Report live MPDD/BTAD download, extraction and manifest readiness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "mpdd": {
        "archive": ROOT / "data" / "downloads" / "MPDD.zip",
        "expected_size": 1825041283,
        "expected_sha256": "69f8da73eea4a31451a50251e5c261e83e0c53f2d1a39a7d4dfc78b5c434ddd6",
        "extract_report": ROOT / "experiments" / "dynamic_fusion" / "v2" / "data_preparation" / "mpdd_archive.json",
    },
    "btad": {
        "archive": ROOT / "data" / "downloads" / "btad.zip",
        "expected_size": 1229193337,
        "expected_sha256": None,
        "extract_report": ROOT / "experiments" / "dynamic_fusion" / "v2" / "data_preparation" / "btad_archive.json",
    },
}


def dataset_status(name: str, spec: dict[str, object]) -> dict[str, object]:
    archive = Path(spec["archive"])
    actual_size = archive.stat().st_size if archive.is_file() else 0
    expected_size = int(spec["expected_size"])
    extract_report = Path(spec["extract_report"])
    extraction = None
    if extract_report.is_file():
        extraction = json.loads(extract_report.read_text(encoding="utf-8"))
    manifest = ROOT / "data" / "splits" / name / "manifest.json"
    split_report = (
        ROOT
        / "experiments"
        / "dynamic_fusion"
        / "v2"
        / "data_preparation"
        / f"{name}_split_validation.json"
    )
    split_validation = None
    if split_report.is_file():
        split_validation = json.loads(split_report.read_text(encoding="utf-8"))
    archive_complete = actual_size == expected_size
    extraction_passed = bool(extraction and extraction.get("status") == "passed")
    split_passed = bool(
        split_validation
        and split_validation.get("error_count") == 0
        and split_validation.get("checksum_match") is True
    )
    if split_passed:
        status = "ready"
    elif extraction_passed:
        status = "extracted_waiting_for_manifest"
    elif archive_complete:
        status = "downloaded_waiting_for_verification"
    elif actual_size:
        status = "downloading"
    else:
        status = "missing"
    return {
        "status": status,
        "archive": str(archive),
        "archive_size_bytes": actual_size,
        "expected_size_bytes": expected_size,
        "download_fraction": actual_size / expected_size,
        "expected_sha256": spec["expected_sha256"],
        "extract_report": str(extract_report),
        "manifest": str(manifest),
        "split_validation_report": str(split_report),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    datasets = {name: dataset_status(name, spec) for name, spec in SPECS.items()}
    ready = all(value["status"] == "ready" for value in datasets.values())
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if ready else "not_ready",
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if ready or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
