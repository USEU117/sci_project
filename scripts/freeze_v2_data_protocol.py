"""Freeze verified MPDD/BTAD archives, dataset audits and nested manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "experiments/dynamic_fusion/v2/data_preparation/mpdd_archive.json",
    "experiments/dynamic_fusion/v2/data_preparation/btad_archive.json",
    "experiments/dynamic_fusion/v2/data_preparation/mpdd_validation.json",
    "experiments/dynamic_fusion/v2/data_preparation/btad_validation.json",
    "data/splits/mpdd/manifest.json",
    "data/splits/mpdd/manifest.sha256",
    "data/splits/btad/manifest.json",
    "data/splits/btad/manifest.sha256",
    "experiments/dynamic_fusion/v2/data_preparation/mpdd_split_validation.json",
    "experiments/dynamic_fusion/v2/data_preparation/btad_split_validation.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    files = []
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"required data evidence missing: {path}")
        files.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        passed = expected.get("files") == files
        print(json.dumps({"status": "passed" if passed else "failed", "files": len(files)}))
        return 0 if passed else 1
    for dataset in ("mpdd", "btad"):
        dataset_report = json.loads((ROOT / f"experiments/dynamic_fusion/v2/data_preparation/{dataset}_validation.json").read_text(encoding="utf-8"))
        split_report = json.loads((ROOT / f"experiments/dynamic_fusion/v2/data_preparation/{dataset}_split_validation.json").read_text(encoding="utf-8"))
        if dataset_report.get("error_count") != 0 or not dataset_report.get("category_count_matches"):
            raise SystemExit(f"{dataset}: dataset validation did not pass")
        if split_report.get("error_count") != 0 or split_report.get("checksum_match") is not True:
            raise SystemExit(f"{dataset}: split validation did not pass")
    payload = {
        "schema_version": 1,
        "status": "data_protocol_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {"development": "mpdd", "holdout": "btad"},
        "shots": [1, 2, 4],
        "seeds": [0, 1, 2],
        "nested_sampling": True,
        "parameters_frozen": False,
        "holdout_metrics_allowed": False,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "files": len(files), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
