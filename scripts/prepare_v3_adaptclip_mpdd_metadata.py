"""Prepare frozen-manifest AdaptCLIP MPDD metadata for a V3-only Gate A."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_ad.fusion.v3_adaptclip_mpdd import build_adaptclip_mpdd_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata, audit = build_adaptclip_mpdd_metadata(args.data_root, manifest, args.seed, args.shot)
    audit.update(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": f"v3_adaptclip_mpdd_s{args.seed}_k{args.shot}_metadata_v1",
            "status": "validated" if args.validate_only else "passed",
            "validate_only": args.validate_only,
            "data_root": str(args.data_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "metadata_written": not args.validate_only,
        }
    )
    if not args.validate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = args.output_dir / "meta.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        audit["metadata_path"] = str(metadata_path.resolve())
        audit["metadata_sha256"] = sha256(metadata_path)
        audit_path = args.output_dir / "audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
