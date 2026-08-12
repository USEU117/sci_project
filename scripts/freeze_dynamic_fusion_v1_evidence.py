"""Create or verify a hash manifest for immutable DynamicFusion V1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    "configs/dynamic_fusion.yaml",
    "src/industrial_ad/fusion/calibration.py",
    "src/industrial_ad/fusion/router.py",
    "experiments/dynamic_fusion/final_validation_audit_20260808/final_validation_audit.json",
    "experiments/summaries/dynamic_fusion_scientific_analysis_20260809/summary.json",
    "outputs/dynamic_fusion/final_validation/summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> list[dict[str, object]]:
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = [path if path.is_absolute() else ROOT / path for path in (args.paths or [Path(value) for value in DEFAULT_PATHS])]
    current = snapshot(paths)
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        passed = expected.get("files") == current
        print(json.dumps({"status": "passed" if passed else "failed", "files": len(current)}))
        return 0 if passed else 1
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "protect DynamicFusion V1 evidence before V2 development",
        "files": current,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "frozen", "files": len(current), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
