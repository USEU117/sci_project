"""Freeze or verify the DynamicFusion V2 code/protocol contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILES = (
    "configs/dynamic_fusion_v2.yaml",
    "configs/dynamic_fusion_v2_data_protocol.yaml",
    "src/industrial_ad/fusion/v2_calibration.py",
    "src/industrial_ad/fusion/v2_router.py",
    "src/industrial_ad/fusion/v2_diagnostics.py",
    "src/industrial_ad/fusion/v2_ablation.py",
    "scripts/fit_dynamic_fusion_v2_calibration.py",
    "scripts/run_dynamic_fusion_v2_cache.py",
    "scripts/audit_dynamic_fusion_v2_calibration.py",
    "scripts/prepare_v2_dataset_archive.py",
    "scripts/audit_v2_data_readiness.py",
    "scripts/freeze_v2_data_protocol.py",
    "scripts/complete_v2_data_preparation.ps1",
    "scripts/start_v2_data_preparation.ps1",
    "scripts/prepare_splits.py",
    "scripts/validate_splits.py",
    "scripts/validate_dataset.py",
    "tests/test_dynamic_fusion_v2.py",
    "tests/test_v2_data_protocol.py",
    "experiments/dynamic_fusion/v2/20260810_v2_cpu_contract_v2/report.json",
    "experiments/dynamic_fusion/v2/20260810_visa_s0_k1_retrospective_calibration/audit.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records() -> list[dict[str, object]]:
    result = []
    for relative in CONTRACT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--test-count", type=int, default=47)
    args = parser.parse_args()
    current = records()
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        passed = expected.get("files") == current
        print(json.dumps({"status": "passed" if passed else "failed", "files": len(current)}))
        return 0 if passed else 1
    payload = {
        "schema_version": 1,
        "status": "code_protocol_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_count_passed": args.test_count,
        "parameters_frozen": False,
        "holdout_metrics_allowed": False,
        "reason_parameters_not_frozen": "MPDD development and BTAD holdout data/caches are not yet ready",
        "forbidden_parameter_sources": ["btad_holdout_metrics", "mvtec_final_metrics", "visa_final_metrics"],
        "files": current,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "files": len(current), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
