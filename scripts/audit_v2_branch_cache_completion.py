"""Audit the completed V2 normal-reference cache matrix and calibrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--runtime-status", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_status.read_text(encoding="utf-8"))
    failures: list[str] = []
    if runtime.get("status") != "complete":
        failures.append("runtime status is not complete")
    if runtime.get("failed"):
        failures.append("runtime contains failed jobs")
    if runtime.get("test_metrics_computed") is not False:
        failures.append("runtime test_metrics_computed must be false")

    cache_audits = []
    for job in queue["jobs"]:
        run_id = str(job["run_id"])
        audit_path = args.runtime_root / "cache_audits" / f"{run_id}.json"
        if not audit_path.is_file():
            failures.append(f"cache audit missing: {run_id}")
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("all_passed") or audit.get("failed") != 0:
            failures.append(f"cache audit failed: {run_id}")
        if audit.get("test_predictions_used") or audit.get("test_labels_used"):
            failures.append(f"forbidden test evidence in cache audit: {run_id}")
        cache_audits.append(
            {"run_id": run_id, "path": str(audit_path.resolve()), "sha256": sha256(audit_path)}
        )

    calibration_audits = []
    for dataset, role, categories in (("mpdd", "development", 6), ("btad", "holdout", 3)):
        for seed in (0, 1, 2):
            for shot in (1, 2, 4):
                pair_id = f"v2_{dataset}_s{seed}_k{shot}_branch_cache_v1"
                calibration = args.runtime_root / "calibrations" / f"{pair_id}.json"
                visual_dir = ROOT / "outputs" / "dynamic_fusion" / "v2_branch_cache" / pair_id / "anomalydino_visual"
                text_dir = ROOT / "outputs" / "dynamic_fusion" / "v2_branch_cache" / pair_id / "anomalyclip_text"
                audit_path = args.runtime_root / "calibration_audits" / f"{pair_id}.json"
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                if not calibration.is_file():
                    failures.append(f"calibration missing: {pair_id}")
                    continue
                calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
                if calibration_payload.get("dataset_role") != role:
                    failures.append(f"dataset role differs: {pair_id}")
                for field in (
                    "test_predictions_used",
                    "test_labels_used",
                    "test_masks_used",
                    "test_set_statistics_used",
                ):
                    if calibration_payload.get(field) is not False:
                        failures.append(f"{field} must be false: {pair_id}")
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_dynamic_fusion_v2_calibration.py"),
                    "--calibration-json", str(calibration),
                    "--visual-reference-dir", str(visual_dir),
                    "--text-reference-dir", str(text_dir),
                    "--output", str(audit_path),
                ]
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    failures.append(f"calibration audit failed: {pair_id}: {completed.stderr.strip()}")
                    continue
                result = json.loads(audit_path.read_text(encoding="utf-8"))
                audited_categories = {str(row["category"]) for row in result.get("rows", [])}
                if result.get("status") != "passed" or len(audited_categories) != categories:
                    failures.append(f"calibration audit content failed: {pair_id}")
                calibration_audits.append(
                    {"run_id": pair_id, "path": str(audit_path.resolve()), "sha256": sha256(audit_path)}
                )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "queue": str(args.queue.resolve()),
        "queue_sha256": sha256(args.queue),
        "runtime_status": str(args.runtime_status.resolve()),
        "runtime_status_sha256": sha256(args.runtime_status),
        "cache_jobs_expected": 36,
        "cache_audits_passed": len(cache_audits),
        "calibrations_expected": 18,
        "calibration_audits_passed": len(calibration_audits),
        "development_dataset": "mpdd",
        "holdout_dataset": "btad",
        "holdout_metrics_read": False,
        "failures": failures,
        "cache_audits": cache_audits,
        "calibration_audits": calibration_audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "cache_audits_passed", "calibration_audits_passed", "failures")}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
