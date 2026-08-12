"""Validate every V2 branch-cache command without starting GPU inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    if queue.get("execution_authorized") is not False:
        raise SystemExit("queue must explicitly forbid execution during dry-run")
    rows: list[dict[str, object]] = []
    for job in queue["jobs"]:
        command = [str(value) for value in job["validate_only_command"]]
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        rows.append(
            {
                "run_id": job["run_id"],
                "dataset": job["dataset"],
                "role": job["role"],
                "branch": job["branch"],
                "seed": job["seed"],
                "shot": job["shot"],
                "exit_code": completed.returncode,
                "status": "passed" if completed.returncode == 0 else "failed",
                "command": command,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    failures = [row for row in rows if row["status"] != "passed"]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "mode": "validate_only",
        "gpu_inference_started": False,
        "queue": str(args.queue.resolve()),
        "queue_sha256": sha256(args.queue),
        "jobs_checked": len(rows),
        "failures": len(failures),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "mode", "jobs_checked", "failures")}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
