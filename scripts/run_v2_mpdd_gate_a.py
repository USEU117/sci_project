"""Run and audit the MPDD seed-0 1-shot two-branch prediction Gate A."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        return subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--authorized-by-user", action="store_true")
    args = parser.parse_args()
    if not args.authorized_by_user:
        raise SystemExit("Gate A requires explicit authorization")
    root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / "v2_mpdd_s0_k1_gate_a"
    runtime = args.status.parent
    log_root = runtime / "logs"
    manifest = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
    data_root = ROOT / "data" / "mpdd_raw" / "MPDD"
    checkpoint = ROOT / "methods" / "AnomalyCLIP-main" / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"
    jobs = [
        (
            "anomalydino_visual",
            [str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "export_anomalydino_mpdd_predictions.py"), "--manifest", str(manifest), "--data-root", str(data_root), "--output-dir", str(root / "anomalydino_visual"), "--seed", "0", "--shot", "1"],
        ),
        (
            "anomalyclip_text",
            [str(ROOT / ".venv-anomalyclip" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "export_anomalyclip_mpdd_predictions.py"), "--manifest", str(manifest), "--data-root", str(data_root), "--checkpoint", str(checkpoint), "--output-dir", str(root / "anomalyclip_text"), "--seed", "0", "--shot", "1"],
        ),
    ]
    state = {"schema_version": 1, "status": "running", "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(), "current": None, "completed": [], "failed": [], "dataset": "mpdd", "dataset_role": "development", "btad_accessed": False, "metrics_computed": False}
    write_json(args.status, state)
    for name, command in jobs:
        report = root / name / "export_report.json"
        if report.is_file() and json.loads(report.read_text(encoding="utf-8")).get("status") == "passed":
            state["completed"].append(name)
            continue
        state["current"] = name
        write_json(args.status, state)
        code = run(command, log_root / f"{name}.stdout.log", log_root / f"{name}.stderr.log")
        if code:
            state.update(status="failed", current=None)
            state["failed"].append({"job": name, "exit_code": code})
            write_json(args.status, state)
            return code
        state["completed"].append(name)
    state["current"] = "audit"
    write_json(args.status, state)
    audit_command = [str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "audit_v2_mpdd_prediction_pair.py"), "--data-root", str(data_root), "--visual-dir", str(root / "anomalydino_visual"), "--text-dir", str(root / "anomalyclip_text"), "--seed", "0", "--shot", "1", "--output", str(runtime / "gate_a_audit.json")]
    code = run(audit_command, log_root / "audit.stdout.log", log_root / "audit.stderr.log")
    if code:
        state.update(status="failed", current=None)
        state["failed"].append({"job": "audit", "exit_code": code})
        write_json(args.status, state)
        return code
    state.update(status="passed", current=None, completed_at=datetime.now().astimezone().isoformat())
    write_json(args.status, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
