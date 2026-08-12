"""Run the resumable MPDD-only V2 full prediction-cache matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
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


def report_passed(directory: Path) -> bool:
    path = directory / "export_report.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--authorized-by-user", action="store_true")
    args = parser.parse_args()
    if not args.authorized_by_user:
        raise SystemExit("MPDD matrix requires explicit authorization")
    cutoff_text = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", args.cutoff.strip("'\""))
    cutoff = datetime.fromisoformat(cutoff_text)
    gate_a = ROOT / "experiments" / "dynamic_fusion" / "v2" / "mpdd_prediction_gate_a" / "runtime" / "gate_a_audit.json"
    if not gate_a.is_file() or json.loads(gate_a.read_text(encoding="utf-8")).get("status") != "passed":
        raise SystemExit("MPDD seed0 K1 Gate A has not passed")

    runtime = args.status.parent
    logs = runtime / "logs"
    output_root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions"
    gate_output = output_root / "v2_mpdd_s0_k1_gate_a"
    manifest = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
    data_root = ROOT / "data" / "mpdd_raw" / "MPDD"
    state = {"schema_version": 1, "status": "running", "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(), "cutoff": cutoff.isoformat(), "dataset": "mpdd", "dataset_role": "development", "btad_accessed": False, "current": None, "completed_pairs": [], "failed": []}
    if args.status.is_file():
        previous = json.loads(args.status.read_text(encoding="utf-8"))
        state["completed_pairs"] = list(previous.get("completed_pairs", []))
        state["failed"] = list(previous.get("failed", []))
    write_json(args.status, state)

    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            pair_id = f"v2_mpdd_s{seed}_k{shot}_full_v1"
            pair_root = output_root / pair_id
            visual = pair_root / "anomalydino_visual"
            text = pair_root / "anomalyclip_text"
            audit = runtime / "audits" / f"{pair_id}.json"
            if audit.is_file() and json.loads(audit.read_text(encoding="utf-8")).get("status") == "passed":
                if pair_id not in state["completed_pairs"]:
                    state["completed_pairs"].append(pair_id)
                continue
            if (cutoff - datetime.now().astimezone()).total_seconds() <= 15 * 60:
                state.update(status="stopped_at_cutoff", current=None, updated_at=datetime.now().astimezone().isoformat())
                write_json(args.status, state)
                return 0
            state["current"] = f"{pair_id}:visual"
            write_json(args.status, state)
            if not report_passed(visual):
                if seed == 0 and shot == 1:
                    command = [str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "reuse_v2_mpdd_prediction_cache.py"), "--source-dir", str(gate_output / "anomalydino_visual"), "--output-dir", str(visual), "--branch", "anomalydino_visual", "--seed", "0", "--shot", "1"]
                else:
                    command = [str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "export_anomalydino_mpdd_predictions.py"), "--manifest", str(manifest), "--data-root", str(data_root), "--output-dir", str(visual), "--seed", str(seed), "--shot", str(shot)]
                code = run(command, logs / f"{pair_id}.visual.stdout.log", logs / f"{pair_id}.visual.stderr.log")
                if code:
                    state.update(status="failed", current=None)
                    state["failed"].append({"pair": pair_id, "stage": "visual", "exit_code": code})
                    write_json(args.status, state)
                    return code
            state["current"] = f"{pair_id}:text_reuse"
            write_json(args.status, state)
            if not report_passed(text):
                command = [str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "reuse_v2_mpdd_prediction_cache.py"), "--source-dir", str(gate_output / "anomalyclip_text"), "--output-dir", str(text), "--branch", "anomalyclip_text", "--seed", str(seed), "--shot", str(shot), "--invariant-to-few-shot"]
                code = run(command, logs / f"{pair_id}.text.stdout.log", logs / f"{pair_id}.text.stderr.log")
                if code:
                    state.update(status="failed", current=None)
                    state["failed"].append({"pair": pair_id, "stage": "text_reuse", "exit_code": code})
                    write_json(args.status, state)
                    return code
            state["current"] = f"{pair_id}:audit"
            write_json(args.status, state)
            command = [str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "audit_v2_mpdd_prediction_pair.py"), "--data-root", str(data_root), "--visual-dir", str(visual), "--text-dir", str(text), "--seed", str(seed), "--shot", str(shot), "--allow-invariant-text-reuse", "--output", str(audit)]
            code = run(command, logs / f"{pair_id}.audit.stdout.log", logs / f"{pair_id}.audit.stderr.log")
            if code:
                state.update(status="failed", current=None)
                state["failed"].append({"pair": pair_id, "stage": "audit", "exit_code": code})
                write_json(args.status, state)
                return code
            state["completed_pairs"].append(pair_id)
            state["current"] = None
            state["updated_at"] = datetime.now().astimezone().isoformat()
            write_json(args.status, state)
    state.update(status="complete", current=None, completed_at=datetime.now().astimezone().isoformat())
    write_json(args.status, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
