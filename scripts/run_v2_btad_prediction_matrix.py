"""Run the frozen, resumable BTAD holdout prediction matrix and pair audits."""

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


def report_passed(directory: Path) -> bool:
    path = directory / "export_report.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--authorized-by-user", action="store_true")
    args = parser.parse_args()
    if not args.authorized_by_user:
        raise SystemExit("BTAD holdout experiments require explicit authorization")

    freeze_path = ROOT / "experiments" / "dynamic_fusion" / "v2" / "parameter_freeze" / "manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "parameters_frozen" or freeze.get("holdout_metrics_allowed_after_this_freeze") is not True:
        raise SystemExit("formal MPDD parameter freeze is missing or invalid")

    runtime = args.status.parent
    logs = runtime / "logs"
    audits = runtime / "audits"
    output_root = ROOT / "outputs" / "dynamic_fusion" / "v2_btad_predictions"
    gate_root = output_root / "v2_btad_s0_k1_full_v1"
    manifest = ROOT / "data" / "splits" / "btad" / "manifest.json"
    data_root = ROOT / "data" / "btad_raw" / "BTech_Dataset_transformed"
    checkpoint = ROOT / "methods" / "AnomalyCLIP-main" / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"
    py_visual = ROOT / ".venv-patchcore" / "Scripts" / "python.exe"
    py_text = ROOT / ".venv-anomalyclip" / "Scripts" / "python.exe"

    state = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "dataset": "btad",
        "dataset_role": "holdout",
        "parameter_freeze": str(freeze_path.resolve()),
        "selected_candidate": freeze.get("selected_candidate"),
        "current": None,
        "completed_pairs": [],
        "failed": [],
        "metrics_computed": False,
    }
    if args.status.is_file():
        previous = json.loads(args.status.read_text(encoding="utf-8"))
        state["completed_pairs"] = list(previous.get("completed_pairs", []))
        state["failed"] = list(previous.get("failed", []))
    write_json(args.status, state)

    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            pair_id = f"v2_btad_s{seed}_k{shot}_full_v1"
            pair_root = output_root / pair_id
            visual = pair_root / "anomalydino_visual"
            text = pair_root / "anomalyclip_text"
            audit = audits / f"{pair_id}.json"
            if audit.is_file() and json.loads(audit.read_text(encoding="utf-8")).get("status") == "passed":
                if pair_id not in state["completed_pairs"]:
                    state["completed_pairs"].append(pair_id)
                continue

            # Gate A is the first pair. No later pair can run until its audit passes.
            state["current"] = f"{pair_id}:visual"
            write_json(args.status, state)
            if not report_passed(visual):
                command = [str(py_visual), str(ROOT / "scripts" / "export_anomalydino_mpdd_predictions.py"), "--dataset", "btad", "--dataset-role", "holdout", "--manifest", str(manifest), "--data-root", str(data_root), "--output-dir", str(visual), "--seed", str(seed), "--shot", str(shot)]
                code = run(command, logs / f"{pair_id}.visual.stdout.log", logs / f"{pair_id}.visual.stderr.log")
                if code:
                    state.update(status="failed", current=None)
                    state["failed"].append({"pair": pair_id, "stage": "visual", "exit_code": code})
                    write_json(args.status, state)
                    return code

            state["current"] = f"{pair_id}:text"
            write_json(args.status, state)
            if not report_passed(text):
                if seed == 0 and shot == 1:
                    command = [str(py_text), str(ROOT / "scripts" / "export_anomalyclip_mpdd_predictions.py"), "--dataset", "btad", "--dataset-role", "holdout", "--manifest", str(manifest), "--data-root", str(data_root), "--checkpoint", str(checkpoint), "--output-dir", str(text), "--seed", "0", "--shot", "1"]
                else:
                    source = gate_root / "anomalyclip_text"
                    command = [str(py_visual), str(ROOT / "scripts" / "reuse_v2_mpdd_prediction_cache.py"), "--dataset", "btad", "--dataset-role", "holdout", "--source-dir", str(source), "--output-dir", str(text), "--branch", "anomalyclip_text", "--seed", str(seed), "--shot", str(shot), "--invariant-to-few-shot"]
                code = run(command, logs / f"{pair_id}.text.stdout.log", logs / f"{pair_id}.text.stderr.log")
                if code:
                    state.update(status="failed", current=None)
                    state["failed"].append({"pair": pair_id, "stage": "text", "exit_code": code})
                    write_json(args.status, state)
                    return code

            state["current"] = f"{pair_id}:audit"
            write_json(args.status, state)
            command = [str(py_visual), str(ROOT / "scripts" / "audit_v2_mpdd_prediction_pair.py"), "--dataset", "btad", "--dataset-role", "holdout", "--data-root", str(data_root), "--visual-dir", str(visual), "--text-dir", str(text), "--seed", str(seed), "--shot", str(shot), "--output", str(audit)]
            if not (seed == 0 and shot == 1):
                command.insert(-2, "--allow-invariant-text-reuse")
            code = run(command, logs / f"{pair_id}.audit.stdout.log", logs / f"{pair_id}.audit.stderr.log")
            if code:
                state.update(status="failed", current=None)
                state["failed"].append({"pair": pair_id, "stage": "audit", "exit_code": code})
                write_json(args.status, state)
                return code
            if seed == 0 and shot == 1:
                # Gate artifacts live in the regular first-pair directory; the audit is the gate marker.
                gate_marker = runtime / "gate_a_audit.json"
                gate_marker.write_text(audit.read_text(encoding="utf-8"), encoding="utf-8")
            state["completed_pairs"].append(pair_id)
            state["current"] = None
            state["updated_at"] = datetime.now().astimezone().isoformat()
            write_json(args.status, state)

    state.update(status="complete", current=None, completed_at=datetime.now().astimezone().isoformat())
    write_json(args.status, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
