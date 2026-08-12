"""Run the prepared DynamicFusion V2 branch-cache queue safely and resumably."""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def gpu_state() -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    values = [int(value.strip()) for value in completed.stdout.strip().split(",")]
    return dict(zip(("utilization", "memory_used", "memory_free", "temperature"), values))


def run_recorded(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    return completed.returncode


def output_dir_from(command: list[str]) -> Path:
    index = command.index("--output-dir")
    return Path(command[index + 1])


def expected_categories(dataset: str) -> int:
    return {"mpdd": 6, "btad": 3}[dataset]


def audit_paths(audit_root: Path, run_id: str) -> tuple[Path, Path]:
    return audit_root / f"{run_id}.json", audit_root / f"{run_id}.csv"


def cache_complete(job: dict, audit_root: Path) -> bool:
    command = [str(value) for value in job["command"]]
    export_report = output_dir_from(command) / "export_report.json"
    audit_json, _ = audit_paths(audit_root, str(job["run_id"]))
    if not export_report.is_file() or not audit_json.is_file():
        return False
    return read_json(export_report).get("status") == "passed" and read_json(audit_json).get(
        "all_passed"
    ) is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="ISO timestamp with timezone")
    parser.add_argument("--authorized-by-user", action="store_true")
    parser.add_argument("--minimum-free-mib", type=int, default=3800)
    parser.add_argument("--maximum-temperature", type=int, default=80)
    parser.add_argument("--latest-start-minutes", type=int, default=15)
    args = parser.parse_args()
    if not args.authorized_by_user:
        raise SystemExit("GPU execution requires --authorized-by-user")

    cutoff_text = args.cutoff.strip("'\"")
    cutoff_text = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", cutoff_text)
    cutoff = datetime.fromisoformat(cutoff_text)
    if cutoff.tzinfo is None:
        raise SystemExit("cutoff must include a timezone")
    queue = read_json(args.queue)
    if queue.get("status") != "prepared_not_started" or len(queue.get("jobs", [])) != 36:
        raise SystemExit("prepared 36-job queue not found")
    if queue.get("development_dataset") != "mpdd" or queue.get("holdout_dataset") != "btad":
        raise SystemExit("MPDD-development/BTAD-holdout boundary differs")
    if queue.get("holdout_metrics_allowed") is not False:
        raise SystemExit("BTAD holdout metrics must remain disabled")

    args.log_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.status.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+b")
    lock_handle.seek(0)
    if lock_handle.tell() == 0:
        lock_handle.write(b"0")
        lock_handle.flush()
    lock_handle.seek(0)
    try:
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("another V2 queue runner holds the lock")

    state = {
        "schema_version": 1,
        "status": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "pid": os.getpid(),
        "cutoff": cutoff.isoformat(),
        "authorized_by_user": True,
        "development_dataset": "mpdd",
        "holdout_dataset": "btad",
        "holdout_metrics_allowed": False,
        "test_metrics_computed": False,
        "current_job": None,
        "completed": [],
        "skipped": [],
        "failed": [],
        "calibrations": [],
    }
    if args.status.is_file():
        previous = read_json(args.status)
        for key in ("completed", "skipped", "failed", "calibrations"):
            state[key] = list(previous.get(key, []))
    write_json(args.status, state)
    audit_root = args.status.parent / "cache_audits"
    calibration_root = args.status.parent / "calibrations"
    patchcore_python = ROOT / ".venv-patchcore" / "Scripts" / "python.exe"

    try:
        for job in queue["jobs"]:
            run_id = str(job["run_id"])
            if cache_complete(job, audit_root):
                if run_id not in state["skipped"]:
                    state["skipped"].append(run_id)
                continue
            remaining = (cutoff - datetime.now().astimezone()).total_seconds()
            if remaining <= args.latest_start_minutes * 60:
                state.update(status="stopped_at_cutoff", current_job=None, updated_at=now_iso())
                write_json(args.status, state)
                return 0

            while True:
                gpu = gpu_state()
                state["gpu"] = gpu
                state["updated_at"] = now_iso()
                write_json(args.status, state)
                if (
                    gpu["memory_free"] >= args.minimum_free_mib
                    and gpu["temperature"] <= args.maximum_temperature
                ):
                    break
                if (cutoff - datetime.now().astimezone()).total_seconds() <= args.latest_start_minutes * 60:
                    state.update(status="stopped_at_cutoff", current_job=None, updated_at=now_iso())
                    write_json(args.status, state)
                    return 0
                time.sleep(30)

            command = [str(value) for value in job["command"]]
            output_dir = output_dir_from(command)
            if output_dir.exists() and not (output_dir / "export_report.json").is_file():
                archived = output_dir.with_name(
                    output_dir.name + ".failed_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                shutil.move(str(output_dir), str(archived))
            state.update(current_job=run_id, current_started_at=now_iso(), updated_at=now_iso())
            write_json(args.status, state)
            stdout_path = args.log_root / f"{run_id}.stdout.log"
            stderr_path = args.log_root / f"{run_id}.stderr.log"
            exit_code = run_recorded(command, stdout_path, stderr_path)
            if exit_code != 0:
                state["failed"].append(
                    {"run_id": run_id, "stage": "export", "exit_code": exit_code, "at": now_iso()}
                )
                state.update(status="failed", current_job=None, updated_at=now_iso())
                write_json(args.status, state)
                return exit_code

            manifest = ROOT / "data" / "splits" / str(job["dataset"]) / "manifest.json"
            audit_json, audit_csv = audit_paths(audit_root, run_id)
            audit_command = [
                str(patchcore_python),
                str(ROOT / "scripts" / "audit_normal_reference_cache.py"),
                "--cache-dir", str(output_dir),
                "--manifest", str(manifest),
                "--dataset", str(job["dataset"]),
                "--branch", str(job["branch"]),
                "--seed", str(job["seed"]),
                "--shot", str(job["shot"]),
                "--expected-categories", str(expected_categories(str(job["dataset"]))),
                "--report-json", str(audit_json),
                "--report-csv", str(audit_csv),
            ]
            audit_exit = run_recorded(
                audit_command,
                args.log_root / f"{run_id}.audit.stdout.log",
                args.log_root / f"{run_id}.audit.stderr.log",
            )
            if audit_exit != 0:
                state["failed"].append(
                    {"run_id": run_id, "stage": "audit", "exit_code": audit_exit, "at": now_iso()}
                )
                state.update(status="failed", current_job=None, updated_at=now_iso())
                write_json(args.status, state)
                return audit_exit
            if run_id not in state["completed"]:
                state["completed"].append(run_id)

            dataset = str(job["dataset"])
            seed = int(job["seed"])
            shot = int(job["shot"])
            pair_id = f"v2_{dataset}_s{seed}_k{shot}_branch_cache_v1"
            visual_dir = ROOT / "outputs" / "dynamic_fusion" / "v2_branch_cache" / pair_id / "anomalydino_visual"
            text_dir = ROOT / "outputs" / "dynamic_fusion" / "v2_branch_cache" / pair_id / "anomalyclip_text"
            calibration_path = calibration_root / f"{pair_id}.json"
            if (
                (visual_dir / "export_report.json").is_file()
                and (text_dir / "export_report.json").is_file()
                and not calibration_path.is_file()
            ):
                calibration_command = [
                    str(patchcore_python),
                    str(ROOT / "scripts" / "fit_dynamic_fusion_v2_calibration.py"),
                    "--visual-dir", str(visual_dir),
                    "--text-dir", str(text_dir),
                    "--visual-branch", "anomalydino_visual",
                    "--text-branch", "anomalyclip_text",
                    "--dataset", dataset,
                    "--dataset-role", str(job["role"]),
                    "--seed", str(seed),
                    "--shot", str(shot),
                    "--run-id", pair_id,
                    "--output", str(calibration_path),
                ]
                calibration_exit = run_recorded(
                    calibration_command,
                    args.log_root / f"{pair_id}.calibration.stdout.log",
                    args.log_root / f"{pair_id}.calibration.stderr.log",
                )
                if calibration_exit != 0:
                    state["failed"].append(
                        {"run_id": pair_id, "stage": "calibration", "exit_code": calibration_exit, "at": now_iso()}
                    )
                    state.update(status="failed", current_job=None, updated_at=now_iso())
                    write_json(args.status, state)
                    return calibration_exit
                state["calibrations"].append(pair_id)
            state.update(current_job=None, updated_at=now_iso())
            write_json(args.status, state)

        state.update(status="complete", current_job=None, completed_at=now_iso(), updated_at=now_iso())
        write_json(args.status, state)
        return 0
    finally:
        try:
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
