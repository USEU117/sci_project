"""Validated, restart-safe PromptAD MVTec queue.

Existing complete combinations are detected from their unified evaluation
reports, so an interrupted queue never needs to rerun valid work.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMBOS = tuple((seed, shot) for seed in (0, 1, 2) for shot in (1, 2, 4))
STATE = ROOT / "outputs/logs/promptad_mvtec_resumable_queue"
STATUS = STATE / "status.json"
LOG = STATE / "queue.log"
TIMEOUT_SECONDS = 7 * 60 * 60


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(**values: object) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    values["updated_at_utc"] = now()
    STATUS.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    STATE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def valid_combo(seed: int, shot: int) -> bool:
    evaluation = ROOT / f"outputs/unified/promptad_mvtec_seed_{seed}_shot_{shot}/evaluation_report.json"
    predictions = ROOT / f"outputs/promptad/mvtec/seed_{seed}_shot_{shot}/predictions"
    if not evaluation.exists() or len(list(predictions.glob("*.npz"))) != 15:
        return False
    report = json.loads(evaluation.read_text(encoding="utf-8"))
    return (
        report.get("category_count") == 15
        and report.get("sample_count") == 1725
        and report.get("validation_errors") == 0
    )


def macro_metrics(seed: int, shot: int) -> dict[str, str]:
    summary = ROOT / f"outputs/unified/promptad_mvtec_seed_{seed}_shot_{shot}/summary.csv"
    with summary.open(encoding="utf-8") as handle:
        return next(row for row in csv.DictReader(handle) if row["category"] == "macro_mean")


def run_combo(seed: int, shot: int) -> bool:
    label = f"s{seed}_k{shot}"
    if valid_combo(seed, shot):
        log(f"SKIP {label}: passed unified validation")
        return True
    log(f"START {label}")
    combo_log = STATE / f"{label}.stdout.log"
    with combo_log.open("a", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "_run_promptad_mvtec.ps1"), "-Seed", str(seed), "-Shot", str(shot)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True,
                timeout=TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT {label}")
            return False
    if completed.returncode == 0 and valid_combo(seed, shot):
        metrics = macro_metrics(seed, shot)
        log(f"PASS {label}: image_auroc={metrics['image_auroc']} pixel_auroc={metrics['pixel_auroc']} aupro={metrics['aupro']}")
        return True
    log(f"FAIL {label}: exit={completed.returncode}; see {combo_log}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    completed = [f"s{s}_k{k}" for s, k in COMBOS if valid_combo(s, k)]
    pending = [f"s{s}_k{k}" for s, k in COMBOS if not valid_combo(s, k)]
    if args.validate_only:
        print(json.dumps({"completed": completed, "pending": pending}, indent=2))
        return 0
    failures: list[str] = []
    log(f"QUEUE START completed={completed} pending={pending}")
    for seed, shot in COMBOS:
        label = f"s{seed}_k{shot}"
        if label in completed:
            continue
        write_status(state="running", current=label, completed=completed, pending=[f"s{s}_k{k}" for s, k in COMBOS if f"s{s}_k{k}" not in completed], failures=failures)
        if run_combo(seed, shot):
            completed.append(label)
        else:
            failures.append(label)
            write_status(state="blocked", current=label, completed=completed, pending=[f"s{s}_k{k}" for s, k in COMBOS if f"s{s}_k{k}" not in completed], failures=failures)
            return 1
    write_status(state="completed", current=None, completed=completed, pending=[], failures=[])
    log("QUEUE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
