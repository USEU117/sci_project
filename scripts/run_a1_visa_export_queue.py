"""Serial, resumable A1 VisA feature-export queue (post-freeze validation).

GPU discipline: single process at a time; every step writes an export_report.json
marker; on restart, already-passed steps are skipped. Never runs two jobs at once.

Phases:
  1. dino  s0/k1 full export        (.venv-patchcore,   dinov2_vitb14)
  2. clip  s0/k1 full export        (.venv-anomalyclip, AnomalyCLIP ViT-L/14@336)
  3. dino  ref-only for the other 8 (seed,shot) combos (base = dino s0/k1)
  4. clip  ref-only for the other 8 (seed,shot) combos (base = clip s0/k1)

Rationale for ref-only: A1 test patch features depend only on the fixed test
images, not on (seed, shot); only the reference patch features change.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PY_DINO = str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe")
PY_CLIP = str(ROOT / ".venv-anomalyclip" / "Scripts" / "python.exe")
CHECKPOINT = ROOT / "methods" / "AnomalyCLIP-main" / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"
FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"

# dataset -> (manifest, data_root, dino prefix, clip prefix, log dir name)
DATASET_INFO = {
    "visa": (
        ROOT / "data" / "splits" / "visa" / "manifest.json",
        ROOT / "data" / "visa_raw",
        "visa_features_vitb14",
        "visa_features",
        "a1_visa_export_queue",
    ),
    "mvtec": (
        ROOT / "data" / "splits" / "mvtec" / "manifest.json",
        ROOT / "data" / "mvtec",
        "mvtec_features_vitb14",
        "mvtec_features",
        "a1_mvtec_export_queue",
    ),
}

SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def report_passed(directory: Path) -> bool:
    path = directory / "export_report.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=ROOT, stdout=handle, stderr=handle, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("visa", "mvtec"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-phase-2-jobs", type=int, default=0,
                        help="0 = unlimited; >0 = run at most N ref-only jobs per branch per call (manual chunking)")
    args = parser.parse_args()

    manifest_path, data_root, dino_prefix, clip_prefix, log_dir = DATASET_INFO[args.dataset]
    STATUS = ROOT / "outputs" / "logs" / log_dir / "status.json"
    MANIFEST = manifest_path
    DATA_ROOT = data_root

    state = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "dataset": args.dataset,
        "dataset_role": "holdout",
        "completed": [],
        "failed": [],
        "current": None,
    }
    if STATUS.is_file():
        prev = json.loads(STATUS.read_text(encoding="utf-8"))
        state["completed"] = list(prev.get("completed", []))
        state["failed"] = list(prev.get("failed", []))
    write_json(STATUS, state)

    def step(step_id: str, python: str, script: str, args_list: list[str], out_dir: Path, phase: str) -> bool:
        if step_id in state["completed"]:
            return True
        if args.validate_only:
            print(json.dumps({"check": step_id, "exists": report_passed(out_dir)}))
            return True
        state["current"] = step_id
        write_json(STATUS, state)
        if not report_passed(out_dir):
            cmd = [python, str(ROOT / "scripts" / script)] + args_list
            log = STATUS.parent / f"{step_id}.log"
            code = run(cmd, log)
            if code or not report_passed(out_dir):
                state["failed"].append({"step": step_id, "exit_code": code, "log": str(log)})
                state.update(status="failed", current=None)
                write_json(STATUS, state)
                print(json.dumps({"step": step_id, "status": "failed", "exit_code": code}))
                return False
        state["completed"].append(step_id)
        state["current"] = None
        state["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(STATUS, state)
        print(json.dumps({"step": step_id, "status": "ok"}), flush=True)
        return True

    dino_s0k1 = FEATURES_ROOT / dino_prefix / "s0_k1" / "anomalydino_visual"
    clip_s0k1 = FEATURES_ROOT / clip_prefix / "s0_k1" / "anomalyclip_text"
    common = ["--manifest", str(MANIFEST), "--dataset", args.dataset, "--dataset-role", "holdout",
              "--data-root", str(DATA_ROOT)]

    # Phase 1: dino s0/k1 full
    if not step("dino_s0_k1_full", PY_DINO, "export_a1_visa_features.py",
                common + ["--branch", "dino", "--model-name", "dinov2_vitb14",
                          "--output-dir", str(dino_s0k1), "--seed", "0", "--shot", "1"],
                dino_s0k1, "dino_full"):
        return 1

    # Phase 2: clip s0/k1 full
    if not step("clip_s0_k1_full", PY_CLIP, "export_a1_visa_features.py",
                common + ["--branch", "clip", "--checkpoint", str(CHECKPOINT),
                          "--output-dir", str(clip_s0k1), "--seed", "0", "--shot", "1"],
                clip_s0k1, "clip_full"):
        return 1

    # Phases 3/4: ref-only for the other 8 combos per branch
    ref_jobs = []
    for seed in SEEDS:
        for shot in SHOTS:
            if (seed, shot) == (0, 1):
                continue
            ref_jobs.append(("dino", seed, shot))
            ref_jobs.append(("clip", seed, shot))

    run_count = {"dino": 0, "clip": 0}
    for branch, seed, shot in ref_jobs:
        limit = args.max_phase_2_jobs
        if limit and run_count[branch] >= limit:
            print(json.dumps({"info": f"max_phase_2_jobs={limit} reached for {branch}, stopping"}))
            break
        if branch == "dino":
            base = dino_s0k1
            python = PY_DINO
            extra = ["--model-name", "dinov2_vitb14"]
            out_dir = FEATURES_ROOT / dino_prefix / f"s{seed}_k{shot}" / "anomalydino_visual"
        else:
            base = clip_s0k1
            python = PY_CLIP
            extra = ["--checkpoint", str(CHECKPOINT)]
            out_dir = FEATURES_ROOT / clip_prefix / f"s{seed}_k{shot}" / "anomalyclip_text"
        step_id = f"{branch}_s{seed}_k{shot}_refonly"
        if not step(step_id, python, "export_a1_visa_ref_only.py",
                    common + ["--branch", branch, "--base-cache", str(base)] + extra +
                    ["--output-dir", str(out_dir), "--seed", str(seed), "--shot", str(shot)],
                    out_dir, "refonly"):
            return 1
        run_count[branch] += 1

    state.update(status="complete", current=None, completed_at=datetime.now().astimezone().isoformat())
    write_json(STATUS, state)
    print(json.dumps({"status": "complete", "completed": len(state["completed"]), "failed": len(state["failed"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
