"""Build a machine-readable snapshot of the current project state."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def active_training_processes() -> list[dict[str, object]]:
    script = r"""
$patterns = 'train_cls.py|train_seg.py|run_promptad_mvtec|_overnight_queue.py|run_gpu_job_scheduler.ps1|run_dynamic_fusion_reference_pipeline.ps1|few_shot.py'
Get-CimInstance Win32_Process |
  Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match $patterns } |
  Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine |
  ConvertTo-Json -Compress
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    return payload if isinstance(payload, list) else [payload]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    summaries = ROOT / "experiments" / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    visa_path = summaries / f"visa_result_audit_{args.date}.json"
    mvtec_path = summaries / f"mvtec_method_seed_shot_completeness_{args.date}.json"
    fusion_path = ROOT / "experiments/dynamic_fusion/final_validation_audit_20260808/final_validation_audit.json"
    queue_path = ROOT / "outputs/logs/promptad_mvtec_resumable_queue/status.json"
    legacy_queue_path = ROOT / "outputs/logs/overnight_status.json"

    required = [visa_path, mvtec_path, fusion_path, queue_path, legacy_queue_path]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required evidence: {missing}")

    visa = read_json(visa_path)
    mvtec = read_json(mvtec_path)
    fusion = read_json(fusion_path)
    queue = read_json(queue_path)
    legacy_queue = read_json(legacy_queue_path)
    active = active_training_processes()
    promptad_partial_dir = ROOT / "outputs/logs/promptad/mvtec/seed_1_shot_2"
    promptad_partial_markers = sorted(path.name for path in promptad_partial_dir.glob("*.complete"))

    mvtec_counts: dict[str, int] = {}
    mvtec_missing: dict[str, list[str]] = {}
    for method in ("PatchCore", "WinCLIP+", "AnomalyDINO", "PromptAD", "DynamicFusion"):
        rows = [row for row in mvtec["rows"] if row["method"] == method]
        mvtec_counts[method] = sum(row["status"] == "complete" for row in rows)
        mvtec_missing[method] = [
            f"s{row['seed']}_k{row['shot']}" for row in rows if row["status"] != "complete"
        ]

    config_text = (ROOT / "configs/dynamic_fusion.yaml").read_text(encoding="utf-8")
    errors: list[str] = []
    if visa.get("status") != "passed" or visa.get("audited_runs") != 36:
        errors.append("VisA baseline audit is not 36/36 passed")
    expected_counts = {
        "PatchCore": 9,
        "WinCLIP+": 9,
        "AnomalyDINO": 9,
        "PromptAD": 4,
        "DynamicFusion": 9,
    }
    if mvtec_counts != expected_counts:
        errors.append(f"MVTec counts differ from expected snapshot: {mvtec_counts}")
    if fusion.get("status") != "passed" or len(fusion.get("runs", [])) != 17:
        errors.append("dynamic-fusion audit is not 17/17 passed")
    if queue.get("state") != "paused_by_schedule":
        errors.append("authoritative PromptAD queue is not paused_by_schedule")
    if sorted(queue.get("pending", [])) != sorted(mvtec_missing["PromptAD"]):
        errors.append("PromptAD queue pending list differs from MVTec completeness matrix")
    if legacy_queue.get("phase") != "superseded":
        errors.append("legacy overnight queue is not marked superseded")
    if active:
        errors.append(f"active training processes detected: {len(active)}")
    for expected in (
        "phase: frozen_final_validation_complete",
        "visual_branch: anomalydino",
        "text_guided_branch: anomalyclip",
        "design_frozen: true",
    ):
        if expected not in config_text:
            errors.append(f"dynamic fusion config missing: {expected}")

    method_rows = [
        {"dataset": "VisA", "method": method, "complete": 9, "expected": 9, "status": "complete", "notes": "3 seeds x 1/2/4-shot"}
        for method in ("PatchCore", "WinCLIP+", "AnomalyDINO", "PromptAD")
    ]
    method_rows.extend(
        [
            {"dataset": "VisA", "method": "DynamicFusion", "complete": 6, "expected": 6, "status": "complete", "notes": "independent final validation: seeds 1/2 x 1/2/4-shot"},
            *[
                {"dataset": "MVTec", "method": method, "complete": mvtec_counts[method], "expected": 9, "status": "complete" if mvtec_counts[method] == 9 else "partial", "notes": ",".join(mvtec_missing[method]) or "none"}
                for method in ("PatchCore", "WinCLIP+", "AnomalyDINO", "PromptAD", "DynamicFusion")
            ],
            {"dataset": "MVTec", "method": "AnomalyCLIP", "complete": 1, "expected": 1, "status": "zero_shot_only", "notes": "not a 1/2/4-shot matrix"},
            {"dataset": "MVTec", "method": "ReMP-AD", "complete": 0, "expected": 1, "status": "gate_a_pending", "notes": "environment ready; manifest and NPZ adapters pending"},
            {"dataset": "MVTec", "method": "AdaptCLIP", "complete": 0, "expected": 1, "status": "gate_a_blocked", "notes": "checkpoint missing; 6 GB VRAM smoke pending"},
        ]
    )

    csv_path = summaries / f"current_method_status_{args.date}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)

    snapshot = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "active_training_processes": active,
        "gpu_queue": {
            "authoritative_status": str(queue_path.relative_to(ROOT)),
            "state": queue.get("state"),
            "completed": queue.get("completed", []),
            "pending": queue.get("pending", []),
            "legacy_status": str(legacy_queue_path.relative_to(ROOT)),
            "legacy_phase": legacy_queue.get("phase"),
            "current_partial": {
                "run": "s1_k2",
                "complete_marker_count": len(promptad_partial_markers),
                "expected_marker_count": 30,
                "complete_markers": promptad_partial_markers,
                "resume_at": "capsule_seg",
            },
        },
        "visa": {"baseline_runs": visa.get("audited_runs"), "status": visa.get("status")},
        "mvtec": {"complete_counts": mvtec_counts, "missing": mvtec_missing},
        "dynamic_fusion": {
            "audit_status": fusion.get("status"),
            "audited_runs": len(fusion.get("runs", [])),
            "visual_branch": "AnomalyDINO",
            "text_branch": "AnomalyCLIP",
            "image_temperature": 0.50,
            "pixel_temperature": 0.20,
            "design_frozen": True,
        },
        "evidence": {
            "visa_audit": str(visa_path.relative_to(ROOT)),
            "mvtec_completeness": str(mvtec_path.relative_to(ROOT)),
            "dynamic_fusion_audit": str(fusion_path.relative_to(ROOT)),
            "method_status_csv": str(csv_path.relative_to(ROOT)),
        },
    }
    output = summaries / f"project_state_snapshot_{args.date}.json"
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0 if snapshot["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
