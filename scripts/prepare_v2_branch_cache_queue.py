"""Prepare (but never execute) the DynamicFusion V2 GPU branch-cache queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "mpdd": {
        "role": "development",
        "root": ROOT / "data" / "mpdd_raw" / "MPDD",
        "manifest": ROOT / "data" / "splits" / "mpdd" / "manifest.json",
    },
    "btad": {
        "role": "holdout",
        "root": ROOT / "data" / "btad_raw" / "BTech_Dataset_transformed",
        "manifest": ROOT / "data" / "splits" / "btad" / "manifest.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "dynamic_fusion" / "v2" / "branch_cache_queue",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "methods"
        / "AnomalyCLIP-main"
        / "checkpoints"
        / "9_12_4_multiscale_visa"
        / "epoch_15.pth",
    )
    parser.add_argument("--skip-view-generation", action="store_true")
    args = parser.parse_args()

    freeze_path = (
        ROOT
        / "experiments"
        / "dynamic_fusion"
        / "v2"
        / "data_protocol_freeze"
        / "manifest.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "data_protocol_frozen":
        raise SystemExit("V2 data protocol is not frozen")
    if freeze["datasets"] != {"development": "mpdd", "holdout": "btad"}:
        raise SystemExit("development/holdout boundary differs from the frozen protocol")
    if freeze["holdout_metrics_allowed"] is not False:
        raise SystemExit("BTAD holdout metrics must remain disabled")
    if not args.checkpoint.is_file():
        raise SystemExit(f"AnomalyCLIP checkpoint missing: {args.checkpoint}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, object]] = []
    view_reports: list[dict[str, object]] = []
    for dataset, spec in DATASETS.items():
        manifest_path = Path(spec["manifest"])
        data_root = Path(spec["root"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for seed in freeze["seeds"]:
            for shot in freeze["shots"]:
                run_id = f"v2_{dataset}_s{seed}_k{shot}_branch_cache_v1"
                views_dir = args.output_root / "reference_views" / run_id
                views_json = views_dir / "reference_views.json"
                if not args.skip_view_generation and not views_json.exists():
                    command = [
                        sys.executable,
                        str(ROOT / "scripts" / "prepare_normal_reference_views.py"),
                        "--manifest",
                        str(manifest_path),
                        "--data-root",
                        str(data_root),
                        "--dataset",
                        dataset,
                        "--seed",
                        str(seed),
                        "--shot",
                        str(shot),
                        "--output-dir",
                        str(views_dir),
                    ]
                    completed = subprocess.run(command, cwd=ROOT, check=False)
                    if completed.returncode != 0:
                        raise SystemExit(f"reference-view preparation failed: {run_id}")
                if not views_json.is_file():
                    raise SystemExit(f"reference views missing: {views_json}")
                views = json.loads(views_json.read_text(encoding="utf-8"))
                if views["status"] != "passed":
                    raise SystemExit(f"reference views failed: {run_id}")
                if views["test_images_used"] or views["test_labels_used"]:
                    raise SystemExit(f"forbidden test evidence in views: {run_id}")
                view_reports.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "role": spec["role"],
                        "seed": seed,
                        "shot": shot,
                        "categories": len(manifest["categories"]),
                        "views": views["views"],
                        "path": str(views_json.resolve()),
                        "sha256": sha256(views_json),
                    }
                )
                output_base = ROOT / "outputs" / "dynamic_fusion" / "v2_branch_cache" / run_id
                commands = {
                    "anomalydino_visual": [
                        str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe"),
                        str(ROOT / "scripts" / "export_anomalydino_normal_references.py"),
                        "--views-json", str(views_json),
                        "--manifest", str(manifest_path),
                        "--data-root", str(data_root),
                        "--output-dir", str(output_base / "anomalydino_visual"),
                        "--dataset", dataset,
                        "--device", "cuda:0",
                    ],
                    "anomalyclip_text": [
                        str(ROOT / ".venv-anomalyclip" / "Scripts" / "python.exe"),
                        str(ROOT / "scripts" / "export_anomalyclip_normal_references.py"),
                        "--views-json", str(views_json),
                        "--checkpoint", str(args.checkpoint),
                        "--output-dir", str(output_base / "anomalyclip_text"),
                        "--device", "cuda",
                    ],
                }
                for branch, command in commands.items():
                    jobs.append(
                        {
                            "run_id": f"{run_id}_{branch}",
                            "dataset": dataset,
                            "role": spec["role"],
                            "seed": seed,
                            "shot": shot,
                            "branch": branch,
                            "status": "prepared_not_started",
                            "gpu_required": True,
                            "test_predictions_used": False,
                            "test_labels_used": False,
                            "holdout_metrics_allowed": False,
                            "command": command,
                            "validate_only_command": command + ["--validate-only"],
                        }
                    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_started",
        "execution_authorized": False,
        "gpu_used": False,
        "data_protocol_freeze": str(freeze_path.resolve()),
        "data_protocol_freeze_sha256": sha256(freeze_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "development_dataset": "mpdd",
        "holdout_dataset": "btad",
        "holdout_metrics_allowed": False,
        "reference_views": view_reports,
        "jobs": jobs,
    }
    queue_path = args.output_root / "queue.json"
    queue_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "views": len(view_reports), "jobs": len(jobs), "queue": str(queue_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
