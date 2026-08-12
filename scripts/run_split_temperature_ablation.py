"""CPU-only seed-0 split-temperature ablation with a K=1 pixel consistency audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)
CALIBRATIONS = {
    1: ROOT / "outputs/dynamic_fusion/normal_reference_predictions/20260731_visa_s0_k1_real_reference_v6_q99/calibration.json",
    2: ROOT / "outputs/dynamic_fusion/normal_reference_predictions/20260804_visa_s0_k2_real_reference_v1_q99/calibration.json",
    4: ROOT / "outputs/dynamic_fusion/normal_reference_predictions/20260804_visa_s0_k4_real_reference_v1_q99/calibration.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(category: str, calibration: Path, output: Path, *, image: float, pixel: float) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts/run_dynamic_fusion_cache.py"),
        "--visual-cache", str(ROOT / f"outputs/anomalydino/unified_matrix/seed_0_shot_1/predictions/{category}.npz"),
        "--text-cache", str(ROOT / f"outputs/anomalyclip/visa_all_518_cached/{category}.npz"),
        "--text-sidecar", str(ROOT / f"outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified/{category}.sample_ids.npz"),
        "--calibration-json", str(calibration), "--category", category,
        "--fusion-mode", "dynamic", "--temperature", "0.20",
        "--image-temperature", str(image), "--pixel-temperature", str(pixel),
        "--decision-margin", "0.15", "--min-weight", "0.05",
        "--output", str(output),
    ]


def run(command: list[str], log: list[str]) -> None:
    log.append(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_calibration(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError(f"Calibration did not pass: {path}")
    if payload.get("test_predictions_used") or payload.get("test_labels_used"):
        raise ValueError(f"Calibration is not normal-reference-only: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260805_visa_s0_split_temperature_k1_check")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        raise ValueError("Use one evaluator worker to avoid the documented memory failure.")

    output_root = ROOT / "outputs/dynamic_fusion/split_temperature" / args.run_id
    experiment_root = ROOT / "experiments/dynamic_fusion" / args.run_id
    if output_root.exists() or experiment_root.exists():
        raise FileExistsError(f"Run ID already exists: {args.run_id}")
    for calibration in CALIBRATIONS.values():
        ensure_calibration(calibration)

    output_root.mkdir(parents=True)
    experiment_root.mkdir(parents=True)
    commands: list[str] = []
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "purpose": "VisA seed-0 split-temperature ablation and K=1 pixel consistency check",
        "dataset": "visa", "seed": 0, "shots": [1, 2, 4],
        "gpu_used": False,
        "test_predictions_used_by_router": False,
        "test_labels_used_by_router": False,
        "evaluation_uses_ground_truth_only_after_fusion": True,
        "image_temperature": 0.50, "pixel_temperature": 0.20,
        "decision_margin": 0.15, "min_weight": 0.05,
        "calibrations": {str(k): {"path": str(v.resolve()), "sha256": sha256(v)} for k, v in CALIBRATIONS.items()},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (experiment_root / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        # K=1 control proves that changing only image temperature leaves pixel
        # weights/maps identical to the single-temperature 0.20 implementation.
        control_dir = output_root / "k1_single_t020_control"
        split_dirs = {shot: output_root / f"k{shot}_image_t050_pixel_t020" for shot in (1, 2, 4)}
        for directory in (control_dir, *split_dirs.values()):
            directory.mkdir(parents=True)
        for category in CATEGORIES:
            run(command_for(category, CALIBRATIONS[1], control_dir / f"{category}.npz", image=0.20, pixel=0.20), commands)
        for shot, directory in split_dirs.items():
            for category in CATEGORIES:
                run(command_for(category, CALIBRATIONS[shot], directory / f"{category}.npz", image=0.50, pixel=0.20), commands)
            evaluation = directory / "evaluation"
            run([sys.executable, str(ROOT / "scripts/evaluate_unified.py"), "--cache-dir", str(directory), "--output-dir", str(evaluation), "--workers", "1"], commands)

        checks = []
        for category in CATEGORIES:
            with np.load(control_dir / f"{category}.npz", allow_pickle=False) as control, np.load(split_dirs[1] / f"{category}.npz", allow_pickle=False) as split:
                checks.append({
                    "category": category,
                    "pixel_weights_identical": bool(np.allclose(control["visual_pixel_weights"], split["visual_pixel_weights"], rtol=0, atol=1e-7)),
                    "pixel_maps_identical": bool(np.allclose(control["anomaly_maps"], split["anomaly_maps"], rtol=0, atol=1e-7)),
                    "image_weights_changed": bool(not np.allclose(control["visual_weights"], split["visual_weights"], rtol=0, atol=1e-7)),
                })
        if not all(row["pixel_weights_identical"] and row["pixel_maps_identical"] and row["image_weights_changed"] for row in checks):
            raise RuntimeError("K=1 split-temperature consistency audit failed")
        summaries = {}
        for shot, directory in split_dirs.items():
            summaries[str(shot)] = next(__import__("csv").DictReader((directory / "evaluation/summary.csv").open(encoding="utf-8")))
        report = {"schema_version": 1, "status": "passed", "run_id": args.run_id, "scope": "development_only_seed_0", "k1_consistency": checks, "summaries": summaries, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        (experiment_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (experiment_root / "decision.md").write_text("# Split-temperature ablation\n\nK=1 confirms that only image-level routing changed; pixel weights and maps remain identical to the 0.20 control. K=1/2/4 metrics are seed-0 development evidence only.\n", encoding="utf-8")
    except Exception as error:
        (experiment_root / "report.json").write_text(json.dumps({"schema_version": 1, "status": "failed", "failure": str(error)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        (experiment_root / "command.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
