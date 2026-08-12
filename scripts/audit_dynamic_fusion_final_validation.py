"""Audit frozen dynamic-fusion final-validation outputs without rerunning models."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "outputs/dynamic_fusion/final_validation"
AUDIT_ROOT = ROOT / "experiments/dynamic_fusion/final_validation_audit_20260808"
SUMMARY_ROOT = ROOT / "experiments/summaries"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(data[key]).item())


def parse_name(name: str) -> tuple[str, int, int]:
    parts = name.split("_")
    dataset = "mvtec" if "mvtec" in parts else "visa"
    seed = int(next(part[1:] for part in parts if part.startswith("s") and part[1:].isdigit()))
    shot = int(next(part[1:] for part in parts if part.startswith("k") and part[1:].isdigit()))
    return dataset, seed, shot


def audit_run(directory: Path) -> dict[str, object]:
    dataset, seed, shot = parse_name(directory.name)
    evaluation_path = directory / "evaluation/evaluation_report.json"
    npz_paths = sorted(path for path in directory.glob("*.npz") if path.name != "summary.npz")
    errors: list[str] = []
    if not evaluation_path.exists():
        errors.append("missing_evaluation_report")
        evaluation: dict[str, object] = {}
    else:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    expected_categories = 12 if dataset == "visa" else 15
    expected_samples = 2162 if dataset == "visa" else 1725
    if len(npz_paths) != expected_categories:
        errors.append(f"npz_count={len(npz_paths)}/{expected_categories}")
    if evaluation.get("category_count") != expected_categories:
        errors.append(f"category_count={evaluation.get('category_count')}/{expected_categories}")
    if evaluation.get("sample_count") != expected_samples:
        errors.append(f"sample_count={evaluation.get('sample_count')}/{expected_samples}")
    if evaluation.get("validation_errors") != 0:
        errors.append(f"validation_errors={evaluation.get('validation_errors')}")

    calibration_paths: set[str] = set()
    calibration_shas: set[str] = set()
    categories: set[str] = set()
    image_temperatures: set[float] = set()
    pixel_temperatures: set[float] = set()
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            needed = {"calibration_path", "calibration_sha256", "calibration_category", "router_image_temperature", "router_pixel_temperature"}
            missing = needed.difference(data.files)
            if missing:
                errors.append(f"{path.name}:missing={sorted(missing)}")
                continue
            calibration_paths.add(scalar(data, "calibration_path"))
            calibration_shas.add(scalar(data, "calibration_sha256"))
            categories.add(scalar(data, "calibration_category"))
            image_temperatures.add(float(np.asarray(data["router_image_temperature"]).item()))
            pixel_temperatures.add(float(np.asarray(data["router_pixel_temperature"]).item()))
    calibration_ok = True
    calibration_flags: list[dict[str, object]] = []
    for value in calibration_paths:
        path = Path(value)
        if not path.exists():
            errors.append(f"missing_calibration={path}")
            calibration_ok = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_sha = sha256(path)
        expected_sha = next(iter(calibration_shas), "") if len(calibration_shas) == 1 else ""
        flag = {
            "path": str(path), "status": payload.get("status"),
            "test_predictions_used": payload.get("test_predictions_used"),
            "test_labels_used": payload.get("test_labels_used"),
            "sha256_matches_embedded": actual_sha == expected_sha,
        }
        calibration_flags.append(flag)
        if not (flag["status"] == "passed" and flag["test_predictions_used"] is False and flag["test_labels_used"] is False and flag["sha256_matches_embedded"]):
            calibration_ok = False
            errors.append("calibration_provenance_failed")
    scope = "independent_final_validation" if dataset == "mvtec" or seed in (1, 2) else "seed0_supplementary_recheck"
    return {
        "run": directory.name, "dataset": dataset, "seed": seed, "shot": shot, "scope": scope,
        "expected_categories": expected_categories, "npz_count": len(npz_paths),
        "evaluation_category_count": evaluation.get("category_count"), "evaluation_sample_count": evaluation.get("sample_count"),
        "evaluation_validation_errors": evaluation.get("validation_errors"), "calibration_file_count": len(calibration_paths),
        "calibration_category_count": len(categories), "calibration_ok": calibration_ok,
        "image_temperatures": sorted(image_temperatures), "pixel_temperatures": sorted(pixel_temperatures),
        "status": "passed" if not errors else "failed", "errors": "; ".join(errors),
        "calibrations": calibration_flags,
    }


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [audit_run(directory) for directory in sorted(FINAL_ROOT.iterdir()) if directory.is_dir() and ("visa_final" in directory.name or "mvtec_final" in directory.name)]
    simple = [{key: value for key, value in row.items() if key != "calibrations"} for row in rows]
    fields = list(simple[0])
    with (AUDIT_ROOT / "final_validation_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(simple)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed", "runs": rows}
    (AUDIT_ROOT / "final_validation_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mvtec = [row for row in rows if row["dataset"] == "mvtec"]
    visa = [row for row in rows if row["dataset"] == "visa"]
    scope = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visa_paper_primary_runs": [row["run"] for row in visa if row["scope"] == "independent_final_validation"],
        "visa_seed0_supplementary_runs": [row["run"] for row in visa if row["scope"] == "seed0_supplementary_recheck"],
        "mvtec_final_runs": [row["run"] for row in mvtec],
        "policy": "VisA seed 1/2 are independent final validation; seed 0 remains development/supplementary and is excluded from paper generalization aggregates.",
    }
    (AUDIT_ROOT / "validation_scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (AUDIT_ROOT / "decision.md").write_text("# Final-validation audit\n\nAll listed outputs were audited from frozen NPZ metadata and unified evaluation reports. VisA seed 0 rechecks are supplementary only; paper validation aggregates must use seed 1/2. MVTec outputs require complete baseline matrices before cross-method ranking.\n", encoding="utf-8")

    metrics = []
    for directory in sorted(FINAL_ROOT.iterdir()):
        if not directory.is_dir() or "mvtec_final" not in directory.name:
            continue
        summary = directory / "evaluation/summary.csv"
        if summary.exists():
            with summary.open(encoding="utf-8") as stream:
                metrics.append({"run": directory.name, **next(row for row in csv.DictReader(stream) if row["category"] == "macro_mean")})
    with (SUMMARY_ROOT / "mvtec_dynamic_fusion_final_summary_20260808.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    (SUMMARY_ROOT / "mvtec_dynamic_fusion_final_summary_20260808.json").write_text(json.dumps({"status": "passed", "run_count": len(metrics), "runs": metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
