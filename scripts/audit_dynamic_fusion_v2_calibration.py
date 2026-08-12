"""Audit V2 calibration rank preservation, saturation and leakage fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import calibration_diagnostics, load_v2_category_calibrations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--visual-reference-dir", type=Path, required=True)
    parser.add_argument("--text-reference-dir", type=Path, required=True)
    parser.add_argument("--minimum-spearman", type=float, default=0.999)
    parser.add_argument("--maximum-boundary-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for category in sorted(payload.get("categories", {})):
        visual_calibration, text_calibration = load_v2_category_calibrations(payload, category)
        for branch, directory, fitted in (
            ("visual", args.visual_reference_dir, visual_calibration),
            ("text", args.text_reference_dir, text_calibration),
        ):
            path = directory / f"{category}.npz"
            with np.load(path, allow_pickle=False) as data:
                raw_image = data["image_scores"]
                maps = np.asarray(data["pixel_maps"])
            if maps.ndim == 4 and maps.shape[1] == 1:
                maps = maps[:, 0]
            raw_pixel_tail = np.quantile(maps.reshape(len(maps), -1), 0.99, axis=1)
            for level, raw, calibrator in (
                ("image", raw_image, fitted.image),
                ("pixel_tail", raw_pixel_tail, fitted.pixel),
            ):
                stats = calibration_diagnostics(raw, calibrator.transform(raw))
                row = {"category": category, "branch": branch, "level": level, **stats}
                rows.append(row)
                if stats["spearman_raw_vs_calibrated"] < args.minimum_spearman:
                    failures.append(f"{category}/{branch}/{level}: rank preservation failed")
                boundary_rate = stats["lower_boundary_rate"] + stats["upper_boundary_rate"]
                if boundary_rate > args.maximum_boundary_rate:
                    failures.append(f"{category}/{branch}/{level}: boundary saturation failed")
    report = {
        "schema_version": 2,
        "status": "passed" if rows and not failures else "failed",
        "calibration": str(args.calibration_json.resolve()),
        "minimum_spearman": args.minimum_spearman,
        "maximum_boundary_rate": args.maximum_boundary_rate,
        "test_predictions_used": False,
        "test_labels_used": False,
        "test_masks_used": False,
        "test_set_statistics_used": False,
        "rows": rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "failures": len(failures)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
