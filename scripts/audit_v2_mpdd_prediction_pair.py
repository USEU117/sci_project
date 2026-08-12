"""Schema, alignment and provenance audit for an MPDD V2 prediction pair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from v2_mpdd_prediction_common import index_dataset, sha256


REQUIRED = {"gt_sp", "pr_sp", "imgs_masks", "anomaly_maps", "sample_ids", "dataset", "dataset_role", "branch", "seed", "shot", "score_direction"}


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return str(value.reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--dataset-role", choices=("development", "holdout"), default="development")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--allow-invariant-text-reuse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indexed = index_dataset(args.dataset, args.data_root)
    failures: list[str] = []
    rows = []
    total = 0
    for category, samples in sorted(indexed.items()):
        expected_ids = [sample.sample_id for sample in samples]
        expected_labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
        branch_data = {}
        for branch, directory in (("anomalydino_visual", args.visual_dir), ("anomalyclip_text", args.text_dir)):
            path = directory / f"{category}.npz"
            try:
                with np.load(path, allow_pickle=False) as data:
                    missing = REQUIRED.difference(data.files)
                    if missing:
                        raise ValueError(f"missing keys {sorted(missing)}")
                    ids = [str(value) for value in data["sample_ids"].tolist()]
                    labels = np.asarray(data["gt_sp"], dtype=np.int64)
                    scores = np.asarray(data["pr_sp"], dtype=np.float64)
                    masks = np.asarray(data["imgs_masks"])
                    maps = np.asarray(data["anomaly_maps"], dtype=np.float64)
                    if ids != expected_ids:
                        raise ValueError(f"sample_ids differ from deterministic {args.dataset.upper()} index")
                    if not np.array_equal(labels, expected_labels):
                        raise ValueError(f"image labels differ from {args.dataset.upper()} folders")
                    if len(ids) != len(scores) or masks.shape[0] != len(ids) or maps.shape[0] != len(ids):
                        raise ValueError("array lengths differ")
                    if masks.shape != maps.shape:
                        raise ValueError("mask/map shapes differ within branch")
                    if not np.isfinite(scores).all() or not np.isfinite(maps).all():
                        raise ValueError("non-finite predictions")
                    if scalar(data, "dataset") != args.dataset or scalar(data, "dataset_role") != args.dataset_role:
                        raise ValueError("dataset boundary metadata differs")
                    if scalar(data, "branch") != branch:
                        raise ValueError("branch metadata differs")
                    source_seed = int(scalar(data, "seed"))
                    source_shot = int(scalar(data, "shot"))
                    if (source_seed, source_shot) != (args.seed, args.shot):
                        if branch != "anomalyclip_text" or not args.allow_invariant_text_reuse:
                            raise ValueError("seed/shot metadata differs")
                    if scalar(data, "score_direction") != "higher_is_more_anomalous":
                        raise ValueError("score direction differs")
                    branch_data[branch] = {"ids": ids, "labels": labels}
                rows.append({"category": category, "branch": branch, "status": "passed", "samples": len(ids), "path": str(path.resolve()), "sha256": sha256(path)})
            except Exception as exc:
                failures.append(f"{category}/{branch}: {exc}")
                rows.append({"category": category, "branch": branch, "status": "failed", "errors": [str(exc)], "path": str(path.resolve())})
        if len(branch_data) == 2:
            if branch_data["anomalydino_visual"]["ids"] != branch_data["anomalyclip_text"]["ids"]:
                failures.append(f"{category}: cross-branch sample alignment failed")
            if not np.array_equal(branch_data["anomalydino_visual"]["labels"], branch_data["anomalyclip_text"]["labels"]):
                failures.append(f"{category}: cross-branch labels differ")
        total += len(samples)

    for directory in (args.visual_dir, args.text_dir):
        report_path = directory / "export_report.json"
        if not report_path.is_file():
            failures.append(f"export report missing: {report_path}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "passed" or report.get("dataset") != args.dataset or report.get("dataset_role") != args.dataset_role:
            failures.append(f"export report failed or role differs: {report_path}")
        if directory == args.text_dir and (report.get("seed"), report.get("shot")) != (args.seed, args.shot):
            failures.append(f"text export report target seed/shot differs: {report_path}")
        if directory == args.text_dir and args.allow_invariant_text_reuse:
            if report.get("prediction_invariant_to_few_shot_selection") is not True:
                failures.append(f"text reuse was not explicitly declared invariant: {report_path}")
        for field in ("test_predictions_used_for_parameter_fit", "test_labels_used_for_parameter_fit", "test_set_statistics_used_for_calibration"):
            if report.get(field) is not False:
                failures.append(f"{field} must be false: {report_path}")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "seed": args.seed,
        "shot": args.shot,
        "categories": len(indexed),
        "samples": total,
        "branches": 2,
        "metrics_computed": False,
        "btad_accessed": args.dataset == "btad",
        "failures": failures,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "categories", "samples", "branches", "failures")}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
