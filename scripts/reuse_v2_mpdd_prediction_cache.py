"""Create an audited hard-link view of an invariant MPDD prediction cache."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from v2_mpdd_prediction_common import sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--dataset-role", choices=("development", "holdout"), default="development")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--invariant-to-few-shot", action="store_true")
    args = parser.parse_args()
    source_report = args.source_dir / "export_report.json"
    source = json.loads(source_report.read_text(encoding="utf-8"))
    if source.get("status") != "passed" or source.get("branch") != args.branch:
        raise SystemExit("source export report is not a passed matching branch")
    if (source.get("seed"), source.get("shot")) != (args.seed, args.shot) and not args.invariant_to_few_shot:
        raise SystemExit("cross-shot/seed reuse requires --invariant-to-few-shot")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in source["categories"]:
        source_path = Path(row["output"])
        target = args.output_dir / source_path.name
        if target.exists():
            if sha256(target) != sha256(source_path):
                raise SystemExit(f"existing target hash differs: {target}")
        else:
            os.link(source_path, target)
        rows.append({"category": row["category"], "samples": row["samples"], "output": str(target.resolve()), "sha256": sha256(target), "source": str(source_path.resolve())})
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "branch": args.branch,
        "seed": args.seed,
        "shot": args.shot,
        "prediction_invariant_to_few_shot_selection": args.invariant_to_few_shot,
        "reused_from": str(args.source_dir.resolve()),
        "source_export_report_sha256": sha256(source_report),
        "test_predictions_used_for_parameter_fit": False,
        "test_labels_used_for_parameter_fit": False,
        "test_set_statistics_used_for_calibration": False,
        "categories": rows,
    }
    (args.output_dir / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "branch": args.branch, "files": len(rows), "invariant_reuse": args.invariant_to_few_shot}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
