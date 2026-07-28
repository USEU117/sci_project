"""Summarize unified per-run CSV results as mean and sample standard deviation."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

METRICS = [
    "image_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_auroc",
    "pixel_ap",
    "aupro",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    pattern = re.compile(
        rf"^{re.escape(args.prefix)}_seed_(?P<seed>\d+)_shot_(?P<shot>\d+)$"
    )
    rows: list[dict] = []
    for directory in sorted(args.root.iterdir()):
        match = pattern.match(directory.name) if directory.is_dir() else None
        summary = directory / "summary.csv"
        if not match or not summary.is_file():
            continue
        with summary.open(newline="", encoding="utf-8") as stream:
            values = next(csv.DictReader(stream))
        rows.append(
            {
                "seed": int(match.group("seed")),
                "shot": int(match.group("shot")),
                "sample_count": int(values["sample_count"]),
                **{metric: float(values[metric]) for metric in METRICS},
                "source": str(summary.resolve()),
            }
        )
    expected = {(seed, shot) for seed in (0, 1, 2) for shot in (1, 2, 4)}
    actual = {(row["seed"], row["shot"]) for row in rows}
    if actual != expected:
        raise SystemExit(
            f"incomplete matrix: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )

    by_shot: list[dict] = []
    for shot in (1, 2, 4):
        selected = [row for row in rows if row["shot"] == shot]
        result = {"shot": shot, "seed_count": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values)
        by_shot.append(result)

    write_csv(args.output_dir / "per_run.csv", rows)
    write_csv(args.output_dir / "by_shot.csv", by_shot)
    for row in by_shot:
        print(row)


if __name__ == "__main__":
    main()
