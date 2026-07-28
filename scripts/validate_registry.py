"""Validate the experiment registry structure and required identifiers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "registry", type=Path, nargs="?", default=Path("experiments/registry.csv")
    )
    args = ap.parse_args()
    with args.registry.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise SystemExit("registry is empty")
    header = rows[0]
    errors: list[str] = []
    ids: set[str] = set()
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            errors.append(
                f"line {line_number}: expected {len(header)} columns, got {len(row)}"
            )
            continue
        experiment_id = row[0].strip()
        if not experiment_id:
            errors.append(f"line {line_number}: empty experiment_id")
        elif experiment_id in ids:
            errors.append(f"line {line_number}: duplicate experiment_id {experiment_id}")
        ids.add(experiment_id)
    print(
        f"registry={args.registry} columns={len(header)} experiments={len(rows)-1} "
        f"errors={len(errors)}"
    )
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
