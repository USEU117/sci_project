"""Aggregate per-category VisA metrics across methods, shots and seeds."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "outputs" / "unified"
OUT = ROOT / "experiments" / "summaries"
METHODS = {
    "PatchCore": "patchcore_visa_seed",
    "WinCLIP+": "winclip_visa_seed",
    "AnomalyDINO": "anomalydino_visa_seed",
    "PromptAD": "promptad_visa_seed",
}
METRICS = ["image_auroc", "image_ap", "image_f1_max", "pixel_auroc", "pixel_ap", "aupro"]


def category_name(value: str) -> str:
    return value.removeprefix("mvtec_")


def main() -> None:
    long_rows: list[dict[str, str]] = []
    for method, prefix in METHODS.items():
        for seed in (0, 1, 2):
            for shot in (1, 2, 4):
                path = UNIFIED / f"{prefix}_{seed}_shot_{shot}" / "per_category.csv"
                if not path.is_file():
                    raise SystemExit(f"missing per-category file: {path}")
                with path.open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        out = {
                            "method": method,
                            "seed": str(seed),
                            "shot": str(shot),
                            "category": category_name(row["category"]),
                            "sample_count": row["sample_count"],
                        }
                        out.update({metric: row[metric] for metric in METRICS})
                        long_rows.append(out)

    OUT.mkdir(parents=True, exist_ok=True)
    long_path = OUT / "visa_per_category_long_20260803.csv"
    with long_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    for row in long_rows:
        key = (row["method"], int(row["shot"]), row["category"])
        grouped.setdefault(key, []).append(row)
    aggregate: list[dict[str, str]] = []
    for (method, shot, category), rows in sorted(grouped.items()):
        out = {"method": method, "shot": str(shot), "category": category, "seed_count": str(len(rows))}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            out[f"{metric}_mean"] = f"{statistics.mean(values):.10f}"
            out[f"{metric}_std"] = f"{statistics.stdev(values):.10f}" if len(values) > 1 else "0.0000000000"
        aggregate.append(out)
    aggregate_path = OUT / "visa_per_category_mean_std_20260803.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(f"wrote {long_path} ({len(long_rows)} rows)")
    print(f"wrote {aggregate_path} ({len(aggregate)} rows)")


if __name__ == "__main__":
    main()
