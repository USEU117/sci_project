"""Build a reproducible cross-method VisA mean/std table from saved summaries."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = ROOT / "experiments" / "summaries"
OUTPUT = SUMMARY_ROOT / "visa_baseline_main_table_20260803.csv"
METHODS = {
    "PatchCore": SUMMARY_ROOT / "patchcore_visa_unified" / "by_shot.csv",
    "WinCLIP+": SUMMARY_ROOT / "winclip_visa_unified" / "by_shot.csv",
    "AnomalyDINO": SUMMARY_ROOT / "anomalydino_visa_unified" / "by_shot.csv",
    "PromptAD": SUMMARY_ROOT / "promptad_visa_unified" / "by_shot.csv",
}
METRICS = [
    "image_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_auroc",
    "pixel_ap",
    "aupro",
]


def main() -> None:
    rows: list[dict[str, str]] = []
    for method, path in METHODS.items():
        if not path.is_file():
            raise SystemExit(f"missing summary: {path}")
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                output = {"method": method, "shot": row["shot"], "seed_count": row["seed_count"]}
                for metric in METRICS:
                    output[f"{metric}_mean"] = row[f"{metric}_mean"]
                    output[f"{metric}_std"] = row[f"{metric}_std"]
                rows.append(output)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
