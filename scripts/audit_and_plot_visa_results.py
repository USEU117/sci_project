from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_DIR = {
    "PatchCore": "patchcore",
    "WinCLIP+": "winclip",
    "AnomalyDINO": "anomalydino",
    "PromptAD": "promptad",
}
METHOD_COLORS = {
    "PatchCore": "#4C78A8",
    "WinCLIP+": "#F58518",
    "AnomalyDINO": "#54A24B",
    "PromptAD": "#E45756",
}
SHOTS = (1, 2, 4)
SEEDS = (0, 1, 2)
METRICS = (
    "image_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_auroc",
    "pixel_ap",
    "aupro",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def audit(root: Path) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    unified = root / "outputs" / "unified"
    for method, prefix in METHOD_DIR.items():
        for shot in SHOTS:
            for seed in SEEDS:
                run_id = f"{prefix}_visa_seed_{seed}_shot_{shot}"
                run_dir = unified / run_id
                required = {
                    "report": run_dir / "evaluation_report.json",
                    "summary": run_dir / "summary.csv",
                    "category": run_dir / "per_category.csv",
                    "image": run_dir / "per_image.csv",
                }
                missing = [name for name, path in required.items() if not path.is_file()]
                if missing:
                    errors.append(f"{run_id}: missing {', '.join(missing)}")
                    continue
                try:
                    report = json.loads(required["report"].read_text(encoding="utf-8"))
                    summary = read_single_row(required["summary"])
                    category = pd.read_csv(required["category"])
                    image = pd.read_csv(required["image"])
                except Exception as exc:
                    errors.append(f"{run_id}: read failure: {exc}")
                    continue

                checks = {
                    "score_direction": report.get("score_direction") == "higher_is_more_anomalous",
                    "category_count": int(report.get("category_count", -1)) == 12,
                    "sample_count": int(report.get("sample_count", -1)) == 2162,
                    "validation_errors": int(report.get("validation_errors", -1)) == 0,
                    "per_category_rows": len(category) == 12,
                    "per_image_rows": len(image) == 2162,
                    "metric_fields": all(metric in summary for metric in METRICS),
                }
                values = {}
                for metric in METRICS:
                    try:
                        value = float(summary[metric])
                    except (KeyError, TypeError, ValueError):
                        value = math.nan
                    values[metric] = value
                checks["finite_metrics"] = all(math.isfinite(value) for value in values.values())
                for name, passed in checks.items():
                    if not passed:
                        errors.append(f"{run_id}: failed check {name}")

                records.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "dataset": "VisA",
                        "shot": shot,
                        "seed": seed,
                        "category_count": report.get("category_count"),
                        "sample_count": report.get("sample_count"),
                        "score_direction": report.get("score_direction"),
                        "validation_errors": report.get("validation_errors"),
                        "target_normal_tuning": method == "PromptAD",
                        **values,
                        "report_path": str(required["report"].relative_to(root)),
                        "report_sha256": sha256(required["report"]),
                        "summary_path": str(required["summary"].relative_to(root)),
                        "summary_sha256": sha256(required["summary"]),
                    }
                )
    return records, errors


def plot_method_comparison(main: pd.DataFrame, output: Path) -> None:
    metrics = [("image_auroc_mean", "Image AUROC"), ("pixel_auroc_mean", "Pixel AUROC"), ("aupro_mean", "AUPRO")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x = np.arange(len(SHOTS))
    width = 0.19
    methods = list(METHOD_DIR)
    for axis, (column, title) in zip(axes, metrics):
        for index, method in enumerate(methods):
            rows = main[main["method"] == method].set_index("shot").loc[list(SHOTS)]
            axis.bar(
                x + (index - 1.5) * width,
                rows[column] * 100,
                width,
                label=method,
                color=METHOD_COLORS[method],
            )
        axis.set_title(title)
        axis.set_xticks(x, [f"{shot}-shot" for shot in SHOTS])
        axis.set_ylabel("Score (%)")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("VisA baseline comparison (mean over 3 seeds)")
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shot_trends(main: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("image_auroc_mean", "image_auroc_std", "Image AUROC"),
        ("image_ap_mean", "image_ap_std", "Image AP"),
        ("image_f1_max_mean", "image_f1_max_std", "Image F1-max"),
        ("pixel_auroc_mean", "pixel_auroc_std", "Pixel AUROC"),
        ("pixel_ap_mean", "pixel_ap_std", "Pixel AP"),
        ("aupro_mean", "aupro_std", "AUPRO"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (mean_col, std_col, title) in zip(axes.flat, metrics):
        for method in METHOD_DIR:
            rows = main[main["method"] == method].set_index("shot").loc[list(SHOTS)]
            axis.errorbar(
                SHOTS,
                rows[mean_col] * 100,
                yerr=rows[std_col] * 100,
                marker="o",
                capsize=3,
                linewidth=2,
                label=method,
                color=METHOD_COLORS[method],
            )
        axis.set_title(title)
        axis.set_xticks(SHOTS, ["1", "2", "4"])
        axis.set_xlabel("Number of normal references (shot)")
        axis.set_ylabel("Score (%)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Effect of shot count on VisA performance (mean +/- std over 3 seeds)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_category_heatmap(category: pd.DataFrame, output: Path) -> None:
    categories = list(dict.fromkeys(category["category"].tolist()))
    methods = list(METHOD_DIR)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "image_auroc_mean", "4-shot image AUROC by category"),
        (axes[1], "aupro_mean", "4-shot AUPRO by category"),
    ):
        subset = category[category["shot"] == 4]
        matrix = np.array(
            [
                [float(subset[(subset["method"] == method) & (subset["category"] == cat)][metric].iloc[0]) * 100 for cat in categories]
                for method in methods
            ]
        )
        image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=100)
        axis.set_title(title)
        axis.set_yticks(np.arange(len(methods)), methods)
        axis.set_xticks(np.arange(len(categories)), categories, rotation=45, ha="right")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center", fontsize=6, color="white" if matrix[row, col] < 55 else "black")
        fig.colorbar(image, ax=axis, fraction=0.018, pad=0.01, label="Score (%)")
    fig.suptitle("VisA per-category comparison (mean over 3 seeds)")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--date", default="20260804")
    args = parser.parse_args()
    root = args.root.resolve()
    summary_dir = root / "experiments" / "summaries"
    figure_dir = summary_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    records, errors = audit(root)
    audit_csv = summary_dir / f"visa_result_audit_{args.date}.csv"
    pd.DataFrame(records).to_csv(audit_csv, index=False)
    report = {
        "schema_version": 1,
        "dataset": "VisA",
        "expected_methods": list(METHOD_DIR),
        "expected_shots": list(SHOTS),
        "expected_seeds": list(SEEDS),
        "expected_runs": 36,
        "audited_runs": len(records),
        "errors": errors,
        "status": "passed" if len(records) == 36 and not errors else "failed",
        "score_direction": "higher_is_more_anomalous",
        "promptad_target_normal_tuning": True,
        "audit_csv": str(audit_csv.relative_to(root)),
        "audit_csv_sha256": sha256(audit_csv),
    }
    audit_json = summary_dir / f"visa_result_audit_{args.date}.json"
    audit_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    main_table = pd.read_csv(summary_dir / "visa_baseline_main_table_20260803.csv")
    category_table = pd.read_csv(summary_dir / "visa_per_category_mean_std_20260803.csv")
    plot_method_comparison(main_table, figure_dir / f"visa_method_comparison_{args.date}.png")
    plot_shot_trends(main_table, figure_dir / f"visa_shot_trends_{args.date}.png")
    plot_category_heatmap(category_table, figure_dir / f"visa_category_heatmap_{args.date}.png")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
