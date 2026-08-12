"""Build an evidence-only MVTec completeness matrix and paper-table template."""
from __future__ import annotations

import csv
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/summaries"
COMBOS = [(seed, shot) for seed in (0, 1, 2) for shot in (1, 2, 4)]
METHODS = ("PatchCore", "WinCLIP+", "AnomalyDINO", "PromptAD", "DynamicFusion")


def paths(method: str, seed: int, shot: int) -> tuple[Path, Path]:
    if method == "PatchCore":
        root = ROOT / f"outputs/unified/patchcore_mvtec_seed_{seed}_shot_{shot}"
    elif method == "WinCLIP+":
        root = ROOT / f"outputs/unified/winclip_mvtec_seed_{seed}_shot_{shot}"
    elif method == "AnomalyDINO":
        root = ROOT / f"outputs/unified/anomalydino_mvtec_seed_{seed}_shot_{shot}"
    elif method == "PromptAD":
        root = ROOT / f"outputs/unified/promptad_mvtec_seed_{seed}_shot_{shot}"
    else:
        root = ROOT / f"outputs/dynamic_fusion/final_validation/20260805_mvtec_final_validation_s{seed}_k{shot}/evaluation"
    return root / "evaluation_report.json", root / "summary.csv"


def read_row(method: str, seed: int, shot: int) -> dict[str, object]:
    report_path, summary_path = paths(method, seed, shot)
    row: dict[str, object] = {"method": method, "seed": seed, "shot": shot, "status": "missing", "report_path": str(report_path.relative_to(ROOT)), "summary_path": str(summary_path.relative_to(ROOT)), "category_count": "", "sample_count": "", "validation_errors": "", "image_auroc": "", "pixel_auroc": "", "pixel_ap": "", "aupro": ""}
    if not report_path.exists() or not summary_path.exists():
        return row
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row.update({"category_count": report.get("category_count"), "sample_count": report.get("sample_count"), "validation_errors": report.get("validation_errors")})
    if report.get("category_count") != 15 or report.get("sample_count") != 1725 or report.get("validation_errors") != 0:
        row["status"] = "invalid"
        return row
    with summary_path.open(encoding="utf-8") as stream:
        macro = next((candidate for candidate in csv.DictReader(stream) if candidate.get("category") == "macro_mean"), None)
    if macro is None:
        row["status"] = "invalid"
        return row
    for key in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro"):
        row[key] = macro.get(key, "")
    row["status"] = "complete"
    return row


def display(values: list[float]) -> str:
    if len(values) != 3:
        return f"-- (n={len(values)}/3)"
    return f"{np.mean(values) * 100:.2f} ± {np.std(values, ddof=1) * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y%m%d"),
        help="Date suffix used for generated evidence files (YYYYMMDD).",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [read_row(method, seed, shot) for method in METHODS for seed, shot in COMBOS]
    fields = list(rows[0])
    completeness = OUT / f"mvtec_method_seed_shot_completeness_{args.date}.csv"
    with completeness.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    table_rows = []
    for method in METHODS:
        for shot in (1, 2, 4):
            selected = [row for row in rows if row["method"] == method and row["shot"] == shot and row["status"] == "complete"]
            table_rows.append({
                "method": method, "shot": shot, "complete_seeds": ",".join(str(row["seed"]) for row in selected) or "none",
                "image_auroc_mean_std": display([float(row["image_auroc"]) for row in selected]),
                "pixel_auroc_mean_std": display([float(row["pixel_auroc"]) for row in selected]),
                "pixel_ap_mean_std": display([float(row["pixel_ap"]) for row in selected]),
                "aupro_mean_std": display([float(row["aupro"]) for row in selected]),
                "paper_ready": "yes" if len(selected) == 3 else "no",
            })
    template = OUT / f"mvtec_paper_main_table_template_{args.date}.csv"
    with template.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0])); writer.writeheader(); writer.writerows(table_rows)
    payload = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "paper_table": table_rows, "policy": "Only rows with all three seeds completed receive mean ± std and paper_ready=yes. AnomalyCLIP zero-shot, ReMP-AD and AdaptCLIP are excluded from the few-shot template."}
    (OUT / f"mvtec_method_seed_shot_completeness_{args.date}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
