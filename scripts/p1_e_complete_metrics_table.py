"""Aggregate the 36 existing A1 complete-metric reports into paper tables.

This is read-only with respect to experiment inputs: it does not run feature
extraction or inference. It validates 4 datasets x 3 seeds x 3 shots, checks
the six image/pixel metrics, and reconciles Pixel-AP with the P0 reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "experiments/dynamic_fusion/v3_direction_a/a1_complete_metrics_20260819"
P0_ROOT = ROOT / "submission_repro_20260827/evidence/per_config"
OUT_ROOT = ROOT / "submission_repro_20260827/evidence/p1"
DATASETS = {"mpdd": 6, "btad": 3, "visa": 12, "mvtec": 15}
ROLES = {
    "mpdd": "development",
    "btad": "external_frozen_validation",
    "visa": "in_domain_frozen_validation",
    "mvtec": "external_frozen_validation",
}
SEEDS = (0, 1, 2)
SHOTS = (1, 2, 4)
METHODS = ("concat", "dino")
METRICS = ("image_auroc", "image_ap", "image_f1_max", "pixel_auroc", "pixel_ap", "pixel_aupro")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(method_payload: dict) -> dict[str, float]:
    return {
        "image_auroc": float(method_payload["mean"]["image"]["image_auroc"]),
        "image_ap": float(method_payload["mean"]["image"]["image_ap"]),
        "image_f1_max": float(method_payload["mean"]["image"]["image_f1_max"]),
        "pixel_auroc": float(method_payload["mean"]["pixel"]["pixel_auroc"]),
        "pixel_ap": float(method_payload["mean"]["pixel"]["pixel_ap"]),
        "pixel_aupro": float(method_payload["mean"]["pixel"]["pixel_aupro"]),
    }


def summarize(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        rec = dict(zip(group_keys, key))
        rec["n"] = len(items)
        for metric in METRICS:
            values = np.asarray([r[metric] for r in items], dtype=np.float64)
            rec[f"{metric}_mean"] = round(float(values.mean()), 6)
            rec[f"{metric}_std"] = round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = []
    sources = []
    max_p0_delta = 0.0
    for dataset, n_categories in DATASETS.items():
        for seed in SEEDS:
            for shot in SHOTS:
                path = REPORT_ROOT / dataset / f"seed{seed}_k{shot}" / "metrics_report.json"
                if not path.is_file():
                    raise SystemExit(f"missing report: {path}")
                report = json.loads(path.read_text(encoding="utf-8"))
                if (report["dataset"], report["seed"], report["shot"], report["dataset_role"]) != (
                    dataset, seed, shot, ROLES[dataset]
                ):
                    raise SystemExit(f"metadata mismatch: {path}")
                if report.get("image_score") != "max_pool" or report.get("stride") != 8:
                    raise SystemExit(f"metric protocol mismatch: {path}")
                p0 = json.loads((P0_ROOT / f"{dataset}_s{seed}_k{shot}.json").read_text(encoding="utf-8"))
                for method in METHODS:
                    values = flatten(report[method])
                    if len(report[method]["per_category"]) != n_categories:
                        raise SystemExit(f"category count mismatch: {path} {method}")
                    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values.values()):
                        raise SystemExit(f"invalid metric: {path} {method}")
                    p0_key = "mean_concat_pixel_ap" if method == "concat" else "mean_feature_dino_only_pixel_ap"
                    delta = abs(values["pixel_ap"] - float(p0[p0_key]))
                    max_p0_delta = max(max_p0_delta, delta)
                    rows.append({
                        "dataset": dataset,
                        "role": ROLES[dataset],
                        "seed": seed,
                        "shot": shot,
                        "method": "A1_concat" if method == "concat" else "feature_DINO_only",
                        **{k: round(v, 6) for k, v in values.items()},
                        "p0_pixel_ap_abs_delta": round(delta, 6),
                    })
                sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    if len(rows) != 72 or max_p0_delta > 5e-4:
        raise SystemExit(f"acceptance failed: rows={len(rows)}, max_p0_delta={max_p0_delta}")

    shot_wise = summarize(rows, ("dataset", "role", "shot", "method"))
    dataset_wise = summarize(rows, ("dataset", "role", "method"))
    dataset_deltas = []
    for dataset in DATASETS:
        concat = next(r for r in dataset_wise if r["dataset"] == dataset and r["method"] == "A1_concat")
        dino = next(r for r in dataset_wise if r["dataset"] == dataset and r["method"] == "feature_DINO_only")
        dataset_deltas.append({
            "dataset": dataset,
            "role": ROLES[dataset],
            **{f"delta_{metric}": round(concat[f"{metric}_mean"] - dino[f"{metric}_mean"], 6)
               for metric in METRICS},
        })
    payload = {
        "schema_version": 1,
        "kind": "p1_e_complete_metrics_table",
        "source_reports": sources,
        "protocol": {"image_score": "max_pool", "pixel_stride": 8, "seeds": list(SEEDS), "shots": list(SHOTS)},
        "checks": {"reports": 36, "config_method_rows": 72, "six_metrics_complete": True,
                   "max_p0_pixel_ap_abs_delta": round(max_p0_delta, 6), "all_passed": True},
        "config_rows": rows,
        "shot_wise": shot_wise,
        "dataset_wise": dataset_wise,
        "dataset_deltas_A1_minus_DINO": dataset_deltas,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "p1_e_complete_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT_ROOT / "p1_e_complete_metrics_per_config.csv", rows)
    write_csv(OUT_ROOT / "p1_e_complete_metrics_shot_wise.csv", shot_wise)

    md = [
        "# P1-E A1 完整指标投稿表",
        "",
        "输入为已存在的36份 complete-metrics reports；本步骤只聚合，不重新运行模型。图像分数为 anomaly map max-pool，像素指标 stride=8。",
        "",
        f"验收：36/36 reports，72 method-config rows，六项指标完整；相对 P0 Pixel-AP 最大绝对差 `{max_p0_delta:.6g}`。",
        "",
        "## 按数据集汇总（9配置 mean ± std）",
        "",
        "| dataset | role | method | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in dataset_wise:
        cells = [f"{r[m + '_mean']:.4f} ± {r[m + '_std']:.4f}" for m in METRICS]
        md.append(f"| {r['dataset']} | {r['role']} | {r['method']} | " + " | ".join(cells) + " |")
    md += [
        "",
        "## A1 − feature-DINO-only（9配置均值之差）",
        "",
        "| dataset | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in dataset_deltas:
        cells = [f"{r['delta_' + metric]:+.4f}" for metric in METRICS]
        md.append(f"| {r['dataset']} | " + " | ".join(cells) + " |")
    md += [
        "",
        "边界：四数据集稳定正结论针对 Pixel-AP。BTAD 的 Image-AP 与 Image-F1-max 为负，不得写成所有检测/定位指标全面提升。",
        "",
        "注意：9配置是同一测试集上的3 seed×3 shot参考采样，不是9个独立数据集。VisA为in-domain frozen validation。",
    ]
    (OUT_ROOT / "p1_e_complete_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "passed", "reports": 36, "rows": 72,
                      "max_p0_pixel_ap_abs_delta": max_p0_delta}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
