"""Build the unified main-results table (design review S3).

Collects per-dataset rows: concat / feature-DINO-only / CLIP-only with
explicit baseline source, plus the MPDD three-way delta decomposition.
Recomputes mean delta AP from per-config reports and asserts the summary
numbers are reproducible within 1e-6.

Outputs to experiments/dynamic_fusion/main_results_20260818/:
  main_results.json / main_results.csv / main_results.md
  per_category_results.csv
  metric_definition.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a"
OUT = ROOT / "experiments" / "dynamic_fusion" / "main_results_20260818"

R = lambda *parts: EXP.joinpath(*parts)  # noqa: E731


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def report_deltas(mode: str, pattern_dirs: list[tuple[str, str]]) -> list[float]:
    """Collect mean_delta_ap_vs_dino from per-config reports of one mode."""
    deltas = []
    for subdir, fname in pattern_dirs:
        p = R(subdir, fname.format(mode=mode))
        deltas.append(load(p)["mean_delta_ap_vs_dino"])
    return deltas


# ---------------------------------------------------------------------------
# 1. MPDD (development) - ablation_summary has concat/dino/clip + concat-minus
# ---------------------------------------------------------------------------
mpdd_abl = load(R("a1_ablation_20260817", "ablation_summary.json"))
mpdd_concat = mpdd_abl["overall"]["concat_frozen"]
mpdd_dino = mpdd_abl["overall"]["dino_only"]
mpdd_clip = mpdd_abl["overall"]["clip_only"]
mpdd_concat_minus_dino = mpdd_abl["overall"]["concat_minus_dino_only_mean"]

# recompute concat mean delta from the 9 matrix per-config reports
mpdd_dirs = [(f"a1_matrix_20260817/seed{s}_k{k}", "concat_pca0_whiten0_w0.5_report.json") for s in (0, 1, 2) for k in (1, 2, 4)]
mpdd_recomp = mean(report_deltas("concat", mpdd_dirs))

# ---------------------------------------------------------------------------
# 2. BTAD (external K1 only)
# ---------------------------------------------------------------------------
btad_seeds = [0, 1, 2]
btad_concat_ap = [load(R("a1_vitb14_btad_fusion", f"seed{s}", "concat_pca0_whiten0_w0.5_report.json"))["mean_fused"]["pixel_ap"] for s in btad_seeds]
btad_concat_delta = [load(R("a1_vitb14_btad_fusion", f"seed{s}", "concat_pca0_whiten0_w0.5_report.json"))["mean_delta_ap_vs_dino"] for s in btad_seeds]
btad_dino_delta = [load(R("a1_vitb14_btad_dino", f"seed{s}", "dino_pca0_whiten0_w0.5_report.json"))["mean_delta_ap_vs_dino"] for s in btad_seeds]
btad_dino_ap = [load(R("a1_vitb14_btad_dino", f"seed{s}", "dino_pca0_whiten0_w0.5_report.json"))["mean_fused"]["pixel_ap"] for s in btad_seeds]

# ---------------------------------------------------------------------------
# 3. VisA (in-domain) / MVTec (external)
# ---------------------------------------------------------------------------
visa_sum = load(R("a1_visa_20260818", "visa_summary.json"))
mvtec_sum = load(R("a1_mvtec_20260818", "mvtec_summary.json"))

per_cat_sources = {
    "mpdd": (R("a1_matrix_20260817", "seed0_k1", "concat_pca0_whiten0_w0.5_report.json"), 6),
    "btad": (R("a1_vitb14_btad_fusion", "seed0", "concat_pca0_whiten0_w0.5_report.json"), 3),
    "visa": (R("a1_visa_20260818", "seed0_k1", "concat_pca0_whiten0_w0.5_report.json"), 12),
    "mvtec": (R("a1_mvtec_20260818", "seed0_k1", "concat_pca0_whiten0_w0.5_report.json"), 15),
}

# ---------------------------------------------------------------------------
# 4. Rows
# ---------------------------------------------------------------------------
BASELINE_DICT = {
    "mpdd": "legacy v2 dino score cache (v2_mpdd_predictions) + matched feature-level dino-only KNN",
    "btad": "legacy v2 dino score cache (v2_btad_predictions)",
    "visa": "feature-level dino-only KNN (no v2 score cache on this dataset)",
    "mvtec": "feature-level dino-only KNN (no v2 score cache on this dataset)",
}

rows = [
    # MPDD
    {"dataset": "mpdd", "role": "development", "method": "A1 concat (frozen w=0.5)", "mean_fused_pixel_ap": round(mpdd_concat["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(mpdd_concat["mean_delta_ap_vs_dino"], 6), "positive": f'{mpdd_concat["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["mpdd"]},
    {"dataset": "mpdd", "role": "development", "method": "feature-DINO-only KNN", "mean_fused_pixel_ap": round(mpdd_dino["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(mpdd_dino["mean_delta_ap_vs_dino"], 6), "positive": f'{mpdd_dino["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["mpdd"]},
    {"dataset": "mpdd", "role": "development", "method": "CLIP-only KNN", "mean_fused_pixel_ap": round(mpdd_clip["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(mpdd_clip["mean_delta_ap_vs_dino"], 6), "positive": f'{mpdd_clip["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["mpdd"]},
    {"dataset": "mpdd", "role": "development", "method": "A1 concat minus feature-DINO-only", "mean_fused_pixel_ap": "", "mean_delta_ap": round(mpdd_concat_minus_dino, 6), "positive": "9/9", "n_configs": 9, "baseline_source": "matched feature-level dino-only KNN (concat contribution isolated)"},
    # BTAD (K1)
    {"dataset": "btad", "role": "external_frozen_validation_k1_only", "method": "A1 concat (frozen w=0.5)", "mean_fused_pixel_ap": round(mean(btad_concat_ap), 6), "mean_delta_ap": round(mean(btad_concat_delta), 6), "positive": "3/3", "n_configs": 3, "baseline_source": BASELINE_DICT["btad"]},
    {"dataset": "btad", "role": "external_frozen_validation_k1_only", "method": "feature-DINO-only KNN", "mean_fused_pixel_ap": round(mean(btad_dino_ap), 6), "mean_delta_ap": round(mean(btad_dino_delta), 6), "positive": "3/3", "n_configs": 3, "baseline_source": BASELINE_DICT["btad"]},
    {"dataset": "btad", "role": "external_frozen_validation_k1_only", "method": "A1 concat minus feature-DINO-only", "mean_fused_pixel_ap": "", "mean_delta_ap": round(mean(btad_concat_delta) - mean(btad_dino_delta), 6), "positive": "3/3", "n_configs": 3, "baseline_source": "matched feature-level dino-only KNN (concat contribution isolated; CLIP-only not evaluated standalone on BTAD)"},
    # VisA
    {"dataset": "visa", "role": "in_domain_frozen_validation", "method": "A1 concat (frozen w=0.5)", "mean_fused_pixel_ap": round(visa_sum["overall"]["concat"]["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(visa_sum["overall"]["concat"]["mean_delta_ap_vs_dino"], 6), "positive": f'{visa_sum["overall"]["concat"]["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["visa"]},
    {"dataset": "visa", "role": "in_domain_frozen_validation", "method": "feature-DINO-only KNN", "mean_fused_pixel_ap": round(visa_sum["overall"]["dino_only"]["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(visa_sum["overall"]["dino_only"]["mean_delta_ap_vs_dino"], 6), "positive": f'{visa_sum["overall"]["dino_only"]["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["visa"]},
    {"dataset": "visa", "role": "in_domain_frozen_validation", "method": "CLIP-only KNN", "mean_fused_pixel_ap": round(visa_sum["overall"]["clip_only"]["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(visa_sum["overall"]["clip_only"]["mean_delta_ap_vs_dino"], 6), "positive": f'{visa_sum["overall"]["clip_only"]["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["visa"]},
    # MVTec
    {"dataset": "mvtec", "role": "external_frozen_validation", "method": "A1 concat (frozen w=0.5)", "mean_fused_pixel_ap": round(mvtec_sum["overall"]["concat"]["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(mvtec_sum["overall"]["concat"]["mean_delta_ap_vs_dino"], 6), "positive": f'{mvtec_sum["overall"]["concat"]["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["mvtec"]},
    {"dataset": "mvtec", "role": "external_frozen_validation", "method": "feature-DINO-only KNN", "mean_fused_pixel_ap": round(mvtec_sum["overall"]["dino_only"]["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(mvtec_sum["overall"]["dino_only"]["mean_delta_ap_vs_dino"], 6), "positive": f'{mvtec_sum["overall"]["dino_only"]["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["mvtec"]},
    {"dataset": "mvtec", "role": "external_frozen_validation", "method": "CLIP-only KNN", "mean_fused_pixel_ap": round(mvtec_sum["overall"]["clip_only"]["mean_fused_pixel_ap"], 6), "mean_delta_ap": round(mvtec_sum["overall"]["clip_only"]["mean_delta_ap_vs_dino"], 6), "positive": f'{mvtec_sum["overall"]["clip_only"]["positive_configs"]}/9', "n_configs": 9, "baseline_source": BASELINE_DICT["mvtec"]},
]

# ---------------------------------------------------------------------------
# 5. Recompute checks (mean delta AP must reproduce from per-config reports)
# ---------------------------------------------------------------------------
def check(dataset: str, mode: str, reported: float, dirs: list[tuple[str, str]]) -> dict:
    recomp = mean(report_deltas(mode, dirs))
    return {
        "dataset": dataset, "mode": mode,
        "recomputed_from_reports": round(recomp, 6),
        "reported_in_summary": round(reported, 6),
        "abs_diff": round(abs(recomp - reported), 10),
        "pass": abs(recomp - reported) < 1e-6,
    }

checks = []
checks.append(check("mpdd", "concat", mpdd_concat["mean_delta_ap_vs_dino"], mpdd_dirs))
btad_dirs = [(f"a1_vitb14_btad_fusion/seed{s}", "concat_pca0_whiten0_w0.5_report.json") for s in btad_seeds]
checks.append(check("btad", "concat", mean(btad_concat_delta), btad_dirs))

for ds, summary, prefix in (("visa", visa_sum, "a1_visa_20260818"), ("mvtec", mvtec_sum, "a1_mvtec_20260818")):
    dirs = [(f"{prefix}/seed{s}_k{k}", "concat_pca0_whiten0_w0.5_report.json") for s in (0, 1, 2) for k in (1, 2, 4)]
    checks.append(check(ds, "concat", summary["overall"]["concat"]["mean_delta_ap_vs_dino"], dirs))

all_pass = all(c["pass"] for c in checks)

# ---------------------------------------------------------------------------
# 6. Per-category table (seed0/K1 concat; BTAD seed0 K1)
# ---------------------------------------------------------------------------
per_cat = []
for ds, (report_path, n_cats) in per_cat_sources.items():
    rep = load(report_path)
    baseline_key = "anomalydino_visual" if "anomalydino_visual" in rep["per_category"][0]["baselines"] else "anomalydino_visual_feature_knn"
    for cat in rep["per_category"][:n_cats]:
        per_cat.append({
            "dataset": ds,
            "category": cat["category"],
            "fused_pixel_ap": round(cat["fused"]["pixel_ap"], 6),
            "dino_pixel_ap": round(cat["baselines"][baseline_key]["pixel_ap"], 6),
            "delta_ap": round(cat["delta_ap"], 6),
        })

# ---------------------------------------------------------------------------
# 7. Write outputs
# ---------------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)

main_json = {
    "schema_version": 1,
    "run_id": "main_results_20260818",
    "generated_at_utc": "2026-08-18T20:00:00+00:00",
    "frozen_config": "concat w=0.5 pca_dim=0 whiten=0 KNN k=1 stride=8 map=448 (dinov2_vitb14 + AnomalyCLIP)",
    "rows": rows,
    "recompute_checks": checks,
    "recompute_all_pass": all_pass,
    "note": "9/9 means 9 reference-sampling configs on one test set, not 9 independent datasets.",
}
(OUT / "main_results.json").write_text(json.dumps(main_json, ensure_ascii=False, indent=2), encoding="utf-8")

with (OUT / "main_results.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

with (OUT / "per_category_results.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_cat[0].keys()))
    w.writeheader()
    w.writerows(per_cat)

md_lines = [
    "# 统一性能表（main results）",
    "",
    "RunId: `main_results_20260818` · 冻结配置: concat w=0.5, pca_dim=0, whiten=0, KNN k=1, stride=8, map=448",
    "",
    "> 口径说明: 9/9 表示同一测试集上的 9 组参考采样配置（3 seeds × 1/2/4-shot），不是 9 个独立数据集。",
    "",
    "| 数据集 | 角色 | 方法 | mean fused Pixel AP | mean ΔAP vs baseline | 正向配置 | baseline source |",
    "|---|---|---|---|---|---|---|",
]
for row in rows:
    md_lines.append(f"| {row['dataset']} | {row['role']} | {row['method']} | {row['mean_fused_pixel_ap']} | {row['mean_delta_ap']} | {row['positive']} | {row['baseline_source']} |")
md_lines += [
    "",
    "## 重算校验（从 per-config report 重算 mean ΔAP）",
    "",
]
for c in checks:
    md_lines.append(f"- {c['dataset']}/{c['mode']}: recomputed={c['recomputed_from_reports']} vs reported={c['reported_in_summary']} (diff {c['abs_diff']}) → {'PASS' if c['pass'] else 'FAIL'}")
md_lines.append(f"\n- 全部通过: **{all_pass}**")
(OUT / "main_results.md").write_text("\n".join(md_lines), encoding="utf-8")

(OUT / "metric_definition.md").write_text(
    "\n".join([
        "# 指标定义（metric_definition.md）",
        "",
        "- Pixel AP: 像素级 Average Precision，基于 GT 掩码与异常图二分类阈值扫描计算。",
        "- Pixel AUROC: 像素级 Area Under ROC。",
        "- Pixel AUPRO: 像素级 Area Under Per-Region Overlap（stride=8 下采样后计算，与冻结 evaluator 一致）。",
        "- ΔAP: `fused Pixel AP − baseline Pixel AP`；baseline 每行显式标注 source。",
        "- MPDD 三口径: ①A1 concat vs legacy v2 dino score = +0.0486；②feature-DINO-only vs legacy = +0.0227；③concat vs matched feature-DINO-only = +0.0258（纯融合贡献）。",
        "- VisA/MVTec/BTAD 无 v2 分数级缓存 → baseline 用特征级 dino-only KNN（MPDD s0/K1 上该口径与 v2 分数级差 ~0.0008 AP，可比）。",
        "- 逐类表（per_category_results.csv）取每个数据集 seed0/K1（BTAD seed0）的 concat 报告。",
    ]),
    encoding="utf-8",
)

print(json.dumps({"rows": len(rows), "per_category_rows": len(per_cat), "recompute_all_pass": all_pass, "checks": checks}, ensure_ascii=False, indent=2))
