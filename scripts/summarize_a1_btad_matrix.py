"""Summarize the A1 BTAD post-freeze validation matrix (9 configs x concat/dino).

BTAD report layout differs from VisA/MVTec:
  - K1 (shot=1):  concat -> a1_vitb14_btad_fusion/seed{s}/
                  dino  -> a1_vitb14_btad_dino/seed{s}/
  - K2/K4:        a1_vitb14_btad_20260819/seed{s}_k{shot}/
  - only concat + dino modes exist (no standalone CLIP-only on BTAD).

Baseline (anomalydino_visual) is the legacy v2 dino *score* cache
(v2_btad_predictions, matched per (seed, shot)); it is NOT a feature-level
dino-only KNN (that is the `dino` mode's `fused` result).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

EXP = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a"
FUSION = EXP / "a1_vitb14_btad_fusion"
DINO_K1 = EXP / "a1_vitb14_btad_dino"
MATRIX = EXP / "a1_vitb14_btad_20260819"

SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]
CATEGORIES = ["01", "02", "03"]
N_CATS = 3
TEST_TOTAL = 741


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals))


def report_path(seed: int, shot: int, mode: str) -> Path:
    fname = f"{mode}_pca0_whiten0_w0.5_report.json"
    if shot == 1:
        base = FUSION if mode == "concat" else DINO_K1
        return base / f"seed{seed}" / fname
    return MATRIX / f"seed{seed}_k{shot}" / fname


def main() -> int:
    rows = []
    for seed in SEEDS:
        for shot in SHOTS:
            concat_p = report_path(seed, shot, "concat")
            dino_p = report_path(seed, shot, "dino")
            if not concat_p.is_file():
                raise SystemExit(f"missing concat report: {concat_p}")
            if not dino_p.is_file():
                raise SystemExit(f"missing dino report: {dino_p}")
            cr = json.loads(concat_p.read_text(encoding="utf-8"))
            dr = json.loads(dino_p.read_text(encoding="utf-8"))

            concat_ap = cr["mean_fused"]["pixel_ap"]
            concat_delta = cr["mean_delta_ap_vs_dino"]
            legacy_dino_ap = cr["mean_dino_baseline_ap"]
            dino_ap = dr["mean_fused"]["pixel_ap"]
            dino_delta = dr["mean_delta_ap_vs_dino"]
            concat_minus_dino = concat_ap - dino_ap

            rows.append(
                {
                    "seed": seed,
                    "shot": shot,
                    "concat_ap": concat_ap,
                    "concat_auroc": cr["mean_fused"]["pixel_auroc"],
                    "concat_aupro": cr["mean_fused"]["pixel_aupro"],
                    "concat_delta_vs_legacy_dino": concat_delta,
                    "dino_only_ap": dino_ap,
                    "dino_only_delta_vs_legacy_dino": dino_delta,
                    "legacy_dino_baseline_ap": legacy_dino_ap,
                    "concat_minus_dino_only": concat_minus_dino,
                    "positive_categories": int(
                        sum(1 for c in cr["per_category"] if c["delta_ap"] > 0)
                    ),
                    "max_regression": round(
                        float(min(c["delta_ap"] for c in cr["per_category"])), 6
                    ),
                }
            )

    concat_aps = [r["concat_ap"] for r in rows]
    concat_deltas = [r["concat_delta_vs_legacy_dino"] for r in rows]
    dino_aps = [r["dino_only_ap"] for r in rows]
    dino_deltas = [r["dino_only_delta_vs_legacy_dino"] for r in rows]
    c_minus_d = [r["concat_minus_dino_only"] for r in rows]

    def agg_delta(vals: list[float]) -> dict:
        by_seed = {str(s): mean([v for v, r in zip(vals, rows) if r["seed"] == s]) for s in SEEDS}
        by_shot = {str(k): mean([v for v, r in zip(vals, rows) if r["shot"] == k]) for k in SHOTS}
        return {
            "mean_delta_ap_vs_dino": mean(vals),
            "positive_configs": int(sum(1 for v in vals if v > 0)),
            "by_seed_delta": by_seed,
            "by_shot_delta": by_shot,
        }

    overall = {
        "concat": {
            "mean_fused_pixel_ap": mean(concat_aps),
            "mean_fused_pixel_auroc": mean([r["concat_auroc"] for r in rows]),
            "mean_fused_pixel_aupro": mean([r["concat_aupro"] for r in rows]),
            **agg_delta(concat_deltas),
        },
        "dino_only": {
            "mean_fused_pixel_ap": mean(dino_aps),
            **agg_delta(dino_deltas),
        },
        "concat_minus_dino_only": {
            "mean_delta_ap": mean(c_minus_d),
            "positive_configs": int(sum(1 for v in c_minus_d if v > 0)),
        },
    }

    # per-category concat aggregates (vs legacy dino + vs feature-level dino-only)
    cat_agg = {}
    for cat in CATEGORIES:
        deltas = []
        deltas_minus_dino = []
        for seed in SEEDS:
            for shot in SHOTS:
                cr = json.loads(report_path(seed, shot, "concat").read_text(encoding="utf-8"))
                dr = json.loads(report_path(seed, shot, "dino").read_text(encoding="utf-8"))
                cat_concat = next(c for c in cr["per_category"] if c["category"] == cat)
                cat_dino = next(c for c in dr["per_category"] if c["category"] == cat)
                deltas.append(cat_concat["delta_ap"])
                deltas_minus_dino.append(cat_concat["fused"]["pixel_ap"] - cat_dino["fused"]["pixel_ap"])
        cat_agg[cat] = {
            "mean_delta_ap_vs_legacy_dino": round(mean(deltas), 6),
            "positive_configs_vs_legacy": int(sum(1 for d in deltas if d > 0)),
            "mean_delta_ap_vs_dino_only": round(mean(deltas_minus_dino), 6),
            "positive_configs_vs_dino_only": int(sum(1 for d in deltas_minus_dino if d > 0)),
        }

    report = {
        "schema_version": 1,
        "run_id": "a1_btad_matrix_20260819",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "phase_7_post_freeze_external_validation",
        "dataset": "btad",
        "dataset_role": "external_frozen_validation",
        "frozen_config": "concat pca_dim=0 whiten=0 dino_weight=0.5 (dinov2_vitb14 DINO + AnomalyCLIP), KNN k=1, stride=8, map=448",
        "baseline_source": "legacy v2 dino score cache (v2_btad_predictions, matched per (seed, shot))",
        "n_configs": len(rows),
        "overall": overall,
        "per_category_concat": cat_agg,
        "rows": rows,
    }
    out = MATRIX / "btad_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# A1 BTAD 冻结后验证（post-freeze, 阶段七）",
        "",
        f"- RunId: `a1_btad_matrix_20260819`（{report['created_at_utc']} UTC）",
        f"- 数据集: BTAD（外部冻结后验证，{N_CATS} 类，{TEST_TOTAL} 测试图）。",
        "- 冻结配置: concat + KNN(k=1) + distance/2，pca_dim=0，whiten=0，w=0.5，stride=8，map=448（与 freeze_manifest 一致，**未调参**）。",
        "- baseline 口径: `anomalydino_visual` = legacy v2 dino **score** cache（`v2_btad_predictions`，按 (seed, shot) 匹配）；`dino` 模式的 `fused` 才是特征级 dino-only KNN。",
        "- 特征导出: K1 全量（`raw_patch_features`）+ K2/K4 ref-only（测试特征复用 K1，黄金结论：测试特征与 (seed,shot) 无关）。",
        "- 评估: CPU/faiss，18 个评估（9 配置 × concat/dino），无 standalone CLIP-only（与 MPDD 口径一致）。",
        "",
        "## 汇总（mean Pixel AP）",
        "",
        "| 模式 | mean fused AP | mean ΔAP vs legacy DINO | 正向配置 |",
        "|---|---|---|---|",
        f"| feature-DINO-only KNN | {overall['dino_only']['mean_fused_pixel_ap']:.4f} | {overall['dino_only']['mean_delta_ap_vs_dino']:+.4f} | {overall['dino_only']['positive_configs']}/9 |",
        f"| **concat + KNN（冻结）** | {overall['concat']['mean_fused_pixel_ap']:.4f} | **{overall['concat']['mean_delta_ap_vs_dino']:+.4f}** | **{overall['concat']['positive_configs']}/9** |",
        "",
        "## 三口径分解（对照 MPDD）",
        "",
        "| 口径 | mean ΔAP | 正向配置 |",
        "|---|---|---|",
        f"| ① concat vs legacy DINO score | {overall['concat']['mean_delta_ap_vs_dino']:+.4f} | {overall['concat']['positive_configs']}/9 |",
        f"| ② feature-DINO-only vs legacy DINO score | {overall['dino_only']['mean_delta_ap_vs_dino']:+.4f} | {overall['dino_only']['positive_configs']}/9 |",
        f"| ③ concat vs matched feature-DINO-only（纯融合贡献） | {overall['concat_minus_dino_only']['mean_delta_ap']:+.4f} | {overall['concat_minus_dino_only']['positive_configs']}/9 |",
        "",
        "## 9 配置（concat）",
        "",
        "| seed | shot | concat AP | legacy DINO AP | ΔAP | 正向类别 | 最大退化 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['shot']} | {r['concat_ap']:.4f} | {r['legacy_dino_baseline_ap']:.4f} "
            f"| {r['concat_delta_vs_legacy_dino']:+.4f} | {r['positive_categories']}/{N_CATS} | {r['max_regression']:+.4f} |"
        )
    lines += [
        "",
        "## 逐类（concat，跨 9 配置平均 ΔAP）",
        "",
        "| category | mean ΔAP vs legacy DINO | 正向 | mean ΔAP vs DINO-only | 正向 |",
        "|---|---|---|---|---|",
    ]
    for cat, v in cat_agg.items():
        lines.append(
            f"| {cat} | {v['mean_delta_ap_vs_legacy_dino']:+.4f} | {v['positive_configs_vs_legacy']}/9 "
            f"| {v['mean_delta_ap_vs_dino_only']:+.4f} | {v['positive_configs_vs_dino_only']}/9 |"
        )
    lines += [
        "",
        "## 结论",
        "",
        f"1. **BTAD 冻结后验证（9 配置）**：concat mean ΔAP vs legacy DINO {overall['concat']['mean_delta_ap_vs_dino']:+.4f}（{overall['concat']['positive_configs']}/9 正）；纯融合贡献（vs matched feature-DINO-only）{overall['concat_minus_dino_only']['mean_delta_ap']:+.4f}。",
        "2. 与 MPDD（+0.0486 / +0.0258）、VisA（+0.0524）、MVTec（+0.0320）的结论一致：concat 增益来自 CLIP 互补。",
        "3. BTAD baseline 为 legacy v2 dino score cache（与 MPDD 同源），区别于 VisA/MVTec 的 feature-level dino-only KNN。",
        "",
        "## 产物",
        "",
        "- 评估报告: `a1_vitb14_btad_20260819/seed{{s}}_k{{2,4}}/{{concat,dino}}_pca0_whiten0_w0.5_report.json` + K1 旧目录 `a1_vitb14_btad_{fusion,dino}/seed{{s}}/`",
        "- 本汇总: `a1_vitb14_btad_20260819/btad_summary.json` + `btad.md`",
        "- 特征缓存: `outputs/dynamic_fusion/v3_direction_a/features_vitb14_btad_s{{seed}}_k{{shot}}/anomalydino_visual/` 与 `features_btad_s{{seed}}_k{{shot}}/anomalyclip_text/`",
    ]
    (MATRIX / "btad.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"overall": overall, "per_category": cat_agg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
