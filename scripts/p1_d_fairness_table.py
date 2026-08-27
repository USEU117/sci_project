"""P1-D: fairness/protocol comparison table.

Explicitly lists, per method, the protocol dimensions that matter for a fair
comparison: backbone, input resolution, shot/seed protocol, source-domain
training, target-normal tuning, test-time adaptation, whether text participates
in inference, evaluator, and this project's baseline source.

Every cell is grounded in project evidence (files referenced in the 'source'
column of the MD output). Cells that are not verifiable in this repo are marked
with the method's official protocol instead of being invented.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "submission_repro_20260827" / "evidence" / "p1"

ROWS = [
    {
        "method": "A1 concat (this paper)",
        "visual_backbone": "DINOv2 ViT-B/14 + AnomalyCLIP ViT-L/14@336 image tower",
        "input_resolution": "DINO 448 / CLIP 518",
        "shot_protocol": "1/2/4-shot normal reference memory bank; 3 seeds",
        "training": "none (pretrained backbones frozen; memory bank built at inference)",
        "target_normal_tuning": "no",
        "test_time_adaptation": "no",
        "text_in_inference": "no",
        "evaluator": "unified frozen evaluator (stride=8)",
        "baseline_source": "matched feature-level dino-only KNN (same pipeline)",
    },
    {
        "method": "feature-DINO-only KNN (matched control)",
        "visual_backbone": "DINOv2 ViT-B/14",
        "input_resolution": "448",
        "shot_protocol": "1/2/4-shot; 3 seeds",
        "training": "none (frozen, KNN memory bank)",
        "target_normal_tuning": "no",
        "test_time_adaptation": "no",
        "text_in_inference": "no",
        "evaluator": "unified frozen evaluator (stride=8)",
        "baseline_source": "feature-level KNN in the same pipeline",
    },
    {
        "method": "CLIP-only KNN",
        "visual_backbone": "AnomalyCLIP ViT-L/14@336 image tower",
        "input_resolution": "518",
        "shot_protocol": "1/2/4-shot; 3 seeds (MPDD/VisA only)",
        "training": "none (frozen, KNN memory bank)",
        "target_normal_tuning": "no",
        "test_time_adaptation": "no",
        "text_in_inference": "no",
        "evaluator": "unified frozen evaluator (stride=8)",
        "baseline_source": "feature-level KNN in the same pipeline",
    },
    {
        "method": "PatchCore",
        "visual_backbone": "WideResNet-50 (layer2/3)",
        "input_resolution": "official resize/imagesize",
        "shot_protocol": "1/2/4-shot; coreset sampling",
        "training": "none (ImageNet-pretrained features; coreset memory bank)",
        "target_normal_tuning": "no",
        "test_time_adaptation": "no",
        "text_in_inference": "no",
        "evaluator": "unified evaluator on project caches",
        "baseline_source": "v2 score prediction caches (PLAN.md P0 protocol)",
    },
    {
        "method": "AnomalyDINO",
        "visual_backbone": "DINOv2 ViT-S/14 (project run: dinov2_vits14_448)",
        "input_resolution": "448 (project run)",
        "shot_protocol": "1/2/4-shot normal reference nearest-neighbor; 3 seeds",
        "training": "none (training-free frozen DINOv2 features)",
        "target_normal_tuning": "no (normal references only build the non-parametric bank)",
        "test_time_adaptation": "no",
        "text_in_inference": "no",
        "evaluator": "unified evaluator on project caches",
        "baseline_source": "v2 score prediction caches",
    },
    {
        "method": "PromptAD",
        "visual_backbone": "CLIP ViT-L/14@336",
        "input_resolution": "official (336)",
        "shot_protocol": "1/2/4-shot; prompt learning",
        "training": "yes (only-normal prompt learning)",
        "target_normal_tuning": "true (must be labeled target_normal_tuning=true)",
        "test_time_adaptation": "no",
        "text_in_inference": "yes (prompts, training-time)",
        "evaluator": "unified evaluator on project caches",
        "baseline_source": "v2 score prediction caches",
    },
    {
        "method": "WinCLIP+",
        "visual_backbone": "OpenCLIP ViT-B/16-plus-240 (project env)",
        "input_resolution": "official",
        "shot_protocol": "project unified 1/2/4-shot normal references; 3 seeds (zero-shot WinCLIP reported separately)",
        "training": "none (frozen text/image features)",
        "target_normal_tuning": "no",
        "test_time_adaptation": "no",
        "text_in_inference": "yes (text prompts)",
        "evaluator": "unified evaluator on project caches",
        "baseline_source": "v2 score prediction caches",
    },
    {
        "method": "AnomalyCLIP",
        "visual_backbone": "CLIP ViT-L/14@336 (text + image)",
        "input_resolution": "official",
        "shot_protocol": "zero-shot",
        "training": "none (zero-shot)",
        "target_normal_tuning": "no",
        "test_time_adaptation": "no",
        "text_in_inference": "yes",
        "evaluator": "unified evaluator on project caches",
        "baseline_source": "fixed zero-shot text cache (reused by hash)",
    },
    {
        "method": "ReMP-AD",
        "visual_backbone": "CLIP ViT-L/14@336 (official)",
        "input_resolution": "official (518)",
        "shot_protocol": "train once, test 4/2/1-shot",
        "training": "yes (official training before testing; source-domain)",
        "target_normal_tuning": "no (official protocol)",
        "test_time_adaptation": "no",
        "text_in_inference": "yes (retrieval of text descriptions)",
        "evaluator": "unified evaluator (official image-score 0.5*(text prob + few-shot max))",
        "baseline_source": "outputs/unified remp_ad_mvtec_* (official protocol)",
    },
    {
        "method": "AdaptCLIP",
        "visual_backbone": "CLIP ViT-L/14@336 + 3 adapters",
        "input_resolution": "518 (features 6/12/18/24)",
        "shot_protocol": "1-shot; seeds 0/1/2",
        "training": "yes (source-domain adapter training; official ckpt train_on_visa)",
        "target_normal_tuning": "no (adapters pretrained on source domain)",
        "test_time_adaptation": "no",
        "text_in_inference": "yes (textual adapter)",
        "evaluator": "unified evaluator on project caches",
        "baseline_source": "outputs/unified adaptclip_mvtec_seed_*_shot_1",
    },
    {
        "method": "SubspaceAD",
        "visual_backbone": "DINOv2 (frozen)",
        "input_resolution": "official",
        "shot_protocol": "few-shot normal subspace PCA",
        "training": "none (training-free PCA subspace on normal patches)",
        "target_normal_tuning": "no",
        "test_time_adaptation": "no",
        "text_in_inference": "no",
        "evaluator": "official / project cross-check",
        "baseline_source": "external training-free visual baseline (see introduction research notes)",
    },
]


def main() -> int:
    out = OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    fields = list(ROWS[0].keys())
    with (out / "p1_d_fairness_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(ROWS)

    md = [
        "# P1-D 公平性/协议对照表",
        "",
        "目标：所有对照方法显式区分协议维度，禁止把 zero-shot、target-normal tuning、源域训练与 training-free "
        "方法混在同一条件下直接比较。",
        "",
        "| 方法 | 视觉 backbone | 输入分辨率 | shot 协议 | 训练 | 目标正常图调优 | 测试时适应 | 文本参与推理 | 评测 | baseline source |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ROWS:
        md.append("| " + " | ".join(str(r[k]).replace("|", "\\|") for k in fields) + " |")
    md += [
        "",
        "## 信息来源（项目内证据）",
        "",
        "- `submission_repro_20260827/METHOD_SPEC_V2.md`、`docs/CURRENT_DYNAMIC_FUSION_STATUS.md`：A1/feature-DINO/CLIP-only 口径",
        "- `PLAN.md`、`scripts/run_anomalydino_unified.ps1`、`scripts/run_winclip_unified.ps1` 与对应 evaluation reports：AnomalyDINO/WinCLIP+ 的实际 backbone、1/2/4-shot 和 seed 口径",
        "- AnomalyDINO 官方 WACV 2025 论文/仓库：方法为 frozen DINOv2 patch nearest-neighbor、training-free，不做 fine-tuning",
        "- `PLAN.md`、PromptAD 运行记录：PromptAD 的 shot/seed 协议与 `target_normal_tuning=true` 标注",
        "- `docs/remp_ad_adaptclip_audit.md`、`NEXT_ACTIONS.md`：ReMP-AD/AdaptCLIP 官方协议、ViT-L/14@336/518px、源域训练、统一导出",
        "- `docs/environment_matrix.md`：WinCLIP+ 使用 OpenCLIP ViT-B/16-plus-240",
        "- `docs/introduction_research_20260825/INTRODUCTION_LITERATURE_MASTER_20260826.md`：SubspaceAD 训练-free 定义",
        "- 外部方法的具体像素数值请见 `evidence/paper_tables/` 与 `outputs/unified/` 统一报告，本表只做协议对照。",
    ]
    (out / "p1_d_fairness_table.md").write_text("\n".join(md), encoding="utf-8")

    payload = {
        "schema_version": 1,
        "kind": "p1_d_fairness_table",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": ROWS,
        "note": "Protocol comparison only; numbers live in paper_tables and outputs/unified.",
    }
    (out / "p1_d_fairness_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "methods": len(ROWS), "output": str(out / "p1_d_fairness_table.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
