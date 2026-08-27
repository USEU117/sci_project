# P1-D 公平性/协议对照表

目标：所有对照方法显式区分协议维度，禁止把 zero-shot、target-normal tuning、源域训练与 training-free 方法混在同一条件下直接比较。

| 方法 | 视觉 backbone | 输入分辨率 | shot 协议 | 训练 | 目标正常图调优 | 测试时适应 | 文本参与推理 | 评测 | baseline source |
|---|---|---|---|---|---|---|---|---|---|
| A1 concat (this paper) | DINOv2 ViT-B/14 + AnomalyCLIP ViT-L/14@336 image tower | DINO 448 / CLIP 518 | 1/2/4-shot normal reference memory bank; 3 seeds | none (pretrained backbones frozen; memory bank built at inference) | no | no | no | unified frozen evaluator (stride=8) | matched feature-level dino-only KNN (same pipeline) |
| feature-DINO-only KNN (matched control) | DINOv2 ViT-B/14 | 448 | 1/2/4-shot; 3 seeds | none (frozen, KNN memory bank) | no | no | no | unified frozen evaluator (stride=8) | feature-level KNN in the same pipeline |
| CLIP-only KNN | AnomalyCLIP ViT-L/14@336 image tower | 518 | 1/2/4-shot; 3 seeds (MPDD/VisA only) | none (frozen, KNN memory bank) | no | no | no | unified frozen evaluator (stride=8) | feature-level KNN in the same pipeline |
| PatchCore | WideResNet-50 (layer2/3) | official resize/imagesize | 1/2/4-shot; coreset sampling | none (ImageNet-pretrained features; coreset memory bank) | no | no | no | unified evaluator on project caches | v2 score prediction caches (PLAN.md P0 protocol) |
| AnomalyDINO | DINOv2 ViT-S/14 (project run: dinov2_vits14_448) | 448 (project run) | 1/2/4-shot normal reference nearest-neighbor; 3 seeds | none (training-free frozen DINOv2 features) | no (normal references only build the non-parametric bank) | no | no | unified evaluator on project caches | v2 score prediction caches |
| PromptAD | CLIP ViT-L/14@336 | official (336) | 1/2/4-shot; prompt learning | yes (only-normal prompt learning) | true (must be labeled target_normal_tuning=true) | no | yes (prompts, training-time) | unified evaluator on project caches | v2 score prediction caches |
| WinCLIP+ | OpenCLIP ViT-B/16-plus-240 (project env) | official | project unified 1/2/4-shot normal references; 3 seeds (zero-shot WinCLIP reported separately) | none (frozen text/image features) | no | no | yes (text prompts) | unified evaluator on project caches | v2 score prediction caches |
| AnomalyCLIP | CLIP ViT-L/14@336 (text + image) | official | zero-shot | none (zero-shot) | no | no | yes | unified evaluator on project caches | fixed zero-shot text cache (reused by hash) |
| ReMP-AD | CLIP ViT-L/14@336 (official) | official (518) | train once, test 4/2/1-shot | yes (official training before testing; source-domain) | no (official protocol) | no | yes (retrieval of text descriptions) | unified evaluator (official image-score 0.5*(text prob + few-shot max)) | outputs/unified remp_ad_mvtec_* (official protocol) |
| AdaptCLIP | CLIP ViT-L/14@336 + 3 adapters | 518 (features 6/12/18/24) | 1-shot; seeds 0/1/2 | yes (source-domain adapter training; official ckpt train_on_visa) | no (adapters pretrained on source domain) | no | yes (textual adapter) | unified evaluator on project caches | outputs/unified adaptclip_mvtec_seed_*_shot_1 |
| SubspaceAD | DINOv2 (frozen) | official | few-shot normal subspace PCA | none (training-free PCA subspace on normal patches) | no | no | no | official / project cross-check | external training-free visual baseline (see introduction research notes) |

## 信息来源（项目内证据）

- `submission_repro_20260827/METHOD_SPEC_V2.md`、`docs/CURRENT_DYNAMIC_FUSION_STATUS.md`：A1/feature-DINO/CLIP-only 口径
- `PLAN.md`、`scripts/run_anomalydino_unified.ps1`、`scripts/run_winclip_unified.ps1` 与对应 evaluation reports：AnomalyDINO/WinCLIP+ 的实际 backbone、1/2/4-shot 和 seed 口径
- AnomalyDINO 官方 WACV 2025 论文/仓库：方法为 frozen DINOv2 patch nearest-neighbor、training-free，不做 fine-tuning
- `PLAN.md`、PromptAD 运行记录：PromptAD 的 shot/seed 协议与 `target_normal_tuning=true` 标注
- `docs/remp_ad_adaptclip_audit.md`、`NEXT_ACTIONS.md`：ReMP-AD/AdaptCLIP 官方协议、ViT-L/14@336/518px、源域训练、统一导出
- `docs/environment_matrix.md`：WinCLIP+ 使用 OpenCLIP ViT-B/16-plus-240
- `docs/introduction_research_20260825/INTRODUCTION_LITERATURE_MASTER_20260826.md`：SubspaceAD 训练-free 定义
- 外部方法的具体像素数值请见 `evidence/paper_tables/` 与 `outputs/unified/` 统一报告，本表只做协议对照。