# Few-shot Industrial Anomaly Detection — Dual-Encoder Patch Fusion

> 投稿收尾入口（2026-08-26）：[SCI 四区投稿与复现总交接](docs/PAPER_SUBMISSION_HANDOFF_AND_REPRODUCIBILITY_PLAN_20260826.md)；[投稿复现包审计入口](docs/submission_reproducibility_20260826/README.md)。

本仓库当前论文主线是**双编码器视觉 patch 固定融合 + 正常记忆库**，用于少样本工业异常检测。早期视觉—语言动态路由是已关闭的探索路线，仅作为负结果与研究边界保留。

## 当前状态 (2026-08-26)

- **A1 主结果（双视觉固定融合）**: DINO `dinov2_vitb14` + AnomalyCLIP `ViT-L/14@336` image-tower patch 特征 concat + KNN(k=1) normal memory bank（冻结 w=0.5，非动态路由、非显式文本融合）。相对 matched feature-DINO-only KNN 的纯融合 mean ΔPixel-AP 在 MPDD/BTAD/VisA/MVTec 分别为 +0.025830 / +0.024895 / +0.052353 / +0.031962，均为 9/9 配置非负。
- **V4 视觉—文本动态融合扩展**: 已按决策 D 关闭（官方 SubspaceAD G2 审计 FAILED，`paper_eligible=false`，G4–G11 永久阻断）。
- 唯一权威状态: [docs/CURRENT_DYNAMIC_FUSION_STATUS.md](docs/CURRENT_DYNAMIC_FUSION_STATUS.md)；权威计划: [docs/DYNAMIC_FUSION_NEXT_STEPS.md](docs/DYNAMIC_FUSION_NEXT_STEPS.md)。

详细实验记录、交接文档: [HANDOFF.md](HANDOFF.md)  
项目状态: [PROJECT_STATUS.md](PROJECT_STATUS.md)  
初始计划: [PLAN.md](PLAN.md)

## 核心目标

- 数据集: MPDD (6类), BTAD (3类), VisA (12类), MVTec AD (15类)
- Shot: 1 / 2 / 4 张正常参考图
- 基础方法: AnomalyCLIP, WinCLIP+, PatchCore, PromptAD, AnomalyDINO, ReMP-AD, AdaptCLIP
- 输出: 图像级检测、像素级定位、效率统计与可复现实验记录

## 环境

| 虚拟环境 | 用途 |
|---|---|
| `.venv-anomalyclip` | AnomalyCLIP GPU 推理 |
| `.venv-anomalydino` | AnomalyDINO GPU 推理 |
| `.venv-patchcore` | PatchCore 推理 + V3.3/V3.5 CPU 评估 |
| `.venv-winclip` | WinCLIP+ 推理 |
| `.venv-promptad` | PromptAD 训练/推理 |
| `.venv-adaptclip` | AdaptCLIP 推理 + 部分 CPU 评估 |
| `.venv-remp_ad` | ReMP-AD 推理 |

GPU: NVIDIA RTX 3060 Laptop, 6 GB VRAM

## 快速复现

A1 冻结配置的完整复现步骤见 [experiments/dynamic_fusion/freeze/a1_mpdd_w05/REPRODUCE.md](experiments/dynamic_fusion/freeze/a1_mpdd_w05/REPRODUCE.md) 与 [METHOD_CARD.md](experiments/dynamic_fusion/freeze/a1_mpdd_w05/METHOD_CARD.md)。
