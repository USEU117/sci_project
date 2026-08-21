# Few-shot Industrial Anomaly Detection — Dynamic Fusion Research

本仓库研究**基于不确定性路由的视觉-语言证据融合**方法，用于少样本工业异常检测。

## 当前状态 (2026-08-19)

- **A1 主结果（双视觉固定融合）**: DINO `dinov2_vitb14` + AnomalyCLIP `ViT-L/14@336` patch 特征 concat + KNN(k=1) normal memory bank（冻结 w=0.5，非动态路由）。MPDD/BTAD/VisA/MVTec 四数据集 9/9 全正，mean ΔAP vs DINO 分别为 +0.0486 / +0.0766 / +0.0524 / +0.0320。
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
