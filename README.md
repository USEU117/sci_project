# Few-shot Industrial Anomaly Detection — Dynamic Fusion Research

本仓库研究**基于不确定性路由的视觉-语言证据融合**方法，用于少样本工业异常检测。

## 当前状态 (2026-08-12)

- **V3.3 加权集成**: DINO visual + AnomalyCLIP text 的 z-score 校准融合，MPDD 上 3/3 seeds Gate B 通过，mean LoO ΔAP = +0.0704【论文主结果候选】
- **V3.5 方向C (图像级分层融合)**: 完成验证 — oracle 上限仅 +0.01 ΔAP，无法超越 V3.3
- **V3.5 方向B (缺陷词增强)**: 完成验证 — 手写缺陷词远不如 learned prompts (-0.112 ΔAP gap)

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

```powershell
# V3.3 加权集成 (MPDD, seed 0)
.\.venv-patchcore\Scripts\python.exe scripts/evaluate_v3_3_pipeline.py --seed 0
```

详见 [HANDOFF.md](HANDOFF.md) 第5节。
