# 项目交接文档 — 少样本工业异常检测动态融合研究

> 日期: 2026-08-12  
> 目标平台: SLE.Work克  
> 当前状态: V3.3 加权集成为主力方案，V3.5 (B/C方向) 实验已完成

---

## 1. 项目概述

本项目研究**基于不确定性路由的视觉-语言证据融合**方法，用于少样本工业异常检测。

- **数据集**: MPDD (6类) / BTAD (3类) / VisA (12类) / MVTec AD (15类)
- **Shot**: 1 / 2 / 4 张正常参考图
- **基础方法**: AnomalyCLIP, WinCLIP+, PatchCore, PromptAD, AnomalyDINO, ReMP-AD, AdaptCLIP
- **核心贡献**: DINO visual + AnomalyCLIP text 的 z-score 校准加权融合 (V3.3)

## 2. 环境与依赖

### Python 环境

| 虚拟环境 | Python | 主要用途 | PyTorch/CUDA |
|---|---|---|---|
| `.venv-anomalyclip` | 3.10 | AnomalyCLIP 推理/导出 | 2.0.0+cu118 |
| `.venv-patchcore` | 系统 | PatchCore 推理 | — |
| `.venv-winclip` | 系统 | WinCLIP+ 推理 | — |
| `.venv-anomalydino` | 系统 | AnomalyDINO 推理 | — |
| `.venv-promptad` | 系统 | PromptAD 训练/推理 | — |
| `.venv-adaptclip` | 3.10 | AdaptCLIP 推理 + V3.5评估 | numpy 1.24.4 |
| `.venv-remp_ad` | 系统 | ReMP-AD 推理 | — |

### 硬件

- GPU: NVIDIA RTX 3060 Laptop, 6 GB VRAM
- 系统: Windows 11
- Shell: PowerShell 5
- Git: 2.54.0

### 注意事项

- 使用 `$env:PYTHONDONTWRITEBYTECODE=1` 避免 sandbox pycache 写入错误
- RTX 3060 6GB 显存限制：批量处理需要 batch_size=1，文本编码需要分批 (batch_size=40)
- 某些脚本会打印 `pkg_resources is deprecated` 警告，不影响运行

## 3. 目录结构

```
sci_project/
├── configs/                    # 配置文件 (YAML 协议、GPU队列配置)
├── data/
│   ├── splits/                 # 数据集划分 (mpdd/, btad/)
│   ├── mpdd_raw/MPDD/          # MPDD 原始数据 (6类, 458测试图)
│   └── btad_raw/               # BTAD 原始数据 (空占位)
├── docs/                       # 设计文档、协议、分析报告 (.md/.docx)
│   └── few_shot_industrial_ad_project_overview_expanded_20260812_v2.docx  # 项目方向文档
├── experiments/dynamic_fusion/ # 所有实验记录
│   ├── v3_3/                   # ★ 主力方案: 加权集成 (Gate B 通过)
│   ├── v3_5_defect_ensemble/   # 方向B: 缺陷词增强 (失败)
│   ├── v3_5_hierarchical/      # 方向C: 图像级分层融合 (oracle仅+0.01)
│   ├── v2/                     # V2 动态路由及相关
│   ├── v3/                     # V3 可靠性路由/AdaptCLIP相关
│   ├── v3_4/                   # V3.4 Gate A相关
│   └── summaries/              # 汇总报告
├── methods/                    # 7种基线方法源码
│   ├── AnomalyCLIP-main/
│   ├── anomalydino/
│   ├── patchcore/
│   ├── winclip/
│   ├── promptad/
│   ├── adaptclip/
│   └── remp_ad/
├── outputs/dynamic_fusion/     # 预测缓存、NPZ导出、评估结果
│   ├── v2_mpdd_predictions/    # MPDD seed 0/1/2 预测缓存
│   └── v3_5_defect_ensemble/   # 方向B 缺陷词导出
├── scripts/                    # 所有运行/评估/分析脚本 (~95个)
├── src/industrial_ad/fusion/   # 核心融合框架代码 (21个模块)
├── HANDOFF.md                  # ← 本交接文档
├── PROJECT_STATUS.md           # 项目状态详细记录
├── PLAN.md                     # 初始计划文档
└── README.md                   # 项目简介
```

## 4. 关键实验结论

### 4.1 V3.3 加权集成 (✅ 成功 — 论文主结果候选)

**方法**: Z-score 校准 DINO visual 和 AnomalyCLIP text 两个分支 → 加权平均  
**权重**: DINO=0.60, AnomalyCLIP=0.40  
**数据集**: MPDD, k=1, 3 seeds (0/1/2)

| Seed | LoO Mean ΔAP | 正类别 | Gate B |
|------|-------------|--------|--------|
| 0 | +0.0463 | 5/6 | PASSED |
| 1 | +0.0752 | 5/6 | PASSED |
| 2 | +0.0898 | 6/6 | PASSED |

- **3-seed 平均 LoO ΔAP = +0.0704**
- 5/6 类别满足 2/3 重复性要求
- metal_plate 未通过 (DINO 基线 AP=0.847，天花板效应)

**关键文件**:
- `experiments/dynamic_fusion/v3_3/overview.md` — 完整分析
- `experiments/dynamic_fusion/v3_3/cross_seed_report.json`
- `src/industrial_ad/fusion/v3_3_strategies.py` — 核心实现
- `scripts/evaluate_v3_3_pipeline.py` — 评估流水线

### 4.2 V3.5 方向C: 图像级分层融合 (❌ 失败)

**方法**: 3种图像级gate策略（离散gate / 连续sigmoid / cross-modal agreement）  
**结论**: 图像级 oracle 上限仅 +0.01 ΔAP，无法超越 V3.3 静态融合  
**原因**: 图像级 weight 无法利用文本分支的像素级信息

关键文件: `experiments/dynamic_fusion/v3_5_hierarchical/s0_report.json`

### 4.3 V3.5 方向B: 缺陷词Prompt Ensemble (❌ 失败)

**方法**: 6个手写缺陷词变体 ("damaged bracket", "broken bracket"等) 的 prompt ensemble  
**结果**: Mean ΔAP = -0.0364 (比 learned prompts 的 +0.0754 差 0.112)  
**结论**: 手写缺陷词无法替代 learned prompts

关键文件: `experiments/dynamic_fusion/v3_5_defect_ensemble/s0_eval.json`

### 4.4 V2 动态路由 / BTAD Holdout

- V2 MPDD 开发后冻结为 `visual_only_safe_fallback` (text权重=0)
- BTAD 独立验证: Image AUROC 0.942, Pixel AUROC 0.965, AUPRO 0.722 (27行均值)

### 4.5 VisA & MVTec 基线矩阵

- VisA 4方法 × 3 shot × 3 seed = 36 runs，全部通过审计
- MVTec PatchCore/WinCLIP+/AnomalyDINO 各 9/9 完成
- DynamicFusion MVTec: 1/2/4-shot 各 3 seeds 完成
- PromptAD MVTec: s0/k1-k4 + s1/k1 完成，其余暂停

## 5. 复现方法

### V3.3 加权集成 (MPDD)

```powershell
# 前置条件: V2 MPDD 预测缓存已生成
# 单 seed 运行
.\.venv-patchcore\Scripts\python.exe scripts/evaluate_v3_3_pipeline.py --seed 0
.\.venv-patchcore\Scripts\python.exe scripts/evaluate_v3_3_pipeline.py --seed 1
.\.venv-patchcore\Scripts\python.exe scripts/evaluate_v3_3_pipeline.py --seed 2

# Gate B 评估
.\.venv-patchcore\Scripts\python.exe scripts/evaluate_v3_3_gate_b.py
```

### V3.5 方向B: 缺陷词导出

```powershell
# GPU 导出 (需要 .venv-anomalyclip)
$env:PYTHONDONTWRITEBYTECODE=1
.\.venv-anomalyclip\Scripts\python.exe scripts/export_anomalyclip_defect_ensemble.py `
  --manifest data/splits/mpdd/manifest.json `
  --data-root data/mpdd_raw/MPDD `
  --output-dir outputs/dynamic_fusion/v3_5_defect_ensemble/s0_shot1 `
  --seed 0 --shot 1 --fast

# CPU 评估 (需要 .venv-adaptclip)
.\.venv-adaptclip\Scripts\python.exe scripts/evaluate_v3_5_defect_ensemble.py `
  --defect-ensemble-dir outputs/dynamic_fusion/v3_5_defect_ensemble/s0_shot1 `
  --seed 0
```

### V3.5 方向C: 分层融合

```powershell
# CPU only
.\.venv-patchcore\Scripts\python.exe scripts/evaluate_v3_5_hierarchical.py
```

## 6. 核心源码映射

| 模块 | 文件 | 说明 |
|------|------|------|
| V3.3 策略 | `src/industrial_ad/fusion/v3_3_strategies.py` | 加权平均 / max_z / two_stage 三种策略 |
| V3.5 策略 | `src/industrial_ad/fusion/v3_5_strategies.py` | 图像级 gate + oracle |
| V3.3 评估 | `scripts/evaluate_v3_3_pipeline.py` | 自动化多 seed 流水线 |
| V3.5 评估 | `scripts/evaluate_v3_5_hierarchical.py` | 分层融合评估 |
| 缺陷词工具 | `scripts/defect_ensemble_utils.py` | 文本特征构建 (batched encode_text) |
| 缺陷词导出 | `scripts/export_anomalyclip_defect_ensemble.py` | GPU 导出 |
| V3.5 缺陷评估 | `scripts/evaluate_v3_5_defect_ensemble.py` | 缺陷词 vs 原始对比 |
| 预测通用 | `scripts/v2_mpdd_prediction_common.py` | 数据集索引/校验 |

## 7. 仍存在的缺口

1. **BTAD holdout 对 V3.3 的验证**: V3.3 在 MPDD 上通过 Gate B，但未在 BTAD holdout 上验证
2. **图像级指标**: V3.3 当前仅评估像素级 (AP/AUPRO)，需补图像级 AUROC/AP
3. **metal_plate 天花板**: DINO 基线 AP=0.847，融合在该类上的增益不稳定
4. **PromptAD MVTec**: s1/k2-k4, s2/k1-k4 未完成
5. **ReMP-AD / AdaptCLIP Gate A**: 需要 manifest/NPZ 适配
6. **论文**: 英文草稿在 `outputs/paper_draft_20260810/`

## 8. 注意事项

- V3.5 方向B 使用的是标准 CLIP 模型 (design_details=None)，不是 AnomalyCLIP 的 DAPM_replace 版本
- V3.3 的预测缓存来自 V2: `outputs/dynamic_fusion/v2_mpdd_predictions/`
- BTAD 是独立保持集，在 MPDD 参数冻结之前不能用于调参
- `experiments/dynamic_fusion/v3_5_defect_ensemble/s0_shot1/` 下的 s0_shot1 目录包含 6 个类别 NPZ 和 export_report.json
- `data/mpdd_raw/MPDD` 下只有 bracket_black 有 test/hole 异常样本，其他类别的异常样本需要确认

## 9. 迁移到 SLE.Work克 的步骤

1. **复制整个项目目录** (包含 .venv-* 虚拟环境)
2. **安装 CUDA 依赖**: 需要 CUDA 11.8+ 和对应 PyTorch
3. **验证环境**: 
   ```powershell
   .\.venv-anomalyclip\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
   ```
4. **验证数据完整性**: 
   ```powershell
   .\.venv-patchcore\Scripts\python.exe scripts/validate_dataset.py --dataset mpdd
   ```
5. **运行 V3.3 复现**: 参见上面的复现命令
6. **注意**: 如果 GPU 显存不足 6GB，某些导出脚本的 batch_size 需要进一步降低
