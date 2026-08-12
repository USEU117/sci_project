# V3.3 动态融合方案 — 概要

## 背景

V3.2 的层级选择性救援（Hierarchical Selective Rescue）架构完全失效：
- AdaptCLIP textual_adapter 分数比 DINO visual 低 10-20 倍
- text > visual 的像素绝大多数是 false positive（harm rate 57-67%）
- "reject first, rescue rarely" 策略意味着 0 次有效 rescue，Gate B FAILED

**V3.3 放弃层级 rescue，改用简单加权集成**：z-score 校准每个分支 → 加权平均。

## 三种候选策略

| 策略 | 描述 | 复杂度 | 结果 |
|------|------|--------|------|
| **weighted_ensemble** | Z-score 校准 → 加权平均 (DINO + AnomalyCLIP) | 低 | **优胜** |
| two_stage_calibrated | AdaptCLIP 内部融合 → 校准 → 与 DINO 合并 | 中 | 有效但弱（需 AdaptCLIP 缓存） |
| max_z_selection | 每像素选 z-score 最高的分支 | 低 | **失败** |

## 评估协议

- **数据集**：MPDD（6 类工业缺陷），k=1
- **Seeds**：0, 1, 2（3 个独立 seed，满足文档 2/3 重复性要求）
- **Cross-validation**：Leave-one-out（每次留出 1 类，在其余 5 类上选最优候选参数）
- **Gate B 通过条件**：LoO mean ΔAP > 0 **且** positive_categories >= 3
- **文档参照**：`outputs/project_progress_report_20260811/` — 路线B标准

## 基线

| 分支 | Seed | 像素 AUROC | 像素 AP | 像素 AUPRO |
|------|------|-----------|--------|------------|
| DINO visual | s0 | 0.9578 | 0.2802 | 0.8622 |
| DINO visual | s1 | 0.9584 | 0.3030 | 0.8728 |
| DINO visual | s2 | 0.9541 | 0.2725 | 0.8569 |
| AnomalyCLIP text | all | 0.8776 | 0.2444 | 0.7234 |

## Gate B 跨 Seed 验证结果（核心）

### 逐 Seed Gate B 汇总

| Seed | LoO mean ΔAP | 正类别 | Gate B | WE mean ΔAP |
|------|-------------|--------|--------|-------------|
| seed=0 | **+0.0463** | 5/6 | PASSED | +0.0898 |
| seed=1 | **+0.0752** | 5/6 | PASSED | +0.0851 |
| seed=2 | **+0.0898** | 6/6 | PASSED | +0.1011 |

**3/3 seeds Gate B 全部通过！**

### 逐类别 LoO ΔAP（跨 seed 对比）

| 类别 | seed=0 | seed=1 | seed=2 | 2/3重复？ |
|------|--------|--------|--------|----------|
| bracket_black | +0.007 | +0.114 | +0.146 | ✅ 3/3 |
| bracket_brown | +0.004 | +0.004 | +0.003 | ✅ 3/3 |
| bracket_white | +0.071 | +0.119 | +0.139 | ✅ 3/3 |
| connector | +0.092 | +0.083 | +0.094 | ✅ 3/3 |
| metal_plate | **-0.034** | **-0.036** | +0.011 | ❌ 1/3 |
| tubes | +0.137 | +0.166 | +0.146 | ✅ 3/3 |

### 聚合统计

- **3-seed mean LoO ΔAP = +0.0704**
- 标准差 = 0.0222
- 5/6 类别满足 2/3 重复性 → **合格**
- metal_plate 是唯一未通过类别（DINO 基线 AP=0.847，天花板效应）

## 与文档标准对照

| 文档要求（路线B） | V3.3 状态 |
|------------------|----------|
| 至少 2/3 seed 重复 | ✅ 3/3 seeds Gate B 通过，5/6 类别通过 |
| 图像指标无退化 | ⚠️ 仅评估像素级，图像级待补 |
| 独立保持集验证 | ⚠️ BTAD holdout 待执行 |
| Gate B 通过 | ✅ PASSED |

**结论：V3.3 weighted_ensemble 满足文档路线B的量化标准，可以作为论文主结果候选。**

## 策略级对比

| 策略 | Mean ΔAP | 正类别数 |
|------|-------------------|---------|
| weighted_ensemble | **+0.0920** (3-seed均值) | 6/6 × 3 |
| max_z_selection | -0.0276 | ~2/6 × 3 |
| two_stage_calibrated | N/A（缺 AdaptCLIP s1/s2 缓存） | — |

## 推荐冻结方案

**weighted_ensemble: DINO weight=0.60, AnomalyCLIP weight=0.40**

这是跨 seed 最稳健的配置：s0 LoO 选中 4/6 类，s1 LoO 选中 5/6（以 dino=0.70），s2 LoO 也倾向 DINO 权重 0.60-0.70 区间。

## 横向对比：全长历史

| 版本 | Gate B | LoO ΔAP | 正类别 | 备注 |
|------|--------|---------|--------|------|
| V1 | N/A | N/A | N/A | 校准饱和，动态融合低于视觉基线 |
| V2 | FAILED | ~0.000 | 0/6 | 安全回退为纯视觉，text权重=0 |
| V3.2 | FAILED | 0.000 | 0/6 | 层级rescue，0次有效rescue |
| **V3.3** | **PASSED** | **+0.070** | **5-6/6** | 加权集成，3/3 seeds通过 |

## 仍存在的缺口

1. **BTAD holdout**：需要在 3 个 BTAD 类别上冻结验证
2. **图像级指标**：当前仅评估像素级，需补图像级 AUROC/AP
3. **metal_plate 天花板**：DINO 基线 0.847，融合收益不稳定

## 关键文件

### 实验输出
- `experiments/dynamic_fusion/v3_3/cross_seed_report.json` — 跨 seed 对比报告（本文档的 JSON 版）
- `experiments/dynamic_fusion/v3_3/s0_k1/report.json` — seed=0 完整结果
- `experiments/dynamic_fusion/v3_3/s1_k1/report.json` — seed=1 完整结果
- `experiments/dynamic_fusion/v3_3/s2_k1/report.json` — seed=2 完整结果
- `experiments/dynamic_fusion/v3_3/gate_b/report.json` — Gate B 正式报告（含 V3.2 横向对比）
- `experiments/dynamic_fusion/v3_3/pipeline_comparison/report.json` — s0 完整流水线（26 变体）
- `experiments/dynamic_fusion/v3_3/fusion_decisions.json` — 每类推荐配置

### 实验日志
- `experiments/dynamic_fusion/v3_3/logs/pipeline_run.log` — s0 执行日志
- `experiments/dynamic_fusion/v3_3/s1_k1/pipeline.log` — s1 执行日志
- `experiments/dynamic_fusion/v3_3/s2_k1/pipeline.log` — s2 执行日志

### 数据集划分元数据
- `experiments/dynamic_fusion/v3_3/staged_mpdd_s0_k1/meta.json` — s0 数据划分
- `experiments/dynamic_fusion/v3_3/s1_k1/meta.json` — s1 数据划分
- `experiments/dynamic_fusion/v3_3/s2_k1/meta.json` — s2 数据划分

### 源码
- `src/industrial_ad/fusion/v3_3_strategies.py` — 三种融合策略核心实现
- `scripts/evaluate_v3_3_pipeline.py` — 自动化流水线（支持 --seed 0/1/2）

### 数据依赖（外部缓存，V3.3 无 GPU 推理）
- `outputs/dynamic_fusion/v2_mpdd_predictions/v2_mpdd_s{0,1,2}_k1_full_v1/` — DINO + AnomalyCLIP 预测
- `outputs/dynamic_fusion/v3_2_branches/v3_2_mpdd_s0_k1/` — AdaptCLIP 预测（仅 s0）

## 复现方法

```bash
# 单 seed
python scripts/evaluate_v3_3_pipeline.py --seed 0
python scripts/evaluate_v3_3_pipeline.py --seed 1
python scripts/evaluate_v3_3_pipeline.py --seed 2

# Gate B 正式评估（s0）
python scripts/evaluate_v3_3_gate_b.py
```

注意：需要 V2 和 V3.2 分支缓存已存在（见数据依赖）。
