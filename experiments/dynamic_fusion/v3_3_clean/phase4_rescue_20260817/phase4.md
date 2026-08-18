# V3.3-clean 阶段四：视觉锚定文本局部救援（MPDD seed0/K1，2026-08-17）

RunId: `v3_3_clean_phase4_rescue_20260817_mpdd_s0_k1` · CPU，无 GPU，复用冻结缓存。
参考：`docs/DYNAMIC_FUSION_NEXT_STEPS.md` 阶段四。

## 方法（预注册固定流程）

```
视觉异常图 -> 视觉候选区域(仅视觉, q95 参考支持) -> 背景拒绝(边界+低纹理)
-> prompt/增强稳定性(参考视图极差) -> 有界单向文本残差(cap) -> 视觉回退
```

- 校准全部来自 K 张正常参考的**池化**统计（median/IQR/q95，与 v3_3_clean 一致）。
- 候选区域：`visual_z > q95(参考 visual z)`，边界 margin=4px 剔除。
- 文本残差：`min(max(text_z - q95(参考 text z), 0), residual_cap)`，仅在视觉候选内。
- 区域原因代码：`no_visual_candidate / reference_in_support / prompt_unstable /
  background_rejected / bounded_text_residual / visual_fallback`（逐像素保存）。
- 五个泄漏字段全 false；未用测试真值拟合任何阈值。

## 关键实现教训（本轮实测）

1. **逐像素参考统计不可用**（K=5 视图）：逐像素 IQR 过噪导致 z-score 爆炸，
   pixel AP 崩塌至 0.0866（vs visual 0.2802）。改用池化 median/IQR 后恢复。
2. **分位数 z 换算 bug**：`(q_value - center)/scale` 必须用**参考分位数数值**
   （如 q95=0.009），不能直接用分位数水平 0.95（会把 support_z 算成 427 而拒绝全部候选）。
3. 边界剔除必须用 `cand[..., :margin, :]`（3D 数组）而非 `cand[:margin, :]`
   （会切 batch 维）。

## 结果（6 类聚合，Pixel AP = mean）

| 方法 | Pixel AP | Δ vs visual | Pixel AUROC | AUPRO | Img AUROC |
|---|---|---|---|---|---|
| visual_only | 0.2802 | — | 0.9578 | 0.8622 | 0.6643 |
| v33_clean_w0.40（阶段三最优） | 0.2975 | **+0.0173** | 0.9609 | 0.8741 | 0.6794 |
| rescue_cap1.0 | 0.2816 | +0.0014 | 0.9582 | 0.8644 | 0.6657 |
| rescue_cap2.0 | 0.2829 | +0.0027 | 0.9584 | 0.8649 | 0.6673 |
| rescue_cap4.0 | 0.2853 | +0.0051 | 0.9587 | 0.8666 | 0.6689 |
| rescue_unstable（文本强制不可靠） | 0.2821 | +0.0018 | 0.9580 | 0.8632 | 0.6674 |

逐类（cap2.0 Δ vs visual）：bracket_white +0.0082、metal_plate +0.0034、
connector +0.0024、tubes +0.0010、bracket_black +0.0011、bracket_brown +0.0002。
**6/6 全正，无单类退化。**

## 解读

- 局部救援**方向正确、安全**（每类全正、AUPRO 不降、文本不可靠时纯视觉回退），
  但增益（cap4 ≈ +0.005）**显著小于** v3.3-clean 固定加权（w0.40 ≈ +0.017）。
- 原因：文本残差只在「视觉候选 ∩ 文本超参考支持」的 ~6~13% 像素上生效，
  保守阈值（q95）限制了文本的贡献面；而 clean 的全图加权能更充分地利用互补性。
- **决策含义**：按计划"再判断动态路由是否超过最佳固定融合"——本 Gate 数据下
  动态/局部救援（+0.005）**未超过**最佳固定融合 v33_clean_w0.40（+0.0173），
  当前保留 w=0.40 固定加权作为 V3.3-clean 主配置；局部救援保留为安全回退组件
  （`visual_fallback` 路径，任何不可靠条件均回退视觉）。

## 产物

- 模块：`src/industrial_ad/fusion/v3_3_rescue.py`（LocalRescueConfig + local_rescue_fusion）
- 测试：`tests/test_v3_3_rescue.py`（13 个 CPU 测试，通过；连同 clean 共 28/28）
- 评估：`scripts/evaluate_v3_3_rescue_phase4.py`
- 报告：`experiments/dynamic_fusion/v3_3_clean/phase4_rescue_20260817/report.json`
