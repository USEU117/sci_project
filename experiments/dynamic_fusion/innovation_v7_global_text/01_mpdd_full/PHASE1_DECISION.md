# PHASE1 DECISION（机器起草；task book 17 s.3.5）

## G1 硬门评估（Full MPDD 图像级，TEXT vs A1-max，配对逐图）

| 门 | 条件 | 实测 | PASS |
|---|---|---|:---:|
| G1a | 9 配置 macro mean ΔImage-AP ≥ +0.015 | **+0.0288**（bootstrap 全精度 +0.0286） | ✅ |
| G1b | ≥ 7/9 配置 ΔImage-AP > 0 | **6/9** | ❌ |
| G1c | seed1/2 六配置 mean ≥ +0.010 且 ≥4/6 正 | mean **+0.0308**，4/6 正 | ✅ |
| G1d | paired bootstrap 95% CI 下界 > 0 | **−0.0226**（mean 0.0268, P(d>0)=0.858） | ❌ |
| G1e | macro mean ΔImage-AUROC ≥ −0.020 | **+0.0017** | ✅ |
| G1f | ≥3/6 类 mean Δ>0 且 worst cat ≥ −0.100 | 3/6 类正（bb/brb/con），worst metal_plate **−0.0581** | ✅ |
| G1g | 有限、无 ID 错位、无外部集、可重放 | 全通过（Phase 0 audit + 确定性 seed） | ✅ |

**整体 G1 = FAIL**（G1b、G1d 两道硬门未过）。

## 关键事实

- discovery seed0：mean ΔAP +0.0249（2/3 配置正，k4 为负）；
  confirmation seed1/2：mean ΔAP **+0.0308**（4/6 正）——confirmation 并未复现 seed0 的"普遍为正"。
- 逐类（9 配置均值）：bracket_brown +0.0739（9/9 正）、connector +0.1586（8/9）、bracket_black +0.0455（5/9）；
  bracket_white **−0.0415**、metal_plate **−0.0581**（0/9）、tubes **−0.0055**。
- 负配置全部集中在 **k4（多 shot）**：bracket_white k1/2/4 全负、bracket_black k4 −0.137、connector k4、metal_plate 恒定 −0.0581。
  机制：A1 图像分随 k 增加而增强（更优正常记忆 → max-pool 变可靠），而 zero-shot 文本概率逐图恒定，
  两者优势随 shot 收敛；文本只在低 shot 与特定类（connector/bracket）补强 A1。
- bootstrap：mean ΔAP 0.0268，95% CI [−0.0226, 0.0737]，P(Δ>0)=0.858；
  类别聚类 bootstrap CI [−0.0394, 0.1027]，P=0.746 → 6 类数量导致宽区间；
  Wilcoxon p=0.064（单侧）、符号检验 6/9 p=0.254（辅助，配置共享图像非独立）。
- micro pooled（secondary）ΔAP **+0.0259**（正，但被类别样本数主导的 secondary 口径）。

## 结论 → Scenario C（task book 17 s.7）

1. **文本 +0.0249 降级为 seed0 exploratory observation**，不构成稳定的图像级第二贡献；
2. **不运行 BTAD/MVTec 文本验证、不开发 TCRR/R0/R1/R2、不冻结 GLSD**；
3. A1 保持唯一主方法；S1-HGLC/文本证据进入 Appendix/Discussion 负结果；
4. 论文走"简单双视觉表征 + 严格实证与复现 + 完整边界刻画"路线（或按任务书 §8 评级：仅 A1 → 中等偏弱）。

## 证据
- `01_mpdd_full/per_config.csv` · `per_category.csv` · `summary.json` · `bootstrap.json`
- `config_delta_heatmap.png` · `bootstrap_delta_plot.png`
- 复现：`scripts/innovation_v7_global_text/run_mpdd_full_image_evidence.py` +
  `run_paired_bootstrap.py`（B=10,000, seed=20260903，确定性）
