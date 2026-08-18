# V3.3 协议审计报告（2026-08-17）

RunId: `v3_3_audit_20260817` · 类型：协议泄漏审查 · 只读审计，未修改任何旧数值。

参考：`docs/DYNAMIC_FUSION_NEXT_STEPS.md` 阶段一。

## 结论

**V3.3 全部旧结果存在校准泄漏，标记为 development-only，不可作为论文证据。**

泄漏链路（三个策略全部命中）：

```
BranchData.gt_masks（来自缓存 npz 的 imgs_masks = 测试掩码）
  └─> estimate_robust_stats(maps, gt_masks, normal_only=True)
        └─> normal_mask = ~masks.any(axis=(1,2))   # 选「完全正常测试图」
              └─> center=median / scale=IQR         # 用测试图像素估计校准统计
                    └─> compute_z_score(maps, center, scale)  # 校准后加权融合
```

- `weighted_ensemble_fusion`：`v3_3_strategies.py` L116-119
- `max_z_fusion`：`v3_3_strategies.py` L183-188
- `two_stage_calibrated_fusion`：`v3_3_strategies.py` L265-273

即：**校准统计（median/IQR）由测试掩码挑选出的正常测试图计算**。五个泄漏字段：

| 字段 | 值 |
|---|---|
| `test_predictions_used` | false |
| `test_labels_used` | false |
| `test_masks_used` | **true** |
| `test_dataset_statistics_used` | **true** |
| `test_normal_selection_used` | **true** |

## 附加泄漏（safe 退火）

`weighted_ensemble_fusion_safe`（metal_plate_annealing）除 z-score 校准泄漏外，还以「DINO 基线测试 AP > 0.80」作为退火触发信号 —— 属于「按测试指标挑选类别规则」，`test_labels_used` 层面的类别级选择泄漏。

## 受影响报告

`experiments/dynamic_fusion/v3_3/` 下全部结果报告（s0/s1/s2_k1、gate_b、cross_seed、k_cross_shot、btad_holdout、metal_plate_annealing、pipeline_comparison、weight_grid、fusion_decisions、overview）。

统一标记：`development_only_leaky_calibration=true`、`paper_eligible=false`。

## 下一步

阶段二：实现 V3.3-clean —— `RouterInput`/`EvaluationTarget` 分离；校准仅来自当前 seed/shot 的 K 张正常参考图（median/IQR 或 MAD+q95/q99）；五项泄漏字段全 false；补齐 CPU 测试后再重跑开发 Gate。
