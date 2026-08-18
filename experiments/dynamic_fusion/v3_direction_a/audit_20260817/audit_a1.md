# 7.1-A1 协议审计报告（2026-08-17）

RunId: `v3_direction_a_a1_audit_20260817` · 类型：协议泄漏审查 · 只读审计。

参考：`docs/DYNAMIC_FUSION_NEXT_STEPS.md` 阶段五 5.1。

## 结论

**A1（静态 concat + KNN memory bank）未发现泄漏**，是当前最值得继续严格审计的候选。

审计对象：`export_anomalydino_mpdd_features.py`、`export_anomalyclip_mpdd_features.py`、`evaluate_a1_feature_fusion.py`。

## 关键证据

1. **正常参考来源**：两个导出脚本的 `ref_patch_features` 均只来自 `manifest["categories"][cat][seed][shot]` 预声明的 K 张正常参考图（[export_anomalydino_mpdd_features.py](file:///d:/STUDY/My_github/sci_project/scripts/export_anomalydino_mpdd_features.py#L71-L79)、[export_anomalyclip_mpdd_features.py](file:///d:/STUDY/My_github/sci_project/scripts/export_anomalyclip_mpdd_features.py#L110-L119)），与测试图特征分开存储。
2. **拟合与记忆库只使用正常参考**：[evaluate_a1_feature_fusion.py](file:///d:/STUDY/My_github/sci_project/scripts/evaluate_a1_feature_fusion.py#L87-L101) 中 PCA/whitening 仅 fit 于 `ref_flat`，KNN memory bank 由 `ref_proj` 构建；测试特征只做 transform/search。
3. **concat 变换无信息源违规**：concat 前各分支 patch 仅做 L2-normalize（L134-140），无去中心化、无投影对齐。
4. **测试真值仅限评价**：`imgs_masks`/`gt_sp` 只在 `compute_metrics` 阶段用于指标计算，不进入任何拟合/校准/路由路径（L211-215、L228-239）。

## 泄漏字段（五项全 false）

| 字段 | 值 |
|---|---|
| `test_predictions_used` | false |
| `test_labels_used` | false |
| `test_masks_used` | false |
| `test_dataset_statistics_used` | false |
| `test_normal_selection_used` | false |

## 下一步

按阶段三：MPDD seed0/K1 CPU Gate —— 固定唯一 A1 配置（w=0.5, pca_dim=0），预注册小网格，对照 DINO/CLIP/50-50/固定权重/V3.2，报告 Image AUROC/AP/F1、Pixel AUROC/AP/AUPRO、逐类 delta、rescue/harm/coverage/risk-coverage。
