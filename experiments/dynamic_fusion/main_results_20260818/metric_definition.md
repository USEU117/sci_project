# 指标定义（metric_definition.md）

- Pixel AP: 像素级 Average Precision，基于 GT 掩码与异常图二分类阈值扫描计算。
- Pixel AUROC: 像素级 Area Under ROC。
- Pixel AUPRO: 像素级 Area Under Per-Region Overlap（stride=8 下采样后计算，与冻结 evaluator 一致）。
- ΔAP: `fused Pixel AP − baseline Pixel AP`；baseline 每行显式标注 source。
- MPDD 三口径: ①A1 concat vs legacy v2 dino score = +0.0486；②feature-DINO-only vs legacy = +0.0227；③concat vs matched feature-DINO-only = +0.0258（纯融合贡献）。
- VisA/MVTec/BTAD 无 v2 分数级缓存 → baseline 用特征级 dino-only KNN（MPDD s0/K1 上该口径与 v2 分数级差 ~0.0008 AP，可比）。
- 逐类表（per_category_results.csv）取每个数据集 seed0/K1（BTAD seed0）的 concat 报告。