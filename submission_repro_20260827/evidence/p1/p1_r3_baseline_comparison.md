# R3 最终 baseline 对照表（macro mean ± std，跨 seed/shot 组合）

生成命令：`python scripts/build_cross_method_comparison_table.py`（输入 `experiments/dynamic_fusion/v3_direction_a/a1_complete_metrics_20260819/` 与 `outputs/unified/**/summary.csv`，输出本 CSV）。

## 数据说明（协议边界，禁止混列）

- **A1 (concat, ours)**：零训练双视觉固定融合（DINO vitb14 + AnomalyCLIP image-tower，KNN normal memory bank），4 数据集 × 3 seed × 1/2/4-shot，共 9 个 seed/shot 组合；image score = 448 异常图 max 池化（与 PatchCore/WinCLIP+ 约定一致）；pixel 指标 stride=8。
- **PatchCore / WinCLIP+ / AnomalyDINO / PromptAD**：MVTec 与 VisA 统一 1/2/4-shot × 3 seed；PromptAD 为 `target_normal_tuning=true`；WinCLIP+ 使用 OpenCLIP ViT-B/16-plus-240。
- **AnomalyDINO**：MVTec/VisA 上多数指标最强（图中 `*` 标记 best）；A1 位居第二，仍超过其余已训练基线。
- **AnomalyCLIP (zs)**：零样本、单组结果（无 seed/shot），单独报告，不与 few-shot 混列。
- **ReMP-AD**：官方协议 train-once、MVTec 4/2/1-shot（3 组，无 seed），图像分数为官方口径 0.5×(文本概率+few-shot max)。
- **AdaptCLIP**：官方 VisA 源域 checkpoint、MVTec 1-shot × 3 seed（3 组），源域 adapter 训练。
- **MPDD/BTAD**：无同协议完整外部训练基线，仅列出 A1（不能暗示 SOTA 对照）。

## 结论边界

- 四数据集稳健提升针对 **Pixel-AP**（相对 matched feature-DINO-only）；BTAD 图像级 Image-AP/F1 有下降，摘要不得写成所有图像/像素指标全面提升。
- 跨方法表中 A1 未超过 AnomalyDINO 的宏平均；A1 的论文定位是「零训练双视觉融合在大多数像素指标上持平/接近最强监督基线」，不得改写为全面 SOTA。
