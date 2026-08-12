# 动态融合候选参数完整像素评估

更新日期：2026-08-05

## 1. 评估范围

- 数据：VisA seed 0 开发集，K=2、K=4。
- 候选参数：`temperature=0.50`、`decision_margin=0.15`、`min_weight=0.05`。
- 范围：12 个类别、每个 shot 共 2,162 张测试图。
- 指标：Image AUROC、Image AP、Image F1-max、Pixel AUROC、Pixel AP、AUPRO。
- 安全边界：路由器未使用测试预测进行拟合，也未使用测试标签；报告字段均为 `false`。
- 资源：使用已有冻结分支预测重新融合并在 CPU 上统一评估，没有占用 GPU。

## 2. 完整结果

| Shot | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|---:|---:|
| K=2 | 82.26% | 85.62% | 80.70% | 94.88% | 18.16% | 80.40% |
| K=4 | 82.11% | 85.42% | 80.60% | 94.86% | 17.88% | 78.50% |

机器可读结果：

- `outputs/dynamic_fusion/selected_candidate_20260805/k2/evaluation/summary.csv`
- `outputs/dynamic_fusion/selected_candidate_20260805/k4/evaluation/summary.csv`
- `experiments/dynamic_fusion/selected_candidate_20260805/report.json`

## 3. 与原始动态路由和最佳固定权重比较

| Shot | 方法 | Image AUROC | Pixel AUROC | Pixel AP | AUPRO |
|---|---|---:|---:|---:|---:|
| K=2 | 原始 dynamic | 81.56% | 94.87% | 18.03% | 81.55% |
| K=2 | 最佳固定权重（w=0.75） | 82.07% | 94.88% | 18.13% | 77.41% |
| K=2 | 候选参数 T=0.50 | **82.26%** | 94.88% | **18.16%** | 80.40% |
| K=4 | 原始 dynamic | 81.44% | 94.86% | 17.82% | **79.60%** |
| K=4 | 最佳固定权重（w=0.75） | 82.07% | **94.88%** | **17.89%** | 70.78% |
| K=4 | 候选参数 T=0.50 | **82.11%** | 94.86% | 17.88% | 78.50% |

## 4. 结论

候选参数解决了此前最主要的图像级问题：K=2 和 K=4 的 Image AUROC 都略高于最佳固定权重。Pixel AUROC 和 Pixel AP 与最佳固定权重基本持平，同时 AUPRO 仍明显高于固定权重。

但候选参数没有在所有像素指标上超过原始动态路由：AUPRO 相比原始 dynamic，K=2 下降约 1.15 个百分点，K=4 下降约 1.10 个百分点。这说明同一个温度不一定同时适合图像分数和像素图。

因此，本轮结论是：

1. 可将 `temperature=0.50` 暂时锁定为图像级路由候选；
2. 暂不把它直接锁定为像素级最终参数；
3. 下一步只在 VisA seed 0 上比较“双温度路由”：图像级使用 0.50，像素级保留原始 0.20，并与单温度 0.50 做一项最小消融；
4. 完成这项消融和 K=1 一致性检查后再冻结结构，冻结前不使用 VisA seed 1/2 或 MVTec 最终结果。

## 5. 验收记录

- K=2、K=4 均生成 12 个类别预测并完成统一评估。
- 两个汇总的 `sample_count` 均为 2,162。
- 执行报告为 `status=passed`。
- 标准错误日志为空。
- 评估进程已正常退出，没有残留的本次任务进程。
