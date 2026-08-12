# VisA 实验协议、公平性边界与结果章节初稿

更新日期：2026-08-04

## 1. 本文档用途

本文档整理当前已经完成的 VisA 基线实验，可直接作为论文“实验设置”和“实验结果”章节的初稿。本文只报告本地统一评测结果，不把论文原文数字、MVTec 未完成矩阵或 ReMP-AD/AdaptCLIP 未复现结果写成本地实验结果。

## 2. 实验对象

当前可比较的方法为 PatchCore、WinCLIP+、AnomalyDINO 和 PromptAD。数据集为 VisA，共 12 个类别。每种方法均使用同一份 1/2/4-shot manifest，并运行 seed 0、1、2。每个配置的统一测试集包含 2,162 张图像。

少样本中的 shot 表示每个类别可使用的正常参考图像数量：1-shot 使用 1 张，2-shot 使用 2 张，4-shot 使用 4 张。不同 seed 用于改变正常参考图的抽样，测试集保持不变。

## 3. 数据和信息使用边界

- 基线方法使用冻结的 VisA manifest，不允许临时更换正常参考图。
- 分数方向统一为 `higher_is_more_anomalous`，即分数越高越可能异常。
- 测试标签只用于最终指标计算，不能用于校准、选择权重或调整路由器。
- 动态融合开发只允许使用 VisA seed 0；seed 1/2 和 MVTec 留作设计冻结后的最终验证。
- PromptAD 使用目标类别正常样本进行提示调优，因此必须标记 `target_normal_tuning=true`。
- PatchCore、WinCLIP+、AnomalyDINO 和 PromptAD 的训练方式并不完全相同。当前表格是统一数据划分和统一指标下的工程比较，不应描述为训练预算完全相同。

## 4. 评测指标

- Image AUROC：判断整张图是否异常的排序能力，越高越好。
- Image AP：图像级精确率—召回率曲线下面积，越高越好。
- Image F1-max：在所有可能阈值中取得的最高图像级 F1，仅用于描述可达到的分类平衡。
- Pixel AUROC：判断每个像素是否异常的排序能力，越高越好。
- Pixel AP：像素级精确率—召回率曲线下面积，对异常区域较小的数据更敏感。
- AUPRO：按异常区域计算的重叠表现，本项目固定最大假阳性率为 0.3。

所有论文主表使用三个 seed 的均值和标准差。逐类别表同样先按 seed 汇总，不用最好的一次 seed 代替均值。

## 5. 统一结果主表

下表单位均为百分数，格式为三个 seed 的均值。标准差保存在机器可读主表中。

| 方法 | shot | Image AUROC | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|---:|
| PatchCore | 1 | 68.03 | 85.96 | 22.54 | 50.60 |
| PatchCore | 2 | 72.91 | 90.04 | 25.53 | 57.96 |
| PatchCore | 4 | 78.68 | 91.95 | 28.53 | 62.21 |
| WinCLIP+ | 1 | 69.96 | 89.98 | 10.33 | 67.34 |
| WinCLIP+ | 2 | 71.57 | 90.50 | 10.87 | 68.32 |
| WinCLIP+ | 4 | 72.58 | 90.88 | 11.19 | 69.05 |
| AnomalyDINO | 1 | 89.40 | 97.97 | 39.07 | 92.21 |
| AnomalyDINO | 2 | 91.40 | 98.28 | 41.46 | 93.10 |
| AnomalyDINO | 4 | 92.58 | 98.45 | 42.97 | 93.69 |
| PromptAD* | 1 | 80.93 | 96.09 | 28.15 | 81.40 |
| PromptAD* | 2 | 81.78 | 96.66 | 29.66 | 82.62 |
| PromptAD* | 4 | 81.55 | 97.07 | 32.25 | 83.73 |

`* PromptAD`：本地协议使用目标类别正常样本进行提示调优，`target_normal_tuning=true`。

## 6. 当前结果可以支持的结论

在当前统一协议下，AnomalyDINO 在三个 shot 设置的 Image AUROC、Pixel AUROC、Pixel AP 和 AUPRO 上均为四种方法中的最高值。它从 1-shot 增加到 4-shot 后，Image AUROC 提高约 3.18 个百分点，AUPRO 提高约 1.48 个百分点。

PatchCore 对参考图数量最敏感：从 1-shot 增加到 4-shot 后，Image AUROC 提高约 10.65 个百分点，AUPRO 提高约 11.61 个百分点。这说明它在极少参考图时不稳定，但增加正常参考样本后改善明显。

WinCLIP+ 的变化较平缓，4-shot 相比 1-shot 的 Image AUROC 提高约 2.61 个百分点，AUPRO 提高约 1.71 个百分点。它的 Pixel AP 明显低于其他方法，因此不能只看 Pixel AUROC 判断像素定位质量。

PromptAD 的 Pixel AUROC、Pixel AP 和 AUPRO 随 shot 总体提高，但 Image AUROC 在 2-shot 后没有继续提高：2-shot 为 81.78%，4-shot 为 81.55%。因此不能简单写成“shot 越多所有指标都越高”。

逐类别热力图显示，不同方法的薄弱类别并不相同。例如 PatchCore 在 capsules 上的 4-shot Image AUROC 约为 50.5%，AnomalyDINO 在同一类别约为 98.4%。这些差异为后续动态融合提供了动机，但不能仅凭类别平均结果证明动态路由一定有效。

## 7. 暂时不能写入论文结论的内容

- 不能声称动态融合已经优于固定权重；当前 seed-0 开发结果还不支持这一结论。
- 不能写 ReMP-AD 和 AdaptCLIP 的本地结果，因为 checkpoint 和 Gate A 尚未完成。
- 不能写完整 MVTec 横向结论，因为完整矩阵尚未运行。
- 不能根据 VisA seed 1/2 或 MVTec 的最终结果反向修改动态融合规则。

## 8. 可追溯产物

- 主表：`experiments/summaries/visa_baseline_main_table_20260803.csv`
- 逐类别明细：`experiments/summaries/visa_per_category_long_20260803.csv`
- 逐类别均值/标准差：`experiments/summaries/visa_per_category_mean_std_20260803.csv`
- 36 配置审计：`experiments/summaries/visa_result_audit_20260804.json`
- 审计明细：`experiments/summaries/visa_result_audit_20260804.csv`
- 方法比较图：`experiments/summaries/figures/visa_method_comparison_20260804.png`
- shot 趋势图：`experiments/summaries/figures/visa_shot_trends_20260804.png`
- 逐类别热力图：`experiments/summaries/figures/visa_category_heatmap_20260804.png`

图表和审计文件可使用以下命令重新生成：

```powershell
& .venv-patchcore\Scripts\python.exe scripts\audit_and_plot_visa_results.py
```
