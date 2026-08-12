# VisA 基线整理状态（2026-08-03）

## 数据和统一评测

- 数据集：VisA，12 个类别。
- 每个完整配置：2,162 张测试图。
- shot：1、2、4。
- seed：0、1、2。
- 统一指标：Image AUROC、Image AP、Image F1-max、Pixel AUROC、Pixel AP、AUPRO。
- 实验登记：`experiments/registry.csv`，当前 53 条记录，校验错误数为 0。

## 已完成完整矩阵的方法

| 方法 | 1/2/4-shot × 3-seed | 统一汇总文件 | 备注 |
|---|---|---|---|
| PatchCore | 已完成 | `experiments/summaries/patchcore_visa_unified/` | 视觉检索基线 |
| WinCLIP+ | 已完成 | `experiments/summaries/winclip_visa_unified/` | 文本/视觉 CLIP 基线 |
| AnomalyDINO | 已完成 | `experiments/summaries/anomalydino_visa_unified/` | 视觉强基线 |
| PromptAD | 已完成 | `experiments/summaries/promptad_visa_unified/` | `target_normal_tuning=true`，单独解释 |

## PromptAD 均值 ± 样本标准差

| Shot | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|---:|
| 1-shot | 80.93% ± 1.47% | 83.66% ± 0.88% | 96.09% ± 0.13% | 28.15% ± 0.35% | 81.40% ± 0.28% |
| 2-shot | 81.78% ± 1.51% | 84.37% ± 1.25% | 96.66% ± 0.11% | 29.66% ± 0.03% | 82.62% ± 0.61% |
| 4-shot | 81.55% ± 1.95% | 83.99% ± 1.55% | 97.07% ± 0.15% | 32.25% ± 0.73% | 83.73% ± 0.53% |

数值由 `scripts/summarize_unified_matrix.py` 根据 9 个 `outputs/unified/promptad_visa_seed_*` 配置自动生成，未手工修改。

## 尚未形成完整 VisA 对比表的内容

- AnomalyCLIP 的完整少样本统一矩阵仍需单独核验。
- 需要把四种方法按 shot 合并成论文主表，并补充逐类别结果、运行时间、显存和失败案例。
- PromptAD 因为使用目标正常样本调优，不能与零样本方法直接混成一列解释。

## 整理验收

- [x] 9 个 PromptAD 配置均有 `summary.csv`、`per_category.csv`、`per_image.csv` 和 `evaluation_report.json`。
- [x] 每个配置的样本数为 2,162。
- [x] 实验登记表校验：53 条记录，0 个错误。
- [x] 所有结果保留原始逐运行目录，可重新生成汇总。
