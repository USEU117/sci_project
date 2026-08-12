# 动态融合 K=2/K=4 GPU 工作完成报告

更新日期：2026-08-04

## 1. 本轮范围

本轮只使用 VisA seed 0 的正常参考图和冻结的 seed-0 测试预测缓存。目标是完成 K=2、K=4 的真实正常参考预测、分支审计、校准和开发期融合对照。没有使用 VisA seed 1/2 或 MVTec 的最终结果调参，也没有启动第五阶段基线矩阵。

## 2. GPU 参考预测与校准

| shot | RunId | 视觉导出 | 文本导出 | 校准 | 测试信息进入校准 |
|---:|---|---|---|---|---|
| 2 | `20260804_visa_s0_k2_real_reference_v1_q99` | 12/12 | 12/12 | passed | predictions=false, labels=false |
| 4 | `20260804_visa_s0_k4_real_reference_v1_q99` | 12/12 | 12/12 | passed | predictions=false, labels=false |

每个类别均记录了参考源数量、视图数量、缓存 SHA256 和校准参数。K=2 为 24 个正常源、120 个增强视图；K=4 为 48 个正常源、240 个增强视图。两次 GPU 流水线均保存了 `run.json`、`report.json`、`command.txt`、分步 stdout/stderr 和 decision 文件。

## 3. 开发矩阵

两个 shot 都完成了以下 8 种预登记模式：visual、text、fixed-w0、fixed-w0.25、fixed-w0.5、fixed-w0.75、fixed-w1、dynamic。统一评测均覆盖 12 类、2,162 个测试样本，使用原有 200 步 AUPRO 协议。

### K=2 seed-0 开发结果

结果目录：`outputs/dynamic_fusion/development_matrix/20260804_visa_s0_k2_calibrated_development_matrix_v3`

| 模式 | Image AUROC | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|
| visual | 60.48% | 84.30% | 5.73% | 32.997% |
| text | 81.97% | 93.74% | 17.77% | 75.519% |
| fixed-w0.25 | 81.99% | 94.81% | 18.06% | 77.079% |
| fixed-w0.5 | 82.02% | 94.86% | 18.09% | 77.323% |
| fixed-w0.75 | 82.07% | 94.88% | 18.13% | 77.406% |
| dynamic | 81.56% | 94.87% | 18.03% | 81.546% |

动态路由分布：视觉主导 1,631 张，加权融合 531 张；文本完全主导 0 张。动态的 AUPRO 高于固定权重，但 Image AUROC 低于最佳固定权重。

### K=4 seed-0 开发结果

结果目录：`outputs/dynamic_fusion/development_matrix/20260804_visa_s0_k4_calibrated_development_matrix_v1`

| 模式 | Image AUROC | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|
| visual | 59.80% | 83.74% | 5.95% | 32.942% |
| text | 81.97% | 93.14% | 17.41% | 73.278% |
| fixed-w0.25 | 81.99% | 94.80% | 17.78% | 73.341% |
| fixed-w0.5 | 82.01% | 94.86% | 17.83% | 73.143% |
| fixed-w0.75 | 82.07% | 94.88% | 17.89% | 70.783% |
| dynamic | 81.44% | 94.86% | 17.82% | 79.599% |

动态路由分布：视觉主导 1,932 张，加权融合 230 张；文本完全主导 0 张。动态的 AUPRO 高于固定权重，但 Image AUROC 仍低于最佳固定权重。

## 4. 资源问题和解决记录

- K=2 开发矩阵第一次运行因外层等待超时中断；原失败 RunId 保留，之后用断点续算完成。
- K=4 使用 Workers=4 评估时出现内存不足 traceback；该失败记录保留，之后改用 Workers=1 完成。
- GPU 参考导出和校准均正常；K=2/K=4 开发矩阵本身使用冻结缓存做 CPU 融合和评估。

## 5. 当前结论

K=2 和 K=4 的真实校准输入已经准备完毕，动态路由对像素区域的 AUPRO 有改善，但图像级 AUROC 没有超过最佳固定权重。因此当前证据支持“动态路由可能改善区域定位”，不支持直接冻结为最终方法。

下一步仍应只在 seed 0 开发轨完成失败类别分析、路由阈值和特征消融；只有设计冻结后，才运行 VisA seed 1/2 和 MVTec 的最终动态融合验证。
