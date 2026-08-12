# CPU 工作完成报告（2026-08-03）

## 1. 本次目标

本次只处理不需要 GPU 的工作：整理 VisA 结果、生成逐类别表、分析动态融合开发结果、准备 K=2/K=4 参考视图、检查 MVTec Gate A 脚本，并审计 ReMP-AD 与 AdaptCLIP 的运行条件。没有启动训练，也没有修改已有预测文件。

## 2. 已完成的结果整理

### VisA 主表

- 主表：`experiments/summaries/visa_baseline_main_table_20260803.csv`
- 结果状态：4 个方法（PatchCore、WinCLIP+、AnomalyDINO、PromptAD）均已覆盖 1/2/4-shot、3 个 seed。
- 汇总规模：12 行方法配置，指标包括 image AUROC、image AP、image F1-max、pixel AUROC、pixel AP、AUPRO。
- 状态说明：PromptAD 的结果仍需在论文中标注 `target_normal_tuning=true`，不能和完全零样本方法混写。

### VisA 逐类别表

- 明细表：`experiments/summaries/visa_per_category_long_20260803.csv`（432 行）
- 均值/标准差表：`experiments/summaries/visa_per_category_mean_std_20260803.csv`（144 行）
- 已统一 PatchCore 的 `mvtec_` 类别前缀，便于四种方法按同一类别名比较。
- 生成脚本：`scripts/build_visa_per_category_table.py`

## 3. 动态融合开发结果分析

- 分析脚本：`scripts/analyze_dynamic_fusion_development.py`
- 输出：`experiments/dynamic_fusion/20260803_cpu_route_analysis/route_stats.csv`
- 输入：VisA seed 0、1-shot 的已冻结预测缓存；未读取测试真值、测试标签或测试集整体统计量。
- 共分析 2162 个样本：视觉主导 770、文本主导 30、加权融合 1362；视觉权重均值约 0.6416，标准差约 0.1882。
- 该结果说明路由器确实会在不同样本间改变分支权重，但它只是开发期诊断，不能当作最终泛化结论。正式锁定前仍需 K=2/K=4 校准和独立 seed 验证。

## 4. K=1/2/4 参考视图准备

参考视图是用正常训练图像生成的身份、亮度和对比度轻微变化版本，供后续可靠性/不确定性校准使用。

| shot | 输出目录 | 状态 | 规模 | 备注 |
|---|---|---|---:|---|
| K=1 | `outputs/dynamic_fusion/reference_views/20260730_visa_s0_k1_v1` | passed | 12 个类别、60 个视图 | 已有验证结果 |
| K=2 | `outputs/dynamic_fusion/reference_views/20260803_visa_s0_k2_v2` | passed | 12 个类别、24 个正常源、120 个视图 | 使用已验证的 `data/visa_raw` |
| K=4 | `outputs/dynamic_fusion/reference_views/20260803_visa_s0_k4_v1` | passed | 12 个类别、48 个正常源、240 个视图 | 使用已验证的 `data/visa_raw` |

K=2 的 `20260803_visa_s0_k2_v1` 是一次错误数据根目录的失败尝试，已保留作为过程记录；成功版本使用了新的 `v2` 目录，没有覆盖失败证据。

四个参考视图 JSON 都记录了 manifest SHA256、源图 SHA256、生成视图 SHA256，并明确 `test_images_used=false`、`test_labels_used=false`。

## 5. MVTec Gate A 预检

已检查以下脚本的路径、manifest、统一评估器和退出码检查：

- `scripts/run_patchcore_mvtec_gate.ps1`
- `scripts/run_winclip_mvtec_gate.ps1`
- `scripts/run_anomalydino_mvtec_gate.ps1`
- `scripts/run_promptad_gate.ps1`

脚本层面已经具备运行条件，但本次没有执行 Gate A，因为这些脚本会启动 GPU 推理。当前 MVTec 尚未形成四种方法的完整矩阵，后续应按 Gate A → Gate B → 全矩阵的顺序排队。

## 6. ReMP-AD 与 AdaptCLIP 审计

- ReMP-AD：代码、README、requirements 和配置文件存在；方法目录没有找到可直接使用的 `.pth/.pt/.ckpt/.bin` 权重。
- AdaptCLIP：代码、README、训练/测试脚本和 requirements 存在；README 要求把 checkpoint 放入 `adaptclip_checkpoints`，当前目录没有找到对应权重。
- 因此两种方法目前是“代码可审计、权重未到位”，不是“已完成复现”。这属于外部 checkpoint/依赖阻塞，不应通过改协议或临时下载未知权重解决。

## 7. 结论与交接

本轮 CPU 工作已完成。现在可以直接用于论文前期整理的内容包括 VisA 主表、逐类别均值/标准差表、动态路由分布分析和 K=1/2/4 参考视图。仍需 GPU 或外部资源的内容是：MVTec Gate A/完整矩阵、动态融合 K=2/K=4 实际特征校准与正式验证、ReMP-AD/AdaptCLIP 的 checkpoint 获取及复现。

建议在 8 月 13–25 日的空闲窗口运行已审核过的 GPU 队列，保持 6 GB 显存设备串行执行，并在每个任务结束后写入 marker、统一评估报告和日志；论文撰写期间不需要等待这些长耗时任务。
