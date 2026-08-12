# 第二阶段动态融合执行计划

更新日期：2026-08-09

> 当前权威状态：第二阶段的设计、双温度冻结、VisA seed 1/2 最终验证和 MVTec
> 3-seed × 1/2/4-shot 最终验证均已完成并通过审计。下方带日期的早期“当前/待办”
> 内容只保留为历史过程记录，不再代表现在的执行状态。

## 目标和边界

第二阶段以“不确定性路由”为唯一核心创新。少样本是实验设置，动态融合是实现方式。后台基线训练继续使用已经冻结的数据划分、指标和运行协议；设计轨不修改基线预测。

设计开发只允许使用：

- 合成数据；
- VisA seed 0 的冻结预测缓存；
- 少样本训练集中选出的正常参考图；
- 预先声明的源域验证数据。

在设计冻结前，禁止使用 VisA seed 1/2 和 MVTec 的最终融合结果调参。路由器接口禁止接收测试真值、掩码、类别测试标签和完整测试集统计量。

## 工作包状态

- [x] S0：统一输入输出接口、确定性路由器、配置和基础测试。
- [x] WP1：预测缓存、sample ID、标签、掩码和跨-shot顺序对齐。
- [x] WP2：仅使用正常参考图的图像/像素稳健校准与来源审计。
- [x] WP3：熵、增强一致性、跨-shot敏感度和响应集中程度；内部层一致性因
  冻结缓存不含层特征而明确延期，不是最终验证阻塞项。
- [x] WP4：分支一致性与冲突特征。
- [x] WP5：视觉、文本、五个固定权重和动态融合正式对照。
- [x] WP6：连续图像级规则路由。
- [x] WP7：连续像素级路由与图像/像素双温度消融。
- [ ] WP8：数值诊断完成；最终成功/失败案例热图和论文可视化仍待整理。
- [x] Freeze：参数冻结为图像温度0.50、像素温度0.20、margin 0.15、
  min_weight 0.05；视觉分支为AnomalyDINO，文本分支为AnomalyCLIP。
- [x] Final：VisA seed 1/2共6组和MVTec seed 0/1/2共9组完成并通过审计。

当前第二阶段剩余工作只有结果解释和论文材料整理，不再进行V1参数调优。若以后开发
V2，必须建立新的开发/验证边界，不能利用已查看的最终验证集继续调参。

## 串行依赖

`WP1 对齐 → WP2 校准 → WP3/WP4 特征 → WP5 固定基线 → WP6 图像级路由 → WP7 像素级候选 → WP8 分析 → Freeze → Final`

WP3 和 WP4 可以并行准备，但都依赖 WP1；WP5 可以与 WP2 同时写代码，但正式结果必须使用 WP2 冻结的校准方案。

## 与后台训练并行的安排

- GPU 忙：执行接口、单元测试、缓存头审计、统计、文档和 CPU 评测。
- GPU 空闲：优先续跑第一阶段基线；第二阶段不主动占用长期 GPU。
- 每完成一个基线配置：检查 `.complete`、NPZ、统一评测和日志后，再启动下一个配置。
- 每完成一个第二阶段工作包：保存命令、输入文件、配置、报告和测试结果。

## 每项任务的完成判定

一项任务只有同时满足以下条件才标记完成：

1. 代码或文档已经落盘；
2. 有可重复执行的命令；
3. 有机器可读报告；
4. 自动测试通过；
5. 已在本文件和 `PROJECT_STATUS.md` 中登记；
6. 没有使用禁止信息；
7. 失败时保留错误记录，不覆盖成功产物。

## 已完成的第二阶段记录

- `20260730_visa_s0_k1_alignment_anomalydino_winclip`：12 类通过。
- `20260730_visa_s0_k124_shot_consistency`：1/2/4-shot 跨 shot 样本顺序通过。
- `20260730_anomalyclip_sidecar_attempt1_failed`：首次掩码复核失败，原因已记录并保留。
- `outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified`：修正预处理后 12 类 sidecar 全部通过。
- 两次 candle smoke 均出现 200/200 路由到文本分支；这只是分数尺度不一致的诊断，不是性能结论。
- WP2 已完成“正常参考分布校准”接口和 16 项回归测试；尚未用真实参考 shot
  拟合参数，也没有用任何测试集统计量调参。

## 当前下一步

1. 固化 WP1 的三组审计报告和 sidecar 记录。
2. 为两个实际分支导出允许的正常参考 shot 预测。
3. 用参考预测拟合图像和像素两个校准器，并保存参数快照。
4. 校准冻结前继续让 PromptAD seed 1/2 后台训练，不运行 VisA seed 1/2 或 MVTec 动态融合最终验证。

## 2026-07-31 执行更新

### 已完成

- [x] 建立正常参考预测统一格式，包含样本编号、原图编号、增强编号、图像分数和像素图。
- [x] 建立正常参考缓存审计工具，并用冻结 manifest 检查类别、来源、视图数量、数组形状、有限数值和 SHA256。
- [x] 建立逐类别校准拟合工具；视觉分支与文本分支的图像分数、像素图分别拟合。
- [x] 完成 12 类合成参考预测全流程，两个分支均 12/12 通过；两次失败尝试单独保留。
- [x] 从 VisA seed 0、1-shot 冻结清单生成真实正常参考视图：12 张原图、60 个固定视图、0 个失败。
- [x] 完成 AnomalyDINO 视觉分支与 AnomalyCLIP 文本分支的正常参考导出脚本，并完成语法检查。
- [x] 将校准 JSON 接入冻结缓存融合脚本。加载器会拒绝使用过测试预测或测试标签的参数文件。
- [x] 完成合成参数接入冒烟测试，200 个输出的图像分数和像素图均为有限数值；该结果不作性能结论。
- [x] 动态融合与统一评估回归测试共 20 项通过。
- [x] 新增真实正常参考一键流水线，并通过语法、输入路径、五步命令和 GPU 占用保护检查。

### 当前等待

真实正常参考校准已在 `20260731_visa_s0_k1_real_reference_v6_q99` 通过：12/12
类别、两个分支均已审计，且 `test_predictions_used=false`、
`test_labels_used=false`。像素图校准改为每个参考视图的 q99 尾部分数，避免稀疏
正常图的背景零值令 MAD 退化。GPU 调度器已跳过旧失败记录，继续 PromptAD VisA
seed 1、2-shot。

### GPU 释放后的严格顺序

1. 先验收当前 PromptAD 配置：24 个 `.complete`、12 个类别 NPZ、合并结果、统一评估和日志退出码。
2. 根据第一阶段队列决定是否立即续跑下一组 PromptAD；若安排短时设计窗口，先导出 AnomalyDINO 正常参考预测。
3. 审计 AnomalyDINO 缓存，必须 12/12 类通过。
4. 导出 AnomalyCLIP 文本分支正常参考预测。
5. 审计 AnomalyCLIP 缓存，必须 12/12 类通过。
6. [x] 拟合真实 VisA seed 0、1-shot 校准参数，保存输入 SHA256。
7. [ ] 只在 VisA seed 0 冻结测试缓存上做视觉、文本、预注册固定权重和动态路由对照；不读取 VisA seed 1/2 或 MVTec 最终融合结果。
8. [ ] 基于预先声明的规则冻结校准、特征、路由结构和全部超参数；冻结后不再修改。
9. [ ] 基线矩阵完成后，对 VisA seed 1/2 和 MVTec seed 0/1/2 各运行一次独立最终验证。

真实参考预测完成前，WP2 状态仍是“工程线路完成，真实参数未冻结”。

推荐恢复命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_dynamic_fusion_reference_pipeline.ps1 `
  -RunId 20260731_visa_s0_k1_real_reference_v1
```

如果 PromptAD 仍在运行，脚本会主动退出且不创建输出；同一个 `RunId` 已存在时也会拒绝覆盖。

## 与 PromptAD 串行队列的衔接

旧监督器 PID `24720` 已因 PowerShell 单对象 `.Count` 问题退出，当前训练未中断。
统一主调度器 `scripts/run_gpu_job_scheduler.ps1` 已由 PID `23512` 接管。
任务配置为 `configs/gpu_job_queue.json`，它不会与 PromptAD 并发使用 GPU。

执行条件：

1. 当前 PromptAD seed 1、1-shot 进程全部结束；
2. 24 个训练阶段标记齐全；
3. 12 个类别预测 NPZ 齐全；
4. 统一评估覆盖 12 类、2,162 个样本、零 schema 错误。

满足条件后才执行真实正常参考流水线。真实校准无论成功或失败都会保留独立记录；
随后继续 PromptAD seed 1、2-shot，避免第二阶段问题阻塞基线矩阵。

主调度器会在真实校准失败后自动使用新 RunId 重试一次；成功后自动切换到下一组
PromptAD。当前队列运行期间无需用户手动输入命令或发起新对话。

## 2026-08-03 状态更新

- [x] GPU 串行队列 `20260731_full_gpu_queue_v2` 已完成，退出码为 0；PromptAD VisA
  seed 1/2 的 1/2/4-shot 配置均已完成统一评测。
- [x] 真实正常参考校准 `20260731_visa_s0_k1_real_reference_v6_q99` 已通过，明确
  `test_predictions_used=false`、`test_labels_used=false`。
- [x] VisA seed 0、1-shot 的校准开发矩阵已生成 visual/text/fixed/dynamic 五类对照，
  但 dynamic 当前并未稳定优于最佳固定权重，因此只能作为开发诊断，不能冻结或宣称
  方法有效。
- [ ] 仍需补齐 K=2/4 的正常参考校准与开发对照；确认图像级路由是否值得进入冻结。
- [ ] 像素级路由、消融、权重/失败案例可视化和最终冻结尚未完成。

当前第二阶段的正确入口是：先保存并分析 VisA seed 0 开发矩阵，解释 dynamic
低于固定权重的原因；在没有明确改进依据前，不查看 VisA seed 1/2 或 MVTec 融合结果
来反向调参。

## 2026-07-31 GPU 忙时完成的 CPU 工作

- [x] 实现视觉单分支、文本单分支和固定权重统一输出接口。
- [x] 预先登记固定视觉权重 `0/0.25/0.5/0.75/1`，禁止看测试指标后选择。
- [x] 实现图像/像素分支一致性与冲突特征。
- [x] 实现像素响应集中程度。
- [x] 实现正常参考图增强一致性。
- [x] 实现跨 1/2/4-shot 敏感度。
- [x] 缓存运行脚本支持 `visual/text/fixed/dynamic` 四种模式。
- [x] 完成 8 项 CPU 合成线路检查和 28 项回归测试。

以上工作没有读取真值、测试预测或测试指标，也没有使用 GPU。WP3 仍保留“层间一致性”
待办；WP5 只有工程接口完成，必须等待真实校准后才能生成正式对照结果。

## 2026-08-04 K=2/K=4 真实校准与开发矩阵更新

- [x] 将真实正常参考流水线从固定 1-shot 扩展为只允许 VisA seed 0 的 1/2/4-shot，
  并把 shot 写入审计、校准和运行报告。
- [x] 完成 K=2 视觉/文本参考预测、两次缓存审计和正常参考校准。
- [x] 完成 K=4 视觉/文本参考预测、两次缓存审计和正常参考校准。
- [x] 完成 K=2 和 K=4 的 visual/text/fixed-w0/fixed-w0.25/fixed-w0.5/
  fixed-w0.75/fixed-w1/dynamic 开发对照。
- [x] 保留外层超时和 Workers=4 内存不足的失败记录，并加入断点续算模式；失败不覆盖成功产物。
- [x] 历史问题已解决：K=2/K=4 初始 dynamic 的 Image AUROC 低于最佳固定权重；
  后续诊断、双温度消融和参数冻结已经完成。

详细记录：`docs/dynamic_fusion_k2_k4_completion_report_20260804.md`。

## 2026-08-04 seed-0 路由诊断更新

- [x] 完成 K=2/K=4 dynamic 与最佳固定权重的逐类别差异分析。
- [x] 确认 dynamic 的主要问题是图像级 AUROC，AUPRO 在大多数类别提高。
- [x] 确认图像权重和像素权重存在明显差异，不能用一个权重解释两种任务。
- [x] 确认 `temperature=0.20` 使图像权重大量饱和到 0.95，造成视觉主导过多；文本完全主导比例为 0。
- [x] 输出 pcb4、pipe_fryum、fryum、pcb1、cashew 的优先失败类别清单。
- [ ] 预登记并运行 seed-0 温度（0.20/0.35/0.50）和 decision margin（0.05/0.10/0.15）敏感性实验。
- [ ] 根据敏感性结果决定是否冻结连续图像权重、连续像素权重和路由结构。

诊断报告：`docs/dynamic_fusion_seed0_diagnostic_analysis_20260804.md`。

## 2026-08-05 temperature/margin 敏感性更新

- [x] 完成 VisA seed 0、K=2/K=4 的 temperature `{0.20,0.35,0.50}` ×
  decision margin `{0.05,0.10,0.15}` 预登记敏感性分析。
- [x] 确认 margin 只改变 visual/text/weighted 标签数量，不改变连续融合分数和图像指标。
- [x] 确认 K=2/K=4 的图像级最佳候选均为 `temperature=0.50`；K=2 Image AUROC
  82.26%，K=4 Image AUROC 82.11%。
- [x] 确认提高温度降低视觉权重饱和，但仍没有稳定的文本完全主导样本。
- [x] 生成理论像素权重统计和图像级实际指标，保持 `min_weight=0.05`。
- [x] 对 `temperature=0.50, decision_margin=0.15` 的 K=2/K=4 完成完整像素级
  Pixel AUROC、Pixel AP 和 AUPRO；12 类、2,162 张测试图均通过统一评估。
- [x] 确认该候选在 K=2/K=4 的 Image AUROC 均略高于最佳固定权重，Pixel AUROC/AP
  基本持平，AUPRO 明显高于固定权重但比原始 dynamic 低约 1.1 个百分点。
- [x] 已完成最后一项最小消融：图像温度使用0.50、像素温度保留0.20，
  K=1一致性检查通过，图像级和像素级路由参数已经冻结。

敏感性报告：`docs/dynamic_fusion_temperature_margin_sensitivity_20260805.md`。

完整像素评估报告：`docs/dynamic_fusion_selected_candidate_pixel_evaluation_20260805.md`。

## 2026-08-09 WP8 科学分析与可视化完成

本节是第二阶段的最新状态，覆盖上方历史待办。

- [x] 审计 17 个冻结最终运行和 231 个类别—运行组合。
- [x] 比较原始 AnomalyDINO、AnomalyCLIP、固定权重、单温度和双温度动态融合。
- [x] 完成图像权重、像素权重、正常/异常权重、路由比例和类别优势相关性分析。
- [x] 定位主要失败链路：正常参考尺度过小导致 sigmoid 校准饱和；大量分数并列破坏视觉排序；二元熵又把饱和值误判为高置信。
- [x] 完成 K=1/2/4 全部现有消融结果整理。
- [x] 生成校准诊断、跨 shot 对比、逐类别差值热图、路由统计和双温度消融图。
- [x] 按客观定位对比度变化自动选择成功/失败案例，生成原图、真值、两分支热图、融合图和像素权重图。
- [x] 生成机器可读 CSV/JSON、9-sheet Excel 工作簿和两份论文材料说明；输入来源 SHA256 覆盖 693 行、285 个唯一文件。
- [x] WP8 完成；动态融合 V1 的工程、冻结、最终验证和科学分析均已结束。

结论边界：V1 可报告局部定位收益、完整失败分析和协议贡献，但不能宣称全面优于最强单分支。未来如开发 V2，必须重建开发/验证边界，不能继续使用已查看的 VisA seed 1/2 或 MVTec 最终结果调参。

材料入口：

- `docs/dynamic_fusion_scientific_analysis_20260809.md`
- `docs/dynamic_fusion_ablation_and_visualization_20260809.md`
- `experiments/summaries/dynamic_fusion_scientific_analysis_20260809/`
- `outputs/dynamic_fusion/figures/20260809_scientific_analysis/`
- `outputs/dynamic_fusion/analysis_20260809/dynamic_fusion_scientific_analysis_20260809.xlsx`
