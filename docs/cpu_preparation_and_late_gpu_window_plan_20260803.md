# CPU 准备工作与 8 月 13—25 日 GPU 批处理计划

更新日期：2026-08-03

## 结论

可以把 MVTec 完整矩阵、ReMP-AD/AdaptCLIP 完整矩阵等长耗时 GPU 工作集中到 8 月 13—25 日运行。当前先做 CPU 整理和 Gate A 准备是合理的，因为这些工作不会改变已有预测，也不会占用 GPU。

前提是：在离开前完成数据、环境、checkpoint、命令、队列和自动验收的预检查；电脑全程接通电源，禁止睡眠/休眠，保留足够磁盘空间，并准备远程查看日志的方式。

## 当前已完成的 CPU 工作

- VisA 四种方法的 mean/std 汇总已经生成。
- PromptAD 的 9 个 VisA 配置已登记为 12 类、2,162 样本、统一评测无 schema 错误。
- PromptAD 的 `target_normal_tuning=true` 已确认并需要单独标注。
- 动态融合 VisA seed 0、1-shot 开发矩阵已经有 visual/text/fixed/dynamic 对照。
- 实验登记表当前 53 条记录，校验错误数为 0。

## 仍可在 CPU 完成的工作

1. 合并 VisA 四种方法的论文主表和逐类别表。
2. 检查所有结果的样本数、指标字段、标签方向、配置路径和 SHA256。
3. 分析动态融合开发矩阵中 dynamic 低于最佳 fixed-weight 的类别和失败模式。
4. 整理 K=2/4 正常参考校准所需的 views JSON、manifest 和输出目录；正式分支预测导出仍需要 GPU。
5. 固化 MVTec Gate A/B 命令、输出目录和验收规则。
6. 审计 ReMP-AD、AdaptCLIP 的 requirements、README、运行脚本和 checkpoint 状态。

## GPU 工作分层

### 短任务：离开前可先做 Gate A

- 单类别、1-shot、seed 0。
- 每个方法约 30—90 分钟，具体取决于模型和缓存状态。
- 目的：只验证数据入口、环境、显存和 NPZ 导出，不进入完整矩阵。

### 长任务：集中在 8 月 13—25 日

- MVTec 15 类、1/2/4-shot、3-seed。
- PromptAD MVTec 完整矩阵。
- 通过 Gate A 后的 ReMP-AD/AdaptCLIP 矩阵。
- 单卡必须串行，不能同时启动两个训练/推理进程。

## 8 月 13—25 日的建议队列

1. 验证已有 MVTec Gate A 结果和统一评测。
2. WinCLIP+ MVTec 2/4-shot 与 seed 1/2。
3. PatchCore MVTec Gate B，再扩展完整矩阵。
4. AnomalyDINO MVTec Gate B，再扩展完整矩阵。
5. PromptAD MVTec Gate A，再扩展完整矩阵。
6. ReMP-AD Gate A；只有通过后才排完整矩阵。
7. AdaptCLIP Gate A；只有 checkpoint 和显存检查通过后才排完整矩阵。
8. 每个方法完成后立即做统一评测和结果登记，再进入下一个方法。

## 离开前必须完成的检查

- [ ] 所有数据目录和 manifest 存在，manifest SHA256 已记录。
- [ ] MVTec 15 类目录完整，正常图、测试图和掩码数量通过校验。
- [ ] WinCLIP+、PatchCore、AnomalyDINO、PromptAD 环境可以启动。
- [ ] ReMP-AD 和 AdaptCLIP 的 checkpoint 已下载并计算 SHA256；目前项目目录还没有发现对应 checkpoint 文件。
- [ ] 每个 Gate A 命令已经手动或 dry-run 验证过。
- [ ] GPU 队列只允许一个调度器实例，支持断点恢复和失败保留。
- [ ] 笔记本接通电源，睡眠和休眠策略已验证。
- [ ] 磁盘空间足够保存预测 NPZ、热图、日志和中间缓存。
- [ ] 已确认可以远程查看 `status.json`、`scheduler.log` 和最新文件时间。

## 不建议的做法

- 不建议离开前直接启动完整矩阵而不先跑 Gate A。
- 不建议把 ReMP-AD 或 AdaptCLIP 的官方论文数字当成本地复现结果。
- 不建议在队列运行期间修改数据划分、分辨率或指标定义。
- 不建议通过关闭日志、删除失败目录或覆盖旧结果来“保持队列继续”。

## 最终时间判断

如果所有 Gate A 在离开前通过，8 月 13—25 日的 12 天窗口足以运行主要 MVTec 长任务，但不保证所有 ReMP-AD/AdaptCLIP 组合都能完成。它们是否纳入完整矩阵，必须由 Gate A 的显存、checkpoint 和输出稳定性决定。
