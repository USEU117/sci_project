# 当前阶段执行计划（2026-08-03 起）

## 一、阶段目标

在 8 月 13 日前完成所有不需要 GPU 的整理、审计、论文材料和 GPU 队列准备；8 月 13—25 日只运行已经通过预检的长耗时任务。第二阶段动态融合继续遵守“只用 VisA seed 0 做开发，冻结后才用 seed 1/2 和 MVTec 做最终验证”的规则。

## 二、现在至 8 月 6 日：结果整理和论文材料

### 任务 1：锁定 VisA 基线数据表

工作内容：检查主表、逐类别表、均值/标准差、样本数、指标方向、PromptAD 调优标记和文件哈希。

完成标准：四种方法、三个 shot、三个 seed 全部有记录；每个配置能追溯到统一评测报告；PromptAD 单独标注 `target_normal_tuning=true`。

产物：主表、逐类别表、数据字典、结果审计记录。

### 任务 2：制作论文图表草稿

工作内容：制作 shot 对比图、四方法平均性能图、逐类别热力图和均值±标准差表。

完成标准：图表只使用已经完成的 VisA 结果，不把尚未完成的 MVTec 或 ReMP-AD/AdaptCLIP 写成实测结果。

产物：`experiments/summaries/` 下的 CSV/PNG，论文结果章节草稿。

### 任务 3：写清楚实验协议和公平性说明

工作内容：记录数据划分、1/2/4-shot 定义、3-seed 规则、指标定义、PromptAD 调优边界，以及后续动态融合不允许使用的信息。

完成标准：任何表格都能说明训练数据、验证数据和最终测试数据的来源；零样本、少样本、target-normal tuning 方法分开描述。

## 三、现在至 8 月 8 日：动态融合开发轨

### 任务 4：完成 K=2/K=4 校准输入审计

工作内容：核对 K=1/2/4 reference views 的 JSON、manifest SHA256、源图数量、增强视图数量、有限数值和 `test_images_used=false`。

完成标准：K=1、K=2、K=4 均为 passed；不允许出现测试图像或测试标签进入校准输入。

### 任务 5：只在 VisA seed 0、1-shot 上分析路由失败模式

工作内容：比较 visual、text、fixed weight、dynamic 四类输出；按类别统计路由比例、平均权重、冲突程度和异常响应集中度。

完成标准：找出 dynamic 低于最佳 fixed weight 的类别，并给出可验证的原因假设；不查看 seed 1/2 或 MVTec 的最终融合性能。

### 任务 6：预先登记路由消融

固定候选包括：

- visual-only；
- text-only；
- fixed weight：0、0.25、0.5、0.75、1；
- uncertainty-only dynamic；
- consistency-only dynamic；
- uncertainty + consistency dynamic。

完成标准：在看性能之前登记全部候选，避免只挑选最好的结果。

### 任务 7：完成图像级路由接口和可视化

工作内容：输出每张图的视觉权重、文本权重、路由类别、冲突特征和异常分数；制作权重分布图、类别路由图和失败样本清单。

完成标准：接口输入输出固定；不依赖测试真值；能够替换 PatchCore、WinCLIP+、AnomalyDINO 或 PromptAD 的预测缓存。

像素级动态路由暂时只完成接口和合成测试，不在基线未冻结前进行正式调参。

## 四、现在至 8 月 10 日：MVTec 和方法外部依赖预检

### 任务 8：MVTec Gate A 命令 dry-run

工作内容：不启动完整矩阵，只检查 15 类目录、manifest、输出目录、统一评估器、显存参数、断点 marker 和失败保留机制。

完成标准：四个 Gate A 脚本都能在命令层面完成输入检查；每个任务有独立 RunId；不会覆盖已有结果。

### 任务 9：ReMP-AD/AdaptCLIP 依赖和 checkpoint 处理

工作内容：确认 Python/PyTorch 版本、requirements、权重放置位置、权重 SHA256、许可要求和最小单类命令。

完成标准：没有 checkpoint 时明确标记 blocked；checkpoint 到位后先做单类 Gate A，不直接排完整矩阵。

## 五、8 月 10—12 日：离开前总验收

检查项目：

1. 所有 VisA 表格和论文图表可以重新生成；
2. K=1/2/4 参考视图全部通过；
3. MVTec 15 类和 manifest 校验通过；
4. GPU 队列只有一个调度器实例；
5. 每个任务有独立日志、marker、统一评估和失败重试规则；
6. 电脑接通电源，关闭睡眠/休眠；
7. 磁盘空间足以保存 NPZ、热图和日志；
8. 可以查看 `status.json` 和 `scheduler.log`。

## 六、8 月 13—25 日 GPU 队列

严格串行执行：

1. MVTec PatchCore、WinCLIP+、AnomalyDINO、PromptAD Gate A；
2. 通过 Gate A 的方法扩展到 1/2/4-shot、3-seed；
3. ReMP-AD Gate A；
4. AdaptCLIP Gate A；
5. checkpoint 和显存稳定后再进入它们的完整矩阵；
6. 所有基线完成后，执行动态融合 K=2/K=4 校准和正式验证；
7. 最后只运行一次 VisA seed 1/2 和 MVTec 的冻结后最终融合验证。

每个 GPU 任务的验收条件：`.complete` 标记、预测 NPZ、统一评估报告、样本数正确、schema 错误为 0、日志无 traceback。

## 七、预计资源和耗时

| 工作 | 预计耗时 | GPU |
|---|---:|---|
| VisA 表格和图表整理 | 0.5—1 天 | 不需要 |
| 协议、论文方法和公平性说明 | 0.5—1 天 | 不需要 |
| 动态路由失败分析和可视化 | 1—2 天 | 不需要或只做 CPU |
| MVTec Gate A dry-run | 0.5 天 | 不需要 |
| ReMP/Adapt 依赖审计 | 0.5—1 天 | 不需要 |
| MVTec Gate A 实际推理 | 每方法约数十分钟至数小时 | 需要 |
| MVTec 完整矩阵 | 数天至十多天，取决于缓存和失败重试 | 需要 |
| 最终动态融合验证 | 基线完成后再估算 | 需要 |

## 八、阶段出口条件

达到以下条件后，才认为可以从“准备阶段”进入“GPU 批处理阶段”：VisA 结果表和论文图表可追溯、动态融合规则和消融已登记、MVTec Gate A 命令通过预检、ReMP-AD/AdaptCLIP 的 checkpoint 状态明确、自动队列和断点恢复经过检查。
