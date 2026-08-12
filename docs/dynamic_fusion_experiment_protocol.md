# 动态融合实验记录规范

## 目录约定

小型记录放在：

`experiments/dynamic_fusion/<run_id>/`

大型预测和可视化放在：

`outputs/dynamic_fusion/<work_package>/<run_id>/`

每个 `run_id` 使用：

`YYYYMMDD_数据集_seed_shot_任务_分支`

例如：

`20260730_visa_s0_k1_alignment_anomalydino_winclip`

## 每次运行必须保存

- `run.json`：运行目的、时间、Git 状态、数据范围、输入和输出路径；
- `command.txt`：可以直接复现的完整命令；
- `stdout.log`：标准输出和错误输出；
- `config.yaml`：本次使用的配置快照；
- `report.json`：机器可读结果；
- `report.csv`：方便人工查看的表格；
- `decision.md`：结论、限制、失败原因和下一步。

如果运行失败，也保存同样的文件，并把状态写为 `failed`。不删除或覆盖已经通过的运行。

## 数据来源字段

`run.json` 至少记录：

- dataset、seed、shot、categories；
- visual_branch、text_branch；
- 输入缓存绝对路径、文件大小和 SHA256；
- manifest 路径与 SHA256；
- 是否使用 GPU；
- 是否读取测试真值；
- 真值只用于“审计/最终评测”还是被错误用于“路由/调参”；
- Git commit 和工作区是否有未提交改动；
- 开始时间、结束时间和退出码。

## 禁止信息检查

路由器的输入对象不允许出现：

- `gt`、`label`、`mask`；
- 测试类别的最终结果；
- 完整测试集的均值、标准差、最大值、最小值和分位数；
- 根据测试集表现选择的阈值或权重。

审计工具和最终评测器可以读取真值，但必须与路由模块分开，并在记录中注明用途。

## 结果解释规则

- seed 0 结果只用于开发和排错，不写成最终泛化结论；
- 单类别结果只叫“冒烟测试”，不叫“方法有效”；
- 动态融合必须与视觉单分支、文本单分支和固定权重融合比较；
- 分数未校准前，不比较两分支的熵，也不声明路由性能；
- 最终表同时报告图像指标、像素指标、均值、标准差和额外运行时间。

## 正常参考校准补充规范（2026-07-31）

### 允许的数据

- 只允许读取冻结 few-shot manifest 中的正常训练参考图。
- 同一张参考图可使用预先写死的确定性视图；当前版本为原图、亮度 0.90/1.10、对比度 0.90/1.10。
- 视觉分支和文本分支必须使用同一组 `source_id` 与 `augmentation_id`。

### 强制审计

每个正常参考 NPZ 必须保存 `sample_ids`、`source_ids`、`augmentation_ids`、
`image_scores` 和 `pixel_maps`。审计报告必须明确写出：

- dataset、seed、shot、branch 和 category；
- manifest 路径与 SHA256；
- 原图数量、视图数量、数组形状和缓存 SHA256；
- `test_predictions_used=false`；
- `test_labels_used=false`。

### 校准参数进入路由的条件

校准报告只有同时满足以下三项才可加载：

1. `status=passed`；
2. `test_predictions_used=false`；
3. `test_labels_used=false`。

融合输出必须记录校准文件的绝对路径、SHA256 和类别。真实参考参数完成前，
合成参数运行只算接口测试，不能报告为算法性能。
