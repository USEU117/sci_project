# Failed normal-reference pipeline attempt 2

视觉和文本两组 12 类参考缓存审计均已通过。校准参数已经在内存中计算，但拟合脚本组装最终 JSON 时再次出现小写 `false` 的 Python 语法错误，因此没有生成可信的最终参数文件。

处理：保留两份通过的审计报告和本次失败记录，修正拟合脚本后使用 `calibration_v3.json` 重新执行。
