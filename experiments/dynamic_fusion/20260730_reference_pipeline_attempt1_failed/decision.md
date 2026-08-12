# Failed normal-reference pipeline attempt

合成参考缓存已成功生成，审计脚本在组装报告时因为 Python 布尔值误写成小写 `false` 而终止。缓存内容尚未被判定失败，问题属于报告代码错误。

处理：保留本次失败记录，把 `false` 修正为 `False`，使用新的报告文件名重新执行审计和拟合。
