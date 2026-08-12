# 结论

正常参考预测的统一格式、逐类别审计和校准参数拟合线路已经打通。12 个类别均通过，图像分数和像素图分别拟合参数。

本次输入是合成预测，只用于验证代码和记录流程，不代表真实方法效果。此前两次脚本错误仍分别保存在 `20260730_reference_pipeline_attempt1_failed` 和 `20260730_reference_pipeline_attempt2_failed`。

下一步是在 GPU 空闲且不影响基线训练时，导出 AnomalyDINO 与 AnomalyCLIP 的真实正常参考预测，再运行同样的审计和拟合流程。
