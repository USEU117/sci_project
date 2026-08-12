# WP1.2 decision

状态：通过。

VisA seed 0、1-shot 的 AnomalyDINO 与 WinCLIP+ 冻结预测共 12 类、2,162 张测试图。十二类的规范化 sample_id 集合、顺序和图像标签全部一致，没有缺失项、重复项或错配。

两个分支的异常图空间尺寸不同：AnomalyDINO 保留各类别的原始长宽，WinCLIP+ 为 240×240。因此融合前必须把文本分支异常图按样本缩放到视觉分支尺寸。异常分数图使用双线性插值；真值掩码不进入路由器。

本结果只证明缓存可以安全对齐，不证明动态融合性能。分数校准尚未完成，不能据此比较两分支的不确定性。

下一步：

1. 审计 seed 0 的 2-shot 和 4-shot 缓存；
2. 验证三种 shot 的测试 sample_id 完全一致；
3. 为 AnomalyCLIP 旧缓存建立可复核 sample_id sidecar。
