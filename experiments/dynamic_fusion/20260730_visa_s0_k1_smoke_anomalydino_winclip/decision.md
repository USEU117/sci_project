# Pre-calibration smoke decision

接口、样本对齐、像素图缩放、统一 NPZ 输出和统一评测均正常。200 张 candle 图片全部被路由为 `text`。

这不是融合性能结论。AnomalyDINO 和 WinCLIP+ 的原始分数不在同一尺度，当前路由器用原始值计算熵，因此不能公平比较不确定性。WP2 必须先完成图像分数和像素图校准。
