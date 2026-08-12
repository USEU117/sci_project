# Pre-calibration sidecar smoke decision

AnomalyCLIP 旧缓存通过 sidecar 接入后，接口、样本对齐、518→视觉图尺寸缩放、统一 NPZ 输出和统一评测均正常。200 张 candle 图片全部被路由为 `text`。

这同样不是融合性能结论。结果再次说明校准必须先于不确定性路由。
