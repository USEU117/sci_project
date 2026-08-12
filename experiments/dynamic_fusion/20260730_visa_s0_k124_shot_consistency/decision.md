# WP1.4 decision

状态：通过。

VisA seed 0 的 1/2/4-shot 缓存均完成十二类审计。每个 shot 都覆盖相同的 2,162 张测试图；AnomalyDINO 与 WinCLIP+ 的 sample_id、顺序和图像标签一致。跨 shot 比较也确认十二类测试 sample_id 和顺序完全不变。

这说明后续可以按同一个 sample_id 直接比较 1/2/4-shot 的校准敏感度和路由变化，不会把测试样本变化误当成算法变化。

仍未完成的 WP1 内容是 AnomalyCLIP 旧缓存的 sample_id sidecar。该缓存没有 sample_id，不能仅凭当前数组顺序直接进入融合。
