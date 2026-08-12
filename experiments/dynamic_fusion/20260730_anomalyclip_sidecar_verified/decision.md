# WP1.3 decision

状态：通过。

AnomalyCLIP 的 12 个旧缓存没有 `sample_ids`，本次按照原始 `meta.json` 的测试列表顺序生成 sidecar，并根据旧版 `dataset.py` 的 `shuffle=False` 证据复核顺序。所有类别的样本数、图像标签和 518×518 掩码均通过。随后与 AnomalyDINO seed 0、1-shot 缓存完成 12 类配对审计。

sidecar 只保存 sample_id，不保存测试真值；真值只在离线路径审计中使用。后续融合脚本通过显式 `--text-sidecar` 参数读取它。
