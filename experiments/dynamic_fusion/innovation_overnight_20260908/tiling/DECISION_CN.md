# 2 × 2 原图裁块重编码探针决策

状态：`complete`；决策：`NO_POSITIVE_TILING_SIGNAL_IN_THIS_PROBE`。

该 probe 只使用 MPDD manifest 的 seed 0、k2 support 原图，在每个 query support 图上留出该图并只用另一张图建 memory。thin_scratch、cutpaste 和光度 nuisance 的 renderer seed 固定，未读取 test 图或 test 标签。

评价使用相同的 1024 × 1024 原图 mask。`random_area_ap` 是 mask 正像素比例，作为大样本/随机排序参考；有限样本 AP 是非线性的，不把它称作有限样本 AP 的精确期望。正常 FP 阈值是留出 memory 图的 LOO p99，合成缺陷的 FP 只统计 mask 外正常背景。

`tile_reencode64` 是四个 512 × 512 crop 的冻结 DINO 独立前向并拼接 64 × 64；`full_feat_interp64` 仅作特征插值控制，`full_map_interp` 仅作距离图插值控制。

实际墙钟时间：`208.730 s`。详细每 episode 记录见 `outputs/dynamic_fusion/overnight_20260908_tiling/RESULTS.json` 与 `PER_EPISODE.jsonl`。

边界：本轮只有 2 类、k2、support 合成 episode；即使出现正向差值，也只能作为候选信号，不能写成论文创新已证实。
