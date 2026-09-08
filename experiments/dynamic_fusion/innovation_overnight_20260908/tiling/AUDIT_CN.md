# Tiling 结果独立质量复核

复核对象是 `outputs/dynamic_fusion/overnight_20260908_tiling/RESULTS.json`、`PER_EPISODE.jsonl`、本 probe 脚本，以及既有 v14 DINO support cache。没有重跑 GPU，没有改变原始数值，也没有新增机制。

## 1. k2 单图 memory 的 LOO p99

对每个类别和 query support 图 `h`，脚本令 `bank = rels[1-h]`，所以 memory 只有同类的另一张 support 原图。每个 arm 的 bank cell 数分别为 full 32 × 32 的 1024 和高分辨率 64 × 64 的 4096。

`_loo_dist` 先对 bank cell 做 L2 归一化，计算 bank 与自身的相似度；当前 cell 的对角项被置为 `-inf`，再取除自身以外的最近 cell 距离。因此这里没有 exact self-memory match。随后把这张 bank 的 LOO 距离网格按该 arm 的 map 函数映射到 1024 × 1024，阈值取其 p99。代码位置是 `probe_tiling.py:191-202` 和 `252-270`。

这仍然不是跨图的正常分布：对角线排除后，1024 或 4096 个 cell 都来自同一张图，空间相邻或外观相近的同图 patch 仍可互为最近邻；k2 还没有足够的 image-level normal variation。用既有 full-DINO cache 做的独立诊断显示，加入 Chebyshev 半径 1 的同图邻域排除后，LOO map p99 会明显升高。该半径 1 数值只是审计对照，没有替换原结果：

|类别|bank 原图|probe 使用的 full_dino32 p99|再排除半径 1 的 p99|相对变化|
|---|---|---:|---:|---:|
|bracket_black|281.png|0.5384|0.6265|+16.4%|
|bracket_black|271.png|0.4902|0.5830|+18.9%|
|metal_plate|012.png|0.4803|0.6461|+34.5%|
|metal_plate|029.png|0.5025|0.6816|+35.6%|

原 probe 的 `normal_fp_rate` 因此只能作同一 arm、同一类别和同一 query 口径下的诊断。它不应被当作跨方法的 calibrated FPR 或用于方法排名：各 arm 的阈值由各自单图 bank 的 LOO 距离和 map 平滑方式得到，且没有估计图像间正常波动。尤其 tile 的 clean FP `0.7412`、nuisance FP `0.7942` 说明其跨图/高分辨率匹配存在明显误报，但不能单凭这个 p99 口径宣称比其他方法差。若后续需要比较 FPR，应先预注册一个含多张 support normal 图的共同 calibration memory，仍将 query 图完全留出，并对所有 arm 使用同一 calibration 规则；本轮不回填该修正。

独立用 v14 full-DINO cache 重算 24 个 full_dino32 synthetic AP，和结果文件最大绝对差约 `2.1e-5`，说明 full arm 的 score、mask 对齐和 AP 记录一致。

## 2. random_area_ap 的准确含义

代码中的 `random_area_ap = mask.mean()` 是正像素比例。它是随机排序在大样本下的参考值，也可作为类别 prevalence baseline；有限样本 average precision 含非线性分母以及排序/并列效应，因此不能把该字段称作有限样本 AP 的精确期望。原始 AP、mask、FP 和 runtime 数值全部保留，只修正了协议、决策和脚本文案。

## 3. mask 完整性与 v14 对应关系

选定的两个 synthetic family 各使用 2 类 × 2 张 query support × 3 个固定 seed：每个类别、每个 family 有 6 个 unique 1024 × 1024 masks。每个 mask 在 4 个 arm 上各评价一次，因而结果文件共有 96 个 mask-arm rows，而 unique mask SHA256 为 24 个且每个出现 4 次。96 行的 `renderer_cache_mask_verified` 均为 `true`。

|类别|family|unique masks|对应 v14 `syn_masks` entries|mask-arm rows（4 arms）|
|---|---|---:|---:|---:|
|bracket_black|thin_scratch|6|6|24|
|bracket_black|cutpaste|6|6|24|
|metal_plate|thin_scratch|6|6|24|
|metal_plate|cutpaste|6|6|24|
|合计|选定两族|24|24|96|

既有 v14 cache 在这两个类别中还包含未纳入本轮的 `local_erasure`：全三族共 36 个 unique entries；本 probe 明确只取其中 24 个，不把遗漏的 12 个写成已测结果。

## 4. 类别 × family 的 tile − full 差值

这里 `full` 指 `full_dino32`，数值为 6 个 mask 的 episode 平均；每个格子样本数都是 6，避免两类宏平均掩盖方向相反的结果。

|类别|family|full_dino32 AP|tile_reencode64 AP|tile − full|
|---|---|---:|---:|---:|
|bracket_black|thin_scratch|0.1932|0.3279|+0.1347|
|bracket_black|cutpaste|0.7803|0.5226|−0.2577|
|metal_plate|thin_scratch|0.2346|0.3582|+0.1236|
|metal_plate|cutpaste|0.8209|0.7035|−0.1174|

按类别把两族等权后，`bracket_black` 为 `−0.0615`，`metal_plate` 为 `+0.0031`；两类合计为 `−0.0292`。所以 tile 对 thin_scratch 的增益在两类都出现，但 cutpaste 在两类都下降，宏观负结论并非由单一类别改善被隐藏造成。

## 5. 最有价值的下一图像域探针（只建议，不执行）

建议下一步先做 **full-image native 64 × 64 DINO**：仍用现有 DINOv2 ViT-B/14 权重和冻结 backbone，把完整 1024 × 1024 图像按既有预处理的 `smaller_edge_size=896` 编码为原生 64 × 64，再与当前 `tile_reencode64` 使用完全相同的 2 类、k2、24 个 mask、留一图 memory 和 AP 口径。这样直接检验 tile 的正向 thin-scratch 信号究竟来自 token density，还是来自局部裁块的输入上下文；不需要训练或下载新模型。显存应先以单类、batch 1 smoke 检查，保持首轮短。

- `full_native64 ≈ tile_reencode64`：主要是输入 token density / 分辨率效应，2 × 2 裁块方案本身缺少额外证据。
- `full_native64 > tile_reencode64`：非重叠裁块的上下文丢失或边界效应是主要代价。
- `tile_reencode64 > full_native64`：局部裁块重编码可能保留独立的候选信号，才值得再设计有上下文的重叠切块。

该下一 probe 的正常 FP 仍应标为描述性诊断，直到改用多图共同 calibration memory；本建议不启动长训或自动扩展矩阵。

