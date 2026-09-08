# Full 896 探针运行归档

## 运行状态

- 状态：`complete`；执行会话 ID：`20834`（进程已正常退出）。
- 命令：见 [`COMMAND.txt`](COMMAND.txt)。使用 `cuda:0`、batch 1、PyTorch threads 2；未下载新模型，复用现有冻结 DINOv2 ViT-B/14 权重。
- 启动时间：2026-09-08 02:14:20（北京）；墙钟时间：177.342 s；硬截止仍按 `2026-09-07T23:00:00+00:00` 检查。
- 结果：88 行，即 2 类 × 2 query support ×（1 clean + 6 synthetic + 15 nuisance）。24 个 synthetic mask 行全部通过 v14 原图 mask 校验；query 图均从 memory 排除，未发生同图 self-memory。
- 无 OOM、无降分辨率；full896 原生特征缓存已保存，后续可复用 clean / synthetic 输入。

## 结果

full896 的 synthetic mean AP 为 `0.561246`，相对已有 `full_dino32` 为 `+0.053995`，相对已有 `tile_reencode64` 为 `+0.083205`。逐类逐族均值如下；`Δ full` 和 `Δ tile` 分别是相对两个只读控制的 AP 差值。

| 类别 | 族 | n | full896 AP | Δ full | Δ tile |
|---|---|---:|---:|---:|---:|
| bracket_black | thin_scratch | 6 | 0.313783 | +0.120583 | -0.014153 |
| bracket_black | cutpaste | 6 | 0.759135 | -0.021168 | +0.236551 |
| metal_plate | thin_scratch | 6 | 0.363602 | +0.129019 | +0.005445 |
| metal_plate | cutpaste | 6 | 0.808464 | -0.012456 | +0.104976 |

正常与 nuisance 项只记录分数分布，不沿用旧的单图 LOO p99 阈值或 FP 结论：full896 clean score p95 均值为 `0.611117`，nuisance score p95 均值为 `0.651408`。本轮 `random_area_ap` 是 mask 正像素 prevalence 的大样本 / 随机排序参考，不是有限样本 AP 的精确期望。

## 资源与缓存

- full896 产生 4096 tokens，attention-work proxy 为 `16,777,216`；4 crop tile64 合计 4096 tokens，但 proxy 为 `4,194,304`。full-image proxy 约为 tile 的 4 倍，因此不能称等算力。
- 编码时间均值为 `380.056 ms`（clean 缓存复用会出现 0 ms；synthetic / nuisance 单次约 0.38–0.39 s），score 时间均值为 `495.258 ms`。
- 运行结束显存：allocated `355.132 MB`，reserved `616.563 MB`，peak allocated `503.212 MB`，peak reserved `616.563 MB`；GPU 总显存记录为 `6441.927 MB`。
- 缓存：[`features/bracket_black.npz`](../../../outputs/dynamic_fusion/overnight_20260908_full896/features/bracket_black.npz) 和 [`features/metal_plate.npz`](../../../outputs/dynamic_fusion/overnight_20260908_full896/features/metal_plate.npz)，各含 clean `(2, 64, 64, 768)`、synthetic `(2, 6, 64, 64, 768)` 与原图 mask `(2, 6, 1024, 1024)`；float32 特征未压缩理论大小各 `176,160,768` bytes。

## 决策边界

本轮是 2 类、k2、support-synthetic 短探针。full896 整体优于 tile64，主要来自 cutpaste；thin_scratch 相对 full_dino32 在两类均上升，但 `bracket_black` 相对 tile64 略低。结果只能作为输入分辨率 / 全图上下文的候选信号，不能写成论文创新已证实。

