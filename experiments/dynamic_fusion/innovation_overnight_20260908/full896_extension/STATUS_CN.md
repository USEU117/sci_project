# Full 896 扩类探针运行归档

## 运行状态

- 状态：`complete`；执行会话 ID：`61848`（进程已正常退出）。
- 命令：见 [`COMMAND.txt`](COMMAND.txt)。单 GPU `cuda:0`、batch 1、PyTorch threads 2；复用本地冻结 DINOv2 ViT-B/14 checkpoint，未下载新模型。
- 启动时间：2026-09-08 03:13:06（北京）；墙钟时间：`334.437 s`；硬截止按 `2026-09-07T23:00:00+00:00` 检查。
- 仅新增 4 类：每类 2 query、2 个 synthetic family、每族 3 seed，共 48 个 unique mask；每个 episode 有 full44832 与 full896 两行，共 192 行。另取每类每 query 5 个预定 photometric nuisance key。
- query 图均严格排除出同类 memory；48 个 synthetic mask 全部通过 v14 原图 mask 校验；无 OOM、无分辨率临时调整。

## 新四类同 episode 结果

`Δ` 定义为同一 episode 的 `full896 AP - full44832 AP`。每类每族 `n = 6`（2 query × 3 seed）。

| 类别 | 族 | full44832 AP | full896 AP | Δ |
|---|---|---:|---:|---:|
| bracket_brown | thin_scratch | 0.161637 | 0.297815 | +0.136177 |
| bracket_brown | cutpaste | 0.839498 | 0.831908 | -0.007590 |
| bracket_white | thin_scratch | 0.165968 | 0.256604 | +0.090636 |
| bracket_white | cutpaste | 0.625350 | 0.602492 | -0.022858 |
| connector | thin_scratch | 0.195535 | 0.431633 | +0.236097 |
| connector | cutpaste | 0.701889 | 0.693576 | -0.008313 |
| tubes | thin_scratch | 0.168101 | 0.348919 | +0.180818 |
| tubes | cutpaste | 0.670133 | 0.715110 | +0.044976 |

四类合并的 full44832 AP 为 `0.441014`，full896 AP 为 `0.522257`，Δ 为 `+0.081243`。六类合并（复用既有两类 full896 及其旧 tiling full44832 控制，不重跑旧类）为：full44832 `0.463093`、full896 `0.535253`、Δ `+0.072160`。完整逐类数据见 [`SUMMARY.json`](SUMMARY.json)。

## 正常分布、时间与显存

- 正常项不使用旧 LOO p99 或 FP 阈值，仅报 score p95：full44832 clean / nuisance 为 `0.568524 / 0.581891`；full896 为 `0.533767 / 0.550305`。
- warmup（不计入正式 episode）：448 edge `432.786 ms`，896 edge `394.503 ms`。正式平均编码：full44832 `76.846 ms`，full896 `368.520 ms`；正式平均 score：`80.935 / 494.582 ms`。
- 正式运行结束：torch allocated `355.132 MB`，torch reserved `631.243 MB`，torch peak allocated `503.212 MB`；整设备 free `4699.718 MB`，total `6441.927 MB`。
- full44832 为 1024 tokens，full896 为 4096 tokens；attention-work proxy 比为 `16 : 1`，不能称等算力。

## 缓存与边界

四类双分辨率 clean / synthetic cache 位于 [`outputs/dynamic_fusion/overnight_20260908_full896_extension/features/`](../../../outputs/dynamic_fusion/overnight_20260908_full896_extension/features/)。每类包含 clean `(2, 32, 32, 768)` / `(2, 64, 64, 768)`、synthetic `(2, 6, 32, 32, 768)` / `(2, 6, 64, 64, 768)` 与 `(2, 6, 1024, 1024)` mask，valid 全部为 1。

这是 4 类、k2、support-synthetic 短探针。正向差值只能作为输入分辨率 / 全图上下文候选信号；不能把旧两类 cutpaste 相对 448 的轻微下降写成改善，也不能宣称论文创新已证实。

