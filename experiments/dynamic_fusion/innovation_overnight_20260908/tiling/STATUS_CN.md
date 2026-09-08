# Tiling probe 状态

首轮固定命令已完成：

```powershell
.venv-anomalyclip\Scripts\python.exe scripts\innovation_overnight_20260908\probe_tiling.py --device cuda:0 --torch-threads 2
```

- 进程 PID：`9932`（已正常退出）。
- 状态：`complete`；352 行（2 类 × 2 query support × 22 episode × 4 arm）。
- 墙钟时间：`208.730 s`；DINO model load `8605.817 ms`。
- mask renderer 与既有 v14 cache：96 个 synthetic arm 行均逐字节一致；所有 352 行 `strict_leave_image_memory=true`，未发现 query 与 bank 同图。

support synthetic 平均 Pixel-AP（thin_scratch + cutpaste 等权 episode 平均）：

| arm | thin_scratch | cutpaste | 两族平均 |
|---|---:|---:|---:|
| full_dino32 | 0.2139 | 0.8006 | 0.5073 |
| full_map_interp | 0.1990 | 0.7692 | 0.4841 |
| full_feat_interp64 | 0.2350 | 0.8172 | 0.5261 |
| tile_reencode64 | 0.3430 | 0.6130 | 0.4780 |

tile 重编码对 thin_scratch 有候选正向差值（相对 full_dino32 `+0.1292`），但 cutpaste 为 `-0.1876`，两族合计为 `-0.0292`；tile clean normal FP `0.7412`、nuisance FP `0.7942`，高于 full_dino32 的 `0.3377`、`0.4087`。因此本轮决策为 `NO_POSITIVE_TILING_SIGNAL_IN_THIS_PROBE`，只保留“细划痕候选信号”诊断，不能作论文创新已证实。

详细逐 episode 指标、random-area AP、mask 外正常 FP、LOO p99 阈值及实际提取/评分时间：`outputs/dynamic_fusion/overnight_20260908_tiling/RESULTS.json` 和 `PER_EPISODE.jsonl`。到时 partial 会保存在同目录 `PARTIAL_RESULTS.json`。

