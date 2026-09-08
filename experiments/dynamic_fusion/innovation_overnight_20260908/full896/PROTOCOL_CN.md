# Full 896 原生 64 × 64 图像域探针协议

## 固定范围

- 类别：`bracket_black`、`metal_plate`。
- support：MPDD manifest seed 0、k2；每个 query support 图只用同类另一张 support 原图建 memory。
- synthetic：与既有 tiling probe 完全相同的 24 个 unique masks（2 类 × 2 query support × `thin_scratch` / `cutpaste` × seed 0、1、2），mask 由既有 v14 renderer 和 `_variant_seed` 重建并逐字节校验。
- nuisance：同一 15 项固定光度变换；只在线处理，不保存 nuisance feature cache。
- 不读取 test 图或 test 标签，不拟合参数，不覆盖既有 tiling 结果。

## 输入和控制

使用现有缓存的 DINOv2 ViT-B/14 权重，`eval()` / inference mode，完整 1024 × 1024 图像以 `smaller_edge_size=896` 编码，原生得到 64 × 64 × 768 `full896_native64` feature grid。

结果按 episode 键读取既有 `overnight_20260908_tiling/RESULTS.json` 的 `full_dino32` 和 `tile_reencode64` AP 作为控制；控制结果只读，不覆盖、不重算旧值。三者使用同一 query、bank、renderer seed 和 1024 × 1024 原图 mask。

`full896_native64` 的 clean / synthetic feature（2 类各 2 个 support 图、24 个 synthetic mask 对应 episode）保存为输出目录 `features/<category>.npz`，供后续同输入跨尺度或 context + detail 短 probe 复用；nuisance feature 不缓存。

## 指标

- synthetic：原图 1024 × 1024 mask 上的 Pixel-AP；同时保留 `random_area_ap` 的 mask 正像素比例，作为大样本 / 随机排序 prevalence 参考，不称作有限样本 AP 的精确期望。
- 记录 `full896 - full44832` 和 `full896 - tile64` 的每类、每族 episode AP 差值。
- 不复用旧单图 LOO p99 阈值，不输出 calibrated FP。clean、nuisance 以及 synthetic mask 外只保存 score map 的 mean / p50 / p95 / p99 / max / std，作为正常分数分布诊断。
- 记录每次 full 896 编码的 wall time、模型加载时间、CUDA peak allocated / reserved VRAM、token 数和 attention-work proxy。full 896 的全图 token attention 与 4 crop 的总 token 数相同但二次 attention 成本不同（约 4 倍 proxy），不称等算力。

## 截止、资源和 OOM 退路

- batch 1，torch threads 2，GPU 仅当前 probe 使用，目标 RTX 3060 6 GB。
- 硬截止：`datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()`，并设本轮最长 45 分钟；每次提取和 episode 循环前后检查，partial 结果逐 episode 落盘。
- 若 896 full image CUDA OOM：保持 896 分辨率并写 `partial_oom`、峰值 VRAM 和明确错误；安全退路是另行以同一 896 edge、batch 1、CPU 复现，禁止把分辨率临时调低或从 OOM 结果挑选配置。

复现命令：

```powershell
.venv-anomalyclip\Scripts\python.exe scripts\innovation_overnight_20260908\probe_full896.py --device cuda:0 --torch-threads 2
```

