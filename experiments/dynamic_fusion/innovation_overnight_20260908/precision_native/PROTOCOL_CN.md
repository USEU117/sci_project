# Precision native32 探针协议（2026-09-08）

## 研究问题

上一轮 precision 只在 16 × 16 pooled grid 上验证 memory-only FP16 roundtrip 与对称 per-channel INT8 roundtrip。本轮固定相同 A1 融合和量化定义，切换到 v14 k4 的原生 32 × 32 cell grid，确认更细离散网格下的 AP、最近邻和距离误差是否仍稳定。该探针只作 support-only 复核，不声称真实 MPDD 泛化或新的算法增益。

## 数据与留出

- 使用 `outputs/dynamic_fusion/v14_p1_support/v14_p1_support_{dino,clip}_s0_k4/{category}.npz`，seed 0，固定六类 MPDD support。
- DINO 保留原生 32 × 32；CLIP 从原生 37 × 37 仅重采样到 32 × 32 以便 A1 对齐；不做 16-grid pooling。
- 每类 K = 4；每折 heldout 1 张 support 图，memory 只取其余 3 张 clean 图的 3 × 1,024 cells。heldout clean、synthetic、nuisance 都不进入 memory。
- synthetic 使用 9 个 episode（cutpaste、local_erasure、thin_scratch，各 3 个 seed）。原始 1,024 × 1,024 renderer mask 保留；32 × 32 score 用固定 32 × 32 nearest repeat 回到原图像素计算 AP。
- normal 查询只使用 heldout clean 与五个固定第 0 强度 nuisance（`exposure_0`、`gamma_0`、`white_balance_0`、`lr_brightness_gradient_0`、`specular_blob_0`），只计算 score error，不重复全部 15 个 nuisance。

## 固定候选

1. `control_f32`：clean memory 保持 float32。
2. `fp16_roundtrip`：memory cast 为 float16 存储，检索前 cast 回 float32 并重新 L2 normalize。
3. `int8_pc_roundtrip`：仅 memory 做对称 per-channel INT8，`scale[c] = max(abs(memory[:, c])) / 127`，round-to-nearest 并 clip 到 `[-127, 127]`；检索前解码回 float32 并重新 L2 normalize。

query 始终保持 float32；不运行 FP16/INT8 runtime matmul，不使用 query 标签或 nuisance 族做路由，不删除 clean memory 单元。每个候选记录实际 bank 行数、存储 bytes、scale bytes、storage ratio，以及解码后 resident bytes。

## 记录与判据

- 每类、每个 synthetic family 记录 candidate/control AP、ΔAP、最差 episode ΔAP、NN identity agreement、距离误差（mean/p95/p99/max）和 bank storage ratio。
- normal 查询记录 control 对比的 score distance error；不把 normal score error 当作 AP。
- 复用前一轮 precision 的保守质量门：FP16 `Δmacro AP ≥ −0.005`、最差 episode `≥ −0.02`、距离 p99 `≤ 0.002`、NN `≥ 0.995`；INT8 对应 `−0.02`、`−0.05`、`0.02`、`0.95`。storage 门分别为 ≤ 0.51 和 < 0.30。门只用于 support-only 归档，不把通过写成真实泛化确认。
- 小正信号只原样报告 per-category/per-family ΔAP；不按结果调参数、不选择赢家、不改论文。

## 资源与硬截止

- CPU only，`CUDA_VISIBLE_DEVICES=""`，OMP/MKL/OpenBLAS/torch threads 固定为 2；一次处理一类。
- 硬截止为 `datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)`，即北京时间 2026-09-08 07:00。脚本在类别、heldout fold、候选和 episode 边界检查；到点或异常会保存 `PARTIAL_RESULTS.json`、`RESULTS.json`、`DECISION_CN.md`、`RUN_MANIFEST.json` 和已有 `PER_EPISODE.jsonl`。
- 首轮预算 45 分钟；本轮不读取 `/test/` 图像、mask 或 label。

## 复现命令

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_precision_native.py
```
