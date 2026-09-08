# Resolution-storage 组合探针协议（2026-09-08）

## 研究问题

full896 的 64 × 64 memory patch 数是 full448 的 32 × 32 的 4 倍。固定已确认的对称 per-channel INT8 memory-only roundtrip 配方，测试它能否把 full896 的持久化 bank bytes 压到接近 full448 FP32，同时保留高分辨率 AP。该探针只回答 support-only 的 AP / 存储关系，不把它写成整体推理成本相等、编码延迟改善或新的算法。

## 输入与缓存

- seed 0、k2、六类 MPDD support；每类两张图互相留出，query 图不进入 memory。
- `full896` 旧缓存：`outputs/dynamic_fusion/overnight_20260908_full896/features/` 提供 `bracket_black`、`metal_plate` 的 clean/synthetic 64 × 64 DINO features。
- `full896_extension` 双分辨率缓存：`outputs/dynamic_fusion/overnight_20260908_full896_extension/features/` 提供 `bracket_brown`、`bracket_white`、`connector`、`tubes` 的 clean/synthetic 32 × 32 与 64 × 64 DINO features。
- 旧两类的 full448 控制使用已存在的 `outputs/dynamic_fusion/v14_p1_support/v14_p1_support_dino_s0_k2/{category}.npz` 32 × 32 clean/synthetic support features；脚本会按 family/seed 对齐并与 full896 mask、`ref_rel` 逐字节/逐项校验。这是缓存补齐，不重新编码图像。
- 三个 arm 使用同一个已缓存的 query、heldout mask 与另一张 support 图 memory；full448、full896 都使用 DINO 768 维 cell。

## 固定 episode 与 arm

- synthetic 只取 `thin_scratch`、`cutpaste` 两族各 seed 0。因此 unique synthetic query episode = `6 类 × 2 张 heldout 图 × 2 族 = 24`；三臂共享这些 episode。
- 本轮只重算 clean/synthetic cache 中已冻结的 query；不为本组合探针在线重编码 nuisance。固定 family/seed 口径只指 synthetic 的两族各 seed 0。
- `full448_f32`：每折另一张 clean 图的 1,024 rows × 768 维，float32。
- `full896_f32`：每折另一张 clean 图的 4,096 rows × 768 维，float32。
- `full896_int8_roundtrip`：只对 full896 memory 做已确认的对称 per-channel INT8，`scale[c] = max(abs(memory[:, c])) / 127`，round-to-nearest、clip `[-127,127]`；检索前解码为 float32 并重新 L2 normalize。query 始终 float32，不运行 runtime INT8 matmul。

## map、指标与内存口径

- 三臂均先计算 native grid 的 nearest cosine distance，再复用 `src.utils.dists2map` 投影到同一原始 `1024 × 1024` map；synthetic AP 使用原始 renderer mask，保留 thin_scratch。
- 每类、每个 synthetic family 记录三臂 AP、`full896_f32 - full448_f32`、`full896_int8_roundtrip - full448_f32`，以及每次对 f32 full896 的 INT8 NN identity、distance error。
- 记录 native memory rows、持久化 bank bytes（INT8 含 768 × float32 scale = 3,072 bytes）、storage ratio；同时记录 decoded resident bytes 与 decode 时 stored + scale + decoded 的 bank-array 峰值。roundtrip 解码后 resident bytes 与 full896 FP32 相同，不能声称运行内存降低。
- 编码时间不在本探针重测；features 来自既有缓存，输出明确标记 `encoding_latency_measured=false`。可记录检索 wall time，但不把它外推成整体推理等成本。
- 只把 full896 FP32 相对 full448 FP32 的 AP 增益作为高分辨率参考；INT8 是否保留该增益只作本轮 support-only 诊断，不调参、不改论文。

## 资源与硬截止

- CPU only，`CUDA_VISIBLE_DEVICES=""`，OMP/MKL/OpenBLAS/torch threads 固定 2；一次处理一类；不运行旧 runtime 性能坏分支。
- 06:30（北京时间，即 2026-09-07 22:30 UTC）后不开始新的 category 重算；硬截止 07:00（2026-09-07 23:00 UTC），脚本在 category、heldout、arm、episode 边界检查，并保存已有 partial。
- 到截止或异常时保存 `PARTIAL_RESULTS.json`、`RESULTS.json`、`DECISION_CN.md`、`RUN_MANIFEST.json`、`PER_EPISODE.jsonl`；不访问 `/test/` 图像、mask 或 label。

## 复现命令

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_resolution_storage.py
```
