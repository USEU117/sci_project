# Precision 探针协议（2026-09-08）

## 目的

在不删除任何 normal memory 单元的前提下，测量 A1 memory 的 FP16 roundtrip、对称 per-channel INT8 量化，以及 INT8 常驻内存近似检索。结果只作为 support-only 小规模效率探针，不替换冻结 A1，也不触碰 MPDD test。

## 输入与范围

- 数据：`outputs/dynamic_fusion/v14_p1_support/` 的 `v14_p1_support_{dino,clip}_s0_k2/{category}.npz`。
- 类别：`bracket_black`、`bracket_brown`、`bracket_white`、`connector`、`metal_plate`、`tubes`。
- shot/seed：固定 `k2`、`seed0`；每类轮换留出每一张 support 图 `h`，memory 为其余正常 support 图。
- 特征：复用 `scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py::load_cells`，16×16 cell、A1 `0.5/0.5` concat 和整体 L2 normalize；不改旧脚本。
- 评价：支持图派生的 cutpaste、local_erasure 合成掩码；所有 256 query cells 参与打分。`thin_scratch` 若 mask 无正 cell，只记录为无效 episode，不人为填分。normal nuisance 只作排序稳定性诊断。
- 数据角色：量化 scale 仅从当前留出折的 normal memory 行估计；合成 mask 只用于 support proxy 指标；没有 `/test/` 标签或图像进入拟合、量化或选择。

## 冻结候选

1. `control_f32`：A1 全精度 float32 memory，作为唯一 control。
2. `fp16_roundtrip`：memory cast 为 float16 作为存储形式，运行前 cast 回 float32 并重新 L2 normalize 后精确矩阵检索。该项只声称存储压缩，不声称运行内存下降。
3. `int8_pc_roundtrip`：memory 做对称 per-channel INT8，`scale[c] = max(abs(memory[:,c]))/127`，下限 `1e-12`；round-to-nearest、clip 到 `[-127,127]`。运行前解码到 float32、重新 L2 normalize，再做精确矩阵检索。scale 仅占额外 `float32[D]`。
4. `int8_pc_runtime`：沿用同一 INT8 memory 和 scale，query 用逐行对称动态 scale `max(abs(query))/127` 量化；用分块的 INT8×INT8 乘积和 per-channel scale 累加近似 cosine，不解码完整 memory bank。候选不删除行，分块大小固定为 256 channels。
5. `fp16_runtime`：memory/query 均 cast 为 float16，直接以 float16 矩阵乘法取最大相似度，用于区分 FP16 常驻内存与 roundtrip 解码；不作为存储-only 候选的替代。

所有候选均保持与 control 相同的 memory 行数、query 数和 k=1 最近邻规则。禁止按类别、缺陷类型或结果调节 scale、位宽、分块、阈值或 top-k。

## 记录项

每个类别、留出图和候选记录：合成 AP、相对 control 的 AP、normal nuisance AUC、距离绝对误差 mean/p95/p99/max、异常分数 Spearman 排名相关、top-10 重叠率、最近邻身份一致率、memory 行数、存储 bytes、常驻 bank bytes、量化/解码和检索实际 wall time。聚合同时给出 macro 和最差类别/episode。

存储 bytes 按持久化 memory bank 计算：float32 为 `rows×D×4`，FP16 为 `rows×D×2`，INT8 为 `rows×D×1 + D×4`。常驻 bank bytes 单独记录，FP16/INT8 runtime 不把 query 临时 workspace 计入持久 bank，但记录分块大小。

## 预注册门限

- `G-P1 rows`：所有候选每折 memory 行数必须等于 control（不允许 coreset 删除）。
- `G-P2 fp16`：`macro AP Δ >= -0.005`、最差 episode `ΔAP >= -0.02`、距离 p99 `<= 0.002`、最近邻身份一致率 `>= 0.995`。
- `G-P3 int8`：`macro AP Δ >= -0.02`、最差 episode `ΔAP >= -0.05`、距离 p99 `<= 0.02`、最近邻身份一致率 `>= 0.95`。
- `G-P4 storage`：FP16 持久化 bytes 比率必须 `<= 0.51`，INT8（含 scale）必须 `< 0.30`；只在对应候选通过质量门时报告“存储压缩可用”。
- `G-P5 runtime`：只有常驻 bank bytes 比率 `< 0.50` 且实际检索 wall time 比 control 更低时，才报告“运行内存与时间同时受益”；否则只报告测得的常驻内存或负时间结果，不外推。

## 复现

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_precision.py --cats bracket_black,bracket_brown,bracket_white,connector,metal_plate,tubes
```

执行无论通过或失败都应在本目录写入 `RESULTS.json`、`DECISION_CN.md` 和 `RUN_MANIFEST.json`。
