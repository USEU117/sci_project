# Precision 探针结论

决策：`PRECISION_PROBE_PASS`

范围：support-only，完成类别 bracket_black、bracket_brown、bracket_white、connector、metal_plate、tubes，seed0/k2、每类轮换留出 support 图；未读取 MPDD test。
候选：FP16 roundtrip、FP16 runtime、对称 per-channel INT8 roundtrip、INT8 runtime；所有候选均保留全部 256 行 memory。

可支持的效率结论：
- FP16 roundtrip 达到预注册质量门并提供约 0.5 倍持久化 memory
- 对称 per-channel INT8 解码检索达到预注册质量门并提供低于 0.3 倍持久化 memory

关键聚合（相对 control_f32）：

| 候选 | AP Δ | 最差 episode AP Δ | 距离 p99 最差类 | NN 身份一致率 | 存储比 | 常驻比 | 检索时间比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control_f32 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| fp16_roundtrip | 0.000000 | 0.000000 | 0.000016 | 0.999925 | 0.500000 | 1.000000 | 1.037133 |
| fp16_runtime | -0.000332 | -0.023810 | 0.000307 | 0.995665 | 0.500000 | 0.500000 | 124.921159 |
| int8_pc_roundtrip | -0.000331 | -0.023810 | 0.000309 | 0.998375 | 0.253906 | 1.003906 | 0.967650 |
| int8_pc_runtime | 0.000688 | -0.023810 | 0.002206 | 0.990612 | 0.253906 | 0.253906 | 2.334746 |

解释：roundtrip 候选的常驻 bank 先解码到 float32，因此只支持存储压缩；INT8 runtime 的常驻 bank 保留 int8 和 scale，分块临时 workspace 为固定 256 channels。实际 CPU wall time 以 perf_counter 记录，不能将存储比直接解释为延迟比。

复现命令：

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_precision.py --cats bracket_black,bracket_brown,bracket_white,connector,metal_plate,tubes
```
