# Precision native32 探针结论

状态：`complete`；决策：`PRECISION_NATIVE_PROBE_PASS`。

范围：v14 support-only k4，原生 DINO 32 × 32 网格；每折 heldout 1 张、memory 使用其余 3 张 clean 图。未读取 MPDD test、未运行 FP16/INT8 runtime。
FP16 与 INT8 均只量化 memory 并 roundtrip 解码为 float32；query 保持 float32。每族 nuisance 只取固定第 0 强度，作为 normal score-error 查询。

每类汇总的 family AP Δ、NN 一致率、距离 p99 与 bank storage 比例见 `RESULTS.json`；六类宏平均如下：

| 候选 | macro AP Δ | 最差 episode ΔAP | NN 一致率 | 距离 p99 最差类 | normal score error p99 最差类 | bank storage 比 | scale bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| control_f32 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.0 |
| fp16_roundtrip | -0.000000 | -0.000093 | 0.999859 | 0.000019 | 0.000017 | 0.500000 | 0.0 |
| int8_pc_roundtrip | -0.000129 | -0.023129 | 0.995581 | 0.000606 | 0.000590 | 0.250326 | 6144.0 |

原始 mask AP 使用固定 32 × 32 score 的 32 × 32 nearest repeat 回到 1024 × 1024；thin_scratch 不因 mask 下采样被删掉。`family_metrics` 在每个 category 和全局 aggregate 中给出三族 AP 差、NN 一致率、距离误差与 bank bytes 比例。

小正信号（仅描述原始 ΔAP，不作为筛选门）：

- `fp16_roundtrip` 宏平均 ΔAP = `-0.000000`；family ΔAP：cutpaste -0.000001，local_erasure 0.000000，thin_scratch 0.000000。
- `int8_pc_roundtrip` 宏平均 ΔAP = `-0.000129`；family ΔAP：cutpaste -0.000013，local_erasure -0.000330，thin_scratch -0.000042。

复现命令：

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_precision_native.py
```

`PARTIAL_RESULTS.json` 和 `PER_EPISODE.jsonl` 保留截止或异常前已完成证据。
