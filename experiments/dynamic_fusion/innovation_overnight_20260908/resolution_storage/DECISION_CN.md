# Resolution-storage 组合探针结论

状态：`complete`；决策：`RESOLUTION_STORAGE_SUPPORT_ONLY_COMPLETE`。

本轮是 support-only 诊断：六类 k2、每个 heldout 图各取 thin_scratch 与 cutpaste 的 seed 0，实际 unique synthetic episodes 为 24；三臂在同一 episode 上配对。未读取 test 图像/标签。

| arm | macro AP | ΔAP vs full448 FP32 | ΔAP vs full896 FP32 | native memory rows | persistent bank bytes | scale bytes | decoded resident bytes | decode peak bank-array bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full448_f32 | 0.48246142 | 0.00000000 | -0.07869321 | 1024.00000000 | 3145728.00000000 | 0.00000000 | 3145728.00000000 | 3145728.00000000 |
| full896_f32 | 0.56115463 | 0.07869321 | 0.00000000 | 4096.00000000 | 12582912.00000000 | 0.00000000 | 12582912.00000000 | 12582912.00000000 |
| full896_int8_roundtrip | 0.56123490 | 0.07877347 | 0.00008027 | 4096.00000000 | 3148800.00000000 | 3072.00000000 | 12582912.00000000 | 15731712.00000000 |

full896 FP32 相对 full448 FP32 的配对宏平均 AP 增益：`0.07869321`。
full896 INT8 roundtrip 相对 full448 FP32：`0.07877347`；相对 full896 FP32：`0.00008027`。
INT8 持久化 bank bytes / full448 FP32：`1.00097656`；/ full896 FP32：`0.25024414`。INT8 scale 已计入持久化 bytes。
INT8 相对 full896 FP32 的 NN identity：`0.99404907`；distance error p99（episode 均值/最差）：`0.00084723` / `0.00090742`。

full896 INT8 只在持久化表示上压缩；解码后的 resident bank bytes 与 full896 FP32 相同，decode 峰值还包含 quantized bank、scale 和 decoded bank，因此不能据此声称运行内存降低。编码延迟未测量（缓存复用），也不能外推端到端推理速度、总内存或等成本，更不能把本探针称为新颖算法。

每类/每族 AP、NN 一致率、距离误差和 memory 字段见 `RESULTS.json` 与 `PER_EPISODE.jsonl`。

复现命令：

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
$env:NUMEXPR_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_resolution_storage.py
```

截止或异常时仍保留 `PARTIAL_RESULTS.json`、`PER_EPISODE.jsonl` 和最终状态字段。
