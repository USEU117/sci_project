# Shift-memory 探针结论

状态：`complete`；决策：`SHIFT_MEMORY_SIGNAL_SUPPORT_ONLY`。

本轮只使用 V14 support-only cache，seed0，未读取 MPDD test；每折 memory 只来自非 heldout support 图。部署候选 `augmented_all5` 不读取 query nuisance 族；`augmented_loo_family` 仅是留族交叉诊断。

关键结果（相对 clean-only）：

| 候选 | synthetic AP Δ | 最差 episode ΔAP | nuisance mean abs shift Δ | nuisance p95 abs shift Δ | equal-budget p95 Δ | memory 行数 |
|---|---:|---:|---:|---:|---:|---:|
| clean_only | 0.000000 | 0.000000 | NA | NA | NA | 256 |
| augmented_all5 | 0.001253 | -0.043294 | -0.002178 | -0.004343 | -0.004343 | 1536 |
| duplicate_clean_equal_all5 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | NA | 1536 |
| augmented_loo_family | 0.001253 | -0.043294 | -0.001527 | -0.003206 | -0.003206 | 1280 |
| duplicate_clean_equal_loo | 0.000000 | 0.000000 | -0.000000 | 0.000000 | NA | 1280 |

解释：原始 1024 × 1024 mask 用固定 16 × 16 score 的 64 × 64 nearest repeat 评价，因此 thin_scratch 不因 16-grid 阈值为空而被删掉。equal-budget duplicate-clean 只控制 memory 行数增加，不增加光度内容。

复现命令：

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_shift_memory.py --shot 2 --cats bracket_black,bracket_brown,bracket_white,connector,metal_plate,tubes
```

详细逐 episode 记录：`PER_EPISODE.jsonl`；运行中断时保留 `PARTIAL_RESULTS.json`。
