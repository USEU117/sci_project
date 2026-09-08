# Shift-memory 探针协议（2026-09-08）

## 研究问题

部署时正常参考图与查询图可能存在光度偏移。固定 memory 行数之外，加入来自其他 normal support 图的光度变体，是否能让 heldout nuisance 的异常分数更接近 heldout clean 分布，同时不损 synthetic AP？本轮只做 support-only 诊断，不访问 MPDD test。

## 输入与数据角色

- 复用 `outputs/dynamic_fusion/v14_p1_support/v14_p1_support_{dino,clip}_s0_k{shot}/{category}.npz`；默认 seed0/k2，可用 `--shot 4` 做同协议扩展。
- 类别固定为 MPDD 六类：`bracket_black`、`bracket_brown`、`bracket_white`、`connector`、`metal_plate`、`tubes`。
- 复用 `scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py` 的 cell loader 组件和 A1 16×16 cell / 0.5–0.5 concat / L2 规范；旧脚本不修改。
- 每类每折留出一张 support 图 `h`。memory 只使用其余 `K-1` 张 support 图的 clean 与预定 nuisance feature；query 的同图 clean、synthetic、nuisance 永不入 memory。
- synthetic 使用 V14 原始 1024×1024 mask；16×16 anomaly score 以固定 64×64 nearest repeat 回到 1024×1024 后计算 AP，保留细划痕原始 mask，不用 16-grid 阈值把细划痕删掉。
- nuisance 使用 V14 `nui_keys` 元数据中的 5 族 × 3 强度。部署预算冻结为每族第 0 强度：索引 `[0,3,6,9,12]`；不用结果挑强度。

## 固定候选

1. `clean_only`：其他 support 图的 256 clean cells。
2. `augmented_all5`：每个 clean memory 图追加 5 个预定族的第 0 强度，共 `1+5=6` 份行；不读取 query 族，不做路由。
3. `duplicate_clean_equal_all5`：将 clean rows 重复 6 份，行数与 `augmented_all5` 相同，控制 memory 变大本身。
4. `augmented_loo_family`：对每个 heldout nuisance 族，memory 排除该族的 nuisance rows，只保留 clean 与其他 4 族；仅用于 leave-family-out 诊断，不代表部署路由。
5. `duplicate_clean_equal_loo`：clean rows 重复 5 份，行数与 `augmented_loo_family` 相同。

每个候选的 memory 行数、特征维度、float32 bytes、构建时间和所有 query 的检索 wall time 均记录。候选不训练参数、不改变 A1 权重、不删除 clean 行、不用 query label 选择实际部署候选。

## 指标与预注册判据

- synthetic：cutpaste、local_erasure、thin_scratch 的原始 mask AP；逐 episode、逐族和宏平均，报告相对 `clean_only` 的 ΔAP。
- nuisance 分布：heldout clean 与 15 个 heldout nuisance 变体的 pooled AUC（越接近 0.5 越好）、nuisance mean/p95 相对 clean 的绝对偏移、nuisance 超过 heldout clean p99 的比例；按族逐项记录。
- equal-budget 归因：`augmented_all5` 对 `duplicate_clean_equal_all5`，以及 `augmented_loo_family` 对 `duplicate_clean_equal_loo`；若只优于 clean-only 而不优于同预算 control，不归因于光度内容。
- `G-SYN`（筛选门）：实用候选 `augmented_all5` 的 synthetic macro ΔAP ≥ −0.01，且最差有效 episode ΔAP ≥ −0.05。
- `G-SHIFT`（筛选门）：`augmented_all5` 的 pooled nuisance mean absolute shift 和 p95 absolute shift 都不高于 `clean_only`，并且至少一项严格降低；同预算 duplicate control 上也必须不差。该门只表示本轮候选信号，不表示真实泛化。
- `G-LOO`（诊断）：`augmented_loo_family` 只报告 leave-family-out 结果；不因其较好就把 query 族当作可用部署标签。

## 资源与硬截止

- CPU only；`CUDA_VISIBLE_DEVICES=""`，OMP/MKL/OpenBLAS/torch threads 固定 2；一次处理一类。
- 硬截止显式为 `datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc)`（北京时间 2026-09-08 07:00）。脚本在模型/类别/留出折/episode边界检查；到点先写 `PARTIAL_RESULTS.json`、`RESULTS.json`、`DECISION_CN.md`、`RUN_MANIFEST.json` 再退出。
- 预计首轮不超过 30 分钟；超时状态为 `partial_deadline`，不伪装成机制失败或完成。

## 复现命令

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
python scripts/innovation_overnight_20260908/probe_shift_memory.py --shot 2 --cats bracket_black,bracket_brown,bracket_white,connector,metal_plate,tubes
```
