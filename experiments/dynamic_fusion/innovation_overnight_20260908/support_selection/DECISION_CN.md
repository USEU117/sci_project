# Support 图级 facility coverage 选择：首轮结果

状态：`complete`；决定：`NO_ROBUST_FACILITY_SIGNAL_IN_THIS_PROBE`。

本轮只使用 v14 k4、seed 0 的 normal support cache。每折留出一张 support 图，选择器只读取其余三张 clean 图的全部 16 × 16 fused cells；held-out feature 和 mask 只在选择完成后进入评价。

facility 的单位是整张图，目标是候选池全部 patch 到被选图 patch bank 的平均 `1 − cosine` 距离。T4 C2 的 farthest-point 是 patch 行级 coreset，本轮不删除选中图内的 patch。

AP 是 16 × 16 score grid 双线性回到原始 1024 × 1024 mask 的 Pixel-AP。thin_scratch 若有原像素正像素则保留；没有正像素只记 invalid，不因 grid 可见性人为填分。normal nuisance AUC 使用固定 4-stride cell lattice，只作稳定性诊断。

完成 folds：`24`；墙钟时间：`388.453 s`。逐折选择路径、预算和 per-family/per-seed 结果见 `RESULTS.json`。

总体摘要（原像素 mask AP；每张图 256 cells）：

- facility k 1：macro `0.5561207992045852`；相对等预算组合均值 `Δ = -0.004195893155697017`；normal nuisance AUC `0.5429958767361112`；coverage 选择耗时均值 / p 95 `37.41992083318261 / 51.59710999978415 ms`。
- facility k 2：macro `0.5904414658950066`；相对等预算组合均值 `Δ = 0.006264171746302338`；normal nuisance AUC `0.5468261718749999`；coverage 选择耗时均值 / p 95 `37.41992083318261 / 51.59710999978415 ms`。
- 对照 macro：uniform k 1 `0.5603166923602823`、uniform k 2 `0.5841772941487042`、all 3 `0.5980278261528845`。

逐类别 facility 选择使用 `selected_source_indices`（原始四张 support 的 index；不是 held-out index）：

| category | k 1 facility − uniform | k 2 facility − uniform | k 1 选图次数 | k 2 选图次数 |
|---|---:|---:|---|---|
| bracket_black | `0.008589502500092133` | `0.009974476667116566` | 0 × 1, 1 × 1, 3 × 2 | 03 × 1, 12 × 1, 23 × 2 |
| bracket_brown | `0.0016604189139750591` | `-0.0005009461793733516` | 1 × 2, 2 × 2 | 02 × 1, 03 × 2, 13 × 1 |
| bracket_white | `-0.03795599896630761` | `0.03017750398668101` | 1 × 2, 3 × 2 | 01 × 1, 02 × 2, 23 × 1 |
| connector | `0.005288651836544311` | `-0.002743127833671588` | 0 × 2, 2 × 2 | 03 × 1, 12 × 1, 13 × 2 |
| metal_plate | `-0.0027688026900692386` | `0.0024016572421985405` | 0 × 1, 1 × 1, 2 × 1, 3 × 1 | 02 × 2, 13 × 1, 23 × 1 |
| tubes | `1.0869471583796475e-05` | `-0.0017245334051372851` | 0 × 3, 3 × 1 | 03 × 1, 12 × 1, 23 × 2 |

边界：即使 facility 在本轮 proxy 中优于对照，也只能作为候选信号；不能写成论文创新已证实，也不能外推到真实 MPDD test。
