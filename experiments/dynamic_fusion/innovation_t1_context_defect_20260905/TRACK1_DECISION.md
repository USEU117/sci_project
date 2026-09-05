# Track-1 结案（Probe-C1：冻结特征中是否含可用的关系信息）— 归档

日期：2026-09-05　立项：`docs/.../30_NEW_OBJECTIVE_LOGICAL_CONTEXT_DEFECTS_PROJECT_CN_20260905.md`（doc30 §4）
脚本：`scripts/innovation_t1_context_defect_20260905/probe_c1_context_variants.py`
数据：复用 v14 support-only 合成缓存（dino/clip k2/k4，A1 融合 → 16 网格 mean-pool）；无 /test/、无真实缺陷。
设计：每 (cat, shot) × 留出图 h：memory=(K−1) 图 clean cells；在留出图 h 的留族 episode 上按描述子 `s=1−max_cos` 算 Pixel-AP。描述子：C0=A1 cell；C1=concat(cell, 3×3 邻域均值)；C2=concat(cell, 上/下/左/右邻)。

## 结果（6 类宏；thin_scratch 不可评）
| shot | 指标 | C0 | C1 | C2 |
|---|---|---|---|---|
| k2 | cutpaste AP | 0.765 | **0.779** | 0.645 |
| k2 | erasure AP | 0.963 | **0.979** | 0.906 |
| k4 | cutpaste AP | 0.829 | **0.843** | 0.686 |
| k4 | erasure AP | 0.963 | **0.969** | 0.935 |
| 宏 | nuisance-AUC(clean vs 15 photometric) | 0.542 | 0.550 | 0.540 |

## 门判定（预注册，按结果未调）
- **G-C1（关系信息存在：最优关系变体 cutpaste Δ ≥ +0.05）失败**：C1 在两 shot 均仅 **+0.014**（k2/k4 同值），远低于门。
- G-C2（erasure 不损失 ≥−0.01）通过：C1 为 +0.016/+0.006。
- G-C3（normal 路径稳定，尺度无关 AUC ≤ +0.05 且 ≤0.60）通过：C0/C1/C2 ≈0.54–0.55。

**决策：`TRACK1_PROBE_FAIL_ARCHIVE`**（doc30 §4：任一 shot 未达 +0.05 即归档 Track-1 → 依序启动 Track-2）。

## 诚实解读
1. 局部一阶邻域描述子（C1）在两 shot 给出**一致但微小**的正增益（+0.014，且不伤结构缺陷、不增 nuisance 敏感度），只回收了 cutpaste 上下文缺口的一小部分；方向性 4 邻 concat（C2）因高维拼接稀释 cosine 匹配而明显变差（−0.08~−0.14）。
2. 结论：冻结的 16 网格融合特征里存在少量可利用的局部关系信息，但远不足以支撑一个"逻辑/上下文缺陷检测"主线；若未来重启 Track-1，需要更高分辨率（32 网格）、跨尺度关系或显式部件级结构先验，而非一阶邻域特征——这已超出"轻量关系描述子"范围，按立项纪律不在此轮无限扩展。
3. 真实 MPDD 的 parts_mismatch 子组门未触碰（正确：未过机制门不进真实门）。

## 产物
`PROBE_C1_RESULTS.json`（320 行明细）、本结案记录。Track-1 关闭，转 Track-2（多层/中段互补性测量，doc31 立项）。
