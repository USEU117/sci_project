# 方向 5 立项：高分辨率（32-grid）关系描述子（Track-1 重启，doc30 §4 明确建议路径）

日期：2026-09-05　上游：Track-4 C2 经用户裁决**接受为通过**（farthest-point coreset，见 doc34 §6 与 TRACK4_DECISION.md）→ 依 doc34 §7 队列启动方向 5；doc30 §4 Track-1 归档时明确写「若重启 Track-1，需要**更高分辨率（32 网格）**、跨尺度关系或显式部件级结构先验」。
性质：**合成探针先行**（同 Track-1 Probe-C1 结构，但分辨率升到 32-grid）——先回答「真实 A1 使用的 32-grid patch 分辨率下，局部邻域/跨尺度关系描述子是否回收上下文型缺陷（cutpaste）的可观缺口」；通过才谈真实门，失败即归档转方向 6。

## 1. 为什么升分辨率（假设）
- Track-1 在 16-grid（2×2 mean-pool cell）测 3×3 邻域描述子，只回收 cutpaste Δ=+0.014（门 +0.05 未过），结论指向「pool 后邻域信息已被平均抹平」。
- A1 真实协议在 **32-grid patch** 打分（448 map，stride 8）；32-grid 下邻域半径小一倍、保留更多空间细节 → 关系描述子可能比 16-grid 更有信息。
- 问题预注册：**在 32-grid patch 上拼接邻域/跨尺度关系项到 A1 描述子，能否把留出 cutpaste Pixel-AP 提高 ≥ +0.05 而不伤 erasure 与 normal 路径？**

## 2. 数据、描述子与红线
- 数据：复用 v14 support-only 合成缓存（dino/clip k2/k4，32-grid；无 /test/、无真实缺陷、不训练）。
- 基线网格改为 **32**（不再 pool 到 16）；描述子统一行 L2：
  - **C0** = A1 fused patch `z`（1536-D，32-grid）：`s=1−max_cos(patch, memory)`；
  - **C1** = concat(`z`, 3×3 邻域均值 `z̄`)，3072-D（32-grid 邻域）；
  - **C2** = concat(`z`, 上下左右 4 邻 `z`)，5×1536-D（方向性邻域，32-grid）；
  - **C3（跨尺度）** = concat(`z`, 16-grid 邻域均值上采样对齐)，3072-D——刻画"该 patch 相对粗粒度局部背景"的偏移（Track-1 结论第二条）。
- 结构：每 (cat, shot∈{2,4}) × 留图 h：memory=(K−1) 图 clean 32-grid patches；Pixel-AP 按 mask-in-patch（mask 下采样到 32）；normal 路径 = scale-free AUC(clean h vs 其 15 光度变体)。
- 红线：同 doc30 §4（fit/select 只 support；不读 /test/；按结果不调；失败即归档）。

## 3. 门（预注册，按结果不调）
- **G-H1**：cutpaste 留出宏 AP：最优关系变体 − C0 ≥ **+0.05**（k2 或 k4 任一 shot 成立即可视为信息存在）；
- **G-H2**：同变体 erasure 宏 AP − C0 ≥ **−0.01**（不伤结构缺陷）；
- **G-H3**：normal 路径：最优关系变体 nuisance-AUC − C0 ≤ +0.05 且 ≤ 0.60。
- 通过 → 立项方向 5 主线（真实 MPDD parts_mismatch 子组 + 六类门）；失败 → 归档转方向 6（parts_mismatch 纯诊断测量，零关系假设）。

## 6. 主线执行结果（doc36，2026-09-05）
- 探针 PASS → 立项 doc36 真实门 → **D5_REAL_GATE_FAIL**：32-grid 关系描述子在真实 MPDD 两 shot 全负（C1 宏 ΔAP −0.014/−0.034；parts_mismatch 子组 −0.012/−0.025）→ 合成增益未向真实泛化（与 Track-3 coreset 方向相反的同类教训）。详见 `experiments/dynamic_fusion/innovation_t5_relation32_20260905/D5_REAL_DECISION.md`。顺带量化了 A1 在 parts_mismatch 子组的绝对缺口（k4=0.278 < 总体 0.388），供方向 6 使用。

## 4. 产物
- `scripts/innovation_t5_relation32_20260905/probe_h1_relation32.py`
- `experiments/dynamic_fusion/innovation_t5_relation32_20260905/`（PROBE_H1_RESULTS.json、TRACK5_DECISION.md）

## 5. 执行提示
> 写 probe_h1_relation32：读 v14 support 缓存；32-grid A1 fused patch；C0–C3 描述子（3×3 邻域 / 4 方向邻域 / 16-grid 邻域均值上采样）；LOO memory KNN Pixel-AP（cutpaste 主测、erasure 对照）；scale-free nuisance-AUC；按 G-H1/H2/H3 判定。只评价不拟合。
