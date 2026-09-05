# N4 / PMC：coreset 保真与成本（doc27 §8）决策

日期：2026-09-04（夜间轮）。实现：`scripts/innovation_v13_overnight_20260904/run_n4_pmc.py`（CPU）。
方法：平衡条件（行列）下对 support fused bank（k2:2048 单元 / k4:4096 单元）做 25%/50% 预算的
coreset，冻结 A1 协议 KNN 评估（MPDD 六类真实掩码 Pixel-AP@56）。选择器：random×3、
concat_greedy（farthest-first）、tri_greedy（dino/clip/concat 三覆盖 minimax）、branch_merge
（分支各半贪心并集）；full 为 A1 旁路。预注册常数见脚本头部。

## 1. REDUNDANT 检查（doc27 §8「先分析是否等价 concat 普通 greedy」）

- tri_greedy 与 concat_greedy 选出的单元 Jaccard：25% 预算均值 0.53（0.35–0.61），
  50% 均值 0.72（0.60–0.78）→ **集合并不近似相同**（数学上不等价，三覆盖会改变选取）。
- 但两者 AP 几乎一致（50% 宏：concat_greedy 0.3633 vs tri_greedy 0.3634）→ **AP 等价**。
  结论：分支私有覆盖保护对 AP 无增量，普通 concat coreset 已足够；按 doc27 §8，只记录
  **工程优化**，不构成升格研究线。

## 2. R1 工程门（50% 预算）结果

宏（12 cat-shot）：
- full 0.3660；concat_greedy50 0.3633（loss 0.0027）；tri_greedy50 0.3634（loss 0.0026）；
  random50 0.3553（loss 0.0107）；branch_merge50 0.3616。
- 宏损失 ≤0.003 ✓（concat 与 tri 均达标）。
- 最差类：**connector k2 concat_greedy loss 0.0142 / tri 0.0107**（>0.01 上限 ✗）；
  其余 11/12 类 loss ≤0.0027–0.0098。→ 最差类门不满足。
- 匹配阶段加速：bank 减半 → 每 query 的距离计算减半（≥25% 匹配加速，bank 规模代理）✓；
  双编码器推理不在匹配阶段，不称端到端加速。
- 25% 预算：concat_greedy 宏 0.3659 ≈ full，明显优于 random（0.3426，+0.023）→ greedy
  coreset 在低预算下显著保值。

## 3. 判定

- R1 工程门：宏损失达标、最差类超标（connector k2）→ **未整体通过**；PMC 不升格为效率
  候选，记录为**工程优化观察**：concat-farthest coreset 在 50% 预算保宏 AP（loss 0.0027），
  25% 预算显著优于随机抽样（+0.023 宏）；瓶颈是 connector 类在低 shot 下的少数关键单元。
- 分支私有覆盖保护（tri）无 AP 收益 → 不支持"双分支私有覆盖"作为 PMC 研究线理由。
- 产物：N4_pmc/R0.json（含 per-cat×shot 标签与 jaccard）、本决策。
