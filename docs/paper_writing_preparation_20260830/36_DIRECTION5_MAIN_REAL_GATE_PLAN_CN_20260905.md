# 方向 5 主线立项：真实 MPDD 关系描述子门（doc35 探针通过后的落地验证）

日期：2026-09-05　上游：doc35 Probe-H1 **PASS**（32-grid 邻域/4 邻关系描述子回收 cutpaste Δ=+0.152（k2）/ +0.148（k4），erasure 不伤、normal 稳定）；A1 冻结基线同 doc33/doc34（concat pca0/whiten0/w0.5，真实 MPDD seed0_k2 mean Pixel-AP=0.3437）。
性质：**真实门验证（只读评估）**——把合成代理上巨大增益的 32-grid 关系描述子放到真实 MPDD test 上，验证其对真实缺陷（含 parts_mismatch 上下文类）像素排序的提升，作为可写进论文的关系一致性表述的机制证据。

## 1. 验证对象与口径
- **A1 基线**：冻结 concat（同 doc33），`s = L2dist(test_patch, ref_bank)` → 448 map → Pixel-AP（stride 8）。
- **关系候选**（在真实 32-grid fused patch 上，逐图计算图内描述子，memory 与查询同规则）：
  - **R5-C1** = concat(z, 3×3 邻域均值 z̄)，3072-D；
  - **R5-C2** = concat(z, 上/下/左/右 4 邻 z)，5×1536-D。
- **评估**：对每 test 图独立打分得到 448 map（描述子行 L2 后 `1−max_cos` 语义经 faiss L2 距离还原），A1 口径 Pixel-AP/AUROC；六类总体宏 + **parts_mismatch 子组**（仅取 sample_ids 含 `/parts_mismatch/` 的图集合，预先声明机制子组，见 doc28）。
- **范围（预注册 first-shot）**：seed0，k2 与 k4 两 shot 都跑（关系增益需两 shot 一致才可信）。
- 数据：A1 冻结特征缓存只读；无拟合、无 /test/good 进 memory、无真实缺陷调参。

## 2. 门（预注册，按结果不调；两 shot 均需满足主门）
- **G-R1（关系增益，主门）**：两 shot（k2 且 k4）六类宏 Pixel-AP：最优关系变体 − A1 ≥ **+0.01**；
- **G-R2（无灾难类）**：同一变体，两 shot 每类 ΔPixel-AP ≥ −0.03；
- **G-R3（真实上下文子组，预先声明）**：同一变体，两 shot parts_mismatch 图集合宏 Pixel-AP − A1 ≥ **+0.00**（不得损伤真实上下文类缺陷；子组门不替代总体门）；
- **G-R4（整体不损）**：同一变体，两 shot 宏 ΔAUROC ≥ −0.005。
- 通过 → 方向 5 主线成立：真实 MPDD 上 32-grid 关系描述子给 A1 带来真实像素增益（含 parts_mismatch 子组不损），可进入论文（关系一致性 + 分辨率消融）；失败 → 归档（合成增益未向真实泛化），转方向 6。

## 3. 产物
- `scripts/innovation_t5_relation32_20260905/run_d5_real_gate.py`
- `experiments/dynamic_fusion/innovation_t5_relation32_20260905/REAL_D5_s0_k{shot}.json`、`D5_REAL_DECISION.md`

## 4. 执行提示
> 跑 run_d5_real_gate：每 cat 加载 A1 冻结缓存 → 32-grid fused concat → 对 ref/test 逐图构造 C0/C1/C2 描述子 → faiss memory KNN → 448 map → A1 口径 Pixel-AP/AUROC（总体 + parts_mismatch 子组按 sample_ids 过滤）；输出逐类/宏/子组 Δ；按 G-R1..R4 判定。
