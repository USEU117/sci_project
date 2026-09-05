# 方向 5 探针结案（Probe-H1：32-grid 关系描述子）— 通过

日期：2026-09-05　立项：`docs/.../35_DIRECTION5_RELATION32_PLAN_CN_20260905.md`（doc35）
脚本：`scripts/innovation_t5_relation32_20260905/probe_h1_relation32.py`
数据：复用 v14 support-only 合成缓存（dino/clip k2/k4，32-grid patch）；无 /test/、无真实缺陷、零拟合。
设计：每 (cat, shot) × 留出图 h：memory=(K−1) 图 clean 32-grid patches；Pixel-AP（mask 下采样到 32）按留出族；描述子 = A1 fused patch `z`（1536-D）及其邻域/跨尺度拼接：
- C0 = z（基线）；C1 = concat(z, 3×3 邻域均值)；C2 = concat(z, 上/下/左/右邻)；C3 = concat(z, 16-grid 3×3 邻域均值上采样)（跨尺度）。

## 结果（6 类宏）
| shot | 指标 | C0 | C1(3×3) | C2(4 邻) | C3(跨尺度) |
|---|---|---|---|---|---|
| k2 | cutpaste AP | 0.632 | **0.784** | 0.783 | 0.659 |
| k2 | erasure AP | 0.951 | 0.965 | 0.963 | 0.934 |
| k4 | cutpaste AP | 0.664 | 0.801 | **0.811** | 0.681 |
| k4 | erasure AP | 0.953 | 0.974 | 0.975 | 0.947 |
| 宏 | nuisance-AUC | 0.536 | 0.541 | 0.532 | 0.543 |

## 门判定（预注册 doc35，按结果未调）
- **G-H1（cutpaste 最优 − C0 ≥ +0.05）通过**：k2 C1 **+0.152**；k4 C2 **+0.148**（均远超门）。
- **G-H2（erasure 不损失 ≥ −0.01）通过**：k2 +0.014、k4 +0.022（不伤且略升）。
- **G-H3（normal 稳定：AUC ≤0.60 且 −C0 ≤ +0.05）通过**：C1 0.541、C2 0.532、C3 0.543。

**决策：`D5_PROBE_PASS`**（doc35：通过 → 立项方向 5 主线：真实 MPDD parts_mismatch 子组 + 六类总体门）。

## 诚实解读
1. **分辨率是关键变量**：同一种"3×3/4 邻域关系描述子"在 16-grid 只回收 cutpaste +0.014（Track-1 归档），在真实 A1 的 32-grid patch 分辨率回收 **+0.15**（10× 差距）。16-grid 的 2×2 mean-pool 把可用于匹配的邻域关系信息抹平了。
2. 跨尺度（C3）弱于同尺度邻域（C1/C2），说明信息在同尺度细邻域而非粗尺度背景；4 邻方向性（C2）在 k4 最优。
3. erasure 不伤反升：邻域拼接对结构缺陷也无损；normal 路径 AUC 变化 ≤ +0.007（C2 甚至更低），无光度敏感度抬升。
4. 该增益为冻结特征内已存在的关系信息（零训练），可直接叠加在 A1 描述子维度；真实 MPDD 门（含 parts_mismatch 子组）是下一步验收，不按此结果调参。

## 产物
`PROBE_H1_RESULTS.json`、本结案记录。方向 5 探针关闭 → 转方向 5 主线（真实门，doc36）。
