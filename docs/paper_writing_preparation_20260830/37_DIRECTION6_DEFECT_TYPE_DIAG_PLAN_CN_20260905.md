# 方向 6 立项：A1 真实 parts_mismatch / defect-type 剩余缺口纯诊断

日期：2026-09-05　上游：doc36（方向 5 主线真实门）**D5_REAL_GATE_FAIL** → [D5_REAL_DECISION.md](../../../experiments/dynamic_fusion/innovation_t5_relation32_20260905/D5_REAL_DECISION.md) 明确「转方向 6：真实 parts_mismatch 子组诊断，**不再加关系机制**」。
性质：**只读纯诊断**——无候选、无拟合、无新机制；只对 A1 冻结 concat 在真实 MPDD test 上的像素级结果按**真实 defect-type**（test 子目录）做细粒度切分，量化剩余缺口的结构。

## 1. 背景（为什么做纯诊断）
- A1 冻结 concat（pca0/whiten0/w0.5，32-grid，faiss k1 → 448 map，Pixel-AP stride 8）是当前唯一冻结方法（MPDD s0/k2 mean Pixel-AP=0.3437，k4=0.3883）。
- doc36 真实门暴露：A1 在 **parts_mismatch 上下文类子组**绝对水平明显低于总体（k4 pm 宏 0.278 vs 总体 0.388；k2 pm 宏 0.246 vs 总体 0.344），但方向 5 的"加 3×3/4 邻关系描述子"不能回收该缺口（真实全负）。
- 结论缺口未量化到 **defect-type 粒度**：哪些真实类型最弱、上下文族（parts_mismatch / bend_and_parts_mismatch）相对表面类缺陷落后多少、是"检测不出"还是"定位质量差"，尚无只读证据。方向 6 补这一层量化，作为论文机制/局限表述的依据；**不立项任何新机制**。

## 2. 诊断对象与口径
- **评估对象**：A1 冻结 concat（与 doc36 C0 完全同一管道）在真实 MPDD seed0 test 上的逐图 448 map。
- **数据（全部只读、无 /test/good 进 memory）**：
  - compact concat patch maps：`submission_repro_20260827/predictions_compact/maps/mpdd/s0_k{2,4}/{cat}.npz`（`concat_patch_map` float16 [n,32,32]，与冻结报告 replay ΔAP ≤ 5e-5 验证过）；
  - masks / sample_ids：`outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k{2,4}/anomalydino_visual/{cat}.npz`（`imgs_masks`、`sample_ids`；与 compact 缓存同一 test 序）。
- **范围（预注册）**：seed0，shot k2 与 k4（与 doc36 一致）。
- **分组键**：从 sample_id（`…/{cat}/test/{defect_type}/{stem}.png`）取 `{defect_type}` 目录名；`good` 为正常图。

## 3. 分析项
- **P0（口径复现）**：每类全图 Pixel-AP/AUROC/AUPRO 与 REAL_D5 C0 / 冻结报告对账（k2 mean 0.343706、k4 mean 0.388328）。
- **P1（defect-type 切分表）**：每 (cat, type) 行：n、Pixel-AP、Pixel-AUROC、Pixel-AUPRO（同 doc36 子组口径：type 图集合内部计算，与总体公式一致）。
- **P2（类型族汇总）**：按 defect type 跨类宏（等类权，A1 宏口径）；单列**上下文族** = `parts_mismatch` + `bend_and_parts_mismatch`（仅 bracket_brown 含后者），与全部表面缺陷型对比；parts_mismatch 跨类宏与 REAL_D5 pm_mean（k2 0.2464 / k4 0.2781）复现对账。
- **P3（检测 vs 定位）**：每 (cat, type) 图像级 image-AUROC/AP（map max 聚合，type 图为正、同 cat `good` 为负），判断弱类型是"检出难"还是"定位差"。
- **P4（诚实读解）**：把剩余缺口结构写成量化结论 + 论文表述建议（上下文类缺陷 = A1 已知边界 / future work），并指明超出当前数据/范围才可启动新机制的边界。

## 4. 决策（无 pass/fail 门）
纯诊断按结果输出 `D6_REPORT_s0_k{2,4}.json` + 汇总 + `D6_DECISION.md`：给出量化的剩余缺口结构、上下文族 vs 表面类的差距、以及是否值得（在授权新数据/新范围下）另立项的边界判断；本方向自身**不产生新方法**。

## 5. 产物
- `scripts/innovation_t6_defect_diag_20260905/run_d6_defect_type_diag.py`
- `experiments/dynamic_fusion/innovation_t6_defect_diag_20260905/{D6_REPORT_s0_k2.json, D6_REPORT_s0_k4.json, D6_DECISION.md}`
