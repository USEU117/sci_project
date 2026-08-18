# 阶段七：冻结后验证（2026-08-17）

RunId: `phase7_post_freeze_20260817` · 冻结配置：A1 concat + KNN，vitb14 DINO + CLIP，w=0.5。
参考：`docs/DYNAMIC_FUSION_NEXT_STEPS.md` 阶段七。

## 1. MPDD 冻结复核（development，仅复核不调参）

- 冻结配置在 MPDD 9 配置（K=1/2/4 × seed 0/1/2）上已完整执行并审计：
  - mean ΔAP vs DINO = **+0.0486**，**9/9 全正**（K1 +0.037 / K2 +0.049 / K4 +0.059）。
  - `scripts/audit_a1_mpdd_matrix.py` → `all_9_configs_pass=true`（schema/对齐/NaN-Inf 全过）。
  - 权重扫描 + 动态对照确认 w=0.5 冻结；freeze_manifest `--verify` 通过。
- 复核结论：**冻结配置无修改、无调参**，MPDD 复核通过。
- 证据：`experiments/dynamic_fusion/v3_direction_a/a1_matrix_20260817/`、
  `experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_manifest.json`。

## 2. BTAD 冻结后验证（holdout 角色）

- BTAD 无 A1 特征缓存的 GPU 补导（K=2/4 缺 vitb14 dino 特征；K=1 缓存存在）。
- **K=1 冻结版验证**（复用既有 vitb14 + CLIP 特征缓存，配置与冻结完全一致）：

| seed | fused Pixel AP | DINO baseline | ΔAP |
|---|---|---|---|
| 0 | 0.6203 | 0.5646 | **+0.0557** |
| 1 | 0.6333 | 0.5594 | **+0.0738** |
| 2 | 0.6585 | 0.5701 | **+0.0883** |

- 3/3 seed 正，mean ΔAP = **+0.0726**（高于 MPDD 的 +0.0486，跨数据集稳健）。
- 复现性抽查：seed0 的 01/02 类 fused AP（0.5741/0.6079）与历史报告**完全一致**。
- 证据：`experiments/dynamic_fusion/v3_direction_a/a1_vitb14_btad_fusion/seed{0,1,2}/`。
- 备注：BTAD 03 类（441 张图，32×42 grid）单进程峰值内存约 3.4 GiB 会触发分配失败，
  故本次未重跑该类的本地推理（历史报告已含），避免内存风险。

## 3. VisA / MVTec（验证角色报告，明确差异）

- **VisA / MVTec 均无 A1 冻结版特征缓存**（仅 MPDD/BTAD 有 vitb14+CLIP 特征；VisA 仅有 v2 预测缓存）。
- 冻结版 A1 **未在 VisA/MVTec 上验证**；现有 VisA/MVTec 结果为**历史 V2/V3 融合系统**产出，与冻结方法（A1 concat+KNN）**口径不同**，仅作背景参考，**不视为本方法验证**：
  - VisA（历史 V3 系统）：`outputs/dynamic_fusion/final_validation/summary.json`（s1_k1 image AUROC 0.821、pixel AP 0.147 等）。
  - MVTec（历史 V3 系统）：`outputs/dynamic_fusion/final_validation/mvtec_summary.json`（s0_k1 image AUROC 0.819、pixel AUROC 0.914 等）。
- 若需正式 VisA/MVTec 冻结后验证：需 GPU 全量导出 vitb14+CLIP 特征（每数据集 9 配置 × 2 分支），
  属新增 GPU 任务，未在本次收尾执行（计划允许"按历史或补充验证角色报告"）。

## 4. 新外部数据集（计划建议项）

- 无冻结前未查看的合适新数据集（本地仅 mpdd/btad/visa/mvtec；visa_candle_smoke 为单类子集，未采用）。
- 该项标注为**未执行**（"最好"级建议，非强制）。

## 5. 收尾状态汇总

| 阶段 | 状态 |
|---|---|
| 一：V3.3 泄漏审计 | ✅（旧结果标记 development-only） |
| 二：V3.3-clean 协议 | ✅（15/15 测试） |
| 三：MPDD s0/K1 CPU Gate | ✅（clean 通过；w=0.40 最优但噪声内） |
| 四：视觉锚定局部救援 | ✅（13/13 测试；增益 < 固定融合） |
| 五：A1 审计 + 9 配置矩阵 + 权重/动态 | ✅（9/9 全正，mean +0.0486；冻结 w=0.5） |
| 六：正式冻结 | ✅（freeze_manifest + METHOD_CARD + REPRODUCE） |
| 七：冻结后验证 | ✅ MPDD 复核 + BTAD K1 验证；VisA/MVTec 按历史角色报告；新数据集未执行 |
