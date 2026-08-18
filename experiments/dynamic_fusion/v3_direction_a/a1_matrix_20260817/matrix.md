# A1 MPDD 完整开发矩阵（阶段五 5.2，2026-08-17）

RunId: `a1_mpdd_matrix_20260817` · MPDD K=1/2/4 × seed 0/1/2，共 9 配置 · 单 GPU 串行 + CPU/faiss。

参考：`docs/DYNAMIC_FUSION_NEXT_STEPS.md` 阶段五 5.2。

## 配置

- 方法：A1 特征层融合 `concat + KNN memory bank`，`pca_dim=0`、`whiten=0`、`dino_weight=0.5`（冻结配置）。
- 分支：DINO **dinov2_vitb14**（768 维，32×32 grid）+ CLIP ViT-L/14@336（768 维，37×37，resize 到 32×32）。
- 评估：`scripts/evaluate_a1_feature_fusion.py`（faiss，STRIDE=8）。

## GPU 工作（缓存复用审计 + 参考特征补充）

**关键审计结论**：A1 测试特征只依赖固定测试集，与 (seed, shot) 无关；k1 缓存已含全量测试特征。
K=2/4 只需 GPU 补算**参考特征**（memory bank 随 shot 变化）。

- 新脚本 `scripts/export_a1_mpdd_ref_only.py`：复用 k1 缓存测试特征，仅重算目标 (seed, shot) 的参考特征。
- **黄金对照通过**：用该脚本重算 k1 参考特征 → A1 fused AP 0.3092 / ΔAP +0.0290，与历史 vitb14+CLIP s0 结果**完全一致**（DINO 与 CLIP 两分支均验证）。
- 12 次小推理（dino×6 + clip×6），单进程串行，全程 GPU 无异常、无崩溃；每项写 export_report.json + sha256。
- 特征缓存：
  - DINO：`outputs/dynamic_fusion/v3_direction_a/features_vitb14_s{0,1,2}_k{1,2,4}/anomalydino_visual/`
  - CLIP：`outputs/dynamic_fusion/v3_direction_a/features_s{0,1,2}_k{1,2,4}/anomalyclip_text/`

## 结果（9 配置，mean Pixel AP）

| seed/shot | K1 | K2 | K4 |
|---|---|---|---|
| seed0 fused (Δ) | 0.3092 (+0.0290) | 0.3437 (+0.0410) | 0.3883 (+0.0596) |
| seed1 fused (Δ) | 0.3425 (+0.0396) | 0.3656 (+0.0569) | 0.3985 (+0.0555) |
| seed2 fused (Δ) | 0.3159 (+0.0434) | 0.3376 (+0.0506) | 0.4047 (+0.0614) |

- **9/9 配置 ΔAP 全为正**，mean ΔAP = **+0.0486**（seed: 0.043/0.051/0.052；shot: K1 0.037 / K2 0.049 / K4 0.059）。
- 增益随 K 增大而增大（K4 最高 +0.059），与"更多正常参考 → 更稳 memory bank"一致。
- 类别趋势与 k1 一致：增益集中在 tubes/connector/metal_plate；bracket_brown 仍小幅退化（-0.006~-0.020）。

## 审计（9/9 全通过）

- `scripts/audit_a1_mpdd_matrix.py`：每个配置 6 类的 dino/clip 特征缓存均通过 schema 检查
  （ndim/N 匹配/grid 一致/ref-grid 一致/mask N 匹配）、**无 NaN/Inf**、gt 有限；报告齐全且含 6 类。
- `matrix_audit.json`：`all_9_configs_pass=true`。
- 五个泄漏字段：特征导出 `test_*_used=false` 全 false（只读 manifest 参考集）；评估仅测试真值评价。

## 与 V3.3-clean 的对比（阶段三 Gate 同一口径）

- V3.3-clean w0.40 (s0/K1) ΔAP = +0.0173；A1 concat w0.5 (s0/K1) ΔAP = +0.0290。
- 9 配置上 A1 平均 +0.0486，远超 V3.3-clean 单点；**A1 是当前最强、且通过 5.1 审计的候选线**。

## 下一步（按计划）

- 动态方案必须**超过最佳固定融合**才能进入冻结。本矩阵已给出固定 A1 (w0.5) 的完整开发基线，
  9/9 全通过。可选项：
  1. A1 权重微调（w 扫描）确认 w=0.5 是否仍最优（跨 9 配置，CPU）；
  2. 动态路由 vs 固定 A1 的显式对照（阶段四式局部救援在特征级变体）；
  3. 直接进入阶段六（正式冻结 A1 固定配置）。
