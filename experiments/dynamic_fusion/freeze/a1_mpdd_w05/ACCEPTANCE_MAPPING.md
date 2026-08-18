# 验收对账总表（GPT 验收意见 → 产物映射）

生成日期：2026-08-18 · 目的：逐条映射 GPT 验收意见到已存档产物，防止"文档任务未完成"的误判，并如实列出仍缺口项。

## 一、阶段验收点逐条对账

| # | GPT 验收点 | 状态 | 证据产物 | 说明 |
|---|---|---|---|---|
| 1 | V3.3 协议审计，确认 `audit.json/audit.md` | ✅ 完成 | `experiments/dynamic_fusion/v3_3/audit_20260817/{audit.json,audit.md}` | 3 个旧策略全部命中 `gt_masks` 挑选正常测试图做 z-score 校准，12 个报告全部标记 `development_only_leaky_calibration=true`、`paper_eligible=false`，只读保留不修改数值 |
| 2 | V3.3-clean 协议修复（RouterInput/EvaluationTarget 分离 + 正常参考校准） | ✅ 完成 | `src/industrial_ad/fusion/v3_3_clean.py` + `tests/test_v3_3_clean.py`（15/15 通过）+ `experiments/dynamic_fusion/v3_3_clean/` | `RouterInput`（预测+K 张正常参考+sample IDs，禁止 gt 字段）/`EvaluationTarget`（仅测试真值，只允许 evaluator）；校准仅来自 K 张正常参考；五项泄漏字段全 false |
| 3 | MPDD seed0/K1 CPU Gate（无泄漏） | ✅ 通过 | `experiments/dynamic_fusion/v3_3_clean/gate_20260817/{report.json,gate.md}` | 复用冻结缓存；**关键发现：泄漏把旧 V3.3 增益虚增约 7 倍**（leaky +0.0754 vs clean w0.60 +0.0108）；clean w0.40 6/6 类全正 +0.0173 |
| 4 | 视觉锚定文本局部救援（预注册 Gate） | ✅ 完成 | `src/industrial_ad/fusion/v3_3_rescue.py` + `tests/test_v3_3_rescue.py`（13/13）+ `experiments/dynamic_fusion/v3_3_clean/phase4_rescue_20260817/{report.json,phase4.md}` | 固定 6 步流程 + 6 类 reason code；rescue_cap4 mean ΔAP +0.0051，安全但未超过固定融合 +0.0173，保留为安全回退 |
| 5 | 7.1-A1 特征级融合严格审计 | ✅ 完成 | `experiments/dynamic_fusion/v3_direction_a/audit_20260817/{audit_a1.json,audit_a1.md}` | 五项泄漏全 false；正常参考仅来自 manifest；PCA/KNN 只 fit 于正常参考；测试真值仅评价；`paper_eligible=pending_gate_review` |
| 6 | MPDD 1/2/4-shot × 3 seeds 完整矩阵 | ✅ 完成 | `experiments/dynamic_fusion/v3_direction_a/a1_matrix_20260817/{matrix.md,matrix_summary.json,matrix_audit.json}` + 9 个 per-config report | **9/9 全正，mean ΔAP +0.0486**（by_seed 0.043/0.051/0.052；by_shot 0.037/0.049/0.059）；审计 9/9 通过 |
| 7 | 方法冻结（freeze manifest） | ✅ 完成 | `experiments/dynamic_fusion/freeze/a1_mpdd_w05/{freeze_manifest.json,METHOD_CARD.md,REPRODUCE.md}` | 9 代码文件 + 2 checkpoint + manifest + evaluator + 108 特征 npz + 108 baseline npz 全部 sha256；`--verify` 通过 |
| 8 | 冻结后外部验证 | ✅ 完成 | `freeze/a1_mpdd_w05/phase7_post_freeze_20260817.md` | MPDD 复核 ✅；BTAD K1 3/3 正（mean +0.0726）✅；**VisA 冻结后验证已完成（2026-08-18，9/9 全正 mean ΔAP +0.0524）**；**MVTec 冻结后验证已完成（2026-08-18，9/9 全正 mean ΔAP +0.0320）**；新外部数据集未执行（需新数据集授权） |

## 二、GPT 对 A1 的 6 项待确认点对账

| # | 待确认点 | 结论 | 证据 |
|---|---|---|---|
| 1 | 三个 seed 是否真的使用同一固定配置 | ✅ 是 | `run_a1_mpdd_matrix.py` 固定 `concat pca_dim=0 whiten=0 w=0.5`，9 配置同一条命令模板，仅 seed/shot 变化 |
| 2 | 是否根据测试结果选择过融合权重 | ✅ 无违规 | 权重扫描（w∈{0.3..0.7}×9 配置）在 **development 集**（MPDD）上进行，属合法开发流程；结果 w0.4 仅比 w0.5 高 +0.0009（噪声内），最终冻结 w=0.5（等权、无超参）见 `a1_weight_scan_20260817/weight_scan.md` |
| 3 | PCA、KNN memory bank 是否只用正常参考图拟合 | ✅ 是 | `audit_a1.md` 证据 2：PCA/whitening 仅 fit 于 `ref_flat`，memory bank 由 `ref_proj` 构建；测试特征只 transform/search |
| 4 | 测试标签是否只用于最终评价 | ✅ 是 | `audit_a1.md` 证据 4：`imgs_masks`/`gt_sp` 仅 `compute_metrics` 使用；五项泄漏字段全 false |
| 5 | DINO/CLIP patch 的 sample_id、空间位置和尺寸是否严格对齐 | ✅ 是 | 两分支均 32×32 grid；CLIP 37×37 经 bilinear resize 到 32×32；sample_id 逐图对齐；`audit_a1.md` + `matrix_audit.json`（schema/对齐/NaN-Inf 9/9 通过） |
| 6 | 当前结果是不是只完成 K=1 | ✅ 否 | K=1/2/4 × seed 0/1/2 完整 9 配置矩阵已跑完（见一.6） |

## 三、A1 特征级消融补充（2026-08-18 新增，回应"消融缺失"）

| 模式 | mean fused Pixel AP | mean ΔAP vs DINO | 正向配置 |
|---|---|---|---|
| DINO 单分支（特征级 KNN） | 0.3304 | **+0.0227** | 8/9 |
| CLIP 单分支（特征级 KNN） | 0.2854 | **-0.0222** | 0/9 |
| **concat + KNN（冻结 w=0.5）** | 0.3562 | **+0.0486** | 9/9 |

结论：concat 相对 dino-only 再 +0.0258，强于任一单分支 → **增益来自 CLIP 互补，非偶然**。产物：`experiments/dynamic_fusion/v3_direction_a/a1_ablation_20260817/{ablation_summary.json,ablation.md}`。

## 四、已归档的负结果路线（GPT 建议停止扩展，均已保留证据）

| 路线 | 结果 | 证据 |
|---|---|---|
| A2 注意力式跨模态调制 | 增益仅 +0.0005~0.0018 | 项目记忆"方向A 实验"节 |
| A2b CCA | Pixel AP -0.152~-0.157 | 同上（centering 破坏 KNN 原点语义） |
| A3 共享子空间 | 多数下降，正值不稳定 | 同上 |
| 手写缺陷词 prompt | 平均 AP -0.036 | 同上 |
| 图像级 gate | Oracle 上限仅约 +0.010 AP | 同上 |
| 动态路由（per-category KNN 紧凑度） | 54/54 全选 w0.4，退化为固定，差 +0.0009 | `a1_dynamic_vs_fixed_20260817/dynamic_vs_fixed.md` |

## 五、整体判定

- **GPT 意见基于旧状态**（认为约 30% 完成）。实际阶段一~七产物齐全（上表一.1~7 全部完成）。
- 剩余真实缺口：
  1. ~~A1 特征级消融对照~~ → 已于 2026-08-18 补齐（见三）。
  2. ~~VisA 冻结后验证~~ → 已于 2026-08-18 完成（见七）。
  3. ~~MVTec 冻结后验证~~ → 已于 2026-08-18 完成（见八）。
  4. 新外部数据集验证 → 未执行（需新数据集授权）。
- 冻结配置（concat w=0.5, pca_dim=0, whiten=0）保持不动；后续仅按冻结版本做外部验证，不得调参。

## 七、VisA 冻结后验证（2026-08-18 完成）

- **结果**：9/9 配置全正，mean ΔAP **+0.0524**（vs DINO feature-level baseline；by_seed 0.054/0.049/0.054；by_shot 0.056/0.052/0.049）。
- **逐类**：10/12 类 9/9 全正（capsules/cashew/fryum/macaroni1/macaroni2/pcb1-4/pipe_fryum）；candle（-0.020）与 chewinggum（-0.039）小幅退化，无灾难类。
- **审计**：285 项检查全过（schema/类别/NaN/均值一致性/对齐/泄漏字段/grid 一致性）。
- **产物**：`experiments/dynamic_fusion/v3_direction_a/a1_visa_20260818/{visa.md,visa_summary.json,visa_audit.json}` + 27 个 per-config report；特征缓存 `outputs/dynamic_fusion/v3_direction_a/visa_features*/`（导出队列 18/18 成功）。
- **baseline 口径说明**：VisA 无 v2 分数级缓存 → 用特征级 dino-only KNN 作 DINO baseline（MPDD s0/K1 上该口径与 v2 分数级差 ~0.0008 AP，可比）。

## 八、MVTec 冻结后验证（2026-08-18 完成）

- **结果**：9/9 配置全正，mean ΔAP **+0.0320**（vs DINO feature-level baseline；by_seed 0.036/0.027/0.034；by_shot 0.035/0.031/0.030）。
- **逐类**：15 类中 11 类 9/9 全正（toothbrush +0.108 / carpet +0.080 / pill +0.077 / screw +0.068 / tile +0.051 等）；capsule（-0.016）、grid（-0.006）、hazelnut（-0.030）、leather（-0.043）小幅退化，无灾难类。
- **消融对照**：dino-only 0（基准）；clip-only -0.0573 全负（0/9）→ concat 增益来自 CLIP 互补，与 MPDD/VisA 结论一致。
- **审计**：327 项检查全过（schema/类别/NaN/均值一致性/对齐/泄漏字段/grid 一致性/mask 尺寸 dino 448 vs clip 518），n_failed=0。
- **产物**：`experiments/dynamic_fusion/v3_direction_a/a1_mvtec_20260818/{mvtec.md,mvtec_summary.json,mvtec_audit.json}` + 27 个 per-config report；特征缓存 `outputs/dynamic_fusion/v3_direction_a/mvtec_features*/`（导出队列 18/18 成功）。
- **基线口径**：MVTec 无 v2 分数级缓存 → 与 VisA 同口径（特征级 dino-only KNN）。
- **脚本参数化（不触碰冻结清单哈希）**：复用 VisA 的六个脚本新增 `--dataset`（visa/mvtec），`v2_mpdd_prediction_common.py` 新增 `index_mpdd` 复用分支（`index_dataset` 路由 MVTec→`index_mpdd`，目录约定一致）；MPDD/BTAD/VisA 行为不变。

## 六、待办清单（按文档规定顺序）

1. [x] V3.3 协议审计（development_only_leaky_calibration）
2. [x] V3.3-clean 实现 + 15/15 CPU 测试
3. [x] MPDD seed0/K1 CPU Gate（泄漏虚增约 7 倍结论）
4. [x] 视觉锚定文本局部救援（安全回退）
5. [x] A1 严格审计 + 9 配置矩阵 + 权重扫描 + 动态对照
6. [x] 方法冻结（freeze_manifest / METHOD_CARD / REPRODUCE）
7. [x] VisA 冻结后验证（2026-08-18 完成，9/9 全正 mean ΔAP +0.0524）
8. [x] MVTec 冻结后验证（2026-08-18 完成，9/9 全正 mean ΔAP +0.0320）
9. [ ] 新外部数据集验证（未授权，需新数据集）
