# 动态融合权威状态（CURRENT STATUS）

最新人工复核：2026-09-01 · 原机器快照 RunId：`current_dynamic_fusion_status_20260818`
机器可读历史快照：`docs/current_dynamic_fusion_status.json`
状态快照：`experiments/dynamic_fusion/reconciliation/dynamic_fusion_state_reconcile_20260818_v1/state.json`

本文件保留算法路线和机器状态的权威历史记录。面向导师汇报和论文写作的最新、浅显总览为 `docs/paper_writing_preparation_20260830/10_PROJECT_STATUS_FOR_SUPERVISOR_CN_20260901.md`；最终方法细节以 `submission_repro_20260827/METHOD_SPEC_V2.md` 为准。旧历史报告（阶段七 2026-08-17、V3/V2 各版本）只读保留，不代表当前结论。

---

## 1. 方法（Method）

**名称**：Frozen Dual-Encoder Visual Feature Fusion with a Normal Memory Bank（即 A1）

- 分支：DINO `dinov2_vitb14` patch 特征 + AnomalyCLIP `ViT-L/14@336` patch 特征
- 融合：各分支 L2-normalize → CLIP grid 对齐 DINO grid → concat → L2-normalize → KNN(k=1) normal memory bank → distance/2 = 像素异常图
- 冻结配置：`pca_dim=0, whiten=0, dino_weight=0.5, stride=8, map=448`
- memory bank 只由当前 seed/shot 的 K 张正常参考图构建；测试标签/掩码只进 evaluator
- 最终推理只使用两个图像编码器，不计算文本 embedding；concat 维度为 1536，不是历史误记的 1152。
- 五项泄漏字段全 `false`；**不是动态路由**（固定融合，权重不随测试图变化）

## 2. 数据集角色（权威口径）

| 数据集 | 角色 | 说明 |
|---|---|---|
| MPDD | `development` | 冻结配置与权重在此开发、矩阵 9/9 全正 |
| BTAD | `external_frozen_validation` | 9 配置（3 seeds × 1/2/4-shot）全覆盖 |
| VisA | `in_domain_frozen_validation` | AnomalyCLIP checkpoint 在 VisA 训练过，**非独立 holdout** |
| MVTec | `external_frozen_validation` | 冻结后新验证，9/9 全正 |

## 3. 主结果（baseline source 均显式）

| 数据集 | 角色 | 正向配置 | mean ΔAP vs DINO | baseline source |
|---|---|---|---|---|
| MPDD | development | 9/9 | **+0.0486**（vs legacy v2 dino score） | legacy v2 score 缓存 + matched feature-level dino-only KNN |
| MPDD（口径拆分） | development | — | concat-minus-feature-DINO-only **+0.0258** | feature-DINO-only 自身 +0.0227（vs legacy） |
| BTAD | external | 9/9 | **+0.0766**（vs legacy v2 dino score） | legacy v2 score 缓存（matched per seed/shot） |
| BTAD（口径拆分） | external | — | concat-minus-feature-DINO-only **+0.0249** | feature-DINO-only 自身 +0.0517（vs legacy） |
| VisA | in-domain | 9/9 | **+0.0524** | feature-level dino-only KNN |
| MVTec | external | 9/9 | **+0.0320** | feature-level dino-only KNN |

要点：
- `+0.0486` 是相对 legacy DINO 分数基线；**纯 concat 贡献看 `+0.0258`**（相对 matched feature-DINO-only）。
- 9/9 是同一测试集上 9 组参考采样的鲁棒性，**不是 9 个独立数据集**，不做伪独立显著性。
- 已归档路线（V3.3-leaky / V3.4 / V3.5 / A2 / A2b / A3）不进入主结果。

## 4. 阶段状态

| 阶段 | 状态 |
|---|---|
| 一：V3.3 泄漏审计 | ✅ completed（12 报告 development-only） |
| 二：V3.3-clean | ✅ completed（15/15 测试） |
| 三：MPDD s0/K1 Gate | ✅ completed（+0.0173，泄漏虚增约 7 倍结论） |
| 四：local rescue | ✅ completed（13/13 测试，安全回退） |
| 五：A1 审计+矩阵+权重+动态 | ✅ completed（9/9 全正，冻结 w=0.5） |
| 六：正式冻结 | ✅ completed（freeze_manifest + METHOD_CARD + REPRODUCE） |
| 七：冻结后验证 | ✅ completed（MPDD/BTAD 9 配置/VisA/MVTec） |
| GPT 验收对账 | ✅ completed（8 点全映射） |
| S1：只读 verifier | ✅ completed（229 项全过 + 8/8 篡改测试） |
| S2：文档与角色修正 | ✅ completed（本文件 + 60 处 JSON 角色修正 + 链接检查） |
| S3：统一性能表 | ✅ completed（13 行主表 + 36 行逐类，重算 <1e-6 全 PASS） |
| S4：正式方法包 | ✅ completed（METHOD_CARD/REPRODUCE 修订 + 伪代码 + schema + 资源统计） |
| S5：Git 归档 | ✅ completed（分 3 批提交并 push） |
| P0：投稿技术复现包 | ✅ completed（P0A–P0I；324 maps；独立 CPU 重算） |
| P1：统计、效率、公平性、失败案例、完整指标 | ✅ completed（P1-A/B/C/D/E 门禁全过，`p1_acceptance.json` `p1_complete=true`；P1-C 已补齐预热端到端稳态 benchmark 与峰值进程 RAM，见 `scripts/p1_c_benchmark.py` 与 `p1_c_efficiency.*`） |
| S6/P2：论文交付 | ⏳ pending（先关闭论文前准备 Gate） |
| D0：动态 headroom 门 | ✅ passed（MPDD 9 配置逐像素 best-of-3 Oracle，mean headroom +0.5807） |
| D1：可预测性门 | ❌ **failed → 本项目停止该路线**（LOCO mean AUROC 0.592 < 0.60，特征置乱不下降 0.616 → 当前无标签特征不能可靠预测A1的修正时机） |

## 5. 剩余工作

1. P1 论文实验收尾：✅ 已完成（P1-A bootstrap CI、P1-B 失败边界、P1-C 效率表+稳态 benchmark+峰值 RAM、P1-D 公平性表、P1-E 完整指标；`p1_acceptance.json` `p1_complete=true`，证据在 `submission_repro_20260827/evidence/p1/`）。
2. ✅ 36份 image+pixel 完整指标报告已汇总进 `p1_e_complete_metrics.*`；无需重跑模型。论文须保留 BTAD 图像级 AP/F1 下降的边界。
3. ✅ A1 成功/失败定性图已生成（7 张固定案例，`scripts/build_a1_qualitative_figures.py`；图文件在 `outputs/p1_b_figures/`，source manifest 在包内 `evidence/p1/p1_b_figures_manifest.*`）；P1-B sample IDs 已用于选例，未据图调参。
4. ✅ 带数字的跨方法 baseline 对照表已入包（`evidence/p1/p1_r3_baseline_comparison.*`，由 `scripts/build_cross_method_comparison_table.py` 生成，显式标注协议边界，不宣称全面 SOTA）。
5. 公开发布前人工事项：自研代码 LICENSE 已于 2026-08-27 选定为 **MIT**；2026-08-30 后续核验已确认 MPDD 为 CC BY-NC-SA 4.0、BTAD 为 CC BY-SA 4.0。数据原图与第三方权重继续不入包，正式 release 时需刷新许可说明和哈希。
6. S6/P2 论文交付：关闭上述准备 Gate 后按7节结构写作，全部数字可追溯。
7. ~~路线 D~~ → **本项目停止该路线**（D0 通过但 D1 失败：像素级虽有互补上限，但当前无标签特征不能可靠预测A1的修正时机；按设计审查第 12 节第 9 条停止扩展）。

## 6. 链接检查

详见 `experiments/dynamic_fusion/reconciliation/dynamic_fusion_state_reconcile_20260818_v1/link_check.json`。

## 7. V4 研究扩展（2026-08-19）

权威计划：`docs/DYNAMIC_FUSION_NEXT_STEPS.md`（取代旧执行路线）。V4 是用户新授权的扩展，不影响上文 A1 结论。

| V4 Gate | 状态 | 关键结果 |
|---|---|---|
| G0 审计 | ✅ passed | A1 = `dual_visual_fixed_fusion`（7 项语义证据全过）；A1 冻结 verify 229 项 `all_ok`；候选来源锁定表完成 |
| G1 契约 | ✅ passed | `v4_contracts` + 回归测试 **49 passed**（RouterInput/EvaluationTarget 隔离 + 五项泄漏字段） |
| G2 强视觉锚点 | ❌ V1 failed / ❌ V2 official audit **FAILED** | `subspace_style_same_backbone` vs matched DINO-KNN：pca0.95 mean ΔAP **-0.0124**、pca0.99 **+0.0012**；非负配置 ≤4/9、最差类 ΔAP **-0.135** → V1 归档（`02_visual_gate/v1_archived.json`）。官方 SubspaceAD（giant/672/aug30，commit `ef56d5c`）完整 54 配置审计：mean ΔAP **+0.0472**、9/9 非负、9/9 配置类别正，但 **worst 类 connector = -0.1167**，跌破 -0.020 底线 → **gate_passed = false**（证据：`06_v2_g2_audit/g2_audit_report.json`）。代理失败不成立，但官方版本仍未通过 Gate |
| G3 文本分支 | ⚠️ partial | T0 显式文本 map 方向翻转通过、6 类导出成功；文本单分支 P-AP 仅 **0.139**；Oracle headroom 仅对**弱视觉**成立（vs DINO-KNN +0.359 / vs V1 +0.324），强视觉不存在 → 不构成 H2 证据 |
| 第 12 节决策 | **D（最终，已锁定）** | V2 smoke 2/2 PASS 曾推翻基于 V1 的 D；完整官方审计（54/54）FAIL（connector 单类 -0.117）→ 按预注册判据与停止规则回到 **D**：停止 V4 算法扩展，诚实交付当前 A1（`paper_eligible = false`） |
| G4–G11 | 全部阻断（永久） | 依赖 H1（强视觉锚点）成立；完整官方审计失败后不再进入 |

- 证据目录：`experiments/dynamic_fusion/v4_vision_text_20260819/`（00_g0_audit、02_visual_gate、03_text_gate、04_gate_decision、05_v2_smoke、06_v2_g2_audit）。
- 决策汇总：`experiments/dynamic_fusion/v4_vision_text_20260819/04_gate_decision/gate_decision.md`；官方完整审计：`06_v2_g2_audit/g2_audit_report.json`；官方 smoke：`05_v2_smoke/smoke_report.md`。
- V4 扩展已按第 12 节决策 **D 关闭**：完整官方 G2 审计失败（worst 类 connector），G4–G11 永久阻断，`paper_eligible = false`。剩余主线仅 S6 论文交付。

## 8. 动态融合新方向探索（2026-08-22）

在决策 D 关闭 V4 后，按用户要求对四个候选方向做最后一轮「按顺序试跑、有价值则深入、无价值则放置」的实测（VisA 5 类，同 backbone `dinov2_vitb14`；脚本 `scripts/explore_dynamic_fusion_visa.py`，结果 `experiments/dynamic_fusion/explore_visa_fusion/report.json`）。结论一致为负，进一步坐实决策 D：

| 方向 | 内容 | 结果 | 判定 |
|---|---|---|---|
| A | KNN 距离 + PCA 重建残差逐像素融合（max/mean/加权/秩） | 两映射 Pearson 相关 0.62–0.86（均值 0.75），融合 AP 一律劣于最优单分支（mean 0.313 vs PCA 0.461 / KNN 0.439） | 无价值 |
| B | 类别级 oracle / 逐类选最优 | oracle 均值 0.4747，仅比最优单分支 +0.014，无互补 headroom | 无价值 |
| C | connector「检索型异常」研究 | MPDD 原始数据已清理，无法重跑特征实验；VisA 无同类缺陷可替代 | 不可行 |
| D | backbone 放大（giant） | giant 权重 4.23GB > free RAM 4.0GB，6GB GPU 无法承载，fp64 PCA 必 OOM | 不可行（资源） |

要点：方向 A/B 从像素级与类别级两个层面再次证明「视觉 KNN 距离」与「视觉 PCA 残差」同 backbone 高度冗余、无互补信号，与 Route-D 的 D1 失败（LOCO 0.592）一致——当前特征下不存在可被无标签 gate 利用的互补修正时机。动态融合探索到此正式收敛为「A1 固定融合（w=0.5）+ 负结果叙事」。

## 9. 投稿复现包状态（2026-08-27）

- P0 四数据集研究数值重建：✅ passed。648 个分支特征 NPZ、36 个配置报告齐全；MPDD/BTAD/VisA/MVTec 相对 matched feature-DINO-only 的 ΔPixel-AP 分别为 +0.025829/+0.024895/+0.052353/+0.031962，均在历史值绝对误差 5e-4 内。
- P0 smoke：✅ passed。实测 DINO 768 维、AnomalyCLIP image-tower 768 维、concat **1536 维**；旧文档 1152 为错误记录。
- CPU 回归：✅ 历史快照 81 passed；当前 `tests/` 独立复验 **122 passed in 5.80s**。
- P0 技术复现包：✅ **最终通过（submission_repro_package_complete=true，P0A–P0I 全门禁）**。
  - `predictions_compact/maps/`：324 个逐 `dataset×seed×shot×category` float16 patch maps（含 `sample_ids`、concat/DINO map、grid/map/stride、`ref_ids`、特征缓存 SHA256），逐类重放与 p0_3 报告容差 5e-3（唯一最差项 mvtec s1/k4 wood dino-AUPRO 3.58e-3，纹理大类对 float16 量化最敏感；concat 与 AP/AUROC 均在 ~1e-5）。
  - 包内独立 CPU 脚本 `recompute_tables.py`：`--verify-only` 结构校验 324/324 通过；完整重算经 mpdd s0/k1 与 mvtec s1/k4（含 wood 超差项）冒烟，配置级聚合相对参考表 ≤1e-5，远在 5e-4 内。
  - `rebuild_manifest_v2.json`：324 个 compact npz SHA256，`numerically_equivalent_to_historical=true`、`byte_identical_to_historical=false`；历史 `freeze_manifest.json` 原样保留。
  - `SOURCE_COMMIT.txt`：最终源码提交 `12e1fcf`，dirty=false；提交后已重新生成 `SHA256SUMS`（447 条受校验记录全通过；包内 448 个文件含清单自身）。
  - `METHOD_SPEC_V2.md` 已固定 1536/双视觉语义，`anomalyclip_text` 仅为历史目录名。
  - `LICENSES_AND_DATA.md` 是 2026-08-27 的冻结快照，当时尚未完成 MPDD/BTAD 许可核验。2026-08-30 后续证据已确认 MPDD 为 CC BY-NC-SA 4.0、BTAD 为 CC BY-SA 4.0；为保持冻结哈希，不静默修改旧包，正式 release 时统一刷新该文件和校验清单。
- 历史 freeze byte identity：❌ 仍不成立（仅声明数值等价，见 `rebuild_manifest_v2.json`）。
- 权威验收：`docs/submission_reproducibility_20260826/P0_ACCEPTANCE_REVIEW_20260827.md`；机器审计：`P0_ACCEPTANCE_AUDIT_20260827.json`（`submission_repro_package_complete=true`）。

## 10. 当前下一步与交付标准（2026-08-27）

当前不再开发动态路由，也不重导 648 个分支特征。按以下顺序推进：

1. **P1-A 统计**：✅ 已完成。`scripts/p1_stats_bootstrap.py` 从 324 个 compact maps + 用户 mask 生成 36 配置的 category bootstrap 与异常图像级 per-image ΔAP bootstrap 95% CI；`p1_a_bootstrap_ci.*` 含 dataset×shot 三 seed mean±std；四数据集均值 0.025839/0.024896/0.052361/0.031957，与主表差 ≤8e-6（≤5e-4）。统计层级已明确（类别=论文口径 pooled AP；图像=异常图 per-image ΔAP）。
2. **P1-B 失败边界**：✅ 已完成。`p1_b_failure_boundaries.md` 列出每 dataset worst category（mvtec leather −0.043、visa chewinggum −0.039、mvtec hazelnut −0.030 等）与 10 个负增益 dataset@category；`p1_b_failure_samples.csv` 含每配置 top-5 逐图失败样例 ID（仅 ID，不复制原图）。
3. **P1-C 效率**：✅ 已入包。`p1_c_efficiency.*`：0 训练参数；稳态 DINO 0.0626s / CLIP 0.3049s / 对齐+concat+KNN 0.0471s，端到端 0.4146s、2.412 image/s；峰值 VRAM 约 2073MB，峰值 RAM 3980.9MB；按 dataset×shot 记忆库 patch 数与 float32 MB；compact 包约 186.8MB。
4. **P1-D 公平性**：✅ 已纠错并完成。AnomalyDINO 项目实际为 `dinov2_vits14_448`、training-free；WinCLIP+ 为项目统一1/2/4-shot三 seed 矩阵，zero-shot WinCLIP 单独报告。
5. **P1-E 完整指标主表**：✅ 已聚合。`p1_e_complete_metrics.*` 汇总四数据集36份报告、72个method-config rows和六项指标，输入哈希齐全；相对P0 Pixel-AP最大差 `3e-6`。重要边界：四数据集稳定提升针对Pixel-AP；BTAD Image-AP约 `−0.0131`、Image-F1-max约 `−0.0237`，不得宣称所有检测/定位指标全面提升。
6. **定性图**：✅ 已完成（R4）。`scripts/build_a1_qualitative_figures.py` 从 P1-B 固定 sample IDs + compact concat/DINO maps + 合法本地原图/GT 生成 7 张固定成功/失败案例图（含 DINO-only 与 A1 对照、逐图 Pixel-AP）；图文件本地保存于 `outputs/p1_b_figures/`（gitignored，含不可再分发原图），包内 `evidence/p1/p1_b_figures_manifest.*` 记录选择规则、source IDs 与文件哈希；未据图调参。
7. **发布人工 Gate**：代码 LICENSE 已选 **MIT**；MPDD 官方仓库为 CC BY-NC-SA 4.0，BTAD 原作者仓库链接确认 CC BY-SA 4.0。数据原图和第三方权重继续不进入 compact 包；最终公开前仍需刷新 release notice、哈希和归档 URL。
8. **P2/P3**：上述准备完成后重写论文，再实时核验目标 SCI 四区期刊的最新分区、scope 与格式。

P1-A/B/D 已同时包含机器可读 JSON/CSV、Markdown 表、生成脚本、输入 source pointer 与无测试标签调参声明。P1-C 的预热端到端稳态 benchmark 与峰值进程 RAM 已实测并入包（`scripts/p1_c_benchmark.py`，MVTec bottle s0/k1：DINO 0.0631s / CLIP 0.3047s / concat 0.0471s，端到端 0.4146s、2.412 img/s，峰值进程 RAM 3980.9MB）。

## 11. 2026-09-01 论文准备增量

1. BTAD/MVTec CLIP-image-only：✅ 18/18 配置、六指标完成；A1 Pixel-AP 在两数据集全部 18 个配置中高于 CLIP-only。结果在 `experiments/dynamic_fusion/v3_direction_a/clip_only_controls_20260830/`。
2. 数据许可：✅ BTAD 确认为 CC BY-SA 4.0；✅ MPDD 官方仓库为 CC BY-NC-SA 4.0。不得把 VT-ADL 代码的 MIT 许可证误写为 BTAD 数据许可证。
3. 论文图件：✅ 11 张图完成，包括方法、协议、配置增益、类别边界、六指标、三分支、效率及成功/失败案例；定量图均有 SVG/PDF/600-dpi PNG，QA 通过。
4. 中文导师会议总览：✅ `docs/paper_writing_preparation_20260830/10_PROJECT_STATUS_FOR_SUPERVISOR_CN_20260901.md`。
5. 当前阶段：停止算法扩展，进入目标期刊选择、英文 Method/Experiments/Results 写作、引用核验和投稿声明准备。

## 12. RCEC 创新方向实现与验收（2026-09-01，负结果归档）

按任务书 `docs/paper_writing_preparation_20260830/11_RCEC_INNOVATION_IMPLEMENTATION_AND_ACCEPTANCE_HANDOFF_CN_20260901.md` 完成工程交付与科研验收，**Phase 2 小门失败 → 按第 8 节停止规则早停并归档**，A1 保持论文主方法不变。

| 项 | 结果 |
|---|---|
| Phase 0 输入审计 | ✅ 通过（`experiments/dynamic_fusion/rcec_v1/PHASE0_INPUT_AUDIT.md`；A1 冻结证据未修改） |
| 单元/技术测试 | ✅ 18/18 通过（`tests/test_rcec.py`，patchcore 环境） |
| Phase 2 MPDD 小门（12 候选 × seed0 × shot 1/2/4） | ❌ 0/12 通过（`small_gate_summary.csv`、`SMALL_GATE_REPORT.json`） |
| 早停决定 | ✅ `RCEC_V1_EARLY_STOP_REPORT.json`（winners=[]，按任务书不运行 full/freeze/验证） |
| 最终决策 | **ARCHIVE**（`FINAL_RCEC_DECISION.md`） |

关键数值：12 个预注册候选（direction × k ∈ {1,3,5} × λ ∈ {0.25,0.50}）在 MPDD seed0 全部低于 A1；最佳候选 `dino_to_clip_k5_lam0.25` 平均 ΔPixel-AP **−0.0071**（仅 1/3 shot 正），λ 越大退化越严重，k=5 略优于 k=1，dino_to_clip 平均优于 symmetric。结论如实写入论文 Discussion/Future Work：正常参考条件下的跨编码器邻域分歧没有稳定超过固定拼接，简单互补收益并不必然转化为可利用的局部一致性信号。RCEC 相关源码/配置/runner/测试全部保留，负结果不隐藏、不改门槛、不换主指标。
