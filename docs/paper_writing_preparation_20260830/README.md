# SCI Paper Writing Preparation Hub (2026-08-30)

本目录是当前 A1 主论文的唯一写作入口。它不替代实验、复现包或历史记录，只把这些材料组织成可直接用于英文论文写作的工作台。

## 一句话论文主线

本文不是“新的动态多模态路由方法”，而是一项关于**冻结双视觉编码器互补性**的受控研究：将 DINOv2 与 AnomalyCLIP 图像塔的 patch 特征做固定、等权、零训练融合，在严格 normal-only few-shot 协议下，与 matched DINO-only 管线比较，并系统报告跨数据集收益、失败边界、计算代价和可复现证据。

## 当前不可改变的事实

- 最终方法只使用两个**图像编码器**；没有显式文本特征，没有动态路由，没有参数训练。
- concat 维度为 `1536 = 768 + 768`，不是旧文档中的 1152。
- MPDD 是 development dataset；BTAD 与 MVTec AD 是 external frozen validation；VisA 是 in-domain frozen validation。
- 四个数据集、每个数据集 9 个 seed/shot 配置的 Pixel-AP 平均增益均为正；这一结论不能扩写成“全部指标均提高”。
- A1 不是新的 SOTA；论文价值应落在受控证据、简单融合的稳定收益、负迁移边界与复现完整性，而不是排行榜第一。
- `anomalyclip_text` 是历史目录名，不表示 A1 推理使用了文本特征。

## 建议阅读顺序

1. [项目材料审计](00_PROJECT_MATERIAL_AUDIT.md)
2. [论文定位与可声明贡献](01_PAPER_POSITIONING_AND_CLAIMS.md)
3. [英文主稿结构蓝图](02_MANUSCRIPT_BLUEPRINT_EN.md)
4. [英文 Introduction 工作稿](03_INTRODUCTION_WORKING_DRAFT_EN.md)
5. [Related Work 与近邻差异图](04_RELATED_WORK_AND_NOVELTY_MAP.md)
6. [主张—证据矩阵](05_CLAIM_EVIDENCE_MATRIX.md)
7. [图表规划](06_TABLE_FIGURE_PLAN.md)
8. [缺口与投稿前清单](07_MISSING_MATERIALS_AND_CHECKLIST.md)
9. [项目结构与权威文件索引](08_PROJECT_STRUCTURE_INDEX.md)
10. [BTAD/MVTec CLIP 图像单分支对照结果](09_BTAD_MVTEC_CLIP_ONLY_CONTROL_RESULTS.md)
11. [导师会议版中文项目总览](10_PROJECT_STATUS_FOR_SUPERVISOR_CN_20260901.md)
12. [论文图件包、英文图注与 Word 使用说明](figures_20260830/README.md)
13. [图件完整性与视觉 QA 报告](figures_20260830/QA_REPORT.md)
14. [BTAD 许可证证据记录](BTAD_LICENSE_EVIDENCE.md)
15. [参考文献审计](references/REFERENCE_AUDIT.md) 与 [合并 BibTeX](references/curated_references.bib)
16. [RCEC 创新方法实现与验收任务书（含执行复核与下一步）](11_RCEC_INNOVATION_IMPLEMENTATION_AND_ACCEPTANCE_HANDOFF_CN_20260901.md) —— 已执行并于 2026-09-02 独立复核：专项测试 18/18、项目自有测试 141/141 通过；Phase 2 小门 0/12 通过，按任务书早停归档，A1 保持主方法。决策见 `experiments/dynamic_fusion/rcec_v1/FINAL_RCEC_DECISION.md`；小门和早停证据位于 `experiments/dynamic_fusion/rcec_v1/development_mpdd/`。结论已写入 `docs/CURRENT_DYNAMIC_FUSION_STATUS.md` 第 12 节与中文初稿 Discussion 10.5。
17. [多路线算法创新执行与验收任务书（A2 Innovation Program）](12_MULTI_ROUTE_ALGORITHM_INNOVATION_EXECUTION_AND_ACCEPTANCE_CN_20260902.md) —— 已执行。27 个预注册候选均未通过 MPDD 小门，A1 保持主方法；原始决策见 `experiments/dynamic_fusion/innovation_v2/FINAL_DECISION.md`。2026-09-02 独立复核发现 DEVA 几何映射/combined 变换和 NCPRA 最佳权重/随机种子存在实现有效性问题，因此 D/E 只能视为当前实现未通过，须按 A3 Wave 0 做限定纠错，不能写成路线已被彻底证伪。
18. [表征层算法突破执行与验收任务书（A3 Breakthrough Program）](13_REPRESENTATION_LEVEL_BREAKTHROUGH_EXECUTION_AND_ACCEPTANCE_CN_20260902.md) —— 待执行。先限定修正 A2 D/E，再独立尝试 CASF“多层特征 + 跨分支非对称伪异常监督”和 DC-SZoom“双线索稀疏高分辨率记忆”；包含候选上限、机制控制、训练随机性、MPDD 两级门、唯一 winner 冻结及一次性外部验证纪律。
19. [更宽算法创新版图与路线优先级（A4 Research Portfolio）](14_BROAD_ALGORITHM_INNOVATION_PORTFOLIO_CN_20260902.md) —— 研究构思文件，不授权全部实施。把搜索空间扩展到 RG-MCR 参考引导掩码上下文修复、SF-NM 频谱正常记忆、RG-OT 关系图最优传输、组件图、跨数据集元学习、扩散反事实修复和 conformal evidence；推荐先完成三个信息价值诊断，最多选择两条路线形成下一份正式任务书。
20. [A4 信息价值诊断矩阵与路线决策](../../experiments/dynamic_fusion/innovation_v4_diagnostics/README_STATUS.md) —— 已按 14 号 §10 完成 D1/D2/D3（全部 MPDD development，seed 0）。结果：D1（SF-NM/DC-SZoom）headroom 全 0 未通过；D2（RG-MCR/RG-OT）context 仅 perm/dup 部分正、missing 失败且未达 0.80，node-OT ≈ 随机，未通过；D3（CASF）mean headroom +0.0668 ≥ 0.02 通过但呈类条件（3/6 类）。推荐 1 条路线（CASF 类条件设计）进入下一份正式任务书；其余路线暂归档。详见 D1/D2/D3_SUMMARY.json。
21. [CASF 类条件算法创新与实验方案（A4 入选路线任务书）](15_CASF_CATEGORY_CONDITIONAL_ALGORITHM_AND_EXPERIMENT_PLAN_CN_20260902.md) —— **已执行并归档（Wave 0 后提前停止）**。放大合成探针（24ep/家族×3 seeds）冻结 Gset = {bracket_white}（仅 1/6 类达标；bracket_brown 早期 +0.59 系小样本 sym 崩溃，不稳健）；绝对 Dice ≤0.096 且 pooled-6 小门需该单类 ≥+0.03 ΔPixel-AP，算术不可达 → 用户确认提前归档，A1 保持主方法。证据与决策见 `experiments/dynamic_fusion/innovation_v5_casf/`（PROBE_SUMMARY/GSET/FINAL_CASF_DECISION）。
22. [负结果之后的新融合方向与分阶段验收任务书](16_POST_NEGATIVE_RESULTS_NEW_FUSION_ROUTES_AND_ACCEPTANCE_CN_20260902.md) —— 当前建议先做 **DG-SAFE 双正常性几何安全融合**：A1 的实例最近邻残差与官方 SubspaceAD 的多层子空间重建残差已有真实互补证据（MPDD 54 个类别×配置单元中 38 个为正，对 A1 总平均 ΔPixel-AP +0.0213），先导出逐像素图并验证正常-only 稳定性是否能保护 connector；全局—局部校准与概率原型流形仅作顺序备选。所有名称和贡献仍是候选，Full MPDD 通过前不得写成已验证创新。
23. [全局文本证据确认、算法升级与未来分支执行任务书](17_GLOBAL_TEXT_EVIDENCE_CONFIRMATION_AND_FUTURE_ALGORITHM_HANDOFF_CN_20260903.md) —— 面向下一位 AI 的完整执行规格。先用现有 cache 完成 MPDD 3 seeds × 3 shots、paired bootstrap 和未看 seed1/2 confirmation；通过后冻结“文本图像筛查 + A1 像素定位”的 GLSD 双输出系统。若仍需要更强算法创新，再依次执行文献边界审计、TCRR 区域级信息价值门、最小候选/强控制、Full MPDD 和一次性 BTAD/MVTec 验证。文档预注册 Scenario A–E，明确失败时自动归档、VisA source-trained 边界以及哪些结果能/不能写成论文贡献。**已执行（2026-09-03）→ Scenario C**：Phase 0 审计/14 测试通过；Full MPDD 3×3 上 G1 未过（9 配置 macro ΔImage-AP +0.0288，但仅 6/9 配置为正；paired bootstrap 95% CI [−0.0226, +0.0737]，下界 <0），seed0 的 +0.0249 降级为 exploratory observation；GLSD 未冻结、BTAD/MVTec 未验证、TCRR 未开发，A1 保持唯一主方法。决策与全表见 `experiments/dynamic_fusion/innovation_v7_global_text/01_mpdd_full/PHASE1_DECISION.md`。

## 权威来源优先级

发生冲突时，按以下顺序判断：

1. `submission_repro_20260827/METHOD_SPEC_V2.md` 与 `submission_repro_20260827/evidence/`
2. `docs/paper_writing_preparation_20260830/10_PROJECT_STATUS_FOR_SUPERVISOR_CN_20260901.md`（面向人的最新总览）
3. `docs/PRE_MANUSCRIPT_READINESS_AUDIT_20260827.md`
4. `docs/PAPER_DETAILED_CHINESE_DRAFT_20260827.md`
5. `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`（算法历史和机器状态）
6. `docs/introduction_research_20260825/` 中的文献资料
7. 历史 DOCX、旧动态融合文档和旧 `METHOD_CARD.md`

第 6 层只能用于追溯过程或回收通用背景文字，不能覆盖当前方法口径。

## 主稿建议工作流

先锁定目标期刊模板和篇幅，再按 `Method → Experiments → Results → Introduction → Related Work → Abstract → Conclusion` 的顺序写英文初稿。这样可以让 Introduction 的贡献点由已经成立的证据约束，而不是反过来制造过强叙事。

## 本目录的维护规则

- 新的主稿论断必须先登记到 `05_CLAIM_EVIDENCE_MATRIX.md`。
- 新引用先进入 `references/REFERENCE_AUDIT.md`，核对元数据后再写入 BibTeX。
- 不在本目录复制数据集原图、第三方权重或大体积输出。
- 不静默修改冻结复现包；若复现包必须更新，应重新运行其审计与哈希流程。
- 历史动态融合失败结果保留，但只用于 Discussion/Appendix，不重新启动为主方法。
