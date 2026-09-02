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
17. [多路线算法创新执行与验收任务书（A2 Innovation Program）](12_MULTI_ROUTE_ALGORITHM_INNOVATION_EXECUTION_AND_ACCEPTANCE_CN_20260902.md) —— 待执行。包含 6 条不同假设路线、统一 MPDD 小门/完整门、机制消融、单一 winner 冻结以及外部验证纪律。任何候选通过前，不改变 A1 主方法口径。

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
