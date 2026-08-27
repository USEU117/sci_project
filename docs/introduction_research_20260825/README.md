# Introduction 文献与写作资料包

建立日期：2026-08-25  
用途：为少样本工业视觉异常检测论文的 Introduction 和 Related Work 提供可追溯资料。  
状态：研究笔记，不是最终投稿稿；投稿前需按目标期刊格式、最终版本和 DOI 再核对一次。

## 文件

- `INTRODUCTION_LITERATURE_MASTER_20260826.md`：当前统一主资料包，整合本目录资料和 2026-08-10 的既有文献筛选/路线文件。
- `project_status_for_paper.md`：截至 2026-08-25 的项目状态、可写结论和不可写结论。
- `literature_notes.md`：按 Introduction 逻辑链整理的文献笔记和官方链接。
- `introduction_outline.md`：建议的英文 Introduction 段落结构、研究缺口和贡献表述。
- `references.bib`：当前核心参考文献的 BibTeX 草稿。

## 使用原则

1. Introduction 只讲问题、背景、已有路线和清晰缺口；具体实验数字放 Results。
2. 视觉—文本融合不能写成已证明的动态路由优势。当前可防守的主线是：DINO patch 特征与 AnomalyCLIP patch 特征的 reference-conditioned fixed fusion（A1）。
3. AnomalyCLIP 在本项目中作为冻结文本分支/zero-shot来源；不要把它写成 1/2/4-shot 矩阵方法。
4. PromptAD 的 `target_normal_tuning=true` 必须在方法比较和协议说明中显式标注。
5. 不把使用测试 mask 的 V3.3、V4 失败路线或探索性结果作为论文主结论。

## 已整合的历史资料包

以下文件保留原样，作为详细档案；写作时优先从主资料包进入：

- `docs/related_literature_screening_2026_20260810.md`
- `docs/representative_literature_and_validation_plan_20260810.md`
- `docs/sources.md`
