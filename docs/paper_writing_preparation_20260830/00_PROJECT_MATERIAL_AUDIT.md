# 00. Project Material Audit

## 1. 审阅范围与方法

本次总览覆盖仓库目录结构、文件类型、现有 Markdown、Word 草稿、BibTeX、关键实验表、P0/P1 审计材料、复现包说明和历史动态融合记录。

- 约 140 个 Markdown 文件均完成路径/标题/章节级扫描；当前主线、Introduction、论文草稿、复现与 P1 证据文件做了正文级阅读。
- 11 个 DOCX 均完成内容抽取与结构检查。旧英文 V0.3 可读，但其方法是已废弃的 uncertainty routing / visual-text fusion；5 个中文进度 DOCX 与若干动态融合设计 DOCX 主要作为历史档案。
- 14 个 PDF 多数是 DOCX 的渲染或 QA 副本，以及第三方概览材料；已做清点，不将重复 PDF 当成新的论文证据。
- 数据集图像、缓存、NPZ、模型输出和权重按目录、格式、数量与用途盘点，没有逐张视觉阅读。逐张读取约六万张图像既不必要，也不属于论文材料审阅；其科学有效性由现有 manifest、评测报告和复现审计覆盖。
- 未发现仓库级 `AGENTS.md`。

## 2. 仓库体量快照

以下数值排除了 `.git` 和虚拟环境，目的是帮助管理材料，不作为论文结果：

| 顶层目录 | 约文件数 | 约体量 | 主要用途 |
|---|---:|---:|---|
| `outputs/` | 2,238 | 317.18 GB | 原始/统一评测输出与大体积中间产物 |
| `data/` | 23,154 | 13.98 GB | 四个数据集本地副本 |
| `methods/` | 39,128 | 5.13 GB | 第三方方法与依赖代码 |
| `experiments/` | 2,218 | 1.34 GB | A1、动态融合与消融记录 |
| `submission_repro_20260827/` | 469 | 186.8 MB | 投稿级 compact 复现包 |
| `docs/` | 52+ | 小 | 项目状态、论文草稿、审计与文献笔记 |

主要扩展名约为：46,638 个 JPG、13,940 个 PNG、1,952 个 BMP、1,056 个 JSON、990 个 NPZ、514 个 CSV、415 个 Python、140 个 Markdown、32 个 PTH、14 个 PDF、11 个 DOCX、2 个 BibTeX。

## 3. 已有论文材料的可用性

### A. 可直接作为当前主稿依据

- `submission_repro_20260827/METHOD_SPEC_V2.md`：最终方法定义和命名纠错。
- `submission_repro_20260827/evidence/p1/`：统计、失败边界、效率、公平性和六指标完整表。
- `submission_repro_20260827/evidence/paper_tables/`：主结果机器可读版本。
- `docs/PAPER_DETAILED_CHINESE_DRAFT_20260827.md`：当前最完整的中文论文骨架。
- `docs/PRE_MANUSCRIPT_READINESS_AUDIT_20260827.md`：主稿前材料完备性审计。
- `docs/PAPER_SUBMISSION_HANDOFF_AND_REPRODUCIBILITY_PLAN_20260826.md`：复现和投稿交接逻辑。

### B. 可回收，但必须改写

- `docs/introduction_research_20260825/`：数据集、PatchCore、DINOv2、CLIP、WinCLIP、AnomalyCLIP、PromptAD 及 2025–2026 方法的文献笔记。
- `outputs/paper_draft_20260810/Leakage-Safe_Uncertainty_Routing_English_SCI_Draft_V0.3.docx`：工业背景、few-shot 动机、协议公平性和负结果的英文句式可回收。
- `docs/related_literature_screening_2026_20260810.md`、`docs/representative_literature_and_validation_plan_20260810.md`：近邻工作筛选思路可继续利用。

回收时必须删除或替换以下旧表述：`language evidence`、`text branch`、`uncertainty router`、`dynamic fusion`、`multimodal inference`、1152-D concat，以及把 VisA 称为独立外部验证的说法。

### C. 仅作为历史档案

- 旧中文/英文 V0.1–V0.3 主稿中的动态路由方法和贡献列表。
- `experiments/dynamic_fusion/` 下 V3.3、V3.5、V4、SubspaceAD gate 等已关闭路线。
- 冻结时刻的旧 `METHOD_CARD.md`，其中 1152 维和 multimodal 命名已经由 V2 规格纠正。
- 大量 `outputs/` 原始中间件；论文引用时应优先使用 compact 包中的重建表，而不是人工抄录旧输出。

## 4. Introduction 材料审计结论

项目内确实已有较完整的 Introduction 资料包，但不能原样拼接成当前论文：

1. 工业检测、异常样本稀缺、normal-only few-shot 的问题背景仍然适用。
2. PatchCore/AnomalyDINO 所代表的视觉记忆方法，以及 CLIP/AnomalyCLIP 所代表的视觉语言预训练背景仍然适用。
3. 原资料把第二分支解释为“文本/语言证据”，与最终 A1 的纯图像推理不一致。
4. 2026 年已经出现 Sea-CLIP、PAPL 等显式结合 CLIP 与 DINO 的方法，因此不能声称首次融合两者。
5. 更可靠的 gap 是：在不训练融合器、不使用测试标签选规则的条件下，异构冻结 patch 表征的简单组合是否产生稳定收益，其收益边界、失败类别和代价是否被完整报告。

当前重写版本见 [03_INTRODUCTION_WORKING_DRAFT_EN.md](03_INTRODUCTION_WORKING_DRAFT_EN.md)。

## 5. 项目结构优化决策

本轮采用“索引优化”而非“物理搬迁”：

- 新增本目录作为唯一论文写作入口。
- 不移动 `outputs/`、`experiments/`、`methods/` 或复现包，避免破坏脚本路径、manifest 和哈希。
- 用 [08_PROJECT_STRUCTURE_INDEX.md](08_PROJECT_STRUCTURE_INDEX.md) 标注每类目录的用途、权威等级和是否允许进入主稿。
- 新建一份合并且可审计的 BibTeX，不覆盖旧的两份参考文献文件。

这种结构把“可写作材料”和“大体积实验资产”分离，同时保留全部可追溯路径。

## 6. 工作树保护记录

开始本轮前，仓库已有用户改动：

- 修改：`docs/submission_reproducibility_20260826/VERSIONED_EVIDENCE.sha256`
- 未跟踪：`docs/PROJECT_PROGRESS_AND_MANUSCRIPT_PLAN_FOR_SUPERVISOR_EN_20260827.md`

本轮不覆盖、不还原这两项；所有新增内容集中在本目录。

