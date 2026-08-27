# Introduction / Related Work Literature Master Pack

更新时间：2026-08-26  
检索范围：截至 2026 年已公开的 CVPR、ICCV、WACV、ICLR、ECCV、AAAI、Pattern Recognition、Expert Systems with Applications 及公开预印本。  
用途：为 SCI-I 论文 Introduction、Related Work、Motivation 和公平性讨论提供统一资料包。

## 0. 与项目已有资料的整合关系

本主包整合并更新以下已有文件：

- `docs/introduction_research_20260825/literature_notes.md`
- `docs/introduction_research_20260825/introduction_outline.md`
- `docs/related_literature_screening_2026_20260810.md`
- `docs/representative_literature_and_validation_plan_20260810.md`
- `docs/sources.md`

原文件全部保留。它们提供更细的历史检索、复现建议和动态融合审查；本文件只作为当前写作入口。

## 1. 当前 Introduction 最重要的论证链

### 1.1 工业异常检测不是普通分类

工业缺陷通常稀有、类别开放、形态多变，且很多缺陷只占据很小的局部区域。MVTec AD 提供了多类工业物体/纹理、正常训练图、异常测试图和像素级标注；VisA 进一步扩大对象与领域覆盖，并强调小缺陷和高精度定位。Introduction 应从“异常样本难以穷举”和“检测必须同时输出是否异常及其位置”开始，而不是先从 CLIP 或某个网络开始。

推荐引用：MVTec AD、VisA。

### 1.2 Few-shot / normal-only 是实际约束

工业生产中通常更容易获得合格品图像，而缺陷类型和缺陷数量不足以支持常规监督学习。因此，给定 K 张正常参考图建立正常性模型，是比大量异常标注更贴近冷启动的设定。PatchCore 用正常 patch memory bank 进行 nearest-neighbor 异常检测，构成经典 reference-based 路线。

推荐引用：PatchCore、WinCLIP+、AnomalyDINO、UniVAD、SubspaceAD、FastRef。

### 1.3 视觉 foundation features 已经很强

DINOv2/AnomalyDINO 说明，强的自监督视觉 patch 表征本身就能在 1/2/4-shot 下提供很强的局部异常证据。2026 年 SubspaceAD 进一步用冻结 DINOv2 特征和正常子空间重建残差完成 training-free few-shot detection，直接提出一个重要问题：当视觉表征已经很强时，复杂的多模态融合是否真的带来可重复的额外收益？

这应成为本项目的强视觉锚点，而不是被融合结果掩盖。

### 1.4 Vision-language 方法提供迁移语义，但不等于细粒度定位

CLIP 通过大规模图文对比学习建立共同空间；WinCLIP 将 prompt ensemble、窗口特征和正常参考用于 zero-/few-normal-shot 异常分类与分割；AnomalyCLIP 用 object-agnostic normality/abnormality prompts 减弱前景对象语义的干扰；PromptAD 则允许在只有正常样本时学习目标类别 prompt。

这些工作共同支持“文本先验有价值”，但也留下一个边界：图文相似度、视觉距离、局部 patch map 的统计意义不同，不能把它们当作同一种概率直接相加。

### 1.5 2025–2026 的最新趋势

最新研究大致出现五条路线：

1. **更强的 training-free 视觉正常性建模**：SubspaceAD 用 PCA 正常子空间；FastRef 在推理阶段细化有限正常原型；DCP-SFR 保护浅层缺陷线索并改善结构特征。
2. **更强的 anomaly-aware CLIP**：AA-CLIP 调整文本和 patch 视觉空间；FAPrompt 学习分解式 fine-grained abnormality prompts；Bayesian Prompt Flow 约束 prompt 分布以改善 zero-shot 泛化。
3. **检索增强和多模态 prompt fusion**：ReMP-AD 用 intra-class token retrieval 减少参考 memory bank 噪声，并用 vision-language prior 引导特征。
4. **动态专家或 patch 级适配**：MoECLIP 将 patch 路由到专门化 LoRA experts；它是“动态路由”相关工作，但路由对象是模型内部专家，不是本项目的视觉/文本异常证据。
5. **回到纯视觉或扩大真实基准**：VisualAD 研究不依赖文本分支的 zero-shot 路线；Real-IAD 和 Real-IAD D³ 通过更多对象、多视角及 RGB/伪 3D/3D 数据缓解 MVTec/VisA 饱和和单视角局限。

## 2. 高相关最新论文清单

### A. 优先写入 Introduction / Related Work

#### SubspaceAD — CVPR 2026

Lendering, Akdag, Bondarau, “SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling.” 冻结 DINOv2，用少量正常 patch 特征拟合 PCA 正常子空间，以重建残差检测异常；不需要 prompt tuning 或 memory bank。它是本项目 A1 的直接强视觉对照，必须用于说明“多模态不是天然必要”。

官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html>

#### FastRef — CVPR 2026

Li et al., “FastRef: Fast Prototype Refinement for Few-shot Industrial Anomaly Detection.” FastRef 在推理阶段用当前查询图像进行 prototype refinement，并用 optimal transport 抑制异常特征被吸收到正常原型中；论文覆盖 MVTec、VisA、MPDD 和 Real-IAD，并 explicitly discusses 1/2/4-shot。它与本项目的 normal reference bank 高度相关，但必须在信息边界中区分“单张查询适配”和“使用测试集整体统计”。

官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html>

#### DCP-SFR — CVPR 2026

Jiang et al., “Defect Cue-Preserved Structural Feature Refinement for Few-Shot Anomaly Detection.” 该工作强调浅层缺陷线索在深层特征提取中可能被削弱，因此通过结构特征细化保留细小缺陷。它对本项目的启发是：像素级失败可能来自上游表征丢失，不一定能由末端 score fusion 修复。

官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html>

#### ReMP-AD — ICCV 2025

Ma et al., “ReMP-AD: Retrieval-enhanced Multi-modal Prompt Fusion for Few-Shot Industrial Visual Anomaly Detection.” ReMP-AD 用 intra-class token retrieval 处理少量参考图的类内变化，并引入 vision-language prior fusion。它是与本项目“正常参考 + 检索/多模态融合”最接近的近期工作之一，但应单独说明其训练、检索和协议与 A1 fixed frozen fusion 不同。

官方页面：<https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html>

#### PGAD — Pattern Recognition 2026

Zhou et al., “One-shot Unsupervised Industrial Anomaly Detection via Global and Local Feature Fusion.” PGAD 的 Histogram-Based Score Fusion 直接关注全局不变特征与多尺度局部 patch 分数来自不同分布时如何融合。这篇期刊论文与本项目最直接的共同问题不是“是否使用多模态”，而是“异构异常分数在融合前是否需要分布对齐”。建议在 Introduction 的最后一段 Related Work 和方法动机中重点引用。

出版社页面：<https://www.sciencedirect.com/science/article/pii/S0031320325014220>  
DOI：<https://doi.org/10.1016/j.patcog.2025.112759>

#### TGRF-CLIP — Expert Systems with Applications 2026

Yan and Xu, “TGRF-CLIP: CLIP-Based Text-Guided Fusion of Visual Residuals for Few-Shot Anomaly Detection.” 该工作从文本引导视觉残差融合，代表近期更接近“视觉 residual + language guidance”的期刊路线。由于其在线优先状态、训练方式和目标域信息使用边界需要投稿前复核，适合放入 Related Work，不宜直接作为同协议数值对比。

出版社页面：<https://www.sciencedirect.com/science/article/pii/S0957417426017306>  
DOI：<https://doi.org/10.1016/j.eswa.2026.132817>

#### UniVAD — CVPR 2025

Gu et al., “UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection.” UniVAD 以统一模型覆盖工业、逻辑和医学异常，代表 training-free foundation-model features 向跨域统一检测发展的方向。它适合放在“从单类工业基线到跨域 foundation model”这一段，但数值不要与本项目不同协议的矩阵直接混排。

官方页面：<https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html>

#### AA-CLIP — CVPR 2025

Ma et al., “AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP.” AA-CLIP 通过 anomaly-aware text anchors、patch-level visual alignment 和 residual adapters 改善 CLIP 的 normal/abnormal discrimination。它可以帮助 Introduction 更准确地描述当前 CLIP 路线的进展：研究重点已从“手写 prompt”转向“异常感知的视觉—文本空间适配”。

官方页面：<https://openaccess.thecvf.com/content/CVPR2025/html/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.html>

#### FAPrompt — ICCV 2025

Zhu et al., “Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection.” FAPrompt 指出 coarse abnormal prompts 难以覆盖多样细粒度异常，提出分解和互补的 abnormality prompts。它适合支撑本项目对 AnomalyCLIP 文本 heatmap 背景激活、局部细节不足和语义粒度限制的讨论。

官方页面：<https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Fine-grained_Abnormality_Prompt_Learning_for_Zero-shot_Anomaly_Detection_ICCV_2025_paper.html>

### B. 需要写入 Related Work，但不要混入同协议主表

#### MoECLIP — CVPR 2026

Park et al., “MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection.” 它将不同 patch 动态路由到 LoRA experts，并用正交/ETF 约束减少专家冗余。可用来说明 2026 年动态路由正在深入 patch-level experts，但必须明确它不是本项目的视觉—文本 evidence routing，也不是 1/2/4-shot normal-reference protocol。

官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Park_MoECLIP_Patch-Specialized_Experts_for_Zero-shot_Anomaly_Detection_CVPR_2026_paper.html>

#### FB-CLIP — CVPR 2026

Hu et al., “FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement.” 该工作从文本多视角表示、前景—背景分离和语义一致性正则化改善复杂背景下的 zero-shot localization。适合用于解释文本分支的背景干扰问题，但不与 A1 的 few-shot 主表直接比较。

官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Hu_FB-CLIP_Fine-Grained_Zero-Shot_Anomaly_Detection_with_Foreground-Background_Disentanglement_CVPR_2026_paper.html>

#### VisualAD — CVPR 2026

Hou et al., “VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer.” VisualAD 使用视觉 token 学习 normality/abnormality，不依赖文本编码器，说明“开放泛化”和“文本分支”并非同义词。它能帮助论文避免把 multimodal 叙事写成唯一正确路线。

官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VisualAD_Language-Free_Zero-Shot_Anomaly_Detection_via_Vision_Transformer_CVPR_2026_paper.html>

#### Real-IAD — CVPR 2024；Real-IAD D³ — CVPR 2025

Real-IAD 提供 30 类、150K 高分辨率、多视角图像，并指出 MVTec/VisA 上的高 AUROC 可能已接近饱和；Real-IAD D³ 进一步扩展到 RGB、伪 3D 和 3D 点云。它们适合放在“benchmark realism and generalization”段落，说明仅凭 MVTec/VisA 的高分不能完全代表生产线泛化。

Real-IAD：<https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.html>  
Real-IAD D³：<https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Real-IAD_D3_A_Real-World_2DPseudo-3D3D_Dataset_for_Industrial_Anomaly_Detection_CVPR_2025_paper.html>

#### MMR-AD — CVPR 2026

Yao et al., “MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with Multimodal Large Language Models.” 这是更偏 general anomaly detection 和 multimodal large language model 的新基准，不是当前 2D few-shot patch fusion 的直接基线，但可在 Introduction 结尾作为“从检测到解释/推理”的新趋势一笔带过。

官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Yao_MMR-AD_A_Large-Scale_Multimodal_Dataset_for_Benchmarking_General_Anomaly_Detection_CVPR_2026_paper.html>

## 3. 研究缺口的更新版

旧资料把缺口重点写成“动态不确定性路由”。结合 2025–2026 新文献后，建议收紧为四层：

1. **表征层**：DINOv2、DCP-SFR、SubspaceAD 表明视觉表征和正常子空间本身已很强；融合不能只凭“多模态”标签宣称有效。
2. **语义层**：AA-CLIP、FAPrompt、FB-CLIP 表明文本分支正在从 coarse prompts 走向 anomaly-aware、fine-grained、foreground-background disentangled semantics。
3. **参考层**：PatchCore、FastRef、ReMP-AD 说明少量正常参考图的选择、检索、原型更新会显著影响结果；K 很小时，reference uncertainty 是核心变量。
4. **验证层**：Real-IAD 系列指出主流 benchmark 可能饱和，因此需要多类别、多个 shot、多个 seed、外部数据集和逐类别负迁移统计，而不是只报一个平均 AUROC。

因此最安全的论文 gap 是：

> Existing studies have substantially improved visual representations, anomaly-aware prompts, and few-shot prototype construction, yet the reproducibility and failure boundary of combining heterogeneous patch-level evidence under a strict normal-only protocol remain insufficiently characterized.

## 4. 建议的 Introduction 结构（更新版）

1. 工业检测需求、稀有缺陷、细粒度定位：MVTec AD + VisA。
2. Normal-only / few-shot 和 normal memory bank：PatchCore。
3. Foundation visual features：DINOv2 + AnomalyDINO + SubspaceAD。
4. Vision-language anomaly detection：CLIP + WinCLIP + AnomalyCLIP + PromptAD。
5. 最新改进：UniVAD、AA-CLIP、FAPrompt、ReMP-AD、FastRef、DCP-SFR。
6. 现实性和公平性：Real-IAD、Real-IAD D³；说明 MVTec/VisA 饱和风险和协议差异。
7. 本文研究问题：DINO patch 与 AnomalyCLIP patch 的 reference-conditioned fixed fusion 是否在严格 1/2/4-shot、3-seed 条件下提供稳定互补？
8. 贡献：统一可审计协议、A1 固定融合、matched controls、跨数据集与失败边界分析。

## 5. 不应写成本文核心贡献的内容

- 已验证的动态路由全面优于强视觉分支；
- 仅凭视觉—文本拼接就能解决文本 heatmap 背景激活；
- 9 个 seed/shot 组合是 9 个独立实验单位；
- 将 zero-shot、target-normal tuning、frozen few-shot 和 transductive test-time adaptation 混成同一协议；
- 用论文原文的最高单点数字替代本项目统一协议下的 mean±std、逐类别和外部验证结果。

## 6. 文章中最值得新增的引用组合

如果篇幅有限，优先加入：SubspaceAD、FastRef、ReMP-AD、AA-CLIP、FAPrompt、Real-IAD、MoECLIP、VisualAD。它们分别覆盖：强视觉挑战、原型更新、多模态检索、异常感知文本、细粒度 prompt、真实 benchmark、动态专家和无文本视觉路线。
