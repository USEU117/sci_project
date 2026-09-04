# 论文初稿前：在线调研与写作资料准备包

日期：2026-09-03  
范围：当前冻结的 A1 论文主线，不把历史动态融合、显式文本路线或后续失败候选写入主方法  
状态：可供导师/作者审阅；作者已明确以“中科院升级版 SCI 四区”为目标，待最终选刊后可据此生成英文论文 V0.1

---

## 0. 先给结论

在线检索和仓库证据共同支持以下判断：

1. 当前课题应定义为 **normal-only few-shot industrial anomaly detection and localization**，重点是像素级异常定位，同时报告图像级检测结果。
2. A1 是 **冻结双视觉编码器特征融合**：DINOv2 与 AnomalyCLIP 的图像塔；推理时没有文本、没有动态路由、没有可训练参数。
3. 近年已有多项工作将 CLIP 与 DINO/DINOv2 联合使用，包括 Sea-CLIP、PAPL 和 CLIP-DINOv2 DMA/SAP。因此，“首次融合 DINO 与 CLIP”“提出双分支网络”不能作为主要创新。
4. 当前最可辩护的论文问题是：在严格 normal-only、统一参考图、统一记忆库和统一评估器的条件下，加入第二个异构冻结视觉表征，是否对少样本异常定位产生稳定增量；增量在哪些数据集/类别失效；计算代价是多少。
5. 这更适合包装为 **受控实证研究 + 简洁可复现基线**，而不是声称新 SOTA 网络。若坚持使用 “DCFNet” 名称，审稿人很可能追问可学习校准模块、新损失或新网络层在哪里。
6. 当前文献候选库已经从 24 篇扩充为 30 篇，其中 22 篇为 2024--2026 年，占 73.3%，达到导师提出的“近三年至少七成”的暂定比例要求。
7. 作者将目标期刊限定为 **中科院升级版 SCI 四区**。这降低的是选刊门槛，不改变创新事实：当前稿应按“应用型、实证型、可复现”路线组织，不能把基础对齐、归一化或固定拼接改名为新算法模块。

推荐题目：

> **Simple Dual-Encoder Patch Fusion for Normal-Only Few-Shot Industrial Anomaly Detection: Benefits, Limits, and Reproducibility**

备选题目：

> **A Controlled Study of Frozen DINOv2 and CLIP Image Features for Few-Shot Industrial Anomaly Localization**

如果目标期刊不喜欢研究问题或实证研究式题目，可再压缩为：

> **Frozen Dual-Encoder Patch Fusion for Few-Shot Industrial Anomaly Localization**

---

## 1. 哪些内容必须上网搜索

| 检索任务 | 为什么必须查 | 本轮状态 | 写入论文的位置 |
|---|---|---|---|
| 工业异常检测的现实背景与少样本动机 | Introduction 不能只凭常识陈述“缺陷稀少、标注昂贵” | 已用数据集论文和近年方法论文建立证据链 | Introduction 第 1 段 |
| 2024--2026 年研究现状 | 导师要求近三年高质量文献占多数 | 已检索 CVF、ICLR、OpenReview、PMLR、出版社页面 | Introduction 第 2--4 段；Related Work |
| 与 A1 最接近的方法 | 决定创新能说到什么程度 | 已重点核验 Sea-CLIP、PAPL、AnomalyDINO、SubspaceAD、FastRef、ReMP-AD 等 | Introduction gap；Related Work 对比表 |
| DINOv2、CLIP、PatchCore 的原始出处 | 方法基础必须引用原始论文 | 已核验 | Method/Related Work |
| 四个数据集的原始论文、规模和许可 | Dataset 表、Data Availability、图像使用与复现包边界 | 已核验官方页/作者仓库；本地协议统计已生成 | Experiments；Data Availability |
| 指标选择依据 | Pixel-AUROC 在像素极不平衡时可能过于乐观，需要说明主指标为什么是 Pixel-AP | 已找到 WinCLIP 官方补充材料中的明确讨论 | Evaluation Metrics |
| 2026 年新近邻 | 防止用已经拥挤的方向包装创新 | 已加入 ANoCo、DCP-SFR、AnoPLe、VisualAD、MoECLIP 等 | Related Work；Limitations |
| 目标期刊的 scope、模板、字数、声明、图像分辨率 | 决定全文结构和最终排版 | 已完成第一轮候选检索；最终选定后再查作者指南全文 | 投稿格式与最终稿 |
| 中科院分区、是否在目标学科、版面费 | 每年会变化，不能凭旧印象 | 已按“中科院升级版 SCI 四区”初筛；投稿前须用学校认可的中科院分区表终验 | 选刊决策，不写进论文 |
| 每条 BibTeX 的最终 DOI、卷期、页码 | 2026 论文元数据可能继续更新 | 核心条目已核验；投稿前仍需逐条终审 | References |

检索纪律：优先使用 CVF、ICLR Proceedings、OpenReview/TMLR、PMLR、出版社论文页、数据集官方页和作者原始仓库；聚合站只用于发现线索，不作为最终引用源。

---

## 2. 网上检索得到的研究现状

### 2.1 任务背景已经从“全量正常样本”转向更严格的冷启动条件

[MVTec AD](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_CVPR_2019_paper.html) 将工业检测表述为仅以正常图像建模、在测试阶段检测未知异常的任务，并提供像素级标注。其缺陷包含划痕、凹陷、污染和结构变化等多种形态。随后，[VisA](https://github.com/amazon-science/spot-diff)、[MPDD](https://github.com/stepanje/MPDD)、[BTAD/VT-ADL](https://github.com/pankajmishra000/VT-ADL) 扩展了复杂结构、金属部件和真实工业产品等场景。

当前 few-shot 研究通常只给每类 1、2 或 4 张正常参考图。这个设置不是普通“少量有标签缺陷学习”，而是 **只有少量正常参考、未知缺陷在测试时出现**。WinCLIP、PromptAD、AnomalyDINO、FastRef 等都采用或讨论 1/2/4-shot 协议，因此本项目的 shot 设置与当前主流研究问题一致。

### 2.2 现有方法可分为四条主线

#### A. 正常记忆与局部特征匹配

[PaDiM](https://doi.org/10.1007/978-3-030-68799-1_35) 对 patch 特征分布建模；[PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) 用正常 patch 记忆库和最近邻距离实现检测与定位。这条路线的优点是无需异常训练样本、解释直接；不足是性能高度依赖预训练表征以及正常参考是否覆盖外观变化。

A1 的记忆库与 k-NN 评分属于这一谱系，因此不能把“正常 patch 记忆 + 最近邻”写成新方法。

#### B. 冻结视觉基础模型与 training-free few-shot

[DINOv2](https://openreview.net/forum?id=a68SUt6zFt) 提供可迁移的自监督视觉特征。[AnomalyDINO](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) 证明冻结 DINOv2 patch 加最近邻即可形成很强的 few-shot、training-free 方法。[SubspaceAD](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html) 进一步用 PCA 正常子空间代替大记忆库；[ANoCo](https://openaccess.thecvf.com/content/CVPR2026/html/Seo_Anomaly_as_Non-Conformity_via_Training-Free_Graph_Laplacian_Energy_Minimization_CVPR_2026_paper.html) 将异常定义为让查询特征符合正常流形所需的优化代价；[FastRef](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html) 则用查询统计改进少样本原型。

现状判断：2025--2026 年的强邻居说明“方法简单、零训练”本身可以有论文价值，但必须有清晰的新问题、严格对照或新的正常性建模。A1 可以走“受控互补性研究”，却不能只说“我们也用了 DINOv2”。

#### C. CLIP 提示学习和异常语义迁移

[CLIP](https://proceedings.mlr.press/v139/radford21a.html) 通过图文对比预训练建立可迁移表示。[WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html) 用窗口/patch/image 多尺度视觉特征与文本提示完成 zero/few-shot 异常检测；[AnomalyCLIP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html) 学习与具体物体类别无关的正常/异常提示；[PromptAD](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html) 在只有正常样本时学习提示；AA-CLIP、FAPrompt、AnoPLe 等继续发展异常感知、细粒度提示和双向视觉—文本提示。

现状判断：CLIP 路线强调语义先验与跨域泛化，但往往包含提示学习、源域训练、文本推理或适配器。A1 虽使用 AnomalyCLIP checkpoint 的图像塔，但推理时不使用文本，不能归入“本文提出视觉—语言方法”。更准确的说法是：**使用由视觉—语言预训练/异常适配产生的冻结图像特征作为第二种视觉表征**。

#### D. 多表征、CLIP--DINO 联合建模

[Sea-CLIP](https://openaccess.thecvf.com/content/WACV2026/html/Guo_Sea-CLIP_Mining_Semantic-Aware_Representations_for_Few-Shot_Anomaly_Detection_with_CLIP_WACV_2026_paper.html) 已联合 CLIP 与 DINOv2，并使用 patch matching、合成异常、可学习 Anomaly Matching Decoder 和提示训练。[PAPL](https://www.sciencedirect.com/science/article/pii/S0031320326004553) 在零样本框架中结合 CLIP 与 DINO 的层次多尺度特征、粒子提示学习和自适应池化。[CLIP--DINOv2 DMA/SAP](https://www.mdpi.com/2079-9292/14/24/4785) 也使用两类表示、注意力融合和辅助数据训练。[ReMP-AD](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html) 将少量正常参考检索与视觉—语言先验融合。

现状判断：这组文献直接否定“首次组合 CLIP 与 DINO”的主张，但也给 A1 留下一条不同的问题边界：这些方法的最终提升同时受到提示、辅助训练、合成异常和可学习融合器影响，难以单独回答第二个冻结图像表征在完全匹配的正常记忆管线中是否提供稳定增量。

### 2.3 2026 年研究关注点正在变化

近年的新工作显示，研究焦点已经从“换一个 backbone”转向更具体的问题：

- **正常流形与非符合度**：SubspaceAD、ANoCo；
- **查询或原型适配**：FastRef；
- **结构、部件与逻辑异常**：UniVAD、ObjectCore；
- **缺陷线索在深层网络中的衰减与边界细化**：[DCP-SFR](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html)；
- **多专家和 patch 级动态专门化**：[MoECLIP](https://openaccess.thecvf.com/content/CVPR2026/html/Park_MoECLIP_Patch-Specialized_Experts_for_Zero-shot_Anomaly_Detection_CVPR_2026_paper.html)；
- **去除文本分支、重新检验纯视觉异常表征**：[VisualAD](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VisualAD_Language-Free_Zero-Shot_Anomaly_Detection_via_Vision_Transformer_CVPR_2026_paper.html)；
- **更接近真实生产的大规模、多视角评估**：[Real-IAD](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.html)。

因此，A1 不能依靠“ViT、双分支、对齐、归一化”这些通用技术词撑起创新。它的贡献必须落到 **明确的受控研究问题、冻结验证、失败边界和可复现证据** 上。

---

## 3. 当前项目的冻结事实：写作时只能使用这一版

| 项目 | 投稿口径 |
|---|---|
| 任务 | 每类仅 1/2/4 张正常参考图的工业异常检测与像素级定位 |
| 编码器 A | frozen DINOv2 ViT-B/14；输入短边 448；768-D patch |
| 编码器 B | frozen AnomalyCLIP ViT-L/14@336 image tower；实际输入 518；768-D patch |
| 文本 | 推理时不计算文本 embedding、prompt score 或语言相似度 |
| 对齐 | CLIP patch grid 双线性插值到 DINO grid |
| 融合 | 两分支分别 L2；固定 0.5/0.5；concat 为 1536-D；再整体 L2 |
| 正常模型 | 每类 K 张正常参考图构建 FAISS IndexFlatL2 memory bank |
| 异常评分 | patch k=1 最近邻距离 / 2 |
| 后处理 | Gaussian sigma=4；上采样到 448×448；stride=8 评估 |
| matched control | 相同参考图、相同评分与评估器，只移除 CLIP 图像分支 |
| 数据角色 | MPDD development；BTAD/MVTec external frozen validation；VisA in-domain frozen validation |
| 训练 | 0 个可训练参数；但仍需建立数据依赖的正常记忆库 |

禁止复活的旧说法：多模态动态融合、文本分支、语言证据、uncertainty router、1152-D concat、所有数据集/指标/类别均提升、全面 SOTA、无额外计算开销。

---

## 4. 可直接用于论文的数据集资料

### 4.1 官方资料

- MVTec AD：15 个物体/纹理类别，超过 5000 张高分辨率图像，测试异常有像素级标注；官方许可为 CC BY-NC-SA 4.0。来源：[官方数据页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)和[CVPR 2019 论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_CVPR_2019_paper.html)。
- VisA：12 类、10,821 张图，其中 9,621 张正常、1,200 张异常，提供图像级与像素级标注；许可为 CC BY 4.0。来源：[AWS Open Data](https://registry.opendata.aws/visa/)和[作者仓库](https://github.com/amazon-science/spot-diff)。
- MPDD：面向工业金属部件，超过 1000 张图并提供像素级缺陷掩码；来源：[作者仓库](https://github.com/stepanje/MPDD)及 ICUMT 2021 论文 DOI `10.1109/ICUMT54235.2021.9631567`。
- BTAD：三个工业产品，异常图有人工像素级掩码；作者仓库将数据标为 CC-BY-SA。来源：[VT-ADL 作者仓库](https://github.com/pankajmishra000/VT-ADL)及论文 DOI `10.1109/ISIE45552.2021.9576231`。

### 4.2 本项目实际使用的本地协议统计

以下数字从当前仓库的数据根目录和 VisA 官方 `1cls.csv` 重新统计，表示本项目管线可见的正常参考池与测试集，不应和官网宣传总数混写。

| Dataset | Categories | Normal reference pool | Test normal | Test anomalous | Role in this paper |
|---|---:|---:|---:|---:|---|
| MPDD | 6 | 888 | 176 | 282 | development |
| BTAD | 3 | 1,799 | 451 | 290 | external frozen validation |
| VisA | 12 | 8,659 | 962 | 1,200 | in-domain frozen validation |
| MVTec AD | 15 | 3,629 | 467 | 1,258 | external frozen validation |

写作时需要说明：每个 seed/shot 只从 normal reference pool 中选 K 张图建库，完整 pool 不是同时用于 A1 推理。

---

## 5. 可直接用于 Introduction 的证据链

### 第 1 段：工业场景与问题重要性

要写的核心：工业异常往往细小、多样、不可穷举；异常样本及像素级标注难以提前获得；因此 normal-only 建模有现实意义。

建议引用：MVTec AD、MPDD、BTAD，最多 2--3 篇，不要一段塞满四个数据集。

### 第 2 段：为什么进一步需要 few-shot

要写的核心：传统 normal-only 方法仍常假设有较多正常图；新产线、新型号和快速换产时，连正常参考也可能只有 1/2/4 张。few-shot 的难点是正常分布覆盖不足，姿态、纹理、光照和结构变化容易被误判为异常。

建议引用：PromptAD 2024、AnomalyDINO 2025、FastRef 2026，体现问题从提示学习、纯视觉特征到原型改进的演化。

### 第 3 段：第一类解决方案——冻结视觉特征与正常记忆

要写的核心：PatchCore 奠定 patch memory；DINOv2 改善 dense representation；AnomalyDINO、SubspaceAD 和 ANoCo 表明简单 training-free 正常建模仍有竞争力。

需要加的评价：单一表征未必对纹理、结构、细小缺陷和不同产品类别都同样适用；这是研究第二表征增量的动机，但“互补”在未做因果验证前应写成待检验假设。

### 第 4 段：第二类解决方案——CLIP 与多表征融合

要写的核心：WinCLIP、AnomalyCLIP、PromptAD 证明视觉—语言预训练可提供异常先验；Sea-CLIP、PAPL 等已经结合 CLIP 与 DINO。

必须主动承认：encoder pairing 已存在。然后说明近邻方法通常同时引入提示学习、源域训练、合成异常、decoder 或自适应机制，因此不能直接隔离第二种冻结视觉表示本身的效果。

### 第 5 段：研究缺口和本文问题

可直接采用的中心句：

> Existing methods demonstrate the effectiveness of DINO-based visual features, CLIP-based anomaly priors, and learned CLIP--DINO integration. However, it remains insufficiently isolated whether a second frozen image representation provides stable localization gains when the normal references, memory construction, scoring rule, post-processing, and evaluator are all held fixed.

随后提出三个问题：

1. 增益在不同数据集、shot 和 seed 上是否方向稳定？
2. 哪些类别会发生负迁移，说明固定融合的边界在哪里？
3. 第二个编码器带来多少延迟、内存和部署成本？

### 第 6 段：方法概述和结果预告

方法只用 4--5 句：两冻结图像塔、patch grid 对齐、branch-wise L2、固定等权 concat、正常 memory k-NN。

结果预告可写：相对 matched DINO-only，四个数据集的 mean Pixel-AP 分别提高 0.0258、0.0249、0.0524 和 0.0320；四数据集共 36 个 seed/shot 配置的 dataset-level Pixel-AP 差值均为正。必须同段说明 BTAD 的 Image-AP 与 Image-F1-max 下降，并存在 MVTec leather、VisA chewinggum 等持续负迁移类别。

### 第 7 段：贡献列表

建议只列三点：

1. **Matched controlled study**：在统一 normal-memory 管线中隔离第二种冻结视觉表征的增量；
2. **Cross-dataset benefit-and-boundary analysis**：四数据集、36 配置、完整六指标、bootstrap、类别负迁移；
3. **Reproducibility and deployment evidence**：精确方法规格、split/checkpoint hash、重建脚本、泄漏检查和效率测量。

不要把“bilinear alignment”“L2 normalization”“0.5/0.5 concat”分别拆成三个创新点；这些是实现细节，不是独立学术贡献。

---

## 6. Related Work 建议结构

不要简单按年份流水账。建议按问题机制组织成四小节，并在每节末尾加入自己的评价。

### 6.1 Normal-only representation and memory modeling

PaDiM、PatchCore、AnomalyDINO、SubspaceAD、ANoCo。

评论落点：从 CNN patch 分布/记忆发展到 foundation feature、子空间和正常流形；A1 仍采用最透明的 matched k-NN 管线，以便隔离 representation effect。

### 6.2 Vision-language and prompt-based anomaly detection

CLIP、WinCLIP、AnomalyCLIP、PromptAD、AA-CLIP、FAPrompt、AnoPLe、VisualAD。

评论落点：文本/提示是强先验，但协议往往包含辅助训练或提示优化；A1 只使用冻结图像塔，不宣称视觉—语言推理。

### 6.3 Few-shot reference reasoning and adaptation

InCTRL、UniVAD、ReMP-AD、FastRef、DCP-SFR。

评论落点：这些方法对参考检索、部件结构、查询适配或缺陷线索进行更强建模；与 A1 的固定、零训练研究问题不同，结果不可在不说明协议差异时直接横向定胜负。

### 6.4 Multi-representation integration and novelty boundary

Sea-CLIP、PAPL、CLIP--DINOv2 DMA/SAP、MoECLIP。

评论落点：融合两类成熟 backbone 已被覆盖；本文不把 pairing 作为首创，而把 fixed fusion 作为控制变量，研究其收益、失败和成本。

Related Work 的结束句应自然引到本文：

> Rather than introducing another learned fusion module, we use a deliberately fixed fusion rule to isolate representation complementarity under a normal-only few-shot protocol and to expose both positive transfer and failure boundaries.

---

## 7. 30 篇候选文献怎样分配

完整元数据和官方链接已进入 `references/curated_references.bib`。建议不是把 30 篇全部塞入 Introduction，而是按下表分工。

### 7.1 Introduction 核心文献（建议 10--12 篇）

| Key | 作用 |
|---|---|
| `bergmann2019mvtec` | 工业 normal-only 任务与像素级标注背景 |
| `roth2022patchcore` | patch memory 基础 |
| `oquab2024dinov2` | DINOv2 表征来源 |
| `zhou2024anomalyclip` | AnomalyCLIP 来源与协议区别 |
| `li2024promptad` | normal-only few-shot prompt learning |
| `damm2025anomalydino` | 纯视觉、training-free few-shot 直接基线 |
| `ma2025rempad` | few-shot 多模态参考融合近邻 |
| `guo2026seaclip` | 最直接的 CLIP+DINOv2 few-shot 近邻 |
| `lendering2026subspacead` | 强简单 training-free 近邻 |
| `li2026fastref` | reference/prototype refinement 近邻 |
| `seo2026anoco` | 正常流形非符合度新趋势 |
| `jiang2026dcpsfr` | defect cue/结构细化新趋势 |

### 7.2 Related Work 主体文献（其余按需引用）

| 类别 | Keys |
|---|---|
| 数据集与经典方法 | `zou2022spot`, `jezek2021mpdd`, `mishra2021vtadl`, `defard2021padim` |
| CLIP 基础与早期 few-shot | `radford2021clip`, `jeong2023winclip` |
| 2024 generalist/合成异常 | `zhang2024realnet`, `zhu2024inctrl`, `wang2024realiad` |
| 2025 unified/CLIP adaptation | `gu2025univad`, `ma2025aaclip`, `zhu2025faprompt`, `jiang2025clipdino` |
| 2026 prompt/fusion/visual-only | `ma2026papl`, `park2026moeclip`, `hou2026visualad`, `hu2026fbclip`, `lee2026anople` |

当前比例：总计 30 篇；2024--2026 年 22 篇；近三年占 73.3%。最终删除未引用条目后必须重新计算，防止比例再次低于 70%。

---

## 8. 创新点结论：老师意见与网上查重后的合并判断

### 8.1 不能作为主要创新的内容

- 使用 DINOv2；
- 使用 AnomalyCLIP/CLIP 图像塔；
- 双分支；
- bilinear patch-grid alignment；
- 分支 L2 normalization；
- concat 或固定加权；
- patch memory 与最近邻评分；
- “冻结 backbone 所以复杂度低”——冻结只减少训练成本，不会自动消除双编码器推理开销。

### 8.2 当前可以防守的贡献

- **控制变量设计**：所有条件相同，只增加第二种冻结表示；
- **跨数据集稳定性证据**：四个数据集、三 seed、三 shot；
- **主动报告失败边界**：固定融合不是所有类别/指标都提高；
- **计算与复现透明度**：第二分支成本、哈希、泄漏审计、可重建结果；
- **负结果的科学意义**：动态路由和复杂候选未稳定超过固定方案，但这些更适合 Discussion/Appendix，不宜喧宾夺主。

### 8.3 创新强度的诚实评级

| 维度 | 当前判断 |
|---|---|
| 新网络结构 | 弱 |
| 新训练目标/损失 | 无 |
| 新表示融合机制 | 弱；主要是固定对齐拼接 |
| 实验设计与证据完整性 | 强 |
| 跨数据集和 seed/shot 稳定性 | 较强 |
| 失败边界与部署代价分析 | 较强 |
| 适合论文类型 | 应用型、实证型、可复现性强调的 SCI 论文 |

这意味着论文仍然可以写，但题目、摘要和贡献不能假装成高复杂度新网络。若导师坚持“必须有明确算法模块创新”，则需要另立新实验任务；不能只靠重新命名把 A1 变成 DCFNet。

---

## 9. 主图需要怎样准备

仓库已有 Fig01 内容草图及 SVG/PDF/PNG，但最终提交版仍建议按老师要求在 PPT、Visio 或 draw.io 中人工重绘并保留可编辑源文件。

现有图还应做一个关键修正：**正常参考图建库流程和 query 检索流程需要分开画**。不能让读者误以为 query 与 normal references 一起直接进入同一个 memory 构建框。

建议版式：

```text
Normal references (K=1/2/4)                Query image
       │                                      │
       ├─ frozen DINOv2 ─┐                     ├─ frozen DINOv2 ─┐
       └─ frozen CLIP-image ─ alignment ─┐     └─ frozen CLIP-image ─ alignment ─┐
                                         │                                      │
                           branch L2 + fixed concat              branch L2 + fixed concat
                                         │                                      │
                                normal memory bank  ───── k=1 NN distance ───────┘
                                                                                │
                                                              anomaly map + image score
```

图中必须出现：

- `image tower only; no text at inference`；
- DINO 448 / CLIP 518；
- 768-D + 768-D = 1536-D；
- CLIP grid resized to DINO grid；
- branch-wise L2、fixed 0.5/0.5 concat、global L2；
- normal support only；
- FAISS k=1、distance/2；
- sigma=4、448×448 map；
- trainable parameters = 0；
- 旁边用虚线小框表示 matched DINO-only control，不要把对照混入主前向路径。

不得出现：文本 prompt 图标、动态 gate/router、learned calibration、multimodal inference。

---

## 10. 当前实验结果可怎样写

### 10.1 主结果

| Dataset | A1 mean Pixel-AP | DINO-only | Delta | Positive seed/shot configs |
|---|---:|---:|---:|---:|
| MPDD | 0.3562 | 0.3304 | +0.0258 | 9/9 |
| BTAD | 0.6455 | 0.6206 | +0.0249 | 9/9 |
| VisA | 0.3725 | 0.3201 | +0.0524 | 9/9 |
| MVTec AD | 0.5546 | 0.5226 | +0.0320 | 9/9 |

安全结论：四个数据集全部 36 个 dataset-level seed/shot 配置的 Pixel-AP 差值为正。

不能推导为：36 个独立数据集、所有类别提高、所有六指标提高、统计显著性已经自动成立。

### 10.2 必须并列报告的限制

- BTAD mean Image-AP 约下降 0.0131；Image-F1-max 约下降 0.0237；
- MVTec leather mean Delta Pixel-AP 约为 -0.0428；
- VisA chewinggum 约为 -0.0386；
- 还存在 MVTec hazelnut、VisA candle、MPDD bracket_brown 等负迁移；
- 当前稳态端到端约 0.4146 秒/图，约 2.412 图/秒；CLIP 图像分支是主要耗时；
- 结果支持“与互补性解释一致”，但不能证明具体缺陷类别下降/提升的因果机制。

### 10.3 指标策略

主指标建议使用 Pixel-AP，同时报告 Pixel-AUROC、AUPRO、Image-AUROC、Image-AP 和 Image-F1-max。WinCLIP 官方补充材料明确提醒，像素级类别极不平衡时 AUROC 可能给出过于乐观的印象；这可用来解释为什么本文把 Pixel-AP 作为主要定位指标，但不能只报 Pixel-AP 隐藏其他指标。

---

## 11. 生成英文论文初稿前仍需作者确认的事项

必须确认：

1. 论文定位是否接受“controlled empirical study”，还是导师明确要求继续增加算法创新；
2. 在四区候选中确定首投顺序；当前优先核验 `Journal of Electronic Imaging`，其次为 `Machine Vision and Applications`；
3. 最终题目采用上面的推荐题目，还是保留 DCFNet 工作名；
4. 作者顺序、单位、通讯作者、ORCID、基金和致谢；
5. 代码/compact reproducibility package 在投稿时公开、录用后公开，还是按需提供；
6. 主文是否放完整六指标，或将部分完整表放 Supplementary Material。

技术上仍需完成但不要求重跑主实验：

- 用 PPT/Visio/draw.io 重绘主图并分开 support/query 两条流；
- 逐条终审 30 篇 BibTeX，补齐仍含 `and others` 的作者列表；
- 选定期刊后再次核对 2026 文献卷期页码；
- 根据期刊模板生成最终数据集表、baseline fairness 表和声明段落；
- 确定公开仓库/归档 URL 或 DOI。

### 11.1 中科院四区目标下的初步选刊判断（2026-09-03）

这里的“四区”专指 **中科院期刊分区表升级版**，不能与 JCR Q4 混用。分区每年可能调整，而且学校对分区年份、大类/小类、Online/录用时间的认定规则可能不同，因此以下仅作为选刊线索；正式投稿前必须以学校认可的中科院分区表或机构订阅数据库复核。

| 候选期刊 | 与本文匹配度 | 当前分区线索 | 费用/风险 | 初步建议 |
|---|---|---|---|---|
| `Journal of Electronic Imaging` | 高。官方 scope 明列 AI/机器学习、图像分割、industrial 与 material inspection；当前论文的异常定位、工业视觉和图像特征融合均在范围内 | 多个 2025/2026 中文检索源报告为中科院计算机科学 4 区；官方页确认被 SCIE 收录。分区仍需校内数据库终验 | Hybrid，可选非 OA；官方明确预筛方法严谨性、图质量、SOTA 对比和验证充分性，简单融合若证据薄弱会有风险 | **当前首选**。四数据集、36 个 seed/shot 配置、六指标、负迁移和耗时分析正好用于回应其验证要求 |
| `Machine Vision and Applications` | 高。题材上直接对应机器视觉与工业应用 | 2025/2026 中文检索源报告中科院计算机科学 4 区；需要在学校认可数据库中终验 | JCR 分区可能与中科院分区不同，不要因其 JCR Q3 就误判不是中科院 4 区 | **当前第二候选**。更适合将文章强调为工业机器视觉应用与系统性实验 |
| `IET Image Processing` | 中高。官方 scope 覆盖图像处理、分割、纹理、架构和创新应用，SCIE 收录 | 本轮没有获得足够可靠的中科院 4 区证据，暂不列入“已确认四区” | 全 OA；官方当前 APC 为 USD 2,800 / GBP 2,190 / EUR 2,550 | 仅作备选；先核验中科院分区并确认费用可接受再考虑 |

`Journal of Electronic Imaging` 官方页面还明确把低质量图、缺少 SOTA 对比、方法不严谨和验证边缘化列为预筛问题。因此，目标定为四区并不意味着只写“两个分支 1:1 融合”即可；稿件仍必须把控制变量、公平基线、跨数据集稳定性、失败类别、计算成本与复现材料做完整。

---

## 12. 下一轮生成论文时的建议交付范围

作者确认定位后，建议先生成一个不绑定期刊模板的英文 V0.1，包含：

1. Title；
2. Abstract（约 180--220 词，背景--困难--方法--实验--结论）；
3. Keywords；
4. Introduction（按第 5 节七段证据链）；
5. Related Work（按第 6 节四类组织）；
6. Method：Problem Definition、Overview、Feature Extraction、Grid Alignment and Fixed Fusion、Normal Memory and Scoring；
7. Experiments：Datasets、Protocol、Metrics、Baselines、Implementation；
8. Results：Matched Control、Shot/Seed Stability、Failure Boundaries、Efficiency；
9. Discussion and Limitations；
10. Conclusion；
11. 暂定图表位置与 BibTeX 引用键。

写作顺序建议是先把 Method、Experiments 和 Results 的事实段锁定，再完成 Introduction/Related Work，最后写 Abstract 和 Conclusion。这样可避免摘要与引言沿用旧版动态融合叙事。

---

## 13. 本轮已经准备好的文件

- 本在线调研与写作准备包：`23_ONLINE_RESEARCH_AND_DRAFT_PREPARATION_CN_20260903.md`
- 当前 30 篇候选文献库：`references/curated_references.bib`
- 引用核验状态：`references/REFERENCE_AUDIT.md`
- 冻结方法规格：`../../submission_repro_20260827/METHOD_SPEC_V2.md`
- 当前英文 Introduction 工作稿：`03_INTRODUCTION_WORKING_DRAFT_EN.md`
- Related Work 与创新边界：`04_RELATED_WORK_AND_NOVELTY_MAP.md`
- 主张—证据矩阵：`05_CLAIM_EVIDENCE_MATRIX.md`
- 图表规划：`06_TABLE_FIGURE_PLAN.md`
- 投稿前缺口：`07_MISSING_MATERIALS_AND_CHECKLIST.md`

本轮没有直接生成完整论文，目的是先让作者和导师确认：论文究竟按“受控实证研究”投稿，还是必须再做一个通过验证的新算法模块。这个决定会直接改变题目、摘要、Introduction gap、主图和贡献列表。
