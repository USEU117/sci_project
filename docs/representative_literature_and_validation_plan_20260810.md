# 动态融合课题的代表性文献地图与复现验证计划

检索日期：2026-08-10  
适用项目：少样本工业异常检测中的视觉—文本证据动态融合  
当前边界：本文件只做文献筛选、论文定位和实验规划；不启动 GPU，不改动已冻结的 DynamicFusion V1，也不把论文原始数字写入本项目的统一结果表。

## 1. 先给出结论

现有研究能够证明本项目的问题有价值，但也说明“把视觉分支和文本分支动态加权”本身已经不足以作为完整创新点。

本项目更合适的论文主线应当收紧为：

> 在只有少量正常参考图、且禁止使用测试标签和测试集整体统计量的条件下，研究异构异常分数为什么会在动态融合中失真，并设计保序校准、超出正常支持范围检测、视觉安全回退以及图像级/像素级分离路由。

这里的关键词含义如下：

- **异构异常分数**：视觉分支输出距离，文本分支输出图文相似度，两者数值含义和分布不同，不能直接相加。
- **保序校准**：把两种分数变到可比较范围时，尽量不改变同一分支原有的高低顺序。
- **支持范围**：少量正常参考图能够说明的正常分数区间。测试分数远超这个区间时，应标记为“超出已知范围”，不能简单当成高置信度。
- **视觉安全回退**：当路由器不能证明文本证据更可靠时，保留强视觉分支，不让融合明显破坏它。
- **分离路由**：图像级判断和像素级定位使用不同权重、不同可靠性特征和不同验收标准。

这个定位与现有工作的主要差别不是“我们也用了动态权重”，而是：

1. 研究对象是两个已经冻结、输出形式不同的异常检测分支；
2. 校准和路由只能使用 K 张正常参考图；
3. 明确禁止测试标签、测试掩码和测试集整体统计量；
4. 把“融合不能伤害强分支”写成安全约束和验收条件；
5. 同时保留 V1 的失败结果，解释错误从校准、置信度到路由的完整传播链。

## 2. 代表性综述：用于搭建论文背景

### 2.1 中文综述

1. 吕承侃等，《图像异常检测研究现状综述》，《自动化学报》，2022，48(6)：1402–1428，DOI：10.16383/j.aas.c200956。
   - 价值：适合说明异常检测只用正常样本建模的基本定义、传统方法与深度学习方法分类，以及小缺陷和复杂背景问题。
   - 写入位置：引言第一段或相关工作开头。
   - 是否复现：不需要。

2. 郭新茹等，《面向低样本的工业图像异常检测综述》，《计算机工程与应用》，2025，61(13)：26–45，DOI：10.3778/j.issn.1002-8331.2408-0230。
   - 价值：与本项目的 1/2/4-shot 设定最贴近，可用于说明低样本工业异常检测的方法分类和公开数据集。
   - 写入位置：引言和少样本相关工作。
   - 是否复现：不需要。

3. 邢鹏等，《工业视觉异常检测：关于架构、模态及学习范式的综述》，《信号处理》，2026，42(2)：257 起，DOI：10.12466/xhcl.2026.02.012。
   - 价值：2026 年中文综述，从架构、2D/3D 模态和学习范式整理最新趋势，可用于说明基础模型和多源信息融合正在成为新方向。
   - 写入位置：引言末段或相关工作总述。
   - 是否复现：不需要。

### 2.2 英文综述

1. Liu et al., “Deep Industrial Image Anomaly Detection: A Survey,” *Machine Intelligence Research*, 2024, 21:104–135.  
   官方页面：https://doi.org/10.1007/s11633-023-1459-z
   - 价值：从监督程度、网络结构、损失、数据集和指标等角度系统整理工业图像异常检测。

2. Li et al., “A Survey of Deep Learning for Industrial Visual Anomaly Detection,” *Artificial Intelligence Review*, 2025, 58:279.  
   官方页面：https://link.springer.com/article/10.1007/s10462-025-11287-7
   - 价值：覆盖近年的无监督、弱监督、自监督等路线，并包含方法选择和部署讨论。

3. Mao et al., “A Survey on Industrial Image Anomaly Detection: Methods, Benchmarks and Rethinks,” *Measurement*, 2025, 256:118377.  
   官方页面：https://www.sciencedirect.com/science/article/abs/pii/S0263224125017361
   - 价值：适合支持数据集、指标和真实工业部署挑战的介绍。

建议：英文 SCI 稿正文主要引用英文综述；中文综述可以保留 1–2 篇，并在英文参考文献中标注 “in Chinese”，不需要为了数量全部加入。

## 3. 视觉少样本路线：决定“融合是否真的有必要”

### 3.1 已有核心基线

- **PatchCore，CVPR 2022**：使用预训练视觉特征和正常 patch 记忆库，是传统视觉参考库方法的代表。当前项目已完成 MVTec 和 VisA 的 1/2/4-shot、3-seed 统一复现。
- **WinCLIP/WinCLIP+，CVPR 2023**：把 CLIP 的图文知识和少量正常图参考结合起来，是视觉语言少样本异常检测的早期代表。当前项目的 WinCLIP+ 矩阵已完成。
- **AnomalyDINO，WACV 2025**：使用 DINOv2 patch 特征构建强少样本视觉分支。它是本项目当前最强的已完成视觉基线，也是融合必须保护的默认分支。

### 3.2 建议补入论文并优先考虑 Gate A 的方法

1. **FastRecon，ICCV 2023**  
   官方页面：https://openaccess.thecvf.com/content/ICCV2023/html/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.html
   - 核心：从少量正常支持图到查询图进行快速特征重建。
   - 论文作用：作为 DINOv2 基础模型之前的少样本视觉重建代表。
   - 复现建议：只引用即可；资源足够时再做 Gate A，优先级低于 SubspaceAD。

2. **UniVAD，CVPR 2025**  
   官方页面：https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html
   - 核心：训练免费、统一处理工业/逻辑/医学异常，使用 CLIP 与 DINOv2 等基础特征。
   - 论文作用：说明训练免费基础模型方法正在成为重要路线。
   - 复现建议：进入相关工作；是否 Gate A 取决于官方权重、数据协议和 6 GB 显存。

3. **SubspaceAD，CVPR 2026**  
   官方论文：https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html  
   官方代码：https://github.com/CLendering/SubspaceAD
   - 核心：冻结 DINOv2，用 K 张正常图的 patch 特征拟合 PCA 正常子空间，以重建残差作为异常分数。
   - 与本项目关系：它直接挑战“复杂视觉—文本融合是否必要”。如果一个简单视觉子空间已经很强，融合论文必须证明自己提供了额外、稳定且可解释的收益。
   - 复现建议：**新增方法中最高优先级 Gate A**。它训练免费、代码公开，最适合作为 2026 强视觉对比，但需检查官方 DINOv2 规模能否在 6 GB 显存下推理。

4. **FastRef，CVPR 2026**  
   官方论文：https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html  
   官方代码：https://github.com/liyufei25/FastRef
   - 核心：推理时用当前查询图特征更新正常原型，并用最优传输抑制异常信息被吸收到原型中；可接到 PatchCore、WinCLIP 和 AnomalyDINO 上。
   - 与本项目关系：与现有三条基线直接对应，也迫使论文把“单图查询自适应”和“使用整个测试集统计量”区分开。
   - 复现建议：**高优先级 Gate A**，优先测试 AnomalyDINO+FastRef。记录 `query_adaptation=true`，确认每张图独立处理，不使用测试批次整体统计量。

5. **DCP-SFR，CVPR 2026**  
   官方页面：https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html
   - 核心：保护早期网络层的细小缺陷线索，再做结构特征细化。
   - 与本项目关系：说明像素级问题有时应在上游特征中解决，而不是在末端分数上融合。
   - 复现建议：先引用，只有代码、显存和协议都合适时再 Gate A。

## 4. 视觉语言和多模态路线：决定文本分支该如何改进

1. **AnomalyCLIP，ICLR 2024**
   - 核心：学习与具体物体类别无关的正常/异常提示，是当前 DynamicFusion 的冻结文本分支。
   - 项目含义：当前文本热图扩散、背景激活和局部细节不足，既可能是路由问题，也可能是文本分支上限问题。

2. **PromptAD，CVPR 2024**  
   官方页面：https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html
   - 核心：只用目标类别正常样本学习提示。
   - 项目含义：可比较，但必须继续标记 `target_normal_tuning=true`，不能和完全冻结方法混成同一种协议。

3. **AdaCLIP，ECCV 2024**  
   官方页面：https://arxiv.org/abs/2407.15795
   - 核心：结合静态提示和随图像变化的动态提示，并依赖辅助源域训练。
   - 项目含义：支持“固定文本提示不足”的判断，也要求论文明确 `base_dataset_training`。
   - 复现建议：引用为主；不要与 AAAI 2026 的 AdaptCLIP 混淆。

4. **VCP-CLIP，ECCV 2024**  
   官方页面：https://eccv.ecva.net/virtual/2024/poster/1169
   - 核心：从图像的全局和细粒度视觉上下文生成提示，调整文本表示。
   - 项目含义：可用于说明“文本证据也可以由当前图像动态调整”，但这属于提示适配，不等于本项目的分支可靠性路由。

5. **FiLo，ACM Multimedia 2024**  
   DOI：https://doi.org/10.1145/3664647.3680685
   - 核心：生成细粒度异常描述，并加强多尺度、多形状定位。
   - 项目含义：支持“泛化的 abnormal 文本太粗，难以定位具体工业缺陷”的失败分析。
   - 复现建议：相关工作和讨论引用即可。

6. **AA-CLIP，CVPR 2025**  
   官方页面：https://openaccess.thecvf.com/content/CVPR2025/html/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.html
   - 核心：建立异常感知文本锚点，并把 patch 视觉特征对齐到这些锚点。
   - 复现建议：若后续要判断“V1 是路由坏，还是 AnomalyCLIP 文本分支太弱”，可把 AA-CLIP 作为一个**文本分支替换 Gate A**，不必立刻跑完整 1/2/4-shot 矩阵。

7. **ReMP-AD，ICCV 2025**  
   官方页面：https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html
   - 核心：类内 token 检索减少正常记忆噪声，并融合视觉—语言先验。
   - 与本项目关系：是 2025 年最接近的少样本多模态方法之一，但它主要在特征/token 层融合，本项目在冻结输出和可靠性层路由。
   - 复现建议：保留原计划，完成 manifest、NPZ 适配和 bottle Gate A 后再决定是否扩展。

8. **AdaptCLIP，AAAI 2026**  
   官方页面：https://ojs.aaai.org/index.php/AAAI/article/view/42404
   - 核心：通过视觉、文本和 prompt-query 轻量适配器让 CLIP 支持通用异常检测。
   - 复现建议：保留原计划。先核对源域训练、官方 checkpoint、batch size 1 和 6 GB 显存，再允许进入正式矩阵。

9. **AnoPLe，CVPR 2026**  
   官方页面：https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Bidirectional_Multimodal_Prompt_Learning_with_Scale-Aware_Training_for_Few-Shot_Multi-Class_CVPR_2026_paper.html  
   官方代码：https://github.com/YoojLee/AnoPLe
   - 核心：视觉提示和文本提示双向交互，并用全局/局部多尺度训练连接图像级与像素级任务。
   - 与本项目关系：是“视觉—文本 + 图像/像素分离”的强近邻工作。
   - 复现建议：必须引用；正式复现排在 ReMP-AD、AdaptCLIP、SubspaceAD 和 FastRef 之后。

10. **DLVP-CLIP，CVPR 2026**  
    官方页面：https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_DLVP-CLIP_Enhancing_Fine-Grained_Zero-Shot_Anomaly_Detection_via_Dynamic_Local_Visual_CVPR_2026_paper.html
    - 核心：从关键局部区域提取视觉 prompt，同时用高低频分解加强纹理和结构细节。
    - 与本项目关系：进一步说明文本分支局部感知不足是公开问题，也提醒我们不能把所有像素失败都归因于路由器。
    - 复现建议：零样本相关工作引用；如需要升级文本分支，可与 AA-CLIP 二选一做小规模 Gate A。

11. **PAPL，Pattern Recognition 2026，在线优先**  
    官方页面：https://www.sciencedirect.com/science/article/pii/S0031320326004553
    - 核心：把 prompt 学习建模为分布推断，同时层次化融合 CLIP 与 DINO 特征。
    - 与本项目关系：与“CLIP + DINO”组合非常接近，是投论文前必须正面讨论的重叠工作。
    - 差别：PAPL 是上游特征/提示联合建模的零样本方法；本项目是少样本正常参考下、冻结分支输出的安全校准和路由。

12. **TGRF-CLIP，Expert Systems with Applications 2026，在线优先**  
    官方页面：https://www.sciencedirect.com/science/article/pii/S0957417426017306
    - 核心：两阶段训练文本适配器和多层视觉残差，并由文本语义引导视觉残差融合。
    - 与本项目关系：标题和方向都很接近，必须在相关工作中明确区分“训练式特征融合”与“冻结预测的正常样本安全路由”。
    - 复现建议：先核对全文、训练数据和最终出版元数据；如果没有公开代码，不列为近期复现承诺。

## 5. 动态融合、校准和路由：直接解释 V1 为什么失败

1. **Multimodal Dynamics，CVPR 2022**  
   官方页面：https://openaccess.thecvf.com/content/CVPR2022/html/Han_Multimodal_Dynamics_Dynamical_Fusion_for_Trustworthy_Multimodal_Classification_CVPR_2022_paper.html
   - 核心：按样本估计模态和特征的信息量，再动态融合。
   - 对本项目的启发：动态融合的前提是“可靠性估计真的与错误有关”，不能只看权重是否变化。

2. **Provable Dynamic Fusion / QMF，ICML 2023**  
   官方页面：https://proceedings.mlr.press/v202/zhang23ar.html
   - 核心：从理论上说明，动态权重只有在与各模态泛化误差负相关时，才可能稳定优于静态融合；论文用能量不确定性描述模态质量。
   - 对本项目的启发：当前路由权重与真实分支优势的 Spearman 相关性很弱，因此 V1 不满足动态融合成功所需的关键条件。这篇应成为方法动机和失败讨论的核心引用。

3. **PGAD，Pattern Recognition 2026**  
   官方页面：https://www.sciencedirect.com/science/article/pii/S0031320325014220
   - 核心：把全局和局部分数分开建模，并用基于直方图的方式融合不同分布的异常分数。
   - 对本项目的启发：不同分布的异常分数不能直接线性相加；这是 V1 校准问题的直接工业异常检测证据。

4. **Calibrated Feature Fusion，Sensors 2026**  
   官方页面：https://www.mdpi.com/1424-8220/26/7/2164
   - 核心：在融合不同阶段特征前先做表示对齐，避免尺度和语义层级不一致。
   - 对本项目的启发：虽然它做的是“网络阶段特征”而不是“视觉/文本分数”，但共同原则是：先对齐，再融合；对齐失败时，后面的权重设计无法补救。

5. **MoECLIP，CVPR 2026**  
   官方页面：https://openaccess.thecvf.com/content/CVPR2026/html/Park_MoECLIP_Patch-Specialized_Experts_for_Zero-shot_Anomaly_Detection_CVPR_2026_paper.html
   - 核心：把 patch 动态路由到不同的 LoRA 专家。
   - 对本项目的启发：可作为“路由”相关工作，但路由对象不同。MoECLIP 路由的是 CLIP 内部专家，本项目路由的是冻结视觉/文本异常证据。

6. **CADET，IJCAI 2022**  
   官方页面：https://www.ijcai.org/proceedings/2022/278
   - 核心：对未知分布的异常分数做后处理校准，并用 conformal prediction 思路提供更清楚的概率解释和误报控制。
   - 对本项目的启发：异常分数校准应优先考虑正常参考分布和统计意义，而不是把任意 sigmoid 输出直接解释为概率。

7. **Instance-Wise Monotonic Calibration，UAI 2025**  
   官方页面：https://proceedings.mlr.press/v286/zhang25c.html
   - 核心：通过受约束变换保证实例间排序不被校准破坏。
   - 对本项目的启发：可作为 V2“保序校准”的直接方法学依据。需要注意，普通阶梯式 ECDF 或 isotonic regression 可能产生大量并列值，因此必须额外检查 tie rate 和校准前后 Spearman 相关性。

现有稿中的 Guo et al. 温度缩放、Ovadia et al. 分布偏移不确定性、Kull et al. Dirichlet calibration 仍然保留。新增文献不是替换它们，而是把“分类概率校准”推进到“异常分数校准、排序保护和动态融合成立条件”。

## 6. 信息边界：哪些方法不能直接混在同一张表中

以下方法有参考价值，但使用了不同级别的测试时信息：

- **MuSc，ICLR 2024**：多个未标注测试图之间互相打分，属于 batch/transductive zero-shot。  
  官方页面：https://proceedings.iclr.cc/paper_files/paper/2024/hash/096b1019463f34eb241e87cfce8dfe16-Abstract-Conference.html
- **RareCLIP，ICCV 2025**：在线维护历史测试 patch 的稀有度记忆库，属于在线测试时适应。  
  官方页面：https://openaccess.thecvf.com/content/ICCV2025/html/He_RareCLIP_Rarity-aware_Online_Zero-shot_Industrial_Anomaly_Detection_ICCV_2025_paper.html
- **DNPR，Expert Systems with Applications 2026**：在测试流中动态更新正常原型和双记忆库。  
  官方页面：https://www.sciencedirect.com/science/article/pii/S0957417426002447
- **FastRef，CVPR 2026**：使用当前单张查询图改进原型，但不应使用整个测试集的标签或整体统计量。

建议在论文增加一张“信息使用与公平性”表，至少包含：

| 字段 | 含义 |
|---|---|
| target_normal_tuning | 是否用目标类别正常图更新可学习参数 |
| base_dataset_training | 是否在其他工业/医学数据集上训练 |
| single_query_adaptation | 是否只用当前查询图做独立调整 |
| test_batch_statistics | 是否使用多个测试图的整体统计量 |
| online_test_memory | 是否把历史测试图写入记忆库 |
| test_labels_or_masks | 是否使用测试标签或掩码；正式方法必须为 false |

这样可以防止零样本、少样本、源域训练、目标正常微调、单图自适应和 transductive 方法被误认为完全相同的实验条件。

## 7. 应当怎样融合进当前英文论文

### 7.1 引言

建议把引言改为四层逻辑：

1. 工业异常检测通常缺少异常样本，引用 MVTec、VisA 和 2024/2025 综述。
2. 少样本视觉基础模型已很强，引用 WinCLIP、AnomalyDINO、UniVAD、SubspaceAD、FastRef。
3. 视觉语言方法带来语义补充，但局部细节、背景干扰和协议依赖仍存在，引用 AnomalyCLIP、PromptAD、ReMP-AD、AdaptCLIP、AnoPLe、DLVP-CLIP。
4. 真正未解决的问题不是“有没有融合”，而是不同异常分数怎样在无泄漏条件下可靠校准，以及动态权重怎样避免伤害强分支，引用 QMF、PGAD、CADET 和单调校准。

引言中的创新点不要写“首次动态融合视觉和文本”，因为 MoECLIP、PAPL、TGRF-CLIP、ReMP-AD 等已经覆盖相近概念。更稳妥的表述是：

- 建立正常参考限定、可审计的信息边界；
- 识别并量化少样本 MAD 校准饱和造成的排序坍塌；
- 提出或验证保序、超范围感知、视觉回退的安全路由；
- 分开评价图像排序与像素定位，并保留负结果。

### 7.2 Related Work

建议从当前较短的相关工作扩成四个小节：

1. **Few-shot visual anomaly detection**：PatchCore、FastRecon、WinCLIP+、AnomalyDINO、UniVAD、SubspaceAD、FastRef。
2. **Vision-language anomaly adaptation**：AnomalyCLIP、PromptAD、AdaCLIP、VCP-CLIP、FiLo、AA-CLIP、ReMP-AD、AdaptCLIP、AnoPLe、DLVP-CLIP、PAPL、TGRF-CLIP。
3. **Dynamic and uncertainty-aware fusion**：Multimodal Dynamics、QMF、MoECLIP、PGAD、CFF。
4. **Calibration and evaluation boundaries**：Guo、Ovadia、Kull、CADET、单调校准，以及 MuSc/RareCLIP/DNPR 的 transductive 或在线设定。

### 7.3 方法部分

- 在“校准”小节说明：V1 的 median/MAD + sigmoid 是被评估的第一代实现，不把 sigmoid 输出直接称为真实概率。
- 在 V2 中把保序性写成明确条件：同一分支校准前后排名相关性应接近 1，tie rate 受控，校准不得只靠极小 MAD 放大分数。
- 把 QMF 的思想转成可测试要求：路由权重应与分支相对优势有统计相关性。
- 把 PGAD/CFF 的共同原则写成“先对齐分数或表示，再决定如何融合”。
- 把 FastRef、MuSc、RareCLIP、DNPR 放入信息边界对照，说明本方法不使用测试集整体统计量和历史测试记忆。

### 7.4 实验与讨论

新增或强化以下表/图：

1. 方法信息使用和公平性表。
2. V1 校准前后排序、饱和率、并列率和 Spearman 相关性。
3. 动态权重与真实分支优势的相关性；测试标签只用于事后评价，不进入路由器。
4. 强视觉回退触发率、文本接受率、负迁移率。
5. 图像级与像素级路由的独立消融。
6. 与固定权重、视觉单分支、文本单分支和 oracle 上界比较。
7. 成功/失败案例同时展示，不只挑选成功热图。

## 8. 后续复现优先级

### P0：不需要 GPU，先完成

1. 把上述代表文献整理成最终 BibTeX/EndNote，并逐条核对作者、年份、卷期、页码、DOI 和正式/在线优先状态。
2. 把英文稿 19 篇参考文献扩展到约 35–42 篇，但每篇必须服务于一个具体论点。
3. 重写引言和 Related Work，不改动现有实验数字。
4. 从现有冻结 NPZ 完成 V2 前置诊断：原始分数排序、校准后排序、tie rate、饱和率、正常参考支持范围、分支冲突和空间集中度。
5. 写出新的信息使用表和实验注册表字段。

### P1：需要少量 GPU，只做 Gate A

按建议顺序：

1. **ReMP-AD**：因为已在原计划和稿件中承诺，先完成适配与 bottle Gate A。
2. **AdaptCLIP**：因为是 2026 计划基线，完成 checkpoint、batch=1、6 GB 显存 Gate A。
3. **SubspaceAD**：2026 最重要强视觉新增基线；若官方模型不适配 6 GB，可尝试 CPU 特征提取或记录显存阻塞，但不能偷偷换小模型后仍称为官方协议。
4. **FastRef + AnomalyDINO**：直接利用当前已完成的强分支，Gate A 重点审计单图查询适应、额外耗时和统一 NPZ 输出。
5. **AnoPLe**：前四项通过后再做；它可能涉及多尺度训练和模拟异常，资源需求更高。

Gate A 只跑一个类别、一个 shot、一个 seed，检查：结果完整、无标签泄漏、显存、耗时、预测方向、掩码尺寸、样本 ID、官方/统一指标差异。Gate A 未通过的方法不进入完整矩阵。

### P2：只在方法通过 Gate A 后安排完整 GPU 矩阵

最低正式矩阵建议控制在：

- 已完成：PatchCore、WinCLIP+、AnomalyDINO；
- 补完：PromptAD MVTec 剩余 5 组；
- 原计划新增：ReMP-AD、AdaptCLIP；
- 2026 强新增：SubspaceAD；
- 条件新增：FastRef+AnomalyDINO。

这已经是 7–8 条基线加 Ours。AnoPLe、DCP-SFR、UniVAD 不应在没有资源预算的情况下全部承诺为完整矩阵；它们可以先作为高质量引用和 Gate A 备选。

## 9. DynamicFusion V2 的具体验证计划

### 9.1 先解决校准，不先训练复杂路由器

依次比较：

1. 原始 V1 median/MAD + sigmoid；
2. 有尺度下限的 robust calibration；
3. 正常参考分位数/经验分布映射；
4. 严格单调插值和尾部外推，避免阶梯 ECDF 产生大量并列值；
5. 显式超出支持范围特征，而不是把边界饱和当成高置信度。

每个方案必须报告：

- 校准前后 Spearman 相关性；
- 唯一分数比例和 tie rate；
- 分数大于 0.999 或小于 0.001 的饱和率；
- 正常参考范围外的测试比例；
- 视觉单分支 AUROC 是否被校准本身破坏。

### 9.2 再做安全路由

推荐从规则清楚的“视觉默认、文本有条件进入”开始：

- 默认输出视觉分支；
- 只有文本参考视图稳定、文本图空间集中、分支冲突达到条件且视觉处于不确定/超范围状态时，才增加文本权重；
- 图像级要求首先保持排序，像素级要求提高缺陷区域与背景的对比；
- 路由失败或数值异常时自动回退视觉分支。

先做可解释规则，再考虑学习型门控。学习型门控若使用测试标签训练，会直接破坏当前论文的信息边界。

### 9.3 必须重新建立开发/验证边界

MVTec AD 和 VisA 的最终结果已经被查看并用于形成 V2 假设，不能再把它们同时称为全新、独立的 V2 最终验证。

推荐方案：

- 用 **MPDD** 做 V2 开发和参数选择；
- 参数冻结后，用 **BTAD** 做独立小型验证；
- MVTec AD/VisA 只做回顾性对照，明确标注 retrospective/exploratory；
- 如果后续能取得更大数据和算力，再用 Real-IAD 或 MVTec AD 2 做额外外部验证。

若暂时不增加数据集，则 V2 只能作为探索性补充；当前 V1 的冻结验证和失败分析仍然是可发表证据，但不能把再次调参后的 MVTec/VisA 数字称为独立验证。

### 9.4 V2 最小消融矩阵

1. raw visual branch；
2. raw text branch；
3. fixed 0.50 fusion；
4. best fixed weight on development set only；
5. V1 calibration + V1 router；
6. rank-preserving calibration + fixed fusion；
7. rank-preserving calibration + out-of-support flag；
8. 上述方案 + safe visual fallback；
9. image-only router；
10. pixel-only router；
11. complete split image/pixel router；
12. post-hoc oracle branch selector，仅作为上界，不作为可部署方法。

### 9.5 统计和验收标准

- 1/2/4-shot、3 个固定 seed、嵌套 manifest。
- 逐类别计算后做 macro mean，报告 mean ± sample standard deviation。
- 同时报告 Image AUROC/AP/F1-max 和 Pixel AUROC/AP/AUPRO。
- 报告相对最强单分支的负迁移类别比例，而不只报告平均值。
- 对逐类别差值给出 bootstrap 置信区间或配对非参数检验；多重比较时做校正。
- V2 通过的最低条件不是“某一个指标更高”，而是：
  1. 不再出现 V1 那样的大面积校准饱和和排序坍塌；
  2. 图像级不对强视觉分支产生材料性的整体退化；
  3. 像素级收益在独立数据上可重复；
  4. 路由权重与真实分支优势的关系明显强于 V1；
  5. 所有禁止信息字段保持 false。

## 10. 现在最合理的执行顺序

1. 先把文献加入英文初稿，补齐引言、Related Work、方法动机和公平性表。
2. 同时用冻结缓存做 V2 校准与排序诊断，全程 CPU，不占 GPU。
3. 准备 MPDD/BTAD 数据协议和统一 manifest，但在确认许可与存储前不自动下载大数据。
4. 到 GPU 窗口后先完成 PromptAD 剩余 5 组，恢复现有完整性矩阵。
5. 严格按 ReMP-AD → AdaptCLIP → SubspaceAD → FastRef 的顺序做 Gate A。
6. 只让通过 Gate A、协议可公平比较的方法进入长 GPU 队列。
7. V2 在新开发集上冻结后才进入独立验证；不再用已看过的 MVTec/VisA 结果调参。

## 11. 风险与论文备选定位

- 如果 V2 能在不破坏图像排序的前提下稳定提高像素定位：论文可定位为安全、可解释的少样本视觉—文本动态融合。
- 如果 V2 只能做到接近视觉分支但不能持续超越：论文应强调“何时融合、何时不要融合”的风险控制和负迁移分析。
- 如果 V2 仍明显低于视觉分支：保留 V1/V2 失败链条，把工作定位为严格复现、信息泄漏审计和融合失效研究，但需要更强的外部验证和更谨慎的投稿目标。

无论哪种结果，都不应删除当前失败证据，也不应只保留有利类别。
