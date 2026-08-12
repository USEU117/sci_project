# 2026 年少样本工业异常检测相关文献筛选

> 补充说明：本文件集中记录 2026 年候选文献。跨年份代表性研究、论文融入位置、Gate A 优先级和 DynamicFusion V2 验证方案见 `docs/representative_literature_and_validation_plan_20260810.md`。

检索日期：2026-08-10  
用途：英文 SCI 初稿的相关工作扩充、方法定位和后续 Gate A 候选筛选。  
原则：优先正式同行评审论文和出版社/会议官方页面；预印本、普通工作坊和不同任务论文不用于凑数量。

## 一、建议优先进入正文并考虑正式复现

### 1. SubspaceAD（CVPR 2026）

- 论文：Camile Lendering, Erkut Akdag, and Egor Bondarau, “SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling,” CVPR 2026, pp. 28557–28566.
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html>
- 方法：用冻结 DINOv2 提取正常图像的 patch 特征，再用 PCA 建立正常子空间；测试 patch 偏离正常子空间的重建残差就是异常分数。
- 关系：与 AnomalyDINO 同属视觉基础模型路线，却比记忆库和动态融合更简单。它会直接挑战“复杂融合是否必要”，因此是本项目最重要的 2026 对比候选之一。
- 使用建议：相关工作必须加入；若官方代码和显存通过 Gate A，优先进入 1/2/4-shot 正式矩阵。

### 2. FastRef（CVPR 2026）

- 论文：Yufei Li et al., “FastRef: Fast Prototype Refinement for Few-shot Industrial Anomaly Detection,” CVPR 2026, pp. 43040–43049.
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html>
- 方法：在推理阶段用查询图像特征更新少量正常原型，再用最优传输抑制“异常特征被正常原型吸收”的问题；论文把 FastRef 接到 PatchCore、WinCLIP 和 AnomalyDINO 上，并测试 1/2/4-shot。
- 关系：与现有三条基线直接对应，也是“怎样利用查询图像但不破坏正常原型”的最新答案。
- 使用建议：相关工作必须加入；正式比较前需要检查它是否使用单张查询还是测试集整体统计量，避免与本项目的信息边界冲突。

### 3. AnoPLe（CVPR 2026）

- 论文：Yujin Lee et al., “Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection,” CVPR 2026, pp. 35577–35586.
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Bidirectional_Multimodal_Prompt_Learning_with_Scale-Aware_Training_for_Few-Shot_Multi-Class_CVPR_2026_paper.html>
- 方法：文本提示和视觉提示双向交互；同时利用全局图像和局部区域训练，使图像级判断与像素级定位更加一致。
- 关系：它与本项目的“视觉—文本融合”和“图像级/像素级需要分开处理”高度相关，是最接近的 2026 多模态少样本方法之一。
- 使用建议：必须进入相关工作；代码、基础训练数据和目标域调参边界通过审计后，再决定是否进入正式矩阵。

### 4. AdaptCLIP（AAAI 2026）

- 论文：Bin-Bin Gao et al., “AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection,” AAAI 2026, vol. 40, no. 6, pp. 4095–4103.
- 官方页面：<https://ojs.aaai.org/index.php/AAAI/article/view/42404>
- 方法：在 CLIP 的输入或输出端加入视觉、文本和 prompt-query 三类轻量适配器；模型在基础数据上训练后，可对新目标域做零样本或少样本检测。
- 关系：它是项目原计划中的核心近期方法，但协议不是简单的目标域冻结推理，需要清楚标明基础训练域和目标域信息使用方式。
- 使用建议：保留现有引用并扩写方法差异；完成 checkpoint 与 Gate A 后才允许进入正式数值比较。

### 5. DCP-SFR（CVPR 2026）

- 论文：Le Jiang et al., “Defect Cue-Preserved Structural Feature Refinement for Few-Shot Anomaly Detection,” CVPR 2026, pp. 35607–35616.
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html>
- 方法：认为异常线索会在深层特征提取过程中逐渐消失，因此先放大早期异常线索，再用结构感知模块改善异常区域和边界定位。
- 关系：对本项目像素级定位结果很重要，可用于解释为什么视觉细节保持可能比后端分数融合更可靠。
- 使用建议：进入少样本视觉方法相关工作；是否进入矩阵取决于官方代码、训练边界和资源需求。

### 6. PGAD（Pattern Recognition 2026）

- 论文：Jiayin Zhou, Wai Keung Wong, and Fangjian Liao, “One-shot Unsupervised Industrial Anomaly Detection: Enhanced Performance under Extreme Data Scarcity,” Pattern Recognition, vol. 173, art. 112759, 2026.
- 官方页面：<https://www.sciencedirect.com/science/article/pii/S0031320325014220>
- DOI：<https://doi.org/10.1016/j.patcog.2025.112759>
- 方法：分别提取全局不变特征和多尺度局部 patch 特征，并用 Histogram-Based Score Fusion 融合不同分布的异常分数。
- 关系：论文明确关注“直接融合不同分布分数会产生错误”，与本项目发现的校准饱和和融合排序破坏高度一致。
- 使用建议：必须在校准/融合相关工作和失败分析中引用；它是支持本项目问题定义的关键期刊文献。

## 二、建议进入相关工作和讨论，但暂不作为同协议数值对比

### 7. MoECLIP（CVPR 2026）

- 论文：Jun Yeong Park et al., “MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection,” CVPR 2026, pp. 35534–35544.
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Park_MoECLIP_Patch-Specialized_Experts_for_Zero-shot_Anomaly_Detection_CVPR_2026_paper.html>
- 方法：将每个图像 patch 动态路由到不同的 LoRA 专家，并用特征正交和 ETF 约束减少专家功能重复。
- 关系：这是 2026 年与“动态路由”概念最接近的论文，但它路由的是 patch 专家，本项目路由的是视觉/文本证据，不能写成同一种方法。
- 使用建议：在动态路由相关工作中重点讨论；由于它是零样本方法，暂不与本项目 1/2/4-shot 表直接混排。

### 8. FB-CLIP（CVPR 2026）

- 论文：Ming Hu et al., “FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement,” CVPR 2026, pp. 35659–35669.
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Hu_FB-CLIP_Fine-Grained_Zero-Shot_Anomaly_Detection_with_Foreground-Background_Disentanglement_CVPR_2026_paper.html>
- 方法：增强文本表示，并从身份、语义和空间等角度分离前景与背景，减少复杂背景干扰。
- 关系：与本项目文本分支热图扩散、背景激活和像素级失败案例直接相关。
- 使用建议：加入视觉语言异常定位和失败讨论；作为零样本方法单独报告。

### 9. VisualAD（CVPR 2026）

- 论文：Yanning Hou et al., “VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer,” CVPR 2026, pp. 21346–21356.
- 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VisualAD_Language-Free_Zero-Shot_Anomaly_Detection_via_Vision_Transformer_CVPR_2026_paper.html>
- 方法：不使用文本编码器，而是在视觉 Transformer 中加入正常/异常 token，并使用空间交叉注意力和特征自对齐完成零样本检测。
- 关系：它客观说明“文本分支并不是实现开放泛化的唯一方式”，可用于支持本项目对文本证据收益边界的谨慎讨论。
- 使用建议：加入讨论，不直接作为少样本矩阵方法。

### 10. DNPR（Expert Systems with Applications 2026）

- 论文：Shuyun Li et al., “DNPR: Zero-shot Industrial Anomaly Detection via Dynamic Normal Prototype Refinement,” Expert Systems with Applications, vol. 312, art. 131331, 2026.
- 官方页面：<https://www.sciencedirect.com/science/article/pii/S0957417426002447>
- DOI：<https://doi.org/10.1016/j.eswa.2026.131331>
- 方法：在测试流中逐步更新正常原型，用双记忆抑制异常污染，并设计纹理感知校准减少纹理类别的假阳性。
- 关系：它同时涉及动态原型、在线信息使用和分数校准，适合与本项目“禁止测试集整体统计”和“正常参考校准失效”进行对照。
- 使用建议：进入校准和信息边界讨论；它属于 transductive zero-shot，不能直接混入冻结少样本主表。

### 11. DSSM-MAPL（Expert Systems with Applications 2026）

- 论文：Wen Lv et al., “Industrial Anomaly Detection via Prompt Learning with Perturbation-Based Selective State Memory Units,” Expert Systems with Applications, vol. 313, art. 131482, 2026.
- 官方页面：<https://www.sciencedirect.com/science/article/pii/S0957417426003957>
- DOI：<https://doi.org/10.1016/j.eswa.2026.131482>
- 方法：计算查询图像与少量正常提示图像之间的上下文残差，通过选择性状态方程增强异常像素，并用多模态记忆单元完成图像—文本提示融合。
- 关系：与本项目“正常参考 + 查询差异 + 视觉语言融合”相关，但它是跨域零样本统一模型。
- 使用建议：进入视觉语言融合相关工作，不作为相同协议数值比较。

## 三、在线优先观察文献

### 12. PAPL（Pattern Recognition 2026，在线优先/2026-10 卷期）

- 论文：Ruichen Ma et al., “PAPL: Particle-Based Adaptive Prompt Learning for Zero-Shot Industrial Anomaly Detection,” Pattern Recognition, vol. 178, art. 113489, 2026.
- 官方页面：<https://www.sciencedirect.com/science/article/pii/S0031320326004553>
- DOI：<https://doi.org/10.1016/j.patcog.2026.113489>
- 方法：用粒子分布而不是单一提示表示正常/异常语义，并融合 CLIP 与 DINO 的多尺度视觉特征。
- 关系：与本项目的 CLIP + DINO 双分支非常接近。
- 使用建议：可以作为在线优先文献加入，但投稿前需要再次核对最终卷期和页码状态；暂不据其论文数字作本地排名。

### 13. TGRF-CLIP（Expert Systems with Applications 2026，在线优先）

- 论文：Hong-Liang Yan and Xin-Shun Xu, “TGRF-CLIP: CLIP-Based Text-Guided Fusion of Visual Residuals for Few-Shot Anomaly Detection,” Expert Systems with Applications, art. 132817, 2026.
- 官方页面：<https://www.sciencedirect.com/science/article/pii/S0957417426017306>
- DOI：<https://doi.org/10.1016/j.eswa.2026.132817>
- 方法：采用解耦的两阶段训练，让文本信息引导视觉残差融合，以兼顾 CLIP 泛化能力和细粒度工业缺陷检测。
- 关系：从题目和摘要看，它是目前与本项目“少样本 + CLIP + 动态/残差融合”最接近的 2026 期刊论文之一。
- 使用建议：应进入相关工作重点对比；但当前属于在线优先，正式写入前需要下载全文核对训练数据、目标域信息使用方式、卷期和页码。

## 四、不建议为了数量直接加入主线的 2026 论文

- RadioCore：CVPR Workshop 2026，任务相关但优先级低于 CVPR 正会的 FastRef/SubspaceAD/DCP-SFR。
- ObjectCore：WACV 2026，主要研究 MVTec LOCO/CAD-SD 的逻辑异常，不是当前 MVTec AD/VisA 主协议。
- Commonality in Few：AAAI 2026，但主要是 MVTec 3D-AD/Eyecandies 多模态 3D 异常，与当前 2D 视觉—文本分支不同。
- 异常生成、3D 点云和视频异常论文：除非新增专门小节，否则不加入以免稀释论文主线。

## 五、建议的文献数量目标

- 当前英文初稿：19 篇。
- 第一批加入上述 2026 核心文献：建议 10–12 篇。
- 再补齐 2024–2025 的 UniVAD、AA-CLIP、RareCLIP、LogSAD、INP-Former、VCP-CLIP、FiLo、AdaCLIP 等基础过渡文献：建议 8–10 篇。
- 最终合理规模：约 37–42 篇。数量服从论证需要，不以堆到 50 篇为目标。

## 六、写入论文时的边界

1. 引用论文不等于完成本地对比；正文必须区分 cited、planned、Gate A passed 和 fully reproduced。
2. 只有使用官方代码、统一 MVTec AD/VisA 清单、1/2/4-shot、3-seed 和统一指标并通过 Gate A 的方法，才能进入正式比较表。
3. Zero-shot、target-normal tuning、base-dataset adaptation、transductive test-time adaptation 必须分别标明。
4. 不直接复制其他论文的数值进入本项目主表，也不根据不完整矩阵形成方法排名。
