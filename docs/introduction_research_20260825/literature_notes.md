# Literature notes for Introduction

检索日期：2026-08-25。优先保存官方会议页面、期刊/出版社页面或作者官方论文页面；以下链接用于后续核对和引用。

## 1. 工业异常检测问题与基准

### MVTec AD

Bergmann et al., “MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019.

该工作建立了面向真实工业物体与纹理的异常检测基准，包含高分辨率图像、正常训练样本、异常测试样本和像素级异常区域。Introduction 可用它说明：缺陷检测不是普通 closed-set classification；异常样本稀少且缺陷类型难以穷举，因此 one-class/normal-only 设定具有实际价值。

官方页面：https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html

### VisA

Zou et al., “SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation,” ECCV 2022.

该工作发布 VisA，覆盖 12 个对象、3 个领域，并提供图像级和像素级标注。它强调工业异常的三个困难：异常样本难获得、缺陷可能很小、制造检测同时要求高检测准确率和精细定位。Introduction 可用 VisA 说明需要跨对象、跨领域的验证，而不是只在单一物体类别上报告结果。

官方论文：https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900389.pdf
项目页：https://github.com/amazon-science/spot-diff

## 2. 只用正常样本的视觉 memory-bank 路线

### PatchCore

Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR 2022.

PatchCore 用预训练视觉特征的正常 patch 构建具有代表性的 memory bank，并以查询 patch 到正常 memory 的距离检测异常。它是本项目理解“reference-conditioned normality modeling”的重要前置工作，也为 DINO patch KNN 分支提供清晰的叙事背景。需要注意：PatchCore 的成功不等于任何两个 feature branch 的拼接都有效；本项目需要通过 matched visual-only control 证明拼接的增益。

官方页面：https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html

### AnomalyDINO

Damm et al., “AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2,” WACV 2025.

AnomalyDINO 研究 DINOv2 视觉特征在 one-/few-shot 异常检测中的能力，采用 patch-level nearest-neighbor 思路，强调训练免费和视觉特征本身的强基线价值。对本项目而言，它不是普通 baseline，而是必须保护的强视觉锚点：如果文本分支或融合不能稳定超过它，论文应诚实报告互补性边界。

官方论文：https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf
官方代码：https://github.com/dammsi/AnomalyDINO

## 3. Vision-language anomaly detection 路线

### CLIP

Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021.

CLIP 通过大规模图文对比学习建立视觉—文本共同表示空间，使自然语言可以作为下游类别或状态概念的接口。Introduction 中应把 CLIP 写成“提供可迁移语义先验”，而不是直接等同于细粒度工业缺陷定位器。

官方论文页：https://proceedings.mlr.press/v139/radford21a.html

### WinCLIP / WinCLIP+

Jeong et al., “WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation,” CVPR 2023.

WinCLIP 将 prompt ensemble 与 window/patch/image-level 特征结合，并扩展到 few-normal-shot。它证明 vision-language features 可用于异常分类和定位，但也显示文本相似度、窗口聚合和正常参考信息共同决定效果。它适合作为从 CLIP 走向 industrial anomaly detection 的桥接文献。

官方页面：https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html

### AnomalyCLIP

Zhou et al., “AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection,” ICLR 2024.

AnomalyCLIP 指出普通 CLIP 更容易捕获前景对象语义，而不是跨对象共享的 normality/abnormality；因此学习 object-agnostic prompts，并通过 global/local objectives 强化异常区域识别。对本项目的作用是提供冻结文本分支和异常语义先验；协议上必须把它与 1/2/4-shot 方法区分，因为本项目使用的是其 checkpoint/文本证据，而不是把它重新训练成少样本方法。

官方页面：https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html

### PromptAD

Li et al., “PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection,” CVPR 2024.

PromptAD 针对 one-class few-shot 场景学习 prompt，通过 semantic concatenation 构造 anomaly prompts，并用 explicit anomaly margin 处理没有真实异常训练图的问题。它说明视觉语言路线可以从 zero-shot prompt transfer 进一步走向 target-normal tuning；同时也意味着比较时必须明确是否允许目标类别正常样本参与 prompt learning。本项目中 PromptAD 必须标注 `target_normal_tuning=true`，不能和完全冻结分支无标签混排。

官方页面：https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html

## 4. Introduction 应该如何形成研究缺口

已有工作分别证明了：

1. 正常 patch memory 可以在缺少异常训练样本时进行检测；
2. DINOv2 等视觉 foundation features 可以提供很强的 few-shot 局部表征；
3. CLIP/AnomalyCLIP 可以提供跨对象、跨领域的异常语义先验；
4. PromptAD 可以利用正常样本学习更适合目标类别的 prompt。

但这些方向没有自动推出“视觉距离 + 文本相似度直接融合就会更好”。两类分支的表示空间、分数尺度、局部定位能力和错误模式不同；在 K 很小的情况下，正常参考统计量也不稳定。因此更可防守的缺口是：在严格 normal-only、无测试标签/测试集整体统计量的条件下，研究 reference-conditioned heterogeneous feature fusion 是否带来可重复的跨数据集收益，并与 matched single-branch controls、不同 shot 和多 seed 一起验证。

## 5. 推荐在正文中引用的最小集合

MVTec AD、VisA、PatchCore、CLIP、WinCLIP、AnomalyCLIP、AnomalyDINO、PromptAD 共 8 组核心引用足以搭建 Introduction。综述论文可以补 1 篇，但不要用大量二手综述替代上述原始论文。

## 6. 本轮 2025–2026 更新入口

更完整的最新文献、协议边界和与本项目的逐项关系，见同目录的 `INTRODUCTION_LITERATURE_MASTER_20260826.md`。本轮新增重点包括 SubspaceAD、FastRef、DCP-SFR、ReMP-AD、AA-CLIP、FAPrompt、MoECLIP、FB-CLIP、VisualAD、Real-IAD、Real-IAD D³ 和 MMR-AD。
