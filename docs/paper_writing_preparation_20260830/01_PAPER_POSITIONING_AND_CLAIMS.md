# 01. Paper Positioning and Claims

## 1. 推荐定位

**A controlled empirical study of frozen dual-encoder visual feature complementarity for normal-only few-shot industrial anomaly localization.**

这一定义有意弱化“新网络结构”，强化当前项目真正扎实的部分：统一管线、matched control、跨数据集冻结验证、完整指标、失败边界和复现审计。

## 2. 推荐标题

首选：

> **Do Frozen Visual Encoders Provide Complementary Evidence for Few-Shot Industrial Anomaly Detection? A Controlled Four-Dataset Study**

备选：

> **Simple Dual-Encoder Patch Fusion for Normal-Only Few-Shot Industrial Anomaly Detection: Benefits, Limits, and Reproducibility**

> **A Controlled Study of Frozen DINOv2 and CLIP Image Features for Few-Shot Industrial Anomaly Localization**

如果目标期刊偏工程应用，第二个更直接；如果偏实验研究，首选更诚实也更有问题意识。

## 3. 核心研究问题

> Under a strictly normal-only few-shot protocol, does a fixed, training-free combination of two heterogeneous frozen visual patch representations improve anomaly localization over a matched single-encoder baseline, and where does that complementarity fail?

三个子问题：

1. 收益是否在不同参考图随机种子和 1/2/4-shot 条件下稳定？
2. 收益能否跨开发域、外部冻结验证域和训练域内验证域复现？
3. 哪些类别发生负迁移，固定融合付出了怎样的时间和内存代价？

## 4. 方法简述

输入图像分别送入冻结的 DINOv2 ViT-B/14 与 AnomalyCLIP ViT-L/14@336 图像塔。CLIP patch 网格双线性对齐到 DINO 网格；两个 768 维分支分别 L2 归一化后以固定 0.5/0.5 权重拼接为 1536 维，再整体 L2 归一化。每个类别仅用 K 张正常参考图建立 FAISS `IndexFlatL2` 记忆库；查询 patch 的 k=1 最近邻距离除以 2 形成异常分数，随后平滑并上采样到 448。

matched DINO-only control 除去 CLIP 分支和 concat，其余参考图、KNN、后处理与指标口径一致。

## 5. 推荐贡献表述

以下三点足以支撑一篇定位克制的 SCI 论文：

1. **Controlled evidence.** We isolate the effect of adding a second frozen visual representation through a matched DINO-only control under an otherwise identical normal-memory pipeline.
2. **Cross-dataset robustness and boundary analysis.** We evaluate 36 seed/shot configurations over four industrial datasets, report complete image- and pixel-level metrics, and characterize both consistent pixel-level gains and category-specific negative transfer.
3. **Reproducible, training-free protocol.** We provide a zero-trainable-parameter method specification, compact prediction maps, reconstruction scripts, statistical intervals, efficiency measurements, leakage checks, and versioned evidence.

如果审稿人要求“算法创新”，可补充但不能夸大：

4. **A deliberately minimal fusion baseline.** The branch-wise normalization, spatial alignment, fixed concatenation, and nearest-neighbor normal memory form a transparent baseline for testing representation complementarity without a learned fusion module.

## 6. 可安全声明的结果

| 数据集 | A1 Pixel-AP | matched DINO-only | ΔPixel-AP | 配置方向 |
|---|---:|---:|---:|---|
| MPDD | 0.3562 | 0.3304 | +0.0258 | 9/9 positive |
| BTAD | 0.6455 | 0.6206 | +0.0249 | 9/9 positive |
| VisA | 0.3725 | 0.3201 | +0.0524 | 9/9 positive |
| MVTec AD | 0.5546 | 0.5226 | +0.0320 | 9/9 positive |

安全措辞：

> The fixed dual-encoder representation improved mean Pixel-AP over the matched DINO-only control in all four datasets, and the gain remained positive in every seed/shot configuration.

必须同时加上：

> The improvement was not universal across metrics or categories: BTAD showed lower image-level AP and F1-max, while several categories exhibited persistent negative transfer.

## 7. 明确禁止的主张

- “the first CLIP–DINO fusion method”
- “a multimodal/vision-language inference framework”
- “dynamic or adaptive fusion”
- “state-of-the-art on four datasets”
- “consistent improvement across all metrics and categories”
- “nine independent trials/datasets”——3 seeds × 3 shots 共享同一测试集
- “VisA external generalization”——该 checkpoint 与 VisA 存在训练域关系
- “no computational overhead”——CLIP 分支是主要时间开销
- “statistical significance”——除非明确指定 bootstrap 单位、CI 和检验对象

## 8. 论文应主动呈现的负面事实

- MVTec `leather`: mean ΔPixel-AP = -0.0428，9/9 配置为负。
- VisA `chewinggum`: -0.0386，9/9 为负。
- MVTec `hazelnut`: -0.0297，9/9 为负。
- VisA `candle`: -0.0198，9/9 为负。
- MPDD `bracket_brown`: -0.0051，9/9 为负。
- BTAD 的 Image-AP 与 Image-F1-max 相对 matched DINO-only 分别约下降 0.0131 和 0.0237。
- 稳态端到端约 0.4146 s/image（2.412 image/s），CLIP 特征提取占大头。

这些不是“削弱论文”的材料。对于 controlled study，它们恰好说明论文在回答“何时有效、何时失效”，而不是只汇报平均数。

## 9. 与最接近工作区分

- **Sea-CLIP (WACV 2026)** 同样组合 CLIP 与 DINOv2，但使用可学习的 anomaly matching decoder、prompt learning 和合成异常训练；A1 是 normal-only、固定融合、无训练的受控基线。
- **PAPL (Pattern Recognition 2026)** 通过提示分布学习、层次多尺度特征和自适应池化组合 CLIP/DINO；A1 不优化 prompt，也不使用学习式跨编码器融合。
- **CLIP–DINOv2 DMA/SAP (Electronics 2025)** 是在辅助数据上训练的 zero-shot 框架；A1 使用目标类别的少量正常参考图，不使用辅助异常训练。
- **ReMP-AD (ICCV 2025)** 使用检索增强的多模态 prompt fusion；A1 研究的是纯图像 patch 表征的固定互补性。
- **SubspaceAD (CVPR 2026)** 证明冻结 DINOv2 + PCA 可构成强训练自由基线；它提示本文必须把 SubspaceAD 作为强相关方法讨论，但两者研究问题分别是单编码器正常子空间与双编码器互补性。
- **FastRef (CVPR 2026)** 在测试时细化正常原型，可接入 PatchCore/WinCLIP/AnomalyDINO；A1 不利用查询图更新原型。

## 10. 预期审稿意见与回答方向

### “方法过于简单”

回答重点不是复杂度，而是受控问题：简单性使第二编码器带来的净效应可被隔离；36 个配置、四个数据集、完整指标和失败边界使结论比单一排行榜结果更可解释。

### “为何不使用学习式融合？”

论文目标是 normal-only、零训练和泄漏安全。历史动态路由实验可在 Appendix 说明：更复杂规则没有在冻结协议下形成稳定优势，且某些版本涉及 test-mask 选择，因而不进入主方法。

### “为什么选择 AnomalyCLIP 图像塔却不用文本？”

回答应强调该分支是经过异常检测适配的视觉表征，用于测试其局部 patch 与自监督 DINOv2 patch 是否互补。模型来源是视觉语言训练体系，不等于当前推理使用语言证据。

### “为什么不宣称 SOTA？”

统一本地比较中，AnomalyDINO 在 MVTec/VisA 的若干指标仍更强。本文的贡献是 matched representation study 与边界分析，主动避免不公平跨论文数字比较。

