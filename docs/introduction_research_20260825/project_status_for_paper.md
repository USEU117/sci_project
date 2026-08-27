# Project status for paper framing

更新时间：2026-08-25。权威状态来源：`docs/CURRENT_DYNAMIC_FUSION_STATUS.md`（更新到 2026-08-23 的探索记录）。

## 当前论文主线

项目已从“动态视觉—文本路由”收敛为：

> Reference-Conditioned Multimodal Feature Fusion with a Normal Memory Bank：将 DINOv2 patch 特征与 AnomalyCLIP patch 特征在统一空间中对齐、拼接，并用当前 K 张正常参考图建立 KNN memory bank，用于少样本工业异常检测与定位。

核心设定是 1/2/4-shot、3 seeds、只用正常参考图构建 memory bank；测试标签和测试 mask 只进入最终 evaluator。

## 已完成的证据范围

- VisA：PatchCore、WinCLIP+、AnomalyDINO、PromptAD 均完成 9/9（3 seeds × 1/2/4-shot）；DynamicFusion/A1 完成冻结验证。
- MVTec AD：PatchCore、WinCLIP+、AnomalyDINO、PromptAD、DynamicFusion/A1 均已完成 9/9；每组要求 15 类、1,725 个测试样本、零 schema 错误。
- MPDD：开发集，用于冻结配置和早期开发。
- BTAD：冻结后的外部验证集，不用于回头调参。
- 已有英文 SCI 风格稿 V0.2，但 Introduction 仍需要独立补强、引用清理和与当前 A1 结论对齐。

## 可以在 Introduction 中提出的研究问题

1. 工业缺陷稀少、类别多变、像素级定位要求高时，如何只依赖少量正常样本建模正常性？
2. 视觉 foundation features 与 vision-language features 是否提供互补证据？
3. 两种分支的距离/相似度具有不同数值语义时，如何进行公平、reference-conditioned 的融合？
4. 融合是否能在外部数据集和不同 shot 数下保持稳定，而不是只在单一类别或单一 seed 上获益？

## 必须避免的表述

- 不写“动态路由全面优于最强单分支”：当前 D1 无标签可预测性门失败，V4 已按预注册停止规则关闭。
- 不把 A1 称为动态权重路由；冻结 A1 是 fixed fusion，权重不随测试图变化。
- 不使用 V3.3 的 test-mask calibration 结果作为正式论文证据；该路线已标记 development-only / paper-ineligible。
- 不将 9 个 shot/seed 组合当成 9 个独立数据集或伪显著性样本。

## 论文准备度判断

实验材料已经足够支撑一篇“严谨的少样本工业异常检测基线与固定多模态特征融合”论文，但不能支撑“已验证的动态不确定性路由”这一更强标题。Introduction、方法名、摘要和贡献点都应与 A1 及其负结果边界一致。

