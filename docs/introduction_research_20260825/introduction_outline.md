# Suggested Introduction outline

建议篇幅：SCI-I 期刊正文约 6–8 个段落、900–1400 英文词；最终按目标期刊调整。

## Paragraph 1 — Industrial motivation and problem definition

先说明 automated visual inspection 需要发现稀有、未知、细小的缺陷，并指出异常样本通常难以事先收集和穷举。定义任务：给定少量正常参考图，输出 image-level anomaly score 和 pixel-level anomaly map。引用 MVTec AD 与 VisA。

可写的核心句：

> Industrial anomaly detection differs from conventional closed-set recognition because the defective patterns of interest are often rare, heterogeneous, and not fully specifiable during system development.

不要在第一段放本项目结果，也不要先讲算法名。

## Paragraph 2 — Normal-only and few-shot visual modeling

说明 one-class/normal-only 的现实合理性，再介绍预训练 feature + normal memory bank。引用 PatchCore。随后说明 K=1/2/4 比“很多正常训练图”更接近冷启动/小批量产品场景，但会让正常分布估计更不稳定。

## Paragraph 3 — Strong visual foundation features

说明 self-supervised/foundation visual representations 改善 patch-level local matching，引用 DINOv2 和 AnomalyDINO。将 AnomalyDINO 作为 strong visual anchor 引出：单一强视觉分支已经很强，因此任何 multimodal fusion 都必须证明稳定、可重复的额外价值。

## Paragraph 4 — Vision-language transfer

介绍 CLIP 的图文共同空间，再说明 WinCLIP 将 prompt/window/patch cues 用到 zero-/few-normal-shot anomaly classification and segmentation。强调优势是语义迁移与开放类别泛化，局限是工业异常常表现为局部纹理、结构偏差或微小缺陷，文本相似度未必等价于局部异常证据。

## Paragraph 5 — Anomaly-specific prompting and protocol differences

介绍 AnomalyCLIP 的 object-agnostic normality/abnormality prompts，以及 PromptAD 的 only-normal prompt learning。明确两条路线的协议差异：zero-shot/frozen semantic transfer、target-normal prompt tuning、normal memory-bank matching 不是同一条件，不能只按一个排行榜数字比较。

## Paragraph 6 — Why fusion is non-trivial

这是本项目最重要的铺垫。说明视觉 patch distance 与 vision-language similarity 具有异构尺度、不同校准含义和不同 failure modes；简单平均/固定加权并不能保证收益，动态路由还要求一个在无异常标签条件下可靠估计分支可靠性的机制。K 张正常参考图不足以支撑任意复杂校准，因此需要 reference-conditioned、leakage-safe 的设计和严格的 cross-dataset validation。

## Paragraph 7 — Research gap and objective

建议用较窄的 gap：

> Despite progress in visual memory-bank methods and vision-language anomaly detection, the reproducibility and robustness of fusing heterogeneous patch-level evidence under an extreme few-shot, normal-only protocol remain insufficiently characterized.

本项目目标：在统一 1/2/4-shot、3-seed protocol 下，对 DINOv2 patch evidence 与 AnomalyCLIP patch evidence 做 reference-conditioned fixed fusion，比较 matched visual-only controls，并在 VisA/MVTec 及冻结后的 MPDD/BTAD 上检查 image/pixel-level stability。

## Paragraph 8 — Contributions

贡献建议写 3 点，不写动态路由：

1. A leakage-safe and reproducible few-shot protocol with nested normal-reference manifests, multiple seeds, and separate development/external-validation roles.
2. A reference-conditioned multimodal patch-feature fusion method combining DINOv2 and AnomalyCLIP evidence through a normal memory bank.
3. A cross-dataset analysis of when fusion helps or fails, including matched single-branch controls, shot sensitivity, category-level variation, and explicit limitations of dynamic routing under label-free reliability estimation.

如果目标期刊更偏应用，可把第 3 点改为 deployment-oriented robustness and efficiency analysis，但不要把稳定性结果扩大成 universal superiority。

