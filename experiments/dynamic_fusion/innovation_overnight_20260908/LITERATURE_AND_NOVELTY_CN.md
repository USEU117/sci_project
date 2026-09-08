# 文献入口与创新边界（2026-09-08）

以下为本夜实际打开的一手摘要页初筛，不是完整文献综述，也未复现论文结果。

1. [HiAD: Towards High-Resolution Industrial Image Anomaly Detection](https://arxiv.org/abs/2508.12931)，2025-08-18。作者强调缩小输入对细微异常信息的损失，提出双分支多尺度和detector pool。对本项目的启示：应检验输入分辨率是否是瓶颈；单纯“裁块+多尺度”已有工作，不能作为新颖性主张。
2. [Learning to Be a Transformer to Pinpoint Anomalies](https://arxiv.org/abs/2407.04092)，当前v3为2025-06-26，IEEE Access。使用冻结Transformer层间映射的浅MLP teacher-student路线，并提出尺寸稳健性评价。检索摘要仍曾显示旧题Looking for Tiny Defects，引用应以实际版本为准。启示：小缺陷需要单独的尺寸条件评价；层间预测可作为与既有concat不同的未来路线，但属于可训练设定。
3. [SPARC: Subspace Position-Aware Robust Few-Shot Calibration for Distribution-Shifted Industrial Anomaly Detection](https://arxiv.org/abs/2608.18585)，2026-08-19预印本。用少量已验证正常的部署图做空间索引子空间校正；作者报告收益主要在有分布偏移的基准，无人工偏移时收益小且混合。启示：正常域校正应先定义明确偏移目标，不能在无偏移主表上盲目叠加，也不可把现成的逐cell子空间投影称本项目新方法。

另检索命中CVPR2026“Defect Cue-Preserved Structural Feature Refinement for Few-Shot Anomaly Detection”，但CVF正文页返回403，当前仅列待核实入口，不据此陈述具体算法和成绩。

## 本夜实验可以说明什么
- 量化保序：可能是实用效率改进；FP16/int8本身无算法新颖性。
- 跨图几何对应：先检验空间先验的净收益与扰动代价；配准异常检测已有工作，不能仅以配准为创新。
- 原图裁块重编码：先检验细节是否可恢复，以及是否优于插值；若有信号，再研究如何在固定推理预算下决定哪些区域需重编码。区域选择必须正常support标定或冻结规则，不能根据test真值裁图。
- 合成评价修复：是方法学可靠性改进，不能当模型AP提升。

## 对历史结论的使用限制
既有单一实现失败不等于此类机制在数学上不存在收益；训练损失只降2%也不能单独证明最优增益上界。历史“冻结特征上限”“任何邻域增强都无效”等归因，宜改述为“已测试配置下未见可迁移增益”。本夜优先保留可检验的原因，避免从负结果外推不可能性。
