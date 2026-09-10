# 本轮相关工作与新颖性边界

实际打开一手摘要：[HLGFA: High-Low Resolution Guided Feature Alignment for Unsupervised Anomaly Detection](https://arxiv.org/abs/2602.09524)，2026-02-10提交，当前v3 2026-05-09。

摘要明确采用共享冻结backbone的高低分辨率一致性，以结构/细节先验、条件调制及残差校正来检测跨分辨率对齐破坏；并包含抑制nuisance的增强。因此“跨尺度差异/残差”不能直接作为本项目的新颖性主张。

本轮轻量probe只回答：有限normal support下，闭式或无需训练的尺度残差是否有可用独立信号；若通过，仍需核查完整算法、few-shot协议、公平预算及简单late控制，才可讨论差异。当前未复现HLGFA，不引用其成绩作为本地比较。

另已检索到PRN、WeakREST等残差异常检测工作，但没有进一步读取正文。本轮不据检索摘要推断它们与当前算法等价或宣布首创。
