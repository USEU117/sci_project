# 广度探针 R10 预注册（2026-09-09 夜，续）— 互惠最近邻可靠性轴

背景：30 机制族已闭合；经历史轴账本盘点（innovation_v2–v14/overnight/t3–t6/followup），确认两条**从未运行**的离线轴：
`innovation_followup_20260908/neighborhood/AUDIT_CN.md` 明确提出但未运行的"support-side reciprocal coverage weighting"，
及其对称化的"query↔memory 双向互惠 NN 一致性"。二者与 CRAM（query 侧 per-reference 聚合）、RW（典型性加权）、
coreset（删除）、AGR2（两支持图和）、NDW（二阶密度）均不同。

| 族 | 轴 | 成员 | 机制（全部离线，仅用支持集 + query 特征，无 GT） | 假设 |
|---|---|---|---|---|
| **RCW** 支持侧互惠覆盖权重 | 参考可靠性（支持侧） | RCW_10 / RCW_30（β=0.1/0.3） | 对每条 memory 行 m，跨**其他支持图**做前向 NN + 回查互惠，得到稳定度 s_m∈[0,1]（=互惠图数/(k−1)）；打分用 top-K=8 候选重加权 `min_m d(q,m)·(1+β(1−s_m))`。**不使用任何 query 统计**（与测试期适配族彻底区分） | 稳定 memory 行更可信；抑制"仅在某张支持图出现的偶然模式"造成的 FP |
| **BRC** 双向互惠一致性 | query↔memory 互惠 | BRC_10 / BRC_30（β=0.1/0.3） | 记 q 的 top-1 memory 为 m；检查 m 在**该 query 图内**的最近 patch 是否为 q（互惠命中）；`d' = d·(1+β(1−recip))` | 正常 patch 与其最近正常样例互为最近邻；异常 patch 的 m 在 query 图内另有近邻 → 放大异常分。属"异常增强"方向（与 PA/CB 的"救硬类"方向相反） |

预注册门同全 breadth（lead 宏 ΔAP ≥ +0.01 且 worst ≥ −0.03 且 ΔAUROC ≥ −0.005，k2/k4 均须）；
parity k2 0.343706 / k4 0.388328；零 test 标签拟合；不做门后调参。

补充方法学注记（本轮不测、仅记录）：**per-class 单调校准（含支持集 LOO CDF 校准）在"逐类 Pixel-AP 再宏平均"口径下恒等**（单调变换不改变类内排名 → ΔAP≡0），
故此类轴可证明为 null，不入探针清单；跨 shot/跨 seed 距离集成因门按 shot 分列不入协议。
