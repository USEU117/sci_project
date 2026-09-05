# V14 P0 AUDIT_CORRECTIONS（doc28 §2.1 + §4）

日期：2026-09-05。作用：登记 V13 审计发现的三问题及其在 v14 中的修复/降级。**不覆盖任何 V13 文件。**

## 问题 A：N2 不是原计划的软容量匹配（doc28 §2.1-A）

- 事实：`run_n2_cct.py` 的 `sinkhorn_log` 要求方阵、行列边缘均匀 → 平衡 OT；探针用 support
  自锚+16×16 特征域直接复制块；无法覆盖 k2/k4 多参考与 Q≠A。
- v14 修复：新 `semi_ot.solve_semi_ot` = 广义 Sinkhorn 半松弛解（行固定 a、列软 KL 向 ρ，
  支持 Q≠A）。18/18 确定性单测通过（含长方形、τ→0 退化为逐行熵 soft match、大 τ 趋近平衡
  OT、置换不变、identical 零 premium、复制 premium 单调、远行 spillover 低）。
- V13 N2 结论状态：降级为"概念探针（特征域/同图/平衡 OT）"，真实门未跑，不撤回、不重跑。

## 问题 B：N3 使用了目标 test/good（doc28 §2.1-B）

- 事实：NTOF `good_syn_feat` 来自测试集 good 图；V13 DNC 通道响应直接用它 → 违反
  DATA_ROLES「目标类正常 test/good 不得进入拟合」。
- v14 修复：P1 只从 manifest K-shot support 生成合成干预并**图像域重编码**（cutpaste/
  local_erasure/thin_scratch + 5 类 nuisance）；test/good 仅留在冻结后的评估集。
- V13 N3 结论状态：全部数字降级为离线机制探索（"通道稀释"线索保留为假设，不构成方法结果）。

## 问题 C：DNC-C 必然选回 DNC-I（doc28 §2.1-C）

- 事实：旧循环每步先 `argmax(qD)/argmax(qC)`，λ 惩罚只仲裁两个候选谁先入选；配额 256 下
  最终仍是各分支 top-256 → 集合恒等于 DNC-I，"dnc_c==dnc_i" 不是有效阴性证据。
- v14 修复：`dnc_selector.select_dnc_c` 改为分支内全候选 `q_j − λ·max|corr(j, opposite_selected)|`
  贪心（向量化）。单测证明：λ=0 → 集合=DNC-I；构造高 q 高冗余通道在 λ>0 时被替换（集合
  改变、chosen 跨分支相关 0.99→0.05）；256/分支唯一且确定性。
- 待 P1 报告：真实尺度 DNC-C vs DNC-I 的集合 Jaccard、跨分支 max/mean 相关、q 值损失。

## 附带记录
- N4 的 "~2× 快" 是 bank-size/距离数代理，非实测匹配/端到端延迟（P3/收尾如需实测再测）。
- N4 branch_merge 并集可少于目标 k：v14 不使用 branch_merge。
- V13 中 N1 CDF rank 修复后的负结果幅度很大，v14 不复活 JTD。

## 引用
- doc28 §2.1/§2.2 全部要点已映射；实现 hash 见 RUN_MANIFEST.json。
