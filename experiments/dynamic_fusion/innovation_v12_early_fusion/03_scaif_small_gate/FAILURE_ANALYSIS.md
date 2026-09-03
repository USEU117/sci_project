# FAILURE ANALYSIS — V12 SCAIF（cached-feature Stage 1）

date: 2026-09-04
status: ARCHIVED (RSR permanent close candidate; decision: `STAGE1_DECISION.md`)
experiment: 缓存特征版 SCAIF，MPDD seed0, leave-one-category-out, k{1,2,4}

## 1. 一句话结论
在 MPDD s0 × k{1,2,4} 上，<300k 参数、带 support-conditioned 有界门（cap 0.2）的 2 对层
3×3 双向跨分支残差修正器，宏平均 Pixel-AP **低于 A1 且低于其自身 gate=0 静态基线**，
且训练把门推向饱和（66–97%），机制前提被证伪。

## 2. 失败症状（可复现的观测）
- main macro Δ vs A1：k1 −0.048 / k2 −0.021 / k4 −0.056（全负）。
- main macro Δ vs static2（gate=0）：k1 −0.047 / k2 −0.026 / k4 −0.052（训练反而变差）。
- 类别分化：black/brown/connector 常正（+0.005…+0.115），white/metal/tubes 常大负
  （−0.07…−0.17）。负值集中在高 AP / 强颜色语义类别。
- gate mean ~0.19–0.20（cap 0.2），fraction at cap 0.66–0.97：门接近"常开"。
- 训练 loss 中 seg 从 ~1.3 降到 ~0.7，ap/cp 恒 ~0（anomaly-preserve 从不被触发 →
  交互后异常分值从未低于 A1 私流，说明残差只是放大而非修正）。

## 3. 根因
### 3.1 监督把"常开残差"当最优解（gate 饱和的机制学解释）
Episodic seg（BCE@10×pos-weight）下，逐 patch 的最近邻分值只对"是否远离 support"敏感；
对任何 patch 施加指向远离/靠近 support 的固定方向残差都能降 loss。残差方向 Δ(p) 本身由
MLP 按 patch 内容给出，但**标量门是否开启并没有独立的收益信号**——于是稀疏惩罚（1.0）被
seg 梯度压过，门全开。support 距离统计（d_sup/c_sup）虽然进到门输入，但模型不需要它来降
loss，学到"忽略它"。这解释了为何把 sparse 权重 0.05→0.5→2.5、anchors 32→256 都无法让门
稀疏：问题不在惩罚强度，而在**监督目标与"条件化稀疏门"无关**。

### 3.2 跨类别（leave-one-cat）修正与 held-out 类私有统计冲突
训练只见过源 5 类的缺陷/正常分布；held-out 类（尤其 metal_plate/tubes/bracket_white 这种
"正常外观紧凑、缺陷语义强"的类）对任何内容级扰动都极敏感。训练后的变换把它们的正常 patch
推向与自身 support 的"伪远"距离，或把缺陷 patch 推向别的类别模式，直接压低 AP。6 类中
仅"低 AP 结构类"（black/brown/connector，A1 本身 ~0.01–0.3）能被跨类学到的"缺陷=偏离"
先验帮到。

### 3.3 Stage0 oracle headroom 不可兑现的原因
oracle（+0.39）是"逐连通域最优层专家"上限：它允许每个缺陷用不同层/分支单独投票，等价于
一个测试时自适应的分层器。缓存级 SCAIF 只有 ≤300k 参数、固定 2 对层、加性残差、且必须在
每个 patch 上对正常/异常同时负责，表达能力与 oracle 的自由度完全不对等；训练目标也不是
"component-level 最优"。因此 oracle 只能作为"该数据有判别信号"的证据，不能作为"可学习模块
能拿到它"的证据。

### 3.4 另一个佐证：static2 ≈ A1（多层静态 concat 没有红利）
control #2（原始 D9C12+D11C24 concat）宏平均 k1 0.3084 ≈ A1 0.3092、k2 0.3486 > A1、
k4 0.3844 < A1（±0.005）。加上 Stage0 的 FULL 静态多层 < A1，说明 MPDD s0 上"加层不加价"；
A1 的深 层 0.5/0.5 concat 已吸收多层增益的绝大部分。任何试图在中层再加一路信息的路子
（E1/E2）都要先解释这条。

## 4. 已排除/已受控的解释
- 不是参数预算问题：255k<300k；对照按同预算执行。
- 不是实现正确性问题：gate=0 恒等测试行级 0 误差、map AP 与 static2 逐位一致；a1/static2
  参考值与 Stage0/A1 harness ≤1e-6 复现。
- 不是训练不足/发散：600 步/fold loss 平滑下降；同 config 在 bracket_black fold 上对
  black/brown/connector 确有提升，说明优化器工作正常，只是学到的东西对 held-out 负向迁移。
- 不是"没跑机制对照"：P5（饱和<10%）与 M1（≥strongest control +0.004）已独立失败，按 §7
  停止即归档；训练版 shuffled/no-support 对照在此前提下无判别意义（门全开时二者必无差别）。

## 5. 归档启示（供后续路线/论文用）
1. "早期可学习跨分支融合（缓存级）在 MPDD s0 无法超过冻结深静态 concat"——进入 claim-evidence
   负结果表，附 CONTROL_RESULTS.csv。
2. 要救"跨分支交互"这一想法，证据表明：必须 (a) 训练与推理共用逐类 support 统计且 loss 显式
   奖励"条件化稀疏门"（而不是靠 seg 隐式）；或 (b) 换到 memory-retrieval 型（E6）在推理时
   显式重排，而非训练残差。二者都需 supervisor 拍板后再预注册。
3. doc 22 队列 PRS 仍为开放项，优先级建议由 supervisor 决定（doc 23 主线已闭环为负）。
