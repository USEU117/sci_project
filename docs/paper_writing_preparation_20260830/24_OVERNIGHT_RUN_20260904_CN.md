# 夜间自主运行总结（2026-09-03 23:xx → 09-04 ~05:00）— doc23 主线闭环

authority/背景：你入睡前指令"整夜持续监督、按结果动态调整"，执行 doc 23（早期可学习融合 SCAIF 分级路线）。
本夜从"多层缓存已导出"继续，未触碰冻结外部数据；全部按预注册协议与门控自动推进，每次判定后即 commit & push。

## 一句话结论
**doc 23 主线整链闭环：Stage0 可观测性门 PASS（但 headroom 来自不可学习的 oracle 上限）→ Stage1 缓存版 SCAIF FAIL（机制与性能双败）→ 按协议归档，不进入 Stage2。A1 仍是唯一方法，"早期可学习融合"家族在 MPDD s0 上闭合。**

## 提交（main，均已 push）
| commit | 内容 |
|---|---|
| a6ccbd1（上个会话） | Stage0 基础设施+冻结协议 |
| 57c6e60 | Stage0 probe k1 + A1 基准表（map parity ~1e-7） |
| 49a38a8 | **Stage0 DECISION PASS**（k2/k4 + 决策文档） |
| c36593d | Stage1 SCAIF 预注册 CONFIG v4 + 恒等自检通过 |
| 85654ac | **Stage1 DECISION FAIL + 失败分析 + FINAL_DECISION 总账** |

## 关键数字
- **Stage0**（MPDD s0 k1/2/4）：layer/branch oracle headroom 宏 **+0.3885**（每 GT 连通域取最优层专家；仅 evaluator-only 审计）。静态多层 concat 无法超过 A1（D11C18/FULL < A1）——"加层不加价"。g2 72/72 低相关对、g3 ≤0.295、g4 map 级 parity ≤1e-6 全过。
- **Stage1 SCAIF**（≤300k 参数=255,236；留一类别 episodic；2 层对 gated 3×3 残差；gate cap 0.2）：
  - main 宏 Δ vs A1 = **−0.048 / −0.021 / −0.056**（k1/2/4）→ P1 败；
  - 甚至**低于自身 gate=0 静态基线**（−0.047/−0.026/−0.052）→ M1 败；
  - 逐类：black/brown/connector 常正（+0.005~+0.115），white/metal/tubes 大负（−0.07~−0.17）→ P3/P4 败；
  - **gate 饱和 66–97%** → P5 败（模块退化为常开修正，无"支持条件化稀疏门"机制）。
  - 恒等自检（gate=0 时输出与静态基线逐位一致）证明实现正确 → 失败是机制性的。

## 为什么失败（详见 FAILURE_ANALYSIS.md）
1. seg 监督偏爱"常开残差"：逐 patch 最近邻分只对"离 support 远近"敏感，开门的收益压倒稀疏惩罚 → 门不条件化。
2. leave-one-cat 学到的跨类修正与 held-out 高 AP 颜色/语义类私有统计冲突 → 负迁移。
3. Stage0 的 +0.39 oracle 是"每条缺陷恰好由最优层单独投票"的自由度上限，<300k 固定 2 层对加性残差模块表达力与其不对等，训练目标也不同 → 不可兑现。

## 下一步（等你拍板，未自动执行）
- **PRS**（doc 22 §5 扰动响应谱）：NTOF 归档后的既定 next。需先定义扰动阶梯、双编码器 perturbed 导出、normal envelope + 合成门（R0 全 normal-only/合成，5.3 六门），工作量大，建议单独会话预注册执行。
- **E6**（doc 23 §6.4 memory 内交互/检索条件重排）：无训练、贴近 A1 风格，可作为备选。
- 论文侧：两条负结果/可观测性证据建议录入 claim-evidence 矩阵（05 号文档新增行）。

## 待办/已知遗留
- `runs/main_all.json` 为正式 main 结果（600 步/fold v4）；controls 训练矩阵因 P5+M1 决定性失败未跑（按 §7 任一失败即停，避免"搜索续命"）。
- Stage1 训练态控制（shuffled/no-support/symmetric/no_cross/dino_only/clip_only）代码已就绪（run_r3_ef_scaif.py --mode runs --variant <v>），若 supervisor 想复核机制可快速补跑。
- 实验目录：`experiments/dynamic_fusion/innovation_v12_early_fusion/`（00~03 + FINAL_DECISION.md 结构完整）。
