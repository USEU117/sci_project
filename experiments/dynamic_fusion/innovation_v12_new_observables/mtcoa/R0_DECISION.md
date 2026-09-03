# V12 MTCOA — 修正 Oracle 审计决策（2026-09-03）

协议：`R0_PROTOCOL.json`（含 amendment：校准池 = support LOO + test/good 正常；损失 = 区域像素 BCE hit + (1−AP) + FP）
脚本：`scripts/innovation_v12_new_observables/run_r1_mtcoa.py`
范围：MPDD development seed0 × shot {1,2,4} × 6 类，evaluator-only 能力审计（无 router 训练）。
专家池：E0 A1 / E1 text / E2 LLSE / E3 CSS（与 v11 相同，修正 v11 的三个缺口 A/B/C）。

## 结果（分类别 oracle ΔAP = Oracle−A1，全为校准后同尺标）

| 类 | k1 | k2 | k4 |
|---|---:|---:|---:|
| bracket_black | +0.111 | +0.095 | −0.017 |
| bracket_brown | +0.040 | +0.018 | +0.020 |
| bracket_white | +0.095 | +0.100 | +0.138 |
| connector | +0.058 | +0.028 | +0.022 |
| metal_plate | **−0.095** | **−0.098** | **−0.130** |
| tubes | **−0.066** | **−0.120** | **−0.111** |
| **mean Δ** | **+0.0239** | **+0.0040** | **−0.0129** |
| ≥4/6 类 ≥+0.01 | 4/6 | 4/6 | 3/6 |
| component-macro 非A1≥15% | E1 35%/E2 24% | E1 33%/E2 24% | E1 32%/E2 24% |
| small-stratum 第二专家 ΔAP | +0.056 | +0.065 | +0.037 |
| LOO drop（去专家对 meanΔ） | E1 +0.003/E2 +0.003/E3 **−0.001** | 全部 **负** | 全部 **负** |
| 最大单类占正 headroom | 0.37 | 0.42 | **0.77**（bracket_white）|
| identity max diff | 1.02e-4（边缘） | 5.3e-5 | 3.9e-5 |

## 门判定
- g0 identity：k1 1.02e-4 边缘超（float32 饱和 rounding；5/6 类 ≤7e-7），k2/k4 PASS。
- g1 headroom：k1 PASS（+0.0239，4/6）；**k2 FAIL（+0.0040<0.020）**；**k4 FAIL（−0.0129，3/6）**。
- g2 contribution：**三 shot 全 PASS**——text 与 LLSE 各 ≥15% component-macro 选择，small-stratum 第二专家 +0.04~+0.07≥0.02。
- g3 LOO：k1 FAIL（去 E3 反升）；**k2/k4 FAIL（去任一核心专家都提升 oracle，负 drop）**。
- g4 dominance：k1/k2 PASS；**k4 FAIL（bracket_white 占 77%）**。
- g5 direction：失败 shot 存在 → 按 doc 规则关闭。

## 结论：MTCOA FAIL → **RSR 永久关闭**（按 doc 22 §2.3/§13 指令）
在修正了 v11 全部三个技术缺口后（共同校准尺标、doc21 区域损失 BCE+AP+FP、component/size
宏观报告），`{A1, text, LLSE, CSS}` 池仍不满足一次通过的 headroom 门：k2/k4 均值 oracle
Δ 只有 +0.004/−0.013，且去掉核心专家反而改善。RSR（pseudo-regret router）**永久关闭，
任何后续会话不得再训练 router**。

## 审计结论中有价值的部分（不丢失）
1. doc 22 的担忧成立：v11 的 pixel-micro 统计确实误导。改为 component-macro 后，
   **text 与 LLSE 各占 32–35%/24% component 选择（三 shot 一致）**，small-defect（1–4 cells）
   第二专家带来 +0.04~+0.07 AP——bracket 类与小缺陷上的区域互补是**真实且跨 shot 一致**的。
2. 但 headroom 无法宏观平衡：A1 最强的 metal_plate/tubes 在**任何 shot、任何 loss 变体**下
   oracle 都净损 −0.07~−0.13（其它专家的缺陷响应弱于 A1，即使有 GT 组件几何知识也无法补）；
   正 headroom 集中于 bracket_white（k4 达 77%）。
3. 因此"区域级动态专家互补"作为**宏观统一主张**被证据关闭；作为**类别条件现象**仍成立——
   这正是 doc 22 §12 归因于需要新机制专家（NTOF/CECW/PRS）或改变协议（类别条件/元训练）的方向，
   不是固定池 router 能承载的。
