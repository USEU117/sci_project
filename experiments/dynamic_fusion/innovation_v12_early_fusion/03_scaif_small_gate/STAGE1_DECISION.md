# STAGE1 DECISION — 缓存特征版 SCAIF（doc 23 §7 Stage1）：FAIL → 归档

date: 2026-09-04 (overnight autonomous run)
authority: docs/.../23_LEARNABLE_AND_EARLY_FUSION_ROUTES_CN_20260903.md §7 Stage1 / §8
entry: `02_stage0_probe/STAGE0_DECISION.md` = PASS（oracle headroom +0.3885，g1–g4 green）
config: `03_scaif_small_gate/CONFIG.yaml` v4（PRE-REGISTERED，无 post-hoc 搜索）

## 0. 结论

**FAIL — 四项性能门在 MPDD s0 × k{1,2,4} 全部不满足；主模型宏平均低于 A1 与自身的 gate=0 静态基线（control #2），门饱和 66–97%（control 语义退化为常开修正）。**

→ 按 doc 23 §7：**停止缓存版 SCAIF / 早期可学习融合主线，归档**；不实现 Stage 2（in-backbone bridge）；不因加层/加宽/扫温度继续搜索。全部对照与失败分析存档如下。

## 1. 协议与实现（复述，均已预注册）
- leave-one-category-out episodic 训练（target cat 只前向）；源=其余 5 个 MPDD cat（真实缺陷监督）。
- 2 层对：P1=(DINO L9, CLIP L12)、P2=(DINO L11, CLIP L24)；768→u=32 投影、3×3 邻域双向交互 MLP、zero-init 残差解码（init=严格恒等）、support-conditioned 有界门 g=0.2·σ(h)，sparse 权重 1.0、weight_decay 1e-3、600 步/fold、Adam lr 1e-3、seed0。
- 参数：255,236（≤300k，PARAMETER_COUNT.json PASS）。
- 打分：A1 同款 per-branch L2→0.5 concat→L2→精确 1-NN d/2→56 网格 pooled Pixel-AP。
- 恒等自检（control #10）：gate=0 时行级 max-abs 误差 0.0、map AP 与 static2 完全相等（PASS）。
- 静态对照：a1/static2 参考图与 Stage0/A1 基准复现一致（≤1e-6 量级）。

## 2. 结果（宏平均 pooled Pixel-AP @56，MPDD s0，6 held-out cats）

| 配置 | k1 | k2 | k4 |
|---|---|---|---|
| A1（control #1，深 D11+C24 concat） | 0.309212 | 0.343699 | 0.388328 |
| static2（control #2，原始 2 对层 concat = SCAIF@gate0） | 0.308382 | 0.348602 | 0.384439 |
| **SCAIF main（训练后）** | **0.261294** | **0.323109** | **0.331993** |
| Δ vs A1 | **−0.0479** | **−0.0206** | **−0.0563** |
| Δ vs static2（自己的零门基线） | −0.0471 | −0.0255 | −0.0524 |
| 正向类别数（vs A1） | 3/6 | 3/6 | 2/6 |
| gate 饱和比例（g>0.18） | 66–97%（fold 依赖，多数 ≥93%） | 同左 | 同左 |

逐类 Δ vs A1（k1/k2/k4）：bracket_black +0.021/+0.115/+0.006 · bracket_brown +0.013/+0.005/+0.008 ·
bracket_white −0.073/−0.089/−0.084 · connector +0.021/+0.048/−0.070 · metal_plate −0.167/−0.125/−0.110 ·
tubes −0.102/−0.078/−0.087。

（详细逐类 AP：`CONTROL_RESULTS.csv`；逐 shot 门：`MECHANISM_AUDIT.json`）

## 3. 门控判定（CONFIG §4；全部为 AND）

| 门 | 要求 | 实测 | 判定 |
|---|---|---|---|
| P1 | vs A1 mean Δ ≥ +0.006 | −0.048 / −0.021 / −0.056 | **FAIL** |
| P3 | ≥5/6 类正 | 3/6, 3/6, 2/6 | **FAIL** |
| P4 | worst-cat Δ ≥ −0.010 | 最低 −0.167（k1 metal_plate） | **FAIL** |
| P5 | gate 饱和 <10% | 66–97% | **FAIL** |
| M1 | vs strongest control ≥ +0.004 | 低于 static2（−0.047/−0.026/−0.052） | **FAIL** |

（M2 shuffled / M3 no-support / M4 remove-private：主模型连自身零门静态基线都未达到，机制对照
失去判别意义；且 P5 已独立证伪"门是稀疏条件化修正器"这一机制前提。按 §7 任一门失败即停，
不再运行训练版机制对照矩阵，避免"搜索式续命"。）

## 4. 失败分析（详见 FAILURE_ANALYSIS.md）
1. **门饱和 → 退化为常开修正**：即便 sparse 权重提到 1.0、anchors 提到 256，训练仍把 g 推向 cap
   （mean ~0.19，σ 几乎全开）。说明在该监督下"对每个 patch 施加同向残差"几乎总是降低 seg loss，
   模块没有动机产生条件化/稀疏门 → 机制前提（support-conditioned bounded corrector）不成立。
2. **主模型 < 自身零门基线（static2）**：训练后的整体变换在有强语义颜色的类别（bracket_white、
   metal_plate、tubes）上显著破坏 A1 表征（−0.07…−0.17），仅在低 AP 结构类（bracket_black、
   bracket_brown、connector）小幅受益（+0.005…+0.115）。leave-one-cat 学到的跨类别修正与
   各 held-out 类的"私有正常统计"冲突 → 净效应为负。
3. **与 Stage0 oracle 的落差**：oracle headroom（+0.39）来自"每条 GT 连通域恰好选最优层/分支专家"，
   是可观测上限而非可学习信号；本实现证明该 headroom 不能由 <300k 参数的缓存级跨分支门控残差
   兑现——至少在 MPDD s0 的 6 类 leave-one-cat 设定下不能。
4. 对 A1 的解读再次被强化：**深 层 静 态 0.5/0.5 concat 已是难以超越的强基线**（static2 ≈ A1，
   多层静态 < A1，训练变换 < static2）。任何"前期/早期可学习融合"都落在同一结论带内。

## 5. 交付物（doc 23 §8.1）
- `CONFIG.yaml` ✓（v4 frozen）· `PARAMETER_COUNT.json` ✓ · `CONTROL_RESULTS.csv` ✓ ·
  `MECHANISM_AUDIT.json` ✓ · `STAGE1_DECISION.md` ✓ · `FAILURE_ANALYSIS.md` ✓ · `runs/main_all.json` ✓
- Stage 0 产物见 00/01/02；总账见 `FINAL_DECISION.md`（已更新）。

## 6. 后续（不自动执行）
- Stage 2（in-backbone bridge）：按 §7 不再进入。
- doc 23 §6 其余路线（E2–E7）中，仅 **E6 memory 内交互/检索条件重排** 与 doc 22 队列的
  **PRS** 仍有开放地位；其新颖性/对照成本需 supervisor 决策后再预注册。
- 论文层面：本路线的负结果应进入 claim-evidence 矩阵（"early learnable cross-branch fusion does
  not beat frozen deep static concat on MPDD s0"），不包装为正。
