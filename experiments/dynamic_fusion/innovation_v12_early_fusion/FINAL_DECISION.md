# FINAL DECISION — V12-EARLY-FUSION（doc 23 分级路线整链总账）

date: 2026-09-04 (overnight autonomous run)
authority: docs/.../23_LEARNABLE_AND_EARLY_FUSION_ROUTES_CN_20260903.md（本目录的最终判定依据 doc 23 §8.1 结构）

## 链条状态

| 阶段 | 结果 | 证据 |
|---|---|---|
| Stage 0 多层可观测性门 | **PASS** | `02_stage0_probe/STAGE0_DECISION.md`：oracle headroom 宏 +0.3885（g1 OR）；72/72 跨分支低相关对（g2）；top-cat 贡献 ≤0.295（g3）；map 级 parity ≤1e-6（g4 操作门）。**注意**：静态多层基线未超过 A1（D11C18/FULL < A1），信号来自"逐 GT 连通域最优层专家"这一不可训练上限。 |
| Stage 1 缓存特征版 SCAIF | **FAIL → 归档** | `03_scaif_small_gate/STAGE1_DECISION.md` + `FAILURE_ANALYSIS.md`：main 宏 Δ vs A1 = −0.048/−0.021/−0.056（k1/2/4）；低于自身 gate=0 静态基线；gate 饱和 66–97%；P1/P3/P4/P5 全败；机制门无判别意义。 |
| Stage 2 in-backbone bridge | **不进入**（Stage1 未过） | doc 23 §7 |

## 结论（对项目/论文的意义）
1. **A1（冻结 DINOv2-B/14 L11 + AnomalyCLIP L24，0.5/0.5 L2 concat，1-NN）在 MPDD s0 上是极强且难以超越的基线**：
   - 多层静态 concat（FULL / 2-pair static2）≈ A1 或更差；
   - ≤300k 可学习跨分支门控残差训练后 < A1 < static2；
   - 唯一大 headroom 来自 evaluator-only 的"逐连通域最优层专家"oracle，不可由缓存级模块兑现。
2. **早期/前期可学习融合家族在本设定下闭合**（含 E1 SCAIF 缓存版、隐含 E2 bridge 前提、E4 类路由的
   "可学习 router"变体）。任何后续尝试必须先回应 `FAILURE_ANALYSIS.md §3` 的三个机制问题。
3. 继续开放、但需 supervisor 决策后预注册的项：
   - doc 22 队列 **PRS**（上一轮 NTOF 归档后的 next）；
   - doc 23 §6 **E6 memory 内交互/检索条件重排**（无训练、推理时显式重排，最接近 A1 风格）；
   - 论文负结果表新增一条证据链（见 05_CLAIM_EVIDENCE 待更新条目），不包装为正。

## 目录（doc 23 §8.1 交付核对）
```
innovation_v12_early_fusion/
├── 00_protocol/        PROTOCOL_FROZEN.yaml, DATA_LEAKAGE_AUDIT.md          ✓
├── 01_multilayer_cache/ CACHE_MANIFEST.json, ALIGNMENT_REPORT.json,
│                         DEEPEST_PARITY_REPORT.json                          ✓
├── 02_stage0_probe/    LAYERWISE_RESULTS*.csv, SCORE_CORRELATIONS*.csv,
│                         ORACLE_HEADROOM*.json, STAGE0_RESULT*.json,
│                         A1_REFERENCE_MAPS.json, STAGE0_DECISION.md          ✓
├── 03_scaif_small_gate/ CONFIG.yaml, PARAMETER_COUNT.json, CONTROL_RESULTS.csv,
│                         MECHANISM_AUDIT.json, STAGE1_DECISION.md,
│                         FAILURE_ANALYSIS.md, REFERENCES.json, runs/         ✓
└── FINAL_DECISION.md    （本文件）                                            ✓
```

## 下一步（待 supervisor / 下一指令）
- 是否预注册并执行 PRS（doc 22）或 E6（doc 23 §6.4）？两者均为"非早期可学习"路线。
- 论文写作侧：把 Stage0 观测（多层可观测性、oracle 上限、A1 强基线）与 Stage1 负结果纳入
  证据矩阵与 related-work 讨论。
