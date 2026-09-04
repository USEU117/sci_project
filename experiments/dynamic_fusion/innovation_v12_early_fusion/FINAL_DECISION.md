# FINAL DECISION — V12-EARLY-FUSION（doc 23 分级路线整链总账；doc 25/26 修订并入）

date: 2026-09-04（overnight autonomous run；2026-09-04 18:33 起按 doc 26 复核更新）
authority: docs/.../23_LEARNABLE_AND_EARLY_FUSION_ROUTES_CN_20260903.md §8.1；
修订依据：docs/.../25_..._CODE_AUDIT_AND_NEXT_STEPS_CN_20260904.md、docs/.../26_CURRENT_CORRECTION_STATUS_AND_NEXT_BREAKTHROUGHS_CN_20260904.md

## 链条状态

| 阶段 | 结果 | 证据 |
|---|---|---|
| Stage 0 多层可观测性门 | **PASS（操作门口径，raw 腿如实 FAIL）** | `02_stage0_probe/STAGE0_DECISION.md`。⚠️ 修订：oracle headroom +0.3885 经空信息审计后**不再作为可学习性/互补性证据**（`ORACLE_NULL_AUDIT.md`：A1 单调缩放空信息对照 Δ=+0.65>real，GT 特权+尺度差主导）；g4 raw parity <1e-5 不满足（最大 0.007），操作门为 map 级 Pixel-AP <1e-4（`00_protocol/AMENDMENT_20260904_PARITY_GATE.md`）。 |
| Stage 1 缓存特征版 SCAIF v4 | **FAIL → 归档（原实现，含已确认实现缺陷）** | `03_scaif_small_gate/STAGE1_DECISION.md` + `FAILURE_ANALYSIS.md`：main 宏 Δ vs A1 = −0.048/−0.021/−0.056；gate 饱和 66–97%。P0-1/P0-2/P1-3 缺陷见 doc25。 |
| Stage 1 缓存特征版 SCAIF v5（correction，bugfix-only） | **无增益 → 归档（与 static2 完全相等）** | `03_scaif_small_gate/CORRECTION_NOTES.md` §5：18/18 cat×shot AP 与 static2 逐位相等（门收敛至 ~0.0025、解码器权重 ~1e-14），Δ vs A1 三 shot 均值 ~+0.000061 ≪ +0.006 门槛。稀疏梯度修复后门被完全关闭 → 模块退化为自身 gate=0 静态基线。 |
| Stage 2 in-backbone bridge | **不进入**（v4/v5 均未过） | doc 23 §7；doc26 §6B/§6.2 |

## 结论（对项目/论文的意义，doc25/26 修订后口径）
1. **A1（冻结 DINOv2-B/14 L11 + AnomalyCLIP L24，0.5/0.5 L2 concat，1-NN）在 MPDD s0 上是极强且难以超越的基线**：
   - 多层静态 concat（FULL / 2-pair static2）≈ A1 或更差；
   - ≤300k 可学习跨分支门控残差：v4（实现缺陷）训练后 < A1 < static2；v5（bugfix）训练后 ≡ static2（门全关），均无增益。
2. **Stage0 oracle headroom 不再是入场证据**：空信息审计（A1复制=0、A1缩放=+0.65>real、shuffle≈0）证明 +0.3885 主成分为 GT 边界特权+专家尺度差，非跨分支排序信息（`ORACLE_NULL_AUDIT.md`）。不再基于它推断可学习互补性或训练 router。
3. **受限结论（不扩大为“整族已证伪”）**：负结果只覆盖“缓存级 SCAIF v4/v5 + MPDD s0 dev”这一实现家族子集；不扩展到任意 early-fusion 架构、其他数据集或训练协议。doc 26 §4 所列新方向（CL-RPF/PRS/共同坐标细节恢复/匹配目标/蒸馏）为**待证实假说**，各自需先过自身低成本控制与机制门，再谈三 seed/外部冻结确认。
4. 继续开放、需 supervisor 决策后预注册的项（doc 26 §4/§6 顺序）：
   - **CL-RPF**（跨层正常偏离轨迹，复用已导出多层 cache，CPU probe，doc26 §4.1）——已列为本文件之后的第一步低成本验证；
   - **PRS**（双编码器扰动响应谱，doc26 §4.2）；
   - 共同坐标空间细节恢复（FeatUp/AnyUp 类，doc26 §4.3）；匹配目标驱动学习（§4.4）；关系蒸馏降成本（§4.5，需改论文目标）；
   - 原 doc 22 队列 PRS（若与 doc26 §4.2 同一概念则合并入口）。

## 目录（doc 23 §8.1 交付核对；修订文件以 ⚠️ 标出）
```
innovation_v12_early_fusion/
├── 00_protocol/        PROTOCOL_FROZEN.yaml, DATA_LEAKAGE_AUDIT.md,
│                         ⚠️ AMENDMENT_20260904_PARITY_GATE.md                ✓
├── 01_multilayer_cache/ CACHE_MANIFEST.json, ALIGNMENT_REPORT.json,
│                         DEEPEST_PARITY_REPORT.json                          ✓
├── 02_stage0_probe/    LAYERWISE_RESULTS*.csv, SCORE_CORRELATIONS*.csv,
│                         ORACLE_HEADROOM*.json, STAGE0_RESULT*.json,
│                         A1_REFERENCE_MAPS.json, STAGE0_DECISION.md,
│                         ⚠️ ORACLE_NULL_AUDIT.md + ORACLE_NULL_AUDIT_k{1,2,4}.json ✓
├── 03_scaif_small_gate/ CONFIG.yaml(v5 amendment), PARAMETER_COUNT.json,
│                         CONTROL_RESULTS.csv, MECHANISM_AUDIT.json,
│                         STAGE1_DECISION.md, FAILURE_ANALYSIS.md,
│                         REFERENCES.json, runs/(v4 main_all + v5 main_correction_all
│                         + ckpt/log ×6), ⚠️ CORRECTION_NOTES.md              ✓
└── FINAL_DECISION.md    （本文件）                                            ✓
```

## 下一步（按 doc 26 §6 执行顺序；均待 supervisor / 下一指令）
1. ~~短优化健康诊断（§6.2/§3A）~~ → 已执行：见 `03_scaif_small_gate/RUN_OPTIM_HEALTH.md`。
   acceptance：任务梯度可达交互通路（run B 全角色非零），但无 sparse 时固定 episode 任务误差
   仅微降（seg −0.0004/−0.0019），AP-on vs off 为负或 ~0 → **不再开完整训练**；根因在目标/正则
   （decay 与任务梯度同量级、BCE 无中心项、zero-init 闭锁初始任务梯度），属 §3B/§4.4 设计修订范围。
2. ~~CL-RPF cache probe（§6.3/§4.1）~~ → 已执行（k1，6 类）：见 `04_clrpf_probe/CLRPF_PROBE.md`。
   **G1 FAIL**（最佳轨迹 0.2325 < 静态 mean_std 0.3099，Δ=−0.077）→ CL-RPF 轨迹形态归档为负；
   顺带复现静态多层≈A1（mean_std 0.3099 vs A1 0.3092）。
3. PRS / 共同坐标细节恢复 / 匹配目标 / 蒸馏：各自需单独冻结协议 + 预算决策。
4. 论文写作侧：把 A1 强基线、oracle 审计负结果、v4/v5 负结果按受限口径纳入证据矩阵与 related-work。
