# V11 Oracle Audit — R0 决策（2026-09-03）

协议：`R0_PROTOCOL.json`（含 amendment：背景默认 A1 回退语义）
脚本：`scripts/innovation_v11_regret_router/run_r1_expert_oracle_audit.py`
依据：doc 21 §4.6（R0 信息价值门）+ §10 Phase 1 + §13 停止规则 1/4。
专家池：E0 A1 / E1 原始 text 区域证据（v8 text_maps, robust01）/ E2 LLSE residual /
E3 CSS 图内自一致性。统一 56×56 stride-8 grid、同 sample_ids（字符串对齐）。
Oracle：evaluator 端离线分区 = GT 缺陷 8-连通块 + 背景（默认 E0，选择性回退语义）；
缺陷块内选 mean robust01 最高的专家并放置其 raw map。

## 结果（MPDD seed0/k1，全部 6 类）

| 类 | A1 AP | Oracle AP | Δ | 单专家 AP (A1/text/LLSE/CSS) | 缺陷像素选择 (E1/E2/E3) |
|---|---:|---:|---:|---|---|
| bracket_black | 0.0105 | 0.2393 | +0.229 | 0.010/0.022/0.008/0.014 | 24% / 16% / **59%** |
| bracket_brown | 0.0344 | 0.9036 | +0.869 | 0.034/0.013/0.032/0.013 | 7% / **68%** / 25% |
| bracket_white | 0.0848 | 0.1153 | +0.031 | 0.085/0.022/0.088/0.010 | **100%** / 0% / 0% |
| connector | 0.1261 | 0.9999 | +0.874 | 0.126/0.030/0.132/0.010 | 0% / **92%** / 8% |
| metal_plate | 0.8713 | 0.9998 | +0.128 | 0.871/0.484/0.884/0.170 | 0% / **99%** / 1% |
| tubes | 0.7282 | 0.9985 | +0.270 | 0.728/0.264/0.764/0.024 | 4% / **86%** / 10% |
| mean | 0.3092 | 0.7094 | **+0.400** | — | **0.6% / 97.2% / 2.3%** |

LOO 掉点（mean Δ 减少）：E0 0.572 / E1 0.023 / E2 0.223 / **E3 −0.039**。

## 门判定
- g0 identity：PASS（1.2e-5）。
- g1 mean Δ≥+0.020：PASS（+0.400）。g2 ≥4/6 类 ≥+0.010：PASS（6/6）。
- **g3 ≥2 个非 A1 专家各 ≥10% 被选缺陷像素：FAIL**——聚合选择 97.2% 归 LLSE；
  text 0.6%、CSS 2.3%（CSS 仅在 bracket_black 达 59%，text 仅在 bracket_white 单个
  小缺陷块被选）。
- **g4 任一核心专家 LOO 掉点 ≥0.003：FAIL**——去掉 CSS 后 headroom 反而上升
  （−0.039），说明 CSS 在全局 oracle 中不增加覆盖（仅 bracket_black）。
- g5 单类占比≤50%：PASS（36%）。

## 结论：RSR Oracle R0 FAIL → RSR 停止（按 doc 21）
Oracle headroom 幅度充足但**结构失衡**：实际 ≈ "在大块缺陷类上把 blob 值换成 LLSE"，
text/CSS 只覆盖零星小缺陷（数~百像素）。doc 21 §4.6："没有专家互补上限时，任何
router 都只是在过拟合"；§13 规则 1/4：Oracle headroom 只来自单一机制（LLSE 系）
即停止。**{A1, text, LLSE, CSS} 池不足以支撑 regret-router 立项**。

## 解读与后续（doc 21 Phase 4 分叉）
- 不构成对 RSR 思想的否定：只证明**当前池不平衡**（LLSE 系近乎全胜大块缺陷；
  text/CSS 互补度在全数据集像素口径下 <10%）。补入新的独立机制专家（如
  SubspaceAD，需 giant 权重/GPU）或专门为小缺陷设计的专家后才值得重审。
- 按 doc 21 Phase 4："Oracle 不足 → 停止 router，转 **BC-MCR**（盲中心掩码上下文
  修复）创造新的结构证据"——这是当前文档指定的下一实验。
- 观察：bracket_white 的 GT 缺陷块仅 1 cell（≈64px），该块唯一被 text 选——小缺陷
  正是 A1 与 LLSE 都失败的类；text 在此有真实但极稀疏的贡献。
