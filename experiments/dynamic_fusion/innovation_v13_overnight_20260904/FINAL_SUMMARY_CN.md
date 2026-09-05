# V13 夜间创新组合 FINAL_SUMMARY_CN（doc27，2026-09-04 轮）

计划：`docs/paper_writing_preparation_20260830/27_OVERNIGHT_INNOVATION_PORTFOLIO_PLAN_CN_20260904.md`
（本轮 ~23:53 起，单会话连续执行，CPU 为主；未启用 GPU 新进程，未触碰其他任务。）
全部产物在 `experiments/dynamic_fusion/innovation_v13_overnight_20260904/`；代码 hash 见
RUN_MANIFEST.json。可复算命令见文末。

---

## 1. 真正完成了哪些机制，哪些只做了探针，哪些因前提/时间没做

| 路线 | 完成层级 | 结论 |
|---|---|---|
| W0 基线身份审计 | 完整（45′） | 冻结 A1 六类 k1 macro=0.309212；确认 detail-recovery/PSMF 中 "a1" 实为 MEAN_STD7（0.309856），正 A1 另有其人；PRS exposure/gamma 响应 V 形、不随参数序单调；冻结 A1 末层 bilinear-56 k1 +0.0028（廉价工程线，非新候选） |
| N1 JTD | 机制 FAIL（完整 k2/k4×6 类） | R1 双条件全面负向 → 归档。**完成**：toy+真实初筛+归因；修复了一个实现 bug（CDF 秩未对齐，Spearman −0.26→1.0）并保留修复记录 |
| N2 CCT | R0 机制探针 PASS；真实门未跑 | 容量约束匹配能检出"正常样式复制"（A1 自由匹配完全失明）；非 concat-OT 退化；JS 耦合无增量、单分支容量主导 → 只记"匹配改进"。真实 MPDD 门 PROVISIONAL/TIME_BUDGET |
| N3 DNC | R0 初筛 MARGINAL | cutpaste 留出族 +0.046（>+0.02 通过）；erase/scratch 对 best-random 不足；dnc_c≡dnc_i → 通道适配观察，无跨分支机制；不跑真实调 mask |
| N4 PMC | REDUNDANT(AP) + 工程记录 | tri 保护与 concat greedy AP 相同 → 工程优化；50% 宏损失 0.0027 达标但最差类（connector k2）0.014 超标 |
| O1/O2 | 未做（无授权） | doc27 §9 条件备选，需用户批准新增训练/效率目标 |
| 胜者复验 | 未启用 | 无候选过真实门 |

## 2. 有没有超过 A1 且超过必要控制的结果（shot/类别/seed）

**没有**能同时超过冻结 A1 与必要控制、且在真实掩码上成立的结果。

- N1（真实掩码 k2/k4）：candidate 宏 0.207/0.250，低于 rank-only 控制 0.309/0.335 与打乱
  0.307/0.333（差 ≥ −0.08）；任何类别×shot 都不胜 rank-only。
- N2（真实掩码未跑）：R0 探针显示容量匹配 > concat-OT（+0.020）但 < 单分支 DINO 容量
  （0.079 vs 0.046）→ 不能按 doc27 表述为双分支创新；真实门未跑，无 seed 声明。
- N3（合成留出族掩码）：DNC-I 相对 full concat 在三族均为正（+0.049/+0.020/+0.014），
  仅 cutpaste 相对 best-random 超 +0.02；seed0 单配置，非真实缺陷。
- N4：不涨点目标；工程损失门宏达标、最差类超标。

## 3. 有没有不涨点但更便宜的 Pareto 结果（端到端 vs 匹配耗时）

- N4 coreset（k2/k4，MPDD 六类真实掩码）：50% 预算下 concat-greedy coreset 宏 Pixel-AP 损失
  0.0027（≤0.003 达标）；匹配阶段 bank 减半 → 匹配距离计算约 2× 快（≥25%）。**非端到端**
  （双编码器推理与后处理不变）。最差类 connector-k2 损失 0.014 → 整体工程门未过。
- N3 通道剪枝（合成族证据）：512 维 concat（256/分支）相对 1536 维 full 反而更好 → 匹配维度
  缩减的存储/算力收益方向存在，但仅 k1 合成证据，未达真实门。
- 结论：无端到端 Pareto 声明；匹配阶段成本方向有初步工程证据（coreset/剪枝），待真实门。

## 4. 哪条机制值得下一轮，哪条应停止

值得（带条件）：
- **N2 容量匹配（单分支/DINO 视角）**：前提/机制探针强（复制类异常 A1 结构性失明已被实证），
  但需 (a) 真实对应缺陷族（MPDD 复制/占比类，若存在）上的真实掩码 AP；(b) 16→32 网格、
  k2/k4；(c) 锚点=支持集（非自锚）的完整对照。建议以"CCT 的容量失配作为 A1 的补充旁路"形式
  预注册下一轮，不是替代 A1。
- **N3 DNC-I 维度剪枝**：三族一致超过 full concat（尽管幅度不一），是"哪些信息进入融合"的低
  成本初步证据；下一轮应在真实掩码上先做冻结选择的**一次性诊断**（不做 per-class 调参），
  并增加 clip-only 与随机通道的方差复验。

应停止：
- **N1 JTD**（联合尾部稀有度，8×8 直方图版）：真实配对上为负，corner 无负依赖；停止。
- **N3 DNC-C**（跨分支互补选择）：与 DNC-I 输出完全相同，无机制；停止交互表述。
- **N4 tri/分支私有覆盖保护**：AP 与 concat greedy 等价；只保留 concat coreset 工程。

## 5. 已排除的具体实现 vs 仍不能排除的大命题

已排除（本轮具体实现）：
- 8×8 Dirichlet 平滑 + `rankF+0.1R` 的 JTD（含 rank-saturation 边界规则与打乱配对对照）在
  该特征/校准下无正信号；正常尾(1,1)corner 无负联合依赖。
- 平衡熵 OT + JS 行耦合（γ=0.5）的 CCT 双计划没有超过单分支容量；concat-OT 等价性排除。
- 末层原始特征逐通道 "inside−all" 响应的 DNC-C 冗余惩罚不改变选择。
- concat-farthest vs tri minimax coreset 在 AP 上不可区分（50%）。

仍不能排除：
- 联合/交互机制在**训练型/可学习匹配**（O1）、更早层（Stage2 特征）、或更高分辨率联合统计
  上有效——本轮零训练缓存证据不支持即停止，不是这些大命题的反证。
- N2 在真实重复/占比缺陷族上是否检出、N3 在真实掩码上维度剪枝是否保值，均未测（缺真实
  对应缺陷族/预算），不能下结论。
- O1/O2 未获授权未运行，不能排除其可学性。

---

## 成本账单（估算，CPU 单进程；GPU 0 新增）

| 段 | 约耗时 | 主要资源 |
|---|---|---|
| P0/W0 | 1.0 h | CPU（faiss/sklearn/numpy） |
| N1（含 toy 与 bug 修复） | 1.25 h | CPU |
| N2（R0 探针 6 类） | 1.0 h | CPU |
| N3（6 类×3 留出族） | 0.5 h | CPU |
| N4（12 cat-shot） | 1.2 h | CPU |
| 收口/写报告 | 0.8 h | — |
| 合计 | ≈5.7 h | 峰值内存 <8 GB（16 GB 机），磁盘新增 <0.5 GB，GPU 未新增进程 |

## 可复算命令（hash 见 RUN_MANIFEST.json）

```text
python scripts\innovation_v13_overnight_20260904\run_w0_1_prs_axis_audit.py
python scripts\innovation_v13_overnight_20260904\run_w0_2_baseline.py
python scripts\innovation_v13_overnight_20260904\run_n1_jtd.py --toy
python scripts\innovation_v13_overnight_20260904\run_n1_jtd.py            # k2/k4×6类
python scripts\innovation_v13_overnight_20260904\run_n2_cct.py            # R0 六类
python scripts\innovation_v13_overnight_20260904\run_n3_dnc.py            # 六类×3族
python scripts\innovation_v13_overnight_20260904\run_n4_pmc.py --shots 2 4
```

## 早晨 5 问答速览（doc27 §12）

1. 完成：W0 完整；N1 完整机制测试；N2/N3 机制探针/初筛；N4 工程记录。探针：N2 真实门、
   N3 真实门、N4 最差类优化。未做：O1/O2（无授权）、胜者复验（无候选）。
2. 无"超 A1 且超必要控制"的真实掩码结果。
3. 无端到端 Pareto；匹配阶段 coreset 50% 宏损失 0.0027、约 2× 快（工程方向）。
4. 下一轮：N2 容量匹配（单分支补真实门）、N3 DNC-I 剪枝真实诊断；停止：N1、DNC-C、tri。
5. 已排除上述具体实现；训练型/更早融合（O1/Stage2）与真实复制族上的容量假设仍开放。

---

## 附录 A：N1 逐类 × shot（真实掩码 pooled Pixel-AP@56，仅 N1 有完整六类真实结果）

| cat | shot | a1_raw | a1_rank | cand | u_sum | u_max | shuf |
|---|---:|---:|---:|---:|---:|---:|
| bracket_black | 2 | 0.0146 | 0.0086 | 0.0070 | 0.0072 | 0.0069 | 0.0086 |
| bracket_brown | 2 | 0.0334 | 0.0320 | 0.0228 | 0.0346 | 0.0308 | 0.0291 |
| bracket_white | 2 | 0.1060 | 0.0167 | 0.0096 | 0.0296 | 0.0330 | 0.0161 |
| connector | 2 | 0.2991 | 0.2966 | 0.1825 | 0.3260 | 0.1846 | 0.2908 |
| metal_plate | 2 | 0.8784 | 0.8522 | 0.8266 | 0.8450 | 0.9097 | 0.8532 |
| tubes | 2 | 0.7307 | 0.6461 | 0.1962 | 0.6998 | 0.5573 | 0.6426 |
| bracket_black | 4 | 0.1856 | 0.0744 | 0.0489 | 0.0467 | 0.0844 | 0.0743 |
| bracket_brown | 4 | 0.0340 | 0.0305 | 0.0227 | 0.0339 | 0.0272 | 0.0301 |
| bracket_white | 4 | 0.0954 | 0.0131 | 0.0070 | 0.0230 | 0.0235 | 0.0135 |
| connector | 4 | 0.3750 | 0.3626 | 0.2235 | 0.3661 | 0.2160 | 0.3479 |
| metal_plate | 4 | 0.8850 | 0.8594 | 0.8460 | 0.8513 | 0.9240 | 0.8584 |
| tubes | 4 | 0.7550 | 0.6705 | 0.3496 | 0.7205 | 0.6425 | 0.6743 |

cand 无一胜 a1_rank；u_max 仅在 metal_plate 胜 a1_raw（2/12），无普适性 → 不构成候选。

## 附录 B：N4 逐类 × shot（50% 预算损失 vs full，concat_greedy / tri_greedy）

| cat | shot | full | cg50 loss | tri50 loss |
|---|---:|---:|---:|
| bracket_black | 2 | 0.0146 | +0.0002 | −0.0000 |
| bracket_brown | 2 | 0.0334 | +0.0023 | +0.0023 |
| bracket_white | 2 | 0.1060 | +0.0009 | −0.0005 |
| connector | 2 | 0.2991 | **+0.0142** | +0.0107 |
| metal_plate | 2 | 0.8784 | +0.0013 | +0.0023 |
| tubes | 2 | 0.7307 | −0.0002 | +0.0021 |
| bracket_black | 4 | 0.1856 | +0.0016 | +0.0011 |
| bracket_brown | 4 | 0.0340 | +0.0019 | +0.0016 |
| bracket_white | 4 | 0.0954 | +0.0002 | +0.0027 |
| connector | 4 | 0.3750 | +0.0098 | +0.0088 |
| metal_plate | 4 | 0.8850 | −0.0008 | +0.0015 |
| tubes | 4 | 0.7550 | +0.0015 | −0.0008 |

宏损失 cg 0.0027 / tri 0.0027；最差类 connector-k2 cg 0.0142 超 0.01 → 工程门最差类未过。

## 附录 C：复现记录（同会话内复核）

- W0.2 重跑：macro A1_frozen 0.309212 / bilinear56 0.312057 / Δ+0.0028，ml-vs-v3 parity
  max 9.28e-6 → 与 W0_AUDITS 一致（connector/metal 第 6 位小数 ±1e-6 属 float32 舍入）。
- N1 toy 重跑：dep/ind=0.1169/0.0238、单调不变 0.0 → 与 TOY.json 逐位一致。
- N1/N2/N3/N4 脚本 hash 见 RUN_MANIFEST.json。

