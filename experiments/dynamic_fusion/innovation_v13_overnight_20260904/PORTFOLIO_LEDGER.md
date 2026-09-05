# V13 OVERNIGHT PORTFOLIO LEDGER（doc27，2026-09-04 夜间轮）

计划：`docs/paper_writing_preparation_20260830/27_OVERNIGHT_INNOVATION_PORTFOLIO_PLAN_CN_20260904.md`
结果目录：`experiments/dynamic_fusion/innovation_v13_overnight_20260904/`
代码目录：`scripts/innovation_v13_overnight_20260904/`（run_w0_1 / run_w0_2 / run_n1_jtd / run_n2_cct / run_n3_dnc / run_n4_pmc）

## W0 身份审计（45′）
- 状态：DONE
- 假设：PRS 强度轴应按距离恒等而非参数序；detail-recovery/PSMF 所谓 "a1" 实为 MEAN_STD7。
- 结果：exposure/gamma 的像素 RMS 与 dino/clip 响应为 V 形且不随参数序单调；命名核对确认
  A1_FROZEN=0.309212(macro k1) 才为正 A1（static2 0.308382 / MEAN_STD7 0.309856）。
- 冻结 A1 末层 bilinear-56 六类 k1 相对 32 网格匹配：+0.0028（W0_baseline.json）。
- 基线身份：A1 frozen map = v3 cache（REFERENCES kind=a1）；ml-cache 与其逐 map 差 ≤ ~1e-5。
- 产物：W0_AUDITS.md / W0_baseline.json / W0_prs_axis_audit.json。

## N1 JTD（doc27 §5，50′）
- 状态：FAIL_MECHANISM（R1 双条件全面负向）
- 数据角色：normal-only（support LOO）拟合；GT 仅评估。配置：k2/k4 × 六类，Dirichlet α=1，
  gate 0.9，cap 5，w=0.1。对照：a1_raw/a1_rank/u_sum/u_max/shuf/no_gate。
- 修复记录：ECDF 初版 rankdata 未对齐排序导致非单调（Spearman=-0.26），修复后 Spearman=1.0。
- 宏（k2/k4）：cand 0.207/0.250 vs 最强独立/rank-only 0.324/0.340（差 -0.116/-0.091）、
  vs shuf 0.307/0.333（差 -0.099/-0.084）→ 归档。corner_R 全 0；tubes 真实配对远差于打乱。
- 时间 ~75′（含 toy+修复）。输出：N1_jtd/{RESULTS,TOY,N1_DECISION}.json|md。

## N2 CCT（doc27 §6，85′）
- 状态：R0 机制 PASS（探针），真实门 PROVISIONAL/TIME_BUDGET（未跑六类真实 AP）
- 假设：自由匹配掩盖"正常样式复制"；容量约束匹配可局部化检出。
- R0（六类，k1 特征域干预）：copy gap a1_free=0.000 vs dino_ot 0.079 / cct_i 0.044 /
  concat_ot 0.024；erase 各法 0.83–0.97；nuisance FP ≤0.0007；置换不变 0.0；copy gap 随面积
  单调。等价性：双计划容量 > concat-OT +0.020，但 JS 耦合增量 ~0，单分支 DINO 容量最高 →
  按 doc27 只记"单分支容量匹配改进"，撤"双分支融合创新"表述。
- 产物：N2_cct/{R0.json, EQUIVALENCE.md, N2_DECISION.md}。

## N3 DNC（doc27 §7，100′）
- 状态：R0 初筛 MARGINAL（cutpaste +0.046 vs best-random 通过；erase +0.009 / scratch +0.001
  未达 +0.02）；FP 受控（DNC p99/full ≤1.05）；dnc_c≡dnc_i → 无跨分支机制。
- 维度剪枝对 full concat 全族为正（+0.049/+0.020/+0.014，合成留出族）。
- 归档为"通道适配（DNC-I）观察（k1 合成族）"；不跑真实调 mask。
- 产物：N3_dnc/{R0.json, N3_DECISION.md}。

## N4 PMC（doc27 §8，55′）
- 状态：REDUNDANT(AP)/工程优化观察；R1 工程门宏达标、最差类（connector k2 0.014）超标。
- REDUNDANT 检查：tri 与 concat greedy Jaccard 0.53–0.72（集合不同）但 AP 相同（50% 0.3633
  vs 0.3634）→ 分支私有覆盖保护无 AP 增量。
- 50% 预算宏 loss concat_greedy 0.0027（≤0.003 ✓）；25% 时 greedy 明显优于 random（+0.023）。
- 产物：N4_pmc/{R0.json, N4_DECISION.md}。

## 条件路线 O1/O2
- 状态：NO_AUTH（未获用户授权，不启动）。

## 胜者复验
- 状态：无候选过真实门 → 未启用（doc27 §10 允许）。

## 跨路线
- 无组合调参、无测试集拟合、无外部数据、未提交代码、未覆盖旧结果。
- GPU：未新起 GPU 进程（仅快照 1791 MiB 其他任务占用）。CPU 为主，单进程顺序运行。
- 代码 hash 与精确命令：见 RUN_MANIFEST.json 与 FINAL_SUMMARY_CN.md。
