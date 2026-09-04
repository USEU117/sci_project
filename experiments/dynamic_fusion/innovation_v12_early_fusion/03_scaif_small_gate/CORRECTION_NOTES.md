# CORRECTION NOTES — SCAIF v5 bugfix-only 修正与受限结论（doc 25 §6A 第 6 条）

date: 2026-09-04
authority: docs/.../25_OVERNIGHT_VALIDATION_CODE_AUDIT_AND_NEXT_STEPS_CN_20260904.md §6 阶段 A
supersedes: `FAILURE_ANALYSIS.md` §4 中三条“已排除/已受控解释”的**绝对措辞**与
`STAGE1_DECISION.md` §0 中“整族关闭/实现正确/稀疏约束无效”的**未限定表述**。
被修订文档保持历史快照不变（归档只读），本文件为唯一修订入口。

## 1. 为什么需要这份修订（doc 25 代码审计结论）

代码审计在归档实现中发现确认缺陷，因此归档文件里三条“已排除”解释不能按原措辞保留：

- **P0-1**：`SCAIF.refine()` 返回的 gate 被 `.detach()`，稀疏损失 `mean|g|` **没有梯度进入门**。
  归档里“把 sparse 权重 0.05→0.5→2.5 都无法让门稀疏”的实验实际上从未给稀疏项梯度，
  只是把权重乘在了一个常数损失上 → **“稀疏约束已尝试且无效”不成立**。
- **P0-2**：训练 query patch 只用 `qflat[:, :QSUBS]`（32×32 row-major 的前 512 列 =
  只采样图像上半幅），下半幅从未参与训练监督 → “训练不足/样本覆盖”并非真正受控。
- **P1-3**：`no_support`/`shuffled` forward 有维度/索引错误；`dino_only`/`clip_only`
  冻结一半参数量，与 main 差约 2×，不是“同预算对照” → **“整套机制对照已执行、
  整族已证伪”不成立**。
- 另有 support/query 距离尺度不统一（identical token 距离≠0）与 checkpoint/log 缺失。

## 2. 修订后的受限结论（代替原绝对结论）

保留（观测不受实现缺陷影响的部分）：
- gate=0 恒等自检行级 0 误差、map AP 与 static2 逐位一致；a1/static2 参考值与 Stage0/A1
  基准复现一致（本次修正后再次复现，见 §3）。→ 这一条不撤回。
- “v4 归档运行（该实现+该监督下）宏平均 Pixel-AP 低于 A1 与自身 static2”是**该次运行的观测事实**。

撤回/限定（被 P0/P1 缺陷污染的部分）：
1. ~~“实现正确性问题已排除”~~ → **撤回**。P0-1/P0-2/P1-3 为确认缺陷；v4 结果是在
   门无稀疏梯度、训练只覆盖上半幅、部分对照实现错误的前提下产生的。
2. ~~“稀疏约束无效/门必然饱和”~~ → **撤回**。稀疏项梯度被切断，v4 从未真正施加稀疏约束；
   “门饱和是监督本质导致”的机制性结论证据不足，仅可表述为“v4 运行中门饱和为观测事实”。
3. ~~“该整族（缓存级跨分支门控残差）已证伪 / early learnable fusion family closed”~~
   → **限定**。已证伪的只是“带 bug 的 v4 实例（训练目标使然）低于 A1/static2”；
   整族结论需要一次 bugfix-only correction 重跑（doc 25 §6B）后才可判定，未跑之前
   不得写“family closed / 主线归档定论”。

## 3. 阶段 A 验收证据（doc 25 §6A 验收：单测全过、稀疏梯度非零、整幅可采样、控制不报错、A1/static2 复现）

- 回归单测 `tests/test_scaif_correction.py`：13/13 PASS（CPU, .venv-anomalyclip）。
  - 稀疏惩罚梯度非零且随权重 λ=1→2 线性翻倍（P0-1 回归）；
  - 40×4×512 采样覆盖 1024/1024 行且上下半幅都 >0（P0-2 回归）；
  - 7 个 variant（main/no_support/shuffled/symmetric/no_cross/dino_only/clip_only）
    forward/backward smoke 全通过、参数量与 main 差 ≤5%（P1-3 回归）；
  - identical-token support 距离 <1e-5（尺度统一回归）；gate=0 行级 == f0（恒等回归）。
- `--mode selfcheck`：PARAM_COUNT 255,236（≤300k）；gate0 rows max-abs-diff=0；
  gate0 map AP == static2 AP（逐位一致）；a1 AP 复现（bracket_black k1：a1 0.010484 /
  static2 0.011754，与归档 REFERENCES.json 逐位一致）。
- 真实数据训练 smoke（60 步，bracket_black fold，GPU）：loss 平滑下降，gate 0.100→0.058
  （稀疏梯度已实际驱动门），patch coverage **1024/1024** min=154，log/ckpt 正常落盘，eval 正常。
- `--mode references`（全 6 类 × k{1,2,4} × {a1,static2}）：数值与归档 REFERENCES.json 一致
  （见该命令输出；REFERENCES.json 已原地重写，内容不变）。

## 4. 代码改动清单（对应 doc 25 §6A 第 1–5 条）

| 条目 | 文件 | 改动 |
|---|---|---|
| §6A-1 稀疏梯度 | `scripts/.../scaif_common.py` | `refine` 不再 `.detach()` gate；docstring 注明训练需图 |
| §6A-1 整幅采样 | `scripts/.../run_r3_ef_scaif.py` | `train_step` 全图均匀固定 seed 采样，同一索引作用于 s/GT/A1 reference；`np.bincount` 记录 1024 列覆盖 |
| §6A-2 尺度统一 | `scaif_common.py` | `PairBlock.forward` query 侧也先 L2 再 cdist（identical-token 距离≈0） |
| §6A-3 控制修复 | `scaif_common.py` | no_support 门输入维 2u（原 2u+2 错位）；shuffled 偏移改整图 roll（原 5-D buffer 索引错误）；冻结逻辑按 `active_dirs` 且对称 zero-init |
| §6A-3 参数匹配 | `scaif_common.py` | `single_dir_matched_u` 为 dino_only/clip_only 自动加宽使可训参数 ≈ main（≤5%） |
| §6A-5 checkpoint/log | `run_r3_ef_scaif.py` | `runs/log_{v}_{cat}.csv` 逐步日志 + `runs/ckpt_{v}_{cat}.pt`（state_dict/config 标记 CONFIG_v5_correction/覆盖统计） |
| 测试 | `tests/test_scaif_correction.py`（新增） | 见 §3 |

未在本阶段改动（留给后续决策，见 §5）：branch_drop 仍为 CONFIG 声明未实现；BCE 无中心项、
源类顺序效应、目标函数/private-path 设计修订属 doc 25 §6C，不在 bugfix 范围。

## 5. 阶段 B 结果（doc 25 §6B，2026-09-04 已执行）

运行：`.venv-anomalyclip python run_r3_ef_scaif.py --mode runs --variant main --tag correction`
（600 步/fold × 6 held-out，seed0，真实 MPDD s0 特征；log/ckpt 落 runs/，汇总 runs/main_correction_all.json；
归档 main_all.json = v4 未改动。机器 RAM/GPU 共享繁忙，中途曾 OOM，已加 resume：有 ckpt 的 fold 跳过训练只补 eval。）

结果（宏平均 pooled Pixel-AP @56，6 held-out cats）：

| 配置 | k1 | k2 | k4 |
|---|---|---|---|
| A1 | 0.309212 | 0.343699 | 0.388328 |
| static2（gate=0 基线） | 0.308382 | 0.348602 | 0.384439 |
| main v4（归档，buggy） | 0.261294 | 0.323109 | 0.331993 |
| **main correction（v5）** | **0.308382** | **0.348602** | **0.384439** |
| Δ correction vs A1 | −0.0008 | +0.0049 | −0.0039 |
| Δ correction vs static2 | 0.0000 | 0.0000 | 0.0000 |
| gate 饱和（g>0.9·cap） | 0.000 | 0.000 | 0.000 |

关键观测：**18/18 个 cat×shot 单元 correction AP 与 static2 逐位相等（0 误差），训练后门收敛到 ~0（gate mean 0.10→~0.0025）**。
P0-1 修复后稀疏梯度真实生效，优化器把门完全关掉 → 训练后的模块退化为自身的 gate=0 静态基线。
与 v4（门饱和到 cap、退化为常开破坏性修正）形成对称对照：无论门被推向“常开”（v4，因稀疏项无梯度）
还是被压到“全关”（v5，因稀疏项有梯度），跨分支门控残差在 MPDD s0 leave-one-cat 设定下都不能产生
高于 A1 / static2 的可学习增益。

按 doc 25 §6B：main correction 未出现稳定改善（Δ vs A1 均值 ~0.000 ≪ +0.006 门槛，Δ vs static2 = 0），
且门全关意味着模块不工作 —— **停止本实现（缓存级 SCAIF），不进入 Stage 2，不做 Stage2 in-backbone bridge**。
原“整族已证伪”结论仍保持受限（只覆盖本实现两个版本 v4/v5 + MPDD s0 dev），但不影响“本路线到此停止”的操作决策。

## 6. 未决（不自动执行，等待 supervisor 决策）
- ~~doc 25 §6B 附项：Stage0 oracle 空信息控制审计~~ → **已执行**（2026-09-04）：
  02_stage0_probe/ORACLE_NULL_AUDIT.md + ORACLE_NULL_AUDIT_k{1,2,4}.json。
  real oracle Δ 复现 +0.3885；**scale 空信息对照 Δ=+0.65 > real**；a1copy=0；shuffle≈0/负
  → 归档 headroom 主成分为尺度/GT 特权，oracle 入场判断口径废止（doc 25 §5）。
- doc 25 §6C：目标函数/原始流保护的设计修订（需单独预注册 + 预算决策）。
- doc 23 §7 声称满足的原始特征 parity 门（≤1e-5）按 doc 25 §5 需 amendment（当前为 Pixel-AP 门），
  属 Stage0 记录问题，另行处理 → **已闭环**：00_protocol/AMENDMENT_20260904_PARITY_GATE.md。

## 7. 复核登记（doc 26 §3B/§3C，2026-09-04）
### §3B 训练选择/归一化范围（记录，非已证结论）
1. **support-bank transform 停止梯度（run_r3_ef_scaif.py train_step）**：训练时
   `fr, _, _ = model.refine(query...)` 带图，而 `frs = model.refine(support...)` 在 `torch.no_grad()`
   下执行。这是**训练选择**，不是自动成立的"无梯度需求"。v4 与 v5 的差异不能全部归因于 gate-detach
   修复；support 侧同一可学习模块变换 support 但不回传梯度，会使模块只针对 query 学而不同步更新
   support 编码。保留该选择，但正式登记（doc 25 §6C 已列 support stop-grad 为需预注册项之一）。
2. **query 投影归一化的作用范围扩大**：v5 把 query 侧 `pd/pc` 输出 L2 后再进入 cdist，同时
   `ud/uc`（归一化后的 query 投影）也进入交互 MLP 与 gate 输入。即"统一距离尺度"顺带改变了
   MLP/gate 的输入尺度（由未归一化 → 归一化）。诊断需区分"只统一距离尺度"与"统一所有交互输入尺度"，
   不启动层/宽度超参搜索（doc 26 §3B）。
3. 仍属未处理原设计问题（doc 26 §3B 列）：BCE logits 用 `10*s` 且 `s>=0`（正常样本无法负 logit）；
   private stream 未作为独立路径保留；gate cap 不单独保证残差范数受限；源类按类别顺序训练；branch-drop
   仍为配置声明未见实现。以上属独立设计修订，不并入"再修 bug"。

### §3C 控制的科学含义（命名与边界，doc 26 §3C）
1. **dino_only / clip_only 更名为"单方向修正控制"**：v5 中二者只是冻结一个方向的残差/门
   （active_dirs 限制可更新方向），最终 static feature 仍含两分支（另一分支以原始行参与 concat）。
   不声称是"单编码器方法"。若真要表达单编码器，query/support/gate/最终 scoring 全部只用一个编码器。
2. **shuffled 是"错位控制"（misalignment），不是随机无信息控制**：小位移 roll（seed 固定 ±3 内）后
   3×3 邻域仍保留大量局部相关性；不能当作"完全破坏 D↔C 空间对应"的空信息对照。恢复机制实验时应补
   真正打乱对应且保持边际分布的控制。
3. **训练型机制控制完整矩阵未跑**：主模型（v4/v5）无增益，按 doc25/26 不为此花预算全跑；
   不把 smoke PASS 表述为控制有效性通过。
4. **generalization 边界**：MPDD 六类仍是 development；seed1/2 只换正常参考选择、不提供全新测试图，
   不能表述为"独立数据集确认"；源-目标类别分离 ≠ 外部数据集泛化。
