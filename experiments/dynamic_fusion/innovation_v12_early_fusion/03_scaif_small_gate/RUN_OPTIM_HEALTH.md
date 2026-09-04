# RUN OPTIM HEALTH — 短优化健康诊断（doc 26 §3A / §6.2）

date: 2026-09-04
authority: docs/.../26_CURRENT_CORRECTION_STATUS_AND_NEXT_BREAKTHROUGHS_CN_20260904.md §3A/§6.2
方法：固定源类 episode（预注册源类 connector + bracket_white，不按效果挑类），K=2 support，
query=≤4 def+2 nrm 图；patch 采样一次固定（P0-2 全图均匀）；步骤 {0,1,10,50}；
Adam lr 1e-3 wd 1e-3（coupled，同 CONFIG）；run A=CONFIG 损失（sparse 1.0），
run B=task-only（无 sparse 项）。support-bank transform 保持 no_grad（已登记训练选择）。
产物：`RUN_OPTIM_HEALTH.json`；代码：`scripts/.../run_r3_ef_optim_health.py`。

## 1. 诊断结果（两类 × 2 run × 4 step）

### 1.1 修正量级（item 1：||g·Δ||/||F||）
- **run A（CONFIG+sparse）**：d 侧修正量始终 ~0–1e-4，c 侧修正量 s50 收敛到 ~1e-4–9e-4；
  gate 0.10→0.072（s50）。交互通路输出≈0 → 与 v5 全折叠、main≡static2 一致。
- **run B（task-only）**：d 侧修正量仍 ~0–1e-4；c 侧修正量增长（bracket_white pair0 到 0.044、
  pair1 到 0.005；connector 到 0.007/0.0015），gate 维持在 ~0.10。即去掉 sparse 后 clip 侧
  交互确实能动，但 dino 侧几乎不动。

### 1.2 任务梯度是否到达交互通路（item 2）
- init（s0）：seg 梯度只到 dec（0.0004–0.0006），proj/mlp/gate = 0 —— zero-init 残差导致
  初始 seg 对 gate/proj/mlp 无梯度（wd=0 使通路闭锁），这是**初始化结构**造成的，不是 bug。
- s50 run B（无 sparse）：seg 梯度 proj 0.0018–0.0025、mlp 0.0016–0.0073、dec 0.0010–0.0028、
  gate 0.0004–0.0033 —— **任务梯度确实到达交互通路**（acceptance 前半句成立）。
- s50 run A（+sparse）：seg 梯度被压到 ~1e-4–4e-4，sparse 梯度到 gate 0.054、proj 0.015。

### 1.3 Adam coupled decay vs 任务梯度（item 3）★
s50 按角色对比 raw seg-grad L2 与 wd·||θ||（wd=1e-3）：

| 角色 | bracket_white run B: grad | decay | bracket_white run A: grad | decay |
|---|---|---|---|---|
| proj | 0.0025 | 0.0004 | 0.0003 | 0.0006 |
| mlp | 0.0073 | 0.0073 | 0.0004 | 0.0070 |
| dec | 0.0028 | 0.0025 | 0.0004 | 0.0004 |
| gate | 0.0033 | 0.0032 | 0.0002 | 0.0043 |

**decay 与任务梯度同量级（甚至更大，尤其 mlp/gate 与 run A）** → Adam coupled wd=1e-3 在
梯度被 sparse/尺度稀释后足以把交互权重压向 0。这直接支持 doc 26 §2.2 的候选解释：
"正则压制、梯度过弱、初始化与归一化的联合作用"——此处正则（decay）与任务梯度之比可实测为 ~1:1
到 ~20:1（run A mlp）。不能只靠"把 sparse 权重调小"解决。

### 1.4 固定 episode 任务误差（item 4）
- run A：seg 基本不降（connector 1.2759→1.2759，white 1.2209→1.2209）。
- run B（无 sparse）：seg 仅微降 connector 1.2759→1.2755（−0.0004）、white 1.2209→1.2190（−0.0019）。
  正常/异常采样分值几乎不动（pos mean ~0.15–0.25，neg mean ~0.06–0.07，恒正 → 无负 logit，
  与"BCE 无中心项"设计缺陷一致）。

### 1.5 交互 ON vs OFF（item 5，固定 episode）
- s0：ON==OFF（zero-init 恒等，逐位 0 差）✓。
- s50 run B bracket_white：row_rel=0.021、map 差 0.0011，但 **AP(on)=0.4826 < AP(off)=0.4901
  （−0.0074）** → 即使交互真正工作（去掉 sparse），也不改善本 episode 的 Pixel-AP，反而下降。
  connector：AP(on)≈AP(off)（+0.0002）。
- run A：ON≈OFF（通路已折叠）。

### 1.6 输入/激活尺度（item 6）
- 两分支 row-L2 均归一 → ud/uc 范数 78.4、support su/cv 16（恒等）；raw pd/pc 范数几百。
- support 距离特征 d_sup mean 0.05（run A）~0.12（run B）、c_sup mean 0.23~0.25（run A）随投影变化；
  mlp 输入（cat[ud,nc] 等）范数 ~111 稳定。

## 2. 判定（doc 26 §3A acceptance）

- 前半句（任务损失给交互通路有限非零梯度）：**满足**（run B s50 全角色非零；仅 init 因 zero-init
  残差闭锁，属结构非 bug）。
- 后半句（不靠 sparse 能降低固定 episode 任务误差）：**仅极微弱**（seg −0.0004/−0.0019，
  相对 0.03–0.15%；且 AP-on vs off 在能工作的 white 上为负）。**交互可测效果为无增益或负增益**。
- 结论：**不再开完整训练**（与 v4/v5 结论一致，无矛盾）。诊断表明折叠不是"模块太小"或"只差
  sparse 调参"，而是目标/正则层问题：
  1. Adam coupled decay（1e-3）与任务梯度同量级乃至压过任务梯度（mlp/gate 尤甚，run A 达 ~20:1）；
  2. BCE `10*s` 无中心项，正常样本恒正 logit，seg 对交互几乎无有效压差；
  3. zero-init 残差在 init 处闭锁 proj/mlp/gate 的任务梯度（依赖解码器先离 0）；
  4. support-bank stop-grad 与 query 归一化作用范围扩大（登记于 CORRECTION_NOTES §7）。
- 若继续，须先按 doc 26 §3B/§4.4 单独预注册设计修订（中心化 logits/ranking objective、显式
  残差范数约束、source 混合 episodic、明确 support stop-grad、decay 处理），不得并入"再修 bug"。
