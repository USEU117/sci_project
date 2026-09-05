# N3 / DNC：通道选择 R0 初筛决策（doc27 §7）

日期：2026-09-04（夜间轮）。实现：`scripts/innovation_v13_overnight_20260904/run_n3_dnc.py`（CPU）。
数据：k1 缓存（ntof 干预 + syn 掩码）。normal-only 拟合；合成掩码仅评估用。

## 1. 设计（冻结常数）

- 通道响应（每分支每通道）：def_resp(g,kind) = median(值[掩码内]) − median(值[整图])
  （自居中，无需 clean good）；nuisance 尺度 = 15 个光度变体在支持图上的逐通道位移中位；
  正常尺度 s_j = MAD×1.4826。q_j = median|def_resp|/s_j ÷ (nuisance_j + 0.05)。
- 选择：DNC-I = 每分支 top-256(q)；DNC-C = 带跨分支冗余惩罚(λ=0.3, 同 episode 响应相关矩阵)
  的贪心，配额 256/分支；对照：random×3、highvar、low_nui、dino_only/clip_only(各 512)；
  full=1536 全部通道（A1 旁路）。
- 评估：留出族（3 合成族轮换）上 32×32 网格 reduced 融合 KNN（A1 协议）逐图 AP 均值；
  FP 代理 = 光度无缺陷 episode 的 p99 距离（相对 full 比值）。

## 2. 结果（六类宏平均，n=6/格）

| 留出族 | DNC-I/DNC-C | best random | low_nui | highvar | full | dnc−bestrand | dnc−low_nui | dnc−full |
|---|---|---|---|---|---|---|---|---|
| cutpaste | 0.7961 | 0.7501 | 0.7262 | 0.7308 | 0.7467 | **+0.046** | +0.070 | +0.049 |
| local_erasure | 0.9316 | 0.9227 | 0.9099 | 0.9001 | 0.9114 | +0.009 | +0.022 | +0.020 |
| thin_scratch | 0.7633 | 0.7619 | 0.7404 | 0.7263 | 0.7493 | +0.001 | +0.023 | +0.014 |

- DNC-C 与 DNC-I 输出**完全相同**（逐族相等）→ 跨分支响应冗余惩罚在 L11/L24 原始通道上
  不改变选择（分支间相关性弱或 q 主导）；按 doc27 §7「只 DNC-I 有效则改称通道适配」。
- FP（光度 p99 相对 full）：DNC-I 均值 0.878、最大 1.05（≤+10%）✓；dino_only 1.49×（clip
  分支是强 FP 制动器）；clip_only 0.41×（太保守，AP 也最低）。
- 维度剪枝本身有效：DNC-I 在全部三族都超过 full 1536 维 concat（+0.049/+0.020/+0.014）。

## 3. 判定

- doc27 §7 R0 门：合成留出族相对 nuisance-only/random 缺陷 AP ≥ +0.02 且 ≥2 族成立、正常
  FP 不恶化 >10%。
  - 相对 best-random：仅 cutpaste +0.046 达标，erase +0.009、scratch +0.001 不达标 → 严格读法
    1/3 族。
  - 相对 low_nui（nuisance-only 基线）：3/3 族 ≥ +0.02。
  - 结果 **边缘/部分通过**。最稳健的信号：cutpaste（复制类合成缺陷）+0.05 级提升且 FP 受控。
- 结论：不进入真实异常调 mask 的 R1/性能门（doc27：R0 未明确达标则不跑真实调 mask）。
  归档为 **DNC-I 通道适配（channel adaptation）候选观察（合成留出族，k1）**：
  - 维度剪枝相对 full A1 有 +0.01~0.05 合成 AP 收益（噪声通道稀释确实存在）；
  - 无跨分支互补机制证据（DNC-C≡DNC-I），不包装为交互/融合创新；
  - 真实 MPDD 缺陷是否同样受益未验证（合成-真实迁移局限已记录）。

## 4. 产物
N3_dnc/R0.json（rows 含 per-cat×family×method 的 ap 与 fp_p99）、本决策。
