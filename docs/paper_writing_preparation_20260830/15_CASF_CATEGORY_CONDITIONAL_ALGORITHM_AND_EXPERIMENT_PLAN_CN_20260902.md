# CASF 类条件算法创新与实验方案（A4 入选路线正式任务书，2026-09-02）

引用关系：
- 上位任务书：[14_BROAD_ALGORITHM_INNOVATION_PORTFOLIO_CN_20260902.md](14_BROAD_ALGORITHM_INNOVATION_PORTFOLIO_CN_20260902.md)（§10 D3 诊断通过 → 本文件）
- 被细化的原规格：[13_REPRESENTATION_LEVEL_BREAKTHROUGH_EXECUTION_AND_ACCEPTANCE_CN_20260902.md](13_REPRESENTATION_LEVEL_BREAKTHROUGH_EXECUTION_AND_ACCEPTANCE_CN_20260902.md)（§5 路线 G、§7.1 小门、§8 Full MPDD、§13.2 论文口径）
- 诊断证据：`experiments/dynamic_fusion/innovation_v4_diagnostics/D3_supervision_value/D3_SUMMARY.json` 与逐类 JSON；总览见同目录 `README_STATUS.md`

本文件只立项 **1 条路线：CASF（类条件互补性感知合成异常融合）**。D1/D2 诊断失败，SF-NM、DC-SZoom、RG-MCR、RG-OT 暂归档，不进入本任务书。本任务书只写到「小门/验收的完整设计」为止；任何实现与跑分须等用户批准后再执行。

---

## 0. 为什么 CASF 必须改成类条件设计（诊断依据）

A4 的 D3 诊断（MPDD development，seed0/k2，双参考 LOO，asymmetric vs symmetric 各 12 伪异常 episode，
tiny logistic 头，pseudo-Dice @0.5 阈值）逐类结果：

| 类别 | asym Dice | sym Dice | headroom (asym−sym) | 诊断 |
|---|---:|---:|---:|---|
| bracket_black | 0.056 | 0.095 | −0.040 | 不支持 |
| bracket_brown | 0.675 | 0.088 | **+0.587** | 强支持 |
| bracket_white | 0.069 | 0.010 | +0.059 | 支持 |
| connector | 0.070 | 0.116 | −0.046 | 不支持 |
| metal_plate | 0.041 | 0.021 | +0.020 | 支持（边缘） |
| tubes | 0.129 | 0.309 | −0.180 | 不支持 |

汇总：mean headroom **+0.0668 ≥ 0.02**；`categories_supporting_casf = 3`。

> **Wave 0 实测修正（2026-09-02，正式 Gset 以此为准）**：上表为 12-episode/单 seed 的早期小探针。
> 按 §2.4 放大探针（24ep/家族 × 3 family seeds）复测后，6 类 mean headroom 变为
> bracket_black +0.0026 / bracket_brown **−0.0168** / bracket_white **+0.0593** / connector −0.0425 /
> metal_plate +0.0157 / tubes −0.0414；**冻结 Gset = {bracket_white}**（详见
> `experiments/dynamic_fusion/innovation_v5_casf/Wave0_gate_probe/PROBE_SUMMARY.json` 与
> `experiments/dynamic_fusion/innovation_v5_casf/README_STATUS.md`）。bracket_brown 早期 +0.587 系小样本下
> sym 训练崩溃，非稳健信号；metal_plate 亦不足 0.02。§0 表格保留为历史证据，不再作为立项依据。

**结论（强制设计约束）**：跨分支非对称伪异常监督的价值**不是全局同质**的——早期小探针提示 bracket 族与
metal_plate 上有正信号而 connector/tubes 为负；放大探针进一步把可学习分歧收敛到 bracket_white 单类。
无论哪一版探针，把非对称监督无差别强加到全部 6 类都会拖累全局均值。因此：

1. 单一全局融合头 + 全部 6 类的固定伪异常配方 → **禁止**（这正是原 13 号 §5 的隐含假设，被 D3 证伪）；
2. 正确设计 = **类别级证据门控（谁用 CASF 由合成机制探针决定）+ 类别级难度标定（每类只按自己 normal
   LOO 校准强度）+ 共享小评分头**；
3. 门控与标定的所有信息源只允许 development 正常参考与合成 episode，**禁止**接触真实 mask/测试统计。

---

## 1. 科学假设（修订版）

原假设（13 号 §5.1）：在正常 patch manifold 附近构造难度受控、分支非对称的伪异常，可教会极小评分头
识别"某一分支或某一尺度率先偏离正常"的模式。

修订为**类条件版本**：

> 双编码器的"何时该信任哪一支"策略不是类别无关的。用**类别级合成难度探针**先判定该类是否存在
> 可利用的分支非对称信息，再只在这些类别上启用非对称伪异常监督，训练一个共享的、可解释的小评分头；
> 其余类别保持冻结 A1。相比"全局统一伪异常配方"，类条件 CASF 把合成监督的收益集中到机制上真实成立的
> 类别，从而在不接触任何测试信息的前提下，让互补性监督在 pooled 指标上可转化。

对应可证伪预测（都在小门处检验）：
- **P1**：对 CASF-active 类别，候选 vs A1 的 per-cat ΔPixel-AP 为正且能支撑 pooled-6 门槛；
- **P2（机制新增）**：带类条件门控的候选 pooled mean ≥ 无门控（全 6 类启用）版本，说明"门控"本身不是噪声；
- **P3（继承 13 号）**：候选同时胜过 CTRL-SYM 与 CTRL-NODIS——收益来自跨分支非对称分歧信号，而非合成监督或参数量本身。

---

## 2. 算法设计

### 2.1 输入统计量（每个 patch，分支 b∈{D,C}）

沿用 13 号 §5.2，只含可解释统计量，不喂 raw token：

1. 到该类正常 memory 的 1-NN cosine distance `d_b`（deepest 层 DINO/CLIP，来自冻结 A1 同一缓存）；
2. normal reference LOO median/MAD 归一化 `z_b = (d_b − med_b)/MAD_b`，裁剪 `[−5, 10]`；
3. 分歧通道：`Δ = |z_D − z_C|`，`sign·Δ = sign(z_D − z_C)·|z_D − z_C|`；
4. 保留冻结 A1 score `s_A1`（唯一与 mask 无关的全局证据）；
5. 可选 3×3 局部稳健对比 `c_b = z_b − median_3×3(z_b)`。

输入通道默认 **4 维（z_D, z_C, Δ, sign·Δ）**；加 context 的候选再并入 `c_D, c_C`（≤6 维）。

### 2.2 类条件伪异常生成器（只在正常 reference feature 上构造）

沿用 13 号 §5.3 家族，增加**类别级调度**：

- **扰动模式**：`dino`（只扰 DINO）、`clip`（只扰 CLIP）、`both`（同时扰）；每个类别内按
  预注册比例生成；**asymmetric 家族 = {dino, clip, both} 1:1:1**；**symmetric 控制家族 = {both} 全量**；
- **扰动算子**：正交 tangent noise（保持单位模长）+ distant transplant（同类别不同参考/远距正常 block
  替换）；若 Wave 1 多层导出通过，追加 cross-layer break（只替换深层/只替换早层）；
- **mask**：矩形与不规则连通块各 50%，面积按 log-uniform 覆盖 patch map `[0.5%, 15%]`；
- **类别级难度标定（新）**：每类只依据本类 normal reference LOO 的 `s_A1` 分位数定标，使伪异常强度
  落在 clean LOO 的 `P70–P99.5`；gamma 候选网格 `{0.2, 0.35, 0.5}` 只按该类 held-out 伪 episode
  Dice 选择（dev-only 合成信息，禁看真实 mask）。
- 每类保存固定 64 个样例（mask/类型/强度/z-map）供人工检查；feature/mask 可入库，不复制原图。

### 2.3 融合头与损失

沿用 13 号 §5.4 预注册结构：

```text
input stats → 1×1 Linear(C,32)+GELU → [可选 depthwise 3×3 context] → Linear(32,1)
s_final = s_A1 + ρ_c · tanh(residual)
```

- 参数量上限 50,000；backbone 完全冻结；
- `ρ_c` 为该类正常 LOO score IQR 预先固定（只看正常，不看测试）；
- clean patch A1-preservation loss + 伪异常 mask focal BCE + soft Dice + branch-dropout；
- 总损失 `L = L_focal + L_dice + 0.2 L_clean + 0.1 L_branch`，系数不随真实异常调整；
- 训练上限 100 epoch、patience 10，epoch 只由 held-out normal/pseudo episode loss 选择；
- **类别门控的推理实现**：CASF-inactive 类别直接 `ρ_c = 0`（等效纯 A1），head 对它们不产生 delta。

### 2.4 类别门控规则（确定性、预注册，禁止手挑）

在**正式小门运行之前**执行一次标准尺度的合成探针（与 D3 同协议但升至 24 episode/家族 × 3 随机种子，
GPU 无关、可 CPU 完成），对 6 类分别输出 `mean(hr_c)`：

```
c ∈ Gset（CASF-active） ⇔  mean(hr_c) ≥ +0.02  且  ≥ 2/3 seed 的 hr_c ≥ +0.02
```

- 探针只使用正常参考与合成 episode；**不接触任何真实异常**；
- 探针代码即 `innovation_v4_diagnostics` 的 D3 探针的正式放大版，禁在探针阶段另加参数；
- 得到的 `Gset` 必须写入小门报告头部（先于任何小门分数），允许被本文档引用的 D3 结果（seed0/k2 小探针）
  作为先验说明，但正式 `Gset` 以放大探针为准；
- 禁止：按小门真实分数反推门控集合。

### 2.5 与 A1 的关系

- 训练前：`ρ_c=0` 时逐值回归 A1（单元验收）；
- 训练后：只修改 score map，不改变 A1 特征与记忆库本身；
- 论文身份：通过前维持 A1 零训练口径；通过后按 13 号 §13.2 改为"lightweight normal-only adaptation"。

---

## 3. 具体算法创新点（可直接转写为论文 claim）

创新点必须**同时**满足：有可消融证据、不换名复活任何已归档路线、不依赖测试信息。每条附支持证据与风险。

1. **互补性感知的"先偏离者"监督（cross-branch asymmetric pseudo-anomaly supervision）**。
   用只扰动单分支的受控伪异常，显式教模型"某一编码器局部统计先离开正常流形"的信号，而非全局权重或
   无监督可靠性公式。证据：D3 mean headroom +0.0668、CTRL-SYM/CTRL-NODIS 将被设为小门强制门槛。
   风险：易被评"合成+切块组合"，故必须有 CTRL-SHUFFLE 接近失败、参数量 50k 上限等机制证明。

2. **类别级机制门控（category-conditional mechanism gating）——本任务书的新增主张**。
   先在 development-only 正常参考上用放大版合成探针判定每类的可学习分歧信号，预注册
   `Gset = {mean hr ≥ +0.02 且 2/3 seed 达标}`；只有这些类别启用非对称监督，其余保持冻结 A1。
   支持证据：D3 逐类 headroom 呈强类条件（+0.587 / +0.059 / +0.020 vs 三类为负）。
   这是把"哪一类值得训练"也变成**由可审计规则、而非全局固定配方**决定，直接回应"全局统一合成监督
   在多类混合数据上被拖累"的诊断发现。

3. **类别级难度标定 + 可解释小头的受控融合（interpretable anchored residual fusion）**。
   `s_final = s_A1 + ρ_c·tanh(residual)`，ρ_c 仅由该类 normal LOO IQR 固定；干净区有
   A1-preservation，头输入为 4–6 维可解释统计量而非 raw token。证据：A1 冻结证据不受影响，
   参数量、VRAM、时间全程记录。

4. **分支分歧的可审计统计（signed disagreement）**。除 `|z_D−z_C|` 外引入 `sign(z_D−z_C)·|Δ|`，
   使头能区分"D 先偏"与"C 先偏"两种互补性方向；CTRL-NODIS（去掉 Δ 与 sign·Δ 通道）作为强制门槛。

创新幅度定位（对齐 14 号 §11）：中等创新。与 A1 自然衔接、能做完整消融；论文叙事为
"合成监督 + 类条件门控"而不是新网络或动态路由。

---

## 4. 预注册候选与控制

最多 3 个候选，全部启用类条件设计（§2.2/§2.4）：

| ID | 层统计 | context 3×3 | 启用条件 | 用途 |
|---|---|---|---|---|
| P0 | deepest only | 无 | 总是 | 类条件合成监督能否转化真实增益（最小机制） |
| P1 | deepest only | 有 | 总是 | 局部上下文是否增益 |
| P2 | deepest + 早层 | 有 | 仅当 Wave 1 多层导出与身份回归通过 | 多层统计是否增益 |

强制控制（每个控制必须与候选同 episode 预算、同难度目标）：

- `CTRL-A1`：原始冻结 A1；
- `CTRL-SYM`：只在 symmetric（both-only）家族上训练，去掉非对称互补监督；
- `CTRL-NODIS`：删除 `Δ` 与 `sign·Δ` 通道；
- `CTRL-DINO`：只用 DINO 单分支统计，参数量匹配；
- `CTRL-SHUFFLE`：训练 mask 与 corruption 位置随机错位——预期接近失败；
- `CTRL-NOGATE`：同一头对全部 6 类启用（无 §2.4 门控）——专门检验"类条件门控"主张；
- `CTRL-BIG` 禁止（不能以扩大容量替代机制证明）。

候选间只保留一个 winner：字典序 mean ΔPixel-AP > worst-shot > 参数少。

---

## 5. 数据集角色与保护边界（不变）

- **development**：MPDD 6 类（小门 seed0×shot{1,2,4}；Full MPDD 3 split seeds×3 shots，加 3 train seeds）；
- **冻结外部验证**：BTAD / MVTec AD / VisA——在唯一 winner 冻结前不可见，禁止任何形式的触碰；
- 伪异常只能从 MPDD 正常 reference 构造；labels/masks 只由 evaluator 在分数写完后加载；
- A1 冻结证据（`submission_repro_20260827/`、`experiments/dynamic_fusion/freeze/`）不修改；
- 不改 13/14 号既有结论文档，不换名复活 A2 归档路线。

---

## 6. 实验方案与波次

```text
Wave 0  协议审计 + 放大合成探针（§2.4）→ 输出预注册 Gset 与 6 类探针全表
Wave 1  多层特征导出与身份回归（仅 P2 需要；含 13 号 §2 Wave 0 的 D/E 有效性复核不并入本路线）
Wave 2  P0/P1(/P2) 与全部控制：MPDD 小门（seed0 × shot{1,2,4}）
Wave 3  小门 winner → Full MPDD（3 split × 3 shot × 3 train seed）稳定性复验
Wave 4  唯一 winner 冻结（代码/配置/checkpoint hash）→ BTAD+MVTec+VisA 一次性验证
Wave 5  论文口径与决策文档
```

停止规则：小门无候选通过 → 停止，A1 保持；Full MPDD 不达标 → 冻结前停止；
外部验证失败 → 不能测试第二名（一次性纪律）。

### 6.1 MPDD 小门（沿用 13 号 §7.1，作 pooled-6 说明）

数据：MPDD seed0 × shot{1,2,4} × train seed{0,1,2} × 6 类。每候选以训练种子均值汇总 shot delta：

- 三-shot mean ΔPixel-AP vs A1 `≥ +0.005`（**pooled-6**：CASF-inactive 类贡献 0，此门槛要求 active 类
  的真实增益足以覆盖三类零贡献后仍达 +0.5 个点）；
- 3/3 shot 均值为正；worst shot mean delta `≥ −0.003`；
- train-seed Pixel-AP 标准差均值 `≤ 0.005`；
- mean ΔPixel-AUROC `≥ −0.002`；mean ΔImage-AP `≥ −0.010`；
- 相对 `CTRL-SYM` mean Pixel-AP `≥ +0.002`；相对 `CTRL-NODIS` `≥ +0.001`；
- **机制新增**：带门控版本 pooled mean ≥ `CTRL-NOGATE` 版本（否则"类条件门控"主张不成立，退回普通 CASF）；
- 无泄漏、NaN、错位、复现失败。

### 6.2 Full MPDD 与最终选择（沿用 13 号 §8）

小门每路线至多 1 候选进入 3 split seeds × 3 shots × 3 train seeds。最终选择须满足：
mean ΔPixel-AP vs A1 `≥ +0.008`；≥8/9 split/shot 配置为正；train-seed 稳定性同小门；
机制控制门槛同小门。通过后才冻结并进入一次性外部验证。

### 6.3 报告必须公布的内容（防 cherry-pick）

- 放大探针 6 类全表（含 negative 类）与由此推导的 `Gset`；
- 小门全部候选与全部控制分数（不只保留 winner）；
- 6 类逐类 ΔPixel-AP（inactive 类必须显示为 ≈0 而非隐去）；
- 训练随机性：同 seed 两次 score 最大绝对误差 `<1e-7`；换 train seed 改 checkpoint hash 不改 split。

---

## 7. 稳定性、随机性与泄漏纪律（对齐 13 号 §3）

- 随机性来源全部显式固定（numpy/torch 种子、episode 采样顺序）；数据 split 与种子解耦；
- 泄漏硬门：训练/验证选 epoch 代码路径不可读真实 mask/异常；`evaluator` 单点加载 GT；
- 几何纪律：heatmap resize 与坐标方向有单元测试；至少一张 debug 图确认 mask/残差/最终图方向正确；
- sample/ref ID、几何、复现三项测试先行（任何真实异常评价之前）。

## 8. 单元与技术验收测试清单

1. synthetic mask 与被修改 patch 精确一致；
2. tangent noise 后 L2 误差 `<1e-5` 且与向量点积绝对值 `<1e-4`；
3. clean 输入、`ρ=0` 时逐值回归 A1；
4. 同 seed 两次 score 最大绝对误差 `<1e-7`；换 train seed 改 checkpoint hash、不改 split；
5. `CTRL-SHUFFLE` 合成验证 Dice 不得高于正确标签模型；
6. 训练代码无法访问真实 test masks；参数 ≤50k；峰值 VRAM/时间自动记录；
7. 门控探针可复现：给定种子输出相同 `Gset`；
8. `s_final = s_A1 + ρ_c·tanh(residual)` 中 ρ_c 由 normal LOO IQR 得出，有测试锁定。

## 9. 交付物

- `scripts/innovation_v5_casf/`、`src/industrial_ad/innovation_v5_casf/`、`tests/`、
  `experiments/dynamic_fusion/innovation_v5_casf/`（命名待执行时统一，避免与 v4 诊断混放）；
- 放大探针报告 + 预注册 `Gset`（JSON）→ 小门报告 → Full MPDD 报告 → 冻结清单 → 外部验证报告；
- `FINAL_CASF_DECISION.md`（通过/归档，含逐类证据）；
- 全部控制在 6 类的逐类分数表。

## 10. 论文口径变化条件（沿用 13 号 §13）

- 通过前：A1 零训练、normal-only 口径不变；
- 通过后：允许写"lightweight normal-only synthetic-anomaly adaptation + category-level gating"；
  不得声称动态路由或新 backbone；Discussion 需交代类条件门控在外部数据集上的泛化不确定性。

## 11. 风险与边界（诚实申报）

- **绝对 Dice 偏低**：多数类 pseudo Dice ≤0.31，说明合成监督本身难度/表达能力有限，bracket_brown 的
  +0.587 由 sym 崩溃驱动，须由 CTRL-SYM/难度匹配解释，不能直接外推；
- **pooled-6 门槛偏紧**：active 类须在真实 MPDD 上产生可观增益才能让 3 类零贡献后仍过 +0.005；
- **门控的跨数据集泛化未知**：Gset 依 MPDD development 合成探针得出，外部验证集上不重选门控；
- 若 P1–P3 任一不成立 → 本任务书自动停止并归档，A1 保持主方法。

## 12. 给执行 AI 的指令（批准后生效）

1. 新建上述五目录（以 innovation_v5_casf 为基），Wave 0 先跑放大合成探针并冻结 `Gset`；
2. 复用 innovation_v4_diagnostics 的 D3 探针与 guard/evaluator 基础设施，禁止复制改造测试标签；
3. 不实现 DC-SZoom/SF-NM/RG-MCR/RG-OT；不跑 BTAD/MVTec/VisA；
4. 先完成 §8 验收清单再进入小门；任何失败按停止规则归档。
