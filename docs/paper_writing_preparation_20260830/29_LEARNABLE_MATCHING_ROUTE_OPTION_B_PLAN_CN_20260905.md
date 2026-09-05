# 可学习匹配路线（doc28 §8 选择 B）方向文档与下一轮计划

日期：2026-09-05  
上游决策：`28_V13_RESULT_AUDIT_AND_NEXT_DECISION_PLAN_CN_20260905.md`（doc28）§8 选择 B；§5.3/§6.1/§7 的冻结与稳定性门沿用。
状态：方向已确认（§10 Q1–Q4：**方案 E episodic 共享、L-A margin hinge、仅对角重标定、六类 LOO 循环**）；精确超参与命名在启动前一次性冻结为 A2_PROTOCOL。
基线：A1（DINOv2+CLIP-image 末端 patch 特征各自行归一化、0.5/0.5 拼接、再归一化、KNN，`s(x)=½d²` 距离图）。
复用与不复活：复用 v14 合法数据门与 support 合成渲染；**不复活** N1 JTD、旧 DNC-C、PMC tri、SCAIF 复杂交互头、NCPRA 双向回归、CASF 多层头。

---

## 1. 为什么在 v14 全部失败之后选这条路

v14（本轮已归档，见 `experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/FINAL_DECISION_CN.md`）在合法、support-only、图像域验证下证明了三类闭式机制不成立：

1. **DNC-I（手工 q 统计选通道）**：合成留出族上 1/3 通过（cutpaste k2 +0.035，erasure −0.036/−0.055），无类别相对 random-mean 稳定 ≥+0.02。
2. **DNC-C-fixed（跨分支去冗余）**：集合几乎不变（Jaccard 0.9981、冗余仅降 0.2%），AP 增益恒 0。
3. **半松弛容量 OT（P2-A 图像域）**：缺陷区容量 premium 低于正常背景 p95（三分支均负），图像重编码后信号消失。

共同教训：**手工拟合统计量（q、相关、容量势）与真实检测目标之间存在目标失配**——统计量不是从"合成缺陷 patch 应排在 clean/nuisance 之上"这一目标直接优化出来的；一旦放到真实图像重编码与跨图留出，机制增量归零。doc28 §8 因此明确：继续换手工聚合函数/闭式规则的边际价值很低，应转向**匹配目标驱动的轻量可学习度量**：

> 保留 A1 的信息通路，只学习一个极小、有界、身份初始化的每分支通道重标定；损失直接优化"合成缺陷 patch 的 normal-memory 距离排名高于 clean/nuisance"；用 support 内 leave-one-family-out 验收；成功作为 A2/新论文协议报告，不与 zero-training A1 混称同一设置。

这条路不新增 backbone、不改匹配结构、不引入大融合头，只回答一个被 v14 反复绕开的问题：**把打分本身作为优化目标学一点点，是否比任何手工规则更诚实、更可迁移。**

### 1.1 与历史可学习尝试的差异（防重复踩坑）
| 历史 | 做了什么 | 失败/归档原因 | 本路线如何不同 |
|---|---|---|---|
| NCPRA | 最终层双向 D↔C MLP 回归预测 normal，用残差做 anomaly | 全候选退化，best Δ≈−0.005 | 不做跨分支预测/残差；只做每分支**通道重标定**，保留原始距离语义 |
| CASF | 多层统计融合头（Wave 0 预门未过即停） | 仅 1/6 类有 headroom；头依赖大量统计特征 | 参数少 4–5 个数量级（1536 个有界标量），目标单一（排序），可解释 |
| SCAIF | 支持集条件化非对称交互融合 | 复杂交互；与 zero-training 定位冲突 | 本路线显式声明为 A2 新协议（可训练设置），不伪装成 zero-training |
| SCAIF loss | 恒正距离的无中心 BCE | 目标与排序不一致，易退化 | 直接优化**成对/排序 margin**（缺陷距离 > 正常/nuisance 距离） |
| DNC 系 | 闭式 q 排序/冗余选择 | 目标失配（§1） | 同一"排序目标"由优化器闭环，而不是手工统计 |

---

## 2. 目标与不目标

**目标（本方向的验收主张，逐条预注册）：**
1. 构建一个可学习的**每分支正对角通道重标定** `w_D, w_C ∈ [0, U]`（身份初始化），配合 A1 恒等旁路，使 worst case = A1。
2. 损失直接让"support 上合成的缺陷 patch（mask 内 cell）到 normal-memory 的融合距离"**排名高于**同一留出图的 clean 与 nuisance cell。
3. 在 support 内 leave-one-family-out / leave-one-image-out 上证明**留出排序改善**，且 normal 路径（clean/nuisance 打分）不坍缩。
4. 若通过，做**一次冻结**的真实 MPDD 门；通过后进入 seed1/2 稳定性门（doc28 §7）。整体作为 **A2/新论文协议**交付，不与 zero-training A1 同表混称。

**不目标（红线）：**
- 不训练/微调任何 backbone；不插入 in-backbone 交互或注意模块（本轮不做 SCAIF/Cross-Attention/Mamba/LoRA）。
- 不读取目标 `/test/good`；真实缺陷 GT 不进入训练与选择（无授权不改此条）。
- 不使用真实缺陷图像训练；合成干预只渲染在 manifest support 上。
- 本轮不做关系蒸馏到单分支（效率方向；除非用户把目标改为降低端到端成本）。
- 不复活 N1/旧 DNC-C/PMC tri/SCAIF/NCPRA/CASF 的失败实现。

---

## 3. 问题形式化

### 3.1 特征与打分
冻结分支特征（图像域重编码，非 token 编辑）：dino patch 网格与 clip patch 网格对齐到同一空间分辨率（沿用 v14 A1 网格对齐）。每 cell 特征经**对角重标定**后按 A1 协议融合：

```
d̃_i = rowL2norm( w_D ⊙ d_i ),   w_D ∈ [0,U]^768，初始全 1
c̃_i = rowL2norm( w_C ⊙ c_i ),   w_C ∈ [0,U]^768，初始全 1
z_i  = rowL2norm( [0.5 d̃_i, 0.5 c̃_i] )
s(q) = min_{r∈M} ½‖z_q − z_r‖²   （M = 留出结构下的 normal memory）
```

- 对角权重 = 每通道"保留/削弱"旋钮；恒等初始化 + 旁路保证不劣化起点。
- 上下界 `U`（如 1.5）防爆；下界 ≥0 保正。
- **阶段 II（仅当阶段 I 对角度量通过健康探针后）**：可选加一个低秩跨分支项 `w_D ⊙ d + A (w_C⊙c)` 或对角近似交互；不提前混入，避免一步引入不可归因的自由度。

### 3.2 训练目标：排序而非恒正距离
对 (cat, shot) 的留出结构：h 为留出 support 图，M 由其余 (K−1) 图构成。对 h 的第 e 个合成 episode（mask `m_e`），定义 cell 打分 `s_i = s(q_i)`，要求：

```
目标（"合成缺陷 patch 的 normal-memory 距离排名高于 clean/nuisance"）：
对于 mask 内缺陷 cell p 与对照 cell n（h 的 clean、以及 h 的光度 nuisance 变体），
  s_p 应显著高于 s_n。
```

**已确认损失（L-A，margin hinge；方向决策 Q2）：**
- 主损失：`Σ_{(p,n)} max(0, γ + s_n − s_p)`，对照 cell n 从该留出图 h 的 clean + 15 光度 nuisance 变体中均匀采样；γ、采样数启动前冻结。
- 公共正则：`λ_w·Σ(‖w−1‖²)`（向恒等收缩；留出不再提升时自动退回 A1），默认不开 L1 稀疏偏好。
- L-B（AUC/logistic 对）保留为**稳健性对照**（记录其在健康探针上的差异），不作主设置。
- **否定项**：不使用"恒正距离的无中心 BCE"，不使用全局二分类；样本单位是 cell/cell-pair，不以像素独立计数做 CI（doc28 §7 约束）。

### 3.3 为什么这会规避 v14 三类失败
- 对 DNC-I：q 换成从排序目标直接梯度学出的权重；目标是端到端使用的同一打分，无统计量→目标的失配。
- 对 DNC-C：去冗余由优化 + 正则承担（冗余通道若有用会被保留，无用会被压低），无需手工相关阈值 λ。
- 对容量 OT：本轮**不做**容量约束，先验证"纯标定是否带来合法增益"；若对角度量有效且用户同意，容量可作为阶段 II 的**匹配成本层**重新进入（不再是独立 OT 打分）。

---

## 4. 数据协议（合法，全部沿用 v14 门）

- **support-only**：manifest `/train/good/` 的 K-shot（k2/k4 先判；k1 仅自洽探索，不给独立泛化结论——doc28 §5.1 同规则）。
- 合成干预：复用 `ntof_render`（`SYNTHETIC_KINDS` 三族 × 固定种子；`REF_KEYS` 五族光度 nuisance ×3 强度），**在 support 图像 1024 渲染后经冻结提取器重编码**（禁止 feature-token 编辑）。
- 种子/强度规则冻结（沿用 v14 哈希种子 `v14::{cat}::{rel}::{kind}::s{i}` 或升级为 `a2::...`，启动前定）。
- 任何 fit/select/train ID 含 `/test/` 即硬失败（复用 `v14_common.assert_fit_ids_are_support` 语义，独立拷贝进 A2 脚本）。
- 留出结构：leave-one-family-out（3 族轮换）× leave-one-support-image-out；训练在 (K−1) 图 × (2 族 × 3 种子) + nuisance 上，验收在 h 的留族 episode。
- 真实评估只发生在冻结后的最终一次：真实 MPDD 异常图 + test/good 作为评估集。

### 4.1 权重共享方案（已确认：E，episodic 共享 + 六类 LOO）
- **主设置（方向决策 Q1/Q4）**：episodic 共享权重 `(w_D, w_C)`。六类内 leave-one-category-out 循环：每轮 5 个源类在该轮 support-synthetic episodic 上训练，1 个目标类零更新推理；k2/k4 分别声明。每类都当过目标，证据对称，且与 doc28 六类宏门一致。
- 源类的合成留出结构同样用 support 内 leave-one-family-out / leave-one-image-out（第 5 节健康探针在源类留出族上做）。
- 被否选项（记录）：方案 P（每类即时校准）作为测试时优化、推理多一步且难以与 zero-training 表述切割，本轮不采用；"E 共享初始化 + 每类 ≤10 步微调"只在 E 通过后作为消融考虑，不作主设置。

---

## 5. 优化健康探针（P0，先于任何真实门）

- 预算：每 (cat,shot) 组合 50–100 优化步（小 Adam），或按固定 epoch 数换算；步数上限冻结。
- 每 N 步在留出族上测：留出排序指标（缺陷 cell 的 s 相对 clean/nuisance 的分位差，或留出族 AP 代理）；并监控 normal 路径：h 的 clean 图打分分布相对 A1 的移动（应≤ 冻结阈值，如 p95 相对漂移 <+0.05，防"全局抬分"式坍缩）。
- **判定（冻结）**：在 ≥2/3 的 (cat,shot) 组合上出现"留出合成排序改善 ≥ 冻结正阈值 且 normal 路径不坍缩"才进入下一步；否则该设置归档。
- 控制（探针阶段同步跑）：等参数 concat 头、单分支头、随机标签、错配分支、固定随机权重同分布、A1 恒等。探针本身不改损失/步数来凑过门（只允许一次冻结超参档，见 §10 Q2）。
- 失败即按 doc28 §8 选择 A（收口 A1）或终止本轮，不再在本轮尝试 SCAIF 级复杂模块。

## 6. 真实门（P1，单次冻结）

沿用 doc28 §5.3 真实门文本（不按结果调通道数/λ/族/权重公式/损失）：
- 六类、同 shot 宏 ΔPixel-AP 相对 A1 ≥ **+0.006**；
- 每个已声明 shot 非负；按 shot 汇总 ≥5/6 类正；最差类 ≥ −0.01；
- 相对等维最强控制（random-mean / highvar / low_nui / 等参数 concat 头）≥ +0.003；
- 机制破坏控制：错配分支 / 随机标签 / shuffle-mask 的增益应消失或转负（记录差值）。
- 只允许**最多一个**真实胜出者进入 doc28 §7 的 seed1/2 稳定性门（三 seed 宏仍 +0.006、≥2/3 seed 总体正、每 shot 非负、最差 cat×shot×seed ≥ −0.02、控制差保持）。
- 报告定位限制：即使通过，也只作为 A2/新论文协议证据；**不得**写进 zero-training A1 的主方法表或 novelty claim。

## 7. 成本与算力
- 训练侧：每组合 ≤100 步 × 每步 batch（≤ 数千 cell-pair）→ 单卡/多核 CPU 分钟级；参数量 1536（阶段 I）。
- 推理侧：比 A1 多一次逐通道缩放 + 权重存储（KB 级）；匹配成本不变。
- 复用 v14 缓存（support 合成特征已导出，dino/clip × k2/k4，5.9 GB）可显著压缩本轮导出时间；但若训练需要留出图与 bank 的细粒度组合，仍以重新按 A2 协议导出为准，不把 v14 缓存当作已冻结输入（种子前缀若升级需重导出）。

## 8. 产物（建议目录）
```text
experiments/dynamic_fusion/innovation_a2_learnable_matching_2026090X/
  RUN_MANIFEST.json
  DATA_ROLE_AUDIT.json
  HEALTH_PROBE/   (P0 判定)
  CONTROL_TABLE.json
  REAL_RESULTS.json  DECISION.md
  A2_PROTOCOL.json   # 冻结：损失/超参/门/共享方案
FINAL_DECISION_A2_CN.md
```
脚本草案（启动前创建并单测）：
- `a2_export_support_synthetic.py`（复用 v14 导出流程，种子前缀按 A2 冻结）
- `a2_train_diag.py`（对角权重训练 + 健康探针输出）
- `a2_run_real_gate.py`（单次冻结真实 MPDD + 控制）

## 9. 可交给实验助手的提示（方向已确认，执行前只剩冻结超参与命名）
> 按 `29_LEARNABLE_MATCHING_ROUTE_OPTION_B_PLAN_CN_20260905.md` 执行 A2 方向：主设置为 **episodic 共享对角权重（方案 E）+ L-A margin hinge + 六类 LOO 循环**，每轮 5 源 1 目标、目标类零更新，k2/k4 分别声明。先建立 A2 manifest：只允许 manifest support ID；在 support 图像上渲染三族合成缺陷与五族 nuisance 并经冻结提取器重编码；不做 feature-token 编辑；不读 /test/good；不用真实缺陷。每源类先做 50–100 步健康探针，要求留出族合成排序改善且 normal 路径不坍缩；通过后才允许一次冻结的真实 MPDD 门；真实结果不得回调节点、损失、族或类别划分。失败即归档并如实报告；成功也只在 A2/新论文协议语境下主张，不混称 zero-training A1。

## 10. 决策清单（已确认 vs 启动前待冻结）
**已确认（本方向 Q1–Q4）：**
1. 权重共享：**E（episodic 共享，目标类零更新）**；方案 P 不采用。
2. 损失：**L-A（margin hinge，γ 值待冻结）**；L-B 仅作稳健性对照。
3. 阶段范围：**仅对角重标定**（低秩跨分支项 = 阶段 II，仅当阶段 I 过健康探针后另行立项）。
4. 类别划分：**六类 LOO 循环**（每轮 5 源 1 目标），k2/k4 分别声明。

**启动前需一次性冻结（写入 A2_PROTOCOL.json，冻结后不得按结果回退）：**
5. 超参档：优化器/步数上限（50–100 步档内固定）/γ/λ_w/权重上界 U/训练与探针 seed/对照 cell 采样数。
6. 论文定位与命名：确认 A2/新论文协议命名，与 zero-training A1 分离表述（doc28 §8 要求）。
7. 缓存策略：沿用 v14 support 合成缓存（dino/clip×k2/k4，5.9 GB）还是按 A2 种子前缀重导出（若升 `a2::` 前缀则重导出，避免隐式复用 v14 输入）。
