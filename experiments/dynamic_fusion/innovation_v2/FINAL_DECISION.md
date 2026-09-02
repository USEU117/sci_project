# A2 Innovation Program — 最终决策（FINAL_DECISION）

任务书：`docs/paper_writing_preparation_20260830/12_MULTI_ROUTE_ALGORITHM_INNOVATION_EXECUTION_AND_ACCEPTANCE_CN_20260902.md`
日期：2026-09-02

## 1. 结论

**六条路线共 27 个预注册候选全部在 MPDD seed0 Small Gate 失败。按任务书 §17 停止规则第 10 条与 §0 流程，A2 Innovation Program 早停，不做 Full MPDD、不做 winner selection、不做冻结验证。A1（冻结双编码器 KNN 基线）继续作为主方法。**

所有候选只在 MPDD（development）上竞争；BTAD / MVTec / VisA 未作为开发集使用，无需一次性验证（无新 winner）。

## 2. 候选结果总表（MPDD seed0 × shot {1,2,4}，Δ 为 Pixel-AP 相对 A1）

| 路线 | 候选 | mean ΔPixel-AP | 正 shot 数 | worst Δ | 控制名 | beats control |
|---|---|---|---|---|---|---|
| A LNDC | k3 | −0.115466 | 0/3 | −0.143412 | global_density_sham | 否 |
| A LNDC | k5 | −0.095361 | 0/3 | −0.123904 | global_density_sham | 否 |
| A LNDC | k9 | −0.085044 | 0/3 | −0.114123 | global_density_sham | 否 |
| B DSAM | translation_r2 | −0.174726 | 0/3 | −0.195537 | no_alignment_local_window | 是 |
| B DSAM | translation_r4 | −0.068456 | 0/3 | −0.077448 | no_alignment_local_window | 否 |
| B DSAM | translation_r8 | −0.011989 | 0/3 | −0.019003 | no_alignment_local_window | 是 |
| B DSAM | affine_r2 | −0.192037 | 0/3 | −0.207398 | no_alignment_local_window | 否 |
| B DSAM | affine_r4 | −0.082395 | 0/3 | −0.097324 | no_alignment_local_window | 否 |
| B DSAM | affine_r8 | −0.012953 | 0/3 | −0.021010 | no_alignment_local_window | 否 |
| C CEQA | q0.10_eta0.25 | +0.000561 | 3/3 | +0.000339 | a1_rank_only | 否 |
| C CEQA | q0.10_eta0.50 | +0.001022 | 3/3 | +0.000686 | a1_rank_only | 否 |
| C CEQA | q0.20_eta0.25 | +0.001519 | 3/3 | +0.001260 | a1_rank_only | 是 |
| C CEQA | q0.20_eta0.50 | +0.002799 | 3/3 | +0.001994 | a1_rank_only | 否 |
| D DEVA | geometry_tau0.90 | +0.000024 | 3/3 | +0.000017 | unfiltered_augmentation | 否 |
| D DEVA | geometry_tau0.95 | +0.000024 | 3/3 | +0.000017 | unfiltered_augmentation | 否 |
| D DEVA | photometric_tau0.90 | +0.000011 | 3/3 | +0.000001 | unfiltered_augmentation | 否 |
| D DEVA | photometric_tau0.95 | +0.000011 | 3/3 | +0.000001 | unfiltered_augmentation | 否 |
| D DEVA | combined_tau0.90 | +0.000000 | 1/3 | +0.000000 | unfiltered_augmentation | 否 |
| D DEVA | combined_tau0.95 | +0.000000 | 1/3 | +0.000000 | unfiltered_augmentation | 否 |
| E NCPRA | r32_lam0.10 | −0.005227 | 0/3 | −0.008490 | linear_ridge | — |
| E NCPRA | r32_lam0.25 | −0.005215 | 0/3 | −0.008008 | linear_ridge | — |
| E NCPRA | r64_lam0.10 | −0.005244 | 0/3 | −0.006891 | linear_ridge | — |
| E NCPRA | r64_lam0.25 | −0.008093 | 0/3 | −0.009560 | linear_ridge | — |
| F FAGR | mu0.10_iter1 | −0.007591 | 0/3 | −0.009747 | uniform_smoothing | 否 |
| F FAGR | mu0.10_iter3 | −0.007606 | 0/3 | −0.009777 | uniform_smoothing | 否 |
| F FAGR | mu0.50_iter1 | −0.005338 | 0/3 | −0.006890 | uniform_smoothing | 否 |
| F FAGR | mu0.50_iter3 | −0.005183 | 0/3 | −0.006761 | uniform_smoothing | 否 |

Small Gate 门槛（任务书 3.2）：mean ΔPixel-AP ≥ +0.003、≥2/3 shot 正、worst ≥ −0.005、无 NaN/泄漏、控制不得同样好。

## 3. 逐路线失败原因与"不应继续什么"

### A — LNDC 局部正常密度校准（3 候选，全负）
把测试距离除以其局部参考密度中位数让分数系统性退化（k 越小越差），并且**全局密度 sham 也是负的**（−0.007~−0.028），说明"密度除法"本身在 MPDD 上不携带比原始距离更多的异常判别力——归一化只放大了参考记忆本身的 patch 噪声。
不应继续：LOF / Isolation Forest / 大范围局部密度搜索（任务书 5.5 明确禁止）。

### B — DSAM 可变形空间对齐记忆（6 候选，全负，但机制 control 支持）
translation 与 affine 对齐都能稳定地略优于"无对齐 + 同半径窗口"控制（多数候选 beats control），**说明对齐机制本身有效**；但 L∞ 局部窗口把分数约束到对齐点邻域后系统性劣于 A1 的全局 KNN，且 R 越小越差（r2 ≈ −0.19 → r8 ≈ −0.01），R→∞ 收敛回 A1。
不应继续：更精细匹配（SIFT / 光流 / 更多 RANSAC 迭代）只会提高对齐精度，不会解决"局部窗口 vs 全局 KNN"的方向性劣化；该方向与 A1 的全局记忆打分冲突。

### C — CE-CQA 跨编码器共识查询适应（4 候选，全正但未达标）
唯一全线为正的路线（+0.0006~+0.0028），q0.20/eta0.50 达 +0.002799，距门槛 +0.003 仅一步；但：
1. 三个候选被 a1_rank_only 控制追平或反超（q0.20_eta0.50 不 beats control）——说明增益主要来自"按秩挑选查询内 pseudo-normal"这一思路本身，而不是双编码器共识的增量；
2. 由 0.10→0.20、eta 0.25→0.50 单调增大 shift，增益单调上升但未跨门槛，继续增大只是把 memory 移得更远，控制项同向上涨。
不应继续：扩大 q/eta 网格等于把候选预算花在追正结果上（任务书 §17.6 禁止）；不应换名重做（禁止列表含动态路由类方法）。

### D — DEVA 等变性验证的正常记忆扩增（6 候选，Δ≈0）
所有候选 ΔPixel-AP ≤ 3e-5，且 **tau 过滤候选与 unfiltered 控制逐位一致**（candidate == control），说明双编码器等变性滤波对 MPDD KNN 分数没有可观测影响；geometry/photometric/combined 三组增强无差别，说明增强 patch 与原记忆高度同质（MPDD 参考图内自相似高），扩增记忆不改变最近邻结构。
不应继续：更多变换类型 / 更多增强数量 / 调节 tau 都不会改变"扩增对最近邻无影响"的结论。

### E — NCPRA 正常样本预测残差适配器（4 候选，全负，已授权）
正常参考训练的 768→r→768 bottleneck 交叉预测残差作为异常证据系统性劣于 A1 距离本身（−0.005~−0.008）。预测残差与 A1 距离高度相关，但训练后带额外噪声。
不应继续：更大隐藏维 / 更长训练违反"轻量 normal-only"约束，且信号方向为负；增加骨干容量违反冻结口径。

### F — FAGR 特征亲和图细化（4 候选，全负）
4 邻域相似度加权 Jacobi 平滑使分数轻微退化（−0.005~−0.008），与 uniform 平滑控制几乎相同——特征亲和力加权的信息在 MPDD 上不高于均匀平滑，说明 DINO/CLIP 邻近 patch 亲和未给细化带来增益。
不应继续：更大 mu / 更多迭代只是更强平滑；换图拓扑（8 邻域、kNN 图）不在预算内且无机制依据。

## 4. 为什么这组负结果可信

1. **判据实现与任务书一致**（`run_small_gates.py::small_gate_decision`）：mean≥0.003 且 ≥2/3 shot 正且 worst≥−0.005 且无 NaN/泄漏且控制不反超。
2. 27 个候选全部在**冻结 A1 feature cache + 冻结 A1 打分**上做差值，A1 回归误差 <1e-6（`00_input_audit/AUDIT_REPORT.json`）。
3. 每份报告记录 config/code/input SHA256 + leakage_flags 全 false，marker 绑定 config hash（无陈旧复用）。
4. 每条路线都有任务书指定的 sham/control；DSAM/CEQA/FAGR 的 control 结果证明"机制在起作用但方向/幅度不足以超过 A1"。
5. 无测试标签 / mask / 验证集参与任何开发决策。

## 5. 保留的证据（全部归档，不删除）

- 代码：`src/industrial_ad/innovation_v2/*.py`（common + 6 路线）
- 预注册配置：`configs/innovation_v2/route_{a..f}.yaml`
- 运行器：`scripts/innovation_v2/{audit_inputs,run_small_gates,export_deva_references}.py`
- 单测：`tests/innovation_v2/`（A/B/C/D/E/F/common 全部通过）
- Wave 0 审计：`00_input_audit/AUDIT_REPORT.json`
- 逐候选报告 + 控制：`01_small_gates/<ROUTE>/<CANDIDATE>/{report.json,_control/report.json}`
- 每路线决策：`01_small_gates/<ROUTE>/SMALL_GATE_DECISION.json`
- DEVA 增强导出缓存（只增正常参考，已校验 462 行）：`outputs/dynamic_fusion/innovation_v2_deva/`

## 6. 决策与论文口径

- **正式决策：ARCHIVE（A2 全部路线），A1 PROMOTE 保持为最终主方法**。
- 论文不新增 A2 方法贡献；A1 零训练可复现口径不变。
- 负结果可作为 few-shot frozen protocol 的失败边界讨论素材（任务书 §16：最多 3 点贡献，不列工程步骤为创新）。
- A1 / RCEC / 历史负结果证据均未修改。
