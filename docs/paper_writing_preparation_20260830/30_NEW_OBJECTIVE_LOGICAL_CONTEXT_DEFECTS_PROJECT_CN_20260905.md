# 新研究目标立项：逻辑/上下文缺陷检测（Track-1），并规划 Track-2/Track-3 顺序推进

日期：2026-09-05
决策记录：用户确认「另立全新研究目标」= **逻辑/上下文缺陷检测（Track-1）先行**，并指示「有空把三个方向都尝试一下，顺序进行」→ Track-2 多层/中段互补性测量、Track-3 推理效率压缩将在 Track-1 结束后按序各立独立立项文档执行。
上游证据：doc28（v14 三条闭式线全败）、A2 健康探针（可学习对角度量 k2/k4 失败，宏 ΔAP +0.004/−0.0003）、doc23（NCPRA/CASF/SCAIF 历史）。

## 1. 为什么立项此目标（实证缺口，非偏好）
把历次实验的合成与真实证据按**缺陷类型**切开，缺口是结构性的：
| 证据 | 观察 | 含义 |
|---|---|---|
| v14 P1-B 合成宏 | erasure（结构破坏）留出 AP ≈0.88–0.99；cutpaste（复制正常块=上下文异常）仅 ≈0.53–0.70 | patch 相似度对**结构缺陷**已近饱和，对**上下文/组成异常**存在 ~0.2 以上的稳定缺口 |
| doc28 容量探针结论 | 复制型异常在 patch 特征上"低层仍像正常"→ 纯相似度机制结构性失明 | 需要**关系/上下文**归纳偏置，而不是再换相似度聚合 |
| 真实 MPDD | 含 `parts_mismatch`/`bend_and_parts_mismatch` 组（doc28 §6.2 已列为预声明机制子组） | 真实标签中确实存在此类异常；子组门不得替代六类总体门 |
| 历史教训 | DNC/CASF/SCAIF/容量 OT/可学习对角全败于"同构匹配微调" | 新目标必须换**任务框架**（从"补丁像不像"到"布局/部件关系对不对"） |

一句话目标：**在零缺陷训练、support-only 的纪律下，检测"组成部分关系/空间布局被破坏"的异常（逻辑缺陷），并在真实 MPDD 六类总体门与 parts_mismatch 子组门两侧同时接受检验。** 若冻结特征中连可分性都不存在（Probe-C1 失败），则立即归档 Track-1，进入 Track-2，不无限加模型。

## 2. 三条 Track 顺序路线图
| Track | 主题 | 触发 | 状态 |
|---|---|---|---|
| 1 | 逻辑/上下文缺陷检测（关系归纳偏置） | 用户已选，本立项 | **已归档**（Probe-C1：G-C1 失败，C1 cutpaste Δ=+0.014 < +0.05；见 `experiments/dynamic_fusion/innovation_t1_context_defect_20260905/TRACK1_DECISION.md`）→ 转 Track-2 |
| 2 | 多层/中段互补性测量（doc23 遗留空白：CASF Wave1/2 从未跑） | Track-1 归档后启动 | **已归档**（Probe-M1：G-M1 失败，M1 cutpaste Δ=+0.019 < +0.05；见 `experiments/dynamic_fusion/innovation_t2_multilayer_20260905/TRACK2_DECISION.md`）→ 转 Track-3 |
| 3 | 推理效率压缩（单分支蒸馏/coreset memory，目标=端到端成本） | Track-2 结束后 | **归档**：探针通过（Probe-E1 T1=50% coreset 三门全过）→ 立项 doc32/doc33 真实门 → **REAL_GATE_FAIL**（宏无损 Δ≈−0.007 但 connector 单类崩溃 −0.033/−0.053 两 shot 复现；几何 coreset 对像素精确排序非无损）。决策：`experiments/dynamic_fusion/innovation_t3_efficiency_20260905/REAL_GATE_DECISION.md`。三方向全部尝试完毕 |

纪律：每 Track 独立立项文档 + 预注册门 + 独立产物目录；失败即归档并如实报告，不跨 Track 混用结论。

## 3. Track-1 目标、不目标与红线
**目标：** 证明或证伪「在冻结 DINO/CLIP 补丁特征中存在可被轻量关系描述子利用的上下文/布局信息，能把上下文型异常（cutpaste 代理；真实 parts_mismatch 待最终门）从正常中分离」。
**不目标：** 不训练 backbone；不做端到端生成模型；不做图网络堆叠（先验证信息是否存在）；不引入外部数据/新数据集。
**红线（沿用并复用既有合法基础设施）：**
- 拟合/选择/记忆 ID 只允许 manifest support（复用 v14 数据门语义）；
- 训练（若有）只用 support 上渲染的合成扰动；不读 `/test/good`；真实缺陷 GT 不进训练/选择；
- 最终真实 MPDD 门为六类同 shot 宏门（±子组报告），不按真实结果回调节点；
- 失败即停止，不复活旧模块。

## 4. Probe-C1（预注册可行性探针：冻结特征里是否有可用的关系信息）
**问题：** 加入显式空间/邻域关系描述后，上下文型异常的分离度是否显著高于纯 A1 补丁相似度？
**数据：** 复用 v14 support 合成缓存（dino/clip k2/k4、32 网格、LOO memory），全部为 support 渲染+冻结重编码，无 /test/。
**描述子变体（只做评价，不训练）：**
- C0（基线）= A1 融合补丁特征 `z`（1536-D），`s=1−max_cos(cell, memory)`；
- C1 = concat(`z`, 3×3 邻域均值 `z̄`)，3072-D，刻画"该 cell 相对局部上下文的偏移"；
- C2 = concat(`z`, 上/下/左/右四邻 `z`)，5×1536-D，刻画方向性邻域结构。
（若 C1/C2 均不优于 C0 → 特征缺关系信息 → 归档 Track-1。）
**结构：** 每 (cat, shot∈{2,4}) × 留一族（主测 cutpaste=上下文异常；erasure 作结构缺陷对照；scratch 在 32 网格不可评，不计）：memory=其余 (K−1) 图 clean cells，评价=留出图 h 的留族 episode Pixel-AP（mask 内 cell 为正）。
**门（预注册默认，按结果不调）：**
- G-C1：cutpaste 留出族宏 AP：最优关系变体 − C0 ≥ **+0.05**（k2 或 k4 任一 shot 成立即可视为"信息存在"；两个都成立更强）；
- G-C2：同变体 erasure 留出宏 AP 相对 C0 ≥ **−0.01**（不得损害结构缺陷）；
- G-C3：normal 路径稳定：clean p95 与 nuisance image-max p95 相对 C0 上升 ≤ +10%。
- 通过 → 立项 Track-1 主线（关系一致性打分 + 排列/组成合成扰动数据，冻结细节见后续 Track-1 协议）；任一 shot 都不过 → 归档 Track-1 → 启动 Track-2。

## 5. 产物与成本
- 目录：`experiments/dynamic_fusion/innovation_t1_context_defect_20260905/`（本探针结果在此）；脚本 `scripts/innovation_t1_context_defect_20260905/`。
- 成本：仅 CPU 评价（Q×M×dim 距离）；k2 全 6 类 ≈ 10–20 分钟。
- 报告项：逐类×shot×族 AP、宏 Δ、normal 路径、每变体的实际运行配置。

## 6. 执行提示（给实验助手）
> 执行 Probe-C1：读 v14 support 缓存（dino/clip k2/k4, 32 网格）；构造 C0/C1/C2 三个补丁/关系描述子；按 (cat,shot,留族) 在 LOO memory 上算 Pixel-AP（cutpaste 主测、erasure 对照、scratch 不计）；输出逐类/宏 Δ 与 normal 路径；按 G-C1/G-C2/G-C3 判定"冻结特征是否有可用的关系信息"。全部只用 manifest support 合成缓存，不读 /test/good，不用真实缺陷。结果如实报告，失败即归档并转 Track-2。
