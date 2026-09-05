# V13 夜间结果独立审计与下一步决策计划

日期：2026-09-05  
范围：`innovation_v13_overnight_20260904` 的结果、脚本与数据来源。  
性质：结果审计及后续计划；**本轮没有重跑实验、修改 V13 结果、启动训练或访问新的外部数据。**

## 1. 修正后的总判断

昨晚完成了约 5.7 小时的 CPU 探索，工程执行和归档较完整，但**没有产生可以直接升级为论文主方法的创新模块**。最有价值的收获不是某个 AP 数字，而是两个可继续验证的机制线索：

1. 自由最近邻对“复制已有正常样式”存在结构性盲点，容量约束可以在人工特征探针中暴露该变化；
2. 1536 维全部进入匹配未必最佳，部分通道可能稀释合成缺陷信号。

但这两个线索目前都不能直接进入真实结论：N2 只完成特征域、同图、平衡 OT 探针；N3 存在数据角色冲突，DNC-C 又有使其必然等同 DNC-I 的选择逻辑问题。

因此，下一步不应直接补三 seed，也不应立刻尝试更多新模块。先用一轮**协议修复后的决定性验证**回答两个问题：

- 容量信号能否从“直接复制 feature token”迁移到图像域和真实缺陷？
- 合法地只用 K-shot support 选择通道后，真实 MPDD 缺陷是否仍然受益，修正后的跨分支去冗余是否真的改变选择？

只有其中至少一条通过必要对照，才进入稳定性验证。若两条都失败，建议停止继续堆 zero-training 手工融合规则，转向明确授权的“匹配目标驱动可学习模块”，或者收口 A1 论文，不继续无边界找突破口。

## 2. 夜间结果的可信度分级

| 项目 | 夜间结果 | 审计后的级别 | 当前可说什么 |
|---|---|---|---|
| W0 基线身份 | 真 A1 k1=0.309212；七层 mean_std7=0.309856；旧恢复/相位文件的 `a1` 命名不是真 A1 | **可信** | 旧恢复/相位阴性结果只能约束 mean_std7 配方，不能无条件外推到冻结 A1 |
| W0 bilinear-56 | 冻结 A1 k1 宏 +0.0028，类别从 -0.0156 到 +0.0213 | **可信但不足晋级** | 廉价工程对照，不是新机制；不值得单独补完整矩阵 |
| W0 PRS 轴 | 原参数序跨恒等点，响应呈 V 形 | **可信的定义审计** | 旧 FAIL 不翻转；若重开需新单侧阶梯，当前不优先 |
| N1 JTD | k2/k4 候选 0.207/0.250，明显低于 A1/rank/独立尾部/打乱控制 | **足以归档当前实现** | 8×8 联合尾部稀有度当前方向为负，不再调 bin、门或权重 |
| N2 CCT | feature token 复制时 A1 gap=0，容量方法有正 gap；DINO 单分支最强 | **概念探针；非真实门** | 自由复用参考的盲点值得验证；不能声称双分支协同或真实缺陷有效 |
| N3 DNC-I | 合成留出族相对 full 为正 | **探索信号；协议不合格** | 可能存在噪声通道稀释；不能据此进入真实候选声明 |
| N3 DNC-C | 与 DNC-I 完全相同 | **不能作为阴性机制证据** | 当前实现使相同集合成为结构性结果，必须修复后才能判断跨分支去冗余 |
| N4 PMC | 50% coreset 宏损失 0.0027，connector-k2 损失 0.0142；tri 无 AP 增量 | **AP 结果可信，成本结论有限** | 普通 concat coreset 是工程观察；“约 2×”只是 bank-size/距离数代理，未实测端到端或匹配延迟 |

### 2.1 本次审计确认的三个关键问题

#### 问题 A：N2 不是原计划的软容量匹配

`run_n2_cct.py` 的 `sinkhorn_log` 要求代价矩阵为方阵，并把行、列边缘都严格固定为均匀分布。它是**平衡 OT**。doc27 原计划是 query 行质量固定、正常 anchor 列容量使用软 KL 约束的半松弛/非平衡形式，应该支持 `Q × A` 的长方形矩阵以及 k2/k4 多参考。

当前探针还把 support 自身作为 anchor/query，在 16×16 特征网格直接复制块；这排除了真实视觉生成过程、Transformer 上下文变化、跨图正常差异和实际 query/reference 比例。因此 R0 只能说明“全局供给约束能感知 token 多重集合变化”。

#### 问题 B：N3 使用了目标 `test/good`

NTOF 导出脚本通过测试集索引器读取每类 `test/good` 图，再为这些图生成三种合成缺陷。DNC 的通道响应和 q 排序直接使用了 `good_syn_feat`。虽然没有读取真实缺陷 GT，但它知道并利用了目标测试图是 good，这与 V13 `DATA_ROLES.md` 明确写下的“目标类正常 test/good 不得进入拟合”冲突，也不符合严格 K-shot 推理条件。

因此现有 DNC 数字只能作为离线机制探索，不能作为 normal-only K-shot 方法结果。下一轮必须只从 manifest 指定的 K 张 support 构造干预和选择规则；真实 test/good 只能与异常图一起留在冻结后的评估集合中。

#### 问题 C：DNC-C 当前必然选回 DNC-I 集合

当前循环先以 `argmax(qD)`/`argmax(qC)`各取一个分支候选，冗余惩罚只影响这两个候选谁先入选。由于每分支配额固定为 256，循环最终会把各分支 q 排名前 256 的通道全部选完。惩罚不会在同一分支内比较“高 q 高冗余”和“稍低 q 低冗余”通道，因而不能改变集合。

所以 `dnc_c == dnc_i` 不是“分支相关性弱或 q 主导”的可靠实验结论。修正方式是：每一步对每个未选通道计算 `q_j - λ·redundancy(j, opposite_selected)`，再在分支内取修正后最优；或者先生成候选池后做带配额的联合子模优化。必须先用人工相关矩阵单元测试证明惩罚会改变集合。

### 2.2 次要但应修正的表述

- N4 的 bank 减半只保证理论距离计算量约减半，不等于 FAISS 实测匹配时间加速 2×，更不等于端到端加速。
- N4 `branch_merge` 的两个分支集合取并集后可能少于目标 k，若继续使用须补足到严格等预算；当前主要 concat/tri 结论不依赖它。
- N2 脚本顶部文字仍有 GRID=32、软容量等历史描述，而实际常量是 GRID=16、平衡 OT；未来结果必须从运行配置而不是注释推断。
- N1 已记录并修复 CDF rank 对齐 bug；修复后的负结果幅度很大，以上次修复版本为准，不需要再为小实现瑕疵复活路线。

## 3. 下一轮总流程：先修协议，再做真实门

建议一次连续 8–12 小时的决策轮，但不要求失败后跑满。路线按 P0 → P1 → P2 → P3 顺序执行；P1/P2 可在代码准备阶段交错，但 GPU 导出和重评顺序进行。**不创建另一个外部确认集实验，不组合尚未过门的候选。**

| 阶段 | 上限 | 任务 | 决策输出 |
|---|---:|---|---|
| P0 | 1.0 h | 修复数据角色、DNC-C 选择器、半松弛 OT 单元测试；建立 v14 manifest | `AUDIT_CORRECTIONS.md`、测试报告、数据 ID 清单 |
| P1-A | 1.0 h | 只从 support 生成图像域 copy/erase/scratch/nuisance；双 backbone 导出前 smoke/profile | 无 test/good 的 support-only cache |
| P1-B | 2.0 h | 修正 DNC-I/DNC-C 的合成留出族门；k2/k4 优先 | 合法的通道选择机制门 |
| P1-C | 1.0 h | 仅过门者运行一次冻结真实 MPDD 诊断 | 是否存在真实通道选择收益 |
| P2-A | 1.5 h | 图像域容量探针 + 半松弛 OT；先 16 网格 | feature-copy 信号是否迁移到真实图像变化 |
| P2-B | 2.5 h | 过门后跑 MPDD k2/k4 真实查询、必要控制；胜出再 32 网格 | 容量旁路及双分支贡献 |
| P3 | 1.5–3.0 h | 最多一个候选补 k1 可行性、seed1/2 或完整指标；没有候选则收口 | PASS_DEV 或明确归档 |
| 收尾 | 0.5 h | 统一报告、成本账单、下一步授权点 | 不以未完成冒充 FAIL |

如果 P1/P2 都未过真实门，直接进入收尾，预计可提前结束。若一个候选通过，优先把它的必要控制、种子和完整指标做扎实，不再另外发明第五条路线。

## 4. P0：必须先完成的修复和测试

### 4.1 新目录与不可覆盖原则

新建：

```text
experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/
scripts/innovation_v14_decisive_validation_20260905/
```

V13 结果和脚本保持原样作为审计证据；新文档登记它们的结论降级，不回写或覆盖原 JSON。`RUN_MANIFEST.json` 必须保存代码 hash、数据 ID、实际 grid、solver 类型、拟合/评估数据角色、运行环境和中断恢复点。

### 4.2 数据泄漏门

在 cache 写入和加载两侧同时检查：

- fit/select IDs 必须是 `data/splits/mpdd/manifest.json` 对应类别、seed、shot 的 support 路径；
- 任意 fit ID 含 `/test/` 立即失败；
- evaluation IDs 可以含 test，但不能进入任何权重、通道、归一化参数、阈值和超参数选择；
- 合成 mask 可以用于 support-derived proxy objective，但必须明确方法属于“support synthetic adaptation”，不是完全无适配；
- k1 的多个合成 seed 不是多个独立正常样本，报告时不能当作数据量增加或置信区间独立单位。

### 4.3 DNC-C 选择器测试

至少三个确定性测试：

1. λ=0 时，DNC-C 与 DNC-I 集合完全相同；
2. 构造一个高 q 但与另一分支已选通道高度相关的通道，λ>0 后应被次高 q、低相关通道替换；
3. 每分支恰好 256 个唯一索引，固定输入重复运行完全一致。

另外报告 DNC-C vs DNC-I 的集合 Jaccard、所选跨分支最大/均值相关和 q 值损失。只有集合确实改变且冗余下降，才有资格进行机制比较。

### 4.4 半松弛容量匹配测试

目标形式：

`min_P <P,C> + ε ΣP(logP−1) + τ KL(P^T 1 || ρ)`，约束 `P 1 = a`。

其中每个 query 行质量固定；normal anchor 目标容量 `ρ` 来自 support 单元/coreset 的代表质量，列质量允许软偏离。支持 `Q != A`；不把每张 query 强制一一使用所有正常 anchor。

确定性测试：

- 长方形 `Q×A` 可运行，行边缘误差小于 1e-5；
- τ→0 时接近逐行熵 soft matching，不产生虚构的全局一一配对；
- τ 增大时列过载下降；当 Q=A、τ 很大才趋近平衡 OT；
- 同时置换 query 行或 anchor 列，逆置换后的结果一致；
- identical query/support 的容量 premium 接近零；复制某个 normal anchor 后 premium 随复制比例上升；
- 每行局部分数的定义不会因为另一区域异常而给全部背景统一抬分，必须输出 spillover 指标。

## 5. P1：合法重做通道选择，并真正检验跨分支机制

### 5.1 数据和生成方式

只对 K-shot support 生成 cutpaste、local erasure、thin scratch 和五类 nuisance。每个 support 每族固定多个位置/面积种子，种子和强度在运行真实 GT 前冻结。DINO/CLIP 都从**图像变换后重新编码**，不直接编辑 feature token。

k2/k4 作为第一判断，因为可按 support image 做 leave-one-image-out：在 K−1 张选择通道，在剩余一张的留出合成族上测机制，轮换汇总。k1 只能作类别内 support 自合成探索，不能给独立 normal 泛化结论；只有 k2/k4 明确通过才补 k1。

### 5.2 固定候选与控制

候选只保留：

- DNC-I：每分支按 defect-vs-nuisance q 选 256；
- DNC-C-fixed：修正后的跨分支冗余约束，仍保持 256/分支；
- 可选 DNC-soft：把 q 映射为有上下界的连续对角权重，仅当硬选择过门后再研究，不与本轮同时调参。

必要控制：full A1、10 个预先固定 random masks 的均值/区间、high variance、low nuisance、DINO-only、CLIP-only。`best-of-10 random` 只作保守补充，主要比较用随机分布均值和区间，避免“随机基线也按结果挑最佳”的含混解释。

### 5.3 机制门和真实门

机制门同时满足：

1. DNC-I 相对 random-mean 与 low-nuisance 在至少 2/3 留出族 ≥ +0.02；
2. 正常 nuisance p99 相对 full 不超过 +10%；
3. DNC-C-fixed 相对 DNC-I 的集合 Jaccard <0.95，所选跨分支冗余下降至少 10%；
4. DNC-C-fixed 相对 DNC-I 的合成宏 AP ≥ +0.003；错配分支响应或打乱跨分支相关后该增量下降 ≥0.003。

若 1/2 通过、3/4 失败，只允许 DNC-I 进入一次真实诊断，并称“通道适配”；不能称跨分支融合创新。

冻结后真实 MPDD 门：六类、同 shot 宏平均相对 A1 ≥ +0.006；每个已声明 shot 非负；按 shot 汇总至少 5/6 类正；最差类 ≥ −0.01；相对等维 random-mean/highvar/low-nuisance 中最强者 ≥ +0.003。DNC-C 还必须比 DNC-I ≥ +0.003 才能支撑融合创新。

不得因真实 connector 或某个 bracket 结果不好而重选通道数、λ、合成族或融合权重。本轮只有一个冻结配置；真实门失败即归档。

## 6. P2：把容量机制从人工 token 复制推进到图像与真实查询

### 6.1 先做图像域转移门

在 support 图像中复制正常纹理/部件块，再完整通过 DINO/CLIP 提取；同时生成局部擦除、细划痕、光度/轻仿射 nuisance。mask 只用于探针评价。相较直接 feature copy，这一步会包含边界、重采样与 Transformer 上下文，是必须通过的现实性门。

先在 16 网格运行：A1 自由匹配、DINO soft-capacity、CLIP soft-capacity、concat soft-capacity、双分支独立计划、跨分支计划耦合。使用 capacity premium：每行软容量期望成本减去同分支自由 soft/NN 成本，而不是直接以全局 OT 成本替代 A1；所有分数用 support-only normal 统计校准。

图像域进入门：copy inside/outside premium 相对自由匹配 ≥ +0.02，随面积单调；nuisance spillover/FP 在冻结门内；真实配对优于错配；信号不能只来自粘贴边界，可增加同边界、不同内容控制。若真实图像重编码后容量信号消失，停止，不进入 MPDD GT。

### 6.2 真实 MPDD 决定性门

优先 k2/k4，使用缓存的真实 query 与 K support；半松弛 solver 支持 `1024 × (K·1024)`。先 16 网格做所有控制，只有超过 A1/必要控制才在 32 网格复验。逐图流式算，不保存全数据集 dense transport matrix。

固定两种候选上限：

- CAP-D：`A1 + λ·positive(z(capacity_premium_D))`，检验单分支容量旁路；
- CAP-X：A1 加双分支 premium/分歧的一个预注册闭式组合，必须由 support 合成门确定且不可看真实标签选公式。

必要控制：A1、DINO/CLIP 单分支自由匹配、concat soft-capacity、双分支独立容量、无容量双计划、跨分支错配、相同额外计算量的平滑/局部均值。报告 anomaly type，但所有阈值仍以六类总体门为主；`parts_mismatch`、`bend_and_parts_mismatch` 只能作预先声明的机制子组，不能替代总体失败。

真实门沿用 §5.3 公共性能要求。CAP-X 还必须比 CAP-D 与 concat capacity 中最强者 ≥ +0.003，错配分支掉点 ≥0.003，才可称“跨分支容量融合”。如果只有 CAP-D 有效，应诚实定位为 A1 的 DINO 容量辅助路径，而不是双分支协同创新。

### 6.3 特别关注正常背景污染

全局容量约束会把一个异常区域造成的匹配压力转移到其他正常行。每图必须报告：

- mask 内 premium、距 mask 1–2 token 环带 premium、远背景 premium；
- normal image p95/p99 和 image-level max；
- anomaly size 与远背景抬升的相关性；
- 只打乱 query token 位置时分数多重集合是否保持，避免把坐标效应混入容量机制。

若 inside 提升伴随远背景相当幅度抬升，即使 Pixel-AP 偶然上涨，也不能通过机制门。

## 7. P3：何时补 seed，何时停止

只有 P1 或 P2 中最多一个候选通过冻结真实门，才追加 seed1/2。报告完整 cat×shot×seed，而不是只给平均。支持样本 seed 是参考集变化，不是独立数据集；置信区间以图像为重采样单位，不以像素当独立样本。

稳定性门：三 seed 平均仍满足宏增益 +0.006；至少 2/3 seed 总体为正；每个 shot 三 seed 平均非负；最差 cat×shot×seed 不低于 −0.02；必要控制差保持正且机制破坏控制仍成立。未达则记录为探索性观察，不进入主方法。

今晚/下一轮不访问新的外部确认集。MPDD 已是高度消耗的开发集。只有通过稳定性门后，冻结代码、权重/选择规则、主终点、失败规则，由用户明确决定外部数据授权，才能做一次性确认。

## 8. 如果 P1/P2 都失败：不要再继续同类手工规则

这将意味着当前已检验的静态融合、动态权重、文本补偿、局部统计、对齐、跨参考、图能量、跨层标量轨迹、空间恢复、相位、联合尾部、通道筛选和容量匹配均未形成稳定的新机制。继续换聚合函数的边际价值很低。

届时有两个合理选择：

### 选择 A：收口 A1 论文

把贡献限定为：互补的成熟视觉表征在统一 patch memory 下的简单特征级融合；四数据集、完整 single-branch/matched baselines、效率和失败边界。创新强度不靠额外未验证模块拔高。将大量探索作为内部证据，不在主文堆砌。

### 选择 B：另立可学习匹配路线（推荐的下一研究轮，但会改变协议）

比继续 SCAIF 更合理的是**support-synthetic、匹配目标驱动的轻量度量学习**：

1. backbone 全冻结；学习每分支有上下界的正对角通道权重，身份初始化，保留 A1 旁路；
2. 可选一个低秩跨分支交互项，但只有对角度量先通过后才加入；
3. loss 直接优化“合成缺陷 patch 的 normal-memory 距离排名高于 clean/nuisance”，而不是当前恒正距离的无中心 BCE；
4. 用 support 内合成族 leave-one-family-out；不读取目标 test/good；不使用真实源缺陷，除非用户另行授权；
5. 对照 DNC-I 闭式选择、等参数 concat head、单分支头、无 support、错配分支、随机标签；
6. 先做 50–100 步优化健康探针，要求留出合成排序改善且 normal 路径不坍缩，再跑真实门。

这条路线真正回答“增加可学习模块能否让融合更有创新”，也针对 SCAIF 的目标/正则/初始化问题，而不是再次调整旧模块的 sparse/weight decay。成功后应作为 A2/新论文协议报告，不能与 zero-training A1 混称同一设置。

关系蒸馏到单分支属于效率论文方向，只有用户明确把目标改为降低端到端推理成本时才优先；目前不应与增强融合创新同时推进。

## 9. 下一轮产物与验收清单

建议目录：

```text
experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/
  RUN_MANIFEST.json
  DATA_ROLE_AUDIT.json
  AUDIT_CORRECTIONS.md
  TEST_REPORT.md
  P1_dnc_fixed/
    PROTOCOL.json  SYNTH_RESULTS.json  REAL_RESULTS.json  DECISION.md
  P2_soft_capacity/
    PROTOCOL.json  IMAGE_PROBE.json  REAL_RESULTS.json  SPILLOVER.json  DECISION.md
  FINAL_DECISION_CN.md
```

最终报告必须明确：

1. fit/select IDs 中是否出现任何 `/test/`；
2. DNC-C 修复后是否真正改变集合，改变是否带来独立收益；
3. 容量 solver 是平衡还是半松弛，实际 Q/A/grid/ε/τ 是什么；
4. 图像域信号、真实总体性能、必要普通控制和错配控制分别如何；
5. 增益是双分支机制、单分支改进、普通工程收益，还是不成立；
6. 哪些阶段因机制失败停止，哪些因时间/资源未完成；
7. 实测匹配延迟、端到端延迟、峰值显存/内存，不能以理论规模替代全部实测；
8. 是否满足进入 seed/外部确认的预注册门。

## 10. 可直接交给实验助手的提示

> 请按照 `28_V13_RESULT_AUDIT_AND_NEXT_DECISION_PLAN_CN_20260905.md` 执行下一轮决定性验证。不要覆盖 V13 文件。先完成 P0：建立 v14 manifest，强制所有拟合/选择 ID 只能来自 K-shot support，修复 DNC-C 使冗余惩罚能改变分支内通道集合，并用确定性单元测试验证；实现支持 Q≠A、query 行固定、anchor 列软 KL 容量的半松弛 OT。然后只用 support 图像重新生成和编码合成干预，先跑合法 DNC 机制门与图像域容量转移门。只有过门者才运行一次冻结的 MPDD 真实诊断；不按真实结果调通道数、λ、生成族、容量参数或融合公式。最多一个真实胜出者补 seed1/2。若 DNC-C 不胜 DNC-I，不称融合创新；若双分支容量不胜 DINO-only/concat 控制，只记单分支匹配改进。N1 JTD、旧 DNC-C 和 PMC tri 不再复活。本轮不访问新外部数据，不下载模型，不启动未经授权的真实缺陷训练，不改论文主表。最后报告数据 ID、机制控制差、逐类逐 shot 结果、spillover、实际成本和明确停止原因；全部失败时如实结束。

## 11. 主要本地证据

- V13 总结：`experiments/dynamic_fusion/innovation_v13_overnight_20260904/FINAL_SUMMARY_CN.md`
- 数据角色：`experiments/dynamic_fusion/innovation_v13_overnight_20260904/DATA_ROLES.md`
- N1：`experiments/dynamic_fusion/innovation_v13_overnight_20260904/N1_jtd/N1_DECISION.md`
- N2：`experiments/dynamic_fusion/innovation_v13_overnight_20260904/N2_cct/N2_DECISION.md`
- N3：`experiments/dynamic_fusion/innovation_v13_overnight_20260904/N3_dnc/N3_DECISION.md`
- N4：`experiments/dynamic_fusion/innovation_v13_overnight_20260904/N4_pmc/N4_DECISION.md`
- NTOF 数据来源：`scripts/innovation_v12_new_observables/run_r2_ntof_export.py`
- N2 实现：`scripts/innovation_v13_overnight_20260904/run_n2_cct.py`
- N3 实现：`scripts/innovation_v13_overnight_20260904/run_n3_dnc.py`
- N4 实现：`scripts/innovation_v13_overnight_20260904/run_n4_pmc.py`

一句话建议：**先用 v14 把 N2/N3 从“有意思的合成现象”变成合法、真实、可归因的证据；过不了就停止手工融合搜索，转向匹配目标驱动的轻量可学习度量，或者诚实收口 A1。**
