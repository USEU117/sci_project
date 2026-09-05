# 今晚创新点探索计划：历史收口、四条新机制与两条条件学习路线

编制时间：2026-09-04 深夜；工作区核查截至 23:47（Asia/Shanghai）。

性质：**计划与实验交接书，不是执行记录。此次未启动训练、重导出、自动化或外部验证。** 预计执行窗口为启动后 8–10 小时，包含实现、测试、筛选和整理；不是要求无结果也必须跑满。各路线名称为本文工作代号，不代表已证明的新方法。

## 1. 先给结论

**A1 仍是主方法。今晚应从“再换一种融合权重/交互模块”转向“改变匹配约束、选择什么信息进入融合、建模两分支的联合异常性，以及降低双分支成本”。**

推荐顺序：短审计 → 联合尾部统计 JTD → 容量约束双分支匹配 CCT → 缺陷/干扰对比通道选择 DNC → 双分支记忆压缩 PMC → 最多两个胜出者复验。可学习匹配目标 MRL、关系蒸馏 RDS 是条件备选，不默认替换零训练论文主线。

今晚的合格产出不是“必须找到涨点模块”，而是：四个不同机制有可复现的进入/停止结论；至少区分真正的双分支增益、普通算子的收益、工程效率收益、实现失败。若全部失败，应如实结束，不继续换名字或搜参数。

### 1.1 为什么这次不直接继续早期融合

此前不只有末端分数融合：A1 本身就是特征级融合；NCPRA 做过分支间预测；SCAIF 做过两对中间/末层特征上的可学习交互。SCAIF correction 修复后等于 static2，短优化诊断又表明“交互能动”不等于“交互有益”。

尚未做的是 backbone 内部真正逐层插入 bridge 的 Stage2。但它需要更高显存、重导出和更多训练，也更难隔离归因。**目前不值得把今晚预算直接押在 Stage2。** 若要再尝试可学习模块，先用缓存特征验证直接面向匹配排序的新目标，过门后才讨论更早融合。

## 2. 已经尝试到哪里：不要重复安排

下表综合既有路线文档、实验 ledger 和本晚新产物；历史数字来自现有结果文件，本次未重新推理全部图像。不同路线的设置不完全相同，不能把下表的增量当作统一排行榜。

| 尝试方向 | 已有工作与结果 | 今晚处理 |
|---|---|---|
| 局部统计/校准 | LNDC 无稳定收益；CEQA 约 +0.0028 被 A1 rank-only 控制复制 | 不再扫描同类校准参数 |
| 对齐/增强/平滑 | DSAM、DEVA、FAGR 未获独立稳定收益 | 不再换相似平滑器 |
| 可学习跨分支预测 | NCPRA 四种设置均负，最佳仍约 -0.0052 | 不重做原始特征互预测 |
| 动态门控、文本补偿 | v6 DGSafe 失败；v7 文本局部正但不稳；v8 TCRR 外部迁移失败，v9 安全补丁未解决 | 不复活旧文本专家路由 |
| 跨参考一致性 CRAM | k2/k4 两候选均值约 -0.0028/-0.0076；真实配对无清晰优势 | 归档 |
| 等变稳定 MESP | DINO 小幅正，但不胜错配控制；完整双分支未继续 | 不写成“双分支完整实验已失败” |
| 规范位置 CAPM | 可完成对齐，但 AP 约 -0.0275，0/6 类正 | 不再依赖可靠全局配准 |
| 正常参考安全 NORC | 构造性机制就绪，缺少有效伙伴 | 条件关闭，不是已证伪安全性 |
| 频谱 STR | 总体约 -0.027；brown 类别局部正 | 不为单类重新调全局方法 |
| 部件关系 SPRG | 节点匹配率约 36%，未过稳定对应前提 | 不直接训练部件图 |
| 局部线性重构 LLSE | seed0 +0.0089；三 seed 降至 +0.0025，稳定性门失败 | 保留局部性观察，不复活全局融合 |
| 图内自一致 CSS | 存在弱类线索，sum/max 融合总体约 -0.042/-0.025 | 不把弱类补救直接推广 |
| 旧专家池/路由 RSR、MTCOA | 修正 oracle 后 k1 +0.0239、k2 +0.0040、k4 -0.0129；不满足原进入条件 | 固定旧池的路由关闭 |
| 上下文修复 BC-MCR | 未胜必要对照 | 不续重型修复 |
| 跨分支图能量 CECW | k1/k2/k4 约 -0.010/-0.023/-0.044；打乱耦合几乎不变 | 不再搜同构图能量 |
| 干扰切空间 NTOF | normal FP 比中位 0.934，降幅不够；双分支不胜 DINO | 不重搜 rank/照明强度 |
| 中间特征交互 SCAIF v4/v5 | v4 大幅退化；纠错 v5 的 18/18 个 cat×shot AP 等于 static2 | 不因“bug 修完”就继续完整训练 |
| 优化健康短诊断 | 去 sparse 后 CLIP 残差能动、梯度可达；white 固定 episode AP 反降约 0.0074 | 原目标关闭；新目标必须另立协议 |
| 跨层标量轨迹 CL-RPF | 最佳轨迹 AP 0.2325/0.2841/0.2926，三 shot 均输静态多层 | 不重做 slope/persistence；不外推为一切跨层向量机制无效 |
| 干预响应 PRS | normal-only G1 按当前强度轴失败；未进入异常验证 | 保留原 FAIL，另做短定义审计，见 §4 |
| 匹配前细节恢复 | 七层配方 AnyUp 六类 0.2775，低于原配方 0.3099 和 bilinear 0.3186；条件耦合也负 | 不换另一个重型 upsampler 续跑 |
| 多相位 PSMF | 三微缺陷类约 -0.0006；不胜 overlap mean，微缺陷收益约 1e-4 | 本形态关闭，不扩大相位数 |
| CASF/PDMC/Stage2 等 | CASF 在前置探针止步，PDMC 前提未满足，Stage2 未启动 | 标 NOT_STARTED/SKIP_PREMISE，不写成完整阴性实验 |

### 2.1 当前主基线必须固定身份

| MPDD seed0，六类宏平均 Pixel-AP | k1 | k2 | k4 |
|---|---:|---:|---:|
| 冻结 A1 | 0.309212 | 0.343699 | 0.388328 |
| static2，两对层静态 concat | 0.308382 | 0.348602 | 0.384439 |
| SCAIF correction | 0.308382 | 0.348602 | 0.384439 |
| 七层标准化分数均值 mean_std | 0.309856 | 0.349498 | 0.375414 |

A1：DINOv2-B/14 L11 与 AnomalyCLIP 图像塔 CLIP ViT-L/14 L24；分支各 768 维，CLIP 网格按既有实现对齐到 DINO 32×32，分支归一化、0.5/0.5 特征拼接及整体归一化，最近邻距离与后处理均继承冻结实现。代码中的 L11/L24 命名按项目已有约定，不重新解释层索引。

**细节恢复和 PSMF 文件内的 `a1` 臂实际上对应七层 mean_std 配方。** 同配方对照的负结果仍有意义，但不能据此声称已在真正冻结 A1 上全面排除空间恢复/相位机制。今晚只核对命名与廉价基线，不借此重跑整个失败路线。

## 3. 外部资料带来的启发与创新边界

以下为本次检索到的原论文/作者论文页；是设计依据，不是本项目有效性的证据，也不是穷尽性新颖性检索。

| 原工作 | 可借鉴内容 | 本项目必须避免的重复 |
|---|---|---|
| [DFM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.pdf) | 让特征适配直接接受匹配任务的优化信号 | 加可微匹配/MLP 本身不是创新；应验证双分支私有信息与匹配目标的必要性 |
| [Unbalanced Optimal Transport，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.html) | 用软质量约束连接不同匹配规则 | 原研究是目标检测，不能引用成工业异常定位效果证据 |
| [FastRef，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.pdf) | few-shot 原型可通过 query 特征转移与 OT 异常抑制进行细化 | “在少样本异常检测中加入 OT”已经不是新颖点；本计划不更新正常原型，检验容量与跨分支分配约束 |
| [COPOD，ICDM 2020](https://arxiv.org/abs/2009.09463) | 用经验分布/尾部极端程度表征异常 | 排名、copula 名称本身不是创新；必须证明实际联合依赖超越独立边缘与 rank-only |
| [OCR-GAN，TIP 2023](https://arxiv.org/abs/2203.00259) | 不同编码器间可以选择通道交互 | 通道注意力早已存在；这里需要缺陷对干扰的选择性和双分支互补预算证据 |
| [PatchCore，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) | 正常记忆的 coreset 压缩与覆盖目标 | 普通 coreset 是强制基线，不应把缩小 memory bank 当作原创 |
| [VRM，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VRM_Knowledge_Distillation_via_Virtual_Relation_Matching_ICCV_2025_paper.pdf) | 蒸馏关系而非仅模仿特征；关注错误教师关系的传递 | 原工作不证明本项目 AD 有效；须与特征/分数蒸馏对照 |
| [EfficientAD，WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf) | 精度—成本可以成为独立研究目标 | 不把已有师生异常检测包装成全新框架；不照搬论文硬件延迟到本机 |

## 4. W0：只花有限预算收口两个审计问题

总上限 45 分钟，**不是今晚主菜**。先保存旧结果，不覆盖历史协议。

### 4.1 PRS 的响应轴定义审计：最多 20 分钟，CPU 优先

当前 `r(a)=||f(T_a x)-f(x)||` 是距恒等输入的无符号位移，却将跨过 1 的 exposure `[0.70,1.15,1.40]`、gamma `[0.80,1.20,1.50]` 按参数递增当成扰动强度递增。渲染代码确认 exposure 为乘法、gamma 为 `x**(1/g)`；恒等值均为 1。

因此，亮度递增不意味着距原图的扰动量递增，V 形响应不能单独证明“强度轴不存在”。取绝对相关也解决不了 V 形问题。这是定义审计，**不将旧 FAIL 改成 PASS**，也不把重排三点后单调当成异常可分性。

交付：原参数序、`|log(a)|`、实际像素 RMS、特征响应四列对照；记录量化/饱和。既有三点仅可描述，不足以做可靠强度曲线。若未来重开，需要新预注册的单侧阶梯（变亮/变暗分开）和跨族缺陷区分门，不能沿旧异常标签搜新轴。今晚不自动导出该阶梯，不重开 PRS 完整路线。

### 4.2 基线身份和廉价 bilinear 信号：最多 25 分钟

读取恢复/相位脚本，区分 `A1_FROZEN`、`STATIC2`、`MEAN_STD7`。先做冻结 A1 特征/分数 map 小样本 parity；相同 dtype/设备期望逐位一致，否则登记误差来源，浮点容差先定为 atol/rtol 1e-6，不以 AP 接近替代 parity。

若已有缓存可直接计算，则只比较冻结 A1 的“32 网格匹配后插值”与“同样末层特征先 bilinear 到 56 再匹配”，six-class k1。没有足够预算只完成身份审计，记 TIME_BUDGET；不加 AnyUp、guided filter 等候选。原七层 bilinear 的 +0.008704 是**廉价工程线索**，不是已证实 A1 增益，更不是跨分支创新。

## 5. N1 / JTD：两分支联合尾部，而不是再调融合权重

**问题：** 同样的 DINO/CLIP 分数，单独看可能普通，但它们的组合是否在正常样本中罕见？旧 rank-only、局部密度没有回答这个低维联合依赖问题。

实现最小版：

1. 保留 A1 主分数；从两个单分支最近邻分数生成 normal-only 经验分位 `u_D,u_C`。
2. 在正常校准样本上拟合 8×8 二维直方图，带固定 Dirichlet 平滑，与独立边缘乘积作比较。它是显式二维依赖模型，不直接把 COPOD 名字等同于联合依赖估计。
3. 检验高尾组合的额外惊讶度：`R=max(0, log(p_D*p_C/p_joint))`，只在 `max(u_D,u_C)>0.9` 时启用，上限截为 5；避免罕见但双低分的正常组合被当作异常。
4. 候选先固定为 `normal_rank(A1)+0.1*R`。边界、ties、超出经验范围的处理必须记录；秩饱和时保留 A1 路径，不用测试集拟合尾部。
5. k2/k4 使用 support leave-one-image-out 分数。k1 不自动套用该统计：图内带空间排除的校准只作另列探索，不能声称独立样本或有限样本误报保证。

必须对照：A1；A1 rank-only；独立尾部 sum/max；相同边缘但打乱 D/C 正常配对；无高尾门的二维稀有度（诊断）。打乱只破坏拟合联合分布，不得顺便改变测试分数的边缘分布。

R0：不使用真实异常标签，以二维已知依赖 toy case 检查配对/打乱可识别；分支严格单调缩放后重新做同样 normal 校准，输出排序应基本不变；至少两个正常校准分块验证不因某一块崩溃。校准有效块太少则 SKIP_DATA，不作成功结论。

R1 归因：必须比最强独立尾部/rank-only 高至少 0.003，且真实配对比打乱高至少 0.003，再检查 §11 性能门。只改分数尺度或只胜原始 A1、不胜 rank-only，立即归档。

预算 50 分钟；首轮 k2/k4。潜在价值：低容量、无需梯度的联合统计。风险：正常样本太少、背景 patch 相关、尾部分位饱和；目前只是新假设。

## 6. N2 / CCT：正常记忆的容量约束与双分支协同匹配

**问题：** A1 每个 query patch 独立找最近邻，同一个正常 patch 可以被无限次使用。局部外观都像正常时，异常复制、缺失或占比变化是否被这种自由匹配掩盖？

最小实现：正常特征固定，取最多 256 个有真实 D/C 配对身份的正常 anchors；保存其代表的原始 patch 数作为容量。首轮 query 降到 16×16 只测机制，原 A1 32×32 私有路径保留，不以降网格结果作最终公平结论。

为 D/C 分别建代价矩阵 `C_D,C_C`，每行运输质量固定，列质量对正常容量使用软 KL 惩罚，加入两个分支逐行分配概率的 JS 一致性项。概念目标：

`sum_b <P_b,C_b> + entropy(P_b) + tau*KL(column_mass(P_b),rho) + gamma*sum_i JS(P_D[i],P_C[i])`。

这是列容量软约束的半松弛匹配；不强迫异常一定有完美匹配，不依据 query 更新正常原型。正常容量来自 support，不能用测试批次估计，也不能把全部 anchors 强制等频。

**关键等价性检查：** 如果最后只用一个运输计划和 `C=(C_D+C_C)/2`，在匹配的归一化条件下就可能等价于 concat 特征上的 OT。这种实现只能标记 concat-OT，不能宣称双分支协同创新。必须保留可检验的两个私有计划与耦合项。

实施时先用正常代价尺度归一化；预注册固定 entropy、gamma，tau 至多两档。各 query 的容量负担可按其运输权重分摊列过载代价，另输出预期匹配距离和分支分配冲突。三项的定义、normal 校准和加入 A1 的系数在读取真实异常结果前一次冻结。

必须对照：原 A1；同 anchors 的 concat-OT；两分支独立容量匹配；无容量的双计划耦合；打乱 anchor 跨分支配对；DINO-only/CLIP-only。计算预算、anchor 数、后处理一致。

R0：用正常图的局部复制/擦除干预，检查容量信号是否随面积可控改变，并与亮度、轻微尺度变化对照；另做保持 token 多重集合不变的置换。**无坐标约束的容量方法本来不应识别纯排列改变**，否则先排查实现伪信号，不声称检测结构顺序异常。

R1：超过 concat-OT/独立 OT 至少 0.003，去容量和错配控制确实削弱预注册效果；正常 FP 受控，才进入性能复验。若只有单分支容量有效，记录“匹配改进”，撤下“双分支融合创新”表述。

预算 85 分钟（包含 toy tests 与小规模实现）；dense Sinkhorn 只在小矩阵，禁止全测试集 all-pairs 堆 GPU。若机制只在大面积合成复制上有效，应标记局部适用，不扩大训练。

## 7. N3 / DNC：选择对缺陷敏感、对干扰不敏感的通道

**问题：** 两个成熟 backbone 的所有维度是否都适合异常定位？不只问“哪个分支权重大”，而问“哪些通道值得进入融合，哪些通道提供另一分支没有的证据”。

复用已有 normal 图、光度干预和 cutpaste/local-erasure/thin-scratch 生成器；不使用真实缺陷标签选择通道。对分支 b 的通道 j，估计缺陷区域内的响应与正常干扰响应之比：

`q_bj = robust_mean(defect_response_inside_mask) / (robust_mean(nuisance_response)+eps)`。

响应先除以正常通道尺度；截断极端值，不允许小分母单独决定排序。每次用两种合成族选通道，第三族留出，三个族轮换；生成随机种子固定三组，mask 面积在族间匹配。**这属于合成干预驱动的 normal-only 适配，不再简单称“完全不适配”。**

首轮两种配置封顶：

- DNC-I：每分支保留 256 通道，独立选择，保持分支配额；
- DNC-C：同样总维数，但在候选池中抑制跨分支高度冗余的干预响应，保留各分支私有响应。冗余比较的是同一批干预 episode 的响应向量，不直接把 DINO 第 j 维当成 CLIP 第 j 维。

保留冻结 A1 旁路，并分别评估 reduced-feature score 单独使用与一次预注册的轻量残差融合；不能为每类选融合方式。选择规则、输出归一化与固定融合系数在真实异常评估前冻结。

必须对照：等维随机通道（三个固定随机 mask）；高方差通道；仅低 nuisance 通道；打乱合成缺陷/正常干预身份；DNC-I；单分支同总维数。不能只拿 512 维新模型与未匹配容量的单分支比较。

R0：未见合成族上相对 nuisance-only/random 的缺陷 AP 至少 +0.02，正常高尾 FP 不恶化超过 10%，至少两个留出族成立；这些是筛选阈值，不是统计显著性或真实缺陷保证。失败则不跑真实异常调 mask。

R1：DNC-C 需胜 DNC-I 和最强选择基线至少 0.003 才谈互补选择机制；只 DNC-I 有效则改称通道适配，不包装成交互创新。真实异常还需 §11 的公共门。

预算 100 分钟。优先复用缓存，补导出前测 10 张成本；不得默认缓存中的 held-out 变体仍未被历史开发看过。合成验证也应注明开发复用局限。

## 8. N4 / PMC：双分支正常记忆的保真压缩

**目标切换：** 不要求再涨点，而是同等精度下减少 memory 与匹配开销。它仍保留冻结双编码器，比重新训练网络风险小，但创新性预期低于 N2/N3。

普通 concat coreset 可能偏向某个分支的易覆盖区域。尝试在固定配对 anchors 预算下，同时约束 DINO、CLIP、concat 三种距离的正常覆盖，并保护正常校准 query 上分支近邻身份明显不同的区域。不得基于真实缺陷“哪些点重要”选 memory。

最小版是三种归一化覆盖误差的 minimax greedy；附加保护配额只允许一个预注册比例，先分析是否等价于 concat 距离的普通 greedy。若数学上等价或输出基本相同，标记 REDUNDANT，不花整夜实测。

预算比只允许 25% 和 50%。对照：完整 A1 memory、相同比例随机、concat greedy coreset、两分支各自选取后等总预算合并。所有方法使用同一对齐/归一化；构建时间计入一次性成本。

R0：正常留出样本 top-k 近邻排序及分数保真；k1 图内排除不作真实独立校准。R1 工程门：50% memory 下宏平均 AP 损失不超过 0.003，最差类不低于 -0.01，匹配阶段实测加速至少 25%。双编码器开销仍在，**不可将 KNN 加速称端到端加速**。

要升格为研究线，还需比同预算 concat coreset 更好，且差异来自分支私有覆盖保护；普通 coreset 即可达到则只记录工程优化。预算 55 分钟，CPU 为主。

## 9. 两条条件备选：不默认启动，不与零训练结果混写

### O1 / MRL：面向匹配排序的可学习交互

不是 SCAIF 多训练几轮。重新预注册：缓存特征上的小残差交互，保留独立 A1 路径；优化正常/合成异常对的匹配分数排序，或明确中心化的 logits，避免距离直接乘 10 后作为恒正正常 logit。若采用 ranking，则单独定义正负采样及 margin，不同时搜索两类目标。

源数据默认仅使用既有 MPDD 开发中的其他类正常图及合成干预，按 leave-one-class-out；目标类只用规定 shot 正常 support。相较之前 real-source SCAIF，监督协议发生变化，必须另列；不得混用其训练成本/性能。使用真实源缺陷训练须另获明确授权。

先只做固定两个源 episode 的 50–100 步新目标可学性探针；明确 support 是否 stop-grad、query/support 变换是否一致。使用显式残差范数约束，优化器及 decay 固定，记录 task-grad、正则作用、输出相对变化。不是只要求 loss 降，而是留出合成 episode 排序改善且 clean-preserve 不坍缩。

对照：冻结 A1/static2、等参数单分支、concat 小头、无 support、错配分支、无交互。先跑最小主模型与 concat 头；若主模型不胜就停止，不补全昂贵控制矩阵。

仅在用户同意“增加源类训练/适配实验”，且 N1–N4 已完成初筛、没有更值得复验候选时启用，预算最多 90 分钟。完整六 fold 不保证当晚完成。**此路线成功也不是零训练 A1 的直接替代。**

### O2 / RDS：把双分支的正常匹配关系蒸馏给单分支 query

区别于 NCPRA：不预测对方完整 768 维特征，而让轻量学生学 A1 对正常 memory 的邻居排序/相对距离结构。query 推理只用 DINO；若 support 建库仍需 CLIP，必须披露这项一次性成本。

训练使用既有源类正常图和合成变体，源/目标类分离；教师 A1 冻结。保留学生 DINO 原始距离路径，教师关系只保留稳定邻域，不能把教师所有异常响应无条件当真。不给目标测试 batch 在线更新模型。

对照：原 DINO、双分支 A1、等参数 raw-feature MSE 蒸馏、scalar-score 蒸馏、relation 蒸馏。教师分数只用于训练，GT 只用于冻结后的评估。

进入后续条件：相对 A1 宏平均 AP ≥ -0.003、最差类 ≥ -0.01，端到端 query 延迟至少下降 40%，且优于普通蒸馏；这是待检验的工程目标，不是硬件预测。测量必须包含编码和后处理，不只测缓存头。

改变方法定位为“训练时双分支、推理时轻量单分支”，需用户接受这条效率研究线后才启动；与 O1 二选一，最多 90 分钟探针，不在同夜铺两套完整训练。

## 10. 8–10 小时长流程：有条件分支，不是无限搜索

下表全部是**预计上限**。启动后先根据小样本 profiling 调整是否可完成；超时保存中间产物并进入下一项，不通过删控制来赶工。

| 相对时间 | 工作 | 必交付 |
|---|---|---|
| T+0–0:20 | 冻结输入、数据角色、代码 hash；检查现有任务资源，profiling | RUN_MANIFEST、DATA_ROLES、资源预算 |
| T+0:20–1:05 | W0 两个短审计 | 身份/定义审计，不篡改历史结论 |
| T+1:05–1:55 | N1 JTD | 实现测试、k2/k4 初筛或有理由停止 |
| T+1:55–3:20 | N2 CCT | 等价性分析、机制探针、最小真实评估 |
| T+3:20–5:00 | N3 DNC | 留出族选择性、对照、初筛 |
| T+5:00–5:55 | N4 PMC | 等预算保真/成本曲线、性质分类 |
| T+5:55–7:30 | 最多两个候选补 shot/seed/关键控制 | 逐类逐 shot 结果及复现证据 |
| T+7:30–9:00 | 有胜出者则继续复验；否则仅经授权启用 O1 或 O2；无授权则补分析/复现 | 不强行创造新候选 |
| 最后 30–60 分钟；最迟 T+10:00 | 收口、成本审计、报告 | SUMMARY、负结果、可复制命令、下一步 |

四条路线都早停且没有授权备选时，允许提前结束。若一个候选已显示可靠价值，不必为“数量”挤占它的关键对照/种子复验。若实现时间超过预算，写 IMPLEMENTATION_BLOCKED/TIME_BUDGET，不伪装成机制 FAIL。

### 10.1 防止流程失控

- 核查硬件为 RTX 3060 Laptop 6 GB；本次只读快照已有约 1.9 GB 显存占用，不能默认整卡空闲。执行前重新检查，不终止其他任务进程。
- 同时最多一个 GPU 工作进程；CPU 重活最多两个，先控制 BLAS 线程以免互抢。此条是进程资源规则，不要求创建其他 AI 任务。
- 不新增外部数据、模型权重或大依赖；优先既有环境。确需新增下载/改变数据角色时停下请求授权。
- 每路线正常配置最多两种，总主候选不超过八种；必要控制不是调参候选，但同样计入算力账单。实现 bug 修复最多一次，保留修复前后的失败记录。
- 不对测试集逐类选择规则、融合权重、随机种子或伪缺陷族。数据驱动拟合只能使用对应协议允许的正常/源数据。
- 不组合两个失败模块赌“互相补偿”。两个独立过门模块的组合也留到下一轮独立预注册，今晚不默认加交叉组合网格。
- 流式加载，一次一类，保留紧凑缓存；先估算新文件大小，再决定导出。磁盘配额在 manifest 中据可用空间填写，不擅删旧缓存腾空间。
- 不提交/推送代码，不改论文主结果表，不覆盖已有 checkpoints、结果 JSON 或其他任务文件。

## 11. 公共验收：把“有信号”和“可发表创新”分开

### G0：身份、数据与实现

冻结 A1 与匹配控制必须相同 support/query、随机种子、输入预处理、距离定义、map 网格和指标。输出样本 ID 清单、配置、脚本 hash、依赖版本。不能从 cache 文件名推断层/shot 身份，必须核查元数据。

测试集异常标签仅在候选和控制冻结后由评估阶段使用。拟合与评估脚本分离；保存参数生成的数据 ID 清单。目标正常测试图也不得自动进入拟合。只有 shot support 被允许时，不额外用全部 train/good 暗中增加参考数量。

### G1：机制探针

各路线按 §5–§8 的机制门先行，含空信息/破坏/等价性控制。合成结果只证明受控条件下有可检测信号。配对打乱控制若改变边缘分布、容量或算力，必须先修正，再谈归因。

### G2：开发初筛

N1 先 k2/k4，其他默认 MPDD seed0 的 k1/k2/k4；优先缓存小规模探针，进入真实性能门时报告六类，不按旧赢家挑类。N1 的 k1 若校准前提不足，允许明确标注“仅多 shot 候选”，不得填入跨 shot 平均伪装完成。

针对以 AP 为目标的路线，建议晋级门：

1. 所有已声明 shot 的六类宏平均 ΔAP ≥ +0.006；每个 shot 宏平均 ΔAP 非负。
2. 按 shot 平均后至少 5/6 类为正，最差类 ≥ -0.01。
3. 超过最强必要普通算子/等容量控制 ≥ +0.003；相应破坏控制支持机制。
4. 正常误报不恶化超过 10%（原 FP 为零时另报绝对增量，不能除零），同时报告图像级 AP、Pixel-AUROC，不能只截取有利主指标。

以上是本轮人为筛选门槛，需执行前冻结，不来自文献，不能当作显著性判定。PMC/RDS 使用各自保精度降成本门，不因 AP 没涨判失败。

### G3：稳定性和归因复验

最多两个候选追加 seed1/2；种子是 support 抽样种子，不是三个独立外部数据集。报告每类×shot×seed，均值、最差值及相对最强控制差。支持配对重采样时按图像而非像素 bootstrap，重算 AP；记录区间算法与重复次数，不把像素当独立样本制造窄区间。

如时间不足，标 PROVISIONAL/INCOMPLETE，不写“通过三 seed”。若真实分支机制没有胜过控制，即使比 A1 好，也只能保留普通算子收益。

### G4：外部确认——今晚不做

MPDD 已被大量开发消耗；BTAD/MVTec 已在旧路线中使用；VisA 是既有 checkpoint 的源域；未接触的数据集保持未接触。本夜只能产生开发候选，不能给出确认性泛化结论。

只有当配置、候选数量、主要终点和全部停止规则冻结，用户明确选择并授权新的外部确认方案后，才讨论 Real-IAD/MVTec AD 2 等数据。不能把同一已消耗外部集反复检验称“一次性盲测”。

## 12. 产物结构、恢复规则和早晨报告

建议新建而非复用旧 v12 结果目录：

```text
experiments/dynamic_fusion/innovation_v13_overnight_20260904/
  RUN_MANIFEST.json
  DATA_ROLES.md
  PORTFOLIO_LEDGER.md
  W0_AUDITS.md
  N1_jtd/  N2_cct/  N3_dnc/  N4_pmc/
    PROTOCOL.json
    TEST_REPORT.md
    RESULTS.json
    DECISION.md
    logs/
  OPTIONAL_LEARNING_DECISION.md
  FINAL_SUMMARY_CN.md
```

这是未来执行助手的建议结构，本轮仅创建本计划。实现代码使用新的 route 文件；确需改共享代码时先核对其他任务改动和回归测试。

每路线 ledger 至少包含：状态、假设、区别于旧尝试、配置数、数据 IDs、拟合数据角色、基线身份、主要/最差增量、最强控制差、运行时间、峰值内存/显存、输出路径、停止原因。状态区分 NOT_STARTED、RUNNING、PASS_DEV、FAIL_MECHANISM、FAIL_PERFORMANCE、REDUNDANT、SKIP_PREMISE、SKIP_DATA、IMPLEMENTATION_BLOCKED、TIME_BUDGET。

恢复执行先校验 manifest/hash 和已有完成标记；不因终端中断从头重训。原始预测至少保留可复算指标的版本；若磁盘无法保存全部 map，保存确定性复算命令、紧凑分数、样本清单和随机状态，并披露未保存内容。训练路线另存优化器状态、checkpoint 和分项日志。

早晨报告只回答五件事：

1. 真正完成了哪些机制，哪些仅做了探针，哪些因前提/时间没做？
2. 有没有超过 A1 **且超过必要控制**的结果，在哪些 shot/类别/seed 成立？
3. 有没有不涨点但确实更便宜的 Pareto 结果，端到端与匹配耗时各多少？
4. 哪条机制值得下一轮，哪条应该停止，依据是什么？
5. 如果全部阴性，已经排除了什么具体实现，哪些更大命题仍不能排除？

## 13. 可直接交给实验助手的执行提示

> 请按本文件规划执行一轮最长 10 小时的创新点探索。先核对当前实验状态，避免重复已经完成的工作；保存独立 run manifest 和预注册协议。严格冻结 A1 基线身份与数据角色，先做最多 45 分钟的 PRS 定义/基线身份审计，然后依次筛选 JTD、CCT、DNC、PMC。每条路线先测机制与必要控制，再评估 MPDD 开发集；失败按停止条件进入下一条，不追加同构调参。优先给最多两个胜出者补完整 shot、三 seed 和归因，而不是无边界增加模块。O1/O2 不默认运行，必须先得到用户对新增训练或效率目标的授权。本次不访问新的外部验证数据，不下载新模型，不覆盖旧结果，不修改论文主表，不终止其他任务。若时间不足，保存可恢复状态并明确未完成项；若全部失败，如实汇总，不强行产生“创新点”。最后提交中文总结、逐类逐 shot 结果、成本账单、必要控制差和下一轮建议。

## 14. 本地证据索引

工作区根目录：`D:/STUDY/My_github/sci_project`。以下为仓库相对标识，供执行助手定位：

- `docs/paper_writing_preparation_20260830/21_INNOVATION_BREAKTHROUGH_PORTFOLIO_AND_ACCEPTANCE_CN_20260903.md`
- `docs/paper_writing_preparation_20260830/22_POST_V11_RESULT_AUDIT_AND_NEW_INNOVATION_ROUTES_CN_20260903.md`
- `docs/paper_writing_preparation_20260830/23_LEARNABLE_AND_EARLY_FUSION_ROUTES_CN_20260903.md`
- `docs/paper_writing_preparation_20260830/25_OVERNIGHT_VALIDATION_CODE_AUDIT_AND_NEXT_STEPS_CN_20260904.md`
- `docs/paper_writing_preparation_20260830/26_CURRENT_CORRECTION_STATUS_AND_NEXT_BREAKTHROUGHS_CN_20260904.md`（18:33 状态快照；其“尚未运行”部分已被今晚新结果更新）
- `experiments/dynamic_fusion/innovation_v10_portfolio/PORTFOLIO_LEDGER.md`
- `experiments/dynamic_fusion/innovation_v11_regret_router/PORTFOLIO_LEDGER.md`
- `experiments/dynamic_fusion/innovation_v12_new_observables/PORTFOLIO_LEDGER.md`
- `experiments/dynamic_fusion/innovation_v12_early_fusion/03_scaif_small_gate/RUN_OPTIM_HEALTH.md`
- `experiments/dynamic_fusion/innovation_v12_early_fusion/04_clrpf_probe/CLRPF_PROBE.md`
- `experiments/dynamic_fusion/innovation_v12_new_observables/prs/PRS_G1_DECISION.md`
- `experiments/dynamic_fusion/innovation_v12_new_observables/detail_recovery/R0_DECISION.md`
- `experiments/dynamic_fusion/innovation_v12_new_observables/psmf/R0_DECISION.md`
- `scripts/innovation_v12_new_observables/ntof_render.py` 与 `run_r3_prs_g1.py`（本次直接核对强度轴定义）

最终建议：今晚主投入放在 N2 的“如何匹配”和 N3 的“哪些信息进入融合”；用低成本 N1 尽快检验联合统计，以 N4 留住效率方向。可学习/更早融合仍可研究，但应该由新目标的短证据引导，而不是以模块更复杂作为创新充分性的替代。
