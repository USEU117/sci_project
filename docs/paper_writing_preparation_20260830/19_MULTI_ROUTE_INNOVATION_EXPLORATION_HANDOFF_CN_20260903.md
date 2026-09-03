# 19. 多路线算法创新探索与并行执行任务书

日期：2026-09-03  
对象：下一位负责算法研究与实验的 AI 助手  
目标：一次性向多个不同机制方向展开，完成最小验证、严格淘汰和种子确认；不要等待用户逐轮提醒。

---

## 0. 给执行者的直接指令

请先完整阅读本文件、任务书 18、`docs/CURRENT_DYNAMIC_FUSION_STATUS.md` 和现有实验决策文件，然后持续执行到出现以下终止条件之一：

1. 至少一条路线通过 MPDD seed0 最小门，并完成 seed1/2 确认；
2. 所有列出的优先路线均按预注册门失败，并已形成完整负结果档案；
3. 出现必须由用户决定的外部数据许可、超大算力或研究范围变更。

不要每完成一个小步骤就停下来询问用户。对可逆、项目范围内的代码、配置、测试和 MPDD 开发实验自行推进。若环境支持并行子任务，可以并行做代码审计、CPU 信息价值探针和文献核查；GPU 推理必须排队，避免互相争抢显存。

本任务不是要求把六条路线全部做成完整模型，而是用低成本证据迅速筛掉无效方向，再把资源集中到真正通过门控的 1–2 条路线。

---

## 1. 当前事实：为什么必须换方向

### 1.1 已经成立的内容

- A1 是当前唯一可正式报告的主方法。
- v8 TCRR 证明文本热图对 A1 候选区域具有真实空间信息：旋转、平移或跨图打乱后信号明显下降。
- v8 在 MPDD 三个 seed、三个 shot 上 Pixel-AP 稳定提升约 +0.032～+0.036。

### 1.2 已经失败的内容

- v7 全局文本图像分数不够稳定。
- v8 双向区域重排在 BTAD 为 −0.006951，在 MVTec 为 −0.005820。
- v9 正常参考 median/MAD + 只增益虽然在 MPDD 为 +0.023270，但 metal_plate 为 −0.018230，未过安全门。
- 以前的标量融合、门控、子空间残差和多种文本/视觉组合也已有大量负结果。

### 1.3 从失败中得到的约束

- 不再继续调 TCRR 的倍率、阈值或按类别例外；那会形成明显的事后过拟合。
- 不再把“调用两个现成分支并加权”当作创新。
- 不再只看均值；必须同时看跨类别、跨 seed、最差类别和强对照。
- BTAD、MVTec 已被 v8 查看，不能再作为新路线的无偏最终验证集；它们只可用于明确标注的失败分析。
- VisA 是当前 AnomalyCLIP 检查点的来源域，也不适合验证新的文本模块。

---

## 2. 文献边界：哪些想法已经很拥挤

开始编码前必须把每条路线与以下工作做差异表。这里列的是最低限度，不代表穷尽检索。

- [PatchCore（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)：深度 patch 特征、正常记忆库和最近邻异常分数。
- [PNI（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.html)：位置条件与邻域条件的正常分布；因此“给 PatchCore 加坐标”本身不新。
- [AnomalyDINO（WACV 2025）](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)：DINOv2、few-shot 记忆库、增强和稳健尾部聚合；因此“换成 DINOv2”不新。
- [ReMP-AD（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html)：参考检索与多模态提示融合。
- [FastRef（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html)：查询条件下的原型细化、最优传输和异常抑制；因此普通 prototype refinement 已很拥挤。
- [DCP-SFR（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html)：缺陷线索放大与结构细化；普通热图 refinement 不足以构成新意。
- [ObjectCore（WACV 2026）](https://openaccess.thecvf.com/content/WACV2026/html/Fucka_ObjectCore_-_Efficient_Few-shot_Logical_Anomaly_Detection_using_Object_Representations_WACV_2026_paper.html)：对象表示与二分图匹配；对象级逻辑异常方向必须与它明确区分。
- [Anomaly as Non-Conformity（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Seo_Anomaly_as_Non-Conformity_via_Training-Free_Graph_Laplacian_Energy_Minimization_CVPR_2026_paper.html)：图拉普拉斯能量与 non-conformity；图/统计方向需要核对其方法细节。
- [RadioCore（CVPRW 2026）](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html)：RADIO 多尺度特征 + PatchCore；单纯换 backbone 只能作为基线。

执行者需要为每条进入 R1 的路线写一页 `NOVELTY_BOUNDARY.md`，至少回答：已有方法做了什么、我们新增的机制是什么、哪项消融能证明新增机制不是换名字。

---

## 3. 总体探索组合

建议同时启动六条路线，优先级如下。

| 代号 | 路线 | 与现有工作差异 | 成本 | 新颖潜力 | 建议 |
|---|---|---|---:|---:|---|
| A | CRAM：跨参考一致性记忆 | 不只取“最幸运的最近邻”，显式建模 K 个参考间的一致远离和不确定性 | 低 | 中高 | 第一优先 |
| B | MESP：多视图等变稳定与区域持续性 | 不增加分支，用反变换后跨视图持续存在的异常区域抑制偶发热点 | 中 | 高 | 第一优先 |
| C | CAPM：规范化对齐的位置条件记忆 | 先用正常参考完成几何对齐，再做位置约束近邻；低可靠时自动退回全局 A1 | 中高 | 中高 | 第二优先 |
| D | NORC：正常参考的区域级风险校准 | 输出区域 non-conformity/p-value 和可审计回退，不只追求 AP | 中 | 高 | 作为安全模块 |
| E | STR：支持集条件频谱纹理残差 | 用正常参考频域统计补充深层特征对细纹理的盲区，并由支持集自动决定是否启用 | 低中 | 中 | 快速备选 |
| F | SPRG：自监督部件关系图 | 不依赖文本或人工类别知识，建模部件数量、相对位置和关系异常 | 高 | 高 | 高风险长线 |

不得一开始就把六个模块融合。每条路线先单独回答一个明确科学问题，只有分别通过后才讨论兼容组合。

---

## 4. 路线 A：CRAM——Cross-Reference Agreement Memory

### 4.1 科学问题

标准 pooled memory bank 对每个测试 patch 只保留所有参考 patch 中的最小距离。一个偶然相似参考就可能掩盖缺陷。K-shot 情况下，真实异常是否会“对每一张正常参考都远”，而正常变化只对部分参考远？

### 4.2 最小实现

对测试 patch `q` 和第 r 张正常参考分别计算：

`d_r(q) = min_j [1 - cos(q, p_(r,j))]`

保留：

- `d_min = min_r d_r`：与 A1 pooled NN 等价的核心量；
- `d_med = median_r d_r`：跨参考典型距离；
- `gap = d_med - d_min`：是否只有一个“幸运参考”匹配；
- `mad = MAD_r(d_r)`：参考不确定性。

预注册两个候选，不得边跑边加：

- A0：`score = d_min`，必须重放 A1；
- A1：`score = d_min + 0.5 × max(0, d_med-d_min)`；
- A2：`score = d_min × (1 + 0.5 × clip(mad / normal_mad95, 0, 1))`。

k=1 时必须严格退回 A0；主要科学评价是 k=2/4。

### 4.3 正常侧校准

把每张正常参考轮流当 pseudo-query，其余参考建库，估计 `gap/mad` 的正常分位数。不得用测试正常图、测试统计或异常标签。若 k=1 无法 leave-one-out，则使用参考图的确定性轻微增强版本，仅用于校准，不把增强当独立样本做显著性夸大。

### 4.4 强对照

- 重复同一参考 K 次：结果不应虚假变好；
- 随机打乱参考归属：一致性收益应消失；
- 仅用 `d_med`：验证收益不是简单换聚合器；
- A0 identity replay：最大绝对误差 `≤1e-6`。

### 4.5 R0 门

MPDD seed0，k=2/4：

- 两个 shot 的平均 Pixel-AP 增益 `≥+0.005`；
- 至少 4/6 类平均为正；
- 最差类 `≥−0.015`；
- Pixel-AUROC 平均损失 `≥−0.002`；
- 真实参考相对打乱参考至少高 `+0.002`。

若只有 k=4 有效、k=2 失败，可以保留为“多参考专用模块”，但不能声称覆盖 1/2/4-shot。

---

## 5. 路线 B：MESP——Multi-view Equivariant Stability Persistence

### 5.1 科学问题

真实缺陷经过轻微光照变化、翻转或小幅几何变化并逆变换回原坐标后应持续出现；由插值、局部纹理或单次特征匹配产生的假热点通常不稳定。能否利用跨视图等变一致性提高异常图，而不引入第二个模型？

### 5.2 视图集合

只允许事先固定的、可逆或可对齐变换：

- 原图；
- 水平翻转；
- 亮度 `×0.9` 和 `×1.1`；
- 旋转 `−5°/+5°`，有效区域用 mask 排除边界。

禁止随机 crop 导致缺陷被裁掉；禁止为不同类别手选变换。

### 5.3 输出

对每个视图运行同一 A1/AnomalyDINO 路径，逆变换热图到原坐标，得到：

- `median_map`：跨视图中位数；
- `stability = 1 - normalized_MAD`；
- 连接区域的 persistence：该区域在多少视图、多少预注册分位阈值下保持重叠。

候选：

- B1：直接使用 `median_map`；
- B2：`base × (0.75 + 0.25×stability)`，只抑制不稳定热点；
- B3：只对 persistence≥4/5 的 A1 连通区域保留原分数，否则乘 0.8。

### 5.4 必做审计

- 在正常参考图上测逆变换误差，确认不是插值本身制造稳定性；
- 单调复制基线图作为伪多视图，不能通过 persistence 对照；
- 随机错位一个视图，真实方法应优于错位对照；
- 报告每张图增加的推理时间和显存。

### 5.5 R0 门

MPDD seed0，1/2/4-shot：

- 平均 Pixel-AP `≥+0.005`；
- 3/3 shot 为正；
- 至少 4/6 类为正；
- 最差类 `≥−0.015`；
- Pixel-AUROC `≥−0.002`；
- 相对错位视图对照的增益 `≥+0.002`。

若精度通过但推理时间超过 A1 的 5 倍，必须尝试只保留原图+翻转+一个光照视图的 B-lite，并单独报告成本。

---

## 6. 路线 C：CAPM——Canonicalized Alignment-aware Positional Memory

### 6.1 科学问题

全局 PatchCore 允许任意位置 patch 匹配，容易漏掉“局部外观正常但出现在错误位置”的异常。直接限制坐标又会被姿态偏移破坏。能否先把参考图与查询图规范化对齐，再在可信坐标邻域内匹配？

### 6.2 最小实现

1. 使用冻结 DINO patch 特征做 query-reference mutual nearest neighbors。
2. 用 RANSAC 估计 affine 或 homography；不得用 GT mask。
3. 将参考 patch 坐标映射到 query 坐标。
4. 对每个 query patch 同时计算：
   - `d_global`：原 A1 全局最近邻；
   - `d_pos`：仅在对齐坐标半径 2 个 patch 内搜索；
   - `alignment_reliability`：RANSAC inlier ratio、重投影误差、参考间一致性。
5. 若可靠性低于固定门，输出严格等于 `d_global`；否则 `d_global + 0.25×relu(d_pos-d_global)`。

### 6.3 重要边界

PNI 已做位置/邻域建模，RegAD 类工作已做 registration。因此这里的潜在新意只能是“few-shot、训练自由、查询条件几何规范化 + 可验证的可靠性回退”，不能只写 position-aware memory。

### 6.4 强对照

- 不做对齐直接位置限制；
- 随机 homography；
- 全局 A1；
- 仅对齐后仍做全局搜索；
- 可靠性回退关闭。

### 6.5 R0 门

- 首先只跑几何稳定的 MPDD 类和全六类汇总，两者都报告，不能只挑好看的子集。
- 全六类平均 Pixel-AP `≥+0.003`；至少 4/6 类为正；最差 `≥−0.015`。
- 低可靠样本的输出与 A1 最大误差 `≤1e-6`。
- 真对齐相对随机对齐至少高 `+0.003`。
- 对结构/错位型异常的 evaluator-only 分组增益应明显高于纯纹理组；这个分组只用于解释，不能参与方法门控。

---

## 7. 路线 D：NORC——Normal-Only Region Conformalization

### 7.1 科学问题

过去多个模块失败的共同原因是“辅助证据存在，但不知道什么时候可信”。与其再训练一个黑盒 gate，是否能从正常参考构造区域级 non-conformity 分布，为每个候选区域给出可审计的 p-value/置信等级？

### 7.2 最小实现

1. 从正常参考做 leave-one-reference-out A1 推理。
2. 使用固定增强产生有限的正常 pseudo-query；增强族和数量必须预注册。
3. 将正常候选区域按面积档和粗位置档分层，记录区域 max、mean、P90、多视图稳定度等 non-conformity。
4. 测试候选区域使用有限样本 conformal 排名：`p=(1+#calibration_score≥test_score)/(n+1)`。
5. 仅当 `p≤0.05` 时允许辅助模块改变 A1；否则 identity 回退。

### 7.3 不要犯的错误

- 由同一张增强图产生的大量 patch 不是独立样本，不能把 patch 数当有效样本量。
- 不得声称严格覆盖保证，除非交换性条件和校准单元被清楚证明。
- 单调 p-value 变换不会自动提高 AP；该路线的价值主要是跨域安全、正常图误报率和可解释回退。

### 7.4 R0 门

- identity 回退测试必须通过；
- MPDD seed0 Pixel-AP 不低于 A1 `−0.002`；
- 正常图 image-FPR 或正常像素高分率相对未校准辅助模块下降至少 20%；
- 跨 seed 的 nominal coverage 偏差不超过 5 个百分点；
- 如果与路线 A/B/C 组合，最差类别回退必须至少改善 50%，同时保留其平均增益的 50% 以上。

NORC 可以成为论文第二贡献，但前提是风险指标真实改善；不要只报告 AP。

---

## 8. 路线 E：STR——Support-Conditioned Texture Residual

### 8.1 科学问题

DINO/CLIP 深层 patch 对语义和结构强，但可能忽略细小周期纹理、划痕和高频变化。能否用正常参考的频谱统计提供一个计算便宜、与 A1 误差不同的证据？

### 8.2 最小实现

- 输入灰度与颜色对手通道；
- 固定 2 层 Haar/DWT 或 Laplacian pyramid，不训练；
- 每个频带在正常参考上估计局部 median/MAD；
- 测试图生成多频带稳健 z-score，取跨频带 trimmed mean；
- 先做信息价值诊断，证明 STR 在 A1 错误区域上有增量，不要直接融合。

若诊断通过，再试：

- E1：A1 候选区域内 `max(A1, calibrated_STR)`；
- E2：只有支持集频谱熵高且参考间稳定时才启用，否则 identity。

### 8.3 强对照

- 随机相位但相同幅度谱；
- 仅 RGB 梯度幅值；
- 不做正常参考校准；
- 频谱图空间错位。

### 8.4 R0 门

- 区域信息价值 AP 相对 A1 区域分数 `≥+0.05`，且至少 4/6 类为正；
- 真空间 STR 相对错位对照 `≥+0.02`；
- 最小融合后 Pixel-AP `≥+0.003`，最差类 `≥−0.01`；
- 推理额外耗时低于 A1 的 20%。

如果只在一个纹理类别有效，归档为类别专项观察，不升级为统一算法。

---

## 9. 路线 F：SPRG——Self-supervised Part-Relation Graph

### 9.1 科学问题

patch 最近邻能发现局部外观异常，却难以发现部件缺失、数量错误、相对位置错误等逻辑异常。是否能不依赖文本和人工类别知识，从少量正常参考中发现稳定部件并建模关系？

### 9.2 最小实现

1. 使用冻结 DINO patch token，在正常参考内做空间约束聚类，得到 4–16 个稳定 part nodes。
2. 节点特征：外观原型、面积、质心、二阶形状、邻接关系。
3. 查询图产生相同节点集合。
4. 用带 dummy node 的二分图匹配或轻量 Gromov-Wasserstein 比较：
   - node appearance cost；
   - missing/extra node cost；
   - pairwise geometry cost。
5. 将 node cost 投影回 patch 图，A1 保留局部结构缺陷能力。

### 9.3 文献差异要求

ObjectCore 已使用对象表示和二分图匹配。只有在以下差异成立时才继续：

- 不依赖开放词汇检测器或人工对象类别；
- part discovery 完全来自 K-shot 正常参考；
- 同一模型同时输出逻辑分数和可定位 patch 图；
- 有 reference-consistency/identity fallback。

### 9.4 数据与门

MPDD 不是验证逻辑异常的理想数据。先做非常小的可行性探针；若部件发现不稳定，立即停止。若稳定，再把 [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) 作为专门协议候选。该数据为 CC BY-NC-SA 4.0，下载和使用前记录许可；不得静默混入现有结构异常表。

R0 可行性门：

- 正常参考间 node matching 成功率 `≥90%`；
- 轻微正常增强下关系分数的 95 分位稳定；
- 合成“删除/交换一个稳定节点”的 counterfactual 分数明显高于正常增强，效应量 `Cohen's d≥1`；
- 若无法满足，不进入真实异常评估。

---

## 10. 不建议作为主路线的方向

以下内容可以做基线或消融，但不要单独包装为论文创新：

- 再换一个现成视觉 backbone；RadioCore 已说明这种路线竞争激烈。
- 再做固定权重加法、乘法、max 或手工类别 gate。
- 对 BTAD-03、metal_plate、hazelnut 单独设阈值。
- 继续调整 TCRR 的 `q=0.95`、1.5 倍率、z=3/6。
- 直接使用测试图统计做归一化或 gate。
- 没有强对照的“注意力模块”“自适应模块”。
- 大规模 synthetic anomaly 训练作为第一选择；相关 refinement、生成和 cue amplification 方法已经非常密集，成本也高。

---

## 11. 统一实验协议

### 11.1 数据角色

- MPDD：唯一算法开发集。
- seed0：R0 探索与候选选择。
- seed1/2：冻结后的确认；看结果后不得回改公式。
- BTAD/MVTec：已被 v8 消耗，只能做带“post-hoc diagnosis”标签的分析。
- 新最终验证集：在候选方法冻结后再选择、下载和生成 manifest。

候选外部集：

- [Real-IAD](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.html)：30 个对象、约 15 万张、多视角，区分力强但计算和存储成本高。
- [AeBAD](https://arxiv.org/abs/2304.02216)：含视角、尺度和光照域偏移，适合检验 CAPM/MESP 的跨域稳健性。
- MVTec LOCO AD：只适合 SPRG/逻辑异常路线，许可为 CC BY-NC-SA 4.0。

不要现在就为所有路线下载数据。只有通过 seed1/2 的候选才有资格触碰新的外部集。

### 11.2 统一指标

主指标：

- Pixel-AP；
- Pixel-AUROC；
- AUPRO；
- Image-AUROC/AP/F1-max；
- 每类、每 seed、每 shot 变化。

安全指标：

- 最差类别变化；
- 正常图假阳性率；
- FPR@95TPR（适用时）；
- identity fallback 比例；
- 强对照差值；
- 推理时间、峰值显存、额外参数。

### 11.3 禁止泄漏

方法函数签名中不得出现 `gt_masks/gt_labels`。所有地图、gate、可靠性和参数必须先冻结，之后 evaluator 才能加载 GT。新增测试必须扫描实现源码中的禁用键，并做“改变 GT 不改变输出图”的测试。

### 11.4 统一 R1 确认门

某路线通过自身 R0 后，原公式不变跑 MPDD 全 3 seeds × 3 shots：

- 九个 seed-shot 中至少 7 个 Pixel-AP 增益为正；
- 类别配置正增益比例至少 60%；
- 平均 Pixel-AP 增益 `≥+0.005`；
- 最差类别平均 `≥−0.02`；
- Pixel-AUROC 平均损失 `≥−0.002`；
- 类别聚类 bootstrap 95% CI 下界大于 0；
- 对照差异仍满足 R0 预设。

任何一项失败都不能进入外部验证。若方法目标是风险控制而非 AP（NORC），使用其专属安全门，但必须事先写清楚。

---

## 12. 执行顺序与并行安排

### Phase 0：共同基础审计

1. 确认 git 状态，保存用户已有改动。
2. 运行 v7/v8/v9 相关测试和 A1 identity replay。
3. 建立统一目录与结果 schema。
4. 写 `PORTFOLIO_LEDGER.md`，每条路线只有 `NOT_STARTED/RUNNING/PASS/FAIL/ARCHIVED` 五种状态。

### Phase 1：低成本信息价值探针

优先并行：

- Agent/分支 A：CRAM；
- Agent/分支 B：MESP；
- Agent/分支 C：STR；
- 主执行者：CAPM 可行性和 NORC 校准单元设计。

每条路线先用 1–2 个类别做 CPU/小 GPU smoke，只验证形状、identity、无泄漏和运行时间；不得在 smoke 结果上选阈值。随后按预注册一次性跑 MPDD seed0 全六类。

### Phase 2：第一次淘汰

- R0 全门通过：进入 Phase 3。
- 只差一个统计稳定性门、且均值/最差类均安全：允许一次预先说明原因的确认，不允许改公式。
- 精度均值为正但靠单类贡献、最差类越线或对照失败：ARCHIVED。
- 均值为负：立即停止，不做更多调参。

### Phase 3：seed1/2 确认

冻结代码哈希、配置哈希和输入清单，再运行。通过统一 R1 门后，最多保留两条候选。

### Phase 4：兼容性判断

只有两条路线的误差互补且机制不同，才允许组合。例如：

- CRAM + NORC：一致性分数 + 正常风险回退，逻辑上兼容；
- CAPM + MESP：几何对齐 + 多视图稳定，可能兼容但成本高；
- CRAM + FastRef 式原型细化：机制重叠，不建议；
- MESP + STR：都可能提高局部图，必须先证明错误不相关。

组合必须重新预注册，不得从所有模块中做组合搜索。

### Phase 5：新外部验证

先确定数据许可、磁盘、划分和一次性 gate，再运行。外部失败后不得换阈值重跑并继续称其为验证。

---

## 13. 目录与交付规范

建议建立：

```text
configs/innovation_v10_portfolio/
  cram/
  mesp/
  capm/
  norc/
  str/
  sprg/
src/industrial_ad/innovation_v10_portfolio/
scripts/innovation_v10_portfolio/
tests/innovation_v10_portfolio/
experiments/dynamic_fusion/innovation_v10_portfolio/
  PORTFOLIO_LEDGER.md
  <route>/R0_*/
  <route>/R1_*/
outputs/dynamic_fusion/innovation_v10_portfolio/   # 大缓存，gitignored
```

每条路线必须至少交付：

- `R0_PROTOCOL.json`：在结果前创建；
- `R0_RESULT.json`：全数值、逐类、逐配置；
- `R0_DECISION.md`：PASS/FAIL 和每个 gate；
- `FAILURE_ANALYSIS.md`：失败也要写；
- 单元测试与无泄漏测试；
- 运行命令、环境、耗时和哈希；
- `NOVELTY_BOUNDARY.md`：进入 R1 时必需。

主 ledger 至少包含：

| 路线 | 状态 | 主增益 | 最差类 | 对照差 | 成本 | 下一步 |
|---|---|---:|---:|---:|---:|---|

---

## 14. 最终情景与自动决策

### Scenario A：两条及以上路线通过 R1

选择机制最不同、最差类最安全的两条。分别完成文献边界，再做一次小规模兼容性探针。不要自动把所有通过路线融合。

### Scenario B：仅一条路线通过 R1

将其作为唯一新方法候选，补 AUPRO、图像指标、消融、复杂度和新外部集。论文主线改为“A1 + 一个明确的新机制”。

### Scenario C：只有 NORC 通过

论文可转向“few-shot 工业异常检测的风险受控后处理/安全回退”，但必须以误报和覆盖率为核心，而不是虚构 AP 优势。

### Scenario D：所有短期路线失败，SPRG 可行性通过

停止在当前论文上继续堆小模块；把 SPRG 作为后续独立研究项目，当前论文维持 A1。

### Scenario E：全部失败

完整归档负结果，停止算法搜索。继续完善 A1 的复现、统计、消融、图表与论文写作。SCI 四区并不要求一定有复杂新网络，但要求问题、方法、实验和边界陈述自洽。

---

## 15. 交给执行者的首批具体动作

按以下顺序直接开始：

1. 创建 v10 portfolio 目录、统一结果 schema 和 ledger。
2. 同时写 CRAM、MESP、STR 三条 R0 protocol；在任何真实标签结果出现前冻结。
3. CRAM 先验证能否从现有 feature cache 按参考图分离距离；若缓存已丢失 reference-image 归属，先修复导出 provenance，不要猜。
4. MESP 先在 10 张图上验证变换—逆变换几何误差和运行成本。
5. STR 先做区域信息价值和空间错位对照，不直接融合。
6. CAPM 先统计六类 mutual match/RANSAC inlier ratio；若大多数图低于 0.3，路线直接 FAIL，不进入像素评估。
7. NORC 先定义校准单元和有效样本量，写清交换性限制，再编码。
8. 每条通过 smoke 后一次性跑 seed0 全六类，更新 ledger。
9. 只有 R0 PASS 的路线才跑 seed1/2。
10. 最后给用户一个总表：哪些路线做了、为什么成败、哪一条值得写论文、下一步是否需要新数据。

执行过程中以证据为准。如果结果显示所有方向都无稳定增益，应明确停止，而不是继续发明阈值。

