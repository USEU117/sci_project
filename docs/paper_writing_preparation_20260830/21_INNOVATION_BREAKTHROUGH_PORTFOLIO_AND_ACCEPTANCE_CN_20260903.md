# 21. 创新点复盘、动态融合再设计与多突破口实验验收书

日期：2026-09-03  
用途：交给后续 AI 助手继续做方法设计、实现、实验和验收  
状态：研究与执行规划；本文件中的新方法名均为工作名，未通过硬门前不得写入论文贡献  

---

## 0. 先给结论

老师指出“两个视觉分支本身都是成熟方法，只做融合可能创新偏薄”，这个判断是合理的，但不等于项目必须推倒重来。真正需要改变的是创新落点：不能再把“用了 DINO + CLIP、做了 concat/加权”本身当贡献，而要提出并验证一个现有分支没有回答的新科学问题。

综合仓库里已经完成的十轮以上探索、负结果边界和 2024—2026 年近邻论文，本项目目前最值得继续的三个突破口是：

1. **RSR：Regret-Supervised Regional Routing（后悔值监督的区域动态路由）**。这是最贴近最初“动态融合”设想、工程复用度最高的方案。它不再用熵、分数大小或简单分歧猜测哪个分支可靠，而是在正常参考图上生成反事实缺陷，以各专家对已知伪缺陷的真实损失差作为“路由监督”，学习预测“选错专家会损失多少”。推理时按区域选择专家，并在不确定时精确回退 A1。
2. **NR-MoE：Normality-Regime Mixture of Experts（正常性机制混合专家）**。这是创新幅度最大的方案。它不再让两个相近视觉编码器互相竞争，而是组织外观记忆、语义区域、局部流形、上下文修复、组件关系等不同“正常性定义”的专家；根据少量正常 support 的结构/纹理/重复性统计判断当前产品与区域属于哪种机制。项目已有结果已经证明这些专家的优劣会随类别发生明显翻转，具备立题依据。
3. **BC-MCR：Blind-Center Masked Context Repair（盲中心参考条件上下文修复）**。这是最有机会独立成方法模块的非融合路线。中心 patch 完全不可见，只根据邻域和正常 support 预测“这里正常时应该是什么”，检测 A1 无法处理的“局部外观见过但位置、邻接或数量不合理”。旧的固定检索式 context 在 permutation/duplicate 上比 A1 高约 0.21 AUROC，说明方向有信号；它在 missing 上失败，恰好给出了需要训练式盲中心预测器而不是继续 KNN 的理由。

推荐执行顺序不是把三者堆在一起，而是：

```text
现有专家池 Oracle/贡献审计
  ├─ 区域级剩余 headroom 足够 → 先做 RSR
  ├─ 优劣主要按产品/缺陷机制翻转 → 立项 NR-MoE（新协议）
  └─ 专家池 headroom 不够 → 单独做 BC-MCR，创造新的结构证据

任一路线通过 MPDD Full Gate
  → 冻结唯一 winner
  → Real-IAD 小规模确认（可选）
  → MVTec AD 2 private test 一次性最终验证
```

最不建议继续的是：再调固定权重、再换一种熵/置信度、继续扩大 TCRR 倍率、把注意力/LoRA/多层拼接直接称创新、把多个已失败模块无门槛堆叠。这些要么已经被本项目实证否定，要么与近邻工作高度拥挤。

---

## 1. 当前 A1 与最初动态融合的真实位置

### 1.1 当前可靠主方法

A1 是冻结双视觉特征融合：

```text
DINOv2 ViT-B/14 patch（768） ─ L2 ─┐
                                    ├─ 等权 concat（1536）─ L2 ─ normal memory KNN(k=1)
AnomalyCLIP image tower patch（768）─ L2/网格对齐 ─┘
```

- 第二分支是 **CLIP 图像塔 patch feature**，不是文本特征；
- 每个 seed/shot 只用 K 张正常 support 建库；
- `pca_dim=0, whiten=0, dino_weight=0.5, stride=8, map=448`；
- 没有逐图/逐像素动态权重；
- 相对 matched feature-DINO-only，四数据集 Pixel-AP 增益约为 MPDD `+0.0258`、BTAD `+0.0249`、VisA `+0.0524`、MVTec `+0.0320`；
- BTAD 的 Image-AP 与 Image-F1-max 仍下降，不能声称所有指标全面提升。

因此 A1 是一个稳定、干净、可复现的强基线，但论文创新不能只写成“双分支拼接”。

### 1.2 最初动态融合的核心想法

最初方案希望让视觉分支和文本/CLIP 分支根据样本或像素的可靠性动态分工：

```text
两个分支的分数/热图
  → 正常参考校准
  → 熵、分歧、集中度、跨视图稳定性等可靠性特征
  → 图像级/像素级权重或分支选择
  → 动态融合与视觉回退
```

这个研究问题本身仍成立，但旧实现把三个不同概念混在了一起：

1. **异常证据**：离正常记忆多远；
2. **置信度外观**：分数是否尖锐、熵是否低；
3. **专家能力**：在这个区域上，某专家是否真的比另一个更接近正确答案。

旧路由只观测前两类量，却试图推断第三类量。实验已经证明：低熵可能只是 sigmoid 饱和，分支一致可能是一起错，离正常更近也不等于更可靠。新的动态方案必须直接学习或估计“专家 regret/competence”，而不是继续发明分数统计量。

---

## 2. 之前已经做过什么：完整创新探索地图

以下表格按“科学假设—结果—留下的约束”整理，不把工程完成误写成科研成功。

| 阶段/路线 | 核心设想 | 关键结果 | 当前结论与留下的知识 |
|---|---|---|---|
| Dynamic Fusion V1 | 用正常参考校准后的熵与分歧做图像/像素动态权重 | MVTec 视觉校准值 `>=0.999` 比例约 99.99%，强视觉排序被饱和破坏；动态总体低于 AnomalyDINO | 不能把 sigmoid 后低熵当可靠性；图像与像素任务需分开 |
| V2 safe router | 排序保持、support 检查、视觉默认、失败回退 | 工程框架成立，可靠性特征无法预测真实分支优劣 | 保留 RouterInput/EvaluationTarget、fallback、reason code；方法归档 |
| V3/V3.2 region rescue | 视觉提候选，文本只在区域内有界救援 | held-out Pixel-AP 约 0 或微负 | 候选框保护安全，却也禁止发现视觉完全漏检区域 |
| V3.3/V3.4/V3.5 | z-score、逐像素 Oracle、层次 gate、缺陷词 | 旧校准使用测试 mask/label，存在泄漏；clean 固定融合可小幅正，动态不超过静态 | 污染结果不能用于论文；“Oracle 有互补”不等于“可预测” |
| A1 | DINO + CLIP-image patch 等权 concat 后 KNN | 四数据集 Pixel-AP 相对 matched DINO-only 均正 | 当前唯一正式主方法，但本身创新幅度有限 |
| Route-D predictability | 从分歧、A1 分布、DINO/CLIP rank 预测何时修正 A1 | LOCO mean AUROC `0.592`，置乱 `0.616` | 旧的 8 个图像级统计无可预测性；不能换分类器重跑同一特征 |
| V4 explicit text | 强视觉锚点 + 显式文本 + 动态机制 | 官方 SubspaceAD 虽总体正，但 connector `−0.1167` 触发安全失败；文本对强视觉无稳定 headroom | 单个更强视觉分支不能自动解决路由；最差类必须设底线 |
| RCEC | DINO 邻居条件下检查 CLIP 是否认可同一正常邻域 | 12 候选 35/36 candidate-shot 下降；最佳三-shot `−0.0071` | 成对邻域分歧存在不等于能作为异常分数；不要继续调 `k/λ` |
| A2-LNDC | 局部正常密度校准 | 最佳约 `−0.085` | 密度除法放大 few-shot 记忆噪声 |
| A2-DSAM | 对齐后空间局部记忆 | 半径越小越差，最佳仍 `−0.012` | 对齐有效，但 rigid/local window 与 MPDD 的全局匹配需求冲突 |
| A2-CEQA | 双编码器共识 pseudo-normal query adaptation | 4 候选全正，最佳 `+0.002799`，低于门且不胜 A1-rank 控制 | query adaptation 只有微弱信号；FastRef/ConceptADapt 已使该方向拥挤 |
| A2-DEVA/MESP | 通过增强/多视图等变稳定扩库或融合 | 变化近零；MESP 真对齐不优于错位对照 | 增益主要是平滑，不是等变机制；不要扩大增强表 |
| A2-NCPRA | 正常样本非线性交叉预测残差 | `−0.005` 到 `−0.008` | 同位置跨编码器预测没有新增证据；且训练实现曾有复现问题 |
| A2-FAGR | 特征亲和图平滑 | `−0.005` 到 `−0.008`，近似 uniform smoothing | 后处理平滑没有创造新信息 |
| A4-SF-NM | wavelet 高频正常记忆 | 所有尺度层 oracle headroom 为 0，频率 AUROC 约 0.2 | 当前固定频谱描述符不适合 MPDD；不要继续换 wavelet 参数 |
| A4-RG-MCR diagnostic | ring context 预测中心 | permutation/duplicate 相对 A1 约 `+0.21`，missing 失败 | 结构上下文是真信号，但 fixed retrieval 无法重建“缺失” |
| A4-RG-OT | 节点/关系 OT | node-OT 约随机 | patch anchor 关系不足；没有稳定组件时不要做重型 OT |
| CASF | 非对称伪异常监督跨分支融合 | 放大探针仅 bracket_white 1/6 类达标，绝对 Dice ≤0.096 | 伪异常监督价值类条件化；当前分割头/生成方式不成立 |
| DG-SAFE | A1 与官方 SubspaceAD 正常-only 安全融合 | 固定 mean `+0.0164`，但可靠性 `ρ=0.3387<0.40`，connector 无法被识别 | 专家互补是真实的，失败点仍是 competence 预测 |
| GLSD/global text | 文本做图像筛查、A1 做像素定位 | mean ΔImage-AP `+0.0288`，但仅 6/9 正且 bootstrap CI 下界 `−0.0226` | 文本对低 shot/特定类有效，不足以冻结双输出系统 |
| TCRR v8 | 文本条件候选区域重排 | MPDD `+0.032~+0.036` 且空间错位控制通过；BTAD `−0.00695`、MVTec `−0.00582` | 区域文本信息真实，但固定倍率跨域不安全 |
| NC-SafeTCRR v9 | 正常参考校准、只增益、identity fallback | MPDD `+0.02327`，metal_plate `−0.01823` 触发归档 | support 校准仍不能判断语义证据何时可信 |
| CRAM | 多参考一致性记忆 | 均值负，shuffled 不比真实差 | K≤4 的 reference agreement 太粗，不能继续同式变体 |
| CAPM | RANSAC 对齐后位置条件记忆 | mean `−0.0275`，random 对齐不差于真实 | MPDD 位姿变化和 diffuse 抬升使位置记忆有害 |
| NORC | 正常参考区域 conformal 风险门 | K≤4 时 p-value 太离散，且没有通过的伙伴专家 | 可作未来部署模块，不能在当前 few-shot 下单独成主创新 |
| STR | support-conditioned texture residual | region info-value `−0.027` | 当前纹理残差不如 A1；只剩类别专项观察 |
| SPRG | 自监督部件关系图 | 跨样本节点稳定匹配仅 36%（门 90%） | MPDD 不具备稳定部件图；应换 LOCO 类数据或更强实例分割 |
| LLSE | top-8 局部线性流形残差 | seed0 `+0.0089`，三 seed均值约 `+0.0025`；tubes/metal_plate 三 seed稳定正 | 机制真实但 reference-seed 不稳；可保留为专门专家，不可单独升级 |
| CSS | 图内邻域自一致性 | bracket 类可 `+0.08~+0.16`，metal_plate 最差 `−0.365` | reference-free 证据能救 A1 弱类，但大块缺陷内部自一致；强烈支持“机制专家”而非统一固定公式 |

### 2.1 这些负结果共同说明什么

1. **A1 已很强，单调后处理几乎没有空间。** 对最终 map 做平滑、z-score、密度除法、固定加权，通常只是改变刻度或扩散响应。
2. **真正互补存在，但高度条件化。** TCRR 对 MPDD 强、外部失败；Subspace 对多数 MPDD 类正、connector 大幅负；LLSE 对 tubes/metal_plate 稳定正；CSS 恰好救 bracket、重伤 metal_plate。这不是“没有创新点”，而是“不同正常性机制适用于不同缺陷形态”。
3. **旧 gate 学错了目标。** 旧特征尝试从分数形状猜专家正确性，却没有获得“选错专家的损失”监督。
4. **两个成熟分支不是致命问题。** PatchCore、FastRef、G²SF 等工作也建立在成熟 backbone/记忆机制上。关键在于新模块是否定义了新的决策问题、有独立机制控制，并在强基线上产生不可被简单 ensemble 解释的增益。
5. **数据协议已经成为主要风险。** MPDD 被反复用于开发；BTAD/MVTec 已在 v8 外部实验中被查看；VisA 又是 AnomalyCLIP source domain。任何新方法都需要新的最终外部验证集，否则创新即使涨点也很难证明泛化。

---

## 3. 近邻文献边界：哪些路已经拥挤

以下只使用论文主页、CVF、PMLR、OpenReview、官方数据页等一手来源。接手者在写论文前仍须逐篇核对公式、代码许可和最终发表信息。

| 近邻工作 | 已经覆盖的核心 | 对本项目的约束 |
|---|---|---|
| [AnomalyDINO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) | DINOv2 patch + normal memory 的 few-shot 检测 | 不能把 DINO KNN 或换 backbone 写成创新 |
| [FastRef, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html) | query 特征转移、prototype refinement、OT 异常抑制 | 继续 CEQA/query-adaptive prototype 很容易碰撞 |
| [ConceptADapt, 2026 preprint](https://arxiv.org/abs/2608.05743) | 正常 concepts、稀疏重构、dynamic attention、LoRA test-time adaptation | “加 attention/LoRA/正常概念重构”不能直接作为新意 |
| [SubspaceAD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html) | few-shot PCA/subspace residual | PCA/whitening/线性子空间已拥挤且本项目也失败 |
| [RadioCore, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html) | foundation feature 的多尺度 training-free memory | 简单多层/多尺度拼接只能当对照 |
| [DCP-SFR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html) | 缺陷线索放大、结构重构与防 cue fading | 多层缺陷 cue refinement 已是直接近邻 |
| [MoECLIP, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Park_MoECLIP_Patch-Specialized_Experts_for_Zero-shot_Anomaly_Detection_CVPR_2026_paper.html) | CLIP 内部 patch-specialized MoE、专家多样性约束 | “patch router + 多专家”本身不新；本项目须路由不同正常性机制，并以 regret 为目标 |
| [AnoPLe, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Bidirectional_Multimodal_Prompt_Learning_with_Scale-Aware_Training_for_Few-Shot_Multi-Class_CVPR_2026_paper.html) | 双向视觉—文本 prompt、scale-aware training | 双向 prompt 交互或尺度 prefix 不宜作为本项目主创新 |
| [G²SF, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.html) | 各向异性局部距离、可学习几何 score fusion | 学一个局部权重/metric 网络与其高度相邻 |
| [UniVAD, CVPR 2025](https://www.openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html) | component clustering、component-aware matching、graph modeling | 组件图必须有不同的问题定义与强控制 |
| [ObjectCore, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Fucka_ObjectCore_-_Efficient_Few-shot_Logical_Anomaly_Detection_using_Object_Representations_WACV_2026_paper.html) | 对象集合表示与 support-query bipartite matching | 逻辑异常路线不能只做“分组件再匹配” |
| [RealNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html) | 真实感合成异常、feature selection、残差选择 | 伪异常训练必须证明不是学 generator artifact |
| [PGBL, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liao_Multi-Prototype_Compactness_and_Boundary-Aware_Synthesis_for_Unsupervised_Anomaly_Detection_CVPR_2026_paper.html) | 多 prototype normality + boundary-aware synthesis | “在正常边界合成再训练判别器”已有直接近邻 |
| [InCTRL, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.html) | 源域训练的 generalist residual learner，target 用 few-shot normal prompt | 跨数据集元路由可做，但须强调“专家 regret/normality regime”而非普通 residual learning |
| [Sample-Aware Model Selection, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xue_Enhancing_the_Power_of_OOD_Detection_via_Sample-Aware_Model_Selection_CVPR_2024_paper.html) | 每样本选择预训练 OOD detector | 证明“模型选择”是合理问题，也意味着样本级选择不能声称首次提出 |
| [Uncertainty Is Not Enough / VI-MoLE, 2026 preprint](https://arxiv.org/abs/2608.02528) | 预测调用额外专家后的反事实风险/价值，以证书决定继续路由或 abstain | “不确定性不等于专家价值、应预测残余风险/regret”这一一般思想已有直接近邻；RSR 只能主张工业区域、伪缺陷 competence 和 normal-support 条件化的具体方法 |
| [MVTec AD 2 官方页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | 8 个新工业场景、光照分布漂移、public/private test | 最适合作为本项目尚未消费的一次性最终外部验证 |
| [Real-IAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.html) | 30 类、约 150K 高分辨率、多视角 | 可作跨类别元训练/中间确认，但资源与多视角协议要单独处理 |

由此可见，论文创新最安全的表达不是“首次动态融合 DINO 与 CLIP”，而是：

> 在极少正常参考条件下，把动态融合重新定义为“估计不同正常性机制在局部区域的条件 regret”，并通过反事实缺陷、结构化区域路由和保守回退控制选择错误；或者进一步学习由 support 描述的 normality regime，在外观、语义、上下文和关系专家之间进行任务条件化选择。

---

## 4. 突破口一（最高优先）：RSR 后悔值监督的区域动态路由

### 4.1 为什么它比旧动态融合更像一个新方法

旧 gate 的目标隐含为“分数尖锐/稳定 = 专家正确”。RSR 直接定义专家 regret：

对带伪缺陷 mask 的正常图区域 (r)，专家 (e) 的损失为

\[
L_e(r)=\mathrm{BCE}(M_r,S_e)+\alpha\,[1-\mathrm{AP}(M_r,S_e)]
        +\beta\,\mathrm{FP}_{e,\bar r}.
\]

选择专家 (e) 相对区域最优专家的 regret 为

\[
R_e(r)=L_e(r)-\min_j L_j(r).
\]

路由器学习的是 (\hat R_e(r))，而不是直接学习异常 mask。推理时取预计 regret 最小的专家；如果最小 regret 与次小差距不足，或区域超出 support 分布，则回退 A1。

这使创新点落在三个可验证机制上：

1. **regret-supervised competence estimation**：监督“哪个专家在什么区域会错”；
2. **region-structured routing**：以连通候选/超像素为决策单位，不做独立噪声像素权重；
3. **selective fallback**：覆盖率不足时精确返回 A1，不要求每个区域都动态。

必须注意：2026 年 8 月的 VI-MoLE 预印本已经明确提出“不确定性不足以决定是否调用专家”，并学习额外专家带来的反事实风险/价值。因此，RSR **不能**宣称首次提出 regret/value-of-information routing。可能成立的创新边界只剩：工业像素异常检测中的区域级专家 competence、用 normal-support 反事实缺陷产生监督、不同 normality mechanism 专家、空间一致路由及 A1 精确回退。正式投稿前必须把两者目标、监督、决策单位、保证和实验设置逐式比较；若最终公式实质等价，RSR 降为应用迁移而不是核心理论创新。

### 4.2 专家池必须先小后大

第一版最多使用四个已有专家，防止搜索爆炸：

- `E0`: A1（默认专家）；
- `E1`: TCRR raw region evidence（不是 v8 固定倍率输出）；
- `E2`: LLSE residual；
- `E3`: CSS boundary/self-consistency。

SubspaceAD 需要 giant 权重和额外图缓存，可在第一轮 Oracle 证明它有独立贡献后再替换最弱专家，不能直接变成第五个候选。

### 4.3 伪缺陷不是普通 CutPaste

至少四个互相区别的 corruption family：

- 局部纹理/颜色破坏；
- 细划痕/边界型；
- 结构搬移/复制；
- 删除/遮挡/缺件型。

每轮训练留出一个 family 完全不参与训练，用于检验路由器是否学会专家能力而不是生成器 ID。可以先用便宜的 CutPaste/Perlin/swap/delete；只有 transfer 失败才考虑 MIRAGE、RealNet/扩散式更真实生成，不得一开始引入重型生成模型。

### 4.4 RouterInput

区域特征只能来自预测与正常 support：

- 每专家区域内 `mean/max/q90/q99/std`；
- 区域面积、紧致度、边界/内部响应比；
- 专家两两 rank/IoU/disagreement；
- 对轻微光照、翻转、缩放的逆变换一致性；
- 区域到各专家 normal LOO 分布的距离；
- support 的纹理平稳性、位置重复性和参考多样性；
- 不得包含 dataset/category 名、测试 label/mask、异常类型字符串或测试集聚合统计。

### 4.5 最小实现

优先用可审计的小模型：

```text
区域统计（约 40–80 维）
  → 2-layer MLP / monotonic GBM（二选一，预注册）
  → 每专家 predicted regret + epistemic uncertainty
  → Potts/邻接一致性（只合并同图邻接区域，不跨图）
  → margin/uncertainty gate
  → 选专家或 A1 fallback
```

不允许同时搜索 MLP、Transformer、XGBoost、random forest。建议先用两层 MLP；模型参数 `<50k`。

### 4.6 R0 信息价值门（先做，CPU/cache 优先）

在实现路由器前，先用 evaluator 单独计算已有专家的区域级 Oracle：

- Oracle vs A1 mean ΔPixel-AP `>= +0.020`；
- 至少 4/6 MPDD 类 headroom `>= +0.010`；
- 至少两个非 A1 专家各贡献 `>=10%` 的被 Oracle 选择正像素；
- 去掉任一核心专家，Oracle headroom 至少下降 `0.003`；
- headroom 不能 `>50%` 来自单一类别。

若不过，RSR 直接停止。没有专家互补上限时，任何 router 都只是在过拟合。

### 4.7 R1 伪异常能力门

采用 leave-one-corruption-family-out：

- held-out family 的平均 normalized regret 相对固定 A1 至少下降 20%；
- 选中真实最佳专家的 balanced accuracy `>=0.65`；
- shuffled regret label 回到 chance，且性能差至少 `0.10`；
- A1-only router、uniform ensemble、best fixed ensemble 均被完整 RSR 超过；
- clean support 图上的误激活覆盖率 `<=5%`；
- identity fallback 单测逐位一致。

### 4.8 R2 MPDD 小门与 Full Gate

小门：seed0 × shot {1,2,4}：

- mean ΔPixel-AP vs A1 `>=+0.006`；
- 3/3 shot 为正；
- 4/6 类为正；
- worst category `>=−0.010`；
- 相对 best fixed expert/ensemble `>=+0.003`；
- 区域路由必须胜过独立像素路由 `>=+0.002`，否则“结构化区域”解释不成立。

Full：3 seeds × 3 shots：

- mean ΔPixel-AP `>=+0.008`；
- 至少 7/9 配置为正；
- category-cluster bootstrap 95% CI 下界 `>0`；
- mean ΔAUPRO `>=0`，mean ΔPixel-AUROC `>=−0.001`；
- worst category mean `>=−0.015`；
- 每个非 A1 专家的实际路由覆盖率在 `[5%,60%]`，避免专家塌缩；
- route entropy、fallback rate、错误选择案例完整报告。

### 4.9 最大风险

CASF 已证明简单伪异常分割监督很弱。RSR 只有在以下方面严格不同才值得试：监督目标改为专家 regret；决策单位改为区域；必须做 leave-family-out；必须与 A1-only/shuffled/fixed controls 比较。若 held-out family 不通过，不得把伪异常训练集加大十倍碰运气。

---

## 5. 突破口二（创新幅度最大）：NR-MoE 正常性机制混合专家

### 5.1 核心科学问题

当前专家并不是“谁都差不多”：

- CSS 救 bracket，却严重伤害 metal_plate；
- LLSE 对 tubes/metal_plate 三 seed 稳定正；
- 文本区域证据在 MPDD 有效，但在 BTAD/MVTec 失效；
- Subspace 在 5 类多为正，却重伤 connector；
- A1 的全局外观记忆是最稳的默认值。

这提示专家差异对应的是不同正常性定义：

| 专家 | 正常性的定义 | 擅长/失败倾向 |
|---|---|---|
| A1 | 局部外观是否在全局 support memory 中出现 | 稳健默认；不理解关系/上下文 |
| LLSE | patch 是否落在局部正常流形上 | 连贯、大区域偏离可能更强；reference seed 敏感 |
| CSS | patch 与同图邻域是否一致 | 边界/微小局部变化；大块异常内部会被当正常 |
| Text/TCRR evidence | 区域是否符合语义异常概念 | 类别/域依赖，外部泛化风险 |
| BC-MCR | 当前上下文能否生成正常中心 | 搬移、重复、错装；纹理不规则类可能误报 |
| Component expert | 组件集合/数量/关系是否正常 | 逻辑异常；表面纹理缺陷不一定有效 |

NR-MoE 的问题不再是“DINO 还是 CLIP 更可信”，而是：

> 仅从 K 张正常 support 能否识别这个产品/区域的 normality regime，并激活适合该机制的专家？

### 5.2 两级路由

1. **support-level prior**：根据正常 support 统计得到产品级专家先验；
2. **query-region posterior**：在先验约束下，用区域的多专家证据做小幅更新。

support descriptor 建议包括：

- token 空间重复性/位置稳定性；
- 纹理平稳度与边缘密度；
- 参考图间 feature covariance/effective rank；
- reference-to-reference retrieval purity；
- 组件数量/面积的稳定度（若可得）；
- DINO/CLIP/多层间一致性；
- shot 数与支持覆盖度。

### 5.3 为什么不能只在 MPDD 六类上学

类别级路由只有 6 个 MPDD 产品，样本量不足。NR-MoE 必须改变论文协议：用多个 source dataset/category 的真实异常结果产生“专家 regret label”，做 leave-one-category-out 或 leave-one-dataset-out 元训练；目标数据集推理时仍只输入正常 support。

推荐协议：

```text
Source meta-train: 已消费的 MPDD + MVTec AD + BTAD + VisA（按许可和 checkpoint 污染显式标记）
Intermediate confirmation: Real-IAD 的预注册类别子集
Final untouched: MVTec AD 2 private test（只提交冻结模型一次）
```

这不再是“target-normal-only training-free”论文，而是“source-supervised, target few-shot normal-conditioned generalist routing”。必须重写标题、baseline 与公平性说明。

### 5.4 R0/R1 门

先不融合 map，只预测“哪个专家在该 source category 上最优/最安全”：

- leave-one-category-out top-1 balanced accuracy `>=0.60`；
- predicted expert 的平均 regret 相对 always-A1 至少下降 25%；
- 至少 70% held-out 类不劣于 A1；
- shuffled support descriptor 性能显著下降 `>=0.10`；
- 去掉 support-level descriptor 后下降 `>=0.05`，否则“support-conditioned”不成立；
- 至少三个机制专家在不同 held-out 类被真实选择，避免退化成固定 winner。

进入像素/区域融合后，沿用 RSR Full Gate，并额外要求 leave-one-dataset-out 全部 source folds mean ΔPixel-AP 非负。

### 5.5 新意边界

MoECLIP 已在 CLIP 内做 patch expert routing，Sample-Aware Model Selection 已研究 OOD detector 选择，InCTRL 已做 source-trained in-context anomaly residual。NR-MoE 能成立的差异必须是：

- 专家代表不同 normality mechanism，而非同一网络的容量分片；
- support normality regime 作为显式条件；
- 训练目标是跨类别专家 regret；
- 输出带 fallback/risk-coverage，而非无条件 softmax 融合；
- 在 unseen dataset 的 private test 上验证。

若只实现“把几个 anomaly map 输入 MLP”，创新不足。

---

## 6. 突破口三：BC-MCR 盲中心参考条件上下文修复

### 6.1 与旧 RG-MCR/NCPRA 的关键区别

- 旧 NCPRA 看见当前中心 DINO feature，再预测当前中心 CLIP；异常信息已进入输入；
- 旧 RG-MCR diagnostic 用最近邻 ring 固定检索中心，在 missing 上会检索到背景并“填平”；
- BC-MCR 完全遮掉 query 中心，只允许邻域、相对位置和正常 support token 参与；
- 训练目标是正常 support 上的 masked center reconstruction，异常分数是 query center 与反事实正常中心的残差。

### 6.2 最小结构

```text
query 7×7 context（去掉中心 3×3）
normal support 中 top-M context tokens
relative position encoding
  → 共享投影 1536→128
  → 2-layer cross-attention（<0.8M 参数）
  → predicted fused normal center
  → cosine residual + uncertainty
```

训练只用正常 support 的 leave-region-out mask task；不能随机把相邻中心同时放在 train/validation 造成空间泄漏。

### 6.3 先做结构合成 Gate

在正常 feature grid 上生成 permutation、duplicate、missing 三类结构扰动：

- 三类 patch-AUROC 均 `>=0.75`；
- 三类相对 A1 均 `>=+0.10`；
- missing 必须相对旧 fixed retrieval 提高 `>=+0.15`；
- `CTRL-COPY`（允许中心）、`CTRL-CTX`（无 support）、`CTRL-POS`（无 query context）、`CTRL-SHUFFLE` 均不能追平；
- 完整模型相对最强控制平均 `>=+0.05`。

不过门不接触真实异常。

### 6.4 真实 MPDD Gate

- seed0 三 shot mean ΔPixel-AP `>=+0.006`；
- 至少 4/6 类正，worst `>=−0.010`；
- 完整模型分别胜过 context-only 与 support-only `>=+0.002`；
- 不能只由 permutation/duplicate 型单类贡献；
- Full MPDD 使用统一 Gate `+0.008 / 7-of-9 / CI lower>0`。

### 6.5 文献碰撞边界

FastRecon、INP-Former、ConceptADapt 都涉及 feature reconstruction/normal prototypes；BC-MCR 只有在“blind-center + cross-support context + dual-encoder counterfactual + missing-specific control”四点都被消融证明时才可能形成独立贡献。若最终模型只是普通 autoencoder 或 query adaptation，应降级为已有方法复现，不写新方法。

---

## 7. 其余不同突破口（按价值分层）

### 7.1 IC-Router：干预一致性的因果式路由（中优先，可并入 RSR）

思想：对轻微光照、颜色、翻转、缩放做可逆干预；真正缺陷证据经逆变换后应稳定，背景伪响应或插值伪影往往不稳定。与旧 MESP 的区别是：MESP 直接平均/平滑 map；IC-Router 只把干预后的**专家 regret 稳定性**作为 RSR 输入，不把一致性本身当异常分数。

验收：加入干预特征后，held-out corruption regret 至少再下降 10%，真实 MPDD 相对无干预 RSR `>=+0.002`，错位逆变换控制必须显著变差。若只产生平滑增益或 real/misaligned 相同，删除该模块。

### 7.2 AARC：Aliasing-Aware Resolution Cascade（中优先，需 GPU）

把动态决策从“选编码器”改成“何时值得高分辨率复查”：A1 448 全图为一级；support 估计局部 aliasing risk，选择少量窗口用 672/高分辨率 crop 重提 DINO/CLIP；窗口内用局部 memory，最后以边界一致的方式回填。

创新必须是 support-derived aliasing risk + compute-budget routing，而不是普通多尺度。RadioCore、VisionAD、DCP-SFR 已覆盖多尺度/多层/线索保留。

R0：先用 evaluator 做 small-defect oracle，只有高分辨率相对 A1 的 small-tier ΔPixel-AP `>=+0.03` 且 overall headroom `>=+0.008` 才实现 router。最终要求 overall `>=+0.006`、small tier `>=+0.02`、worst cat `>=−0.01`、平均 latency `<2×A1`、窗口覆盖率 `<35%`。旧 wavelet D1 失败不等价于高分辨率 token 无 headroom，但它意味着本路线不能只凭“高频可能丢失”立项。

### 7.3 Meta-RSR：真实源域异常监督的 episodic router（高创新、高协议代价）

如果伪异常无法迁移，就使用已消费数据集的真实 anomaly/mask 训练 region regret predictor，每个 episode 输入 K 张正常 support + query 专家证据；leave-one-dataset-out 训练。目标域只输入 normal support，不微调。

与 NR-MoE 的区别：NR-MoE 主攻产品级 normality regime；Meta-RSR 主攻图像/区域级 competence。两者不能同时首轮实现。

验收：至少三个 leave-one-dataset-out fold 中相对 A1 均非负，平均 `>=+0.006`；目标域类别名移除后性能不变；source dataset ID shuffled 不影响；support 打乱导致性能下降。若只在 source-seen 类有效，不具 generalist claim。

### 7.4 Object-Set Expert（独立逻辑异常论文方向）

若老师允许扩展任务到 MVTec LOCO/装配异常，可用实例/组件集合的数量、存在性和二部匹配作为独立 expert，再由 normality regime 路由外观 A1 与逻辑 expert。

但 UniVAD 和 ObjectCore 已直接覆盖 component clustering/graph/bipartite matching；本项目必须提出新的“support-conditioned expert choice”或“跨视角组件守恒”机制。SPRG 在 MPDD 的节点稳定率只有 36%，因此不得继续在 MPDD 上硬做组件图。

最低门：MVTec LOCO 4-shot image AUROC 相对复现的 ObjectCore/UniVAD 至少 `+1.0 pp` 或在相当性能下降低 30% 计算量；结构/逻辑异常分别报告；component shuffle/count controls 必须通过。此路线应另立论文协议，不与当前四数据集主表混为一谈。

### 7.5 E-Fuse：e-value/conformal selective fusion（适合第二贡献，不适合单独主创新）

把各专家的 region non-conformity 转为有限样本 p/e-value，在明确依赖假设下合并；只有证据达到阈值才偏离 A1。价值是 false-alarm/risk-coverage，而不一定提升 AP。

当前 K≤4 导致 NORC p-value 太粗，因此需要额外正常 calibration set 或 MVTec AD 2 的正常 validation，不能用 patch 数伪装成独立样本。必须按图/region block 校准并声明交换性边界。

验收：在未参与校准的 normal images 上，目标 FPR 1%/5% 的经验覆盖不超过目标 `+2 pp`；risk-coverage 曲线 AURC 相对 A1 降低 `>=10%`；Pixel-AP 非劣 `>=−0.002`。若只有单调校准、AP 不变，应定位为部署贡献。

### 7.6 CP-Metric：互补保持的各向异性联合度量（低优先，碰撞高）

在 A1 concat 空间学习 block-diagonal + low-rank cross term 的局部 Mahalanobis metric，identity 初始化；以正常紧致、伪异常边界和分支贡献不塌缩为损失。它可能比固定 concat 更有表达力。

风险：与 G²SF、PGBL、SubspaceAD 和普通 metric learning 高度相邻，本项目线性 PCA/CCA/局部密度又已失败。除非能证明“两个视觉分支 private/common 子空间的互补保持”是必要机制，否则不建议投入。

最低门：相对等参数普通 MLP metric、G²SF-style local scale、A1 均有 `>=+0.003` 增益；每分支梯度/贡献不小于 20%；shuffled cross-branch pairing 显著退化。不能仅凭比欧氏距离好就写创新。

### 7.7 REF-Diff：参考条件的 residual-evolution 反事实修复（长期、高资源）

用正常 support 条件的扩散/生成模型修复 A1 候选区，不只看单次原图—修复图 residual，而跟踪多去噪步残差是否持续。[Anomaly-Related Residual Fields, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Anomaly-Related_Residual_Fields_for_Cross-domain_Anomaly_Detection_CVPR_2026_paper.html) 指出正常随机残差会逐步被吸收，而异常残差信号可能持续；这比“一次修图相减”更有机制依据。

当前 6GB GPU、生成伪影和近邻拥挤使其不适合作为近期路线。只有获得更强资源且轻量 latent diffusion smoke 能在合成 defect 上让 residual persistence 比单步 residual AUROC 高 `>=0.10`，才进入真实数据。最终必须有 identity preservation、normal false repair、随机噪声轨迹和非扩散修复控制。

### 7.8 Topo-Head：检测—定位解耦的区域拓扑聚合（低成本副线）

A1 的图像分数直接取 map max，BTAD Image-AP/F1 边界明显。可从多阈值连通区域的面积持续性、峰值持续性、边界/内部比和跨尺度重现性构造 image score；像素图保持 A1 不变。

这不是核心像素算法，单独新意较弱，但能把“图像检测与像素定位不应共用 max”做成干净的第二模块。不得使用测试正常/异常标签拟合阈值；只能在 MPDD development 预注册并冻结。

验收：Full MPDD mean ΔImage-AP `>=+0.010`、至少 7/9 正、Image-AUROC 非劣 `>=−0.002`；冻结后新外部集同号；Pixel 指标逐位不变。若只在 BTAD 后验有效，归档。

---

## 8. 方案优先级与选择矩阵

| 路线 | 与现有资产复用 | 预期创新 | 成功概率 | 资源 | 最大风险 | 建议 |
|---|---:|---:|---:|---:|---|---|
| RSR region regret router | 高 | 中高 | 中 | 低—中 | 伪异常 competence 不迁移 | **第一优先** |
| NR-MoE normality regimes | 中 | 高 | 中低 | 高 | 协议改变、source 类别不足 | **第二优先/长期主线** |
| BC-MCR blind-center repair | 中 | 中高 | 中低 | 中 | 与 reconstruction 近邻碰撞 | **第三优先** |
| IC-Router | 高 | 中 | 中 | 中 | 退化为平滑/增强平均 | 作为 RSR 单一扩展 |
| AARC resolution cascade | 中 | 中 | 中低 | 高 | 多尺度近邻拥挤、proposal 漏检 | small-defect oracle 通过才做 |
| Meta-RSR | 中 | 高 | 中 | 高 | 不再 training-free | 老师接受新协议时做 |
| Object-Set Expert | 低 | 中高 | 中 | 中高 | UniVAD/ObjectCore 近邻强 | 另立 LOCO 论文方向 |
| E-Fuse conformal | 高 | 中/部署型 | 中 | 低 | K 太小、保证不成立 | winner 后做第二贡献 |
| CP-Metric | 中 | 低中 | 低 | 中 | G²SF/PGBL 碰撞 | 不优先 |
| REF-Diff | 低 | 高 | 低 | 很高 | 6GB GPU/伪影 | 长期储备 |
| Topo-Head | 高 | 低中 | 中 | 低 | 只改善 image metric | 可独立低成本探针 |

推荐不要同时实现超过三条。一次性广撒网已经在 v10 做过；下一轮应围绕“competence/normality regime”形成一条连贯主线。

---

## 9. 新实验协议：修复多轮搜索后的验证可信度

### 9.1 数据角色重新冻结

| 数据 | 新角色 | 允许 | 禁止 |
|---|---|---|---|
| MPDD | exhausted development | R0、候选选择、消融、失败分析 | 声称独立泛化 |
| BTAD | consumed diagnostic | 只引用 v8 等历史失败 | 再用于选新参数或称 untouched |
| MVTec AD | consumed diagnostic | 历史对照、source meta-train（若改协议） | 再称一次性最终验证 |
| VisA | source/in-domain | 历史结果或 source meta-train | 独立外部验证 |
| Real-IAD | intermediate/new（首次使用前冻结） | 预注册子集确认或 source meta-train 二选一 | 既训练又验证同类 |
| MVTec AD 2 | final untouched | 只运行唯一冻结 winner；优先 private server | public test 调参后再报 private 为一次性 |
| MVTec LOCO | logical-route only | Object/relationship 路线 | 与表面缺陷主表混均值 |

MVTec AD 2 有正常 train/validation、公开小 test 和私有 test。若作为最终验证：只能用正常 train/support 建库；public test 最多做一次兼容性 smoke且不得调参；正式结论以 private evaluation server 的单次冻结提交为准。下载和许可记录必须引用官方页并存 hash。

### 9.2 统一性能 Gate

由于已经尝试过大量路线，后续 winner 需要高于早期 `+0.003/+0.005` 的微小门，减少 winner's curse：

1. MPDD Small：mean ΔPixel-AP `>=+0.006`、3/3 shot 正、4/6 类正、worst `>=−0.010`；
2. MPDD Full：mean `>=+0.008`、7/9 配置正、bootstrap CI lower `>0`、worst cat mean `>=−0.015`；
3. 机制控制：完整方法相对最强同容量/sham control `>=+0.002`；
4. 外部：Real-IAD（若使用）和 MVTec AD 2 的 mean ΔPixel-AP 均 `>0`，平均至少 `+0.005`，任一类 `>=−0.020`；
5. 不能只提升 Pixel-AUROC；主指标仍为 Pixel-AP，共报 AUPRO/Image AP/AUROC/F1；
6. 若方法主张安全/选择性，必须额外通过风险指标，不可用 AP 代替。

### 9.3 统一机制控制

每条 trainable/router 路线至少包含：

- A1 identity；
- 所有单专家；
- best fixed weight/ensemble；
- 同容量但输入只有 A1 的模型；
- shuffled supervision/pairing；
- wrong alignment/geometry；
- 去掉 support 条件；
- 去掉 query 条件；
- 去掉 fallback；
- 参数量、FLOPs、延迟、峰值 RAM/VRAM。

### 9.4 防泄漏合同

预测模块不得读取：`labels`, `gt_masks`, anomaly type, dataset test aggregate, per-category test AP。Evaluator 可在预测落盘后读取 GT 生成离线 regret/metric；训练若使用这些离线标签，artifact 必须标记 `auxiliary_supervised_router=true` 并与推理输入物理分离。

必须测试：

1. 修改/打乱 GT 不改变 inference prediction hash；
2. sample/ref ID 缺失、重复、错位立即失败；
3. route feature 中不存在 category/dataset 泄漏；
4. freeze verify 只读，前后 hash/mtime 不变；
5. NaN/Inf、空专家、缺失文本时精确 fallback；
6. 相同 seed/config 重跑确定；
7. 所有候选和失败结果进入 ledger，不只保存 winner。

---

## 10. 交给下一位 AI 的具体执行任务

### Phase 0：只读审计与新目录（先做）

1. 阅读本文件、18/19/20 号文档、`CURRENT_DYNAMIC_FUSION_STATUS.md`、A1 `METHOD_SPEC_V2.md`。
2. 不修改 `submission_repro_20260827/`、A1 freeze、v8/v9/v10 现有证据。
3. 记录当前 `git status`、HEAD、所有相关 cache hash；现有未提交内容属于用户资产。
4. 新建独立 `innovation_v11_regret_router` 目录和 protocol/ledger；真实指标出现前冻结 R0 protocol。
5. 运行项目自有测试，正确范围为 `tests/`，不要误收集 `methods/` 第三方测试。

建议目录：

```text
configs/innovation_v11_regret_router/
src/industrial_ad/innovation_v11_regret_router/
scripts/innovation_v11_regret_router/
tests/innovation_v11_regret_router/
experiments/dynamic_fusion/innovation_v11_regret_router/
outputs/dynamic_fusion/innovation_v11_regret_router/  # gitignored 大缓存
```

### Phase 1：专家池与 Oracle 审计（第一项真正实验）

1. 统一 A1/TCRR-evidence/LLSE/CSS 的 sample ID、grid、方向和 scale；不得直接使用 v8 最终倍率。
2. 用 evaluator 在 MPDD seed0/k1 生成 region oracle、leave-one-expert-out oracle、专家贡献矩阵。
3. 通过后扩展 seed0 × k{1,2,4}；R0 不过立即停止 RSR。
4. 输出 `ORACLE_REPORT.json/csv/md`、逐类热图、专家选择覆盖和失败类别。

### Phase 2：伪异常 regret 数据与最小 router

1. 四个 corruption family，固定生成数量/强度范围/随机种子；
2. 每 family 单独保存专家损失和 regret，不先融合；
3. 先验证“不同专家确实在不同 family 上胜出”；若一个专家全赢，停止；
4. 实现唯一两层 MLP + fallback；
5. 完成 leave-family-out、shuffled label、A1-only、fixed ensemble 控制；
6. R1 通过才跑真实 MPDD。

### Phase 3：MPDD Small/Full 与唯一 winner

1. Small Gate 只允许最多两个预注册候选：`region-only` 与 `region+intervention`；
2. 未过不追加更多模型；
3. 过 Small 后冻结唯一 candidate，跑 3×3 Full；
4. Full 失败归档，不触碰新外部集；
5. Full 通过才进行文献逐公式审计和 freeze。

### Phase 4：若 RSR 失败，按证据分叉

- Oracle 足够、伪 regret 不迁移：老师接受协议变化时转 Meta-RSR/NR-MoE；
- Oracle 不足：停止 router，转 BC-MCR 创造结构专家；
- Oracle 只在 small defect 明显：先做 AARC high-resolution oracle；
- 只改善 image metric：单独做 Topo-Head，不改 pixel method；
- 所有门失败：停止算法搜索，以 A1 + 系统负结果/复现/边界论文收尾。

### Phase 5：新外部验证

1. 在下载 Real-IAD/MVTec AD 2 之前写数据角色、许可、磁盘预算和一次性规则；
2. 外部集不用于阈值、专家池、router feature、候选或 fallback 调参；
3. MVTec AD 2 只提交唯一 frozen winner；
4. 外部失败后保留结果并归档，不允许改参数后称第二次为首次验证。

---

## 11. 每阶段必须交付的机器可读证据

每个 run：

```text
<run_id>/
  protocol.json            # 结果出现前冻结
  config.json
  command.txt
  environment.json
  git_state.json
  input_manifest.json
  hashes.sha256
  predictions/             # 无 GT
  evaluation/              # evaluator-only
  per_image.csv
  per_region.csv
  per_category.csv
  metrics.json
  controls.json
  leakage_audit.json
  runtime.json
  decision.md
  marker.json
```

`marker.json` 最少字段：

```json
{
  "run_id": "...",
  "status": "passed|failed|blocked",
  "gate_passed": false,
  "paper_eligible": false,
  "dataset_role": "development|intermediate|external_private",
  "method": "...",
  "seed": 0,
  "shot": 1,
  "config_sha256": "...",
  "code_sha256": "...",
  "input_manifest_sha256": "...",
  "prediction_sha256": "...",
  "leakage_flags": {
    "test_labels_used_for_inference": false,
    "test_masks_used_for_inference": false,
    "test_aggregate_used_for_fit": false,
    "external_validation_used_for_selection": false,
    "category_identity_used_by_router": false
  }
}
```

最终必须有：

- `PORTFOLIO_LEDGER.md`：每个候选 `NOT_STARTED/RUNNING/PASS/FAIL/ARCHIVED`；
- `FINAL_DECISION.md`：只允许 `PROMOTE/ARCHIVE/BLOCKED`；
- `FAILURE_ANALYSIS.md`：失败也要解释机制与禁止继续的方向；
- `NOVELTY_AUDIT.md`：与 FastRef、ConceptADapt、MoECLIP、G²SF、InCTRL 等逐项差异；
- `FREEZE_MANIFEST.json`：仅 Full Gate 通过时创建；
- 外部验证前后配置 hash 必须完全一致。

---

## 12. 论文贡献怎样写才不会显得单薄

### 若 RSR 全部通过

可以形成三点贡献：

1. 提出极少正常参考条件下的 **region-wise expert regret estimation**，把动态融合从启发式置信度加权改成可监督、可审计的能力预测问题；
2. 提出由反事实缺陷训练、带空间结构和选择性回退的路由机制，在不暴露目标测试真值的前提下动态选择外观/语义/流形/上下文证据；
3. 在 exhausted development 与全新 private external benchmark 上，以固定专家、普通 ensemble、同容量控制和风险—覆盖分析证明增益、泛化和失败边界。

成熟专家不是问题，因为论文贡献不是重新发明 backbone，而是定义、学习并验证“专家局部能力”。

### 若 NR-MoE 通过

贡献可更强：

1. 把工业异常检测统一为多个 normality regime 的条件模型选择；
2. 用 normal support 描述产品结构，在 unseen category/dataset 上选择正常性机制；
3. 证明固定单方法无法覆盖表面、边界、结构和语义异常，并给出跨域路由证据。

但必须明确这是 source-supervised generalist protocol，不再沿用 A1 的 training-free claim。

### 若只有 BC-MCR 通过

主贡献落在“外观出现过不等于上下文正常”：blind-center cross-support reconstruction + counterfactual residual。A1 作为 appearance expert/强基线，BC-MCR 是结构 normality 模块。只有二者各自消融成立后，才测试组合。

### 若全部新路线失败

不要制造创新。A1 仍可形成一篇偏实证/工程严谨的论文：双视觉 feature fusion 的跨 seed/shot/数据集证据、严格泄漏审计、丰富失败路线、何时复杂动态机制无效的系统边界。创新评级较弱，但比把失败 router 包装成成功方法更可靠。

---

## 13. 最终停止规则

出现任一情况立即停止对应路线：

1. Oracle headroom 不足或只来自单类；
2. shuffled/wrong-alignment control 与真实机制相同；
3. 伪异常 leave-family-out 失败；
4. 动态方法不超过 best fixed ensemble；
5. 完整模型不超过同容量 A1-only 控制；
6. 需要使用 BTAD/MVTec/MVTec AD 2 调参才能通过；
7. 增益只在 seed0 或单 shot 成立；
8. worst category 越过安全线；
9. RouterInput 能访问 GT、category identity 或 test aggregate；
10. 新意与 FastRef/ConceptADapt/MoECLIP/G²SF/UniVAD 实质等价；
11. 资源超过本机可诚实复现能力且无缩小版机制 smoke；
12. 为追门槛临时增加未预注册候选。

---

## 14. 给执行 AI 的一句话任务

> 不要继续为两个成熟分支寻找另一个加权公式。先证明已有不同正常性专家在区域级仍有足够、分散且可重复的 Oracle headroom；若成立，用反事实缺陷的专家 regret 监督一个带空间结构与 A1 回退的区域路由器，并用强控制证明增益来自 competence estimation 而不是普通 ensemble。若专家池本身没有 headroom，就停止路由，转向 blind-center context repair 创造新的结构证据。任何 winner 只允许在 MPDD 开发，冻结后用尚未消费的 MVTec AD 2 private test 一次性验收。
