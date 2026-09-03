# 22. v11 结果复核与下一轮创新突破口

日期：2026-09-03  
用途：复核 21 号文档后续实验，给下一位 AI 助手提供可直接执行的候选路线、控制实验和验收门  
状态：研究决策文档；所有新名称均为工作名，未通过冻结门前不得写成论文贡献  

---

## 0. 执行摘要

21 号文档之后，两个近期首选方案已经得到明确的 R0 结果：

1. **RSR 当前专家池审计失败**。它有很大的离线 headroom：MPDD seed0/k1 上 Oracle 相对 A1 的 mean ΔPixel-AP 为 `+0.400`，6/6 类均超过 `+0.010`；但被选择的非 A1 缺陷像素中，LLSE 占 `97.2%`，text 仅 `0.6%`，CSS 仅 `2.3%`，没有形成按原门定义的平衡专家池。因此 pseudo-regret router 没有启动。
2. **BC-MCR 结构门失败**。permutation/duplicate 上 FULL、无 support 的 CTRL_CTX、旧非参数 CTRL_POS 都约为 `0.72`，说明结构上下文确实有信号，但训练式 support-conditioned 模块没有额外贡献；missing 上 FULL 仅 `0.405`，显著低于 A1 的 `0.582`。真实异常门按协议没有启动。

不过，RSR 结果中存在一个必须单独记录的**审计语义问题**：当前代码以缺陷块内 `robust01` 均值最高来选专家，却把该专家的 **raw map** 拼回输出；同时不以最终 AP 或包含假阳性的区域损失来决定专家。`bracket_black` 中去掉 CSS 后 Oracle AP 从 `0.239` 反而升到 `0.477`，已经证明这个“Oracle”不是对目标指标的真正最优上界。再加上贡献率按像素统计，被 `metal_plate` 的 35,522 个 LLSE 像素强烈支配，而 `bracket_white` 的 text 贡献只有 1 个 cell。因此：

> **可以归档当前 v11 Oracle 实现和当前专家池，不能据此把“区域级动态专家互补”这一科学假设彻底判死。**

下一轮建议不再同时铺开十条路线，而采用五级漏斗：

```text
P0 修正信息审计（不训练 router）
  ├─ 仍无宏观/小缺陷互补 → 永久关闭 RSR
  └─ 存在互补 → 只做尺度平衡的 regret router

P1 光照/成像干预切空间（NTOF）
P2 双编码器顺应功耦合（CECW，可复用缓存）
P3 扰动响应谱（PRS）
P4 跨层残差持续场（CL-RPF）

以上均失败后，再考虑多视角按需采集或概率式结构补全等协议级转向。
```

综合创新性、已有资产、6 GB GPU 约束和文献碰撞，**当前第一新主线推荐 NTOF，最快的低成本探针推荐 CECW；RSR 只允许做一次修正后的 Oracle 审计，不允许直接训练 router。**

---

## 1. 这次究竟完成了什么

### 1.1 已经完成并按当前定义关闭的部分

| 路线 | 已完成阶段 | 关键结果 | 严格结论 |
|---|---|---|---|
| RSR `{A1,text,LLSE,CSS}` | MPDD seed0/k1 区域 Oracle 审计 | mean headroom `+0.400`；LLSE 像素贡献 `97.2%`；CSS LOO drop `-0.039` | 当前 Oracle/专家池未过 g3/g4；Phase 2 不得启动 |
| BC-MCR | MPDD 正常图上的结构合成门 | perm `0.720`、dup `0.716`、missing `0.405`；CTRL_CTX 在 perm 上高于 FULL | 当前盲中心点预测、余弦残差和 missing 代理的组合关闭；不得触碰真实异常门 |
| v10 CRAM/MESP/CAPM/NORC/STR/SPRG | 各自 R0 | 全部 FAIL/ARCHIVED | 不再做同构参数搜索 |
| LLSE/CSS 探索 | MPDD 多 seed 或类别分析 | LLSE 小幅且 seed 不稳；CSS 类别翻转极大 | 可作为诊断专家，不能固定融合升级 |

### 1.2 尚未完成，不能写成“实验失败”的部分

以下是 21 号文档中的储备方案，但没有完成对应实验：

- NR-MoE：需要 source-domain episodic 元训练和老师同意改变 training-free 协议；
- Meta-RSR：同样需要真实源域异常监督；
- AARC：需要重新做高分辨率特征导出和 small-defect oracle；
- Object-Set：应转 MVTec LOCO/组件逻辑异常协议；
- REF-Diff：需要扩散模型和更高算力；
- Topo-Head：仍可做 image metric 的独立低成本探针；
- E-Fuse/CP-Metric：没有 winner 专家，或文献碰撞较高，因此未启动。

所以“多个方向都结束了”的准确表述应为：**现有资产上、保持原 normal-only few-shot 协议的高优先低成本路线已经基本穷尽；需要新观测量或新协议的路线并没有被实验否定。**

---

## 2. v11 RSR：为什么不能只看 `97.2%` 就彻底关闭动态融合

### 2.1 有效的负结论

当前 `{A1,text,LLSE,CSS}` 不适合直接训练四专家 router：

- 以像素质量计，LLSE 几乎垄断非 A1 选择；
- text/CSS 的贡献集中在极少数小块或类别；
- 一个高容量 router 很容易退化成“A1/LLSE 二选一”；
- 当前只有一个 seed、一个 shot 的 Oracle 审计，没有证明 competence 可由 normal-only 输入预测。

因此，**不要跳过 R0，直接训练 MLP/Transformer router**。这条停止规则仍然正确。

### 2.2 当前 Oracle 的三个技术缺口

#### 缺口 A：选择尺度与输出尺度不一致

代码在 `robust01(raw_map)` 上比较缺陷块均值，却把被选专家的 `raw_map` 放回最终图。选择和评价不在同一标度上。不同专家 raw score 的量纲与动态范围不同，可能造成“选中时看似强、拼回后排序反而差”。

#### 缺口 B：best-hit 不是 best-loss

当前规则只看 GT 块内部均值，没有用 21 号文档中已经定义的区域损失：

\[
L_e(r)=\mathrm{BCE}(M_r,S_e)+\alpha[1-\mathrm{AP}(M_r,S_e)]+\beta\mathrm{FP}_{e,\bar r}.
\]

它也没有真正枚举“哪个专家使最终目标指标最好”。CSS 被选后使 `bracket_black` 的最终 AP 更低，就是直接证据。

#### 缺口 C：像素加权掩盖尾部价值

按像素汇总时，大缺陷天然支配贡献率。v11 中：

- `metal_plate` 的 LLSE 被选 35,522 pixels；
- `bracket_white` 的 text 被选 1 cell，却是该类唯一非 A1 贡献；
- `bracket_black` 的 CSS 类内贡献为 59%，但全局只剩 2.3%。

若论文科学问题包含“小缺陷或尾部缺陷的召回”，应同时报告 component-macro、category-macro 和 defect-size-stratified 贡献，不能只报告 pixel-micro。

### 2.3 P0：一次性修正审计，之后必须停止或前进

工作名：**Macro-Tail Calibrated Oracle Audit，MTCOA**。它不是论文创新，只是决定 RSR 是否被误杀的审计。

执行要求：

1. 用每个专家的正常 support leave-one-out 分数拟合共同 empirical-CDF/quantile 标度；不得用 test bad 或 mask 做 calibration。
2. 选择和输出都使用同一 calibrated map；raw map 只作附加报告。
3. 对每个 GT component 枚举专家，以预注册的 `foreground ranking loss + outside-FP penalty` 选最小 loss 专家。
4. 同时报告：pixel-micro、component-macro、category-macro，以及面积 `1–4 / 5–64 / >64 cells` 三个分层。
5. Oracle 输出仍需明确是 evaluator-only capacity audit，不得被当成可部署算法。

一次性通过门：

- calibrated identity A1 与正式 A1 的 Pixel-AP 差绝对值 `<=1e-4`；
- mean Oracle ΔPixel-AP `>=+0.020`，至少 4/6 类 `>=+0.010`；
- 至少两个非 A1 专家各贡献 `>=15%` 的 **component-macro** 选择，或第二专家在 small-defect stratum 单独带来 `>=+0.020` AP；
- 去掉每个声称“核心”的专家，component-macro loss 均恶化 `>=0.003`；
- 最大单类别不贡献超过总正 headroom 的 50%；
- 结果在 seed0 shot `{1,2,4}` 的方向一致。

若不过，RSR 永久关闭。若通过，只允许训练一个 **scale-balanced regret router**，其训练 loss 对 component 和 size-bin 等权，不能再按像素数量加权。

---

## 3. 新路线一：NTOF——成像干预切空间的双编码器异常检测

全名：**Nuisance-Tangent Orthogonal Fusion**。优先级：**最高**。

### 3.1 新科学问题

A1 把所有“离正常 memory 远”的变化都当异常。但 MVTec AD 2 明确引入训练期未覆盖的光照条件；亮度梯度、曝光、色温和镜面高光会让正常 patch 远离 memory。新的问题是：

> 能否仅用正常 support 上已知成像干预产生的有限差分，估计“无害成像变化的局部切空间”，只对其正交方向上的残差报警？

对编码器 \(e\)、support patch \(z\) 和一组光照干预 \(T_j\)，构造：

\[
v_{e,j}=f_e(T_j(x))-f_e(x), \qquad U_e=\operatorname{orth}([v_{e,1},\ldots,v_{e,m}]).
\]

查询 patch 与正常近邻的差 \(d_e=q_e-n_e\) 分解为：

\[
d_e^{nuis}=U_eU_e^\top d_e,\qquad
d_e^{anom}=(I-U_eU_e^\top)d_e.
\]

异常分数主要使用 \(\lVert d_e^{anom}\rVert\)，同时把 DINO 与 CLIP-image 中共同出现的 nuisance 方向看作更可信的成像因子。A1 仍作安全回退。

### 3.2 它与已失败路线不同在哪里

- 不是 DEVA/MESP 的“增强后把多张图或热图平均”；
- 不是 SubspaceAD/LLSE 的“用正常样本总体方差拟合正常流形”；
- 不是再学一个标量动态权重；
- 它使用**带物理语义的配对干预方向**，目标是删除 nuisance 分量，而不是扩大 normal memory。

MVTec AD 2 官方说明测试图包含训练中未必出现的多种光照条件；SuperADD 在该基准上使用 intensity augmentations 和 overlapping patches 获得较强结果。这证明光照稳健性是现实缺口，但“由配对干预估计局部切空间并进行正交残差融合”仍需要本项目自己的机制验证。

### 3.3 最小实现

- 只用 5 组干预：exposure、gamma、white balance、低频左右亮度梯度、轻微 specular blob；
- 每组 3 个强度，强度范围先由正常 support 的可视性边界固定；
- DINO 和 CLIP-image 分别做 rank-`r<=4` 局部 nuisance basis；
- 第一版不训练网络，只做 closed-form QR/SVD；
- 先在 A1 已有 stride-8 grid 上实现，不先上 DINOv3。

### 3.4 R0：完全 normal-only 的机制门

用 support leave-one-out：一张正常图建 memory，另一张正常图施加 held-out illumination，再交换角色。

- 对 held-out illumination，95% 正常 patch 分数或正常图 FP95 相对 A1 降低 `>=25%`；
- 对 CutPaste、局部擦除、细划痕三类合成结构缺陷，响应保留率 `score_NTOF / score_A1 >=0.90`；
- random subspace、wrong-category tangent、打乱干预配对三种强对照均显著差于真实 tangent，差值 `>=0.05` normalized effect；
- nuisance rank 从 2/3/4 的结论稳定，不允许按真实 bad 图选 rank；
- DINO-only、CLIP-only、简单 concat projection 均为必做对照。

### 3.5 R1 与最终门

MPDD seed `{0,1,2}` × shot `{1,2,4}`：

- mean ΔPixel-AP vs A1 `>=+0.006`；
- 至少 7/9 seed-shot 为正；
- 5/6 类总体为正，worst category `>=-0.010`；
- illumination-normal FPR 改善与 real-defect AP 改善不能负相关到 `rho<-0.2`；
- simple augmentation memory、global PCA removal、SubspaceAD-style residual 都必须被超过。

只有通过后，才冻结一次性进入 MVTec AD 2。该数据集的 public test 也不得用于 rank/强度搜索。

### 3.6 最大风险

光照变化未必在深特征中形成线性局部切空间；投影可能同时删除低对比度缺陷。若合成缺陷响应保留率不过 0.90，立即停止，不用真实异常调干预范围。

---

## 4. 新路线二：CECW——双编码器“顺应正常所需做功”

全名：**Cross-Encoder Conformity Work**。优先级：**高；最快缓存探针**。

### 4.1 文献启发与差异

CVPR 2026 的 ANoCo 把异常从“离最近正常 patch 多远”改写为“把查询特征拉回锚定正常图流形需要多大更新”，通过闭式凸 Laplacian energy 得到 non-conformity cost。这个观测量比 KNN 距离更接近本项目需要的独立专家。

直接在 A1 concat feature 上复现 ANoCo 只能算强基线，不能算本项目创新。候选创新是：

1. DINO 与 CLIP-image 各自建立锚定 normal bipartite graph；
2. support-only 学一个低秩正交对齐，把两边 correction vector 投到共同坐标；
3. 联合能量既惩罚总 correction work，也惩罚两个编码器为了顺应正常而提出的**修正方向冲突**。

示意目标：

\[
E=E_D(\delta_D;N_D)+E_C(\delta_C;N_C)
 +\lambda\lVert P_D\delta_D-P_C\delta_C\rVert_2^2.
\]

最终 score 不是两个 anomaly map 的权重和，而是最小联合能量及其跨编码器矛盾项。

### 4.2 为什么值得做

- 可直接复用现有 DINO/CLIP patch cache；
- 能生成与 LLSE 不同的“图顺应做功”专家，可能重新平衡 RSR 池；
- 闭式或小型线性系统，资源可控；
- 能以强对照证明增益是否来自 coupling，而不是图平滑。

### 4.3 硬门

先实现四个基线：ANoCo-DINO、ANoCo-CLIP、ANoCo-A1-concat、两者固定均值。然后才实现 CECW。

- MPDD seed0 shot `{1,2,4}` mean ΔPixel-AP vs A1 `>=+0.006`；
- CECW 相对最强 ANoCo baseline `>=+0.003`；
- 至少 4/6 类为正，worst `>=-0.010`；
- shuffled DINO↔CLIP support correspondence 必须损失 `>=0.003`，否则 coupling 没有信息；
- query-query edge、普通 Laplacian smoothing 和只取 displacement magnitude 是必做对照；
- 与 A1 的逐 patch Spearman 若 `>0.98` 且 AP 增益 `<0.003`，判为等价重标度并归档。

### 4.4 碰撞边界

ANoCo 已经覆盖 training-free graph energy，G²SF 已经覆盖可学习局部几何融合。因此只有“**成对编码器 correction field 的联合最小功与冲突**”有独立贡献且 coupling control 通过时，才能写成创新；否则只可把公开 ANoCo 当补充 baseline。

---

## 5. 新路线三：PRS——扰动响应谱，而不是增强平均

全名：**Perturbation-Response Spectroscopy**。优先级：**中高**。

### 5.1 核心思想

UAI 2026 的 PCU 用已知强度的 perturbation ladder，把“表示对扰动的响应”本身变成不确定性信号，而不是事后查看一次扰动。PCU 当前验证主要是通用/tabular UAD，视觉工业扩展仍需重新设计。

对一组强度 \(a_0<\cdots<a_m\)，记录每个 patch 的双编码器响应：

\[
r_e(a_j)=\lVert f_e(T_{a_j}(x))-f_e(x)\rVert.
\]

从正常 support 得到 slope、curvature、跨编码器相位差和局部稳定性的条件分布。查询 patch 的响应谱若偏离 normal response envelope，则报警。

### 5.2 与 MESP/DEVA 的关键区别

- 不对增强后的 anomaly maps 求均值；
- 不声称所有增强等变；
- 明确使用强度顺序，评价一阶斜率、二阶曲率和 DINO/CLIP 响应差；
- shuffled strength order 必须破坏结果，否则所谓“谱”只是多图平滑。

### 5.3 验收门

- normal support 上，至少 80% patch 的已知强度与响应 rank correlation `>=0.6`；
- held-out perturbation family 的 response-envelope coverage 在目标 90% 时落在 `[0.85,0.95]`；
- 三种合成缺陷相对 clean 的谱偏离 AUROC `>=0.75`；
- shuffled strength、single-strength、map-average、random perturbation basis 均被超过 `>=0.05 AUROC`；
- real MPDD 门沿用 `mean ΔPixel-AP >=+0.006`、worst `>=-0.010`；
- 若真实增益只在同一 perturbation family 上出现，判为学会 corruption detector，不能作为异常检测贡献。

### 5.4 与 NTOF 的关系

两者先独立验证：NTOF 删除已知 nuisance 方向，PRS 检测异常的响应动力学。只有二者各自过门，才允许组合为“tangent removal + off-envelope response”；禁止从一开始堆叠后只报告总涨点。

---

## 6. 新路线四：CL-RPF——跨层残差持续场

全名：**Cross-Layer Residual Persistence Field**。优先级：**中；需重新导出特征**。

### 6.1 核心思想

当前 A1 只使用最终选定层的静态 patch descriptor。候选新观测量是：同一 patch 与正常 memory 的残差，沿 transformer 深度如何演化。

对层 \(l\)：

\[
d_e^l=q_e^l-\operatorname{NN}(q_e^l,N_e^l).
\]

构造：残差能量曲线、相邻层方向夹角、后半层 stationarity、DINO 与 CLIP 的 persistence agreement。假设普通纹理/光照偏差会在深层逐渐被语义吸收，而真实局部缺陷会保持非平稳或在两个编码器中呈现不同的持续模式。

CVPR 2026 的 REF 证明 diffusion reverse trajectory 中“正常变化趋于稳定、异常信号持续”是有价值的建模视角；DCP-SFR 则指出深层传播会导致 defect cue fading。这里不能照搬它们，只有“**双视觉编码器深度轴上的 normal-memory residual field**”本身产生不可由多层拼接解释的增益时才成立。

### 6.2 必做对照

- final-layer A1；
- 同参数量的 multi-layer concat；
- 各层 anomaly map mean/max；
- layer order shuffle；
- 只用能量、不用方向/非平稳统计；
- DINO-only、CLIP-only；
- RadioCore 类 multi-scale memory baseline。

### 6.3 硬门

- R0 small-defect 与 large-defect 两个 size bin 的 Oracle headroom 均 `>=+0.010`；
- persistence model 相对最佳 simple multi-layer baseline `>=+0.004 Pixel-AP`；
- layer-order shuffle 损失 `>=0.003`，否则“演化”解释不成立；
- seed0 shot `{1,2,4}` 全正，worst category `>=-0.010`；
- 峰值显存 `<5.5 GB`，特征逐层落盘，不同时驻留两个大 backbone。

### 6.4 风险

它与 DCP-SFR、REF、RadioCore 的边界都较近。若只表现为“早层分辨率更高”或“多层平均更平滑”，不得包装成 persistence 创新。

---

## 7. 新路线五：PSMF——patch-grid 相位稳定的微缺陷场

全名：**Phase-Stable Micro-defect Field**。优先级：**中低；高可行、创新风险偏高**。

### 7.1 核心思想

对输入做小于 patch/stride 的确定性平移相位，例如 `(0,0),(0,4),(4,0),(4,4)`，分别生成 anomaly map 并反变换回图像坐标。真实微缺陷应固定在图像坐标中；由 token grid 产生的 alias 响应会随相位移动或消失。

不使用简单平均，而报告：

- phase-median evidence；
- phase lower-quantile persistence；
- image-coordinate consensus 与 grid-coordinate variance；
- DINO/CLIP 的相位互补性。

### 7.2 文献边界

SuperADD 已使用 overlapping patch processing，MVTec AD 2 又包含极小缺陷。因此“多裁剪/重叠 patch”本身不新。只有 image-coordinate persistence 显著胜过 overlap average，且 phase-shuffle control 通过，才能作为 AARC 的一个机制模块，而不宜单独做主贡献。

### 7.3 验收门

- 面积最小 25% defect components 的 Pixel-AP 相对 A1 `>=+0.015`；
- 全体 mean ΔPixel-AP `>=+0.004`，worst category `>=-0.010`；
- 相对相同 forward 次数的 overlap-average `>=+0.004`；
- 错误反对齐/phase shuffle 损失 `>=0.003`；
- 计算量报告为 A1 的倍数，并给出 2-phase、4-phase 的 Pareto 曲线。

---

## 8. 新路线六：PDMC——从点预测改为条件分布的结构补全

全名：**Predictive-Distribution Masked Completion**。优先级：**中低；只有 BC-MCR 想复活时做**。

### 8.1 从 v11 学到的关键约束

BC-MCR 的失败不只是“模型不够大”：

- FULL 与 CTRL_CTX 在 permutation/duplicate 上相当，support 没有因果贡献；
- 以周围均值替换中心的 missing 代理，恰好落在上下文点预测附近，余弦残差会反向变小；
- 因此继续加 cross-attention 层数或换 loss 没有依据。

若复活，必须同时改变目标和代理：预测 \(p(z_{center}\mid context,support)\) 的**多峰分布**或 energy，而不是单个均值；missing 代理用真实背景纹理/组件删除后的 inpainting，不能用自身空间均值。

### 8.2 最小候选

- blind-context 输入不含中心；
- mixture density/conditional flow 输出中心 feature 的 NLL 与 predictive variance；
- support 通过检索到的正常同位置/同语义 patch 条件化；
- 对正常多模态纹理允许多个 mode，避免均值预测；
- realistic missing proxy 同时含背景填充、边缘断裂和组件抹除。

### 8.3 文献碰撞

PNI 已经用位置和邻域估计条件正常分布；Spatial Autoregressive DINOv3 也已显式建模 patch context。因此本路线只有在“few-shot support-conditioned、多峰 predictive uncertainty、专门验证 missing/duplicate/permutation 三机制”全部成立时才有新意。

### 8.4 复活门

- realistic missing、permutation、duplicate 三类 patch-AUROC 均 `>=0.75`；
- 每类均 `>=A1+0.10`；
- missing 相对旧 CTRL_POS `>=+0.15`；
- FULL 相对 CTRL_CTX、CTRL_POS、CTRL_COPY、CTRL_SHUFFLE 的三类平均优势 `>=+0.05`；
- support shuffle 损失 `>=0.05`，否则 support 条件化仍是装饰；
- 点预测/余弦残差 baseline 必须复现 v11 失败。

此门不过，不进入真实异常。

---

## 9. 协议级新方向：不是当前 A1 的小修小补

### 9.1 BAVA：预算约束的主动多视角检测

Real-IAD 为每个样本提供五个视角并定义 sample-level evaluation。Multi-Flow 已证明把相邻视图消息传递进 normalizing flow 有效，所以“把五视角一起融合”已经不新。

更不同的动态问题是：

> 先看一个默认视图，根据当前缺陷不确定性决定是否请求第二/第三视图，并选择哪个视图，使检测收益与采集/计算成本共同最优。

工作名 **Budgeted Active-View Anomaly detection（BAVA）**。它把最初动态融合从“给两个现成分支配权重”提升为“按预期风险下降主动获取新证据”。

验收：

- 平均使用 `<=2.5/5` 个视图；
- sample-AP 相对单默认视图 `>=+0.030`；
- 与使用全部五视图的差距 `<=0.010`；
- 随机视图、固定视图顺序、entropy-only policy 均被超过；
- 必须报告 accuracy–views–latency Pareto，而不是只报告最好点。

这是新论文协议，不能与 MPDD 单视角主表混写。近期工作已开始讨论多视角信息泄漏和不完整视图，立项前需要再做一次专门查重。

### 9.2 几何/表面法线分支

Real-IAD-MVN 在 2026 年提出五视角高保真 surface-normal 数据，动机是 RGB 易受纹理/光照影响，稀疏点云又难检测微小几何缺陷。若老师允许改变传感器输入，一个比“双成熟 RGB 分支融合”更有说服力的方向是：

- RGB 分支负责颜色、污染和纹理；
- surface-normal 分支负责凹坑、刮痕和形变；
- 动态模块只决定“外观异常/几何异常/两者一致”，而不是给两个同质 backbone 配权。

这属于新数据与多模态协议，不是当前项目的低成本延伸。它的优点是互补来源具有物理意义；缺点是数据、下载、显存和近邻工作成本都高。

### 9.3 知识约束/缺陷解释分支

ADSeeker 已把视觉文档知识库、RAG、层次 sparse prompt 和 type-level feature 用于工业异常检测与推理。因此“让大模型生成缺陷词再与 CLIP 相似度融合”已经很拥挤。

只有在项目能获得真实工艺约束，例如 BOM、允许组件数、装配顺序、容差或 CAD 属性时，才值得做“工艺规则违反 + 像素证据”的新分支。没有真实工艺知识时，不建议把 VLM 幻觉当作正常性规则。

---

## 10. 方案排序

| 顺位 | 路线 | 新观测量 | 资产复用 | 预计成本 | 创新潜力 | 当前建议 |
|---:|---|---|---|---|---|---|
| 0 | MTCOA 修正 Oracle | 校准后 component/size 宏观互补 | 极高 | 低 | 审计，不是贡献 | 必做一次；不过即永久关 RSR |
| 1 | NTOF | 已知成像干预的 nuisance tangent 与正交残差 | 中 | 中 | 高 | **第一新主线** |
| 2 | CECW | 顺应正常流形所需联合做功与修正冲突 | 高 | 低—中 | 中高 | **最快探针** |
| 3 | PRS | 特征对扰动强度的响应谱 | 中 | 中 | 高 | NTOF 之后独立验证 |
| 4 | CL-RPF | 残差沿网络深度的非平稳/持续性 | 低 | 中高 | 中高 | 需严防 DCP/REF/多层拼接碰撞 |
| 5 | PSMF | 微缺陷跨 token-grid 相位的图像坐标持续性 | 中 | 中 | 低中 | 高可行副模块 |
| 6 | PDMC | blind context 下中心 feature 的条件分布 | 低 | 高 | 中 | 仅 BC-MCR 复活路线 |
| 7 | BAVA | 每次新视图的预期风险下降 | 低 | 很高 | 高 | 独立多视角论文 |
| 8 | surface-normal | 物理几何互补 | 低 | 很高 | 高 | 老师同意换输入后考虑 |

建议只并行准备，不并行消耗数据：先 MTCOA 与 CECW；若 CECW 无信息，立即转 NTOF；NTOF 的 normal-only R0 不过则转 PRS；只有前面出现明确机制信号，才做大规模三 seed 实验。

---

## 11. 统一验收合同

### 11.1 数据角色不变

- MPDD：已耗尽 development，可做 R0、候选选择和失败分析，不能声称独立泛化；
- BTAD/MVTec AD：已用于历史诊断，不再为新方法调参；
- VisA：可作 source/meta-train，但不能同时作独立外部验证；
- Real-IAD：若用于多视角路线，应先冻结协议；
- MVTec AD 2：最终 untouched，只允许唯一冻结 winner 做一次正式验证；
- MVTec LOCO：只用于逻辑/组件路线，不与表面缺陷平均成一个主结果。

### 11.2 每条路线共同的最低性能门

除各路线专属门外，正式候选至少满足：

- 三 seed × shot `{1,2,4}` 的 mean ΔPixel-AP vs A1 `>=+0.006`；
- 至少 7/9 seed-shot 为正；
- 至少 5/6 MPDD 类的跨 seed-shot 平均为正；
- worst category `>=-0.010`；
- bootstrap 95% CI 至少报告，最终外部集要求 CI 下界 `>0`；
- Pixel-AUROC、AUPRO、Image-AP、Image-AUROC 一并报告，不能只挑上涨指标；
- 与相同 backbone、相同 shot、相同输入分辨率和近似 forward 数的 strongest control 比较。

### 11.3 每条路线共同的机制门

- 至少一个“打乱关键关系”的强对照；
- 至少一个相同算力/参数量但没有新机制的对照；
- 至少一个 identity/fallback test；
- 逐类、逐 seed、逐 shot 报告，不只给 mean；
- 新模块相对最强简单对照的独立增益 `>=+0.003 Pixel-AP`；
- 若新分数与 A1 的 Spearman `>0.98` 且无独立增益，判为重标度；
- 若去掉声称的核心输入后性能不降，核心机制声明失败。

### 11.4 搜索纪律

- R0 前冻结 `R0_PROTOCOL.json`、配置、数据列表与代码 hash；
- 每条路线最多一个机制版本 + 一个预注册 amendment；
- 不得同时搜索 MLP/Transformer/XGBoost/Random Forest；
- 不得在 bad test/mask 上拟合 normalization、threshold、rank 或融合权重；
- 未过 normal-only 机制门，不运行完整真实异常；
- 失败结果保留 `R0_RESULT.json`、`R0_DECISION.md`、`FAILURE_ANALYSIS.md`。

---

## 12. 明确禁止继续的方向

以下方向已有足够负证据或文献已高度拥挤：

1. 再调 DINO/CLIP 固定权重、温度、z-score、熵和阈值；
2. 用同一批旧的图像级统计换一个分类器继续猜专家优劣；
3. 继续扩大 TCRR 倍率或用 support 分数决定文本是否可信；
4. 在 BC-MCR 上只加层数、换 attention 或换余弦 loss；
5. 把增强图、错位图或多层 map 简单平均后称等变/持续性；
6. 把 DINOv2 换 DINOv3 或 RADIO 后直接称方法创新；
7. 继续换 wavelet/frequency 参数；
8. K≤4 时做离散 conformal p-value 并声称严格风险保证；
9. 没有稳定组件对应时继续做重型 OT/部件图；
10. 同时堆 LLSE、CSS、text、subspace、diffusion 后只看总涨点。

DINOv3 的官方材料强调其高分辨率 dense features，但 2026 工业迁移研究也报告 frozen transfer 并非在所有工业模态都天然更强。因此 backbone 升级可以是实验因素，不能替代方法问题。

---

## 13. 交给下一位 AI 助手的执行指令

> 先只读核查本文件、21 号文档、v10/v11 ledger 和所有 R0 decision。不得修改历史结果。新建 `experiments/dynamic_fusion/innovation_v12_new_observables/`。第一项只实现 MTCOA：修正 v11 Oracle 的 calibration、loss 和 component/size 汇总，冻结协议后运行 seed0 shot 1/2/4。MTCOA 不过即永久关闭 RSR；通过也不得直接跑 Full，先交付审计结果。第二项并行做 published ANoCo 的三个单分支/concat baseline，再做 CECW 的最小闭式 coupling；若 coupling 对 strongest baseline 独立增益不足 0.003 或 shuffled correspondence 不掉点，立即归档。第三项才实现 NTOF 的 normal-only illumination holdout 和 synthetic-defect preservation 门；normal shift 降幅不足 25%、synthetic response 保留不足 90% 或 random/wrong tangent 对照不输，均停止。任何路线只有通过自己的机制门，才进入 MPDD 三 seed × 三 shot。BTAD/MVTec AD 不得再调参，MVTec AD 2 不得下载或查看 public label，直到唯一 winner、配置和 hash 全部冻结。

每个实验必须交付：

```text
R0_PROTOCOL.json
R0_RESULT.json
R0_DECISION.md
FAILURE_ANALYSIS.md       # FAIL 时必须有
per_category.csv
per_seed_shot.csv
controls.json
environment.json
input_manifest.json
```

验收者只接受机器可读结果和预注册门自动生成的 PASS/FAIL；不接受“热图看起来更好”、只报均值、运行后改门或用最终测试集挑参数。

---

## 14. 论文贡献可能的落点

### 若 NTOF 成功

1. 提出由 normal-only paired imaging interventions 估计局部 nuisance tangent 的 few-shot anomaly formulation；
2. 提出双视觉编码器的 nuisance-orthogonal residual fusion，在保留缺陷方向的同时消除成像漂移；
3. 通过真实光照分布漂移、random/wrong tangent 与 defect-preservation 控制验证机制。

这比“融合两个成熟分支”更完整，因为核心贡献是**从异常距离中因果式剥离无害成像变化**。

### 若 CECW 成功

1. 把双编码器互补从分数加权改写为联合 normal-conformity work；
2. 用跨编码器 correction-direction conflict 检测单个表示中局部可接受、联合表示中不一致的异常；
3. 以 ANoCo 单分支、concat 和 shuffled correspondence 证明 coupling 的独立价值。

### 若只有 PSMF 或 Topo-Head 成功

它们更适合作为 A1 的第二模块或工程贡献，不足以单独承载“新方法主创新”。论文应诚实定位为 strong empirical baseline + alias/topology refinement，并强化系统负结果和鲁棒评估。

### 若所有新观测量仍失败

不要再做第十三轮同协议搜索。此时有三种诚实选择：

- A1 作为简洁强基线，论文重心转为跨数据集、few-shot 稳定性和系统负结果；
- 获得老师许可，转 BAVA/Real-IAD 多视角新任务；
- 获得真实工艺知识或几何模态，重新定义输入信息。

---

## 15. 一手资料与查重边界

- [MVTec AD 2 官方数据页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)：8 个新工业场景、8,000+ 高分辨率图、公开/私有测试和训练期未必出现的光照条件。
- [MVTec AD 2 论文](https://arxiv.org/abs/2503.21622)：透明/重叠对象、高正常方差、极小缺陷和 illumination shift；强方法平均 AU-PRO 仍低于 60%。
- [SuperADD](https://arxiv.org/abs/2605.14808)：DINOv3、overlapping patch、intensity augmentation 与 memory coverage 用于 MVTec AD 2 分布漂移；约束 NTOF/PSMF 必须胜过简单增强与重叠处理。
- [ANoCo, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Seo_Anomaly_as_Non-Conformity_via_Training-Free_Graph_Laplacian_Energy_Minimization_CVPR_2026_paper.html)：以锚定图 Laplacian 优化所需 feature update 作为 anomaly score；是 CECW 的直接强基线。
- [PCU, UAI 2026](https://proceedings.mlr.press/v337/allaoui26a.html)：用已知强度扰动阶梯学习表示敏感度和局部稳定性；启发 PRS，但其工业视觉扩展需独立证明。
- [Anomaly-Related Residual Fields, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Anomaly-Related_Residual_Fields_for_Cross-domain_Anomaly_Detection_CVPR_2026_paper.html)：在 diffusion 时间轴上用非平稳残差持续性区分正常变化和异常；约束 CL-RPF 不能照搬时间轨迹。
- [DCP-SFR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html)：指出深层 defect cue fading 并做结构 refinement；是 CL-RPF 的另一直接边界。
- [RadioCore, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html)：training-free multi-scale foundation feature memory；简单多层 memory 必须作为 CL-RPF 对照。
- [DINOv3 官方研究页](https://ai.meta.com/research/dinov3/)：高分辨率 dense feature 能力；只换 backbone 不构成本项目方法创新。
- [PNI, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.html)：位置/邻域条件正常分布；是 PDMC 的直接近邻。
- [Spatial Autoregressive Modeling of DINOv3 Embeddings](https://arxiv.org/abs/2603.02974)：显式建模 DINO patch 空间依赖；进一步压缩 PDMC 的新意空间。
- [Real-IAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.html)：30 类、约 150K 高分辨率图、五视角和 sample-level metric；BAVA 的数据基础。
- [Multi-Flow, CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/VAND/html/Kruse_Multi-Flow_Multi-View-Enriched_Normalizing_Flows_for_Industrial_Anomaly_Detection_CVPRW_2025_paper.html)：相邻视角 message passing；说明普通 all-view fusion 已有直接近邻。
- [Real-IAD-MVN](https://arxiv.org/abs/2605.07149)：五视角 surface-normal 数据，强调 RGB/稀疏点云对微几何缺陷的局限。
- [ADSeeker, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.html)：视觉文档知识库、Q2K RAG、层次 prompt 和 type-level feature；限制普通 VLM 文本增强路线。
- [AnomalyVFM, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Fucka_AnomalyVFM_--_Transforming_Vision_Foundation_Models_into_Zero-Shot_Anomaly_Detectors_CVPR_2026_paper.html)：多阶段合成数据与低秩 adapter；限制“换 VFM + synthetic anomaly + adapter”路线。
- [Unseen Visual Anomaly Generation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Unseen_Visual_Anomaly_Generation_CVPR_2025_paper.html)：单正常样本条件化的文本引导 unseen anomaly 生成；限制把 diffusion 伪缺陷本身写成新意。

写论文前应重新核对最终版本、公式、许可与开源状态；本文件只作为路线决策和验收合同。
