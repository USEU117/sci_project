# 多路线算法创新执行与验收任务书（A2 Innovation Program）

版本：2026-09-02  
执行对象：下一位负责算法实现、实验运行、结果审计和论文材料更新的 AI 助手  
项目目标：在保护 A1 冻结证据的前提下，系统尝试多条相互独立的算法创新路线，最终只允许一个经过 MPDD 开发、冻结后验证的方案升级为论文方法  
当前状态：**计划已制定，尚未执行；不得把本文任何候选写成已经成立的贡献。**

---

## 0. 给执行 AI 的直接指令

这不是一次“继续随便试参数”的任务。你需要实现并筛选几条回答不同科学问题的路线：

1. 正常记忆库不同区域的密度不均，是否使固定 1-NN 距离产生偏差？
2. A1 完全忽略 patch 位置，是否把产品的不同正常部件错误匹配到一起？
3. 测试图与少量参考图存在整体外观偏移时，能否只用双编码器一致认可的正常 patch 估计安全校正？
4. 能否用两种编码器共同验证哪些正常增强是真正保持语义的，从而扩充极少样本记忆？
5. 线性 CCA/RCEC 失败后，正常样本上的轻量非线性交叉预测是否仍有可利用信息？
6. 固定 Gaussian 平滑是否损害小缺陷和边界，能否用测试图自身的特征亲和关系做边界保持细化？

本任务包含六条路线，但不是六条都跑到四数据集。正确流程是：

```text
共享审计与测试
    ↓
MPDD seed0 小门：多路线低成本竞争
    ↓ 只保留通过者
MPDD 3 seeds × 3 shots 完整确认
    ↓ 只选一个最终方法（最多允许一个已证明必要的双模块组合）
冻结代码、配置和证据
    ↓
BTAD + MVTec AD + VisA 一次性验证
```

如果所有路线都失败，必须停止并归档，A1 继续作为主方法。不能根据 BTAD/MVTec/VisA 的结果回头选择第二名，也不能把多个失败模块堆叠后碰运气。

---

## 1. 开始前必须理解的事实

### 1.1 A1 是不可原地修改的冻结基线

A1 使用冻结的 DINOv2 ViT-B/14 和 AnomalyCLIP ViT-L/14@336 图像塔：patch 对齐、分支 L2、固定 0.5/0.5 concat、整体 L2、FAISS 1-NN、distance/2、Gaussian `sigma=4`、448×448 map。

权威规格：

- `submission_repro_20260827/METHOD_SPEC_V2.md`
- `experiments/dynamic_fusion/freeze/a1_mpdd_w05/REPRODUCE.md`
- `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`

所有新代码写到独立目录。禁止改写 A1 冻结 manifest、历史报告或已有特征缓存。

### 1.2 新方法必须超过 A1

主要判断是：

\[
\Delta AP = AP(\text{new method}) - AP(\text{A1 fixed concat}).
\]

只超过 DINO-only、CLIP-only 或历史 legacy score 不足以证明新模块有效。

### 1.3 数据集角色固定

| 数据集 | 角色 | 本任务允许行为 |
|---|---|---|
| MPDD | development | 允许评价候选、选公式和冻结配置 |
| BTAD | external frozen validation | 最终唯一配置一次性验证 |
| MVTec AD | external frozen validation | 最终唯一配置一次性验证 |
| VisA | in-domain frozen validation | 最终唯一配置一次性验证；必须披露 checkpoint 域内关系 |

算法层不得读取测试标签、mask 或测试集总体统计。标签只能进入 evaluator。

### 1.4 已经关闭、不得换名重做的路线

- V3.3 test-mask 校准；
- 动态 branch router、逐像素选择 DINO/CLIP；
- Route-D 可靠性预测；
- PCA/whitening、CCA、全局共享子空间；
- A2 随机投影 cross-attention、A2b CCA、A3 shared-subspace；
- RCEC v1 的条件邻域分数与同一组 `direction × k × lambda`；
- 普通 DINO KNN + PCA residual 融合；
- 单纯把 backbone 换大、单纯增加维度或重新扫 A1 权重；
- 用测试标签选择类别专属 fallback。

### 1.5 RCEC 失败带来的约束

RCEC v1 的 12 个候选三-shot平均全部低于 A1，说明“先找 DINO 邻居，再加 CLIP 条件距离”不是当前可行方向。本计划中的路线不能重新使用相同公式。尤其：

- 路线 B 使用位置约束来改变**允许匹配的正常部件**，不是增加条件 CLIP 距离；
- 路线 C 估计整体外观漂移并移动参考记忆，不做分支路由；
- 路线 E 学习正常双分支之间的非线性预测残差，必须与线性 CCA/RCEC 做明确区别。

---

## 2. 近邻工作与路线边界

执行 AI 在写论文前必须重新核验文献元数据。下表仅用于规定本计划不能怎样夸大创新：

| 近邻方向 | 已有代表工作 | 对本计划的约束 |
|---|---|---|
| 冻结 DINO patch memory | [AnomalyDINO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) | 不能把冻结特征 + KNN 本身写成新贡献 |
| query-adaptive prototype refinement | [FastRef, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html) | 路线 C 必须突出“双编码器共识 + 有界全局漂移”，不能泛称首次 query refinement |
| test-image intrinsic prototypes | [INP-Former, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_Exploring_Intrinsic_Normal_Prototypes_within_a_Single_Image_for_Universal_CVPR_2025_paper.html) | 不能直接把测试 patch 加进自己的记忆导致自匹配为零 |
| normal subspace | [SubspaceAD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html) | 不再重启 PCA/subspace 路线 |
| multi-scale foundation features | [RadioCore, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html) | 简单多层拼接创新不足，本计划不把它列为独立主路线 |
| cue/structure-aware refinement | [DCP-SFR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html) | 路线 F 只能称 training-free feature-affinity refinement，不能宣称首次结构细化 |
| spatial/neighborhood memory | [N-Pad, CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Jang_N-Pad_Neighboring_Pixel-Based_Industrial_Anomaly_Detection_CVPRW_2023_paper.html) | 路线 B 必须包含 robust deformable alignment，而不是固定同位置匹配 |
| representative normal memory | [PatchCore, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | 不能把 coreset/记忆库本身当新贡献 |

推荐的论文级新意排序：

1. **路线 D：双编码器等变性验证的正常增强**——最符合现有双分支身份，近邻重合较少；
2. **路线 B：可变形空间记忆**——科学问题明确，缓存即可验证；
3. **路线 C：跨编码器共识的有界 query shift**——可解释，但与 FastRef/INP-Former较近；
4. **路线 E：非线性交叉预测 adapter**——算法改动大，但会改变零训练定位；
5. **路线 A：局部正常密度校准**——低成本、可能有效，单独新意中等；
6. **路线 F：特征亲和图细化**——适合作为第二模块，不宜单独支撑整篇方法创新。

---

## 3. 全局实验纪律

### 3.1 候选搜索总预算

| 路线 | 预注册候选上限 | 是否需要新特征 | 首轮成本 |
|---|---:|---|---|
| A LNDC | 3 | 否 | CPU |
| B DSAM | 6 | 否 | CPU/FAISS/OpenCV |
| C CE-CQA | 4 | 否 | CPU/FAISS |
| D DEVA | 6 | 是，仅正常参考增强 | GPU batch=1 |
| E NCPRA | 4 | 可先用缓存；推荐正常增强 | 小型 GPU 训练 |
| F FAGR | 4 | 否 | CPU |

不得静默增加候选。若需要 v2，先完成当前路线失败报告和新的假设说明。

### 3.2 两级 MPDD Gate

#### Small Gate：MPDD seed0 × shot {1,2,4}

每条路线的候选至少满足：

- 三-shot mean ΔPixel-AP vs A1 `≥ +0.003`；
- 至少 2/3 shot 为正；
- worst shot delta `≥ −0.005`；
- 没有 NaN/Inf、样本错位和泄漏；
- 路线特定的 sham/control 不得与候选同样好。

没有候选通过则该路线早停。

#### Full MPDD Gate：3 seeds × 3 shots

小门胜者最多保留每条路线 1 个，完成九配置后必须满足：

- mean ΔPixel-AP vs A1 `≥ +0.005`；
- 至少 7/9 配置为正；
- worst config delta `≥ −0.010`；
- MPDD 至少 4/6 类别九配置均值不低于 A1；
- worst category mean delta `≥ −0.015`；
- mean Image-AP delta `≥ −0.005`；
- mean Image-F1-max delta `≥ −0.010`；
- 路线特定机制消融通过。

### 3.3 多路线选择防止“试到成功”

多路线属于明确的探索性开发，不进行伪显著性声明。最终选择采用字典序：

1. Full MPDD mean ΔPixel-AP 最大；
2. 若差值 `<0.001`，选正配置更多者；
3. 再平局，选 worst category 更高者；
4. 再平局，选无训练方案；
5. 再平局，选延迟和内存增加更小者。

只允许一个最终 winner。不得在验证集上比较多个 winner。

### 3.4 组合限制

只有两个模块分别通过 Full MPDD Gate 时，才允许测试它们的组合；最多测试两个组合，并且必须满足：

- 两模块逐图错误或 patch score 的相关性 `<0.90`；
- MPDD oracle combination headroom `> +0.010 Pixel-AP`；
- 组合相对较强单模块 mean ΔPixel-AP 再增加 `≥ +0.003`；
- 组合没有新增严重失败类别。

禁止组合两个单独失败的模块。

---

## 4. 共享工程结构

建议新增：

```text
configs/innovation_v2/
├── route_a_lndc.yaml
├── route_b_dsam.yaml
├── route_c_cecqa.yaml
├── route_d_deva.yaml
├── route_e_ncpra.yaml
└── route_f_fagr.yaml

src/industrial_ad/innovation_v2/
├── common.py
├── local_density.py
├── deformable_spatial_memory.py
├── consensus_query_adaptation.py
├── equivariant_augmentation.py
├── predictive_adapter.py
└── feature_graph_refinement.py

scripts/innovation_v2/
├── audit_inputs.py
├── run_small_gates.py
├── run_full_mpdd.py
├── select_winner.py
├── freeze_winner.py
├── run_frozen_validation.py
└── summarize_and_decide.py

tests/innovation_v2/
├── test_common.py
├── test_route_a_lndc.py
├── test_route_b_dsam.py
├── test_route_c_cecqa.py
├── test_route_d_deva.py
├── test_route_e_ncpra.py
└── test_route_f_fagr.py

experiments/dynamic_fusion/innovation_v2/
├── 00_input_audit/
├── 01_small_gates/
├── 02_full_mpdd/
├── 03_mechanism_ablations/
├── 04_selection/
├── 05_freeze/
├── 06_frozen_validation/
└── FINAL_DECISION.md
```

共享要求：

- 复用现有 A1 feature cache 和 evaluator；
- 新 loader 的算法视图不得暴露 `gt_sp/imgs_masks`；
- evaluator 单独加载真值；
- 每份报告记录 config/input/code SHA256；
- marker 必须绑定 config hash，不能只看文件名；
- chunked 与 non-chunked 在小数组上误差 `<1e-6`；
- `pytest tests -q` 全过；不要用无范围 pytest 收集第三方方法测试；
- 执行前保护现有未提交工作，不清理、不 reset、不改 A1 freeze。

---

## 5. 路线 A：LNDC——局部正常密度校准记忆

全称：**Local Normal-Density Calibrated Dual-Encoder Memory**。

### 5.1 科学假设

A1 使用绝对 1-NN 距离，但正常记忆的局部密度不均：高变化正常区域天然稀疏，绝对距离较大；重复纹理区域密集，轻微缺陷也可能找到近邻。用正常参考自身的局部尺度校准距离，可能减少这种偏差。

### 5.2 方法

继续使用 A1 的融合 descriptor \(z_i\)。对每个正常参考 patch：

\[
\rho_i=\operatorname{median}_{j\in\mathcal N_k^{ref}(i)}
d(z_i,z_j),\quad d(a,b)=\frac12\lVert a-b\rVert_2^2.
\]

LOO 排除：shot≥2 排除同一参考图；shot=1 排除自身和 Chebyshev 半径1邻域。

测试 patch 的相对异常分数：

\[
s_{LNDC}(q)=\operatorname{median}_{i\in\mathcal N_k(q)}
\frac{d(z_q,z_i)}{\rho_i+\epsilon}.
\]

不使用测试集均值、分位数或类别标签。

### 5.3 预注册候选

- `k ∈ {3,5,9}`；
- 聚合固定 median；
- `epsilon=1e-6`；
- 不加 lambda，不与 raw A1 混合。

### 5.4 必做消融

- A1 raw 1-NN；
- 使用全类别单一 global density 的 sham control；
- LNDC on DINO-only；
- LNDC on A1 fused descriptor。

机制 Gate：A1-LNDC 至少比 global-density sham 高 `+0.003`，且优于 DINO-LNDC，才能把收益归因于“融合正常流形的局部密度”。

### 5.5 风险和停止条件

若 k 增大只带来平滑、类别损失增加，或 global sham 与 LNDC 等价，则停止。不要扩展到 LOF/Isolation Forest 大搜索。

---

## 6. 路线 B：DSAM——可变形空间感知正常记忆

全称：**Deformable Spatially-Aware Dual-Encoder Memory**。

### 6.1 科学假设

A1 允许测试图任一 patch 匹配正常参考图任一位置。对结构化产品，这可能让缺陷位置错误匹配到另一个正常部件。固定同位置匹配又会受到拍摄平移/旋转影响。因此需要先鲁棒对齐，再做局部空间约束。

### 6.2 鲁棒对齐

对每张 query 与每张 reference：

1. 用 DINO patch 计算双向最近邻；
2. 只保留 mutual matches；
3. 使用固定随机种子的 robust estimator：
   - translation：位移向量的坐标中位数；
   - affine：OpenCV RANSAC，固定阈值和 RNG seed；
4. 计算 inlier ratio；
5. 若 mutual matches `<16` 或 inlier ratio `<0.20`，该 reference 回退为 global A1，不得读取标签决定回退。

### 6.3 空间约束评分

给定变换 \(T_r\)，测试位置 \(u_q\) 只允许匹配：

\[
\mathcal C_r(q)=\{i:\lVert u_i-T_r(u_q)\rVert_\infty\le R\}.
\]

分数仍是原始 A1 descriptor 距离：

\[
s_{DSAM}(q)=\min_{r,i\in\mathcal C_r(q)} d(z_q,z_i).
\]

这里不增加 RCEC 条件距离，也不做分支权重路由。

### 6.4 预注册候选

- alignment ∈ `{translation, affine}`；
- radius `R ∈ {2,4,8}` 个 DINO patch；
- 共 6 个候选；
- fallback 固定为 global A1。

### 6.5 必做消融

- global A1；
- 固定同位置/同半径但不对齐；
- DSAM 正确对齐；
- 打乱 reference patch 坐标；
- 只用 DINO descriptor 的 DSAM。

机制 Gate：正确对齐至少比“无对齐 local window”和“坐标打乱”各高 `+0.003`；否则不能声称 deformable spatial memory 有效。

### 6.6 诊断输出

每类报告 mutual match 数、inlier ratio、估计平移/仿射幅度、fallback 比例和候选池平均大小。不得只输出最终 AP。

---

## 7. 路线 C：CE-CQA——跨编码器共识的有界测试时校正

全称：**Cross-Encoder Consensus-Bounded Query Adaptation**。

### 7.1 科学假设

少量参考图可能与测试图存在光照、材质批次或相机造成的整体特征偏移。直接把 query patch 加入自己的 memory 会形成零距离并掩盖异常。本路线只用双分支都认为最正常的 patch 估计一个**全局、有界的 feature shift**，再移动正常参考记忆。

### 7.2 共识 pseudo-normal 选择

分别计算 query patch 的 DINO-only 和 CLIP-only 正常距离，并在**单张 query 内**计算 rank。候选集合：

\[
P_q=\{p:\operatorname{rank}_D(p)\le q,
\operatorname{rank}_C(p)\le q\}.
\]

这里 rank 只用于同一张图内部选择，不使用其他测试图统计。若 `|P_q| < 16`，回退 A1。

### 7.3 有界 shift

对每个 \(p\in P_q\)，找到 A1 最近正常 patch \(n(p)\)。分别估计：

\[
\Delta_D=\operatorname{coordinateMedian}(d_p-d_{n(p)}),
\quad
\Delta_C=\operatorname{coordinateMedian}(c_p-c_{n(p)}).
\]

shift norm 必须截断到 reference-only LOO 差值范数的 95% 分位数。对当前 query 构造：

\[
d_i'=\operatorname{norm}(d_i+\eta\Delta_D),\quad
c_i'=\operatorname{norm}(c_i+\eta\Delta_C),
\]

再按 A1 方式 concat 和 KNN。测试 patch 从不直接加入 memory。

### 7.4 预注册候选

- consensus fraction `q ∈ {0.10,0.20}`；
- shift strength `eta ∈ {0.25,0.50}`；
- 共 4 个候选。

### 7.5 必做消融

- no shift A1；
- 仅用 A1 rank 选 pseudo-normal；
- 仅用 DINO rank；
- DINO∩CLIP consensus；
- 不截断 shift 的 unsafe control；
- 将 query patch 直接加入 memory 的 leakage-like/self-match control 只用于说明风险，不作为有效候选。

机制 Gate：consensus shift 至少比 A1-only selection 高 `+0.003`，且 unsafe control 不能作为最终方法。

### 7.6 与 FastRef/INP-Former 的论文边界

只能声称这是“双编码器共识筛选的有界全局漂移校正”。不得声称首次使用 query feature 或 intrinsic normal prototype。

---

## 8. 路线 D：DEVA——双编码器等变性验证的正常增强

全称：**Dual-Encoder Equivariance-Validated Normal Memory Augmentation**。

### 8.1 科学假设

few-shot memory 缺少正常外观变化。盲目增强可能把结构改变或插值伪影当正常。本路线利用两个具有不同预训练目标的编码器共同判断：一个增强 patch 是否仍与原正常 patch 保持表征一致。只有两个分支都认可的增强特征才进入 memory。

### 8.2 只增强正常参考图

固定 transform packs：

1. `geometry`：水平/垂直平移 ±4% 和小旋转 ±5°；不使用任意大旋转；
2. `photometric`：brightness `{0.90,1.10}`、contrast `{0.90,1.10}`；
3. `combined`：geometry 与 photometric 各取一个固定组合，不做笛卡尔爆炸。

所有变换参数和随机种子写入 manifest。GPU batch=1，只重导出正常参考，不重导出测试特征。

### 8.3 等变性验证

对几何增强，将增强 feature grid inverse-warp 回原坐标；边界无效区域丢弃。对每个有效 patch：

\[
e_D=\cos(d_i,d_i^T),\quad e_C=\cos(c_i,c_i^T).
\]

仅当：

\[
\min(e_D,e_C)\ge\tau
\]

时，增强后的成对 descriptor 才进入正常 memory。原始 reference 永远保留。

### 8.4 预注册候选

- pack ∈ `{geometry, photometric, combined}`；
- threshold `tau ∈ {0.90,0.95}`；
- 共 6 个候选；
- memory search 与 A1 相同。

### 8.5 必做控制

- A1 original memory；
- unfiltered augmentation；
- DINO-only filter；
- CLIP-only filter；
- dual-encoder intersection filter；
- 在相同 memory size 下随机抽取 unfiltered patches，排除“只因 bank 变大”。

机制 Gate：dual filter 至少比 unfiltered 和单分支 filter 中的较强者高 `+0.003`；相同 bank-size control 仍需落后，才能支持“等变性验证”而不是“更多 patch”。

### 8.6 资源与完整性

- 每个增强文件记录原 reference ID、transform、valid mask、DINO/CLIP checkpoint hash；
- 检查 inverse warp 的坐标单元测试；
- 检查 identity transform 与原缓存误差；
- 报告 acceptance ratio、每类 memory 增幅和推理内存变化；
- 若 acceptance ratio `<5%` 或 `>95%`，先判定阈值失去区分力，不扩大阈值扫描。

---

## 9. 路线 E：NCPRA——正常样本非线性交叉预测残差适配器

全称：**Normal-only Cross-Encoder Predictive Residual Adapter**。

### 9.1 权限 Gate

该路线引入训练参数，会把论文从“zero-trainable-parameter”改成“lightweight normal-only adaptation”。开始前必须得到用户或导师明确同意，并在状态文档记录协议变化。未经同意，只能完成设计和合成 smoke，不能跑正式标签评估。

### 9.2 科学假设

CCA/共享子空间是全局线性关系，RCEC 是非学习的邻域关系。双编码器在正常 patch 上可能存在非线性、局部可预测关系，异常会破坏该关系。用小型 bottleneck adapter 学习双向预测，prediction residual 可作为新异常证据。

### 9.3 模型

两个预测器：

```text
g_D2C: 768 → r → GELU → 768
g_C2D: 768 → r → GELU → 768
```

只训练 adapter，backbone 冻结。损失：

\[
L=1-\cos(g_{D2C}(d),c)
+1-\cos(g_{C2D}(c),d)
+\mu L_{equivariance}.
\]

异常残差：

\[
e(q)=\frac12[1-\cos(g_{D2C}(d_q),c_q)
+1-\cos(g_{C2D}(c_q),d_q)].
\]

使用 reference-only LOO 统计校准后与 A1 组合。不得使用异常训练图、mask 或测试统计选 epoch。

### 9.4 正常验证与训练规则

- shot≥2：leave-one-reference-image-out normal validation；
- shot=1：固定增强 view 作为 normal validation，原图训练；
- epochs、optimizer、学习率和 early-stop rule 在首个正式运行前写死；
- early stopping 只看 normal prediction loss；
- 每个 seed/shot/category 独立训练，记录 parameter count 和训练时长。

建议固定初值：AdamW、lr `1e-3`、weight decay `1e-4`、最多 100 epochs、patience 10。若合成 smoke 发现不收敛，允许在正式标签评估前修复数值问题一次，并记录变更。

### 9.5 预注册候选

- bottleneck `r ∈ {32,64}`；
- residual weight `lambda ∈ {0.10,0.25}`；
- 共 4 个候选；
- `mu` 固定 0.1，不扫描。

### 9.6 必做消融

- A1；
- 线性 ridge D→C/C→D；
- 历史 CCA 结果；
- nonlinear adapter without equivariance；
- full NCPRA；
- branch-shuffled pairs（只做机制消融）。

机制 Gate：full NCPRA 至少比 linear ridge 和 no-equivariance 中较强者高 `+0.003`，且 shuffled pairing 必须明显下降。

### 9.7 额外验收

- trainable parameters `<0.5M`；
- 单类别训练峰值 VRAM `<4GB`；
- 不允许 adapter 性能依赖测试标签选择 checkpoint；
- 如果成功，必须重新测训练成本、推理延迟和 A1 零训练公平性，不得沿用旧效率声明。

---

## 10. 路线 F：FAGR——特征亲和图边界保持细化

全称：**Feature-Affinity Graph Refinement**。

### 10.1 科学假设

A1 用固定 Gaussian `sigma=4` 平滑所有区域，可能跨越结构边界传播高分或稀释小缺陷。DINO query patch 之间的特征亲和可以指导只在相似局部传播分数。

### 10.2 方法

在 DINO patch grid 上建立 4-neighbor 图：

\[
w_{pq}=\exp\left(\frac{\cos(d_p,d_q)-1}{\tau}\right).
\]

从原始 A1 patch score \(s^0\) 开始做固定次数 Jacobi 更新：

\[
s_p^{t+1}=\frac{s_p^0+\mu\sum_{q\in N(p)}w_{pq}s_q^t}
{1+\mu\sum_{q\in N(p)}w_{pq}}.
\]

最后双线性上采样；不再叠加 `sigma=4` Gaussian，避免两次平滑。

### 10.3 预注册候选

- `mu ∈ {0.10,0.50}`；
- iterations ∈ `{1,3}`；
- `tau=0.10` 固定；
- 共 4 个候选。

### 10.4 必做控制

- 原 Gaussian A1；
- 无 Gaussian 的 bilinear A1；
- 相同迭代次数但所有边权=1的 uniform smoothing；
- DINO feature-affinity；
- 可选 CLIP affinity 仅作消融，不扩大候选。

机制 Gate：FAGR 相对 uniform smoothing Pixel-AP 或 Pixel-AUPRO 至少提高 `+0.003`，同时 Pixel-AP 相对 A1 不得下降。如果只改善边界可视化而统一指标不变，不升级为主模块。

### 10.5 定位

FAGR 更适合与通过 Full Gate 的主记忆模块组合。它单独通过也不能自动成为整篇论文唯一创新，除非跨数据集 Pixel-AP/AUPRO 证据非常稳定。

---

## 11. 推荐执行顺序

### Wave 0：共享审计与框架

1. 运行 `git status --short` 并保存；
2. 验证 A1 map/metrics 回归；
3. 建立统一算法输入对象，不含标签；
4. 建立统一六指标 evaluator；
5. 建立 route/candidate/config hash schema；
6. 添加防止访问验证数据集的 guard；
7. 完成所有共享单元测试。

### Wave 1：缓存可完成的不同假设

按顺序实现 A、B、C、F：

1. 路线 A LNDC；
2. 路线 B DSAM；
3. 路线 C CE-CQA；
4. 路线 F FAGR。

每条路线先合成 smoke，再 MPDD s0/k1 单类别，再运行正式 Small Gate。单路线失败立即归档，但不阻止其他路线，因为它们假设不同。

### Wave 2：正常增强

执行路线 D DEVA。先只用 MPDD s0/k1 的两个类别验证增强导出、inverse warp 和 memory 构建，再运行完整小门。不得一次生成四数据集增强。

### Wave 3：轻量训练路线

仅在以下条件同时满足时执行路线 E：

- Wave 1/2 没有明显 Full Gate winner，或 NCPRA 在创新性上明显更有价值；
- 用户/导师明确接受论文从零训练改为轻量 normal-only adaptation；
- adapter 合成和 normal-only validation smoke 全部通过。

### Wave 4：Full MPDD 与单一 winner

1. 每条路线最多一个小门胜者进入九配置；
2. 完成机制消融；
3. 生成统一比较表；
4. 按第 3.3 节自动选择 winner；
5. 若允许组合，最多测试两个符合第 3.4 节条件的组合；
6. 生成 `MPDD_SELECTION_DECISION.md/json`。

### Wave 5：冻结与一次性验证

冻结后只运行唯一 winner：

- BTAD 9 配置；
- MVTec 9 配置；
- VisA 9 配置。

若真实缓存覆盖变化，运行前报告。不得缺几个就平均剩余项。

---

## 12. 最终冻结验证 Gate

最终方法升级为论文主方法必须满足：

1. MPDD Small/Full/Mechanism Gates 全通过；
2. 三个验证数据集 mean ΔPixel-AP vs A1 的平均值 `>0`；
3. 至少 2/3 验证数据集平均 Pixel-AP 高于 A1；
4. 任一验证数据集 mean delta `≥ −0.005`；
5. 27 个验证配置至少 18 个为正；
6. 任一验证类别九配置均值不低于 A1 `−0.030`；
7. Image-AP/F1 不出现新的系统性恶化；
8. 六指标、资源、泄漏、输入哈希和重算审计完整；
9. 相对新增复杂度有足够收益；
10. 论文近邻差异能够用一段话清楚说明且不使用“首次”夸大词。

如果冻结验证失败：

- 不验证第二名；
- 不回 MPDD 改参数；
- 将 winner 归档为 development overfit；
- A1 保持论文主线。

---

## 13. 每份报告的最低 schema

```json
{
  "schema_version": 1,
  "program": "innovation_v2",
  "route": "B_DSAM",
  "candidate_id": "affine_r4",
  "dataset": "mpdd",
  "dataset_role": "development",
  "seed": 0,
  "shot": 1,
  "config_sha256": "...",
  "code_sha256": "...",
  "input_manifest_sha256": "...",
  "input_cache_manifest_sha256": "...",
  "leakage_flags": {
    "test_labels_used_by_method": false,
    "test_masks_used_by_method": false,
    "test_distribution_used_for_calibration": false,
    "validation_dataset_used_for_tuning": false,
    "category_specific_test_rules_used": false
  },
  "metrics": {
    "new_method": {},
    "a1": {},
    "dino": {},
    "delta_vs_a1": {}
  },
  "per_category": [],
  "mechanism_diagnostics": {},
  "runtime": {},
  "checks": {}
}
```

指标必须包含：Image-AUROC、Image-AP、Image-F1-max、Pixel-AUROC、Pixel-AP、Pixel-AUPRO@0.30。

---

## 14. 单元测试最低要求

共享：

- A1 numerical regression；
- sample ID alignment、非方形 grid；
- 算法对象拒绝 label/mask 字段；
- config hash/marker 防误复用；
- chunked search 等价；
- frozen runner 拒绝参数覆盖；
- 验证数据集 guard；
- deterministic rerun。

路线特定：

- A：LOO density 不含自身/同图，手算 ratio；
- B：已知 translation/affine 能恢复，低 inlier 正确 fallback；
- C：共识交集、shift cap、空集合 fallback，绝不自匹配；
- D：identity transform、inverse warp、valid mask、双分支接受逻辑；
- E：backbone 无梯度、normal-only split、checkpoint 不读测试指标；
- F：uniform 权重退化为普通平滑、边权非负、迭代稳定。

正确项目测试命令：

```powershell
.\.venv-patchcore\Scripts\python.exe -m pytest tests -q
```

---

## 15. 最终交付物

无论成功或失败，必须交付：

1. 六条路线的代码或明确早停原因；
2. 每条实际运行路线的 pre-registered config；
3. 所有候选逐配置报告，不只保留 winner；
4. 每条路线的 `SMALL_GATE_DECISION.md/json`；
5. 通过路线的 Full MPDD 和机制消融；
6. 自动 winner selection 报告；
7. 若有 winner：独立 freeze manifest、method spec、reproduce 命令；
8. 一次性冻结验证及最终 `PROMOTE/ARCHIVE` 决策；
9. 英文 Method 草稿、伪代码、复杂度、消融表；
10. 更新主张—证据矩阵和导师中文总览；
11. 若全部失败：按路线总结为什么失败和不应继续什么；
12. 不修改/删除 A1、RCEC 和历史负结果证据。

---

## 16. 推荐的最终论文结构（仅成功后）

如果无训练路线成功：

```text
Frozen dual-encoder representation
        +
one validated normal-memory innovation (A/B/C/D)
        +
optional FAGR refinement if independently necessary
```

如果 NCPRA 成功：

```text
Frozen dual encoders
        +
lightweight normal-only cross-encoder adapter
        +
normal memory scoring
```

论文贡献最多写 3 点：一个核心机制、一个严格 few-shot/frozen protocol、一个跨数据集结果与失败边界。不要把每个工程步骤都列成创新。

---

## 17. 最重要的停止规则

1. 任何路线必须超过 A1，不只超过 DINO。
2. 所有候选只在 MPDD 竞争；验证集只看唯一 frozen winner。
3. 小门失败的路线不进入 full matrix。
4. 两个失败模块不能组合。
5. 机制 control 不支持解释时，即使单点 AP 高也不能按该机制写论文。
6. 不增加候选数量来追正结果。
7. 不按类别手写规则，不用测试正常图统计，不碰测试 mask。
8. 只有路线 E 经明确授权后可训练；其他路线保持 training-free。
9. 成功后重新测完整六指标、效率、内存和公平性。
10. 全部失败时停止，保留 A1，不因“需要创新”而接受不可复现或泄漏方法。

---

## 18. 给下一位 AI 的开工清单

开始执行时，先完成以下动作并汇报：

- [ ] 阅读本任务书全文；
- [ ] 阅读 A1 `METHOD_SPEC_V2.md`、当前状态和 RCEC 最终决策；
- [ ] 运行 `git status --short`，列出需要保护的现有修改；
- [ ] 建立 `innovation_v2` 独立目录；
- [ ] 写 route registry 和禁止访问验证集的 guard；
- [ ] 复现一个 MPDD s0/k1 A1 配置；
- [ ] 先实现路线 A 和 B 的纯函数与单元测试；
- [ ] 不要在没有 smoke 的情况下直接运行整矩阵；
- [ ] 每完成一条路线立即生成 Gate 决策；
- [ ] 任何冻结验证前先停下来检查只有一个 winner。

执行完成的定义不是“所有路线都跑完”，而是每条路线都按预算得到可信的继续/停止决定，并且整个过程中 BTAD、MVTec 和 VisA 没有变成隐藏开发集。

