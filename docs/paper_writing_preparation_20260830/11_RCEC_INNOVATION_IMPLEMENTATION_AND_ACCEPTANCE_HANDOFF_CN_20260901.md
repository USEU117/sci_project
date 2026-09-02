# RCEC 创新方法实现与验收任务书（交给下一位 AI）

版本：2026-09-01  
适用项目：Few-shot Industrial Anomaly Detection — Dual-Encoder Patch Fusion  
执行对象：接手代码实现、实验运行、结果审计和论文证据整理的 AI 助手  
任务性质：在不破坏 A1 冻结证据的前提下，开发一个真正利用双编码器关系的新方法候选  

> **执行状态（2026-09-02 独立复核）：`ARCHIVE`。** RCEC v1 工程实现和 MPDD Phase 2 小门已完成；12 个候选均未通过，按本任务书预设规则早停。没有运行 Phase 3—6，没有访问 BTAD/MVTec/VisA 做 RCEC 调参或验证，A1 继续作为论文主方法。详见本文第 15 节和 `experiments/dynamic_fusion/rcec_v1/FINAL_RCEC_DECISION.md`。

---

## 0. 给执行 AI 的直接指令

你要完成的不是“随便再试几个融合权重”，而是把当前的简单双分支拼接方法扩展成一个可解释、无测试泄漏、可以被严格验收的新方法，并留下完整的代码、测试、配置、逐次实验结果、冻结记录和失败边界。

本任务的首选方法暂命名为：

> **RCEC: Reference-Conditioned Cross-Encoder Consistency**  
> 参考样本条件下的跨编码器一致性检测

RCEC 的主要创新不是更换 DINOv2 或 AnomalyCLIP，而是显式利用正常记忆库中 DINO patch 与 CLIP-image patch 的**成对对应关系**：DINO 找到正常结构邻居后，再检查 CLIP 语义是否认可同一批邻居。两个编码器若对正常邻域产生异常分歧，该分歧本身应成为异常证据。

你必须遵守以下工作顺序：

1. 只读审计当前 A1 和已有缓存，确认输入可用；
2. 新建独立的 RCEC 源码、配置、测试和实验目录，不修改 A1 冻结文件；
3. 先在合成数据和一个 MPDD 小配置上验证算法正确性；
4. 只在 MPDD 上完成候选选择；
5. 只有 MPDD Gate 通过后才能冻结唯一配置；
6. 冻结后一次性运行 BTAD、MVTec AD、VisA，禁止根据三者结果改方法；
7. 通过最终 Gate 才允许把 RCEC 升级为论文主方法；否则保留为负结果，论文继续使用 A1。

不要为了“必须成功”而放宽标准、删除失败配置、临时更换指标或在验证数据集上调参。科研上得到可信的否定结论，也比产生不可投稿的正结果更有价值。

---

## 1. 开始前必须理解的项目事实

### 1.1 当前 A1 冻结方法

当前论文基线 `A1 fixed concat` 使用：

- DINOv2 ViT-B/14，448 输入，约 32×32 patch grid，每 patch 768 维；
- AnomalyCLIP ViT-L/14@336 图像塔，实际 518 输入，约 37×37 grid，每 patch 768 维；
- 将 CLIP grid 双线性插值到 DINO grid；
- 两个分支分别 L2 normalize；
- 固定 0.5/0.5 加权 concat，得到 1536 维，再整体 L2 normalize；
- 正常参考 patch 建 FAISS `IndexFlatL2`；
- KNN `k=1`，平方 L2 距离除以 2；
- patch map 经过 `sigma=4` 高斯平滑并上采样到 448×448；
- 当前图像分数为完整异常图最大值。

权威方法规格：

- `submission_repro_20260827/METHOD_SPEC_V2.md`
- `experiments/dynamic_fusion/freeze/a1_mpdd_w05/REPRODUCE.md`
- `docs/paper_writing_preparation_20260830/10_PROJECT_STATUS_FOR_SUPERVISOR_CN_20260901.md`

历史 `METHOD_CARD.md` 中的 1152 维是已知旧错误，不得沿用；正确值为 `768+768=1536`。

### 1.2 当前可靠的主结果

A1 相对 matched feature-DINO-only 的九配置平均 Pixel-AP 为：

| 数据集 | DINO-only | A1 | A1 − DINO | 配置方向 |
|---|---:|---:|---:|---|
| MPDD | 0.3304 | 0.3562 | +0.0258 | 9/9 为正 |
| BTAD | 0.6206 | 0.6455 | +0.0249 | 9/9 为正 |
| VisA | 0.3201 | 0.3725 | +0.0524 | 9/9 为正 |
| MVTec AD | 0.5226 | 0.5546 | +0.0320 | 9/9 为正 |

RCEC 的主要比较对象必须是 **A1**，不能只与 DINO-only 比。因为 RCEC 的目标是证明“跨分支关系建模”优于“两个现成分支直接拼接”。

### 1.3 数据集角色，不得重新定义

| 数据集 | 本任务角色 | 允许行为 |
|---|---|---|
| MPDD | development | 允许使用标签计算候选指标、选择超参数和确定最终公式 |
| BTAD | external frozen validation | 仅允许冻结后一次性验证，不能调参 |
| MVTec AD | external frozen validation | 仅允许冻结后一次性验证，不能调参 |
| VisA | in-domain frozen validation | 仅允许冻结后一次性验证，必须标注 checkpoint 域内关系 |

测试标签和 mask 在所有数据集上都只能被 evaluator 读取。算法构建记忆库、估计分布、计算可靠性或生成 anomaly map 时，均不得读取 `gt_sp`、`imgs_masks` 或由它们产生的统计量。

### 1.4 已经失败或被关闭的方向

禁止把下列旧方案换名后重新包装为创新：

- 逐图或逐像素动态路由：旧实验仅约 +0.0009，稳定性不足；
- Route-D：无标签可预测性 Gate 未通过；
- V3.3：曾使用测试 mask 校准，属于泄漏，不可进入论文主证据；
- PCA、whitening、CCA、全局共享子空间、普通残差 KNN：已有负结果；
- 仅增加更多 backbone、简单多尺度拼接、扫描大量融合权重；
- 使用 BTAD、MVTec 或 VisA 结果反向决定参数。

RCEC 如果最终退化成“根据测试图动态选择 DINO 或 CLIP”，视为偏离任务。

---

## 2. 本任务要回答的科研问题

必须围绕下列问题设计、实现和报告实验：

> 在只有少量正常参考图、没有异常训练数据的条件下，DINO 与 CLIP-image 对同一个正常 patch 邻域的一致或不一致，能否提供超出固定特征拼接的异常证据？

需要验证的三个具体假设：

1. **H1：成对关系有效。** DINO 检索到的正常邻居如果在 CLIP 空间中也与查询 patch 相似，则该查询更可能正常；反之，跨分支分歧可能提示异常。
2. **H2：这种信息不是维度增加造成的。** 复制 DINO 到 1536 维或使用距离保持的维度扩展，不应产生 RCEC 的收益。
3. **H3：像素定位和图像判别需要不同聚合。** 融合图对 Pixel-AP 有利，不代表最大值池化一定适合 Image-AP/F1；可在主模块稳定后单独验证任务解耦评分。

论文方法的最低成立条件是 H1 通过。H2 是必须完成的控制证据。H3 是可独立 Gate 的次级创新，不能拿它掩盖 H1 失败。

---

## 3. 创新一：RCEC 成对正常记忆与条件一致性评分

### 3.1 输入与符号

对一个类别、一个 seed/shot 配置，CLIP grid 对齐到 DINO grid 后，正常参考记忆表示为：

\[
\mathcal{M}=\{(d_i,c_i,m_i)\}_{i=1}^{N},
\]

其中：

- \(d_i\in\mathbb{R}^{768}\)：L2 归一化后的 DINO patch；
- \(c_i\in\mathbb{R}^{768}\)：L2 归一化后的 CLIP-image patch；
- \(m_i\)：必须包含 reference image ID、patch row、patch column；
- `(d_i, c_i)` 必须来自同一张参考图、同一个对齐后 patch 位置。

查询 patch 表示为 \((d_q,c_q)\)。跨分支对齐必须通过现有 `sample_ids` alignment plan 完成，不能假定两个缓存天然同序。

现有 `.npz` 中，测试特征有 `sample_ids`，但 `ref_patch_features` 没有单独保存 reference IDs。参考身份必须按 manifest 中
`categories[category][seed][shot]` 的列表顺序重建：数组第 0 维是 reference index，随后两维是 patch row/column。现有 DINO 与
CLIP 导出器都按该 manifest 列表顺序 append reference block，但执行 AI 仍须审计导出源码，并把 manifest SHA256、列表内容和
`ref_patch_features.shape[0] == shot` 写入报告。如果任一旧缓存无法证明参考顺序一致，不得凭数组位置猜测；应生成经过哈希的
reference-ID sidecar，或只重新导出缺失身份信息的对应缓存。

### 3.2 基本分数

DINO 正常距离：

\[
s_D(q)=\min_i \frac{1}{2}\lVert d_q-d_i\rVert_2^2.
\]

A1 固定拼接分数：

\[
z_q=\operatorname{norm}([0.5d_q;0.5c_q]),
\quad
s_{A1}(q)=\min_i\frac{1}{2}\lVert z_q-z_i\rVert_2^2.
\]

以上两个分数必须与当前冻结实现数值对齐，用于回归测试。

### 3.3 DINO→CLIP 条件一致性

首先只在 DINO 记忆库中找 Top-k 正常邻居：

\[
\mathcal{N}_D^k(q)=\operatorname{TopKNN}(d_q,\{d_i\}).
\]

然后只在这些 DINO 已认可的成对邻居中计算 CLIP 距离：

\[
r_{C|D}(q)=
\min_{i\in\mathcal{N}_D^k(q)}
\frac{1}{2}\lVert c_q-c_i\rVert_2^2.
\]

它回答的问题是：

> “DINO 认为最像正常结构的这些 patch，在 CLIP 语义空间中是否也像当前查询？”

不要把 `r_C|D` 错写成 CLIP 在整个库中的普通最近邻距离。后者只是 CLIP-only，不能表达跨编码器条件关系。

### 3.4 可选的对称项

只有在单向版本完成后，才允许测试对称项：

\[
r_{D|C}(q)=
\min_{i\in\mathcal{N}_C^k(q)}
\frac{1}{2}\lVert d_q-d_i\rVert_2^2.
\]

对称版本的条件分数可定义为：

\[
r_{sym}(q)=\frac{r_{C|D}(q)+r_{D|C}(q)}{2}.
\]

单向 `DINO→CLIP` 是首选 MVP，因为项目证据表明 DINO 是较强主分支，CLIP-image 是较弱但可能互补的分支。对称版本只能作为预注册候选之一，不能看完验证集后再追加。

### 3.5 只用正常参考的校准

`s_A1` 与 `r_C|D` 数值范围可能不同，必须只用正常参考 patch 做稳健校准。禁止使用测试图整体分位数、均值或标准差。

对每个正常参考 patch 执行 reference-only leave-one-out（LOO）评分，得到：

\[
\mathcal{S}_{A1}^{ref},\quad \mathcal{R}_{C|D}^{ref}.
\]

LOO 排除规则固定如下：

- shot≥2：对来自参考图 \(I\) 的 patch，优先从其他参考图检索，即 leave-one-reference-image-out；
- shot=1：排除 patch 自身，同时排除同一图中 Chebyshev 空间半径 1 的邻近 patch，避免自匹配和紧邻位置形成虚假的零距离；
- 如果排除后候选数少于 k，该配置必须报错，不能静默退回包含自身的搜索。

对任意参考分数集合 \(X^{ref}\)，使用：

\[
\operatorname{rz}(x;X^{ref})=
\frac{x-\operatorname{median}(X^{ref})}
{1.4826\cdot\operatorname{MAD}(X^{ref})+\epsilon}.
\]

建议固定：

- `epsilon = 1e-6`；
- robust z 截断到 `[-5, 10]`，防止小 MAD 放大极端值；
- 如果 MAD < `1e-6`，配置判为数值退化并记录，不能使用测试统计补救。

### 3.6 候选最终公式

首选公式：

\[
S_{RCEC}(q)=
(1-\lambda)\operatorname{rz}(s_{A1}(q))+
\lambda\operatorname{rz}(r_{C|D}(q)).
\]

对称候选仅将第二项替换为 `r_sym`。

最终 patch score 仍按现有流程生成 anomaly map：reshape 到 DINO grid、Gaussian `sigma=4`、双线性上采样到 448×448。除非进入创新二，图像分数仍使用 map max，保证第一阶段只检验跨编码器一致性本身。

### 3.7 严格限制候选数量

MPDD 上最多允许评估以下 12 个预注册 RCEC 候选：

- direction ∈ `{dino_to_clip, symmetric}`；
- k ∈ `{1, 3, 5}`；
- λ ∈ `{0.25, 0.50}`。

固定项：

- LOO 排除规则不扫描；
- epsilon、z clip、sigma、map size 不扫描；
- backbone、输入尺寸、特征层、reference split 不改变；
- 不引入 PCA/CCA/whitening；
- 不扫描类别专属参数。

如 12 个候选全部失败，RCEC v1 判定失败。若要提出 v2，必须先写独立的失败机理报告和新的科学假设，不能临时增加几十个超参数。

---

## 4. 创新二：检测—定位解耦评分（RCEC-D，可选）

这一模块只有在 RCEC 主模块通过 MPDD Gate 后才能开始。它的目的不是改变 pixel map，而是解决“Pixel-AP 改善但图像级最大值池化不稳定”的现有边界。

### 4.1 原则

- RCEC pixel map 保持完全不变；
- 只改变一张图如何从 patch/map score 聚合为 image score；
- 参数仍只在 MPDD 选择；
- 不允许专门针对已知 BTAD 结果选择规则。

### 4.2 预注册候选

基于 patch score，而不是插值后重复像素，评估：

1. `max`：当前基线；
2. `top_0.1pct_mean`；
3. `top_1pct_mean`；
4. `top_5pct_mean`。

如果有效 patch 数导致 0.1% 少于一个 patch，至少取 1 个。Top-q 的实现必须对每张图独立，不得使用整套测试图分布。

可选的保守双头仅允许一个固定形式：

\[
I(q)=0.5\,\operatorname{TopQ}(S_{RCEC})+
0.5\,\operatorname{TopQ}(S_D^{cal}),
\]

其中两个 patch score 必须先用各自 reference-only LOO 统计做 robust z。禁止再扫描 0.5 权重。

### 4.3 单独验收

创新二只有同时满足以下 MPDD 条件才可冻结：

- 九配置平均 Image-AP 相对 RCEC-max 提高至少 `+0.005`；
- 至少 6/9 配置 Image-AP 提高；
- 平均 Image-AUROC 不低于 `−0.002`；
- 平均 Image-F1-max 不低于 `−0.005`；
- Pixel 指标逐文件哈希或数值完全不变。

未通过则删除论文方法声明，但保留实验报告。不得因为 BTAD 的 Image-AP 已知下降而在 BTAD 上选 q。

---

## 5. 必做控制实验与消融

### 5.1 维度匹配控制

构造：

\[
z_{dup}=\operatorname{norm}([0.5d;0.5d]).
\]

该表示与原始 DINO 的成对距离理论上相同：

\[
\lVert z_{dup}(x)-z_{dup}(y)\rVert_2^2
=\lVert d(x)-d(y)\rVert_2^2.
\]

必须同时提供：

- 一个单元测试证明随机归一化向量的距离误差 `<1e-6`；
- 一个 MPDD 配置实测，DINO 与 DINO-duplicate 的 map 最大绝对误差 `<1e-5`，Pixel-AP 误差 `<1e-6`。

这项控制用于排除“A1/RCEC 只是因为特征从 768 维变成 1536 维”的解释。

### 5.2 配对破坏控制

仅用于最终选定 RCEC 配置的消融，不参与选参：

- 在固定随机种子下，打乱正常记忆中 CLIP patch 与 DINO patch 的配对；
- 保持每个分支的边际特征集合、数量和维度不变；
- 使用相同的 frozen 参数重新评估 MPDD 九配置；
- shuffle 种子至少 `{0,1,2}`，报告均值和方差。

预期是正确配对优于 shuffled pairing。如果打乱后不降或更好，说明“成对一致性”解释不成立；即使 RCEC 数字较高，也不能按当前创新叙事投稿，必须分析是否只是附加 CLIP 距离产生的普通集成收益。

最低解释性要求：正确配对的平均 Pixel-AP 必须高于三次 shuffle 平均，且差值至少 `+0.003`。该门槛是解释性 Gate，不替代主性能 Gate。

### 5.3 必须保留的比较方法

每个最终结果表至少包含：

- matched DINO-only；
- CLIP-image-only（已有数据集尽量复用现有报告）；
- A1 fixed concat；
- RCEC；
- RCEC shuffled-pairing（消融）；
- DINO duplicate 1536-D control。

不要只报告 RCEC 与 DINO-only 的差值。

---

## 6. 工程实现要求

### 6.1 不得修改的内容

以下内容视为历史/冻结证据，默认只读：

- `submission_repro_20260827/`
- `experiments/dynamic_fusion/freeze/a1_mpdd_w05/`
- 当前 A1 的既有逐配置报告；
- 现有 feature cache `.npz`；
- `scripts/evaluate_a1_feature_fusion.py` 的冻结行为。

如果需要复用函数，优先 import；若冻结脚本不适合被 import，将最小逻辑抽取到**新的**模块并用回归测试证明等价，不要直接重写历史文件。

工作区可能有用户尚未提交的修改。执行前运行 `git status --short`，只编辑本任务新文件和经明确允许的索引文档，不覆盖或清理其他修改，不使用 `git reset --hard` 或 `git checkout --`。

### 6.2 建议新增文件

名称可以细调，但职责不得混在一个巨大脚本中：

```text
configs/
└── rcec_v1.yaml

src/industrial_ad/fusion/
└── rcec.py                         # 成对记忆、条件分数、LOO校准、聚合纯函数

scripts/
├── evaluate_rcec_cached.py         # 单数据集/seed/shot 评估入口
├── run_rcec_mpdd_development.py    # 12候选×9配置，串行、可续跑
├── select_and_freeze_rcec.py       # 按固定规则选择并生成冻结清单
├── run_rcec_frozen_validation.py   # BTAD/MVTec/VisA，只读 frozen config
└── summarize_rcec_results.py       # 汇总、Gate、表格、失败边界

tests/
└── test_rcec.py

experiments/dynamic_fusion/rcec_v1/
├── development_mpdd/
├── freeze/
├── frozen_validation/
├── ablations/
└── reports/
```

不要把实验产物写入 A1 的 `freeze/a1_mpdd_w05` 目录。

### 6.3 建议配置内容

`configs/rcec_v1.yaml` 至少包含：

```yaml
method: rcec_v1
development_dataset: mpdd
seeds: [0, 1, 2]
shots: [1, 2, 4]
directions: [dino_to_clip, symmetric]
neighbor_k: [1, 3, 5]
lambda: [0.25, 0.50]
normal_calibration:
  kind: median_mad
  mad_scale: 1.4826
  epsilon: 1.0e-6
  z_clip: [-5.0, 10.0]
  shot_ge_2_exclusion: leave_one_reference_image_out
  shot_1_exclusion: self_and_chebyshev_radius_1
postprocess:
  gaussian_sigma: 4
  map_size: [448, 448]
  interpolation: bilinear
validation_datasets: [btad, mvtec, visa]
forbid_validation_tuning: true
```

运行时生成的最终配置应从 YAML 序列化并记录 SHA256。冻结后验证脚本只能读取冻结配置，不能再接受 `--lambda`、`--k` 等覆盖参数；如果命令行出现覆盖，应直接报错。

### 6.4 核心 API 建议

`src/industrial_ad/fusion/rcec.py` 至少实现可独立测试的函数：

```python
align_and_normalize_paired_features(...)
build_paired_reference_memory(...)
compute_conditional_scores(...)
compute_reference_loo_statistics(...)
robust_z_from_reference(...)
combine_rcec_scores(...)
aggregate_image_score(...)
validate_no_label_inputs(...)
```

核心方法函数的参数中不得出现 `gt_sp`、`mask`、`labels`、`pixel_gt` 等字段。标签只能进入 evaluator 层。

### 6.5 内存与运行要求

- 优先复用现有 DINO/CLIP feature cache；首轮开发不应重新运行两个 backbone；
- FAISS 搜索必须支持分块 query，避免 BTAD 03 或大类别一次性分配超大矩阵；
- `float32` 用于距离、校准和指标计算；如果读取 float16 compact map，要显式记录；
- 固定所有 NumPy/FAISS shuffle 随机种子；
- 支持 marker 断点续跑，但 marker 必须同时校验 config hash 和输入 hash，不能看到同名文件就无条件跳过；
- 每个配置出现 NaN/Inf、样本错位、类别缺失、参考数不符时立即失败，不允许跳过后计算平均值。

---

## 7. 单元测试和技术验收

在任何完整矩阵运行前，`tests/test_rcec.py` 至少覆盖：

1. DINO 与 CLIP sample ID 顺序不同时能正确对齐；缺失或重复 ID 会报错；
2. 非方形 CLIP grid 可正确 resize 到 DINO grid；
3. `(d_i,c_i)` 的 image/row/column metadata 不错位；
4. 条件分数只在 DINO Top-k 对应的 CLIP 项中搜索；
5. 手工构造的小数组能得到可人工验证的 `r_C|D`；
6. LOO 不会检索自身；shot≥2 会排除同一 reference image；shot=1 会排除半径1邻域；
7. MAD 退化、候选数不足、NaN/Inf 会明确失败；
8. robust z 只接收 reference statistics，不读取测试集统计；
9. 当 `lambda=0` 时，RCEC 排名/输出与校准后的 A1 一致；
10. 现有 A1 原始分数的回归误差满足 map `<1e-5`、指标 `<1e-6`；
11. DINO duplicate 距离保持误差 `<1e-6`；
12. frozen validation runner 拒绝命令行覆盖冻结参数；
13. 输出 schema、leakage flags、输入哈希完整；
14. chunked 与 non-chunked 小样本结果误差 `<1e-6`；
15. 图像聚合不改变 pixel map。

建议命令：

```powershell
.\.venv-patchcore\Scripts\python.exe -m pytest tests/test_rcec.py -q
```

技术 Gate：上述测试全部通过，并且现有项目测试没有因为新增代码出现回归。完整项目测试以当时仓库真实数量为准，不要硬编码历史的 123 tests。

---

## 8. 实验阶段与停止规则

### Phase 0：输入和复现审计

任务：

- 记录 `git status --short`，保护已有修改；
- 列出四数据集 3 seed×3 shot 的双分支缓存可用性；
- 对每个缓存记录类别数、样本数、grid、维度、reference 数、SHA256；
- 任选 MPDD s0/k1 一个类别，重算 A1 与 DINO map；
- 与既有报告/实现核对误差。

验收：输入清单完整；A1 回归一致；不存在未解释的样本 ID、grid 或特征维度冲突。

若缓存缺失：先报告缺口。仅在确实缺少特征时，才按现有冻结导出协议重新导出缺失项，不要全量重跑。

### Phase 1：算法 smoke

任务：

- 完成纯函数和单元测试；
- 用合成数据验证：正常成对一致时分数低、破坏 CLIP 配对时条件分数升高；
- 运行 MPDD s0/k1 的一个类别；
- 输出 A1、RCEC、DINO、CLIP-only 的 map 范围和指标；
- 检查速度、内存、NaN/Inf、确定性。

验收：重复两次运行结果完全一致或浮点误差 `<1e-6`；没有标签进入算法调用栈；结果 schema 合法。

### Phase 2：MPDD 小门

任务：先运行 MPDD seed0 的 shot 1/2/4，最多 12 个预注册候选。

小门通过条件：至少一个候选同时满足：

- 三个 shot 平均 Pixel-AP 不低于 A1；
- 至少 2/3 shot 为正；
- 任一 shot 相对 A1 不低于 `−0.010`；
- 没有数值退化或泄漏 flag。

若无候选通过：停止完整矩阵，输出 `RCEC_V1_EARLY_STOP_REPORT.md/json`。不要继续跑验证数据集。

### Phase 3：MPDD 9 配置开发矩阵

只对通过小门的候选运行 seed `{0,1,2}` × shot `{1,2,4}`。所有跑过的候选必须进入汇总，不得只保留最好的一项。

候选进入最终选择池必须同时满足：

1. 九配置平均 Pixel-AP 相对 A1 `≥ +0.005`；
2. 至少 7/9 配置的 Pixel-AP 高于 A1；
3. 最差单配置 Pixel-AP 差值 `≥ −0.010`；
4. MPDD 六类别中至少 4/6 的九配置平均 Pixel-AP 不低于 A1；
5. 最差类别平均差值 `≥ −0.015`；
6. 平均 Image-AP 相对 A1 `≥ −0.005`；
7. 平均 Image-F1-max 相对 A1 `≥ −0.010`；
8. 五项 leakage flags 全部为 false，且算法日志证明未读取测试统计。

这里的 `+0.005` 是绝对 AP 点，不是相对百分比。

如果多个候选合格，使用以下预先确定的字典序选择，禁止人工挑选：

1. 九配置 mean Pixel-AP 最高；
2. 若差值 `<0.001`，选择正配置数更多者；
3. 再平局，选择 `dino_to_clip` 而非 `symmetric`；
4. 再平局，选择较小 k；
5. 再平局，选择较小 λ。

如果没有候选进入选择池：RCEC v1 性能 Gate 失败，停止，不运行冻结验证。

### Phase 4：解释性消融

对选定候选完成：

- 正确 pairing；
- 三个 shuffled pairing seeds；
- DINO duplicate control；
- A1、DINO-only、CLIP-only；
- 单向与对称项的必要性对照（仅使用已经跑过的候选，不新调参数）。

解释性 Gate：正确 pairing 相对 shuffle 三次均值 Pixel-AP `≥ +0.003`，并且维度复制不产生虚假收益。

若性能 Gate 通过但解释性 Gate 失败：不得宣称“跨编码器对应关系有效”。可以保留为普通 score ensemble 候选，但必须重新评估论文是否仍有足够创新，且不能进入下一阶段前悄悄改叙事。

### Phase 5：冻结

生成独立冻结目录，例如：

```text
experiments/dynamic_fusion/rcec_v1/freeze/rcec_mpdd_v1/
├── FROZEN_METHOD_SPEC.md
├── frozen_config.yaml
├── freeze_manifest.json
├── freeze_verification.json
├── DEVELOPMENT_SELECTION_REPORT.md
└── REPRODUCE.md
```

冻结清单至少哈希：

- RCEC 源码；
- evaluator 和 runner；
- config；
- MPDD manifest；
- 使用的 feature cache；
- 12 候选/通过小门候选的完整结果；
- 最终选择报告。

冻结验证 runner 必须先验证这些哈希，再运行外部数据集。不得修改原 A1 freeze manifest。

### Phase 6：一次性冻结验证

使用唯一 frozen config，分别运行：

- BTAD：3 seeds × 3 shots；
- MVTec AD：3 seeds × 3 shots；
- VisA：3 seeds × 3 shots。

如果真实缓存覆盖与上述矩阵不一致，必须在运行前报告并说明；不得用“缺几个就只平均剩下的”方式掩盖缺失。

整个验证阶段不允许：

- 改 λ、k、direction、LOO 规则、z clip、image pooling；
- 根据某个类别手写 fallback；
- 删除下降的 seed/shot；
- 使用验证集正常测试图重新估计 reference 分布；
- 看到结果后新增候选。

最终升级为论文主方法的 Gate：

1. MPDD 性能 Gate 和解释性 Gate 已通过；
2. BTAD、MVTec、VisA 三个数据集平均 Pixel-AP 差值的平均值相对 A1 `> 0`；
3. 至少 2/3 验证数据集的平均 Pixel-AP 高于 A1；
4. 任一验证数据集平均 Pixel-AP 相对 A1不得低于 `−0.005`；
5. 27 个验证配置中至少 18 个 Pixel-AP 高于 A1；
6. 不增加严重失败类别：任一类别九配置平均差值相对 A1不得低于 `−0.030`；
7. 六项指标完整、无 NaN、类别数和样本数正确；
8. 所有冻结哈希、泄漏审计、复算审计通过。

若仅部分通过，应按下列规则处理：

- 主 Gate 全通过：RCEC 可升级为论文主方法，A1 作为强基线/消融；
- MPDD 通过但冻结验证失败：RCEC 归档为开发集过拟合负结果，A1 保持主方法；
- 性能通过但 pairing 消融失败：不得使用跨编码器一致性创新叙事；
- RCEC 通过而 RCEC-D 失败：保留 RCEC，图像分数继续 max；
- 全部失败：不影响现有 A1 论文证据，完整写入限制和未来工作。

---

## 9. 报告 schema 与证据要求

每个 dataset/seed/shot/candidate 的 JSON 至少包含：

```json
{
  "schema_version": 1,
  "method": "rcec_v1",
  "dataset": "mpdd",
  "dataset_role": "development",
  "seed": 0,
  "shot": 1,
  "candidate": {
    "direction": "dino_to_clip",
    "k": 3,
    "lambda": 0.25,
    "config_sha256": "..."
  },
  "inputs": {
    "manifest_sha256": "...",
    "dino_cache_sha256": "...",
    "clip_cache_sha256": "..."
  },
  "normal_calibration": {
    "kind": "median_mad",
    "exclusion_rule": "...",
    "test_statistics_used": false
  },
  "leakage_flags": {
    "test_labels_used_by_method": false,
    "test_masks_used_by_method": false,
    "test_distribution_used_for_calibration": false,
    "validation_dataset_used_for_tuning": false,
    "category_specific_test_rules_used": false
  },
  "metrics": {
    "rcec": {},
    "a1": {},
    "dino": {},
    "delta_rcec_vs_a1": {}
  },
  "per_category": [],
  "runtime": {},
  "checks": {}
}
```

报告必须含六项指标：

- Image-AUROC；
- Image-AP；
- Image-F1-max；
- Pixel-AUROC；
- Pixel-AP；
- Pixel-AUPRO@0.30。

同时保存：

- 逐配置 JSON；
- 汇总 CSV 和 Markdown；
- 逐类别增益；
- 失败配置和失败类别清单；
- 运行命令、时间、Python/依赖、CPU/GPU/RAM 信息；
- 输入和输出 SHA256；
- 选参全过程，包含未入选候选；
- 至少 3 个改善案例和 3 个退化案例的 sample ID；
- 不受数据再分发许可影响的 map hash/数值证据。

不能只保存一张最终表或终端截图。

---

## 10. 最终必须交付的文件

无论实验成功还是失败，执行 AI 都必须交付：

1. RCEC 独立源码、runner、配置和测试；
2. 输入缓存与 A1 回归审计；
3. 全部实际运行候选的逐配置报告；
4. MPDD Gate 报告；
5. pairing shuffle 和维度匹配消融；
6. 如果 MPDD 通过：独立 freeze 包和冻结验证报告；
7. 如果开始 RCEC-D：单独的图像评分 Gate 报告；
8. `FINAL_RCEC_DECISION.md`，明确写出 `PROMOTE`、`ARCHIVE` 或 `BLOCKED`；
9. 可直接进入论文的英文 Method 草稿、伪代码、复杂度说明和贡献点；
10. 可直接进入论文的主结果表、消融表、失败边界表及图件 source manifest；
11. 对论文准备目录的索引更新，但不要静默覆盖现有 A1 论断。

`FINAL_RCEC_DECISION.md` 至少回答：

- RCEC 是否超过 A1，而不是只超过 DINO；
- 通过了哪些 Gate，失败了哪些 Gate；
- 是否存在验证集调参或其他泄漏；
- pairing shuffle 是否支持“一致性”解释；
- 哪些数据集、类别、shot 得益，哪些退化；
- 增益是否值得增加计算和方法复杂度；
- 论文主方法应使用 RCEC 还是继续使用 A1；
- 所有结论分别指向哪个 JSON/CSV/哈希证据。

---

## 11. 建议执行命令形态

以下脚本是本任务要求新建的接口示意。实现后应能用类似命令完成工作：

```powershell
# 1. 技术测试
.\.venv-patchcore\Scripts\python.exe -m pytest tests/test_rcec.py -q

# 2. 只读审计输入
.\.venv-patchcore\Scripts\python.exe scripts/run_rcec_mpdd_development.py `
  --config configs/rcec_v1.yaml --validate-only

# 3. MPDD seed0 小门
.\.venv-patchcore\Scripts\python.exe scripts/run_rcec_mpdd_development.py `
  --config configs/rcec_v1.yaml --phase small-gate

# 4. MPDD 完整开发矩阵
.\.venv-patchcore\Scripts\python.exe scripts/run_rcec_mpdd_development.py `
  --config configs/rcec_v1.yaml --phase full

# 5. 固定规则选择与冻结
.\.venv-patchcore\Scripts\python.exe scripts/select_and_freeze_rcec.py `
  --config configs/rcec_v1.yaml

# 6. 冻结验证；脚本只能读取 frozen-config
.\.venv-patchcore\Scripts\python.exe scripts/run_rcec_frozen_validation.py `
  --freeze-dir experiments/dynamic_fusion/rcec_v1/freeze/rcec_mpdd_v1

# 7. 总结和最终决策
.\.venv-patchcore\Scripts\python.exe scripts/summarize_rcec_results.py `
  --experiment-root experiments/dynamic_fusion/rcec_v1
```

每个 runner 都应支持：

- `--validate-only`；
- 可续跑；
- 遇错返回非零退出码；
- 输出 machine-readable JSON；
- 不依赖人工复制数字；
- Windows 路径；
- 单进程低内存模式。

---

## 12. 论文写作边界

只有全部主 Gate 通过后，才可以把贡献写成：

> We introduce a reference-conditioned cross-encoder consistency mechanism that explicitly evaluates whether semantic representations agree with the structural normal neighbors retrieved from a paired normal memory bank.

在 Gate 通过前，只能写成“proposed candidate”或“under evaluation”。

无论结果如何，不得写：

- “首次融合 DINO 与 CLIP”——需要全面文献证据，当前不能保证；
- “全面 SOTA”——现有 A1 不是全面 SOTA；
- “所有指标提高”——现有 BTAD Image-AP/F1 有下降；
- “完全独立的 VisA 泛化”——AnomalyCLIP checkpoint 与 VisA 有训练域关系；
- “无监督自适应路由”——RCEC 不是运行时分支选择器；
- “证明了语义互补机理”——除非 pairing shuffle 和其他消融支持该解释。

如果 RCEC 失败，应按实际停止阶段表述。本次只能写成：预注册的参考条件跨编码器一致性公式没有在 MPDD 开发小门稳定超过固定拼接，因此按规则早停，未进入冻结验证。该结论可以进入 Discussion/Future Work，但不能包装成成功方法，也不能写成已经完成跨数据集的机理否定。

---

## 13. 最终完成定义

### 工程完成

以下均满足才算工程任务完成：

- 新代码、测试、配置、runner 和报告齐全；
- A1 冻结证据未被修改；
- 所有运行可从命令和配置复现；
- 候选、失败结果和停止决定没有被隐藏；
- 数字能追溯到逐配置 JSON；
- 最终决策文档明确。

### 科研成功

只有同时满足以下条件才算科研成功：

- MPDD 性能 Gate 通过；
- pairing 解释性 Gate 通过；
- 冻结验证主 Gate 通过；
- 防泄漏和复现审计通过；
- 相对 A1 的收益足以抵消增加的复杂度。

工程完成不等于科研成功。科研失败也不等于交接任务失败，但必须及时停止、完整归档并保留 A1 主线。

---

## 14. 最重要的最后提醒

1. **保护 A1。** RCEC 是新候选，不是在原地改写已经冻结的 A1。
2. **只在 MPDD 开发。** BTAD、MVTec 和 VisA 不能成为隐藏调参集。
3. **必须超过 A1。** 只超过 DINO 不能证明新模块有效。
4. **先验证配对关系。** 如果 shuffled pairing 不下降，就不能声称利用了跨编码器一致性。
5. **控制搜索预算。** 12 个预注册候选全部失败就停止 RCEC v1。
6. **负结果必须保留。** 不删配置、不改门槛、不换主指标。
7. **方法层不读标签。** 所有测试标签和 mask 只能进入最终 evaluator。
8. **最终由证据决定主线。** RCEC 通过才升级；不通过就继续以现有 A1 为论文主方法。

---

## 15. 2026-09-02 执行结果、独立复核与下一步

### 15.1 实际完成范围

- Phase 0 输入与 A1 回归审计：通过；
- RCEC 核心模块、配置、runner、冻结/验证接口和测试：已实现；
- RCEC 专项测试：独立实跑 `18/18 passed`；
- 项目自有回归测试：独立实跑 `pytest tests -q`，`141/141 passed`；
- Phase 2：12 个预注册候选 × MPDD seed0 × shot `{1,2,4}`，共 36 份逐配置报告；
- Phase 3—6：未执行，这是小门失败后的正确停止行为，不属于遗漏；
- RCEC-D：未执行，因为任务书规定只有 RCEC 主模块通过 MPDD Gate 后才能开始。

直接在仓库根运行不限定范围的 `pytest` 会额外收集 `methods/` 内第三方上游测试，并因这些方法需要各自虚拟环境而出现 `thop/patchcore` import error。项目回归的正确命令为：

```powershell
.\.venv-patchcore\Scripts\python.exe -m pytest tests -q
```

### 15.2 关键结果

12 个候选的三-shot平均 Pixel-AP 全部低于 A1。最佳候选为：

```text
direction = dino_to_clip
k = 5
lambda = 0.25
s0/k1 delta = +0.000303
s0/k2 delta = -0.005964
s0/k4 delta = -0.015756
three-shot mean delta = -0.007139
```

36 个 candidate-shot 组合中 35 个下降、1 个微升。λ=0.50 的候选普遍比 λ=0.25 更差，这一趋势与“加大条件一致性分数权重会稀释 A1 有效排序”的解释一致。不过当前整体公式还同时包含 reference-only robust-z 和截断，且没有完整运行 λ=0 指标对照，因此不能把全部下降严格归因于条件项本身。无论原因如何，预注册的 RCEC 整体方法没有通过；这也不是门槛过严造成的，因为最佳候选平均值本身已经为负，而且只在 1/3 shot 上出现几乎为零的正变化。

因此，RCEC v1 没有证明“成对邻域一致性”优于固定拼接。配对打乱消融按照早停规则没有运行，所以也不能从当前结果推断配对关系是否存在可利用机理；论文只能将其作为开发集负结果或 Future Work，不能作为方法贡献。

### 15.3 独立复核结论

| 检查项 | 结论 |
|---|---|
| 36 份报告数量和候选覆盖 | 通过：12 候选 × 3 shot |
| 分数有限性 | 通过：无 NaN/Inf |
| 五项泄漏标记 | 通过：全部 false |
| 小门汇总复算 | 通过：排序、均值、正向数和 worst delta 与报告一致 |
| A1 回归与核心实现测试 | 通过 |
| 是否应继续 Phase 3—6 | 否；继续将违反预注册停止规则 |
| 科研结论 | RCEC v1 失败，A1 不变 |

发现并已纠正一处文档表述：旧 `FINAL_RCEC_DECISION.md` 曾写成“所有候选在所有 shot 均退化”，但实际有一个 `+0.0003` 的微小正值。准确说法是“12 个候选的三-shot平均值全部退化，36 个组合中 35 个下降”。该文字修正不改变早停决定。

低优先级证据缺口：Phase 0 目前以叙述性审计和 A1 已有 freeze manifest 为主；RCEC 每份报告只保存代表性类别缓存哈希，没有为全部约 27 GB MPDD 双分支缓存另建逐文件 RCEC manifest。由于 RCEC 已在早期开发门失败、不会进入论文有效方法和公开复现包，这不影响否定结论。只有未来准备公开 RCEC 负结果复现包时，才需要补全逐缓存机器可读哈希清单。

### 15.4 接下来应该做什么

推荐顺序：

1. **停止 RCEC v1。** 不运行 MPDD 其余 seeds，不运行 pairing shuffle，不冻结，也不碰三个验证集。
2. **锁定 A1 论文主线。** RCEC 只在 Discussion/Future Work 用 1 段或补充材料小表说明，不进入标题、摘要贡献和主方法图。
3. **开始英文主稿。** 优先完成 Method、Experimental Setup、Results，再回写 Introduction、Related Work、Abstract 和 Conclusion。
4. **确定目标期刊。** 根据最新中科院分区、scope、篇幅、开放获取费用和模板筛选 2—3 个 SCI 四区候选；这些信息具有时效性，选择时必须重新联网核验。
5. **完成作者人工项。** 作者顺序、单位、通讯作者、基金、致谢、数据许可声明、代码开放时点和投稿声明仍需确认。
6. **最后做一次论文证据同步。** 正式英文稿完成后，再更新主张—证据矩阵、图表编号、BibTeX、版本哈希和发布包说明。

如果导师明确要求“必须增加新的算法创新”，不要继续调 RCEC 的 λ/k，也不要在 BTAD/MVTec/VisA 上救结果。应单独建立一个新的问题定义和预注册开发协议。可以考虑把“像素定位与图像判别解耦”作为新的 A1-ID 研究，但它主要针对 BTAD Image-AP/F1 的限制，不能冒充 RCEC 成功，也不能保证增强核心 Pixel-AP 创新。更大幅度的 trainable adapter、合成异常训练或替换 backbone 会改变零训练论文定位和实验成本，只有导师决定重开算法阶段后再做。
