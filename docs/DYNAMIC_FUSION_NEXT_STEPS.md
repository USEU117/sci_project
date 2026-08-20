# 动态视觉—文本融合 V4：完整实验计划与 AI 交接手册

更新日期：2026-08-19

项目：`D:\STUDY\My_github\sci_project`

当前基线提交：`ac5c2f1`（本地 `main`，当前比 `origin/main` 超前 1 个提交）

权威现状：`docs/CURRENT_DYNAMIC_FUSION_STATUS.md` 与 `docs/current_dynamic_fusion_status.json`

本文替代本文件的旧执行路线。旧路线中的 V3.3 审计、V3.3-clean、A1、冻结、四数据集验证、S0–S5 和 Route-D D0/D1 均已结束，不得再当作待办重复执行。本文的任务是：**在不否定“视觉 + 文本动态融合”论文主线的前提下，设计并验证一个真正含显式文本证据、强视觉锚点和无泄漏动态机制的 V4。**

任何后续 AI 开始工作前必须完整阅读本文。默认第一轮只允许 CPU、只读审计和现有缓存实验；GPU 空闲不构成启动授权。

---

## 1. 一页结论

### 1.1 论文方向不变，但当前 A1 不能充当最终动态视觉—文本方法

项目的研究问题继续保持为：

> 在少样本工业异常检测中，如何利用视觉正常性证据与文本异常语义证据的互补性，构造无测试泄漏、可解释、可回退且优于固定融合的动态融合方法？

当前冻结 A1 必须保留，但其正确身份是：

> DINO 图像 patch 特征 + CLIP 图像 patch 特征的固定特征拼接与 KNN 正常 memory bank。

代码证据位于 `scripts/export_anomalyclip_mpdd_features.py`：导出时只调用 `model.encode_image(...)`；虽然加载了 `prompt_learner`，但没有生成 text embedding、没有计算 image-text similarity，也没有让 prompt 输出参与被保存的 patch 特征。因此：

- A1 是有价值的双视觉表征融合基线；
- A1 不是显式视觉—文本融合；
- A1 权重固定为 0.5，不是动态路由；
- 论文不得把 A1 的提升直接归因于“文本语义”；
- V4 的首要任务不是再加一个权重公式，而是先构造并验证真正的文本分支。

### 1.2 V4 的目标结构

```text
K 张正常参考图 ──> 强视觉正常性模型 ──> S_visual（安全锚点）
                                          │
类别名/官方提示/检索提示 ──> 文本语义模型 ──> S_text（显式文本证据）
                                          │
正常参考稳定性 + 跨模态一致性 + 无标签测试稳定性
                         ──> reliability gate
                                          │
S_final = S_visual + gate * bounded_text_residual
                                          │
不可靠、缺失或超界时精确回退 S_visual
```

### 1.3 只有同时满足四个条件，V4 才能成为论文主方法

1. 新视觉锚点至少不弱于匹配协议下的官方 AnomalyDINO；
2. 显式文本分支在强视觉锚点之上具有可复现的互补 headroom；
3. 动态 gate 能在无测试真值输入下预测“文本修正是否有益”；
4. V4 稳定超过最佳固定融合，而不只是超过弱视觉基线。

若第 2 条失败，视觉—文本核心假设未得到支持；若第 3/4 条失败，只能称固定多模态融合，不能称动态融合成功。

---

## 2. 当前项目事实与不可改写的结论

### 2.1 已完成状态

- V3.3 泄漏审计完成：使用测试 `gt_masks` 的旧校准结果仅可标为 `development_only_leaky_calibration=true`、`paper_eligible=false`。
- V3.3-clean、局部安全回退、A1 9 配置开发矩阵、冻结、MPDD/BTAD/VisA/MVTec 验证均已完成。
- A1 冻结清单、METHOD_CARD、REPRODUCE、统一结果表和只读 verifier 已完成。
- Route-D：D0 Oracle headroom 通过，但 D1 可预测性失败；LOCO AUROC 0.592，置乱后 0.616。本项目停止使用原 D1 特征继续扩展旧路线。
- 当前唯一未完成的旧主线原本是 S6 论文交付；V4 属于用户新授权的研究扩展，必须使用新目录、新 RunId 和新冻结清单。

### 2.2 当前 A1 与官方 AnomalyDINO 的可比结果

以下为 `outputs/logs/cross_method_comparison_macro_mean.csv` 中 3 seeds × 1/2/4-shot 的 macro mean：

| 数据集 | 方法 | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MVTec | A1 | 0.9583 | 0.9777 | 0.9663 | 0.5546 | 0.9199 |
| MVTec | AnomalyDINO | **0.9668** | **0.9836** | **0.9664** | **0.5710** | **0.9206** |
| VisA | A1 | 0.9046 | 0.9103 | 0.9757 | 0.3725 | 0.9125 |
| VisA | AnomalyDINO | **0.9113** | **0.9182** | **0.9824** | **0.4117** | **0.9300** |

因此 A1 的正确结论是：它稳定优于项目内部 matched feature-DINO-only KNN，但没有在 MVTec/VisA 上总体超过完整官方 AnomalyDINO。V4 不得再只以内部较弱 DINO-KNN 为唯一对照。

### 2.3 旧版本在 V4 中的用途

| 旧路线 | V4 中的角色 | 不得做的事 |
| --- | --- | --- |
| V3.3-leaky | 泄漏反例 | 进入论文主表或选参数 |
| V3.3-clean | 无泄漏分数融合对照 | 包装成强动态方法 |
| local rescue | 安全回退机制参考 | 直接宣称已解决动态选择 |
| A1 | 冻结双视觉固定融合基线 | 称为显式视觉—文本或动态路由 |
| Route-D | 原可靠性特征失败证据 | 不加新证据就重复网格搜索 |
| A2/A2b/A3 | 负结果消融 | 改名后重新试同一假设 |

---

## 3. 科学假设与反证条件

V4 不是以“做出一个更高数字”为唯一目标，而是按以下假设逐层验证。

### H1：强视觉锚点假设

使用更合适的正常子空间、流形或原型建模，可以在相同 K-shot manifest 下达到或超过 AnomalyDINO。

反证：在 MPDD 开发矩阵中不能稳定超过匹配 AnomalyDINO，或增益只来自一个类别/一个 seed。

### H2：显式文本互补假设

由 text embedding 与图像 patch 计算得到的语义异常图，在强视觉锚点犯错的位置仍含独立有效信息。

反证：文本分支对强视觉锚点的 Oracle headroom 很小、仅来自单类，或 prompt 扰动后方向不稳定。

### H3：无标签可靠性可预测假设

仅使用 K 张正常参考、模型自身输出和当前测试图的无标签稳定性特征，可以预测文本修正是否有益。

反证：LOCO 预测接近阳性基率、置乱后不下降，或跨类别方向不一致。旧 Route-D 已对旧特征给出一次反证；V4 只有在分支和特征定义实质变化后才允许重新做一次 Gate。

### H4：动态优于固定假设

受约束的局部动态残差可稳定超过最佳固定融合，并控制最坏类别退化。

反证：动态版本没有超过固定融合、收益小于预注册最小效应，或外部冻结验证出现灾难性回退。

---

## 4. 数据角色与信息边界

### 4.1 数据集角色

| 数据集 | V4 角色 | 可用于什么 | 禁止用于什么 |
| --- | --- | --- | --- |
| MPDD | `development` | 分支选择、LOCO、固定权重和 gate 开发 | 宣称独立泛化 |
| BTAD | `external_validation_exposed` | 冻结后验证 | 冻结前选规则；称完全未见 |
| MVTec | `external_validation_exposed` | 冻结后主要对比 | V4 参数选择 |
| VisA | `in_domain_validation` | 域内稳定性 | 称独立 holdout；AnomalyCLIP checkpoint 在 VisA 训练过 |
| 新外部数据集 | `pristine_external_validation` | 最终一次性验证 | 冻结前查看标签指标 |

由于团队已经看过 BTAD/MVTec 的 A1 结果，它们对 V4 不是严格意义的“从未暴露”数据。若论文要提出强泛化结论，应在冻结后增加一个预先登记的新外部数据集，例如 MVTec AD 2 或获得授权的 Real-IAD。数据取得和许可证未确认前，不得把它写成已完成。

### 4.2 五项泄漏字段

所有 V4 report 必须含以下字段，且正式候选全部为 `false`：

```json
{
  "test_predictions_used_for_parameter_fit": false,
  "test_labels_used_for_parameter_fit": false,
  "test_masks_used_for_parameter_fit": false,
  "test_dataset_statistics_used_for_calibration": false,
  "test_normal_selection_used": false
}
```

MPDD development 的标签可以由独立 evaluator 用来比较候选和生成离线 benefit label，但不得进入推理输入。任何在 MPDD 标签上学到的模型必须明确标为 `auxiliary_supervised_router=true`；本计划的首选 V4-U 不允许这种监督路由。若未来改为监督路由，论文任务定义和公平基线必须重新审查。

### 4.3 强制物理隔离

- `RouterInput`：预测图、显式文本相似度、正常参考统计、稳定性特征、sample ID、grid、模型元数据；不得含 GT。
- `EvaluationTarget`：image label、pixel mask；只允许 evaluator 读取。
- 新缓存不得像旧实验缓存一样把 `gt_sp`、`imgs_masks` 与推理特征打包在同一个可供 router 读取的 NPZ 中。
- 单元测试必须证明：替换、打乱或删除 `EvaluationTarget` 不改变任何 V4 prediction hash。

---

## 5. 候选分支与优先级

### 5.1 视觉分支

| 编号 | 候选 | 作用 | 执行优先级 |
| --- | --- | --- | --- |
| V0 | 官方 AnomalyDINO | 必须超过的强基线 | 必做 |
| V1 | SubspaceAD-style，同 backbone/cache | 用现有 DINO 特征做 PCA 正常子空间重建残差；不是官方复现 | 第一 Gate，CPU |
| V2 | 官方 SubspaceAD | DINOv2-registers-giant、672、增强、PCA residual | V1 通过后 |
| V3 | FoundAD | DINOv2/DINOv3 等基础编码器 + 非线性 manifold projector | 第二候选 |
| V4-alt | FastRef + AnomalyDINO | 查询条件原型细化 | 等官方代码实际发布后 |

注意：截至 2026-08-19，`https://github.com/liyufei25/FastRef` 是空仓库。旧文档中“FastRef 官方代码可复现”的说法无效；不得自行重写后仍称官方复现。

V1 与已失败 A2b 不同：A2b 是 PCA/CCA 后继续 KNN；V1 是以“到正常子空间的重建残差”作为异常度。输出中必须命名为 `subspace_style_same_backbone`，不能写 `SubspaceAD official`。

### 5.2 文本分支

文本分支必须产生可审计的显式文本证据，例如每个 patch 对 normal/abnormal text embedding 的相似度或 logit 差。只导出 CLIP image patch 不算文本分支。

| 编号 | 候选 | 要求 | 执行优先级 |
| --- | --- | --- | --- |
| T0 | 官方 AnomalyCLIP text-conditioned anomaly map | prompt learner 输出必须实际参与 text embedding 和 image-text similarity | 第一 Gate |
| T1 | ReMP-AD retrieval-enhanced prompt | 固定官方检索库/规则，审计是否使用目标测试信息 | T0 互补不足时 |
| T2 | AdaptCLIP | 明确它是否按 K/seed 适配、训练数据和参数来源 | 条件候选 |
| T3 | 手写缺陷词 | 只作负对照 | 禁止作为主候选继续扩展 |

第一轮必须追踪一次完整数据流：

```text
prompt tokens -> prompt learner -> normal/abnormal text embeddings
 -> image patch embeddings -> similarity/logit map -> calibration -> S_text
```

报告必须保存 prompt 文本/learned context hash、tokenizer、checkpoint、text embedding hash 和相似度实现。改变 prompt 或交换 normal/abnormal text embedding 后输出不变化，必须判定实现无效。

---

## 6. 诊断矩阵：先回答“增益来自哪里”

所有方法使用相同 dataset、category、seed、shot、normal-reference manifest、测试样本和 evaluator。不同 backbone/resolution 必须单独标记，不能冒充严格同骨干消融。

| Run | 视觉 | 文本 | 融合 | 回答的问题 |
| --- | --- | --- | --- | --- |
| B0 | 官方 AnomalyDINO | 无 | 无 | 当前强视觉基线 |
| B1 | A1 的 DINO 分支 | A1 的 CLIP image patch | 固定 feature concat | 已冻结双视觉基线 |
| B2 | V1/V2/V3 | 无 | 无 | 视觉升级本身贡献 |
| B3 | 无 | T0/T1 | 无 | 显式文本单支路能力 |
| B4 | 强视觉 | 显式文本 | 最佳固定 score fusion | 多模态互补下限 |
| B5 | 强视觉 | 显式文本 | 最佳固定 feature/prototype fusion | 融合位置对照 |
| V4-U | 强视觉 | 显式文本 | 无监督受约束动态残差 | 最终候选 |
| Oracle | 强视觉 | 显式文本 | 使用 GT 的 best-of-two | 仅诊断可达上限 |

必须同时报告 `B2-B0`（视觉升级）、`B4-B2`（文本固定贡献）、`V4-U-B4`（动态贡献）、`Oracle-B2`（互补上限）、最差类别、harm 数量和 fallback coverage。论文不得把视觉升级写成融合贡献。

---

## 7. V4-U 方法预注册

### 7.1 分数与可靠性

- `S_v`：强视觉锚点 anomaly map。
- `S_t`：显式文本 normal/abnormal logit 构成的 anomaly map。
- `C_v`：视觉在 K 张正常参考及预声明增强下的稳定性。
- `C_t`：文本在固定 prompt ensemble 和预声明增强下的稳定性。
- `A_vt`：视觉/文本候选区域与局部排序一致性。

校准统计只允许来自 K 张正常参考的 leave-one-reference-out/固定增强响应。禁止使用测试集均值、分位数或正常测试样本筛选。

### 7.2 唯一允许的动态候选

在可预测性 Gate 通过前不得实现多个动态公式。通过后只允许一个预注册版本：

```text
R_text = clip(S_t - S_v, 0, cap_up)
gate = valid_text * stable_text * support_ok * agreement_ok
S_final = S_v + gate * R_text
```

约束：

- 默认只允许文本做正向、有界救援，不允许大幅压低视觉异常；
- `cap_up` 在 MPDD development 预注册后冻结；
- `gate=0` 时输出应与 `S_v` bitwise 或容差内一致；
- 文本缺失、NaN/Inf、prompt 不稳定、参考超界时回退 `S_v`；
- 不允许按类别测试 AP 写规则；
- 不允许搜索大量阈值、公式和特征后只汇报最好者。

若 gate 对所有样本恒定，或动态输出与最佳固定融合无实质差异，只能称 reference-conditioned multimodal fusion，不能称 dynamic fusion。

---

## 8. 分阶段执行路线

### G0：状态、语义和来源审计（CPU，只读）

1. 读取本文、权威状态、A1 METHOD_CARD/REPRODUCE、freeze manifest 和统一对比表。
2. 记录 Git、Python/GPU 和进程状态，但不启动任务。
3. 新建 RunId：`v4_g0_modality_and_source_audit_YYYYMMDD_v1`。
4. 审计 A1 CLIP 导出数据流；测试更换/清空 prompt learner 时当前 patch feature 是否不变。
5. 核验 SubspaceAD、FoundAD、ReMP-AD、AdaptCLIP、FastRef 的官方 URL、commit、license、权重、环境和协议。
6. 输出来源锁定表；空仓库、无权重或不可审计方法不得进入队列。

交付物：

```text
experiments/dynamic_fusion/v4_vision_text_20260819/00_g0_audit/
  state.json
  modality_semantics_audit.json
  modality_semantics_audit.md
  candidate_source_lock.json
  environment.txt
  commands.txt
  hashes.sha256
```

验收：A1 标为 `dual_visual_fixed_fusion`；候选来源有 commit/license/状态；旧冻结 hash 不变；没有启动 GPU。

### G1：统一 V4 schema 和防泄漏测试（CPU）

建议新增但当前尚不存在：

```text
src/industrial_ad/fusion/v4_contracts.py
scripts/audit_v4_inputs.py
tests/test_v4_contracts.py
```

要求：RouterInput/EvaluationTarget 不同类型和文件；prediction 不含 GT；sample ID/grid/manifest 不一致立即失败；NaN/Inf、空 prompt、错误 text 顺序失败或回退；改变 GT 不改变 prediction hash；同配置确定；所有拟合记录数据角色和 hash。

验收：全部 CPU 测试通过；失败注入必须失败；旧 V3/A1 测试不回归。

### G2：强视觉锚点 Gate（先 CPU）

第一步只运行 `subspace_style_same_backbone`：复用冻结 DINO raw patch cache，在正常参考 patch 上拟合 PCA 子空间，以测试 patch 重建残差生成 anomaly map。禁止改 backbone、重新导出特征或启动 GPU。

预注册：PCA 能量阈值只允许 `{0.95, 0.99}`；增强数先为 0；map/evaluator 与 matched DINO-only 一致；只在 MPDD 选择；PCA 仅由当前 seed/shot/category 的 K 张正常参考拟合。

对照：matched feature-DINO-only KNN、官方 AnomalyDINO、A1、V1。

Gate G2：

- 9 配置平均 P-AP 相对匹配 AnomalyDINO `>= +0.010`；或 P-AP 基本持平（绝对差 `<=0.003`）且 AUPRO `>= +0.010`；
- 至少 7/9 配置非负、4/6 类正；
- 任一类平均 P-AP 退化不得低于 `-0.020`；
- 泄漏、对齐、确定性、有限值测试全过。

若失败：V1 归档，再决定是否做一次官方 SubspaceAD smoke；不得无限扩 PCA/whiten/centering 网格。若通过：才审计官方 giant/672 版本。6GB GPU 先 validate-only 和单类别/单图 smoke；OOM 时优先 batch=1、CPU offload，不得换小模型后仍按官方方法命名。

### G3：显式文本分支 Gate（CPU 审计，必要时 GPU）

首先实现/复用官方 AnomalyCLIP 的真实 text-conditioned anomaly map，并证明 prompt learner/text embeddings 影响输出。

测试：交换 normal/abnormal prompt 后 logit 方向应变化；prompt checkpoint 变化应改变 embedding/output hash；固定 prompt ensemble 重跑确定；只用官方模板或预注册检索库；参考校准只用 K 正常图。

Text Gate：

- 输出非恒定，方向和 score definition 明确；
- prompt/view 稳定性达到预注册阈值；
- 与强视觉的逐像素 Oracle mean P-AP headroom `>= +0.015`；
- 至少 4/6 类 headroom `>= +0.005`；
- headroom 不得 60% 以上来自单类；
- 保存文本有益和有害位置。

T0 失败后，按 T1 ReMP-AD、T2 AdaptCLIP 顺序各做一次来源/协议 Gate。不同 shot/seed 协议不得混入同一均值。三者均失败则停止 V4，不能声称已有证据支持文本互补。

### G4：2×2 分支因子实验（CPU）

固定一个通过 G2 的视觉和一个通过 G3 的文本，运行：

```text
weak visual only
strong visual only
text only
weak visual + text fixed
strong visual + text fixed
strong visual + shuffled text control
strong visual + wrong-category text control
```

报告 visual upgrade effect、text addition effect、interaction effect、shuffled/wrong-category degradation 和逐类 rescue/harm。若错误类别/打乱文本与正确文本几乎相同，说明文本语义未被利用，停止 V4。

### G5：最佳固定融合强对照（CPU）

动态方法必须超过合法选择并冻结的最佳固定融合。只允许小型网格，例如视觉权重 `{0.25,0.50,0.75}`；校准只允许一个 reference-only robust z/rank 版本。使用 MPDD 类别级 LOCO：5 类选择、第 6 类评估。保存每 fold 和所有未选候选。权重差异 `<0.002` 时选等权或更简单版本。

### G6：新可靠性可预测性 Gate（CPU）

只有分支实质变化后才允许重新测试一次。候选特征最多包括：参考 leave-one-out 稳定性、固定增强 text logit 方差/符号一致性、跨尺度/跨 prompt 一致性、视觉/text 区域 IoU 与局部排序、支持范围超界、背景/边界比例。

benefit label 仅由独立 evaluator 离线生成，不进入 router artifact。采用 MPDD LOCO，并做 label/feature permutation。

Gate G6 必须全部满足：held-out mean AUROC `>=0.65`；AP 高于阳性基率 `+0.10`；至少 4/6 folds 方向一致；置乱后 AUROC 降低 `>=0.05` 且接近随机；输入无 GT/测试整体统计；相对旧 D1 有真实新增信息。

失败则记录 `dynamic_predictability_failed` 并停止动态模块。可以保留固定多模态 B4，但不得新增 V4.1/V4.2。

### G7：V4-U 有界动态残差 Gate（CPU）

G6 通过后实现第 7 节唯一公式。对照包括 B0–B5、A1、V3.3-clean、V4-U、Oracle。

硬标准：

- V4-U vs 最佳固定：mean P-AP `>= +0.005`；
- V4-U vs 官方 AnomalyDINO：mean P-AP `>= +0.010`；
- 至少 7/9 seed-shot 非负、4/6 类正；
- 最大单类平均退化 `>= -0.010`；
- AUPRO 整体下降不超过 `0.003`；
- fallback 测试通过；置乱/去除 reliability 后增益消失；
- gate 非常数，coverage、risk-coverage、原因代码完整。

任一硬条件失败即归档，不进入 GPU 完整矩阵。

### G8：GPU 特征导出与完整矩阵

只有 G2/G3/G6/G7 全通过且用户明确同意后执行：validate-only → 单类别 seed0/K1 smoke → 审计 NPZ/schema/hash/显存 → MPDD 全类别 s0/K1 → 串行补齐 3 seeds × 1/2/4-shot。每项立即生成 marker/audit。队列可恢复，不覆盖旧目录。GPU 仅用于不可复用的特征导出/推理；融合、统计和审计用 CPU。

### G9：冻结

通过后新建：

```text
experiments/dynamic_fusion/freeze/v4_vision_text_<config>/
  freeze_manifest.json
  METHOD_CARD.md
  REPRODUCE.md
  prediction_schema.json
  metric_definition.md
  freeze_verification.json
  freeze_verification.md
```

冻结代码、配置、视觉/text checkpoint、prompt/tokenizer、manifest、calibration、gate、cap、evaluator、cache 和 prediction hash。`--create`/`--verify` 互斥；verify 只读且前后 manifest hash/mtime 不变。

### G10：冻结后验证

顺序：MVTec → BTAD → VisA → 新外部数据集。前三者已对项目暴露；VisA 是 in-domain。不得根据验证结果回调。

最低标准：MVTec/BTAD 相对最佳固定均值均不为负；至少一项动态增益 `>=+0.005`；相对官方 AnomalyDINO 总体不劣；无单类 `<-0.020` 灾难回退；新外部数据若可取得则一次性冻结评估。

### G11：论文与复现交付

完成方法图、伪代码、主表、逐类表、固定/动态对照、错误重叠、risk-coverage、失败案例、资源表和命令。所有数字从 per-config report 自动重算，误差 `<1e-6`。

---

## 9. 建议脚本、命令与产物合同

以下脚本目前尚不存在，是建议接口；创建前先搜索等价实现：

```text
scripts/audit_v4_modality_semantics.py
scripts/audit_v4_inputs.py
scripts/evaluate_v4_visual_anchor.py
scripts/export_v4_text_maps.py
scripts/analyze_v4_complementarity.py
scripts/evaluate_v4_fixed_fusion.py
scripts/analyze_v4_predictability.py
scripts/evaluate_v4_bounded_residual.py
scripts/run_v4_matrix.py
scripts/freeze_v4.py
tests/test_v4_contracts.py
tests/test_v4_fallback.py
tests/test_v4_freeze.py
```

### 9.1 接手后的只读命令

```powershell
Set-Location D:\STUDY\My_github\sci_project
git status --short --branch
git log -3 --oneline --decorate
Get-Content docs\CURRENT_DYNAMIC_FUSION_STATUS.md -Raw
Get-Content docs\DYNAMIC_FUSION_NEXT_STEPS.md -Raw
.venv-patchcore\Scripts\python.exe scripts\freeze_a1_mpdd.py --verify
```

### 9.2 现有回归测试

```powershell
.venv-patchcore\Scripts\python.exe -m pytest `
  tests\test_v3_3_clean.py `
  tests\test_v3_3_rescue.py `
  tests\test_freeze_a1_mpdd.py -q
```

### 9.3 V4 Gate 命令形式

```powershell
.venv-patchcore\Scripts\python.exe scripts\audit_v4_modality_semantics.py `
  --output-dir experiments\dynamic_fusion\v4_vision_text_20260819\00_g0_audit

.venv-patchcore\Scripts\python.exe scripts\audit_v4_inputs.py `
  --validate-only --dataset mpdd --manifest data\splits\mpdd\manifest.json

.venv-patchcore\Scripts\python.exe scripts\evaluate_v4_visual_anchor.py `
  --dataset mpdd --seed 0 --shot 1 `
  --mode subspace_style_same_backbone --pca-ev 0.99 `
  --output-dir experiments\dynamic_fusion\v4_vision_text_20260819\02_visual_gate\s0_k1

.venv-anomalyclip\Scripts\python.exe scripts\export_v4_text_maps.py `
  --validate-only --dataset mpdd --seed 0 --shot 1 `
  --manifest data\splits\mpdd\manifest.json

.venv-patchcore\Scripts\python.exe scripts\analyze_v4_complementarity.py --validate-only
.venv-patchcore\Scripts\python.exe scripts\evaluate_v4_fixed_fusion.py --validate-only

.venv-patchcore\Scripts\python.exe scripts\analyze_v4_predictability.py `
  --dataset mpdd --protocol leave_one_category_out --permutation-tests 100

.venv-patchcore\Scripts\python.exe scripts\evaluate_v4_bounded_residual.py `
  --dataset mpdd --validate-only
```

具体参数只有在脚本实现、预注册配置写入 `config.json` 后才能执行。示例命令不代表脚本已经存在或实验已经完成。

### 9.4 每个 Run 的目录

```text
<RunId>/
  config.json
  command.txt
  environment.json
  git_state.json
  source_lock.json
  input_manifest.json
  hashes.sha256
  stdout.log
  stderr.log
  predictions/              # 不含 GT
  evaluation_targets/       # evaluator-only；可引用而非复制
  per_category.csv
  metrics_report.json
  audit.json
  audit.md
  marker.json
```

`marker.json` 至少包含 RunId、status、gate_passed、paper_eligible、dataset_role、seed、shot、config/prediction hash 和 evaluation report。目录或 marker 存在不代表成功；必须核验进程、日志结尾、预测数量/schema、有限值、audit 和 evaluation report。

---

## 10. 指标、统计与图表

- 主指标：Pixel AP；共同报告 AUPRO、Pixel AUROC。
- 图像级：Image AUROC、Image AP、Image F1-max。
- 安全性：worst-category delta、harm count、fallback rate、coverage、risk-coverage。
- 效率：GPU 峰值、推理/建库时间、磁盘、参数量。
- 3 seeds × 1/2/4-shot 是参考采样鲁棒性，不是 9 个独立数据集。
- 报告 mean±std、逐 seed/shot/类和最坏退化；不得伪独立 t-test。
- bootstrap 如使用，按图像或类别重采样并说明共享测试集相关性。

论文图至少包括：V4 数据流/信息边界；B0–B5/V4 贡献分解；错误重叠/Oracle；risk-coverage/fallback；正向、伤害、回退案例；逐类 delta heatmap；精度—显存—速度图。

---

## 11. 测试清单

GPU 矩阵前至少通过：

1. RouterInput 无标签/掩码；
2. 修改 GT 不改变 prediction hash；
3. sample ID 重复/缺失/错位失败；
4. manifest 与 seed/shot 匹配；
5. prompt 实际影响 text embedding/map；
6. normal/abnormal prompt 交换改变方向；
7. wrong-category/shuffled text control；
8. grid 对齐记录明确；
9. NaN/Inf 失败或回退；
10. 文本缺失精确回退视觉；
11. gate=0 等于视觉；
12. gate/cap 在预注册范围；
13. 重跑确定；
14. LOCO 无类别穿越；
15. permutation 真正打乱；
16. evaluator 唯一读取 GT；
17. validate-only 不写正式预测；
18. freeze verify 严格只读；
19. 旧 A1 freeze 229 项仍通过；
20. 旧 V3.3-clean/rescue/freeze 测试不回归。

---

## 12. 论文交付分支与最终效果定义

### A：完整成功

H1–H4 全通过；V4-U 在开发和冻结验证中超过最佳固定融合，相对 AnomalyDINO 总体不劣。可以保留“动态视觉—文本融合”主标题。贡献是强视觉锚点、显式文本证据、reference-conditioned 可靠性、有界残差/视觉回退、严格冻结评估和失败路由实证。

### B：文本有效，但动态不超过固定

H1/H2 通过，H3/H4 失败。论文改为“可靠的参考条件视觉—文本融合”，动态不作为性能贡献。

### C：只有视觉有效

H1 通过、H2 失败。不能声称视觉—文本性能贡献；选择完成 A1 工程论文，或转为多模态互补性/泄漏风险的负结果研究。

### D：新视觉也未超过 AnomalyDINO

停止算法扩展，诚实交付当前 A1：双视觉特征融合相对 matched internal baseline 稳定有效，但不宣称 SOTA 或动态路由。

完整成功的最低数值门：MPDD 上强视觉 vs AnomalyDINO P-AP `>=+0.010`；固定视觉—文本 vs 强视觉 `>=+0.010`；V4-U vs 最佳固定 `>=+0.005`；7/9 非负、4/6 类正、worst category `>=-0.010`。冻结后 MVTec/BTAD 相对最佳固定均不为负，至少一个外部数据集动态增益 `>=+0.005`，相对官方 AnomalyDINO 总体不劣。

这些是最小实际效应门，不保证达到即可称 SOTA；SOTA 仍需核对 backbone、分辨率、预训练数据、shot/seed 和 evaluator。

---

## 13. 停止规则与资源控制

任一发生即停止对应路线：协议/缓存泄漏；文本未实际使用 text embedding；wrong-category/shuffled text 与正确文本等效；Oracle headroom 不足；G6 未过；动态未超过最佳固定；增益只来自单类/seed；需要用外部验证回调；6GB 无诚实运行方案；官方代码/许可证/权重不可用。

禁止：大量 V4.x 后按测试挑最好；扩写手工 defect prompts；把 CLIP image feature 称 text evidence；把 9 配置称独立实验；覆盖旧冻结目录；因 GPU 空闲开跑；无 Gate 补大矩阵。

---

## 14. 后续 AI 第一轮必须执行

1. 阅读本文、当前状态、A1 METHOD_CARD/REPRODUCE 和 freeze manifest。
2. 检查 Git；`ac5c2f1` 已 commit 未 push，不得覆盖或混入无关修改。
3. 运行 A1 只读 verify 和现有 36 项相关测试。
4. 创建 G0 新 RunId，不修改旧实验。
5. 完成 A1 modality semantics audit，证明当前第二支路是 CLIP image patch，不是显式文本输出。
6. 核验候选官方来源/commit/license/权重。
7. 实现 G1 RouterInput/EvaluationTarget 合同与测试。
8. 在现有 DINO cache 做 `subspace_style_same_backbone` seed0/K1 CPU smoke。
9. 报告 G0/G1/G2-smoke，再请求 GPU 授权。

第一轮不得启动新 GPU 导出、修改 A1 freeze、在外部验证集调参、写 SOTA 结论，或把建议脚本当已实现。

---

## 15. 最终完成定义

1. A1 的双视觉固定融合身份在代码、方法和论文中统一；
2. V4 有显式、可追踪的文本数据流；
3. 强视觉、文本、固定和动态贡献被因子实验拆开；
4. 每个 Gate 有预注册门槛、报告和停止决定；
5. RouterInput/EvaluationTarget 物理隔离，五项泄漏字段全 false；
6. MPDD 3 seeds × 1/2/4-shot 完整审计；
7. 只有超过最佳固定才称动态成功；
8. 冻结后各数据集按正确角色报告，不回调；
9. 尽可能增加真正新外部数据；
10. freeze、METHOD_CARD、REPRODUCE、环境、命令、日志、hash、审计齐全；
11. 汇总可自动重算，误差 `<1e-6`；
12. 正负结果、最坏类别、失败案例、资源和限制进入论文；
13. Git 归档源码/测试/轻量证据，不误提交数据/cache；
14. 根据第 12 节 A/B/C/D 的真实结果选择叙事，不预设成功。

最终目的不是保住“动态融合”这个词，而是得到审稿人可验证的结论：**文本是否在强视觉异常检测器之上提供独立信息，以及这种信息能否被无泄漏、可回退的动态机制可靠利用。**

---

## 16. 外部来源（接手时重新核验）

- AnomalyDINO 论文：https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html
- AnomalyDINO 代码：https://github.com/dammsi/AnomalyDINO
- SubspaceAD 论文：https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html
- SubspaceAD 代码：https://github.com/CLendering/SubspaceAD
- FoundAD 论文：https://openreview.net/pdf?id=YRrlJ8oVEH
- FoundAD 代码：https://github.com/ymxlzgy/FoundAD
- FastRef 论文：https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html
- FastRef 空仓库状态：https://github.com/liyufei25/FastRef
- ReMP-AD 论文：https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html
- ReMP-AD 代码：https://github.com/cshcma/ReMP-AD
- VisionAD：https://github.com/Qiqigeww/VisionAD

外部论文数字只用于候选筛选。只有按本项目统一 manifest、seed/shot、数据角色和 evaluator 完成的结果，才能进入本地公平排名。
