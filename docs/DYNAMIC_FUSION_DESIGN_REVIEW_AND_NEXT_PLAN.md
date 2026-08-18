# 动态融合设计审查与下一步完整执行规划

状态日期：2026-08-18  
用途：交接给后续 AI，作为动态融合设计判断、证据修复、可选新设计 Gate、正式冻结与论文交付的统一执行依据。  
范围：只讨论动态融合主线及其直接依赖，不替代基线总状态报告。  
首轮资源限制：**禁止启动 GPU、禁止覆盖旧实验、禁止先写成功结论。**

---

## 1. 这份规划要解决什么问题

后续工作不能再简单地问“V3.3、V3.4、V3.5 哪个版本号更高”，而必须回答以下五个问题：

1. 哪些版本在工程上实现了，哪些版本在科学协议上有效？
2. 哪些结果只是开发诊断、Oracle 上限或含泄漏的旧证据？
3. 当前最强方法到底是动态路由，还是固定特征融合？
4. 是否还有足够证据支持继续设计一个新动态版本？
5. 后续 AI 最终要交付什么，达到什么标准才算完成？

本规划的总目标不是继续增加版本号，而是形成一条可审计、可复现、可写入论文且不会因泄漏或比较口径被审稿人推翻的主线。

---

## 2. 先给最终判断

### 2.1 当前推荐结论

1. **旧 V3.3 不能作为正式方法。** 它使用测试 `gt_masks` 选择正常测试图并估计 z-score 统计，属于测试信息泄漏；旧数值只能作为 `development_only` 的失败证据。
2. **V3.3-clean 是有效的无泄漏固定分数融合基线。** 它证明参考图校准可以安全工作，但 MPDD seed0/K1 的最佳平均 Pixel AP 增益约为 `+0.0173`，不是当前最强主方法。
3. **V3.3 local rescue 是安全但收益有限的保护模块。** 最好配置约 `+0.0051`，低于 V3.3-clean 固定融合，不应升级为主方法。
4. **V3.4 的实际动态路线没有超过静态融合。** 其报告中静态约 `+0.0769`，改进路线约 `+0.0754/+0.0739/+0.0669/+0.0755`；同时实现仍以测试 masks 进行校准。因此 V3.4 只能保留为“Oracle 有互补上限、现有可靠性代理失败”的开发证据。
5. **V3.5 的图像级 gate 和缺陷词 prompt 路线应停止扩展。** 图像级 Oracle 相对静态方法的额外上限只有约 `+0.010`；手写缺陷词 ensemble 平均约 `-0.036`。V3.5 同样继承了测试标签/掩码校准问题。
6. **当前最强、最完整的候选是 A1：特征级 concat + KNN normal memory bank。** 但 A1 是固定特征融合，不是逐图或逐像素动态路由。论文和文档不得继续把 A1 模糊包装为“动态路由已经成功”。
7. **现在不需要自动启动一个 V3.6/V4 大设计。** 推荐先收敛 A1、修复证据链和论文表述。只有当研究目标明确要求“动态”作为核心创新，而且新的 Oracle/可预测性 Gate 证明 A1 之上仍有可利用空间时，才允许启动一次受限的新设计。

### 2.2 推荐的论文定位

推荐主线名称：

> Reference-Conditioned Multimodal Feature Fusion with a Normal Memory Bank

中文可写为：

> 基于正常参考记忆库的参考条件多模态特征融合

不要直接称为“动态路由方法”，除非后续条件分支真的通过全部 Gate。

现阶段可主张的是：

- 只使用 K 张正常参考图构建 memory bank；
- DINO 与 CLIP patch 特征在对齐、归一化后 concat；
- KNN 距离产生像素异常图；
- 不使用测试标签、测试掩码或测试集统计进行融合参数拟合；
- 在 MPDD 开发矩阵及冻结后的 MVTec 等验证中获得稳定增益；
- 同时保留对分数路由、图像级 gate、缺陷词和泄漏校准失败的系统分析。

---

## 3. 统一术语与数据流

### 3.1 允许使用的数据

- 当前 seed/shot 对应的 K 张正常参考图；
- 冻结的 DINO、AnomalyCLIP 模型与 checkpoint；
- 参考图和测试图的无标签特征；
- sample ID、shape、空间网格、分支名称等对齐元数据；
- 预先声明的、与测试真值无关的配置；
- 最终评价阶段由 evaluator 单独读取的测试标签与掩码。

### 3.2 禁止进入融合/校准/路由的数据

- 测试 `labels`；
- 测试 `gt_masks`；
- 用测试标签或掩码筛出的“正常测试图”；
- 测试集总体均值、分位数、类别难度或最优权重；
- 根据测试 AP/AUROC 选择的类别规则；
- 根据 MVTec/VisA 最终结果回头修改冻结配置。

### 3.3 A1 当前数据流

```text
K 张正常参考图
  ├─ DINO patch features ── L2 normalize ─┐
  └─ CLIP patch features ── 对齐网格 ─ L2 normalize ─┤
                                                   ├─ concat ─ L2 normalize
测试图                                              │
  ├─ DINO patch features ── L2 normalize ─┐         │
  └─ CLIP patch features ── 对齐网格 ─ L2 normalize ─┘

参考 concat features ──> normal KNN memory bank
测试 concat features ──> KNN distance ──> pixel anomaly map
测试真值 ──> 仅 evaluator 读取 ──> AUROC/AP/AUPRO
```

这里的“参考条件”来自每个 seed/shot 的正常 reference bank；当前权重 `w=0.5` 是冻结的，不随测试图动态变化。

---

## 4. 各版本的真实状态与设计评价

## 4.1 V1：早期不确定性路由

核心思路：对视觉分支和文本分支的分数、熵、不确定性和分歧做路由，在图像级或像素级选择分支或加权融合。

完成内容：接口、对齐、校准、路由输出、敏感性和部分独立验证均有工程证据。

已知问题：

- 不同分支分数尺度不同，原始 Bernoulli entropy 不可直接横向比较；
- sigmoid 饱和会制造“低熵但错误地自信”；
- 部分类别被路由到较弱文本分支，导致最坏类别退化；
- 平均指标掩盖逐类 harm；
- 动态路由没有稳定超过强视觉分支。

结论：保留为失败分析和工程基础，不恢复为主方法。

## 4.2 V2：视觉默认的安全路由

核心改进：

- 排序保持校准；
- 支持范围检测；
- 图像级与像素级路由分离；
- 文本只在通过可靠性检查时辅助；
- 失败时回退视觉输出。

优点：协议意识和工程防护明显好于 V1，是后续 `RouterInput`、fallback、reason code 的来源。

问题：

- 可靠性特征不等于“哪个分支在异常像素上更正确”；
- 仅从置信度、熵、集中度和分歧推断分支正确性，预测力不足；
- V3 Gate A2 的 held-out 类别中，正 Pixel AP 类别不足，未通过预设 Gate；
- 它更像安全工程框架，而不是已经证明有效的论文主方法。

结论：保留 API、测试思想与视觉回退原则；不继续扩大旧 V2 路由矩阵。

## 4.3 V3 / V3.2：区域候选与可靠性救援

核心思路：视觉先生成区域候选，再依据背景拒绝、可靠性和局部信号让文本救援。

结果：

- `v3_2_gate_b`：平均 held-out Pixel AP 约 `-0.0000001`，0/6 正，失败；
- `v3_2_gate_b_v2`：平均 held-out Pixel AP 约 `-0.000248`，0/6 正，失败。

核心原因：

- 视觉候选区保证安全，但也限制了文本发现视觉完全漏检区域的能力；
- 可靠性代理与真实分支优劣关联太弱；
- 局部修正幅度受限后，剩余可获得增益很小；
- 放大文本权重又会重新引入 hallucination 和背景误报。

结论：区域候选、背景拒绝、视觉回退可作为安全模块保留；V3.2 方法本身归档。

## 4.4 V3.3 原版：z-score 融合、多策略选择

实现包含：

- weighted ensemble；
- per-pixel max-z selection；
- AdaptCLIP 内部融合后再与 DINO 两阶段融合；
- safety annealing 等类别规则。

致命问题：`estimate_robust_stats(maps, masks)` 使用测试 `gt_masks` 选择完全正常的测试图，再估计 median/IQR。由此得到的旧 V3.3 数值包含标签泄漏。

此外：

- metal_plate safety annealing 使用测试基线 AP 判断是否触发，属于按测试指标选类别规则；
- 网格搜索候选多，容易形成测试集多重尝试；
- 旧结果不能通过补写五个 `false` 字段变成有效结果。

结论：旧 V3.3 全部标记：

```text
development_only_leaky_calibration=true
paper_eligible=false
```

旧数值只用于展示泄漏会虚增结果，不再作为主要性能证据。

## 4.5 V3.3-clean：参考图校准的无泄漏固定融合

核心修复：

- `RouterInput` 与 `EvaluationTarget` 分离；
- 校准统计只由 K 张正常参考图生成；
- 测试标签/掩码只进入 evaluator；
- 缺失、错位、重复 sample ID、NaN/Inf、确定性和视觉回退均有测试。

当前证据：

- CPU 测试 15/15；
- 与 local rescue 合计测试 28/28；
- MPDD seed0/K1 中，`w=0.40` 相对视觉平均 Pixel AP `+0.0173`，6/6 类正；
- 旧泄漏 V3.3 同口径约 `+0.0754`，泄漏将增益放大约 7 倍。

评价：这是有效、必要的安全基线，但不是当前效果最强的方法。

## 4.6 V3.3 local rescue：有界文本局部救援

设计：

```text
视觉候选 -> 正常参考超界 -> prompt/增强稳定性 -> 背景拒绝
-> 有界单向文本残差 -> 视觉回退
```

优点：

- 信息边界清楚；
- 文本不能全图改写视觉分数；
- 有明确 reason code；
- 失败模式容易解释。

结果：最佳配置约 `+0.0051`，安全但低于 V3.3-clean 固定融合 `+0.0173`。

结论：可作为 safety/fallback 消融模块，不作为主方法继续扩展。

## 4.7 V3.4：真正逐像素动态加权尝试

V3.4 包含：

- Route 2：空间不确定性加权；
- Route 3：normal-reference gating；
- Route 4：跨模态 agreement gating；
- Oracle：使用 GT 在每个像素选择更有利分支。

开发结果：

- Oracle 相对 DINO 的平均上限约 `+0.4717`；
- 静态融合约 `+0.0769`；
- 多个实际动态路线约 `+0.0754/+0.0739/+0.0669/+0.0755`，没有超过静态融合；
- 早期实现甚至出现约 `-0.08` 的平均退化。

必须正确解释 Oracle：

- Oracle 使用真实掩码逐像素取 max/min，只证明两分支在像素层面存在互补；
- Oracle 不证明当前无标签可靠性特征能够找到这些像素；
- Oracle 相对 DINO 的上限很高，不等于相对 A1 已经存在同样大的剩余上限。

设计错误/局限：

1. `robust_z_score`、`per_pixel_z_score` 仍以测试 masks 选择正常测试样本；正式协议无效。
2. “离正常参考更远”是异常证据，不等于分支不可靠；“更接近正常”也不等于应该信任。Route 3 混淆了 anomaly evidence 与 reliability。
3. 空间方差既可能表示噪声，也可能表示真实小缺陷；简单地降低高方差分支权重会压制异常。
4. agreement 只能判断两个分支是否一致，不能判断一致时是否一起错误。

结论：V3.4 不做 clean 大矩阵。仅保留 Oracle 与失败路线作为新设计前的理论诊断。

## 4.8 V3.5：图像级层次 gate 与缺陷词 ensemble

方向 C：图像级 gate。

- 离散 gate、连续 gate、agreement gate 的分数大致与旧静态 V3.3 接近；
- 图像级 Oracle 相对静态融合的额外 Pixel AP 上限仅约 `+0.010`；
- 图像级决策无法精确解决像素级文本互补问题。

方向 B：手写缺陷词 prompt ensemble。

- `damaged/broken/scratched/...` 等手写词平均约 `-0.036`；
- 明显弱于已有 learned prompts。

协议问题：V3.5 `estimate_robust_stats` 使用 `gt_masks`，图像分数校准使用 `gt_labels==0`；正式结果同样含测试信息泄漏。

结论：V3.5 两条路线都归档，不再补 GPU、不做更多 prompt 列表或图像级阈值网格。

## 4.9 A1：特征级 concat + KNN normal memory bank

冻结配置：

- DINO ViT-B/14 patch feature；
- AnomalyCLIP patch feature；
- CLIP grid 对齐到 DINO grid；
- 分支分别 L2 normalize；
- 等权 concat 后再次 L2 normalize；
- KNN `k=1`；
- memory bank 只使用 K 张正常参考图 patch；
- `pca_dim=0`、`whiten=0`、`w=0.5`。

主要结果：

- MPDD development：9/9 配置全正；相对旧 DINO score baseline 平均 `+0.0486`；
- A1 feature-level DINO-only 自身相对旧 baseline 为 `+0.0227`；
- 因此 concat 相对严格匹配的 feature-level DINO-only 的额外增益为 `+0.0258`；
- CLIP-only 相对旧 baseline 为 `-0.0222`；
- MVTec 冻结后验证：9/9 配置全正，concat 相对 feature-level DINO-only 平均 `+0.0320`；
- VisA：9/9 配置全正，平均 `+0.0524`，但 AnomalyCLIP checkpoint 在 VisA 上训练，必须标为 in-domain validation，而不是独立外部 holdout；
- BTAD：K1 三个 seed 为正，平均约 `+0.0726`，但验证覆盖与 MVTec 9 配置不同。

优点：

- 方法简单、稳定、可复现；
- 正常参考来源明确；
- 比分数后融合更能利用 CLIP 与 DINO 的表征互补；
- 多 seed/shot 与 MVTec 外部验证有支持；
- 不依赖测试标签进行路由或参数拟合。

局限：

- 不是动态路由；
- MPDD 的 `+0.0486` 混合了 feature-level DINO 表征改进，纯 concat 贡献应看 `+0.0258`；
- 9/9 是同一测试集上的 9 组参考采样鲁棒性，不是 9 个独立数据集；
- 少数 MVTec/VisA 类仍有小幅退化；
- VisA 不是完全独立于 checkpoint 训练数据；
- 当前冻结验证脚本存在先重写 manifest 再 `--verify` 的实现错误，需修复证据工具。

结论：A1 是当前唯一适合继续收敛为正式方法的路线。

---

## 5. 已知问题总表

| 问题 | 影响 | 必须采取的动作 |
|---|---|---|
| 旧 V3.3/V3.4/V3.5 使用测试 masks/labels 校准 | 旧结果不能作为正式证据 | 标记 development-only，不修改旧数值 |
| A1 被叫作动态路由 | 方法定义与实现不一致 | 更名为参考条件特征融合 |
| MPDD `+0.0486` 与 matched DINO-only 口径混淆 | 夸大融合本身贡献 | 同时报告 `+0.0486` 与 concat-minus-DINO-only `+0.0258` |
| VisA 被标记 external holdout | checkpoint 在 VisA 训练 | 改为 in-domain frozen validation |
| BTAD 只完整覆盖 K1，且旧报告提到部分历史缓存复用 | 跨数据集矩阵不对称 | 明确 K1-only 与缓存来源，不补写成 9/9 |
| 9 seed/shot 配置共享测试图 | 不能视为独立重复实验 | 报告 reference-sampling robustness，不做伪独立显著性 |
| `freeze_a1_mpdd.py --verify` 先写后验 | 无法证明冻结未变化 | 改成严格只读验证，并验证全部条目 |
| VisA 路径写成 `a1_visa_20260817`，实际为 `a1_visa_20260818` | 复现路径错误 | 修正文档并做链接检查 |
| 旧阶段七报告称 VisA/MVTec 未完成，新表称已完成 | 权威状态冲突 | 新建统一状态快照，不覆盖旧报告 |
| Git 有大量未提交代码与产物 | 结果不可追溯、易丢失 | 做范围审计后分批提交，数据/cache 排除 |
| 某些 JSON `status=passed` 但内部 `gate_passed=false` | 机器状态语义冲突 | 统一状态 schema，外层状态必须反映 Gate |
| 失败路线过多、版本号不断增长 | 多重尝试与选择偏差 | 冻结失败登记表，停止无 Gate 的新版本 |

---

## 6. 是否需要新设计：决策树

```text
论文是否必须以“动态路由”为核心创新？
  ├─ 否：执行路线 S（推荐）
  │     A1 收敛 -> 证据修复 -> 统一表格 -> 论文/复现交付
  │
  └─ 是：先执行路线 D 的 Gate D0
        ├─ A1 之上没有足够 Oracle/可预测性 headroom -> 停止新设计，回路线 S
        └─ headroom 足够且能被无标签特征预测 -> 只实现一个预注册候选
              ├─ CPU Gate 不过 -> 永久归档动态路线
              └─ CPU Gate 通过 -> 冻结后才允许一次外部验证
```

任何后续 AI 不得因为“GPU 空闲”自动选择路线 D。

---

## 7. 路线 S：推荐的收敛与正式交付路线

## S0. 建立新的只读状态快照

目的：把 2026-08-18 的真实状态固定下来，区分旧历史报告与当前权威状态。

动作：

1. 读取本文件、`DYNAMIC_FUSION_NEXT_STEPS.md`、freeze manifest、最新 MPDD/VisA/MVTec audit。
2. 记录 Git commit、`git status --short`、活动进程、GPU、队列状态。
3. 生成新 RunId，例如 `dynamic_fusion_state_reconcile_20260818_v1`。
4. 只新增状态报告，不改旧实验目录。

交付物：

- `experiments/dynamic_fusion/reconciliation/<RunId>/state.json`
- `experiments/dynamic_fusion/reconciliation/<RunId>/state.md`
- `hashes.sha256`

验收标准：

- 明确列出 completed/partial/invalid/optional；
- 所有结论都有证据路径；
- 不把目录存在当成实验完成；
- 不启动训练或 GPU。

## S1. 修复冻结验证工具

目的：让 `--verify` 真正验证既有清单，而不是刷新清单。

实现要求：

1. `--create` 与 `--verify` 必须互斥；
2. `--verify` 首先读取既有 manifest，绝不写入；
3. 验证 code/checkpoint/manifest/evaluator/feature caches/baseline caches；
4. 缺失、size 不同、hash 不同、额外未声明输入都要进入报告；
5. 验证失败返回非零退出码；
6. 加入测试：修改临时副本后验证必须失败，原 manifest mtime/hash 不变。

交付物：

- 修复后的脚本；
- `tests/test_freeze_a1_mpdd.py`；
- 独立 `freeze_verification.json/md`；
- 验证前后 manifest hash 与 mtime。

验收标准：

- 正常状态全量验证通过；
- 故意篡改临时副本时失败；
- `--verify` 前后冻结 manifest SHA256 完全相同；
- 不删除或覆盖任何旧输出。

## S2. 修正文档与数据集角色

目的：消除路径、角色和比较口径冲突。

必须修正：

- `a1_visa_20260817` -> `a1_visa_20260818`；
- VisA `holdout` -> `in_domain_frozen_validation`；
- MVTec -> `external_frozen_validation`；
- MPDD -> `development`；
- BTAD -> `external_frozen_validation_k1_only`；
- 更新阶段七当前报告，但保留旧阶段七历史文件；
- 所有表同时写清 baseline source。

交付物：

- 新的权威 `CURRENT_DYNAMIC_FUSION_STATUS.md`；
- 机器可读 `current_dynamic_fusion_status.json`；
- 文档链接检查报告。

验收标准：

- 文档中所有证据路径存在；
- JSON 和 Markdown 的状态一致；
- 不再出现同一阶段同时“未完成/已完成”；
- 不把 VisA 叫独立外部数据集。

## S3. 统一性能表与统计口径

目的：让论文表格能够经受“比较基线不一致”的审查。

每个数据集至少报告：

- feature-level DINO-only；
- CLIP-only；
- A1 concat；
- legacy DINO score baseline（仅在存在时）；
- V3.3-clean fixed score fusion；
- visual fallback。

MPDD 必须同时显示：

- A1 concat vs legacy DINO：`+0.0486`；
- feature-DINO-only vs legacy DINO：`+0.0227`；
- A1 concat vs matched feature-DINO-only：`+0.0258`。

统计要求：

- seed/shot 平均与标准差；
- 逐类 delta；
- 正收益配置数；
- 最坏类别退化；
- 不把 9 配置当作 9 个独立测试集做显著性结论；
- 如做 bootstrap，以图像或类别为重采样单位，并说明相关性限制。

交付物：

- `main_results.csv/json/md`；
- `per_category_results.csv`；
- `metric_definition.md`；
- 生成表格的脚本与命令。

验收标准：

- 所有汇总数字可从 per-config report 自动重算；
- 重算误差小于 `1e-6`；
- baseline source 每行显式存在；
- 不同协议的 AdaptCLIP/ReMP-AD 单独分表。

## S4. 形成正式方法包

目的：将 A1 从实验候选收敛为可复现方法。

交付物：

- 修订后的 `METHOD_CARD.md`；
- 修订后的 `REPRODUCE.md`；
- `freeze_manifest_v2.json` 或独立 verification manifest；
- 方法伪代码；
- 输入输出 schema；
- 环境与显存/时间/磁盘统计；
- failure cases 与限制章节。

验收标准：

- 一条 validate-only 命令可检查全部输入；
- 一条 CPU evaluation 命令可从冻结缓存重算报告；
- 五个泄漏字段全 false；
- sample ID、shape、类别数、NaN/Inf 全部检查；
- 方法名称不再误称动态路由。

## S5. Git 与证据归档

目的：防止当前大量未提交成果丢失或被后续 AI 覆盖。

要求：

1. 先审计 3 个 tracked 修改与所有 untracked 顶层项；
2. 数据集、checkpoint、大型缓存不得误提交；
3. 代码、测试、轻量 JSON/Markdown 报告按逻辑分批提交；
4. 每个提交记录对应 RunId 和关键 hash；
5. 不使用 `git reset --hard` 或覆盖旧结果。

验收标准：

- `git status` 中剩余项都有明确原因；
- 新增源码、测试和权威文档进入版本历史；
- 数据/cache 排除规则可解释；
- 提交后重新执行只读冻结验证。

## S6. 论文交付

建议论文核心结构：

1. 少样本工业异常检测中的跨模态分数不可比问题；
2. 动态路由为何容易因置信度代理、测试校准和类别规则失败；
3. 参考条件的正常 memory bank 特征融合；
4. 无泄漏协议与冻结流程；
5. MPDD development、MVTec external、VisA in-domain、BTAD K1 的分角色结果；
6. matched feature baseline 消融；
7. 失败路线与限制。

论文完成标准：

- 不引用旧 V3.3 泄漏结果作为主结果；
- 不把 VisA 写成完全独立 holdout；
- 不把 A1 写成逐像素动态路由；
- 同时报告有利与不利类别；
- 所有主要数字能追溯到冻结报告和脚本。

---

## 8. 路线 D：只有必须保留“动态”创新时才启动

路线 D 不是默认待办，而是条件分支。第一轮仍然只允许 CPU 与现有缓存。

## D0. 重新定义 A1 之上的动态 headroom

目的：判断动态机制是否还能在 A1 之上获得有意义收益，而不是继续拿 DINO 当弱基线制造巨大 Oracle。

必须计算的 Oracle：

1. A1 vs feature-DINO-only 的逐像素 Oracle；
2. A1 vs CLIP-only 的逐像素 Oracle；
3. A1 vs V3.3-clean 的逐区域 Oracle；
4. A1 对少数退化类别的 rescue-only Oracle；
5. Oracle 只用于 development 诊断，不进入路由输入。

Gate D0：只有同时满足以下条件才继续：

- 相对 A1 的 mean Pixel AP Oracle headroom `>= +0.015`；
- 至少 4/6 MPDD 类有 `>= +0.005` headroom；
- headroom 不是只来自一个类别；
- MVTec 退化类别中存在相似的无标签可观测现象，但不得用 MVTec 指标调参；
- Oracle 计算代码与融合代码完全隔离。

若未通过：停止全部新动态设计，回到路线 S。

## D1. 可靠性可预测性 Gate

目的：证明无标签特征能预测“何时 A1 应该被修正”。没有这一步，不实现新 router。

候选无标签特征分为三类：

1. **参考稳定性**：正常参考及固定增强下的特征方向方差、neighbor 一致性、prompt/view 一致性。
2. **测试时稳定性**：固定无随机增强下的区域响应一致性、跨尺度一致性、局部邻域一致性。
3. **跨模态关系**：DINO/CLIP 排名一致性、候选区域 IoU、局部 disagreement，但不能把 disagreement 直接当作文本正确。

必须明确区分：

- anomaly evidence：离正常参考有多远；
- reliability：该分支的证据在允许的扰动下是否稳定。

禁止再采用“离正常更近的分支更可靠”这种 V3.4 假设。

评估协议：

- MPDD 做 leave-one-category-out；
- 在 5 类上确定一个简单映射，在第 6 类只评估；
- 报告 AUROC/AP、校准误差和 risk-coverage，用于预测“A1 修正是否有益”；
- MVTec/VisA 不参与特征选择或阈值选择。

Gate D1：

- held-out benefit prediction AP 明显高于阳性基率；
- 6 个 held-out 类中至少 4 类方向一致；
- 特征置乱后性能回到随机水平；
- 不读取测试标签/掩码生成任何输入特征。

若未通过：停止路线 D。

## D2. 唯一允许的新候选：A1-R 有界可靠性残差

建议名称：A1-R（A1 with Reference-conditioned Reliability Residual）。

设计原则：

- A1 始终是默认输出；
- 新模块只在可靠性 Gate 通过的局部区域进行有界修正；
- 不能全图重加权；
- 不能根据类别测试 AP 调整规则；
- 文本或动态模块缺失时精确回退 A1；
- 修正幅度有预注册上限。

概念流程：

```text
A1 anomaly map
  -> A1 candidate regions
  -> reference/test stability features
  -> benefit predictor / deterministic reliability rule
  -> bounded residual cap
  -> A1 fallback
```

候选形式只能预注册一个：

```text
s_final = s_A1 + gate * clip(residual, -cap_down, cap_up)
```

其中：

- `gate` 只能由允许的无标签可靠性特征产生；
- `residual` 是 A1 与单分支/安全分数的局部差异；
- `cap_up`、`cap_down` 在开发前固定；
- 默认优先使用单向或极小负向修正，避免破坏 A1 强区域。

不允许同时实现十几个公式再按测试结果选最好者。

## D3. MPDD CPU Gate

对照必须包括：

- feature-DINO-only；
- CLIP-only；
- A1 frozen；
- V3.3-clean；
- A1-R；
- A1 fallback；
- Oracle（仅诊断列）。

主要 Gate：

- A1-R 相对 A1 mean Pixel AP `>= +0.005`；
- 至少 4/6 类正；
- 最大单类回退不得小于 `-0.01`；
- AUPRO 不整体下降；
- 3 seeds × 1/2/4-shot 中至少 7/9 配置非负；
- 所有泄漏、对齐、NaN/Inf、确定性测试通过；
- 去掉 reliability features 后收益消失，证明不是无关后处理。

任一硬条件失败：A1-R 归档，禁止补大 GPU 矩阵。

## D4. 冻结与一次外部验证

只有 D3 通过才能：

1. 冻结代码、配置、cap、特征定义、checkpoint、manifest 和 evaluator；
2. 生成新的只读 freeze manifest；
3. 在 MVTec 上执行一次冻结验证；
4. VisA 只作为 in-domain 验证；
5. 不根据验证结果回头调参。

外部通过标准：

- MVTec 相对 A1 平均 Pixel AP 不低于 0；
- 至少 9/15 类不退化或退化在预设容忍内；
- 不出现单类灾难性回退；
- 若只在 MPDD 提升而 MVTec 不提升，论文中只能称 development exploration。

---

## 9. 明确停止的路线

后续 AI 不得继续以下工作，除非用户重新明确授权并说明新证据：

- 旧 V3.3 的更多权重网格；
- V3.4 原始 uncertainty/normal-reference/agreement 路由扩展；
- V3.5 图像级 gate；
- 手写缺陷词 prompt 列表扩展；
- A2 attention、A2b/CCA、A3 shared subspace；
- 按类别测试 AP 选择权重；
- 为凑齐矩阵而启动无科学必要性的 GPU；
- 在 VisA/MVTec 最终结果上继续调冻结配置。

---

## 10. 每个阶段的通用交付标准

每个 Run 必须保存：

- RunId、开始/结束时间、机器、命令和完整配置；
- Git commit 与 dirty diff 摘要；
- 数据集角色、类别、seed、shot；
- normal-reference manifest 与 SHA256；
- checkpoint/code/cache/evaluator SHA256；
- sample ID、shape、类别数、样本数；
- NaN/Inf、重复、缺失、错位检查；
- 五个泄漏字段；
- 逐类指标、平均指标、最坏退化；
- stdout/stderr；
- `audit.json`、`audit.md`；
- 明确结论：`passed`、`failed`、`development_only`、`invalid` 或 `optional`。

机器可读状态要求：

```json
{
  "status": "passed|failed|invalid|development_only|optional",
  "gate_passed": true,
  "paper_eligible": true,
  "dataset_role": "development|external_frozen_validation|in_domain_frozen_validation",
  "leakage_flags": {
    "test_predictions_used": false,
    "test_labels_used": false,
    "test_masks_used": false,
    "test_dataset_statistics_used": false,
    "test_normal_selection_used": false
  }
}
```

若 `gate_passed=false`，外层 `status` 不得写 `passed`。

---

## 11. 后续 AI 接手后的第一轮动作

第一轮只允许完成以下任务：

1. 阅读本文件与 `docs/DYNAMIC_FUSION_NEXT_STEPS.md`；
2. 读取 freeze manifest、A1 matrix/audit、VisA/MVTec summary/audit；
3. 检查 Git、活动进程、GPU 和队列，但不启动 GPU；
4. 创建 S0 状态对账 RunId；
5. 修复只读 freeze verifier 并增加测试；
6. 修正文档路径、角色与 baseline 口径；
7. 生成统一状态报告；
8. 向用户报告路线 S 是否已具备收敛条件，以及路线 D 的 D0 是否有必要。

第一轮不得：

- 新建 V3.6/V4 代码；
- 运行新训练；
- 修改冻结配置；
- 用 MVTec/VisA 指标选择新规则；
- 覆盖任何 2026-08-17/18 旧产物。

---

## 12. 项目最终完成定义

动态融合主线只有在以下条件全部满足时才算真正完成：

1. A1 方法名称、协议和实现一致；
2. 旧 V3.3/V3.4/V3.5 全部有明确 invalid/development-only 标记；
3. 冻结验证工具为严格只读，且全量 hash 验证通过；
4. MPDD/MVTec/VisA/BTAD 的数据集角色准确；
5. matched feature baseline 与 legacy baseline 分开报告；
6. 统一结果表能自动重算，误差小于 `1e-6`；
7. Git 中源码、测试和权威文档已归档；
8. REPRODUCE/METHOD_CARD/限制章节完整；
9. 若路线 D 未通过，失败被正式归档且停止扩展；
10. 若路线 D 通过，必须在冻结后 MVTec 验证中不劣于 A1，才可称动态改进。

达到这些条件后，推荐停止继续增加融合版本，将资源转向论文、复现包、效率数据和图表。

