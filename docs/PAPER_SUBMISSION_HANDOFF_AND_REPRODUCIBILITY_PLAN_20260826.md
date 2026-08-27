# SCI 四区投稿：A1 复现包与论文收尾总交接（2026-08-26）

## 0. 文档地位

这是下一位 AI 的**唯一投稿收尾入口**。项目历史状态以 `docs/CURRENT_DYNAMIC_FUSION_STATUS.md` 为事实来源；动态路线设计细节见 `docs/DYNAMIC_FUSION_NEXT_STEPS.md`，但不得据此自动重启实验。

任务不是继续寻找更复杂算法，而是把已有 A1 结果变成可投稿、可复核、口径诚实的 SCI 四区论文与精简复现包。

### 2026-08-27 P0 验收覆盖说明

最新验收以 `docs/submission_reproducibility_20260826/P0_ACCEPTANCE_REVIEW_20260827.md` 为准：四数据集数值重建和 P0 技术复现包均已通过，P0A–P0I 全部为 true，`research_rebuild_complete=true`、`submission_repro_package_complete=true`。不需要再次导出四数据集特征。数据集 URL 已补齐并纠正 MPDD 身份；自研代码 LICENSE 已于 2026-08-27 选定为 **MIT**（根 `LICENSE`），仍须确认 MPDD/BTAD 未明确展示的许可条款。P1 全部完成（`p1_acceptance.json` `p1_complete=true`）；P1-C 效率以 smoke 实测为准，未单列预热端到端 benchmark 与峰值 RAM。

## 1. 最终研究结论与论文边界

### 1.1 最终正结果

A1 使用两个互补的视觉 patch 表征：

1. DINOv2 `dinov2_vitb14`；
2. AnomalyCLIP `ViT-L/14@336px` 的 image tower patch tokens。

两分支分别按 patch 做 L2 归一化，CLIP grid 双线性对齐至 DINO grid，按 `w=0.5/0.5` 拼接，再整体 L2 归一化。仅用 K 张正常参考图的 patch 建立 `faiss.IndexFlatL2`、`k=1` 正常记忆库，以最近邻平方 L2 距离除以 2 形成 patch 异常分数并上采样为像素异常图。没有学习式路由，也不依据测试图改变权重。

相对 matched feature-DINO-only KNN 的纯拼接 Pixel-AP 增益：

| 数据集 | 角色 | 平均增益 | 配置覆盖 |
|---|---|---:|---:|
| MPDD | development | +0.025830 | 9/9 |
| BTAD | external frozen validation | +0.024895 | 9/9 |
| VisA | in-domain frozen validation | +0.052353 | 9/9 |
| MVTec AD | external frozen validation | +0.031962 | 9/9 |

这里的 9/9 是 `3 seeds × 1/2/4-shot` 的参考采样配置，不是 9 个独立数据集。

### 1.2 必须使用的论文表述

建议英文方法名：`Reference-Conditioned Dual-Encoder Patch Fusion with a Normal Memory Bank`。

推荐中文概括：`面向少样本工业异常检测的参考条件双编码器视觉特征融合与正常记忆库方法`。

必须写作：

- dual-encoder visual patch fusion；
- heterogeneous pretrained visual representations；
- training-free / normal-reference-conditioned inference（按论文具体定义谨慎用词）；
- fixed equal-weight fusion；
- leakage-safe development and frozen validation。

禁止写作：

- 动态融合方法已经成功；
- 视觉—文本特征在 A1 推理时显式融合；
- multimodal/text branch 是 A1 的实际输入；
- SOTA；
- VisA 是独立外部泛化验证。

G0 语义审计已证明 A1 的 AnomalyCLIP 导出只调用 image encoder；prompt learner 虽被加载，但没有进入导出的 A1 patch 表征。动态路线和更强视觉替代路线均按预注册门禁失败或停止，可作为负结果讨论。

### 1.3 已关闭的维度错误

旧 `METHOD_CARD.md` 曾同时写 DINO 768 维、CLIP 768 维和 concat 1152 维。P0 smoke 已实测 DINO 768、AnomalyCLIP image tower 768、concat **1536**；投稿版 `METHOD_SPEC_V2.md` 已修正。后续正文、图和表只允许使用 1536，不再重复运行该 smoke，除非代码或权重发生变化。

## 2. 为什么不需要全部重跑

以下内容已有版本化 JSON/CSV/Markdown 与 Git 历史，除非审计发现 hash 或计算错误，不应重跑：

- PatchCore、WinCLIP、PromptAD、AnomalyDINO 等历史基线全矩阵；
- V3.3/V3.4/V4 动态路由探索；
- SubspaceAD V1/V2 gate；
- Route-D oracle/predictability；
- 权重搜索与动态对固定融合的开发期比较。

需要重建的是“投稿复现链”而非“整个研究史”。2026-08-20 清理过 `outputs/dynamic_fusion` 等大缓存，因此旧 `freeze_verification.json` 的 229 项 `all_ok=true` 是当时的历史证据；当前运行 `freeze_a1_mpdd.py --verify` 会因缓存缺失失败，这是预期状态。

## 3. P0 已完成（2026-08-27 最终复验）

本次已完成：

1. 新建 `scripts/audit_submission_repro_package.py`，机器可读地区分版本化证据、数据、权重、环境、缓存和测试门禁。
2. 新建 `docs/submission_reproducibility_20260826/` 作为投稿复现入口。
3. 生成当前 `P0_LIVE_AUDIT.json` 与核心文件 SHA256 清单。
4. 明确“无需全重跑、但必须最小重建”的边界。
5. 把 SCI 目标从一区修正为四区：不再把新增强算法或 MVTec AD 2 设为硬性投稿前置条件。
6. 历史 CPU 回归证据为 `81 passed`；2026-08-27 当前复验扩大为 `python -m pytest -q tests`，`122 passed in 5.80s`。
7. 324 个 compact maps、包内 CPU 重算脚本、独立 rebuild manifest、1536 维方法说明和 source commit 均已入包。
8. `--verify-only` 为 324/324；MPDD s0/K1 与 MVTec s1/K4 从 maps + mask 完整重算均通过。
9. `SHA256SUMS` 有 447 条受校验记录且全部通过；包内共 448 个文件（包含清单自身）。

P0 技术门禁已关闭。第 4 节保留为复现协议，不是待执行清单。

## 4. P0 执行路线（P0-1 至 P0-4 均已于 2026-08-27 完成）

### Gate P0-1：环境与只读输入

目标：证明脚本、数据、split、权重可用，不运行全量推理。

步骤：

1. 运行 P0 审计脚本并阅读 `blockers`，不要只看退出码。
2. 恢复或确认 MVTec AD 本体；目录存在但文件数为 0 也视为缺失。
3. CPU 测试可直接使用已验证的 `.venv-anomalyclip`；无需只为 pytest 改动 `.venv-patchcore`。若以后必须在 patchcore 环境运行，安装前后记录 `pip freeze`，且不要无约束升级 torch/faiss/numpy。
4. 对 DINO/CLIP 导出脚本分别执行 `--validate-only`（MPDD seed0/K1）。
5. 运行 CPU 单元测试；优先范围：

```powershell
.venv-anomalyclip\Scripts\python.exe -m pytest `
  tests\test_freeze_a1_mpdd.py tests\test_v4_contracts.py `
  tests\test_dynamic_fusion.py tests\test_dynamic_fusion_v2.py `
  tests\test_dynamic_fusion_v3.py -q
```

验收：数据/权重/split 全存在；validate-only 通过；测试全过；把实际命令、解释器、版本、Git SHA 和日志保存到新的 `submission_repro_<date>` 目录。

### Gate P0-2：一类一图端到端 smoke

目标：以最小 GPU 成本验证“数据 → 两分支特征 → 对齐/拼接 → KNN → anomaly map → evaluator”。

要求：

- 只选 MPDD 一个类别、seed0、K=1，并限制为一张正常参考和一张测试图；若现有导出脚本不支持样本上限，新增 `--categories`、`--max-test-images` 的非破坏性参数，默认行为必须不变。
- 保存 DINO/CLIP 原始 shape、grid、dtype、sample_id、参考 ID。
- 现场确认 concat 维度，解决 1152/1536 冲突。
- 输出一张 anomaly map 和对应指标 smoke；smoke 指标不得进入论文表。
- 记录峰值显存、墙钟时间、CPU RAM、文件大小。

验收：两个分支 sample_id 完全一致；grid 对齐明确；无 NaN/Inf；五个泄漏 flag 全 false；重复运行的数值在声明容差内一致。

### Gate P0-3：四数据集精简最终证据

目标：恢复论文真正需要的可核验证据，不恢复 324 GB 研究缓存。

最小建议保留：

- 每数据集、每 `(seed, shot)` 的 A1 concat 与 matched DINO-only 的逐图 `image_score`；
- 压缩后的像素 anomaly map，或能从保留 patch-distance map 无损/声明精度地重建指标的最小表示；
- sample_id、label、mask 相对路径、reference IDs、dataset/category/shot/seed；
- 每配置和逐类别指标 JSON/CSV；
- 配置、代码、split、checkpoint、产物 SHA256；
- 运行日志、耗时、峰值 VRAM/RAM、磁盘占用。

执行原则：

1. 只重跑 A1 concat 与 matched feature-DINO-only，不重跑所有历史方法。
2. 先 MPDD seed0/K1，全链通过后再扩至 MPDD 9 配置。
3. 再按冻结配置运行 BTAD、MVTec、VisA；任何验证集结果都不得反向修改 w、归一化、PCA、阈值或规则。
4. K2/K4 仅重算参考特征；测试特征应按数据集复用，避免重复存储。
5. 每完成一个数据集立即生成审计报告和 SHA256，不等四个全部结束。

验收数值采用“重算接近历史结果”而非逐位相同：

- MPDD concat vs matched DINO-only ΔPixel-AP：`+0.025830`；
- BTAD：`+0.024895`；
- VisA：`+0.052353`；
- MVTec：`+0.031962`。

初始容差建议为绝对误差 `5e-4`。超过容差时先检查依赖、图像排序、插值、维度、`distance/2` 和 evaluator；不得为了贴合历史值而调权重。

### Gate P0-4：真正可发布的 compact package

2026-08-27 状态：**P0 技术复现 PASS，P1 全部完成**。真实 compact predictions、包内重算脚本、独立 rebuild manifest、投稿版 1536 维方法说明与源码提交均已完成。许可证索引已建立，自研代码 LICENSE 已于 2026-08-27 选定为 **MIT**（根 `LICENSE`，Copyright 2026 LiYuening）并随包分发；MPDD/BTAD 数据条款的逐项复核仍是公开发布前人工事项；不要把技术 Gate 通过误写成法律许可已经完成。

建议目录：

```text
submission_repro_<date>/
├── README.md
├── LICENSES_AND_DATA.md
├── environment/
│   ├── system.txt
│   ├── patchcore_pip_freeze.txt
│   └── anomalyclip_pip_freeze.txt
├── config/
│   ├── frozen_a1.json
│   └── split_manifest_hashes.json
├── evidence/
│   ├── paper_tables/
│   ├── per_config/
│   ├── per_category/
│   └── negative_results_index.md
├── predictions_compact/
├── logs/
├── manifest.json
└── SHA256SUMS
```

验收：在一个新的目录/机器上，按 README 能完成：

1. CPU-only：从 compact predictions 重算论文全部表格；
2. GPU smoke：从原图重建一个类别的两分支输出和 A1 anomaly map；
3. `audit_submission_repro_package.py` 全门禁通过；
4. 包中不含不能再分发的数据集、第三方权重或许可证禁止内容。

## 5. P1：论文实验与统计收尾（全部完成，门禁通过）

### 已完成

1. category/image bootstrap CI 与 dataset×shot 三 seed mean±std。
2. worst/negative categories 和逐图失败 sample IDs。
3. 11方法公平性协议表；已纠正 AnomalyDINO 与 WinCLIP+ 的实际协议。
4. memory-bank 与包大小统计；原 patch 计数错误已纠正。

### 写论文前仍必做

1. P1-C 最小 benchmark：预热模型后重复测量端到端 latency/throughput、峰值 VRAM 和峰值 RAM；报告硬件、batch=1、类别、shot、重复次数与计时边界。
2. ✅ 已聚合36份 complete-metrics reports：`p1_e_complete_metrics.*` 含 image AUROC/AP/F1-max 与 pixel AUROC/AP/AUPRO、source hash和P0对账。论文须注明BTAD Image-AP/F1下降，四数据集稳健正结论只针对Pixel-AP。
3. 从 compact maps + 本地合法原图生成最终 A1 成功/失败定性图；固定选择规则，只作解释，不参与调参。
4. 将现有 MVTec/VisA baseline 矩阵整理成最终对照表；MPDD/BTAD 若没有同协议完整外部基线，必须明确为空缺，不得暗示已有。
5. 自研代码 LICENSE 已选定为 **MIT**（2026-08-27，根 `LICENSE` 随包分发）；仍须确认 MPDD/BTAD 数据条款；完成前不公开发布代码包。

### 四区投稿的“建议补做”，不是硬门槛

- 一个独立新数据集（如 MVTec AD 2）可增强说服力，但不应在现阶段自动启动；先完成 P0/P1，再按期刊要求、数据可得性和时间预算决定。
- 可以补一张 normalization/alignment 的小消融，但只有在已有缓存可低成本重算时做。
- 不再设计动态路由、新 backbone 或新训练模块。

## 6. P2：论文重写

旧英文稿围绕 `When Should Visual and Language Evidence Be Fused?` 和 uncertainty routing，已与最终方法不符，必须整体重写，不能局部修补。

推荐结构：

1. Introduction：工业缺陷少样本问题 → 正常记忆库 → 异构预训练视觉表征互补 → 严格防泄漏固定融合的研究空缺 → 四数据集结果与失败边界。
2. Related Work：visual memory-bank、vision-language anomaly detection、training-free/few-shot 方法；严格区分 zero-shot、target-normal tuning、transductive/test-time adaptation。
3. Method：两个 image encoder、grid 对齐、双重 L2、等权 concat、KNN、distance/2、指标；明确没有文本推理和动态路由。
4. Experiments：协议、数据角色、matched controls、主表、统计、效率、消融、失败案例。
5. Discussion：为什么简单融合有效；为什么 CLIP-only 弱但能提供互补；为什么动态/同 backbone 子空间路线失败；VisA 域内和非 SOTA 限制。
6. Conclusion：强调稳健性、审计和边界，不夸大复杂性。

Introduction/Related Work 的本地资料入口为 `docs/introduction_research_20260825/INTRODUCTION_LITERATURE_MASTER_20260826.md`。该目录当前已纳入 Git；写稿前仍应人工复核 BibTeX 元数据与原文对应关系。

## 7. P3：SCI 四区选刊与投稿

四区目标使“再造一个强算法”不再是合理的硬要求，但不降低可复现性、对照公平性和表述准确性要求。

选刊时逐本核验当年 JCR/中科院分区、scope、版面费、篇幅、开放获取政策、审稿周期和近两年是否接收工业视觉异常检测。分区会变化，不在本文档固化具体期刊名单。优先选择接受应用型机器视觉、质量检测、工业 AI、可复现经验研究的期刊。

投稿前最终门禁：

- P0 compact package 全通过；
- P1 统计、效率、公平性和失败案例齐全；
- 方法维度冲突已由 smoke 实测解决；
- 全文无 dynamic/multimodal/text-fusion 误称；
- 表格数字可由 compact predictions 一键重算；
- 数据和第三方权重没有被非法打包；
- 稿件、补充材料、代码 README、cover letter 的贡献口径一致。

## 8. 下一位 AI 的执行顺序

严格按以下顺序，不要跳步：

1. 阅读本文、`CURRENT_DYNAMIC_FUSION_STATUS.md`、P0 最终验收与复现包 README。
2. 只读复核 P0 审计；若源码、maps、数据或权重均未改变，不重建四数据集特征。
3. P1 已全部完成（`p1_acceptance.json` `p1_complete=true`）；如论文需要稳态吞吐/峰值 RAM 才补最小预热 benchmark（非门禁项）。
4. 使用已生成的 `p1_e_complete_metrics.*`；不得把Pixel-AP四数据集提升扩写为所有图像/像素指标全面提升。
5. 生成固定的成功/失败定性图及 source-ID 清单。
6. 自研代码 LICENSE 已选定为 **MIT**（2026-08-27，根 `LICENSE` 随包分发）；仍须确认 MPDD/BTAD 许可；公开发布前完成。
7. 按固定双视觉融合口径重写论文、图表和补充材料。
8. 实时检索 SCI 四区候选期刊并适配格式；审阅并提交本轮修正。

若任何 gate 失败：保存命令、日志、环境、hash 和失败原因，停止在该 gate；不得用旧缓存结论或验证集调参绕过。

## 9. 完成交付定义（Definition of Done）

只有同时满足以下条件才可向用户报告“投稿前实验完成”：

- A1 从原图到 anomaly map 的 smoke 可复现；
- 四数据集精简证据存在且 hash 固定；
- CPU 一键重算论文表；
- A1 vs matched DINO-only 的统计与效率完整；
- 所有数据角色、baseline source、泄漏 flag 清楚；
- 1152/1536 冲突已实测消除；
- 论文不宣称动态、文本融合或 SOTA；
- 复现包通过机器审计且许可证清单完成；
- 最终稿与目标 SCI 四区期刊格式一致。
