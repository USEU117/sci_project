# 18. TCRR 区域重排实验：完整结果、当前结论与下一算法路线

日期：2026-09-03  
性质：算法探索与论文证据审计；本文档覆盖 v8，取代“任务书 17 中 TCRR 尚未开发”的旧状态。

## 一句话结论

我们找到了一条真实、可重复的互补信号：**AnomalyCLIP 文本热图能够判断 A1/PatchCore 提出的候选区域是否更像真实缺陷**。它在 MPDD 三个 seed、三个 shot 上能把 Pixel-AP 平均提高约 3.2%–3.6%。但是，当前“按文本分数同时放大或压低候选区域”的固定公式在 BTAD 和 MVTec 都失败，因此 v8 还不能作为论文主方法，也不能声称跨数据集有效。

## 方法到底做了什么

1. A1 仍是主干：它根据少量正常参考图生成异常图。
2. 在 A1 图的最高 5% 区域中做八连通分量，得到候选缺陷区域。
3. 在每个候选区域内读取 AnomalyCLIP 显式文本异常图的第 90 百分位分数。
4. 文本分数只改变 A1 已提出区域的排序，不能凭空创建新区域。
5. 固定倍率为 `exp(log(1.5) × (2p−1))`，范围约为 0.667–1.5。

这个结构暂命名为 **Topology-Constrained Region Re-Ranking（TCRR，拓扑约束区域重排）**。名称可以继续用作内部代号，但在文献检索完成前不要把名字或机制写成“首次提出”。

## 实验链条与结果

### R0：区域信息价值

- 共 23,866 个候选区域，其中正区域 3,925、负区域 19,941。
- A1 区域 AP 为 0.2135；文本 P90 区域 AP 为 0.3832，提升 +0.1697。
- 随机交换图像后的文本 AP 为 0.1952，真实文本比随机交换高 +0.1880。
- 原始 R0 因提升集中度超过预设线而判 FAIL；这条结果没有被篡改。

### R0b：更严格的同图空间对照

- 真实文本区域 AP 0.3832；同图旋转 180° 为 0.2102；同图半幅平移为 0.1318。
- 真实空间对应比最强错位对照高 +0.1730。
- 仅保留 q=0.95 时，相对 A1 仍提升 +0.1593，5/6 类为正。
- 去掉贡献最大的 tubes 后仍提升 +0.0724。
- 1%、10%、25% 三种区域重叠定义下均保持明显正增益。

解释：提升不是只靠“这张图属于哪一类”或文本图整体偏高，而确实依赖缺陷与文本热点的空间对应。

### R1：MPDD seed0 的真实像素图重排

- 平均 Pixel-AP：+0.032147。
- 平均 Pixel-AUROC：+0.002944。
- 三个 shot 全部为正，15/18 个类别×shot 配置为正。
- 旋转、平移对照均下降；真实方法比最佳对照高 +0.058855。
- `metal_plate` 平均 −0.013656，是稳定风险类。

### R2：MPDD seed1/2 确认

- 六个 seed×shot 组合全部为正。
- 平均 Pixel-AP：+0.036330；平均 Pixel-AUROC：+0.002645。
- 29/36 个类别配置为正。
- 类别聚类 bootstrap 95% CI：[+0.002556, +0.088785]。
- 各类平均变化：black +0.0177、brown +0.0037、white +0.0287、connector +0.1609、metal_plate −0.0135、tubes +0.0205。

结论：TCRR 在 MPDD 内部不是偶然波动，机制与像素收益均能跨 seed 重复。

### R3：一次性外部验证

算法、阈值和倍率在看外部结果前已冻结，BTAD 与 MVTec 之间没有改参数。

- BTAD：平均 Pixel-AP −0.006951，FAIL。01/02 分别 +0.0111/+0.0098，但 03 为 −0.0418。
- MVTec：平均 Pixel-AP −0.005820，FAIL。cable、capsule、screw、tile、zipper 为正，其余多类为负；hazelnut 为 −0.0828。
- 两个数据集的 Pixel-AUROC 均有极小正变化，且正确空间文本仍显著优于错位对照。

直白解释：**文本分支确实看到了有用位置，但我们目前把“有用位置”转换成最终分数的办法不安全。** 它会在部分材料、纹理或形状上把 A1 原本正确的排序打乱。

## 为什么会失败

最可能的原因是当前文本图对每张测试图单独做 1%–99% 归一化。即使一张图的文本分支整体不可靠，归一化后也必然出现接近 0 和 1 的区域；随后双向倍率会强行放大或压低 A1 区域。MPDD 上这种相对排序恰好有效，但外部域的纹理与形状改变后，错误置信被放大。

此外，简单的“视觉图 + 文本图区域细化”与现有 CLIP 异常定位研究存在相邻关系。[CLIP-ADA](https://arxiv.org/abs/2403.09493) 已包含异常区域细化，[PALADIN](https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html) 也直接把自监督视觉 patch 与文本原型对齐；[FastRef](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html) 则说明 few-shot 原型改进本身也是活跃路线。因此，论文创新不能只写成“双分支融合”或“用文本细化热图”，必须突出可验证的正常参考校准、安全回退和拓扑约束，并用消融证明这些部件不可替代。当前检索未发现与 TCRR 完全相同的固定实现，但这不是穷尽式新颖性证明。

## 下一版：NC-SafeTCRR（建议 v9）

不再在外部结果上微调 v8。新算法应回到 MPDD 开发集，重新预注册，然后使用新的独立数据作最终验证。

核心改动：

1. **正常参考校准**：对每个 seed/shot 的 K 张正常参考图也生成文本热图；用正常像素的 median/MAD 建立文本基线，不再对每张测试图强制归一化。
2. **只增益、不压低**：只有候选区域的校准文本证据显著高于正常参考（例如 z>3）才提高 A1 分数；证据不足时保持 A1 原值。
3. **可靠性回退**：若正常参考上的文本热点过多、不同参考间不一致或与 A1 正常纹理高度冲突，整个类别/配置直接退回 A1 identity。
4. **拓扑约束不变**：文本仍不能创建候选区域，只能确认 A1 已提出的连通区域。
5. **风险类专项分析**：metal_plate、BTAD-03、hazelnut 只用于解释失败模式，不能据其标签定制规则。

建议验收顺序：

- v9-R0：MPDD seed0，要求 Pixel-AP 平均至少 +0.003，AUROC 不低于 −0.001，至少 4/6 类为正，最差类不低于 −0.01。
- v9-R1：MPDD seed1/2，要求六个 seed-shot 至少 5 个为正，类别聚类 bootstrap 下界大于 0。
- v9-R2：只在另一个事先冻结、未用于设计的验证集上运行一次。BTAD/MVTec 已被 v8 观察，不能再作为 v9 的无偏最终证据。

## 论文当前应该怎样写

- A1 仍是当前可正式报告的主方法。
- v8 可放在“探索性分析/失败经验/消融”中，说明区域级语义互补在 MPDD 成立，但朴素分数转换无法跨域。
- 不能写“TCRR 在多个数据集提升性能”，因为 BTAD 和 MVTec 均为负。
- 在 v9 获得新的独立验证前，不要把 NC-SafeTCRR 写入摘要或 Contributions。

## 可复核文件

- `configs/innovation_v8_tcrr_probe/`
- `src/industrial_ad/innovation_v8_tcrr_probe/`
- `scripts/innovation_v8_tcrr_probe/`
- `tests/innovation_v8_tcrr_probe/`
- `experiments/dynamic_fusion/innovation_v8_tcrr_probe/`
- `outputs/dynamic_fusion/innovation_v8_tcrr_probe/`（大缓存，gitignored）

主要决策文件：

- `R0_region_value/R0_DECISION.md`
- `R0b_spatial_robustness/R0B_DECISION.md`
- `R1_minimal_reranker/R1_DECISION.md`
- `R2_seed_confirmation/R2_DECISION.md`
- `R3_btad/R3_DECISION.md`
- `R3_mvtec/R3_DECISION.md`
- `R3_OVERALL_DECISION.md`

## v9-R0 已执行补记（2026-09-03）

上述 NC-SafeTCRR 的固定小试验已经完成，不再是“尚未执行”的计划：

- MPDD seed0 三个 shot 平均 Pixel-AP +0.023270，Pixel-AUROC +0.002127。
- 5/6 类平均为正；真实空间方法比最佳错位对照高 +0.032433。
- 仅增加 3316/8277 个候选区域（40.1%），其余保持 A1 原值。
- `metal_plate` 仍下降 −0.018230，超过预注册的 −0.01 安全线，因此总门控 **FAIL / ARCHIVE**。

这说明正常参考校准和 identity 回退能保留大部分有效增益，但“一个类别共用一个全局 median/MAD”仍不能可靠识别金属板纹理上的错误文本热点。不得事后把安全线放宽，也不得只删除 metal_plate 后宣布成功。下一次若继续，应研究完全由正常参考决定的区域级可靠性估计（例如参考增强一致性、位置条件基线或 conformal 阈值），并重新准备未被用于设计的验证集。

结果文件：`experiments/dynamic_fusion/innovation_v9_ncsafe_tcrr/R0_mpdd_seed0/R0_DECISION.md`。
