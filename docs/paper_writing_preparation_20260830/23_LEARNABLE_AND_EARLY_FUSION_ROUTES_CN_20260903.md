# 可学习模块与前期双分支融合：历史核查、候选方法与分级验收

日期：2026-09-03  
状态：研究建议；尚未形成已验证方法，不得写入论文结论  
基线：A1（DINOv2 与 CLIP-image 最终 patch 特征分别归一化、0.5/0.5 拼接、再次归一化、KNN）

---

## 1. 先给结论

1. **还能继续做，而且“多层/分支中段交互”是目前少数尚未被本项目真正完成验证的空白。**
2. **我们并不是完全没做过可学习融合。** NCPRA 已用两个瓶颈 adapter 在最终层做 DINO→CLIP、CLIP→DINO 的正常样本预测，但所有候选均退化；CASF 曾计划多层可学习融合头，却在 Wave 0 最终层探针后提前归档，多层导出与 P0/P1 训练没有启动。
3. **不建议直接把 Cross-Attention、Mamba、LoRA 或 MoE 塞进两个 backbone。** 这只能构成结构替换，且近期文献已非常拥挤。新意必须来自一个异常检测特有的问题：在正常条件下保持两分支互补，在疑似缺陷处允许非对称残差被暴露，而不是把差异强行对齐掉。
4. 最值得做的主路线暂命名为 **SCAIF：Support-Conditioned Asymmetric Interaction Fusion（支持集条件化非对称交互融合）**。先在缓存的多层特征上验证，再决定是否真正把 bridge 插入 backbone 中段。
5. 若论文必须保持“目标类别 1/2/4-shot、normal-only、完全零训练”，则不建议训练较大的前期融合模块；只能做支持集生成的有界门控或闭式低秩变换。若允许在源类别上 episodic 训练、目标类别不更新参数，SCAIF 才最有价值。

---

## 2. 当前 A1 到底属于哪一种融合

A1 已经不是最后 anomaly score 的简单相加，而是**最终 patch 表征层融合**：

\[
z = \operatorname{norm}\big[0.5\operatorname{norm}(d),\;0.5\operatorname{norm}(c)\big],
\qquad s(x)=\min_{r\in\mathcal M}\tfrac12\lVert z_x-z_r\rVert_2^2.
\]

因此准确表述应是：

- 已有：两个冻结视觉编码器的**末端特征融合**；
- 未有：中间层之间的信息交换；
- 未有：查询/支持集条件化的可变融合；
- 未有：让一个分支的中层特征改变另一个分支后续计算的真正 in-backbone fusion。

后续不能把“从 score fusion 改成 feature fusion”当作新贡献，因为 A1 已经是 feature fusion。真正可扩展的是**层、位置、方向和样本条件均可变的交互机制**。

---

## 3. 历史核查：哪些做过，哪些没有做过

| 路线 | 是否可学习 | 交互位置 | 是否多层 | 实际结果/状态 | 对新实验的约束 |
|---|---:|---|---:|---|---|
| A1 | 否 | 最终 patch 特征 concat | 否 | 当前基线 | 新方法必须超过它，而非只超过 score average |
| NCPRA | 是 | 最终层同位置 D↔C 预测 | 否 | 4 个候选全负，best mean ΔPixel-AP 约 `-0.0052` | 不得原样复活“双向 MLP 预测 + cycle” |
| AdaptCLIP 内部 adapter | 是 | CLIP 内部视觉/提示分支 | 有内部层 | 属于成熟分支本身 | 不能声称是 DINO–CLIP 融合创新 |
| CASF 计划 | 是 | 多层统计后的小融合头 | 计划是 | **未执行** Wave 1 多层导出和 Wave 2 P0/P1 | 说明多层可学习路线仍未被否定 |
| CASF Wave 0 | 是，小型 probe | 最终层 4 个统计量 | 否 | 仅 1/6 类有 headroom，提前归档 | 不能把它误写成“多层早期融合已经失败” |
| RCEC/CECW 类 | 否/闭式 | 配对 memory/条件距离 | 否 | 已做或已规划 | 新交互需胜过 shuffled correspondence 控制 |
| BC-MCR | 是 | support/context repair | 否 | 未证明训练模块的独立贡献 | 必须设置无训练、无 support、位置控制 |

### 3.1 NCPRA 为什么不等于本次建议

NCPRA 的核心是最终层、同一 patch 位置的双向确定性映射：

\[
g_{D\rightarrow C}:768\rightarrow r\rightarrow768,
\qquad
g_{C\rightarrow D}:768\rightarrow r\rightarrow768.
\]

它只用正常 reference patch 训练，并把跨分支预测误差作为 anomaly residual。它没有：

- 中间层信息；
- 邻域 token 交互；
- 支持集条件化门控；
- private/shared 表征分离；
- 对“不要抹除异常”的显式约束；
- 源类别 episodic 训练与类别外验证。

所以 NCPRA 的失败说明“最终层确定性相互回归”不值得继续调参，但没有否定多层、局部、残差式的前期交互。

### 3.2 CASF 到底做到了哪里

CASF 原计划导出 DINO 三层和 CLIP `[6,12,18,24]`，用分支距离、LOO z-score、分支分歧和局部对比构成多层输入，再训练小型融合头。但实际只运行了 Wave 0：在最终层统计量上用合成非对称扰动测试可分性。由于预门槛只在 `bracket_white` 通过，项目按协议没有启动：

- Wave 1 多层特征导出；
- Wave 2 P0/P1 可学习头；
- 真正的跨分支 token 交互。

结论是：**旧 CASF 不能直接复活，但“真实多层互补性是否存在”仍未被实验回答。**

---

## 4. 文献边界：为什么“加模块”本身不够新

近期工作已经覆盖了大量直觉上容易想到的结构：

1. **MFuser（CVPR 2025）** 已在 DINO 类 VFM 与 CLIP 类 VLM 之间使用 co-adapter/Mamba 融合空间与序列信息。因此“用 Mamba 连接 DINO 和 CLIP”不能单独作为创新。
2. **ReMP-AD（ICCV 2025）** 已将 token retrieval 与 vision-language prior fusion 用于 few-shot 工业异常检测。因此“检索后再融合视觉语言先验”也不是空白。
3. **Crossmodal Feature Mapping（CVPR 2024）** 已在正常样本上学习跨模态映射，并用映射不一致检测异常；**FIND（ICCV 2025）** 又结合了 intra-modal reverse distillation 与 cross-modal mapping。它们虽是 RGB–3D，但与 NCPRA 的方法论邻近。
4. **PALADIN（CVPRW 2026）** 已将冻结 DINOv3 patch 多层特征通过轻量 adapter 投入 CLIP 文本空间，并用正常图加合成异常训练。
5. **VisualAD（CVPR 2026）** 已在冻结 ViT 中插入正常/异常可学习 token，并在多个中间层使用 spatial-aware cross-attention。
6. **AnomalyVFM（CVPR 2026）** 已组合多阶段合成异常、低秩 feature adapter 和像素损失。
7. **DCP-SFR（CVPR 2026）** 已明确以“深层传播导致 defect cue fading”为问题，并做早期异常线索放大与结构细化。
8. **BSMAD（CVPRW 2026，医疗异常）** 已将冻结 DINOv3 的结构 token 以严格受限的弱加性系数注入 CLIP dense path，同时保留独立的 CLIP image-level 决策。这与“有界残差 + 不破坏主分支”的直觉非常接近，是 SCAIF 必须正面区分的最近工作。
9. **MambaAlign（JCDE 2026，RGB-X 工业异常）** 已采用深层、内容条件化的 Cross Mamba Interaction，并用 top-down alignment-aware fusion 恢复低层定位。因此“深层 Mamba 交互 + 低层重建”也不宜作为本项目主张。

因此可投稿的主张不能是：

> 我们首次在 DINO/CLIP 后加 adapter / attention / Mamba / learnable token。

应改成可检验的问题主张：

> 正常区域需要跨分支一致性，但缺陷往往只先触发某一分支或某一深度。我们只在正常支持集证明安全的位置打开有界的、非对称的残差交互，同时保留两个原始 private stream，使模块不能通过过度对齐抹掉缺陷证据。

---

## 5. 推荐主路线：SCAIF

### 5.1 核心直觉

DINO 偏局部结构与细粒度对应，CLIP-image 偏预训练语义和更大范围不变性。直接 concat 假设两者在所有类别、层、空间位置都同样可靠；直接 cross-attention 又可能把某一分支发现的微弱异常“修复”为正常。

SCAIF 不替换两条分支，而是增加第三类信息：**受正常支持集约束的交互残差**。

### 5.2 模块结构

选择语义大致匹配的 2–3 对层，而不是机械配对相同 block index。设归一化后的中层 token 为 \(D_l,C_m\)。

1. 低维投影：

\[
u_D=P_D(D_l),\quad u_C=P_C(C_m),\qquad \dim(u)=64\text{ 或 }128.
\]

2. 局部双向交互，只看对应 patch 的 `3×3` 邻域：

\[
r_{C\rightarrow D}=A_{C\rightarrow D}(u_D,u_C),\qquad
r_{D\rightarrow C}=A_{D\rightarrow C}(u_C,u_D).
\]

3. 支持集条件化有界门：

\[
g_D=0.2\,\sigma(h_D(T_S,T_x)),\qquad
g_C=0.2\,\sigma(h_C(T_S,T_x)).
\]

其中 \(T_S\) 是正常支持集的每层方差、局部一致性、D/C 相关性和 LOO 距离统计；门初始为 0，使模型起点严格退化为 A1/静态多层基线。

4. 残差更新：

\[
\tilde D_l=D_l+g_D W_Dr_{C\rightarrow D},\qquad
\tilde C_m=C_m+g_C W_Cr_{D\rightarrow C}.
\]

5. 不丢弃原分支：最终检测表征至少包含

\[
[D_l,C_m,\tilde D_l-D_l,\tilde C_m-C_m].
\]

原始 D/C 是 private evidence，两个 interaction residual 是 shared/violation evidence。若交互学坏，零门仍可回到原基线。

### 5.3 真正的创新点不在 cross-attention

在 BSMAD 已有“受限弱加性 DINO→CLIP 注入”的前提下，SCAIF 的可辩护新意必须由四项同时构成：

- **support-conditioned**：不同类别、shot 和图像质量产生不同门，而非全局固定参数；
- **asymmetric**：允许 D→C 与 C→D 的方向和空间位置不同；
- **bounded residual safety**：交互只能小幅修正，且原始 private stream 永远保留；
- **anomaly-preservation objective**：明确惩罚“交互后合成/真实缺陷响应下降”，避免正常对齐把异常抹掉。

少一项，方法就容易退化为已有 adapter/cross-attention 的重新包装。

### 5.4 训练选择

#### 方案 A：允许源类别训练（推荐）

- backbone 全冻结；
- 在与目标类别严格不相交的源类别上 episodic 训练；
- 每个 episode 模拟 `K∈{1,2,4}` 正常支持集和 query；
- 可用源数据真实异常监督，并加入多个合成异常族；
- 目标 MPDD 类别只前向，不更新模块参数。

建议损失：

\[
L=L_{seg}+0.2L_{clean-preserve}+0.2L_{anomaly-preserve}
+0.05L_{gate-sparse}+0.05L_{branch-drop}.
\]

其中 `anomaly-preserve` 要求交互后的异常 margin 不低于 private branch；`branch-drop` 随机丢弃一个分支，防止门退化成永远依赖一个专家。

#### 方案 B：坚持目标 normal-only、无源异常训练

只允许：

- 缓存特征上的低秩 adapter；
- 支持集 LOO、自监督 mask/reconstruction；
- 门宽 `≤0.1`，参数 `≤100k`；
- 不能用测试 query 更新权重。

该方案与 NCPRA/Crossmodal Mapping 更接近，成功概率较低。若采用，必须从“确定性同位置回归”升级为“局部多峰预测 + 不确定性”，例如输出多个 prototype 或均值/方差，而不是再做一次 MLP cosine prediction。

---

## 6. 不同突破口及优先级

| ID | 突破口 | 与旧实验的真实差异 | 新颖性潜力 | 实验风险 | 建议 |
|---|---|---|---:|---:|---|
| E1 | SCAIF 缓存多层版 | 多层、局部双向、有界门、保留 private stream | 高 | 中 | **第一主试** |
| E2 | SCAIF in-backbone bridge | 交互结果继续流过后续 Transformer blocks | 中高 | 高 | E1 通过后才做 |
| E3 | 多峰跨分支预测 | 预测分布/多个正常模式，而非 NCPRA 单点回归 | 中 | 中高 | 第二梯队 |
| E4 | 稀疏层路由器 | 每个 patch 选择浅层结构证据或深层语义证据 | 中 | 高 | 先做 oracle/可识别性门 |
| E5 | private/shared 正交分解 | 保留分支特有异常，shared 只学正常共性 | 中高 | 中 | 可并入 E1 消融，不先独立扩张 |
| E6 | 检索条件交互 memory | DINO 检索正常邻居，CLIP 只在候选内重排/交互 | 中低 | 低 | 工程稳健备选 |
| E7 | 可变计算 bridge | 仅疑似 patch 打开交互，正常 patch 走 A1 | 中 | 中 | 若效率是论文问题再做 |

### 6.1 E2：真正的“前期分支融合”

E1 在中间层输出上做 post-hoc 交互，便于快速判断信号；E2 才把 bridge 插到 backbone 中段：

```text
DINO blocks 1...i ── D_i ─┐        ┌─ residual bridge ─ D'_i ─ DINO blocks i+1...end
                            ├─ SCAIF ┤
CLIP blocks 1...j ── C_j ─┘        └─ residual bridge ─ C'_j ─ CLIP blocks j+1...end
```

建议只插 1–2 个 bridge，且冻结 backbone。**不建议从像素输入或前两层就融合**：两模型 patch/grid、位置编码和低级统计差异大，早期强耦合最容易破坏预训练表征。优先插在 `1/2` 和 `3/4` 深度附近，并按实际语义相似度配层。

### 6.2 E3：多峰跨分支预测

正常外观可能是一对多映射，同一 DINO 结构可对应多种 CLIP 语义/颜色状态。NCPRA 用单一预测向量，会把正常多模态性误当异常。可改为：

- mixture prototypes；
- heteroscedastic mean/variance；
- local-neighborhood masked prediction；
- anomaly score 使用负对数似然与双向不确定性，而非单一 cosine residual。

但这条路线与 Crossmodal Feature Mapping、FIND 以及旧 NCPRA 邻近，只有在“多峰 + few-shot support conditioning + 机制对照”同时成立时才值得写。

### 6.3 E4：稀疏层路由

不是把所有层 concat，而是为每个 patch 选择层/分支：微小纹理缺陷偏浅层，缺件/结构错误偏深层。门输入只能用正常统计、query 的无标签稳定性和跨增强一致性，不能偷看异常标签。先证明 layer oracle headroom 可识别，再训练 router。

### 6.4 E6：memory 内交互

DINO 先找结构上相似的正常 anchor，CLIP 在这些 paired anchor 中重排；反向也做一次。相比修改 backbone 更稳、更省显存，但和 RCEC/ReMP-AD/FastRef 更接近，论文新颖性较弱，适合当稳健备选或强基线。

---

## 7. 分级实验，不要直接上大模型

### Stage 0：多层可观测性门（先回答“值得不值得”）

此阶段不训练任何新模块。

1. 导出 DINO 三层（总深度约 `1/2, 3/4, 1`）和 CLIP `[6,12,18,24]`；
2. 统一 sample ID、参考图顺序和空间网格；
3. 报告每层 DINO-only、CLIP-only、静态同层 concat；
4. 计算层间/分支间 patch score 的 Spearman 相关；
5. 只在 MPDD development `seed0 × shot{1,2,4}` 做预注册 probe。

进入 Stage 1 至少满足：

- 最佳静态多层基线相对 final-layer A1 的 mean ΔPixel-AP `≥ +0.003`，或 layer oracle headroom `≥ +0.010`；
- 至少 2 对候选层的跨分支 Spearman `<0.95`；
- 改善不是单一类别贡献超过总正增益的 50%；
- deepest cache 与当前 A1 对齐误差 `<1e-5`。

否则说明当前数据/分辨率没有足够的多层可用信号，停止所有早期可学习模块。

### Stage 1：缓存特征版 SCAIF

- 不回传进 backbone；
- 参数 `≤300k`；
- 局部窗口固定 `3×3`；
- 最多 2 组层对；
- gate 上限 `0.2`，zero-init；
- 先跑 `seed0 × shot{1,2,4}`。

必须有的对照：

1. A1 final concat；
2. static multi-layer concat；
3. 参数量匹配 DINO-only adapter；
4. 参数量匹配 CLIP-only adapter；
5. 无 support 的固定 gate；
6. shuffled DINO↔CLIP 空间对应；
7. 去掉 private stream；
8. 对称 gate（强制 D→C = C→D）；
9. same-parameter MLP，无 cross interaction；
10. `gate=0` 身份回归测试。

机制门：

- 相对 strongest static/parameter-matched control mean ΔPixel-AP `≥ +0.004`；
- shuffled correspondence 至少下降 `0.003`；
- no-support 至少下降 `0.003`；
- 去掉 private stream 后，合成缺陷保真或真实 development Pixel-AP 有显著下降；
- 三个 family seed 方向一致，不允许只报最佳 seed。

性能门：

- 相对 A1 mean ΔPixel-AP `≥ +0.006`；
- `seed × shot` 9 个设置至少 7 个为正；
- 6 类至少 5 类为正；
- worst-category ΔPixel-AP `≥ -0.010`；
- gate 饱和比例 `<10%`，否则说明模块退化为常开修正。

### Stage 2：in-backbone bridge

只有 Stage 1 通过才实现。Stage 2 必须在同一训练数据和参数预算下比缓存版再提高 `≥0.003`，否则不值得承担复杂度。附加约束：

- bridge 总参数 `≤1M`；
- 推理延迟增幅 `≤25%`；
- 峰值显存适配现有 6GB 设备；
- 使用 gradient checkpointing 或逐分支执行，但不能改变 A1 输入分辨率；
- 验证零门时输出与原 backbone max abs error `<1e-5`。

---

## 8. 给实验 AI 的执行指令

> 先只读核查本文件、13 号文档、22 号文档、NCPRA 源码与 CASF 最终决策。不得修改历史结果或宣称 CASF 多层路线已失败。新建独立的 `innovation_v12_early_fusion/` 目录。第一阶段只做多层 cache exporter 和 Stage 0 可观测性门，不训练 SCAIF，不查看冻结外部测试标签。必须复用当前 A1 的 sample/ref 对齐和 CLIP→DINO 网格变换，并证明 deepest feature 与 A1 cache 的 max abs error `<1e-5`。预注册层号后不得按 Pixel-AP 改层。仅当 Stage 0 满足本文件所有门槛，才实现缓存特征版 SCAIF。SCAIF backbone 全冻结、参数不超过 300k、局部窗口 3×3、gate 上限 0.2 且 zero-init，必须保留原始 DINO/CLIP private stream。运行全部十个控制，若相对 strongest control 独立增益不足 0.004、shuffled correspondence 或 no-support 不掉至少 0.003，立即归档；不得通过加层、加宽、扫温度继续搜索。只有缓存版同时通过机制门和 A1 性能门，才提交是否实现 in-backbone bridge 的决策，不得自动进入 Stage 2。

### 8.1 必交文件

```text
experiments/dynamic_fusion/innovation_v12_early_fusion/
├── 00_protocol/
│   ├── PROTOCOL_FROZEN.yaml
│   └── DATA_LEAKAGE_AUDIT.md
├── 01_multilayer_cache/
│   ├── CACHE_MANIFEST.json
│   ├── ALIGNMENT_REPORT.json
│   └── DEEPEST_PARITY_REPORT.json
├── 02_stage0_probe/
│   ├── LAYERWISE_RESULTS.csv
│   ├── SCORE_CORRELATIONS.csv
│   ├── ORACLE_HEADROOM.json
│   └── STAGE0_DECISION.md
├── 03_scaif_small_gate/
│   ├── CONFIG.yaml
│   ├── PARAMETER_COUNT.json
│   ├── CONTROL_RESULTS.csv
│   ├── MECHANISM_AUDIT.json
│   └── STAGE1_DECISION.md
└── FINAL_DECISION.md
```

---

## 9. 最终建议

值得尝试，但只值得尝试一条**有停损条件的分级路线**：

1. 先补上从未完成的多层观测；
2. 有信号再做缓存版 SCAIF；
3. 缓存版证明“跨分支交互本身”有效后，才做真正前期 bridge；
4. 若研究协议不能接受源类别训练，就把优先级下调，不要在 1/2/4 张目标正常图上训练大模块。

最不值得做的是无门槛地增加 Attention/Mamba/LoRA；最值得做的是把**何时交互、向哪个方向交互、交互后是否抹掉缺陷**变成可测量、可被强对照证伪的机制。

---

## 10. 主要参考

- [MFuser, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Mamba_as_a_Bridge_Where_Vision_Foundation_Models_Meet_Vision_CVPR_2025_paper.html)
- [ReMP-AD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html)
- [Crossmodal Feature Mapping, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Costanzino_Multimodal_Industrial_Anomaly_Detection_by_Crossmodal_Feature_Mapping_CVPR_2024_paper.html)
- [FIND, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FIND_Few-Shot_Anomaly_Inspection_with_Normal-Only_Multi-Modal_Data_ICCV_2025_paper.html)
- [PALADIN, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html)
- [VisualAD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VisualAD_Language-Free_Zero-Shot_Anomaly_Detection_via_Vision_Transformer_CVPR_2026_paper.html)
- [AnomalyVFM, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Fucka_AnomalyVFM_--_Transforming_Vision_Foundation_Models_into_Zero-Shot_Anomaly_Detectors_CVPR_2026_paper.html)
- [DCP-SFR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html)
- [FastRef, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html)
- [BSMAD, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/Med-Reasoner/html/Lin_BSMADBridging_Semantic_and_Structural_Manifolds_for_Robust_Cross-Modality_Medical_Anomaly_CVPRW_2026_paper.html)
- [MambaAlign, JCDE 2026](https://academic.oup.com/jcde/article/13/1/514/8405688)
