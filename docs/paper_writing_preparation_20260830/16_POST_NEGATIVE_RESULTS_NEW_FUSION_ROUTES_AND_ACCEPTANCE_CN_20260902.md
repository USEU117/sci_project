# 负结果之后的新融合方向与分阶段验收任务书（2026-09-02）

引用关系：

- 当前主方法与论文证据：`submission_repro_20260827/`，A1 = DINOv2 + AnomalyCLIP patch 特征固定拼接、1-NN 正常记忆；
- 多路线负结果：`experiments/dynamic_fusion/innovation_v2/`、`innovation_v4_diagnostics/` 与 `innovation_v5_casf/`；
- CASF 最终决定：`experiments/dynamic_fusion/innovation_v5_casf/FINAL_CASF_DECISION.md`；
- 官方 SubspaceAD 审计：`experiments/dynamic_fusion/v4_vision_text_20260819/06_v2_g2_audit/g2_audit_report.json`；
- 上位构思文档：[14_BROAD_ALGORITHM_INNOVATION_PORTFOLIO_CN_20260902.md](14_BROAD_ALGORITHM_INNOVATION_PORTFOLIO_CN_20260902.md)；CASF 历史方案：[15_CASF_CATEGORY_CONDITIONAL_ALGORITHM_AND_EXPERIMENT_PLAN_CN_20260902.md](15_CASF_CATEGORY_CONDITIONAL_ALGORITHM_AND_EXPERIMENT_PLAN_CN_20260902.md)。

本文件供后续 AI 助手直接执行。它不宣布新算法已经成功，也不修改 A1 的冻结论文结论。它只根据现有真实结果，
选择仍有信息增量且能被严格证伪的三条新路线。执行优先级固定为 **S0 → S1 → S2**；S0 未完成前禁止启动 S1/S2，
避免再次同时铺开大量公式。

---

## 0. 当前判断：大部分创新实验为负，但不能说“所有方向都为负”

### 0.1 已经明确归档的方向

1. A2 的 27 个局部归一化、对齐、密度、图传播及其控制均未过门；最好的 CEQA 只有约 +0.0028，且机制控制不成立。
2. A4-D1 的频率/尺度代理几乎无判别力，oracle headroom 为 0；不再做频率分支或尺度门控。
3. A4-D2 的 context 在 permutation/duplicate 合成任务上有信号，但 missing 为负，node-OT 接近随机；不支持直接立项图匹配主线。
4. CASF 放大探针只剩 bracket_white 1/6 类，且绝对 Dice 低，无法支撑 pooled-6 门槛，已在 Wave 0 后归档。
5. 禁止将上述方法改名、换激活函数或增加超参数后重新提交；除非引入本文件定义的全新证据源并有独立控制。

### 0.2 被旧的硬门“整体判负”，但实际上存在强正信号的方向

官方 SubspaceAD 完整配置不是弱代理版：DINOv2-with-registers-giant、672 输入、30 次旋转增强、
多层 `-12..-18` 聚合、PCA 0.99、重建残差。在 MPDD 的 3 split seeds × 3 shots 上：

- 相对 matched DINO-KNN：平均 **ΔPixel-AP +0.04715**、平均 **ΔAUPRO +0.06344**，9/9 配置均值非负；
- 旧 G2 仅因 connector 平均 −0.11675 违反“最差类 ≥ −0.02”而整体失败；
- 与当前 A1 逐配置重新对齐后，54 个“类别×seed×shot”单元中 **38/54 为正**，总平均
  **ΔPixel-AP +0.02132**。

| 类别 | 官方 SubspaceAD − A1 平均 ΔPixel-AP | 正配置数 | 最差配置 | 结论 |
|---|---:|---:|---:|---|
| bracket_black | +0.15 | 9/9 | +0.02 | 强互补 |
| bracket_brown | +0.03 | 9/9 | +0.02 | 稳定互补 |
| bracket_white | 约 0.00 | 3/9 | −0.05 | 不稳定 |
| connector | **−0.16** | 0/9 | −0.26 | 明确冲突 |
| metal_plate | +0.04 | 8/9 | −0.01 | 基本互补 |
| tubes | +0.06 | 9/9 | +0.04 | 稳定互补 |

这说明真正尚未解决的问题不是“有没有第二个有效专家”，而是：

> 如何在不使用测试标签、不按类别名称手工写规则的情况下，识别 SubspaceAD 何时可靠，并将其新增异常证据加入 A1，
> 同时保护 connector 等冲突类别。

这个问题比继续修改 DINO/CLIP 拼接权重更有算法价值，也有直接可验收的现有证据。

---

## 1. 路线总览与优先级

| 路线 | 核心新增信息 | 与旧失败路线的区别 | 优先级 | 当前证据 |
|---|---|---|---:|---|
| S0：DG-SAFE 双正常性几何安全融合 | A1 最近邻残差 + 官方 SubspaceAD 子空间重建残差 | 融合“正常性模型”，不是再融合编码器或弱 PCA 代理 | 1 | SubspaceAD 对 A1 平均 +0.0213，5/6 类平均非负 |
| S1：HGLC 全局—局部一致性校准 | 图像级 DINO/CLIP/文本异常证据 + A1 像素图 | 图像级证据只校准整幅图，不做不稳定的逐像素路由 | 2 | 现有缓存无 CLS，须先做小诊断 |
| S2：GPMR 概率原型流形残差 | 多原型后验 + 各向异性残差 | 用概率流形替代 KNN/PCA，不是 KNN 距离再加权 | 3 | 算法创新较强，但风险与工程量最高 |

论文主线最多保留 **S0 一个主模块 + S1 一个辅助模块**。S2 只有在 S0 的“强专家确有互补但可靠性不可识别”时才启动。
不得直接做 A1 + SubspaceAD + 文本三分支大融合；每个证据源必须先独立通过小门。

---

## 2. S0：DG-SAFE（Dual-Geometry Safeguarded Evidence Fusion）

中文名称：**双正常性几何的可靠性保护融合**。

### 2.1 科学假设

A1 与官方 SubspaceAD 对“正常”的描述不同：

- A1：测试 patch 到有限正常实例记忆的最近邻距离，擅长局部外观与跨模态语义表征；
- SubspaceAD：测试 patch 偏离正常多层特征子空间的重建残差，擅长捕捉正常流形之外的变化。

因此两者不是同一 backbone 分数的重复加权。若子空间残差在正常参考的轻微增强、多层子空间和重采样下稳定，
它应当提供 A1 未覆盖的异常证据；若不稳定，则最终分数应退回 A1。可证伪预测为：

1. A1 与 SubspaceAD 的逐图/逐像素错误不是完全同构，固定 oracle 诊断存在明显融合 headroom；
2. 只用正常参考得到的子空间稳定性指标，能把 connector 排入低可靠组，而不是事后按类别名屏蔽；
3. 可靠性保护版本胜过 A1、SubspaceAD、简单平均、简单最大值和打乱可靠性控制；
4. 增益主要出现在 SubspaceAD 已显示正向的类别，而 connector 的损失被限制。

### 2.2 输入与统一几何

1. A1 使用现有 `submission_repro_20260827/predictions_compact/maps/mpdd/` 中的 float16 patch map；
2. 修改 `scripts/run_v4_official_g2_audit.py`，增加 `--export-maps`，保存：
   `sample_ids`、原始 `48×48` 重建残差图、`image_res=672`、层集合、PCA 配置、reference IDs 和模型 hash；
3. 不保存 GT 到预测包；GT 只由 evaluator 在分数生成完成后加载；
4. 按 `sample_ids` 严格对齐；SubspaceAD 原始网格经同一确定性插值到 448，A1 按现有 `dists2map` 回放到 448；
5. 必须先做 identity replay：新导出的 SubspaceAD map 重新计算指标，与现有 G2 报告每类绝对误差 ≤ `5e-4`。

### 2.3 正常样本经验尾概率标定

不得直接平均两个原始分数，因为尺度、分辨率和尾部分布不同。对专家 `b∈{A1,SUB}`，只从正常参考构造
calibration pool：identity、轻微亮度/对比度、±2% 平移、±2° 旋转；几何增强必须逆变换回原坐标。

对每个像素分数计算经验尾概率：

```text
p_b(x) = (1 + #{u in normal_pool: u >= s_b(x)}) / (2 + |normal_pool|)
z_b(x) = clip(-log p_b(x), 0, 12)
```

这一步只统一“在本专家看来有多罕见”，不使用真实异常、测试正常图或测试统计。

### 2.4 可靠性与证据稳定性

对子空间专家生成三类正常-only 可靠性量：

- `U_aug`：正常参考在轻微增强前后、逆变换后的 `z_SUB` 图的 median L1 差；
- `U_layer`：以层组 `{-12,-13,-14}` 和 `{-16,-17,-18}` 各自建子空间后，两张 `z_SUB` 图的 median L1 差；
- `B_tail`：正常增强池中 `z_SUB` 的 P99 尾部负担；尾部越重，正常区域误报风险越高。

用预注册稳健排序产生配置级可靠性，不训练分类器：

```text
q_sub = 1 - mean(percentile_rank(U_aug), percentile_rank(U_layer), percentile_rank(B_tail))
r_sub = clip((q_sub - 0.25) / 0.50, 0, 1)
```

percentile rank 在 MPDD development 的类别×配置正常统计中计算。该公式在查看任何真实异常分数之前冻结；
不得通过类别名称或小门结果单独重写 connector 的 `r_sub`。

### 2.5 三个预注册候选

所有候选都以 A1 为锚，`τ=−log(0.05)`，`λ∈{0.25,0.5,1.0}` 只允许在 MPDD 小门按 pooled 指标选择一次，
不允许逐类选 λ。

```text
C0 tail-add:
z_final = z_A1 + λ * r_sub * ReLU(z_SUB - τ)

C1 stable-discovery:
m = local_consistency_SUB * ReLU(z_SUB - z_A1 - 0.5)
z_final = z_A1 + λ * r_sub * m

C2 guarded-union:
z_union = -log(exp(-z_A1) * exp(-r_sub*z_SUB))
z_final = min(z_union, z_A1 + λ * r_sub * (1 + z_A1))
```

`local_consistency_SUB` 是层组图与增强图在同一位置均超过各自 95% normal 分位数的比例，范围 `[0,1]`。
C1 是首选：它只加入“SubspaceAD 比 A1 更异常、且跨扰动/层组稳定”的新增证据。

### 2.6 强制控制与消融

- `CTRL-A1`：冻结 A1；
- `CTRL-SUB`：官方 SubspaceAD；
- `CTRL-MEAN`：经验 z 分数 0.5/0.5 平均；
- `CTRL-MAX`：经验 z 分数逐像素 max；
- `CTRL-R1`：固定 `r_sub=1`，检验保护门的必要性；
- `CTRL-SHUFFLE-R`：在类别×配置间打乱 `r_sub`，预期退化；
- `CTRL-AUG-ONLY`、`CTRL-LAYER-ONLY`：可靠性来源消融；
- `CTRL-WEAK-PCA`：旧 vitb14 同 backbone PCA 代理，只作“强专家与弱代理不同”的机制控制，不参与选模。

### 2.7 分阶段执行与停止门

#### S0-Wave 0：导出、回放与对齐

- 只做 MPDD seed0 × shot{1,2,4} × 6 类；
- map 回放误差、sample ID、分辨率、参考 ID、数据角色全部通过；
- 任一错位或 map 无法复现旧指标，立即停止，不跑融合。

#### S0-Wave 1：互补上限诊断（允许使用 MPDD development GT，仅用于判断是否值得做）

固定扫描 `A1`、`SUB`、mean、max 和离散 `λ`，报告逐类 Pixel-AP、AUPRO 以及每图赢家；另报告
pixelwise oracle 仅作为不可部署上界。通过条件：

- 至少一种固定、非 oracle 组合相对 A1 与 SUB 中较好的单方法，pooled mean ΔPixel-AP ≥ +0.005；或
- oracle headroom ≥ +0.020 且至少 4/6 类为正，证明存在值得学习/门控的互补性。

若两项都不满足，S0 归档：两个方法虽逐类各有优势，但像素证据不可融合。

#### S0-Wave 2：正常-only 可靠性诊断

先冻结 §2.4 公式，再由 evaluator 读取 GT，检查可靠性是否解释真实风险。通过条件：

- `r_sub` 与“SUB−A1 per-category/config ΔPixel-AP”的 Spearman ρ ≥ +0.40；
- connector 的 `r_sub` 位于全部类别×配置的后 25%，且不得用类别名参与计算；
- 可靠性同 seed 重跑最大误差 `<1e-7`；轻微增强种子变化后类别级排序 Kendall τ ≥ 0.60。

若失败，禁止直接训练一个标签路由器；转入 S2，研究概率流形本身，而不是过拟合 selector。

#### S0-Wave 3：MPDD 小门

数据：seed0 × shot{1,2,4} × 6 类。候选相对 A1 必须同时满足：

- 三-shot pooled mean ΔPixel-AP ≥ +0.008；3/3 shot 为正；worst shot ≥ −0.003；
- mean ΔAUPRO ≥ 0；mean ΔPixel-AUROC ≥ −0.002；
- connector 三-shot平均 ΔPixel-AP ≥ −0.020，任一 shot 不低于 −0.040；
- 胜过 `CTRL-MEAN` 与 `CTRL-MAX` 至少 +0.003；
- 胜过 `CTRL-R1` 或 `CTRL-SHUFFLE-R` 至少 +0.002，证明可靠性保护有用；
- λ 只能选一个全局值；所有候选/控制完整公开。

#### S0-Wave 4：Full MPDD

唯一小门 winner 扩到 3 seeds × 3 shots：

- mean ΔPixel-AP ≥ +0.010；至少 7/9 配置为正；
- worst category 平均 ΔPixel-AP ≥ −0.020；
- AUPRO 不降，机制控制仍成立；
- 通过后冻结代码、配置、模型、PCA、map、报告 hash。

只有 Full MPDD 全部通过，才允许一次性运行 BTAD/MVTec/VisA 外部验证；失败则 A1 保持主方法。

### 2.8 可成立的论文创新表述（通过后才可使用）

1. 提出双正常性几何：把实例记忆最近邻与多层正常子空间重建视为互补异常证据，而非简单多编码器堆叠；
2. 提出只依赖正常参考的跨增强、跨层稳定性可靠度，抑制不稳定子空间专家；
3. 提出以 A1 为锚的稳定新增证据融合，在引入子空间召回的同时限制最差类别退化；
4. 用强控制证明增益不来自分数尺度、简单平均/最大或更大 DINO 模型本身。

文献边界必须诚实：不能声称首次使用子空间、概率原型或几何融合。SubspaceAD 已使用子空间重建；
G2SF 已研究几何引导分数融合；GPFlow 已研究高斯原型与概率流。本文潜在新意只在“**异构正常性模型 +
正常-only 稳定性保护 + A1 锚定新增证据**”这一组合及其严格机制验证。

---

## 3. S1：HGLC（Hierarchical Global–Local Consistency Calibration）

### 3.1 为什么值得看，但不能先做

当前 A1 完全由 patch 最近邻产生像素图，图像分数近似来自局部极值；它没有显式回答“整幅图是否像正常产品”。
现有 DINO/CLIP feature cache 只含 `patch_features/ref_patch_features`，**没有 CLS/global token**，因此这条路线
尚无直接证据。AnomalyCLIP 文本像素图平均 Pixel-AP 仅 0.1392，不能再作为第三张像素图直接融合。

### 3.2 算法边界

重新导出冻结 DINO CLS、CLIP global image embedding，以及可选的 AnomalyCLIP image-level normal/abnormal logit。
只用 normal reference 建立全局 Mahalanobis/nearest-prototype 尾概率 `z_global`，然后对 A1 做整图级校准：

```text
z_final(x,p) = z_A1(x,p) + beta * ReLU(z_global(x)-tau_g) * h(z_A1(x,p))
```

其中 `h` 只能是固定的 `z/(1+z)` 或 top-q mask；同一图所有像素共享一个 global gate，不训练逐像素 router。
它的目标是强化“局部异常与整图异常一致”的样本，不负责定位新区域。

### 3.3 先做的诊断与门槛

仅 MPDD seed0/k{1,2,4}：

1. identity replay 确认重新导出的 patch 特征不改变 A1；
2. 比较 A1 max/top1% image score、DINO CLS、CLIP global、text global 的 Image-AUROC/AP；
3. 固定 beta 网格 `{0.1,0.25,0.5}`，不得逐类选择；
4. 只有当全局单分数相对 A1 image score 平均 Image-AP ≥ +0.010，且校准后 Pixel-AP ≥ +0.005、
   worst category ≥ −0.02，才立项；否则归档。

控制：打乱图像级 gate、DINO-only global、CLIP-only global、text-only global、直接乘法、A1 原图。
若通过，它只能作为 S0 winner 的后续独立模块；必须先证明 `S0+S1 > S0`，否则论文只保留 S0。

---

## 4. S2：GPMR（Gaussian Prototype Manifold Residual）高风险备用路线

### 4.1 触发条件

仅当 S0-Wave 1 证明 oracle 互补充分，但 S0-Wave 2 表明手工正常可靠性无法识别时启动。不要与 S0 并行。

### 4.2 核心设计

对 DINO、CLIP 或拼接后的正常 patch 先做位置粗分桶，再以 shrinkage Gaussian mixture 建模多个正常原型：

```text
w_k(x) ∝ pi_k * N(x | mu_k, Sigma_k + eps I)
x_hat = sum_k w_k(x) mu_k
r_para = projected residual within prototype subspace
r_perp = residual orthogonal to prototype subspace
s = r_perp + eta * uncertainty(w)
```

与旧 LNDC/同 backbone PCA 的区别是：这里使用多原型后验、收缩协方差和正交残差，不再做单一全局 PCA 后接 KNN。
融合 DINO/CLIP 时先各自计算概率尾分数，再共享后验置信度；不要在 raw feature 上无条件拼接协方差。

### 4.3 风险与验收

该方向与 2026 GPFlow 的高斯原型/概率流接近，创新空间比 S0 更窄，必须用简化、training-free、normal-only 的定位区分。
小门前先检验：多原型责任熵是否与真实错误相关；若 Spearman |ρ| < 0.3，立即停止。正式门槛沿用 S0-Wave 3/4，
并强制胜过全局 PCA、KNN、diagonal Gaussian、单原型四个控制。

---

## 5. 暂不建议的融合方向

1. **再加 RADIO/DINOv3/第三视觉骨干**：单纯换骨干是性能工程，不足以成为算法创新；先验证已有强 SubspaceAD 专家的价值。
2. **把 AnomalyCLIP 文本热图直接平均进 A1**：文本热图本身 Pixel-AP 太低，旧 oracle headroom 不能等同于可部署增益。
3. **按类别名称硬切换 A1/SubspaceAD**：在 MPDD 上会很好看，但没有跨数据集可迁移机制，属于开发集过拟合。
4. **训练监督路由器预测哪个专家更好**：few-shot 正常设置下没有合法标签，且会破坏 training-free/normal-only 口径。
5. **继续做频率、尺度、图 OT 或 CASF 变体**：已有诊断不支持，除非先出现全新的独立信息价值证据。
6. **立即跑外部集挑方法**：BTAD/MVTec/VisA 已是冻结验证角色；只能在 MPDD 冻结唯一 winner 后一次性使用。

---

## 6. 后续 AI 助手的具体执行顺序

```text
Step 1  建 experiments/dynamic_fusion/innovation_v6_dgsafe/ 五目录骨架；复制协议，不复制旧结论
Step 2  给官方 SubspaceAD runner 加 --export-maps；只跑 MPDD s0/k1 单类 smoke
Step 3  做 map identity replay、sample-ID/几何/泄漏单元测试
Step 4  完成 MPDD s0 × k{1,2,4} 六类 map 导出
Step 5  运行 S0-Wave 1 互补上限诊断；写 WAVE1_DECISION.md
Step 6  仅 Wave 1 通过才构造 normal augmentation/layer probes，冻结 reliability.json
Step 7  evaluator 读 GT，执行 S0-Wave 2；写 WAVE2_DECISION.md
Step 8  仅 Wave 2 通过才实现 C0/C1/C2 与全部控制，跑 MPDD 小门
Step 9  唯一 winner 跑 Full MPDD；冻结 hash
Step 10  只有 Full MPDD 通过才请求用户批准外部一次性验证
```

每一 Wave 都必须提交：`protocol.json`、`metrics.csv/json`、`leakage_audit.json`、`resource_usage.json`、
`DECISION.md`。失败就停止，不自动转跑下一路线；由用户确认后再决定是否启动 S1 或 S2。

---

## 7. 外部文献定位（执行前需再次核对最终版本）

- SubspaceAD，CVPR 2026：training-free few-shot subspace modeling，界定“子空间重建”先例；
  <https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html>
- G2SF，ICCV 2025：geometry-guided score fusion，界定“几何融合/各向异性距离”先例；
  <https://openaccess.thecvf.com/content/ICCV2025/html/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.html>
- GPFlow，CVPR 2026：Gaussian prototypes 与 probability flow，界定概率原型先例；
  <https://openaccess.thecvf.com/content/CVPR2026/html/Li_GPFlow_Gaussian_Prototype_Probability_Flow_for_Unsupervised_Multi-Modal_Anomaly_Detection_CVPR_2026_paper.html>
- GLAD，WACV 2023：global-to-local anomaly detection，界定全局—局部建模先例；
  <https://openaccess.thecvf.com/content/WACV2023/html/Artola_GLAD_A_Global-to-Local_Anomaly_Detector_WACV_2023_paper.html>
- ReMP-AD，ICCV 2025：retrieval-enhanced multi-modal prompt fusion，界定文本/检索提示融合先例；
  <https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html>

---

## 8. 最终决策

现在不应宣布“项目无创新可做”，也不应继续大规模随机尝试。最合理的下一步是 **S0-DG-SAFE 的低成本诊断**：
现有官方 SubspaceAD 已经提供了高于 A1 的平均性能和 5/6 类正向证据，工程上只缺逐像素图导出与可靠性验证。
这是当前证据最强、故事最连贯、也最容易被严格判假的新突破口。

在 S0 通过 Full MPDD 前，论文主方法仍是 A1；本文件中的名称、公式和 claim 都是“候选”，不能写成已验证贡献。
