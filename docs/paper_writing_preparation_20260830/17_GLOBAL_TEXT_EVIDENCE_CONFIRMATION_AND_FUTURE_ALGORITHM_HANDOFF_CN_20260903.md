# 全局文本证据确认、算法升级与未来分支执行任务书（2026-09-03）

> **接收者**：下一位负责完整执行实验、验收、归档和论文材料更新的 AI 助手。  
> **执行方式**：按本文件的 Phase 0 → Phase 4 顺序执行；只有上一个硬门通过才进入下一阶段。失败时必须写决策文件并停止对应路线，不得临时放宽门槛、扩大参数网格或使用外部验证集救结果。  
> **起始代码基线**：git commit `7c15b6d`。若开始执行时 HEAD 已变化，应先记录实际 HEAD 和工作区状态，保留用户已有修改，禁止 reset/覆盖。  
> **目标**：确认 AnomalyCLIP 图像级文本正结果是否稳定；判断它能否成为论文第二贡献；若且仅若稳定，再检验一个真正具有空间机制的“文本条件区域重排序”候选。  
> **非目标**：不把现成 AnomalyCLIP 分数改名冒充新算法；不继续在 MPDD seed0 上调 scalar gate；不触碰已归档的 DG-SAFE、CASF、GPMR、频率、图 OT 等路线。

关联材料：

- 上位任务书：[16_POST_NEGATIVE_RESULTS_NEW_FUSION_ROUTES_AND_ACCEPTANCE_CN_20260902.md](16_POST_NEGATIVE_RESULTS_NEW_FUSION_ROUTES_AND_ACCEPTANCE_CN_20260902.md)
- 当前最终结果：`experiments/dynamic_fusion/innovation_v6_dgsafe/s1_hglc/S1_HGLC_DECISION.md`
- 图像级明细：`experiments/dynamic_fusion/innovation_v6_dgsafe/s1_hglc/S1_HGLC_DIAG.json`
- 像素校准明细：`experiments/dynamic_fusion/innovation_v6_dgsafe/s1_hglc/S1_CALIB.json`
- 夜间总览：`experiments/dynamic_fusion/innovation_v6_dgsafe/OVERNIGHT_REPORT_20260903.md`
- A1 冻结方法：`submission_repro_20260827/METHOD_SPEC_V2.md`
- A1 紧凑预测：`submission_repro_20260827/predictions_compact/maps/`
- 数据划分：`data/splits/{dataset}/manifest.json`

---

## 0. 当前已经证明和没有证明的内容

### 0.1 已经证明

在 MPDD development、seed0 × shot{1,2,4} 上：

| 信号 | pooled Image-AP | 相对 A1-max |
|---|---:|---:|
| A1-max | 0.7985 | 0 |
| A1-top1% | 0.7536 | −0.0449 |
| DINO CLS | 0.6769 | −0.1216 |
| CLIP global | 0.7035 | −0.0950 |
| AnomalyCLIP 异常文本概率 | **0.8234** | **+0.0249** |

文本信号在 connector、bracket_black、bracket_brown 上明显补强 A1；在 bracket_white、metal_plate 上下降，
tubes 基本持平。文本方向交换后概率严格变为 `1-p`，基本 sanity 已通过。

把同一个图像级文本标量用于放大 A1 像素热图时，最优预注册候选：

```text
z_final = z_A1 + 0.5 * ReLU(p_abn - 0.5) * z_A1/(1+z_A1)
```

pooled ΔPixel-AP = **+0.0040**，低于冻结门槛 +0.005；打乱门控为 −0.0005，说明信号真实但很小。
因此 S1-HGLC **像素融合模块已经归档**，不得继续在 seed0 上换阈值或 beta 追过门槛。

### 0.2 尚未证明

1. 没有证明 +0.0249 能在 MPDD seed1/seed2 和外部数据集保持；
2. 没有 bootstrap 置信区间，不能说优势具有统计显著性；
3. 没有证明“文本图像分数 + A1 像素图”的任务解耦优于强外部方法；
4. 没有证明文本能够改善缺陷位置、轮廓或像素排序；
5. 当前 `p_abn` 是**现有 AnomalyCLIP 的输出**，不是本项目发明，单独使用不能构成算法创新；
6. checkpoint `9_12_4_multiscale_visa/epoch_15.pth` 含 VisA 来源训练的 prompt learner。对 MPDD/BTAD/MVTec 可称
   “target-domain zero-shot transfer”，但不能称整个系统“从未训练”，VisA 也不能被包装成完全独立外部验证。

### 0.3 当前创新性判定

- 作为新的像素级算法：**不够**；
- 作为 A1 论文的图像级辅助发现：**有价值，但必须先做稳定性确认**；
- 作为“全局文本筛查 + 局部视觉定位”的任务解耦系统：创新程度**中等偏弱**，适合 SCI 四区的应用/实证叙事，
  不宜声称全新视觉语言架构；
- 若要提高算法创新，必须让文本参与**空间区域级决策**，且胜过 scalar gate、原始文本图平均和已有方法控制。

---

## 1. 总体路线和硬性顺序

```text
Phase 0  证据/协议审计与新目录骨架
   ↓ 全通过
Phase 1  Full MPDD 图像级文本稳定性 + paired bootstrap
   ├─ G1 失败 → Scenario C：文本发现归档，回到 A1 论文
   └─ G1 通过
          ↓
Phase 2  冻结“全局筛查/局部定位”双输出系统 + 完整内部对照
   ├─ 只需较稳妥投稿 → Phase 3 外部冻结验证
   └─ 需要更强算法创新 → Phase 2R 区域级信息价值小门
                                  ├─ R0 失败 → 不开发区域算法，进入 Phase 3
                                  └─ R0 通过 → R1/R2 新算法小门与 Full MPDD
Phase 3  唯一冻结版本在 BTAD/MVTec 一次性验证；VisA 只作 source/in-domain 附录
Phase 4  按 Scenario A–E 更新论文、图表、claim-evidence matrix 和最终决策
```

**禁止并行开发多个新算法。** Phase 1 未通过前，不实现 Phase 2R；Phase 2R 未通过前，不允许外部集为其选参数。

---

## 2. Phase 0：完整性审计与目录骨架

### 2.1 新目录

创建但不复制大缓存：

```text
configs/innovation_v7_global_text/
scripts/innovation_v7_global_text/
src/industrial_ad/innovation_v7_global_text/
tests/innovation_v7_global_text/
experiments/dynamic_fusion/innovation_v7_global_text/
outputs/dynamic_fusion/innovation_v7_global_text/      # gitignored，大缓存才放这里
```

不得改写：

- `submission_repro_20260827/`
- `experiments/dynamic_fusion/innovation_v6_dgsafe/` 既有 JSON/MD
- BTAD/MVTec/VisA 已冻结预测和结果

### 2.2 输入审计

生成 `00_audit/INPUT_AUDIT.json`，至少包含：

1. 实际 git HEAD、dirty files、Python/Torch/CUDA/GPU 信息；
2. AnomalyCLIP checkpoint 路径、SHA256、prompt 配置、DAPM layer、输入尺寸；
3. MPDD 6 个 cache 的 SHA256、字段、shape、dtype、样本数；
4. 每类 `sample_ids` 唯一、顺序可重放，并与 A1 的 9 个 seed/shot 配置集合完全一致；
5. manifest 的 reference IDs 来自各类 `train/good`，未误入异常图或测试样本；
6. exporter 不读取 GT/mask/标签；evaluator 才读取标签；
7. `text_prob_test` 全有限、范围 `[0,1]`、非恒定；
8. `swap probability = 1 - original probability` 的最大误差；
9. 当前 S1 18 行图像指标能从缓存重放，绝对误差 ≤ `1e-4`；
10. VisA checkpoint 来源边界写入 `checkpoint_provenance.md`。

### 2.3 单元测试

至少实现并通过：

- sample ID set/order alignment；
- seed/shot reference 子集正确，禁止再次把 k4 union 用于 k1/k2；
- A1 map 回放 identity；
- 文本概率方向、finite、repeatability；
- 标签只能在 evaluator 模块加载；
- 相同输入重复计算指标最大绝对误差 `<1e-10`；
- bootstrap 固定 seed 可复现；
- 外部验证开关默认关闭，未生成冻结清单时不能执行。

### 2.4 Phase 0 验收

所有测试通过、重放通过、无泄漏才提交一次 git commit，并写 `00_audit/PHASE0_DECISION.md`。
任一输入对不上，标记 `BLOCKED_INPUT_INTEGRITY`，不得用路径模糊匹配或删样本绕过。

---

## 3. Phase 1：Full MPDD 图像级稳定性确认

### 3.1 为什么这一步不需要重新跑 AnomalyCLIP

文本概率与 A1 的 seed/shot 正常参考选择无关。使用当前 6 个 cache 中每张图唯一的 `p_abn`，分别和
A1 的 3 seeds × 3 shots 对齐即可；不得为不同 seed 重复导出并把相同分数伪装成独立模型运行。

### 3.2 必须实现的脚本

```text
scripts/innovation_v7_global_text/run_mpdd_full_image_evidence.py
scripts/innovation_v7_global_text/run_paired_bootstrap.py
```

输入：

- `experiments/dynamic_fusion/innovation_v6_dgsafe/s1_hglc/cache/{category}.npz`
- `submission_repro_20260827/predictions_compact/maps/mpdd/s{0,1,2}_k{1,2,4}/{category}.npz`
- MPDD manifest/合法 image labels

比较信号：

1. A1-max（主基线）；
2. A1-top1%；
3. TEXT `p_abn`（主候选）；
4. DINO CLS、CLIP global 只复用现有结果或补齐合法缓存，不为它们重新选参数；
5. 官方 AnomalyCLIP image score 若与 `p_abn` 相同，应标注“同一信号”，不得重复列成两个方法。

### 3.3 指标与汇总口径

每个 `(category, seed, shot, signal)` 报告：Image-AP、Image-AUROC、正负样本数。必须同时提供：

- primary macro：先每类平均，再对 6 类平均；
- micro pooled：合并全部图像（只作 secondary，防类别样本数主导）；
- 9 个 seed/shot 配置的 macro delta；
- 6 类跨配置 mean/worst delta；
- discovery 子集：seed0；
- confirmation 子集：seed1/seed2，必须单独报告，不能被 seed0 拉高后隐藏。

### 3.4 统计检验

实现 `B=10,000` 次 paired stratified bootstrap：

1. 每类别内分别对 normal/anomaly 图像有放回采样；
2. 同一 bootstrap 索引同时用于 TEXT 与 A1，保持配对；
3. 对每个配置计算 macro ΔImage-AP 和 ΔImage-AUROC；
4. 输出 percentile 95% CI、median、P(Δ>0)；
5. 再做 category-cluster bootstrap 作为敏感性分析，明确只有 6 类导致的宽区间；
6. 9 配置符号检验/Wilcoxon 仅作辅助，因为配置共享测试图、并非完全独立。

禁止用普通独立样本 t-test 把重复配置当成 9 组独立数据。

### 3.5 G1：Full MPDD 图像级硬门

当前 seed0 已被看过，故同时要求完整结果和未看 confirmation 子集：

1. 9 配置 macro mean ΔImage-AP ≥ **+0.015**；
2. 至少 **7/9** 配置 ΔImage-AP > 0；
3. seed1/seed2 六配置 mean ΔImage-AP ≥ **+0.010**，至少 **4/6** 为正；
4. primary paired bootstrap 95% CI 下界 > 0；
5. macro mean ΔImage-AUROC ≥ **−0.020**（AP 为主指标，但不允许 AUROC 大幅下降）；
6. 至少 3/6 类 mean ΔImage-AP > 0；worst category mean delta ≥ **−0.100**；
7. 所有数值有限、无 ID 错位、无外部集参与、结果完全可重放。

G1 是“图像级文本证据是否稳定”的门，不是算法创新门。

### 3.6 Phase 1 输出

```text
experiments/dynamic_fusion/innovation_v7_global_text/
  01_mpdd_full/
    per_config.csv
    per_category.csv
    summary.json
    bootstrap.json
    bootstrap_delta_plot.png
    config_delta_heatmap.png
    PHASE1_DECISION.md
```

若 G1 失败：进入 Scenario C，停止 Phase 2R/3；不得触碰外部集。

---

## 4. Phase 2：冻结“全局筛查 + 局部定位”双输出系统

仅 G1 通过才执行。本阶段不声称新算法，只把已证明的不同任务优势形成一个可复现系统规格。

### 4.1 冻结系统定义

暂定名称：**GLSD（Global-Language Screening and Local-Visual Defect Localization）**。

```text
Image-level anomaly score  S_img(x) = p_abn(x)                 # AnomalyCLIP 文本概率
Pixel-level anomaly map    M_pix(x) = M_A1(x)                  # 冻结 A1 热图
```

这叫“任务解耦”，不是 score fusion。它表达的是：全局文本负责筛查，局部视觉记忆负责定位。

### 4.2 必须公开的控制

| 系统 | Image score | Pixel map | 目的 |
|---|---|---|---|
| A1/A1 | max(A1 map) | A1 | 主基线 |
| TEXT/A1（GLSD） | text probability | A1 | 候选 |
| TEXT/AnomalyCLIP-map | text probability | 原始文本 map | 现成 AnomalyCLIP 系统对照 |
| CLIP-global/A1 | CLIP global distance | A1 | 证明不是 CLIP 全局向量即可 |
| DINO-CLS/A1 | DINO CLS distance | A1 | 证明不是任意 global token |
| shuffled-TEXT/A1 | 打乱 text score | A1 | 信息性控制 |

像素图完全相同的系统必须报告完全相同 Pixel-AP/AUPRO；若不同，说明实现有泄漏或错位。

### 4.3 论文可用与不可用表述

G1 通过后可以写：

> We observe that target-domain zero-shot global anomaly probabilities from a source-trained vision-language prompt model complement local few-shot visual memory, motivating a task-decoupled detector that uses language evidence for image screening and visual memory for pixel localization.

不能写：

- “we propose AnomalyCLIP text probability”；
- “fully training-free”，因为 prompt learner 有 VisA 来源训练；
- “text improves localization”，因为 Pixel-AP 校准门失败；
- “new multimodal fusion network”，因为 GLSD 没有学习/新网络；
- “VisA external zero-shot validation”。

### 4.4 Phase 2 冻结物

- `METHOD_SPEC_GLSD.md`；
- 一条确定性重放命令；
- 代码/config/checkpoint/cache SHA256；
- latency、峰值 VRAM、参数来源；
- 数据角色表；
- `FREEZE_MANIFEST.json`；
- 当前 commit tag/hash；
- “GLSD 创新程度中等偏弱、主要为任务设计与实证发现”的书面边界。

---

## 5. Phase 2R：可选的真正算法升级——TCRR 文本条件区域重排序

仅当同时满足以下条件才启动：

1. G1 通过；
2. 本次用户请求已明确要求在证据允许时继续追求更强算法创新；
3. 完成 §5.1 文献边界审计；
4. R0 信息价值门通过。

暂定名称：**TCRR（Text-Conditioned Region Re-ranking）**。名称只是占位，R2 通过前不得写进论文贡献。

### 5.1 文献边界审计（先做，不能跳）

必须读原论文/官方代码并形成 `02_region_route/LITERATURE_BOUNDARY.md`：

- AnomalyCLIP：已有文本提示与异常图；
- ReMP-AD：已有 retrieval、global prototype、vision-language prior fusion 和潜在异常区域定位；
- UniVAD：已有 component clustering、component-aware matching 和 component graph；
- PALADIN：已有 DINOv3 与语言提示对齐定位；
- WinCLIP/PromptAD/VCP-CLIP 等近邻方法。

候选只能定位为：

> 对**独立局部正常记忆 A1 产生的候选区域**做 prompt-conditioned semantic re-ranking；不修改视觉编码器注意力，
> 不从文本图直接生成分割，不复现 ReMP-AD 的 VLPF，也不复现 UniVAD 的组件匹配图。

若审计发现完全等价方法，标记 `NOVELTY_COLLISION`，停止 TCRR，不得换名继续。

参考原始来源：

- ReMP-AD：<https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html>
- UniVAD：<https://www.openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html>
- PALADIN：<https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html>

### 5.2 为什么区域级可能比 scalar gate 更合理

当前 scalar gate 对一张图所有像素乘同一个权重，只能改变跨图排序，不能改变图内缺陷位置。TCRR 改为：

1. A1 先提出少量高分连通区域；
2. 在每个区域内汇聚 AnomalyCLIP patch 与 normal/abnormal prompt 的相似度差；
3. 根据“该区域是否具有异常语义”调整该区域，而不是整张图统一放大；
4. 区域外保持 A1 不变，以降低破坏原定位的风险。

### 5.3 R0：只做信息价值诊断，不先造完整方法

数据限制：MPDD seed0 × shot{1,2,4}，development only。

候选区域生成必须与 GT 无关：

- 从 normal-reference A1 score 分布冻结 90/95/97.5% 三个阈值；
- 二值化测试 A1 map，8-connectivity，去除面积 < 4 patch 的区域；
- 阈值和最小面积在读取测试 GT 前写入 `region_protocol.json`；
- GT 只在 evaluator 中给候选区域打“是否与缺陷相交/IoU”的诊断标签。

每个候选区域输出：

- A1 mean/max/top-q severity；
- 区域面积、紧致度；
- patch-level text margin `sim(f_patch,t_abnormal)-sim(f_patch,t_normal)` 的 mean/max/trimmed mean；
- 区域内 text-margin 一致性；
- global `p_abn`，只作控制。

R0 判断“区域文本统计是否比 A1 severity 更能区分真缺陷区域和假警报区域”。至少报告 region-AP、AUROC、
Spearman、每类结果和样本数。

R0 硬门：

1. 最佳**预注册 pooling** 的 region-AP 相对 A1 severity ≥ +0.050；
2. 至少 4/6 类为正；
3. 打乱 prompt/区域对应后增益回落 ≥ 0.030；
4. 不由单个类别贡献超过 50% 的正增益；
5. 至少 100 个正区域和 100 个负区域，否则标记证据不足，不训练/不开发。

R0 失败即归档；不要开发 R1。

### 5.4 R1：最小可解释区域重排序

只允许两个候选，禁止大网格：

```text
C0 semantic-boost:
M'(p in R) = M_A1(p) * [1 + lambda * sigmoid(z_text(R))]

C1 semantic-veto-and-boost:
M'(p in R) = M_A1(p) * clip[1 + lambda*z_text(R), 0.75, 1.50]
```

- `z_text(R)` 由 normal reference 的区域级 prompt margin 做 median/MAD 或经验尾概率标定；
- `lambda` 固定为 0.5；不得逐类、逐 shot 选择；
- 区域外 `M'=M_A1`；
- 不训练神经网络，不读取异常 mask 选公式；
- 若 normal reference 无法形成足够区域，必须明示 fallback，禁止借测试正常图校准。

强制控制：

- A1；
- S1 scalar global gate（+0.0040 基线）；
- raw text map 与 A1 简单平均；
- region pooling 但不含文本；
- shuffled prompt、inverted prompt；
- shuffled region assignment；
- 相同参数量/相同后处理的随机分数控制。

### 5.5 R1 小门

MPDD seed0 × shot{1,2,4}，相对冻结 A1：

- pooled mean ΔPixel-AP ≥ **+0.008**；
- 3/3 shot 为正，worst shot ≥ −0.003；
- mean ΔAUPRO ≥ 0；mean ΔPixel-AUROC ≥ −0.002；
- worst category mean ΔPixel-AP ≥ −0.020；
- 胜过 scalar global gate ≥ +0.003；
- 胜过 raw text-map average ≥ +0.003；
- 胜过 shuffled/inverted prompt ≥ +0.005；
- 所有几何、ID、泄漏和复现测试通过。

最多一个 winner。未通过则 TCRR 归档，GLSD/A1 决策不受影响。

### 5.6 R2 Full MPDD

唯一 winner 原样运行 3 seeds × 3 shots：

- mean ΔPixel-AP ≥ **+0.010**；
- 至少 7/9 配置为正；
- worst category ≥ −0.020；
- 95% paired bootstrap CI 下界 > 0；
- scalar gate/raw map/shuffle 机制控制仍成立；
- 固定 hash 后才允许进入 Phase 3。

R2 通过后，TCRR 才能成为“新算法”；此前一律称候选或诊断。

---

## 6. Phase 3：冻结外部验证

### 6.1 数据角色

| 数据集 | 角色 | 可否选参数 | 说明 |
|---|---|---|---|
| MPDD | development | 仅按上述冻结网格/门槛 | 算法选择与 Full development |
| BTAD | unseen external validation | 否 | 一次性运行 |
| MVTec AD | unseen external validation | 否 | 一次性运行 |
| VisA | source/in-domain audit | 否 | prompt checkpoint 来源相关，不能称独立外部验证 |

### 6.2 进入条件

- GLSD：Phase 1 G1 通过并完成 Phase 2 freeze；
- TCRR：R2 通过并单独 freeze；
- 外部验证时只运行一个最终规格；如果 TCRR 通过则验证 TCRR+GLSD，否则验证 GLSD；
- 不允许看到 BTAD/MVTec 后回 MPDD 换 beta、阈值、prompt、区域面积或模型。

### 6.3 外部执行

复用同一 AnomalyCLIP checkpoint、通用 `object` learned prompt、输入尺寸和 score direction，分别导出 BTAD/MVTec。
对每个数据集报告：

- Image-AP、Image-AUROC；
- Pixel-AP、Pixel-AUROC、AUPRO；
- 逐类、逐 shot、逐 seed；
- 相对 A1 paired delta 与 bootstrap CI；
- 运行时间、显存、缓存体积；
- 失败类别和可视化。

外部推广判定（不是删结果门，所有结果必须公开）：

- BTAD/MVTec 两数据集平均 ΔImage-AP ≥ 0；
- 至少一个数据集为正，另一个不得低于 −0.020；
- 若 TCRR 进入，平均 ΔPixel-AP ≥ +0.005，任一数据集不得低于 −0.020；
- GLSD 的 Pixel metrics 必须与 A1 identity 一致。

VisA 可在最后运行 source/in-domain audit，但单独成表，禁止与两个 unseen external 数据集求平均后宣传泛化。

---

## 7. 预先假设的五种结果与自动决策

### Scenario A：最佳情况——文本稳定，区域算法也通过，外部验证通过

条件：G1、R0、R1、R2、Phase 3 全通过。

行动：

1. 主方法升级为 TCRR + GLSD；
2. 论文贡献可写为：局部正常记忆、全局文本筛查、文本条件候选区域重排序；
3. 必须用 ReMP-AD、UniVAD、AnomalyCLIP 控制划清边界；
4. 重做方法图、主结果表、消融表、效率表、可视化和英文 Method；
5. A1 作为 base model，S1 scalar gate 作为消融；
6. 目标仍定位 SCI 四区/偏应用期刊，不夸大 SOTA。

### Scenario B：文本稳定，但区域算法失败；外部图像级验证通过

条件：G1 通过，R0/R1/R2 任一失败，GLSD 外部图像指标通过。

行动：

1. 保留 A1 作为像素主方法；GLSD 作为任务解耦扩展/第二贡献；
2. 明确“语言改善图像筛查而不改善定位”的边界；
3. 论文定位为 training-free local representation fusion + source-trained global language evidence 的系统实证；
4. 创新程度中等偏弱，但凭四数据集、完整负结果、复现与边界分析可尝试 SCI 四区；
5. 不再追区域算法，转入论文写作和目标期刊筛选。

### Scenario C：Full MPDD confirmation 失败

条件：G1 任一硬门失败。

行动：

1. 文本 +0.0249 降级为 seed0 exploratory observation；
2. 不运行外部文本验证、不开发 TCRR；
3. A1 保持唯一方法；
4. S1 进入 Appendix/Discussion 负结果；
5. 论文走“简单双视觉表征 + 严格实证与复现”路线，或重新评估投稿价值。

### Scenario D：MPDD 通过，但 unseen external 失败

行动：

1. 不回调参数，不验证第二名；
2. 将文本结果定义为 MPDD-specific / limited transfer；
3. 不把 GLSD/TCRR 放主标题；A1 恢复为主方法；
4. 外部失败结果必须保留，写入 limitations；
5. 若 A1 本身外部证据仍稳定，则继续 A1 实证论文；否则先暂停投稿、审查协议与目标期刊。

### Scenario E：数值通过，但创新审计与已有工作碰撞

例如 TCRR 与 ReMP-AD VLPF 或 UniVAD component modeling 实质等价。

行动：

1. 性能结果可作为内部探索或基线，不声称新方法；
2. 不靠改名解决 novelty collision；
3. 回到 Scenario B 的任务解耦/实证论文；
4. 若期刊明确要求较强算法创新，则暂停当前投稿而不是制造夸大 claim。

---

## 8. 论文创新程度的最终评级规则

| 最终证据 | 可给的创新评级 | 投稿策略 |
|---|---|---|
| 只有 A1 | 中等偏弱：固定双视觉表征融合 + 严格实证 | SCI 四区应用/工程/实证型，突出复现与边界 |
| A1 + GLSD，外部通过 | 中等：任务解耦 + 跨模态全局证据，但核心分数来自现成模型 | 可投四区，避免“新网络/SOTA”叙事 |
| A1 + TCRR，Full/外部/机制控制全通过 | 中等偏强：新的区域级跨模态机制 | 可作为主算法论文，完整消融和近邻差异是必要条件 |
| 仅 seed0 或仅 +0.004 像素增益 | 不足 | 只能作为探索/附录，不作主贡献 |

任何“创新足够”的结论必须同时看：数值、稳定性、外部泛化、机制控制、与已有工作的非重复性。只满足其中一项不算通过。

---

## 9. 执行纪律、资源与提交

### 9.1 执行纪律

- MPDD 的 threshold/grid 到本文件为止冻结；后续不得因差 0.001 临时调参；
- labels/masks 只由 evaluator 加载；导出器不接触；
- 所有失败候选保留，不只保存 winner；
- 不删除旧证据、不覆盖 v6 JSON；
- 大 npz 放 gitignored outputs/cache，小 JSON/CSV/MD/图表入库；
- 每个 Phase 一次独立 commit，commit message 写清 PASS/FAIL；
- 执行中发现代码 bug：先写 `BUG_AUDIT.md`，修复后从受影响 Phase 全量重跑，不手改结果文件；
- 外部集一旦打开，记录时间、commit 和 freeze hash；禁止回流调参。

### 9.2 粗略资源规划

| 阶段 | 主要资源 | 预计量级 |
|---|---|---|
| Phase 0 | CPU/磁盘审计 | 1–2 小时 |
| Phase 1 | CPU 指标 + bootstrap | 1–3 小时，无需重跑 AnomalyCLIP |
| Phase 2 | 文档/重放/效率 | 1–2 小时 |
| R0 | AnomalyCLIP patch 导出 + 区域统计 | 数小时 GPU |
| R1/R2 | CPU 融合为主，必要时复用 patch cache | 半天至一天 |
| Phase 3 | BTAD/MVTec AnomalyCLIP 导出 | 数小时 GPU，必须冻结后一次性 |

实际耗时必须记录，不以本表作为验收依据。

### 9.3 每阶段统一交付物

```text
protocol.json
input_hashes.json
per_config.csv
per_category.csv
summary.json
bootstrap.json（适用时）
leakage_audit.json
resource_usage.json
DECISION.md
reproduce_commands.md
```

### 9.4 最终必须更新

- `docs/paper_writing_preparation_20260830/README.md`
- `docs/paper_writing_preparation_20260830/05_CLAIM_EVIDENCE_MATRIX.md`
- `docs/paper_writing_preparation_20260830/07_MISSING_MATERIALS_AND_CHECKLIST.md`
- `docs/paper_writing_preparation_20260830/10_PROJECT_STATUS_FOR_SUPERVISOR_CN_20260901.md`
- `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`
- `docs/PAPER_DETAILED_CHINESE_DRAFT_20260827.md`（只写已经通过的事实）
- 若进入新方法：英文 Method/Experiments/Results、图表计划和复现包规格。

---

## 10. 下一位 AI 的简明执行清单

```text
[ ] 读取 16、17 号任务书和 v6 S1_HGLC_DECISION
[ ] 记录 HEAD/dirty state，完成 Phase 0 审计与测试
[ ] 不重跑文本模型，先用现有 cache 补齐 MPDD 3×3 image evidence
[ ] 做 paired bootstrap 和 seed1/2 confirmation
[ ] 严格判断 G1，并写 PHASE1_DECISION
[ ] G1 失败：归档，更新论文材料，停止
[ ] G1 通过：冻结 GLSD 双输出规格
[ ] 做文献边界审计；若无 novelty collision，执行 R0 区域信息价值诊断
[ ] R0 失败：归档 TCRR；保留 GLSD，进入外部冻结验证
[ ] R0 通过：只实现 C0/C1，跑 R1 小门
[ ] R1 通过：唯一 winner 跑 R2 Full MPDD；否则归档
[ ] 冻结唯一最终规格，运行 BTAD/MVTec 一次性验证
[ ] VisA 仅作 source/in-domain audit，单列
[ ] 按 Scenario A–E 更新全部状态、论文、图表和 claim
[ ] 运行相关测试与 git diff --check，提交最终决策
```

---

## 11. 本任务书的最终判断

最值得立即做的不是继续调 +0.004 的像素 gate，而是先用现有缓存完成 **Full MPDD + bootstrap**，确认文本的
图像级 +0.0249 是否为稳定事实。这一步通过，GLSD 才可成为论文的第二实证贡献；如果还需要更强创新，
再用 R0 判断“区域级文本证据”是否真的能区分 A1 真缺陷与假警报。只有 R0/R1/R2 和外部验证全部通过，
TCRR 才有资格成为新算法。

该顺序同时保护两件事：不把现成 AnomalyCLIP 输出冒充创新，也不因当前多数负结果而错过真实的跨模态信息价值。
