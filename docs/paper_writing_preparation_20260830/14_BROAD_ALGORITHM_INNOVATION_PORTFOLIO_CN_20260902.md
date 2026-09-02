# 更宽算法创新版图与路线优先级（A4 Research Portfolio）

版本：2026-09-02  
用途：突破“只研究双编码器融合”的思维边界，为后续算法选择和预注册提供依据  
状态：**研究构思与路线筛选文件，不代表任何方法已经实现或有效，也不授权一次性运行全部路线。**

---

## 0. 结论先行

当前 A3 的 CASF 和 DC-SZoom 比 A2 更有希望，但仍主要围绕“已有双编码器怎样融合”。如果希望形成更强的算法创新，应该把研究问题从“如何融合两个分数”扩大为：

1. **正常结构应该怎样被定义？**——局部外观正常不等于放在当前位置/上下文中正常；
2. **基础视觉编码器遗漏了什么？**——高频划痕、周期纹理破坏和极细边缘可能在 token 化时消失；
3. **检测单位是否一定是固定 patch？**——部件缺失、数量错误和装配关系异常需要组件/关系层检测；
4. **异常分数是否一定来自最近邻？**——可以来自被遮挡 token 的反事实正常预测、关系图失配或正常修复残差；
5. **是否允许改变学习协议？**——使用辅助源域异常监督可能显著提高性能，但会改变 normal-only few-shot 的公平性与论文身份。

综合创新性、与现有失败路线的独立性、本地可执行性和 SCI 四区论文风险，推荐顺序为：

| 优先级 | 路线 | 创新幅度 | 资源 | 推荐结论 |
|---:|---|---|---|---|
| 1 | I：RG-MCR 参考引导掩码上下文修复 | 高 | 中 | 最值得作为新的主攻路线 |
| 2 | J：SF-NM 频谱—基础特征双正常记忆 | 中 | 低 | 最适合做低成本独立小门 |
| 3 | K：RG-OT 关系图最优传输异常 | 高 | 中/高 | 高风险高新意，先做小规模机制 smoke |
| 4 | A3-G：CASF 合成异常融合 | 中/高 | 中 | 保留，但不再是唯一主路线 |
| 5 | A3-H：DC-SZoom 稀疏高分辨率 | 中 | 高 | 小缺陷证据成立时再做 |
| 6 | L：组件感知正常图 | 中 | 中/高 | 与 UniVAD 较近，需明显差异后再立项 |
| 7 | M：跨数据集 episodic 元学习 | 高 | 高 | 可能涨点，但会改变协议，只作战略备选 |
| 8 | N：扩散反事实正常修复 | 高 | 很高 | 当前 6GB 级 GPU 风险大，不优先 |
| 9 | O：conformal/不确定性评分 | 低/中 | 低 | 适合部署补充，不足以单独支撑主创新 |

新的首选主线不是“造更复杂融合网络”，而是：

> 用正常参考图训练一个只看周围上下文和参考正常 token、看不到中心 patch 的小型预测器，为每个 query patch 生成其“在正常结构下应该是什么”的反事实特征；真实特征与反事实正常特征的残差作为结构异常证据，再与 A1 外观异常证据做严格受控组合。

它直接回答 A1 当前无法回答的问题：**一个 patch 即使能在全局记忆里找到相似正常 patch，放在错误位置或错误邻域中是否仍应判为异常？**

---

## 1. 已关闭路线与新路线边界

新思路不能只是给旧失败方法换名字。

| 已失败/受限方向 | 不能怎样复活 | 新路线为何不同 |
|---|---|---|
| DSAM 局部空间窗口 | 继续缩放窗口或换更精细配准 | RG-MCR 不是限制 KNN 搜索位置，而是遮掉中心后从上下文预测正常中心 |
| FAGR score map 图平滑 | 换 8 邻域或更强平滑 | RG-OT 比较节点与边关系，不在最终分数上做平滑 |
| NCPRA 跨编码器同位置预测 | 换更大 bottleneck | RG-MCR 预测的是“被隐藏的空间中心”，输入不包含中心 token，科学问题是上下文条件正常性 |
| LNDC/LOF 密度 | 换核函数或扩大 k | SF-NM 引入编码器之外的频率信息，不重新估计相同 feature density |
| CEQA/FastRef 类 query shift | 扩大 q/eta | 新路线不把 query patch 写入/移动 normal memory |
| DEVA 参考增强 | 增加增强数量 | SF-NM/RG-MCR 改变观测量或正常性目标，不靠重复增强扩库 |
| PCA/CCA/subspace | 改维度或 whitening | 新路线不做全局线性子空间残差 |
| 动态 branch router | 换权重网络 | CASF 若继续，必须由伪异常监督并胜过对称/去分歧控制；其他路线不依赖 branch routing |

---

## 2. 路线 I：RG-MCR——参考引导掩码上下文修复

英文候选名：**Reference-Guided Masked Context Repair**。

### 2.1 核心科学问题

A1 的全局 KNN 只问“这个 patch 在正常记忆里是否见过”。它不能区分：

- 正常零件出现在错误位置；
- 螺钉、孔洞、线缆等部件数量或排列错误；
- 局部纹理看似正常，但与周围结构不连续；
- patch 被邻近正常区域污染，末层距离仍不大。

RG-MCR 改问：

> 在看不到中心 patch 的前提下，只依据周围上下文和少量正常参考，正常中心特征应该是什么？

如果 query 中心与预测的正常中心差异大，则它在结构上异常。

### 2.2 最小可行算法

对 DINO 和 CLIP 对齐后的 feature grid，逐 patch 构造：

- query context ring：中心周围 `5×5` 去掉中心 `3×3`，避免直接复制异常；
- reference context bank：从 k 张正常参考图取相同归一化位置附近和全局 top-K 相似上下文；
- 2D 相对位置编码；
- 中心位置放置一个 learned mask token，不输入真实中心 feature。

小型模型：

```text
context tokens + reference support tokens
  → shared projection 768→96
  → 2-layer cross-attention transformer
  → predicted normal center feature D_hat / C_hat
```

只在正常参考图上随机遮中心训练：

\[
L_{repair}=1-\cos(\hat d,d)+1-\cos(\hat c,c)
+0.1L_{cross-view}+0.1L_{smooth-prior}.
\]

测试分数：

\[
s_{ctx}=\tfrac12[(1-\cos(\hat d,d_q))+(1-\cos(\hat c,c_q))].
\]

用 reference-only LOO 标准化后与 A1 做有界组合：

\[
s=s_{A1}+\lambda\,\mathrm{softplus}(z_{ctx}-\tau_{ref}).
\]

`lambda`、`tau_ref` 必须预注册并只用正常 LOO 设定；不能用测试 mask。

### 2.3 与 NCPRA 的本质区别

- NCPRA 输入当前中心 DINO，预测当前中心 CLIP，异常中心信息已经进入网络；
- RG-MCR 完全遮掉 query 中心，只从邻域与正常参考预测；
- NCPRA 学跨编码器映射，RG-MCR 学“空间条件下的正常结构”；
- RG-MCR 可检测“局部外观正常但上下文错误”，这是全局 KNN 和同位置跨编码器预测都不直接解决的现象。

### 2.4 关键控制

- `CTRL-COPY`：允许输入中心 token；若它也有效，说明可能只是自编码复制；
- `CTRL-POS`：只用位置与正常参考，不用 query context；
- `CTRL-CTX`：只用 query context，不用正常参考；
- `CTRL-SHUFFLE`：打乱 context-center 对应，应该失败；
- `CTRL-DINO`：只预测 DINO；
- `CTRL-A1`：原始 A1。

要把“参考引导上下文修复”写成贡献，完整方法必须同时胜过 `CTRL-POS` 和 `CTRL-CTX`，证明参考与上下文都必要。

### 2.5 风险与解决办法

| 风险 | 约束 |
|---|---|
| 1-shot 只有一张参考图，训练过拟合 | 随机遮挡产生大量 patch 任务；参数量限制 `<0.5M`；强制 leave-region-out，而不是随机 patch 泄漏 |
| 周围 context 也含异常 | 去掉中心 3×3；对 context attention 做 trimmed top-K；不得使用预测的低分 patch 自适应选 context |
| 预测器输出均值，细节模糊 | cosine residual + reference prototype cross-attention；比较单一均值控制 |
| 纹理类没有固定结构 | A1 bounded residual 保底；不得手写类别 fallback |
| 与 PNI/INP-Former 接近 | 强调 few-shot reference-guided masked dual-encoder counterfactual；写论文前做逐公式近邻核验 |

### 2.6 推荐小门

先用 MPDD seed0 × shot {1,2,4}，候选不超过三种：context ring `{3→5, 5→7}` 与是否使用 cross-attention。必须满足：

- mean ΔPixel-AP vs A1 `≥+0.006`；
- 至少 2/3 shot 正，worst `≥−0.005`；
- 相对 `CTRL-CTX` 和 `CTRL-POS` 各至少 `+0.002`；
- 结构错位合成 smoke 明显优于 A1；
- 真实 MPDD 选择只看预注册总体门，不按类别救规则。

近邻依据：[Predictive Convolutional Attentive Block, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.html)、[FastRecon, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.html)、[INP-Former, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_Exploring_Intrinsic_Normal_Prototypes_within_a_Single_Image_for_Universal_CVPR_2025_paper.html)。

---

## 3. 路线 J：SF-NM——频谱与基础特征双正常记忆

英文候选名：**Spectral-Foundation Normal Memory**。

### 3.1 为什么是独立信息源

DINO/CLIP 的 patch stride、resize 和深层语义会降低高频敏感度。SF-NM 不再从同一 768 维 token 推导新分数，而是直接从原图构造局部频谱描述，针对：

- 细划痕、毛刺、裂纹；
- 周期纹理中断；
- 局部方向频率改变；
- 颜色不明显但表面粗糙度变化。

### 3.2 最小可行算法

对每张原图统一保长宽比预处理，使用固定 stationary wavelet transform 或 steerable filters：

- 2 个尺度；
- `LH/HL/HH` 三个高频带；
- luminance + 两个 opponent-color 通道；
- 每个 DINO patch 覆盖区域计算 log-energy、方向比、局部谱熵和相邻能量差；
- descriptor 维度控制在 32 以内，按正常 reference robust normalize；
- 建立 spectral memory，得到 `s_freq`；
- 与 A1 通过 reference LOO 尾概率组合，而不是训练类别权重。

推荐组合：

\[
p_{A1}=\frac{1+\#\{r:s_r\ge s_{A1}\}}{n+1},\quad
p_f=\frac{1+\#\{r:f_r\ge s_f\}}{n+1},
\]

\[
s_{joint}=-\log(p_{A1}p_f).
\]

少样本下 empirical tail 太粗时，用跨 patch reference LOO + block bootstrap 校正，不能使用测试分布拟合。

### 3.3 机制控制

- `CTRL-LOW`：只用低频 LL；
- `CTRL-BLUR`：输入先高斯模糊再提高频；
- `CTRL-RGBGRAD`：普通 Sobel/gradient magnitude，同维度；
- `CTRL-RANDOMBANK`：打乱频带与位置；
- `CTRL-FREQONLY` 与 `CTRL-A1`。

完整方法必须胜过 RGB gradient 控制，才能证明频谱建模而不只是边缘强度。

### 3.4 优势与限制

- 优势：CPU 可运行、无需新 backbone、与 A2 末层特征后处理真正独立；
- 限制：对缺件、错装等逻辑异常帮助有限；姿态/光照变化可能制造高频假阳性；
- 论文新意中等，单独作为主方法需要清楚的频带选择机制和跨数据集稳定增益；否则更适合作为第二个互补证据分支。

推荐小门：MPDD seed0 三 shots，mean ΔPixel-AP `≥+0.004`，至少 2/3 正，且相对 `CTRL-RGBGRAD ≥+0.0015`。若频域与 A1 逐 patch rank 相关均值 `>0.95`，直接停止。

近邻依据：[Wave-MambaAD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Wave-MambaAD_Wavelet-driven_State_Space_Model_for_Multi-class_Unsupervised_Anomaly_Detection_ICCV_2025_paper.pdf)。该工作说明频率建模已有先例，所以不能把“用了 wavelet”本身写成创新。

---

## 4. 路线 K：RG-OT——关系图最优传输异常

英文候选名：**Relational Graph Optimal-Transport Anomaly Scoring**。

### 4.1 科学问题

固定 patch KNN 忽略 token 之间的关系。DSAM 用局部位置窗口修正这一点，但局部窗口会阻止全局正常匹配。RG-OT 不规定 rigid window，而是同时比较：

- 节点外观是否匹配；
- 节点与其他节点的相对空间/语义关系是否匹配。

因此它允许整体形变，同时对错位、缺件、重复部件、次序错误敏感。

### 4.2 最小可行算法

1. 将 32×32 token 通过正常 reference 上固定 coreset/聚类缩减为 64 个 anchor；
2. 节点属性为 A1 fused descriptor；
3. 边属性为归一化空间距离、DINO 相似度和 CLIP 相似度；
4. query 与每张 reference 之间求 entropic fused Gromov-Wasserstein transport；
5. patch 分数由 node transport cost 与 edge distortion 的 barycentric attribution 得到；
6. 多 reference 取最小或 reference-only softmin；
7. 与 A1 做预注册有界组合。

### 4.3 必须控制

- `CTRL-OT-NODE`：只比较节点、不比较边；
- `CTRL-SHUFFLE-EDGE`：打乱正常图边；
- `CTRL-COORD`：只用空间边；
- `CTRL-FEATURE`：只用特征关系边；
- `CTRL-DSAM`：引用 A2 已有结果，不重跑调参；
- `CTRL-A1`。

只有 RG-OT 胜过 node-only OT，才能证明关系图贡献。

### 4.4 风险

- O(N²) 边和迭代 OT 成本高；先限制 64 anchors、20 次 Sinkhorn；
- 归因回 patch 容易产生粗糙图；必须报告 Pixel-AUPRO 与边界误差；
- 与 UniVAD 的 component graph 方向接近，不能宣称首次图建模。区别必须落在“fused relational transport + patch-level edge distortion attribution”，并在写作前核验公式。

只做一个 smoke 配置：MPDD seed0/shot2。先在由正常图人工打乱 5% anchors 的合成结构异常上验证 `CTRL-OT-NODE` 无法识别而 RG-OT 能识别；机制 smoke 不成立则不进入真实小门。

近邻依据：[UniVAD, CVPR 2025](https://www.openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html)、[PNI, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.pdf)。

---

## 5. 路线 L：组件感知正常图

思路：先把图像分成产品组件/区域，再分别建立组件 appearance memory、数量/面积统计和组件关系图。这比固定 patch 更适合缺件和装配错误。

当前不列为前三优先级，原因：

- 本项目没有现成 SAM/自动组件模型和权重；
- 引入 SAM 会增加 checkpoint、许可、显存和复现负担；
- UniVAD 已包含 component clustering、component-aware matching 和 graph-enhanced modeling，近邻重合较强；
- 四个当前数据集以纹理/局部表面缺陷为主的类别不少，组件级收益可能被宏平均稀释。

如果后续做，优先使用 DINO token 自聚类/normalized-cut，而不是立即引入新大模型。创新必须是“跨编码器组件不确定性或组件关系的具体新机制”，不能只是“用 SAM 切一下再跑 A1”。

---

## 6. 路线 M：跨数据集 episodic 元学习

### 6.1 思路

训练一个 universal reference interpreter：每个 episode 给 k 张正常 support 和带标签 query，从源数据集学习怎样根据 support 判断 query；测试时在新数据集只给正常 reference，不再微调。

可利用 MPDD 作为 source，BTAD/MVTec/VisA 作为 target，或者做 leave-one-dataset-out。模型可以采用 RG-MCR/CASF 结构，但由真实 source anomalies 提供监督。

### 6.2 为什么可能更强

它不再要求只靠一张正常图学习“异常应该长什么样”，而是从其他产品的真实缺陷学到通用偏离模式。InCTRL 等 generalist anomaly detection 工作说明“辅助数据训练 + few-shot sample prompts”是成立的研究设定。[InCTRL, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.pdf)

### 6.3 协议代价

- 不再是严格 target-normal-only training-free 方法；
- 与 A1 的公平性口径改变；
- MPDD 若用于 source train，就不能再作为同意义的 development test；
- 数据集许可、类别重合和 source-target checkpoint 关系要重新审计；
- 需要新的论文标题、baseline 与实验设计。

因此它是“另立论文协议”的战略路线，不应混进当前 A1/A3 实验。如果老师更重视算法结构和性能、能接受辅助监督，可另立 A5；否则暂不做。

---

## 7. 路线 N：扩散反事实正常修复

思路：先由 A1 提候选区域，再用正常参考条件扩散/生成模型把候选区域修复成正常外观，原图与修复图的 feature residual 定位异常。

优点：反事实解释直观，图像层修复可产生很强的论文视觉效果。相关方向已有 AnoDDPM 类工作、AnomalyAny 和 Removing Anomalies as Noises。[AnomalyAny, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Unseen_Visual_Anomaly_Generation_CVPR_2025_paper.html)、[Removing Anomalies as Noises, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.pdf)

缺点：

- 当前机器显存较小，扩散推理和多候选区域成本高；
- 生成伪影容易被误当异常；
- 参考图条件和产品身份保持很难；
- 近邻工作密集，单纯“扩散修图 + residual”创新不足。

除非获得更强 GPU，并能提出“reference identity preservation + uncertainty-aware repair”的明确机制，否则不优先。

---

## 8. 路线 O：不确定性与 conformal evidence

可把每个分支/层/频带的 reference LOO residual 转成有限样本 p-value，再用 e-value、Fisher/Cauchy combination 或 conformal risk control 合并，以获得可解释的异常置信度和固定误报率。

价值主要在：

- 阈值可解释；
- 跨类别分数可比；
- 工业部署时控制 false alarm；
- 可为 image-level decision 提供统计保证。

但 AP/AUROC 对单调变换不敏感，纯 conformal calibration 很可能不涨主指标，因此不应作为当前论文唯一算法创新。它适合 winner 产生后的部署/附录扩展。

---

## 9. 对现有 A3 两条路线的进一步优化

### 9.1 CASF 的优化空间

CASF 最大风险是模型学会识别伪异常生成器，而不是真实缺陷。建议增加：

- corruption-family leave-one-out：一个伪异常家族只用于验证，不参与训练；
- hard anomaly mining：只保留 A1 无法轻易分开的伪异常，不用巨大噪声；
- clean ranking preservation：正常 reference LOO 排序不能被明显破坏；
- cross-dataset frozen generator test：伪异常强度规则在 MPDD 冻结后，不能为其他数据集重设；
- `CTRL-GEN-ID`：给 generator 类型 ID 的上界实验只能做诊断，正式模型不得输入类型 ID。

若 CASF 在伪异常验证 Dice 很高、真实 MPDD 不涨，应立即停止，不扩大网络。

### 9.2 DC-SZoom 的优化空间

DC-SZoom 最大风险是低分辨率 proposal 根本看不到小异常。可做的非标签改进：

- proposal 同时加入固定高频 saliency，但这会与 SF-NM 发生组合，必须先分别过门；
- 保留一个 uniform coverage window，防止全部窗口聚到同一区域；
- reference crop 不强制 rigid 同坐标，可用 query window 的全局 fused descriptor在 reference 上找 top-2 对应窗口；
- local map 与 global map 用正常 LOO quantile 对齐，不能直接比较不同分辨率的 raw distance；
- 先测异常面积分层 headroom，若小面积组没有明显提升潜力则停止。

### 9.3 更合理的角色分配

- CASF：学习“跨编码器证据怎样解释”；
- DC-SZoom：找回小尺度输入信息；
- RG-MCR：学习“正常结构下中心应该是什么”；
- SF-NM：引入独立频率观测；
- RG-OT：检测关系/装配异常。

这些路线回答不同问题。后续只有各自单独通过 Full MPDD，才允许考虑组合；不能一开始堆成一个大系统。

---

## 10. 推荐的实际执行顺序：先做信息诊断，再选择路线

为了避免同时运行 5 条路线造成多重搜索，先做三个固定诊断，全部只在 MPDD development 上：

### D1：缺陷尺度与频率诊断

- 按 GT 异常面积比例预注册分为 small/mid/large；
- 只由 evaluator 做分层，不把 mask 交给方法；
- 比较 A1 在各层的失败程度；
- 测试一个无参数 wavelet score 的 oracle complementarity headroom；
- 若 small 组 headroom `≥+0.03 Pixel-AP` 且 rank correlation `<0.90`，优先 SF-NM/DC-SZoom。

### D2：结构上下文诊断

- 只在正常 reference feature 上合成 patch permutation、missing-block、duplicate-block；
- 比较 A1、context prediction residual、node-only OT；
- 不使用真实异常调模型；
- 若 RG-MCR 对三类结构扰动 AUROC 均 `≥0.80` 且比 A1 高 `≥0.10`，优先 RG-MCR；
- 若 RG-MCR 不行但 relation OT 明显有效，才进入 RG-OT smoke。

### D3：双分支监督价值诊断

- 使用 A3 CASF 已规定的 asymmetric corruption；
- 比较 symmetric/no-disagreement controls；
- 若机制控制差值 `<0.02` pseudo Dice 或真实小门 `<+0.002 Pixel-AP`，CASF 降级，不扩大训练。

### 路由规则

```text
D1 通过 → 先跑 SF-NM；只有 SF-NM 显示高频有效，再决定是否花 GPU 跑 DC-SZoom
D2 通过 → 先跑 RG-MCR；只有 relation-specific smoke 胜过 context 才跑 RG-OT
D3 通过 → 保留 CASF；否则不再以双分支门控为主线
```

最多允许两个方向进入完整 MPDD。外部验证仍只允许一个冻结 winner。

---

## 11. 不同创新幅度对应的论文风险

| 层级 | 例子 | 优点 | 风险 |
|---|---|---|---|
| 低风险增量 | SF-NM、conformal | 实现快、容易解释 | 新意可能不足，只适合作为模块/补充 |
| 中等创新 | CASF、DC-SZoom | 与现有 A1 自然衔接 | 容易被评价为已有合成/多尺度/切块的组合 |
| 较强方法创新 | RG-MCR、RG-OT | 改变正常性定义或匹配目标 | 实现与验证更难，必须有机制消融 |
| 协议级创新 | 跨数据集 episodic meta-learning | 性能潜力与模型复杂度高 | 失去当前 strict normal-only、training-free 身份 |
| 生成式高风险 | diffusion counterfactual | 视觉结果强、可解释 | 资源重、伪影、近邻工作密集 |

对于当前目标——中文构思、英文撰写、SCI 中科院四区——最稳妥的选择不是追求最大网络，而是选择一个科学问题清晰、能做完整消融且能跨数据集验证的方法。RG-MCR 最符合这一点；SF-NM 是最好的低成本互补探针。

---

## 12. 建议下一位 AI 接到的具体任务

下一轮不要直接实现本文件全部内容。建议执行指令为：

1. 保留 A3 Wave 0 的 DEVA/NCPRA 有效性修复；
2. 新建 `innovation_v4_diagnostics`，只实现第 10 节 D1/D2/D3 三个信息价值诊断；
3. RG-MCR 只实现 synthetic structural smoke 和正常 reference masked reconstruction，不立即跑四数据集；
4. SF-NM 只实现一个固定两尺度 wavelet descriptor 和三个控制；
5. RG-OT 只在 RG-MCR 无法识别、但 node relation 有 headroom 时实现；
6. 输出诊断矩阵，最多选两个方向进入新的正式执行任务书；
7. 未选路线归档，不继续调参；
8. 任何路线进入真实异常评价前，先完成 sample/ref ID、几何、复现和泄漏测试；
9. BTAD/MVTec/VisA 在最终唯一 winner 冻结前保持不可见；
10. 不修改 A1 冻结证据和当前论文主张。

建议新增目录：

```text
configs/innovation_v4_diagnostics/
src/industrial_ad/innovation_v4_diagnostics/
scripts/innovation_v4_diagnostics/
tests/innovation_v4_diagnostics/
experiments/dynamic_fusion/innovation_v4_diagnostics/
```

诊断完成后，另写一份只包含最终 1–2 条路线的严格候选/门槛任务书。不要把本研究版图直接当成无限实验授权。

---

## 13. 当前建议

- **主推荐：RG-MCR**，因为它最明显地超出“分数融合”，能形成独立、可解释的算法问题；
- **低成本推荐：SF-NM**，因为它引入 DINO/CLIP 之外的频率信息，可快速判断微缺陷是否还有未利用 headroom；
- **高风险备选：RG-OT**，只有结构 smoke 支持关系信息时才投入；
- CASF 保留但降为同级候选，不再把所有希望押在合成异常融合；
- DC-SZoom 只有在缺陷尺度诊断支持时才投入高分辨率重导；
- 组件图、源域元学习、扩散修复和 conformal 分别作为中长期路线，不与当前实验混跑。

这使项目从“两个视觉分支怎样融合”扩展为四种互补的正常性证据：

```text
外观是否见过          → A1 global memory
在当前上下文是否合理  → RG-MCR counterfactual repair
高频表面是否正常      → SF-NM spectral memory
部件关系是否正常      → RG-OT relational transport
```

只有其中某一条在严格机制控制和跨数据集验证中成立，才能升级为论文创新。

