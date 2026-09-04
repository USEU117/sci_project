# 纠错重跑后的实验状态、剩余验证与下一轮突破口

日期：2026-09-04，18:33 起核查工作区  
性质：独立复核与研究建议；未修改其他任务正在修改的实验代码，未启动新训练、下载模型或运行外部验证。本文中的新门槛为建议，执行前需冻结协议，不是文献结论或已获实验结果。

## 1. 当前最准确的结论

**上次提出的确定性 bugfix、主模型纠错重跑、oracle 空信息审计均已完成。纠错后不再出现旧版的大幅退化，但可学习交互几乎消失，结果与两对层静态拼接相同。A1 仍是主方法，没有可升级的新方法。**

本次独立检查了：

- `CORRECTION_NOTES.md` 与 `ORACLE_NULL_AUDIT.md`；
- `main_correction_all.json` 与静态 `REFERENCES.json`；
- 六个 correction checkpoint 与六份完整训练日志；
- 当前工作树代码改动；
- CPU 回归测试，实跑结果 **13 passed**。

没有独立重算全部图像预测。以下 AP 为现有结果 JSON 核对；参数范数、日志步数与测试结果是本次直接检查所得。

## 2. 已完成与未完成

| 项目 | 当前状态 | 结论边界 |
|---|---|---|
| 稀疏项 detach bug | 已修正，梯度回归测试通过 | 不能再沿用“稀疏惩罚从未生效”的旧结论 |
| query 固定上半幅采样 | 已改全图随机采样，六 fold 覆盖 1024/1024 | 仅说明采样覆盖正确，不等于训练充分 |
| no-support/shuffled 运行错误 | smoke tests 通过 | 不代表控制有效性或性能矩阵已完成 |
| 单分支可训练参数匹配 | 计数测试通过 | 还应核查活跃计算路径/真实单分支语义 |
| correction 主模型 | 六 fold，每 fold 600 条日志，18 个 cat×shot 结果 | 仅 MPDD seed0；不是三 seed 或跨数据集确认 |
| checkpoint 保存 | 六份均存在，可重新评估 | resume 仍需更完整的配置/代码 hash 核验 |
| oracle 空信息控制 | 三 shot 均完成 | 旧 headroom 不再能作训练入场证据 |
| 训练型机制控制完整矩阵 | 未完成 | 主模型无增益时不必立即花预算全跑 |
| 原始特征 parity amendment | 尚未闭环 | 指标接近不能替代逐像素 map/特征一致性 |
| Stage2 backbone 内部 bridge | 未启动 | 维持不进入 |
| PRS / CL-RPF / PSMF | 本次目录检索未找到正式实验结果 | 是未验证方向，不是失败方向 |

### 2.1 性能数值

| 六类宏平均 Pixel-AP，MPDD seed0 | 1-shot | 2-shot | 4-shot |
|---|---:|---:|---:|
| A1 | 0.309212 | 0.343699 | 0.388328 |
| static2，两对层静态 concat | 0.308382 | 0.348602 | 0.384439 |
| 旧版 SCAIF v4 | 0.261294 | 0.323109 | 0.331993 |
| correction v5 | 0.308382 | 0.348602 | 0.384439 |
| correction − A1 | -0.000830 | +0.004903 | -0.003889 |

三个 shot 平均 Δ 约 `+0.000061`，远低于此前 `+0.006` 的晋级门槛。本次程序化比较确认：**18/18 个 cat×shot，correction AP 与 static2 的绝对差为 0.0**。这是 AP 相等，不单凭它宣称每个 feature/map 数值完全相同。

来源：

- `experiments/dynamic_fusion/innovation_v12_early_fusion/03_scaif_small_gate/runs/main_correction_all.json`
- `experiments/dynamic_fusion/innovation_v12_early_fusion/03_scaif_small_gate/REFERENCES.json`

### 2.2 新增发现：残差通路不只是“小门”，而是近乎零参数输出

六 fold 的最后训练 gate mean 为 `0.002511–0.002586`，不是严格等于零。更关键的是：所有 `wd/wc` 解码器权重范数只有约 `1.92e-14–6.43e-14`。

以 bracket_black checkpoint 为例：

- 交互 MLP 第一层权重范数约 `1e-8`；
- 第二层权重范数约 `2e-12`；
- 解码器约 `2e-14–5e-14`；
- 门头仍为正常非零参数量级。

因此，单看“gate 只有 0.0025，所以缩小稀疏权重就能恢复学习”不充分。当前交互 MLP 与解码器同时衰减，须看**任务梯度和正则化作用的相对量级**。正则压制、梯度过弱、初始化与归一化的联合作用是候选解释，尚未通过干预证明根因。

### 2.3 oracle 审计的结果

| mean ΔPixel-AP vs A1 | 1-shot | 2-shot | 4-shot |
|---|---:|---:|---:|
| 真实层专家的 GT 组件拼接 | +0.399753 | +0.375420 | +0.390378 |
| A1 自身复制 | 0 | 0 | 0 |
| A1 单调缩放的空信息专家 | +0.690780 | +0.656249 | +0.611638 |

单调缩放没有增加排序信息，却得到比真实专家更大的 headroom，确认旧 oracle 不能区分真实互补与尺度/GT 特权。应废止这个入场指标。

**谨慎边界：这证明旧证据不能识别互补性，不等于已经定量证明真实 oracle 增益有多少比例由尺度造成，也不等于双编码器不存在互补。**

来源：`innovation_v12_early_fusion/02_stage0_probe/ORACLE_NULL_AUDIT.md`。

## 3. 现在还值得验证什么

### A. 仅一次短的优化健康检查，不再直接跑六 fold

建议在固定的源类训练 episode 上检查初始、1、10、50 步，使用两个预先指定的源类，避免按效果挑类别。不得访问冻结外部验证标签。

必须输出：

1. 各层 `||g*Delta|| / (||F||+eps)`，同时报告归一化前后修正量；
2. 分割、clean-preserve、anomaly-preserve、gate-sparse 分项损失对投影/MLP/解码器/门头的梯度范数；
3. Adam 的 coupled weight decay 项 `lambda*theta` 与原始任务梯度量级；
4. 固定 episode 的分割损失和正常/异常 score 分布；
5. 关闭交互与开启交互的特征差、map 差、AP 差；
6. 输入两分支、support 距离特征和中间激活的尺度。

验收不是“只要参数不为零”，而是任务损失确实给交互通路提供有限、非零梯度，且在不靠稀疏项下降的前提下能降低固定 episode 的任务误差。若无法做到，不再开完整训练；若能做到，也仅证明优化可用，不证明泛化有效。

### B. 纠错版并非完全单变量比较，需补记录

当前 `run_r3_ef_scaif.py:119` 为模型变换后的 support bank 新增 `torch.no_grad()`。同一个可学习模块也在变换 support，停止这一路梯度是一个训练选择，不是自动成立的“无梯度需求”。它可合理采用，但应明确登记，而不是把 v4/v5 差异全部归因于 gate detach 修复。

此外，当前对 query 投影归一化后同时用于距离、交互 MLP 与门头；这也改变了后两者输入尺度。短诊断应区分“只统一距离尺度”与“统一所有交互输入尺度”，不启动层/宽度超参搜索。

尚未处理的原设计问题仍包括：

- 使用 `10*s` 作为 BCE logit，且 `s>=0`，正常样本无法取得负 logit；
- 原始 private stream 没有作为独立路径保留；
- gate cap 不能单独保证实际残差范数受限；
- 源类按类别顺序训练，而非混合 episodic；
- branch-drop 仍是配置声明，未见训练实现。

这些应在独立设计修订中解决，不应无边界地纳入“再修一个 bug”。

### C. 控制的科学含义还要检查，不能只看 smoke PASS

- shifted/rolled spatial control 是“错位控制”，不是随机无信息控制；小位移后仍可能保留大量局部相关性。若恢复机制实验，应增加真正打乱对应、且保持边际分布的控制。
- 当前 dino_only/clip_only 主要限制残差更新方向，而最终 static feature 仍包含两分支；应改名为单方向修正控制。若声称单编码器方法，query、support、gate 与最终 scoring 全部只能用一个编码器。
- 源目标类别分离不等于外部数据集泛化；反复在 MPDD 六类上设计方法后，MPDD 仍是 development。
- seed1/2 仅改变正常参考选择，不提供全新测试图像；不能描述为独立数据集确认。

### D. 论文主线不必等待所有探索

A1 已有主结果、单分支控制、bootstrap 与效率材料，应继续整理。当前更有价值的补齐是：统一 MPDD/VisA CLIP-only 的六指标格式、整理已存在的开发集权重敏感性、明确所有新候选只作 development 负结果。不要为补论文表格再次在 BTAD/MVTec 上选配置。

## 4. 下一轮突破口：排序与差异

不再建议“同样的最终层特征 + 又一个门控头”。选择标准改为：是否引入了此前没有被测试的观测信息，或者明确改变计算/精度目标。

| 优先级 | 路线 | 新信息/新目标 | 本项目历史状态 | 最大风险 |
|---|---|---|---|---|
| 1 | CL-RPF：跨层正常偏离轨迹 | 同一位置跨深度的变化，而不是层分数平均 | 已有设计、多层 cache 现已可用；未找到正式结果 | 等价于加层或分数尺度差 |
| 2 | PRS：双编码器扰动响应谱 | 同一图像面对干预强度的响应轨迹 | 22号文档已提，未找到正式结果 | 只学到扰动强度/边缘，不是异常 |
| 3 | 跨分支共同坐标的细节恢复 | 在 KNN/融合前恢复空间细节，而非事后平滑 | 本次新增具体路线 | 普通 upsampling 已有，容易抹除微缺陷 |
| 4 | 匹配目标驱动的轻量学习 | 将正常匹配的判别目标与交互同步优化 | SCAIF 尚未正确隔离该问题 | 文献拥挤、训练协议改变 |
| 可选 | 双分支关系蒸馏至单分支 | 不追求再涨 AP，而是保持 AP 降低推理成本 | 本次新增研究目标提案 | 改变论文目标；单纯蒸馏不新 |

### 4.1 CL-RPF：优先利用已经付出代价导出的缓存

核心问题：单层分数不够区分正常纹理与缺陷，但它们从中层到深层的正常偏离是否有不同持续模式？

建议先只使用每层经正常支持集尺度标准化的残差距离。不要直接比较不同层的 raw residual 向量夹角，除非先说明跨层坐标对齐；不同层语义坐标不天然可比。

构造每 patch 的：深度斜率、二阶差分、后半层持续高偏离比例、D/C 持续模式差异。校准必须 normal-only；1-shot 无法 leave-one-image-out 时，要单独说明空间分块排除邻域的限制，不能把相邻 patch 当作独立正常图。

低成本控制：

- final-layer A1；各层等权固定 mean/max；static2；
- 顺序打乱/反序层轨迹；
- 只保留每层分数、去掉轨迹项；
- 同一终层重复多次的伪多层轨迹；
- DINO-only / CLIP-only。

建议机制门：相对最强静态多层控制 `>=+0.003 Pixel-AP`；正确顺序对打乱顺序 `>=+0.003`。如果顺序无影响，则它只是多层集成，不作为轨迹创新。最终晋级仍按统一 `>=+0.006 vs A1` 门。

它不是已经被 static2≈A1 否定的路线：静态拼接不利用层变化次序。但它也不因低相关就自动成立。

### 4.2 PRS：让融合使用响应曲线，而不是单次响应值

对同一图像施加少量固定强度的轻微干预，比较两个编码器相对正常支持集的响应斜率/曲率。正常参考给出响应包络，异常候选是偏离包络的位置。

文献启发：[PCU, UAI 2026](https://proceedings.mlr.press/v337/allaoui26a.html) 使用受控扰动阶梯学习敏感度、位移和局部稳定性。其验证主要是表格数据且训练了 encoder；**不能把论文结果直接视为冻结视觉编码器上的成功证据**。本项目需要 independently validate patch-level dual-encoder extension。

与旧路线区别：DEVA/MESP 聚合增强结果，NTOF 删除估计的干扰方向，PRS 则使用响应随强度变化的形状。先选 2 个干预族、3 个强度，做 normal held-out + cutpaste/erase/scratch 合成异常门，不一开始导出所有组合。

必做对照：增强平均、总变化幅值、单编码器响应、打乱强度顺序、错配 D/C 干预。正常 FP 应受约束，且响应谱在 held-out 合成族上的定位 AP 至少高于最强简单对照 0.02；真实 development 阶段要求独立增益 0.003。未过即停止。

### 4.3 新具体路线：融合前的跨分支空间细节恢复

现在 CLIP 37×37 先缩到 DINO 32×32，随后 KNN、平滑与上采样。可以研究是否在进入匹配之前就丢失了影响微缺陷定位的信息。

建议先建立**共同图像坐标**，用冻结的预训练 feature upsampler 分别恢复两分支至较小共同网格（例如 56×56），再构建同坐标正常 memory；query 和 support 必须同处理。不要先上 448×448 全维 KNN。

一手参考：

- [FeatUp, ICLR 2024 官方代码](https://github.com/mhamilton723/FeatUp)：支持 DINO/CLIP 等特征上采样；
- [AnyUp, ICLR 2026](https://arxiv.org/abs/2510.12764)：无需按目标 encoder 重训的通用 feature upsampling；
- [RaysUp, 2026 作者预印本](https://arxiv.org/abs/2606.22749)：轻量、几何引导的通用特征恢复，可作为算力受限候选。其作者报告的效率不能直接套用到本机。

创新不能只是“使用 AnyUp”：应证明双分支在共同图像坐标恢复时能相互约束虚假边缘，同时保留真正微缺陷，而单分支/普通上采样做不到。

必做控制：

1. 当前 A1；
2. 双线性上采样 feature 后 KNN；
3. 原 A1 score 上采样/引导滤波；
4. 单分支 feature upsampling；
5. 两分支独立 upsampling 再固定拼接；
6. concat feature 使用同一 upsampler；
7. 错误 RGB 引导图、错位 cross-branch correspondence。

警告：相同线性算子下，“分别上采样再拼接”与“拼接再上采样”可能代数等价，不能命名成新方法。必须有实际不同的分支条件化非线性/匹配规则，并通过第5/6控制体现独立价值。

机制门：相对最强 generic upsampler 独立增益 `>=0.003`；small-defect 子组 `>=0.01` 且正常纹理 FP 不增加超过 5%；错配引导/分支应明显掉点。旧56网格主指标与完整分辨率辅指标同时报告，不能改变 GT 下采样后宣称胜过原结果。

这与 FAGR 的后处理平滑不同，与 CAPM 的 query-reference 位姿限制也不同；与 PSMF 的 patch-grid 相位实验邻近，若实现效果等价于多移位平均，只算增强基线。

### 4.4 匹配目标驱动的学习：条件性保留，不是当前优先主线

[DFM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.html) 将最近邻匹配纳入可微训练，针对 feature adaptation 与 matching 不协调的问题。这提示当前 SCAIF 的困难可能首先是目标/优化，而不是模块不够大。

如果短优化诊断显示任务梯度能修正、用户接受源类训练，可单独预注册：中心化 distance logits 或 ranking objective、明确 support bank 是否 stop-grad、严格独立 A1 通道、有限范数 interaction。先复现匹配驱动的小基线，再做双分支扩展。

风险：低秩 Mahalanobis metric `M=L^T L` 与线性投影 feature 再做欧氏距离本质等价，不能包装成全新融合。只有支持集条件化、异常证据保留以及强单分支/固定匹配控制通过，才有方法增量。

### 4.5 可选新目标：双分支关系蒸馏，保精度降成本

本项目既有 A1 的效率记录中，CLIP 提取约 0.3049s，DINO 约 0.0626s，CLIP 是主要成本。与其持续追求极小的 AP 增量，可以考虑用 A1 作为离线 teacher，让单视觉分支学习其**相对正常支持集的距离排序**，减少推理时第二分支的需求。

不是 NCPRA：NCPRA 用 D↔C 预测失败作为异常分数；此路线把 teacher 的正常匹配关系压缩到 student，推理直接使用 student memory distance，优化目标是成本—精度，而不是跨分支预测误差。

但蒸馏本身成熟：[EfficientAD, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) 已用高效 student-teacher 做异常检测。RADIO 等多教师表征也使泛泛“融合蒸馏”不新。可研究的差异在 few-shot support-relative rank preservation 与按缺陷尺度保真，而非简单 teacher feature MSE。

建议目标（须用户接受改变论文目标）：mean Pixel-AP 相对 A1 不低于 -0.003、worst-category 不低于 -0.01、同机同分辨率端到端延迟至少下降40%，并超过普通 MSE distillation 和未蒸馏单分支。未满足精度—成本两端，不推广。

## 5. 文献排重后的不推荐方向

- 不再优先普通 PCA/协方差重标度：SubspaceAD、既有 LNDC/LLSE/NTOF/CCA 已覆盖大量邻近尝试。
- [GPFlow, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_GPFlow_Gaussian_Prototype_Probability_Flow_for_Unsupervised_Multi-Modal_Anomaly_Detection_CVPR_2026_paper.html) 已用高斯 prototype 与 covariance-aware 收缩联合 intra/cross-modal reconstruction；“把单点预测改高斯”不能直接视为独立创新。
- [SSVP, arXiv 2026](https://arxiv.org/abs/2601.09147) 已将 DINOv3 多尺度结构融入 CLIP，并结合动态 prompt；目前按作者预印本引用，不能未经核实标成顶会论文。
- 不把 DINO 亲和图传播重新包装成新主线：既有 FAGR 失败，而且近期已有 DINO-guided CLIP score refinement 文献；简单平滑/传播必须先超过均匀控制。
- 不复活旧 fixed expert pool 的 oracle router，不继续依据已废止的 +0.39 做可学习性推断。

## 6. 推荐执行顺序

1. 收口纠错证据：保留 v4、v5、null audit 的独立记录，修订入口与主账链接一致；补 parity amendment。
2. 做一次短优化健康诊断，只定位交互归零的原因，不重跑完整六 fold。
3. 低成本优先：CL-RPF cache probe；有独立顺序信息再推进。
4. 探索新观测：PRS normal/synthetic 小门，先证明相对增强均值的独立信息。
5. 若微缺陷/边缘是主要失败区域，再进入共同坐标 feature lifting/PSMF。
6. 只有某条路线相对 A1 和最强简单控制均有独立增益，才做三 seed、paired image/category bootstrap、冻结外部确认；各类别至少5/6正、均值增益至少0.006、最差不低于-0.01沿用原晋级门。区间应按图像/类别而非独立像素重采样。

若用户更看重尽快完成论文，稳定 A1 主结果的整理可继续，不必等新路线成功；若用户允许改变贡献目标，关系蒸馏的精度—成本路线可另立研究协议。以上新路线均是待证实假说，未承诺会涨点或具备可发表新颖性。
