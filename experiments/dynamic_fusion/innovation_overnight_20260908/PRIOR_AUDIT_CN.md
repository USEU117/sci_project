# 2026-09-08 夜间探索：既有路线只读审计与优先级

审计范围：`docs/paper_writing_preparation_20260830/29–37`、`innovation_t1..t6`、`innovation_v14`、`innovation_a2`，以及对应的脚本、冻结缓存、A1 评价代码和已归档路线。审计时间为 2026-09-08 00:10 后（北京时间）。本文件是唯一新写入的审计产物；没有回退或改写其他 agent 的文件。

## 先给结论

当前可复用的冻结主线仍只有 A1：DINOv2 ViT-B/14 与 AnomalyCLIP 最后层 patch 特征，分别逐行 L2 归一化，`0.5 / 0.5` 拼接，整体 L2 后用 `k = 1` 全局正常 memory。现有机制路线没有通过真实门；合成代理中的正增益不能直接外推到 MPDD test。

今晚最值得做的小规模方向是四个，按优先级为：

1. **输入空间裁块重编码 + 原像素 mask 评价**：它改变的是 encoder 看到的图像视图，和旧 AnyUp 的 token 恢复不同；先做 detail 可行性，不宣称新算法。
2. **support 图级 coverage / active selection**：以整张 normal support 图为选择单位，在固定 memory 预算下选 1 或 2 张；这和 Track-4 的 patch 行级 coreset 是不同问题。
3. **不删 memory 行的 FP16 / INT8 存储与检索审计**：验证保留全部 normal 单元时的存储、误差和速度；这属于可复现的效率路线，不改变 A1 排序目标。
4. **原像素分辨率与缺陷族可检性审计**：native mask 探针已显示 thin scratch 有信号但不同缺陷族取舍相反，适合作为评价修复和 crop 路线的共同 control，不能当作模型增益。

结构先验或多视角 normal manifold 仍是 D6 暗示的远期缺口，但现有 MPDD support 没有稳定的跨图部件对应，今晚不宜用复杂新机制代替上述可证伪的小探针。

## 已有路线的结论和封闭边界

| 路线 | 已读证据 | 结论 | 今晚处理 |
|---|---|---|---|
| v14 DNC-I / DNC-C / semirelaxed OT | `experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/FINAL_DECISION_CN.md`、`DATA_ROLE_AUDIT.md` | 支持域合成 mechanism gate 未通过；P1 只有 `1 / 3` 合成族通过，P2-A 的 capacity premium 为负，k4 还因预计时间过长中止。没有真实 MPDD gate。 | 不重跑 OT、容量或 q/correlation 变体。 |
| A2 learnable matching | `innovation_a2_learnable_matching_20260905/HEALTH_PROBE_DECISION.md` | 对角通道权重健康探针：k2 macro `Δ AP = +0.00403` 但仅 `3 / 12` folds 达到预设增益，k4 `Δ AP = −0.00035`、仅 `2 / 12`；H1 失败，权重移动小且 loss 只降约 `2%`。 | 归档该参数化，不再调 λ、上界或 loss。 |
| T1 context defect | `innovation_t1_context_defect_20260905/TRACK1_DECISION.md` | 16-grid 的 C1 邻域描述子 cutpaste 仅 `+0.0137 / +0.0140`，低于 `+0.05` gate；C2 方向 concat 反而下降。 | 不重做 16-grid 邻域拼接。 |
| T2 multilayer | `innovation_t2_multilayer_20260905/TRACK2_DECISION.md` | DINO block 5 的 cutpaste 仅 `+0.0188`，低于 `+0.05`；CLIP mid 会伤害 cutpaste，虽有少量 nuisance 变化。 | 不继续堆叠中间层。 |
| T3 efficiency | `innovation_t3_efficiency_20260905/TRACK3_DECISION.md`、`REAL_GATE_DECISION.md` | 合成 T1 50% chessboard probe 通过，但真实 k2/k4 macro `Δ AP = −0.0070 / −0.0074`，connector 最差 `−0.0328 / −0.0527`，灾难类 gate 失败。 | 不再做几何均匀删 patch。 |
| T4 preserve coreset | `innovation_t3_efficiency_20260905/TRACK4_DECISION.md` | patch 级 C2 farthest-point 比 chessboard 明显好：k2/k4 macro `Δ AP = −0.0031 / −0.0018`，最差 `−0.0102 / −0.0046`；k2 的 AUPRO `−0.0060` 比严格 `−0.005` 门差约 `0.001`。高引用 hub C1 更差。 | 结果作为 patch coverage 证据；不把它当作图级 selection 已验证。 |
| T5 relation 32 | `innovation_t5_relation32_20260905/TRACK5_DECISION.md`、`D5_REAL_DECISION.md` | 32-grid 合成 C1/C2 cutpaste 大增（约 `+0.15`），但真实 k2/k4 C1 macro `Δ AP = −0.0142 / −0.0335`，C2 更差；所有主要真实 gate 失败。 | 不把合成 relation gain 当真实路线依据。 |
| T6 defect diagnostic | `innovation_t6_defect_diag_20260905/D6_DECISION.md` | `parts_mismatch` 宏弱主要由 bracket_brown 拉低；connector 的该类 Pixel-AP 为 `0.467 / 0.531`。bracket_brown 上下文的 image-AUROC 约 `0.55 / 0.61`，说明核心瓶颈是图像级可检性，不是简单定位差。 | 作为边界诊断，暂不再加 relation 机制。 |
| v2 / v10 / v12 旧 portfolio | `innovation_v2/FINAL_DECISION.md`、`innovation_v10_portfolio/PORTFOLIO_LEDGER.md`、`innovation_v12_new_observables/PORTFOLIO_LEDGER.md` | LNDC、DSAM/CAPM 局部对齐、CEQA、DEVA、NCPRA、FAGR、SPRG、LLSE、CSS、PRS、PSMF、AnyUp 恢复等均已归档或未过 gate。 | 不更换名字重做相同局部对齐、平滑、密度、响应谱或 token 恢复。 |

这些数字来自不同实验目的时只能在各自协议内解释。特别是合成支持 proxy 和真实 MPDD test 的绝对 AP 不可直接相减；真实结果已反复被用于诊断时，新的 test 指标只能作为冻结后 exploratory evidence，不能再用来选方向。

## t3 k4 基线矛盾：不是已证实的 calibration / resize / stride 差异

`experiments/dynamic_fusion/innovation_t3_efficiency_20260905/REAL_GATE_DECISION.md` 的结果表把 k4 `mean full AP` 写成了 `0.3737*`，但同目录 `REAL_GATE_s0_k4.json` 的逐类平均和字段都是 `0.388328`。同一个 `0.388328` 也出现在：

- `innovation_t5_relation32_20260905/REAL_D5_s0_k4.json` 的 `mean_A1_ap`；
- `innovation_t6_defect_diag_20260905/D6_REPORT_s0_k4.json` 与 `D6_DECISION.md` 的 parity；
- `experiments/dynamic_fusion/v3_direction_a/p0_rebuild_20260826/mpdd_s0_k4.json` 的冻结 A1 重建；
- 多个 A1 weight scan / stage 0 control 报告。

主脚本 `scripts/innovation_t3_efficiency_20260905/run_real_gate_coreset.py` 直接导入 `scripts/evaluate_a1_feature_fusion.py` 的 `resize_patches`、`compute_metrics` 和 `STRIDE`，复用 `pca = 0`、`whiten = 0`、`w = 0.5`、32-grid、`dists2map` 到 448、Pixel-AP stride 8。D5 的 C0 也逐项使用同一拼接与评价路径；D6 读取 compact map 后重放 `dists2map`，逐类 AP 与 D5 的最大差为 k2 `4.3e-05`、k4 `1.8e-05`。因此现有证据支持 `0.3737` 是该 Markdown 表的旧值或误植，不能据此推断存在 resize、stride 或 calibration 变化；来源在该表的脚注中没有可复核实现。

今晚每个新方向都应在同一次运行内产生 control 与 candidate，固定以下字段：feature cache manifest、shot、seed、类别、grid、map size、stride、归一化、memory 行数和 evaluator 版本。禁止把 `REAL_GATE_DECISION.md` 的 `0.3737` 与其他表的 `0.388328` 拼在同一条 delta 上；需要基线时以同脚本重新生成的 control 为准，并在新的 overnight 输出目录保存 JSON。

## 图像 tiling / crop 是否重复旧工作

只读搜索当前 A1 路径没有发现图像级 tiling、sliding crop 或 crop-view memory：

- `methods/anomalydino/src/backbones.py` 的 DINO `prepare_image` 是把较短边 resize 到 448 后按 patch 对齐；
- `methods/AnomalyCLIP-main/AnomalyCLIP_lib/model_load.py` 的 `_transform(n_px)` 是固定方形 BICUBIC resize，CenterCrop 代码被注释；
- A1 的 `scripts/evaluate_a1_feature_fusion.py` 只在已导出的 patch feature 上把 CLIP 网格双线性 resize 到 DINO 网格；
- `scripts/innovation_v12_new_observables/run_r3_ef_recovery_probe.py` 的 AnyUp 是对已有 token 做 feature upsampling / recovery，不重新读取原图；
- `methods/winclip/WinClip-master/WinCLIP/CLIPAD/transformer.py` 有 window mask 代码，但没有接入当前 MPDD A1 双 encoder 路径，不能算旧 A1 实验。

所以“原图裁块后重新过 DINO / CLIP，再映射回原像素”与 AnyUp、T4 patch coreset、DSAM/CAPM 局部窗口限制都不同。它仍是已有文献和工程中常见的 crop / multiscale 思路，论文中应表述为 detail-feasibility 或 resolution ablation，不能单凭实现位置宣称算法新颖。tile size、stride、边界 padding、视图聚合和 memory budget 必须在看指标前冻结；query 和 support 使用同一视图规则；掩码评价放回原 1024 像素坐标。

当前 overnight `native_resolution/REPORT_CN.md` 已给出一个重要 control：六类 k2、留一张 normal support 作 query、另一张作 memory 时，16→32 grid 的原像素 Pixel-AP 为：

| 合成族 | 16-grid | 32-grid | 32 − 16 |
|---|---:|---:|---:|
| cutpaste | `0.757421` | `0.717909` | `−0.039511` |
| local_erasure | `0.904036` | `0.951238` | `+0.047203` |
| thin_scratch | `0.090067` | `0.213655` | `+0.123587` |

这证明旧的 `> 50%` 网格覆盖为零只能说明 mask-to-grid 评价不可见，不能推出 frozen feature 没有薄划痕信号；同时也证明更高网格并非对所有缺陷都更好。父 agent 新测的同一趋势（native 16→32，thin_scratch `0.0901→0.2137`、erasure `0.9040→0.9512`，但 cutpaste `0.7574→0.7179`）适合作为 crop 重编码的共同 control。

## support 图级 active selection：没有旧实现，且确实区别于 T4

当前代码和实验记录只找到三类相关行为：manifest 固定选择 k-shot normal refs；`rcec.py` 对 paired feature 做留图 memory / LOO；T4 C2 在拼好的 `[K, G, G, D]` 中按 patch 行做 farthest-point coreset。没有发现按整张 support image ID 计算 coverage、medoid、farthest image、随机图均值或主动选择的实现。

因此 k4 的“3 张 normal candidate + 1 张留出 normal image query，选择 1 或 2 张整图”是一个真正不同的 memory policy：被选整图的全部 patch 保留，选择单位是 image，而 T4 的选择单位是 patch。建议协议如下：

1. 对每个类别的 k4 refs 轮换一张作 held-out normal evaluation image，候选为其余 3 张；固定 manifest 顺序，不读任何 `/test/` 文件。
2. 选择器的 coverage 分数只能由这 3 张候选的 frozen feature 计算，例如候选 patch 到所选图 patch bank 的平均 / 95 分位最近距离；选 `k = 1`、`k = 2`，并预先固定 tie-break。held-out 图只能用来评价选择结果，不能参与选择分数。
3. 同时记录 all-4、固定首图 / 固定 manifest subset、预注册随机 seed 的 k=1/2 control。报告 memory 行数、coverage、normal held-out 距离和合成缺陷 AP；不按类别或缺陷族事后选不同图。
4. 若要在真实 MPDD 上看 exploratory AP，先冻结选择器和所有参数，再对 seed 0 的 k2/k4 统一运行；test mask 只进入最终指标函数，绝不进入 coverage、候选排序、阈值或是否继续实验的决定。

“用 held-out query 选出 coverage 最大的图”会把评估 query 变成选择信号，即便它是 normal，也会产生 transductive bias。若实验目的就是诊断 transductive 上限，必须单列为上限测量，不能叫 support-only selection，也不能与严格 memory policy 混报。

## 可小规模验证的方向

| 优先级 | 方向和真正变化的对象 | 最小验证与通过依据 | 现有入口 / 缓存 | 主要风险 |
|---|---|---|---|---|
| P 0 | 原图 crop / tile 重新编码；变化在输入视图和 encoder token，不改变 KNN 规则 | 先对 1–2 类、k2、预注册 cutpaste / erasure / thin scratch 做 support-only LOO；与全图 A1 和 native-resolution control 并列，记录原像素 AP、可见正像素数、显存和时间。只有在多个预定 episode 同向时才进入下一轮。 | `outputs/dynamic_fusion/v14_p1_support/`；`scripts/innovation_v14_decisive_validation_20260905/export_p1_support_variants.py` 中的 frozen extractor 构造可复用；A1 feature cache 做 no-crop control。 | crop + multiscale 已是常见工程做法；tile 数和 aggregation 很容易后验调参；不能把 native metric 的变化写成 model gain。 |
| P 0 | support 图级 coverage selection；变化在 memory 的 image unit，不是 patch coreset | 每类 k4 留一张 normal support 作 evaluation，候选三张；预注册 image-level medoid / farthest coverage，选 1/2 张，和 all-4、固定、随机 control 同批输出。首要看 held-out normal coverage 和 AP 变化，不能用 held-out 选择。 | `data/splits/mpdd/manifest.json`；`outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k{2,4}/...`；`scripts/innovation_t3_efficiency_20260905/run_real_gate_coreset.py::fused_concat` 可作 A1 拼接参考，但新脚本应写 overnight 子目录。 | 只有 3 个候选，coverage 方差大；整图选择可能把 patch coverage 问题重新包装；若 selector 读 query，会泄漏。 |
| P 0 | FP16 / INT8 全行 memory 的存储和近似检索；变化在数值表示，不删除 normal rows | k2、seed 0、六类 support-only LOO；记录距离 p 99、最近邻身份、top-10 overlap、AP、存储 / 常驻 bytes 和 wall time。按已冻结 precision protocol 判断，不能按类别调 scale / bit width。 | `precision/PROTOCOL_CN.md`；`outputs/dynamic_fusion/v14_p1_support/`；`scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py::load_cells` 的 cell 组织。 | FP16 roundtrip 只说明持久化，不说明常驻运行内存；INT8 近似内积可能改排序，需和 exact decode 分开报告。 |
| P 1 | native pixel / mask-aware scoring；变化在评价分辨率，不是 encoder 或 memory | 固定原 1024 mask、双线性 map，报告 16/32/64、薄划痕覆盖率和每族 AP；扩大预定 episode 后再判断是否值得为 crop 付出 GPU 成本。 | `native_resolution/REPORT_CN.md`、`native_resolution/RESULTS.json`、`resolution_audit/REPORT_CN.md`；A1 `scripts/evaluate_a1_feature_fusion.py` 仅作为旧 stride 8 control。 | 它不能支持真实泛化或论文主表替换；原像素 AP 与历史 stride 8 AP 不可直接相减。 |
| P 2（远期） | 结构先验 / 多视角 normal manifold；变化在数据或跨图结构，不是简单邻域 concat | 需要先得到稳定的跨图对应和新的 normal data split，再做 shuffled-correspondence control；当前 SPRG 跨图 node match 约 `36%` 且 CAPM/DSAM 已负，今晚只记录 gap。 | D6 结案、`innovation_v10_portfolio/sprg`、`capm` 记录。 | 当前 support 不支持可信的结构学习；贸然做 correspondence、OT 或局部窗口会重复已失败路线并有高泄漏风险。 |

前三个方向可以共享同一 A1 scoring harness，但必须把“输入重编码”“图级 memory 选择”“数值压缩”分成互斥实验标签；native-resolution 只作为评价 control。不要把四者组合成一个无法归因的候选。

## 可用缓存和入口

### 冻结 normal / test feature cache

- DINO：`outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k{2,4}/anomalydino_visual/{category}.npz`。
- CLIP：`outputs/dynamic_fusion/v3_direction_a/features_s0_k{2,4}/anomalyclip_text/{category}.npz`。
- 这些文件包含 `patch_features`、`ref_patch_features`、`sample_ids`、`grid_size` 和评价所需的 mask；memory ref 来自 manifest 的 normal train support。使用前检查 sample order 和 ref IDs。
- compact A1 map replay：`submission_repro_20260827/predictions_compact/maps/mpdd/s0_k{2,4}/{category}.npz`。它适合只读诊断，不适合重新选择 memory 或拟合参数。

### 支持域合成缓存

- `outputs/dynamic_fusion/v14_p1_support/v14_p1_support_{dino,clip}_s0_k{2,4}/{category}.npz`：v14 的 support-only synthetic clean / cutpaste / nuisance 变体；DINO 与 CLIP 分支网格不同，需按 A1 规则把 CLIP resize 到 DINO grid。
- `outputs/dynamic_fusion/t2_multilayer_support/k2/{category}.npz`：中间层路线缓存，作为历史证据，不建议重新扩展。

### 评价和探针入口

- A1 规范实现：`scripts/evaluate_a1_feature_fusion.py`，关注 `fuse_category`、`resize_patches`、`score_via_memory`、`compute_metrics`。
- T3 coreset 参考：`scripts/innovation_t3_efficiency_20260905/run_real_gate_coreset.py`；它会写入旧实验目录，若仅复用函数请复制到 overnight 脚本并换输出目录。
- T4 patch C2 参考：`scripts/innovation_t3_efficiency_20260905/run_t4_preserve_coreset.py`；只用于证明 patch 行级 farthest，不要冒充 image selector。
- D6 诊断：`scripts/innovation_t6_defect_diag_20260905/run_d6_defect_type_diag.py`；它会写旧 D6 JSON，读结果即可。
- 历史 support proxy：`scripts/innovation_t1_context_defect_20260905/probe_c1_context_variants.py`、`scripts/innovation_t5_relation32_20260905/probe_h1_relation32.py`、`scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py`；均是已关闭路线的参照，不要用来证明真实增益。
- native resolution：`scripts/innovation_overnight_20260908/probe_native_resolution.py`，结果已经在 `native_resolution/`；只读复核时不要覆盖已有 JSON。

如果今晚需要再导出 crop feature，必须新建 overnight 输出名和 run manifest；不要覆盖 `v3_direction_a`、`v14_p1_support` 或任何历史 paper table。CPU-only 可先做 selection / precision；GPU 只给 input crop re-encode 使用。

## 数据角色、泄漏与比较纪律

1. `data/splits/mpdd/manifest.json` 是唯一 support 身份来源。v14 的 `scripts/innovation_v14_decisive_validation_20260905/v14_common.py::assert_fit_ids_are_support` 已把 `/test/` fit ID 设为 hard fail；沿用这条规则。
2. memory、PCA、量化 scale、coverage selector、阈值和任何模型参数只能使用 `/train/good/` support。不能把 `/test/good/` 当 normal memory，也不能把真实 bad 图用于训练或候选选择。
3. 留一张 support image 的协议中，held-out normal 只用于该折评价；候选选择使用其余 support。若把 held-out image 的距离用于选图，必须显式标为 transductive 上限，不得混入 support-only 结论。
4. 合成 mask 只用于预注册 support proxy；合成 AP、native 1024 AP、旧 grid-level AP、真实 MPDD AP 各自单列，不跨协议计算 delta。
5. 真实 MPDD test 已被多轮 D5/D6/T3 反复诊断。先固定代码、seed、shot、类别和参数，再读取 mask 计算 exploratory metrics；不根据 test 指标挑类别、调 tile、改阈值、决定保留哪个结果。
6. 每个新方向都必须同时记录 control、candidate、feature cache、map/stride、memory 行数、运行时间和完整 per-category / per-episode 结果。没有完整 control 的单个高 AP episode 不能升级为路线结论。
7. 禁止引入外部图像、外部 checkpoint 或未经授权的新数据。不得覆盖 A1 冻结表、历史 JSON、其他 agent 的目录或本文件以外的 overnight 产物。

## 明确避免的重复失败

- 不再做 DNC / OT / semirelaxed capacity、A2 对角 channel weight、cross-branch low-rank 或 q/correlation 重参数化。
- 不再做 16-grid 或 32-grid 的 relation / 3×3 / directional concat 作为主机制；T5 已显示合成大增、真实反向。
- 不再做 DSAM / CAPM 的局部窗口或细粒度几何对齐，不再把 SPRG 的跨图 node matching 作为现成结构先验；已有真实负结果和低 match rate。
- 不再做 chessboard、固定网格或高引用 hub 的 patch 删除；若需要 memory 压缩，优先研究已区分的 image-level selection 或数值压缩，并保持全行 control。
- 不再重做 AnyUp / feature recovery、PSMF smoothing、PRS / NTOF response spectrum、LOF / density / local reconstruction、CSS self-similarity 等已归档方向。
- 不把 `thin_scratch` 在旧 grid coverage 为零解释成“feature 没信号”，也不把 native 32 的局部正结果解释成“高分辨率普遍更好”。

本审计的可执行顺序是：先用同一 A1 control 固定比较口径，再做 native-resolution / crop detail 小探针；并行做全行 memory 的 precision 测量；最后做严格 support-only 的图级 selection。任何路线若在预注册的 support proxy、泄漏检查或同脚本 control 对账上失败，应归档原因，不进行结果导向的补救调参。
