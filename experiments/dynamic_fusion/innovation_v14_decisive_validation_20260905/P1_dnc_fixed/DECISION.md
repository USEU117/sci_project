# P1 DECISION.md — DNC 机制门（合成，support-only）结果与判定

日期：2026-09-05　计划：doc28 §5.3　协议：`P1_dnc_fixed/PROTOCOL.json`（冻结）

## 1. 数据与结构（合法）
- fit/select 仅用 manifest support（`/train/good/`，K-shot）；未读取任何 `/test/`（DATA_ROLE_AUDIT 18/18 ok；导出器每次调用 `assert_fit_ids_are_support`）。
- 合成干预在 support 图像（1024）上渲染后经冻结 DINO/CLIP 提取器重新编码（非 feature-token 编辑）。
- 结构：对每个 (cat, shot∈{2,4})，留一 support 图 h × 留一族 f：在 (K−1) 图 × (2 非留族 × 3 种子) + 15 nuisance 上拟合 q/选择，在 h 的留族 episode 上测 reduced-fused KNN AP。宏平均：先平均 h，再平均 cat（与 shot 汇总）。
- 候选/控制：full、DNC-I(256/branch)、DNC-C-fixed(λ=0.3)、low_nui、highvar、dino_only(512)、clip_only(512)、10 个固定 random masks（比较用其均值）。

## 2. 结果（全量 k2+k4）
宏 AP（pool h×cat）：full≈0.80；DNC-I≈0.784；DNC-C≈0.784（与 DNC-I 逐位相同）；random-mean≈0.786；low_nui≈0.785；highvar≈0.789。

逐类 (shots 合并，DNC-I − random-mean)：
- bracket_black −0.003；bracket_brown −0.013；bracket_white **−0.062**；connector +0.012；metal_plate +0.009；tubes +0.001。

逐留族 (DNC-I − max(random-mean, low_nui))：
- cutpaste：k2 +0.035（通过阈值 +0.02）、k4 +0.019（未过）；
- local_erasure：k2 −0.036、k4 −0.055（不通过）；
- thin_scratch：NaN（划痕在 32 网格下采样后 mask 全空；低于 DINO patch 分辨率，整族不可评）。

## 3. 门判定
| 门 | 阈值（冻结） | 实测 | 判定 |
|---|---|---|---|
| G1 | ≥2/3 留出族上 DNC-I − max(random-mean, low_nui) ≥ +0.02 | 1/3（仅 cutpaste k2） | FAIL |
| G2 | nuisance FP(相对 full) ≤ +10% | full=1.000，DNC-I=1.000（双双饱和，p99 阈值过紧，无区分力） | 不通过（失真，不可解释为通过） |
| G3 | 集合 Jaccard<0.95 且跨分支冗余下降≥10% | Jaccard=0.9981；drop=0.002（0.2%） | FAIL |
| G4 | DNC-C − DNC-I ≥+0.003 且 shuffle 后下降≥0.003 | gain=0.0000；shuf_gain=0.0000 | FAIL |

**决策：`P1_FAIL_ARCHIVE` → claim：none。** 按 doc28 §5.3，G1/G2 失败 → 归档，不做真实 MPDD 运行；P1-C（真实冻结诊断）不执行。

## 4. 机制解读（诚实归因）
1. DNC-C-fixed（λ=0.3，分支内 q−λ·max|corr| 贪心）在 k2/k4 均几乎不改变集合（Jaccard 0.9981，冗余仅降 0.2%），AP 增益恒为 0。原因：q 分布极陡、高 q 通道未被冗余惩罚覆盖，且 episode 数少（k2=6、k4=18）使跨分支相关估计含噪声。修复后集合**有**微小改变，但无可测收益 → 不能支撑"融合创新"。
2. DNC-I 相对随机/low-nuisance 通道选择在合成代理上无系统增益（只有 cutpaste/k2 +0.035 单点）。多数类别 DNC-I ≤ random-mean/low_nui。通道筛选在该 reduced-fused KNN 下无独立机制价值。
3. thin_scratch 族在 DINO patch 分辨率下不可见 → 该合成族在 32 网格天然不可评（记录为冻结协议的测量限制，不计为证据）。
4. G2 测量失真（两种选择均 100% FP）：p99 正常参考基于留出图 bank 的 LOO d2，对跨图光度变化过紧；该指标在本设计下无信息量。

产物：`SYNTH_RESULTS.json`（rows/diags/g2 全量），`GATES.json`（判定），`PROTOCOL.json`（冻结配置）。
