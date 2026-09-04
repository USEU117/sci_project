# CL-RPF cache probe（doc 26 §4.1 / §6.3）— 结果与判定（k1/k2/k4 全 shot）

date: 2026-09-04
authority: docs/.../26_CURRENT_CORRECTION_STATUS_AND_NEXT_BREAKTHROUGHS_CN_20260904.md §4.1 / §6.3
方法：MPDD dev s0 × k{1,2,4} × 6 类；每层（DINO L6/9/11、CLIP L6/12/18/24，统一 32 网格）
取与正常 support 集的最近邻 L2 残差 r_l，按 normal-only 校准（K=1 → support 图内
leave-one-patch-out + 空间邻域排除半径 1；K=2/4 → leave-one-image-out，均已登记）
标准化 z_l=(r_l−μ_l)/σ_l。候选逐 patch score map → dists2map → pooled Pixel-AP @56。
产物：`04_clrpf_probe/CLRPF_PROBE_k{1,2,4}.json`；代码：`scripts/.../run_r3_ef_clrpf_probe.py`。

## 1. 宏平均 Pixel-AP（6 类，分 shot）

| 配置 | k1 | k2 | k4 |
|---|---|---|---|
| a1 | 0.309212 | 0.343699 | 0.388328 |
| static2（归档） | 0.308382 | 0.348602 | 0.384439 |
| 静态对照 mean_std（7 层标准化残差均值） | **0.309856** | **0.349498** | **0.375414** |
| 静态对照 best_single_layer_map | 0.293225 | 0.315610 | 0.361983 |
| 最佳轨迹变体（k1 mean_aslope 0.2325 / k2 late_persist / k4 late_persist） | 0.232529 | 0.284107 | 0.292604 |
| 顺序对照（reversed/shuffled/final_repeat 中最好） | 0.057978 | 0.103944 | 0.105806 |

## 2. 机制门（macro，6 类）

| 门 | k1 | k2 | k4 |
|---|---|---|---|
| G1：最佳轨迹 − 最强静态多层控制 ≥ +0.003 | **FAIL**（−0.077） | **FAIL**（−0.065） | **FAIL**（−0.092） |
| G2：正确顺序 − 打乱/反序 ≥ +0.003 | PASS（+0.175） | PASS（+0.180） | PASS（+0.187） |

## 3. 判定

三 shot 一致：轨迹特征（slope/|slope|/late_persist/second_diff/d_minus_c）相对
**最强的静态多层控制（7 层 normal-only 标准化残差等权均值 mean_std ≈ A1/static2）**
无独立增益（G1 三 shot 全 FAIL）。G2 通过仅说明"顺序会影响 slope 算子自身的输出"，
不足以使轨迹超过静态融合。按 doc26 §4.1 "如果顺序无影响，则它只是多层集成"与
"未过即停止"口径：

**CL-RPF 轨迹路线（本预注册形态，MPDD s0 × k{1,2,4}）正式归档为负。**
顺带三 shot 复现：静态 7 层标准化残差均值 ≈ A1/static2（k1 0.3099 vs A1 0.3092，
k2 0.3495 vs static2 0.3486）→ "多层静态信息已被 A1/static2 吸收"的既有结论在
normal-only 标准化口径下依然成立。

## 4. 局限
- 特征为标量残差距离的逐层标准化；未做跨层 raw 向量对齐（doc26 已提示不宜直接比夹角）。
- "最佳轨迹变体"在预注册集合内取 max；非事后搜索。
- clip 均经 32 网格重采样（同 A1）；37 原生网格轨迹未测。
