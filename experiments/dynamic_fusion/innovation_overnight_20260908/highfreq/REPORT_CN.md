# 原图低层高频 / 梯度描述子探针记录（2026-09-08）

## 旧路线审计与口径边界

旧 `innovation_v10_portfolio` 已经实现并实测了 2-level Haar 高频残差（gray、R-G、B-Y）以及 Sobel gradient control：`scripts/innovation_v10_portfolio/run_r0_str.py` + `src/industrial_ad/innovation_v10_portfolio/spectral.py`。其 k4、真实 A1-missed-region 结果为 mean Δ(STR−A1) = −0.027368、仅 1/6 类为正，已 `FAIL_ARCHIVED`。因此本次结果不能把“高频/梯度描述子”本身称作新颖算法。

本 probe 的差异只用于有限口径审计：1024 原图直接构造 64×64 灰度小维统计，k2 的 bracket_black / metal_plate，24 个与 tiling 相同的 support synthetic masks；全局 NN、memory-only median/MAD 尺度归一化，并列 raw-only 与 DINO full controls。

## 数据角色与冻结协议

- 协议：`highfreq_native64_support_probe_v1`；seed0 × k2；CPU threads=2；无 GPU。
- 每个 query 只用同类另一张 `/train/good/` support 图建 memory；缓存 ID 已调用 `assert_fit_ids_are_support`，没有读取 `/test/`。
- synthetic / nuisance query 都从其自身渲染图直接构造 raw 描述子；clean query 只进入正常光度敏感性控制，没有 clean counterpart oracle。
- 24 个 unique mask 与 v14 cache 逐字节核对；mask 只作评价标签，未参与描述子、memory 或归一化。
- 所有描述子维度、Gaussian sigma、cell 大小、median/MAD、NN 和映射参数在读取 AP 前冻结，没有扫描。

## 首轮结果（support-derived）

| 方法 | thin_scratch AP | cutpaste AP | 两族等权 AP | 正常 clean/光度 AUC | 光度均分−clean均分 |
|---|---:|---:|---:|---:|---:|
| raw_hf_grad64 | 0.0006 | 0.0082 | 0.0044 | 0.4341 | -0.001182 |
| raw_only64 | 0.0006 | 0.0062 | 0.0034 | 0.5150 | +0.000000 |
| dino_full32 | 0.1313 | 0.7120 | 0.4216 | 0.5551 | +0.018979 |

相对 DINO full control 的两族等权 AP：

- raw_hf_grad64：`-0.4172`（thin_scratch `-0.1307`，cutpaste `-0.7037`）。
- raw_only64：`-0.4182`（thin_scratch `-0.1307`，cutpaste `-0.7057`）。

首轮结论：`LOW_LEVEL_SIGNAL_NEGATIVE_OR_NON_INDEPENDENT`。该小样本只说明本口径下的低层信号是否可观察，不能改变旧 STR 的归档结论，也不能写成论文主方法。

mask audit：unique masks=24，cache parity=True；AP rows=72。

复现命令：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_highfreq.py`
