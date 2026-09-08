# 缓存多尺度匹配扩类控制记录（2026-09-08）

## 范围与数据角色

本轮只新增 bracket_brown、bracket_white、connector、tubes 四类，复用 full896_extension/features 的 448-edge 32-grid 与 896-edge 64-grid 缓存；没有 GPU、没有新 feature 提取，不改两类原 scale_fusion 结果。

每个 query 只用同类另一张 k2 support 图建 memory；synthetic query 使用自身双分辨率 cached variant，没有 clean counterpart oracle。48 个 mask 与既有 v14 mask 逐字节核对。normal 只报告 clean score 分布，不当作 FPR。

## 四类新结果

| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |
|---|---:|---:|---:|---:|---:|---:|
| full896_native64 | 0.3337 | 0.7108 | 0.5223 | 0.341917 | 0.533767 | 0.645568 |
| dino448_bilinear64 | 0.2055 | 0.7449 | 0.4752 | 0.336838 | 0.527536 | 0.627311 |
| multiscale_concat64 | 0.3122 | 0.7487 | 0.5305 | 0.374898 | 0.566577 | 0.660915 |
| late_mean64 | 0.3113 | 0.7479 | 0.5296 | 0.339378 | 0.521757 | 0.622261 |

四类直接差值：concat − late = `+0.000874`；concat − 最佳单支 = `+0.008213`；late − 最佳单支 = `+0.007339`。

## 六类汇总

六类汇总将本轮四类按类别等权纳入，并只读复用原 scale_fusion 的 bracket_black / metal_plate 类别摘要；没有跨脚本重算旧两类。

| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |
|---|---:|---:|---:|---:|---:|---:|
| full896_native64 | 0.3354 | 0.7351 | 0.5353 | 0.367780 | 0.559550 | 0.676088 |
| dino448_bilinear64 | 0.2153 | 0.7690 | 0.4922 | 0.371209 | 0.562569 | 0.659811 |
| multiscale_concat64 | 0.3216 | 0.7759 | 0.5488 | 0.402803 | 0.594443 | 0.690353 |
| late_mean64 | 0.3211 | 0.7728 | 0.5470 | 0.369495 | 0.552443 | 0.654012 |

六类直接差值：concat − late = `+0.001806`；concat − 最佳单支 = `+0.013514`；late − 最佳单支 = `+0.011708`。

结论：`EXTENSION_COMPLETE_CONCAT_VS_LATE_REPORTED_NO_NOVELTY_CLAIM`。concat 与 late 的差值已直接呈现；结果只用于本轮 support-derived 控制审计，不能写成论文创新。

mask audit：rows=48，unique masks=48，extension/v14 parity=True；new AP rows=192。

复现命令：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_scale_fusion_extension.py`
