# 跨输入分辨率残差控制探针记录（2026-09-08）

## 历史审计与路线

历史 `probe_resolution_storage.py` 只将 full448 / full896 作为独立 NN 臂比较；既有 `scale_fusion` 只做 high、low、固定 concat 与 late mean。没有完成跨输入分辨率 feature residual 或 low→high normal-memory ridge。故本轮锁定为有界控制，不声称方法新颖。
相关先例：HLGFA（2026, v3，https://arxiv.org/abs/2602.09524）已涉及冻结 backbone 的高低分辨率一致性、structure/detail 分解与 gated residual；本 probe 只检验 few-shot normal-only 闭式 residual 的支持集质量/成本，不主张 residual 概念原创。

## 协议与数据角色

- 协议：`scale_residual_cached64_support_probe_v1`；bracket_black / metal_plate，seed0 × k2，CPU threads=2，无 GPU。
- 复用 full896 64-grid 与 v14 DINO448 32-grid cache；低分辨率固定 bilinear 到 64-grid；未下载、未提取新 feature。
- 每个主 synthetic query 的 high / low 都来自自身 cached variant；memory 仅为同类另一张 clean support 图，未使用 clean counterpart oracle。
- 24 个原始 1024 mask 全部 high/low cache 逐字节一致；normal 只报 clean score 分布，不当 FPR。
- ridge 固定 `alpha=1.0`、无 intercept，仅用另一张 clean support 图拟合；wrong-pair 固定使用同一 query 图的下一个 seed，未按 AP 调整。

## 首轮结果（support-derived）

| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |
|---|---:|---:|---:|---:|---:|---:|
| high_only64 | 0.3387 | 0.7838 | 0.5612 | 0.419506 | 0.611117 | 0.737129 |
| low_only64 | 0.2350 | 0.8172 | 0.5261 | 0.439951 | 0.632634 | 0.724811 |
| late_mean64 | 0.3407 | 0.8227 | 0.5817 | 0.429728 | 0.613814 | 0.717514 |
| direct_residual64 | 0.0032 | 0.0119 | 0.0075 | 0.540355 | 0.753025 | 0.869670 |
| high_residual_concat64 | 0.3347 | 0.7676 | 0.5512 | 0.641930 | 0.844317 | 0.946427 |
| ridge_residual64 | 0.2161 | 0.5728 | 0.3944 | 0.506365 | 0.681847 | 0.754239 |
| wrong_pair_residual64 | 0.1121 | 0.4299 | 0.2710 | nan | nan | nan |

关键差值：direct residual − late = `-0.574166`；ridge residual − late = `-0.187243`；high/residual concat − late = `-0.030542`；wrong-pair residual − direct residual = `+0.263491`。

结论：`NO_RESIDUAL_GAIN_OVER_FIXED_BASELINES`。Direct and ridge residuals are bounded controls. Any positive result is support-derived and cannot be called a new algorithm; the wrong-pair arm is a fixed correspondence sanity control.

mask audit：rows=24，unique masks=24，high/low parity=True；AP rows=168。

复现命令：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_followup_20260908/probe_scale_residual.py`
