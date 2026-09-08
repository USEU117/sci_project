# 缓存多尺度匹配控制探针记录（2026-09-08）

## 目的与边界

本 probe 只复用既有 full896 clean/synthetic feature 与 v14 DINO448 clean/synthetic feature；没有下载模型、没有新特征提取。它检验 context/detail 两尺度联合匹配能否在固定配置下减少单尺度取舍。多尺度拼接与 late mean 都是已知的工程控制，不能称论文新颖性。

## 冻结协议与数据角色

- 协议：`scale_fusion_cached64_support_probe_v1`；seed0 × k2；CPU threads=2；GPU=False。
- full896 是缓存的 64 × 64 × 768 高分支；v14 DINO448 是缓存的 32 × 32 × 768 低分支，低分支只做固定 bilinear → 64 × 64。
- 每个 query 只用同类另一张 support 图建 memory；synthetic query 使用自身 cached variant，未使用 clean 对应图或对齐 oracle。
- 24 个 mask 由 full896 cache 作为标签，并与 v14 cache 逐字节核对；没有读取 `/test/`。
- single、concat、late 四个 arm 同时计算；concat 固定两支等权后 L2，late 固定两张 64-grid score map 均值，没有参数扫描、type routing 或按既有 tiling 结果选参。
- normal 只报告 clean score 分布（mean / p95 / p99），不额外提取 nuisance feature。

## 首轮结果（support-derived）

| arm | thin_scratch AP | cutpaste AP | 两族等权 AP | clean score mean | clean score p95 | clean score p99 |
|---|---:|---:|---:|---:|---:|---:|
| full896_native64 | 0.3387 | 0.7838 | 0.5612 | 0.419506 | 0.611117 | 0.737129 |
| dino448_bilinear64 | 0.2350 | 0.8172 | 0.5261 | 0.439951 | 0.632634 | 0.724811 |
| multiscale_concat64 | 0.3403 | 0.8304 | 0.5854 | 0.458612 | 0.650174 | 0.749229 |
| late_mean64 | 0.3407 | 0.8227 | 0.5817 | 0.429728 | 0.613814 | 0.717514 |

结论：`FIXED_MULTISCALE_CANDIDATE_ONLY`。One fixed two-scale control exceeds the best single arm in this support-synthetic probe; this is not a routing or paper novelty claim.

mask audit：rows=24，unique masks=24，full896/v14 parity=True；AP rows=96。

复现命令：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_scale_fusion.py`
