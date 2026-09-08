# 空间跨图对应探针首轮记录（2026-09-08）

## 范围与数据角色

- 协议：`spatial_global_translation_bwnn_v1`；seed0 × k2 × 六类；CPU threads=2。
- fit/select 只使用每类 manifest 的 `/train/good/` K-shot clean support；每个 held-out 图 h 都从 memory 排除。
- 每个 synthetic / nuisance query 都用自身特征独立估计到 clean support memory 的平移；clean query 只用于正常控制，不能为 synthetic query 提供对应图。
- 未读取 `/test/` 图像、真实 defect 标签或主表；`v14_p1_support` 缓存的 `ref_rel` 已逐类调用 support 断言。

## 冻结机制

将 32 网格 A1 0.5/0.5 融合特征在 16 网格做 2×2 mean-pool；每个实际 query 与每个 clean support memory 对在 ±2 个 16-grid cell 内比较 overlap cosine，丢弃最低 20% 后取 normal-majority 均值，平移映射回 32 网格。查询 cell 只在平移后中心的 Chebyshev 半径 2 内做 NN。

对照为无空间限制的全局 NN，以及保持同一平移估计、只对每个 memory 图做固定坐标置乱的 control。没有根据 AP 或 mask 选择平移范围、带宽或类别。

## 首轮实测（support-derived）

| 方法 | cutpaste AP | erasure AP | 正常 clean/光度 AUC | 光度均分−clean均分 |
|---|---:|---:|---:|---:|
| global_nn | 0.6320 | 0.9512 | 0.5334 | +0.004564 |
| spatial_bw_nn | 0.3762 | 0.8302 | 0.5297 | +0.004565 |
| spatial_bw_nn_coord_shuffle | 0.1956 | 0.7024 | 0.5121 | +0.003319 |

全局 NN 是排序基线；空间带宽方法相对它的 AP 差值：

- cutpaste：`-0.2558`；坐标打乱 control `-0.4365`。
- local_erasure：`-0.1210`；坐标打乱 control `-0.2488`。

首轮结论（负）：独立 query 对齐后的空间带宽 NN 在两类 support-derived 代理上均低于全局 NN（cutpaste −0.2558、erasure −0.1210）；坐标打乱进一步下降，表明带宽限制确实施加了空间约束，但本批数据中该约束损害了异常排序，不能保留为候选增强。

query-to-memory 平移拟合数 264（按 query 类型：{'clean_normal': 12, 'nuisance': 180, 'synthetic': 72}），trimmed overlap mean cosine=0.9411，平均 32-grid L1 平移=0.000。

## 负结果与复现边界

这是有界机制探针，不构成真实 MPDD 泛化或论文主表证据。若空间 AP 未超过全局 NN，结论是本批 support-derived 代理上没有观察到跨图带宽限制的独立收益；若坐标打乱同样提升，则提升不能归因于真实空间对应。正常 AUC 越接近 0.5 越稳定，需与 AP 一起解读。

复现命令：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_spatial.py --shot 2`

复用历史审计：t1/t5 的机制是同图邻域描述子，PSMF 是图像坐标多移位合并；本探针只测试跨图平移与坐标受限检索，避免重复上述路线。
