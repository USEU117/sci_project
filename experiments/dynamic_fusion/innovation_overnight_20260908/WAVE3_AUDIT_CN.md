# WAVE 3 跨脚本口径与样本去重审计（2026-09-08）

本审计只读检查 probe_full896.py、probe_tiling.py、probe_highfreq.py、probe_precision_native.py、probe_scale_fusion.py 及其 PROTOCOL、RESULTS、REPORT、AUDIT 文件。除本文件外没有修改脚本、旧结果或其他产物，也没有重跑大型实验。

## 结论

1. highfreq 报告中的 DINO mean AP = 0.421633 不能作为 full448 baseline 与 tiling full_dino32 = 0.507251 跨表相减。两条代码路径同时改变了 feature 几何和 map 后处理；差异不是单一的插值因素。
2. full896 使用的两个 tiling 控制在 synthetic AP、query / bank 路径和 mask hash 上逐 episode 对齐，可以在 full896 表内作 paired delta；计算量和 normal FP 定义仍不可直接等同。
3. precision_native 的 648 synthetic 行计数完全一致于 24 fold × 9 episode × 3 arm。9 是每 fold 的 3 个 synthetic family × 3 个 seed；不能写成“每族只有 1 个变体”。
4. scale_fusion 已有冻结且可复核的独立 probe protocol，但它是缓存支持集探针，不是独立外部测试。当前结果只能作为探索信号。

## 1. highfreq 0.421633 与 tiling full448 0.507251 的原因

probe_highfreq.py 的 DINO 路径在 bank 上拟合逐维 median / MAD，再对 bank 和 query 做 row L2；随后使用 1 - max cosine，最后只做 cv2.INTER_LINEAR resize。代码位置：probe_highfreq.py:295-332。

tiling full_dino32 和 full896 都走标准路径：raw feature row L2、sqrt(max(2 - 2 × max cosine, 0))、src.utils.dists2map。后者是 INTER_LINEAR resize 后接 Gaussian sigma = 4。代码位置：probe_tiling.py:181-212、probe_full896.py:191-207、src/utils.py:10-12。

对缓存 v14 feature 和同一原始 1024 × 1024 mask 做的只读 CPU 隔离如下；最后一行与 tiling 的数值差只约为 1.5 × 10^-6：

| 重放口径 | feature / distance | map 后处理 | synthetic mean AP |
|---|---|---|---:|
| highfreq 原路径 | memory median / MAD + row L2；1 - cos | linear only | 0.421633 |
| 仅加 Gaussian | memory median / MAD + row L2；1 - cos | linear + sigma = 4 | 0.440392 |
| 去掉 median / MAD | raw row L2；1 - cos | linear only | 0.484406 |
| raw 几何 + 标准 map | raw row L2；1 - cos | dists2map | 0.506205 |
| tiling 等价路径 | raw row L2；Euclidean sqrt(2 - 2 cos) | dists2map | 0.507253 |
| tiling full_dino32 报告值 | 同上 | 同上 | 0.507251 |

因此，Gaussian map 平滑单独已经把 highfreq 原路径从 0.421633 推到 0.440392；raw feature 几何再把 linear-only 结果推到 0.484406，标准 map 后处理到 0.506205，最后 Euclidean 形式得到 tiling 数值。median / MAD 是 memory-only 的变换，但会改变 nearest-neighbor 几何，不能从解释中省略。

执行口径：禁止计算 0.507251 - 0.421633 并称为 resolution gain。0.421633 应标为“highfreq MAD + linear diagnostic”，不应标为“full448”。如果需要跨脚本复核，必须先统一 raw row L2、Euclidean distance、dists2map、mask 和 AP 的整条 scoring pipeline。

## 2. full896 与 full448 / tile 控制的可比性

full896 和 tiling full-image 控制使用相同冻结 DINO wrapper、checkpoint、ImageNet normalization、BICUBIC + antialias 预处理；输入 edge 分别是 896 与 448。tile_reencode64 是四个 512 × 512 crop 各自以 edge 448 重编码成 32 × 32，再拼接为 64 × 64。因此输入 scale / receptive field 是实验因素，不能把它们描述成同一 feature。

可比部分是评分与标注：两边都使用 raw row L2、Euclidean nearest distance、dists2map（linear resize + Gaussian sigma = 4）、原始 1024 × 1024 mask 和 average_precision_score。probe_full896.py 还逐 control 检查 query_rel、bank_rel 和 synthetic mask sha；只读复核中 24 个 synthetic episode 的 48 条 control reference（full_dino32 + tile_reencode64）全部匹配，未发现 path / mask mismatch。full896 与 tiling synthetic 行均带 strict_leave_image_memory = True。

所以 full896 表内的 paired synthetic 平均差值可用：

- full896 = 0.561246；
- 相对 full_dino32 = 0.507251，delta = +0.053995；
- 相对 tile_reencode64 = 0.478041，delta = +0.083205。

这些 delta 只表示当前同 episode、同 mask、同标准 scorer 下的控制差异。不能据此作 equal-compute 结论：full896 为 64 × 64 = 4096 token，attention proxy 约 16,777,216；四 tile 合计同为 4096 token，但每 tile 的 proxy 合计约 4,194,304，约相差 4 ×。normal FP 也不能跨表比较：full896 结果中的 normal_fp_rate 为 None，tiling normal 行使用由 p99 得到的 threshold / FP 定义。

## 3. precision_native 的样本去重和分母

只读统计 precision_native/PER_EPISODE.jsonl：

- 总行数 1080；
- synthetic = 648，normal_nui = 360，normal_clean = 72；
- 三 arm 各 360 行；
- folds 为 24 = 6 category × 4 heldout image；
- 每 fold 有 9 = 3 synthetic family × 3 seed 个 synthetic episode；
- 每个 synthetic episode 在 control_f32、fp16_roundtrip、int8_pc_roundtrip 三 arm 重复一次。

因此：

648 = 24 fold × 9 synthetic episode / fold × 3 arm。

每个 family 每个 arm 是 24 × 3 = 72 行，三个 family 合计每 arm 216 行。每族一个变体只适用于 nuisance normal 查询：每个 nuisance family 固定一个 strength 0 样本（每 heldout 共 5 个 nuisance normal）；不适用于 synthetic，synthetic 每 family 有 3 个 seeds。报告和后续表格应同时写清 family、seed、arm 和 fold 分母，避免把 648 当成 24 个 unique synthetic episode。

## 4. scale_fusion protocol 是否独立、可复核

scale_fusion/PROTOCOL.json 已在运行前冻结，script_version = scale_fusion_probe_v1、protocol_version = scale_fusion_cached64_support_probe_v1；脚本的 _freeze_protocol 会在已有 protocol 与当前参数不一致时失败。protocol 明确写出：

- full896 native 64 × 64 × 768 与 v14 DINO448 32 × 32 × 768 的 cache 角色；
- 以 ref_rel 对齐 episode，不依赖行位置；
- full896 / v14 support manifest、heldout 角色、cache shape 和原始 mask byte parity 检查；
- full896_native64、dino448_bilinear64、multiscale_concat64、late_mean64 四个固定 arm；
- 所有 arm 共用 raw row L2 → Euclidean → dists2map；
- memory 只取其他 support image，heldout query 不进 memory；无 test、无参数拟合、无 type routing，main_table = false。

结果计数也闭合：24 个 unique synthetic masks × 4 arms = 96 AP rows；4 个 clean normal × 4 arms = 16 rows；MASK_AUDIT 为 24 行、24 个 unique mask，full896 / v14 mask parity 为 true。

“独立”应限定为 protocol / scorer / cache-role 可审计的独立 probe。它仍复用旧运行生成的 cache 和 renderer mask，只覆盖 2 个 category、k2、seed 0、24 个 synthetic masks，且没有 nuisance normal，因此不能当作独立外部测试、泛化验证或 shift robustness 证据。当前 concat / late mean 的提升只能留作探索信号；protocol 没有参数扫描，这一点可保留为固定 control 的优点。

## 可执行建议

1. 所有后续表格把 highfreq 0.421633 标成 MAD + linear diagnostic，从 full448 / full896 delta chain 排除。
2. 需要比较 resolution、tiling 或 fusion 时，统一一个 scorer：raw row L2 → Euclidean nearest distance → dists2map，并在结果中保存 query / bank / mask identity；AP 与 normal FP 分开报告。
3. precision_native 文案写成“每 family 3 个 synthetic seeds、24 个 folds、3 个 arms”，把 nuisance 的“每 family 一个固定 strength 0 变体”另行表述。
4. scale fusion 下一步应做同 scorer 的 per-category paired comparison，并补充未参与 cache 选择的 test / 外部 split 和 nuisance stability；在此之前不进入论文主表或作鲁棒性结论。

证据入口：scripts/innovation_overnight_20260908/probe_highfreq.py、probe_tiling.py、probe_full896.py、probe_precision_native.py、probe_scale_fusion.py，以及 experiments/dynamic_fusion/innovation_overnight_20260908/ 下对应的 highfreq、tiling、full896、precision_native、scale_fusion 目录。

