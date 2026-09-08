# WAVE 4 跨实验口径与初步深研排序审计（2026-09-08）

本审计只读检查 full896_extension、scale_fusion、real_resolution 的 protocol、脚本、cache 和结果入口。除本文件外没有修改脚本或旧产物，也没有重跑实验。real_resolution 的脚本和 protocol 在审计期间出现；截至本次 snapshot 已有 PARTIAL_RESULTS.json / PER_IMAGE.jsonl，状态仍为 running，尚未有最终 RESULTS.json，因此只审实现和数据角色，不能把它标成完成结果。

## 结论

1. full896_extension 的新增四类是严格 paired 的 fresh full44832 / full896：48 个 synthetic episode 中 query、bank、mask、grid 和两 arm 均对齐，mask cache 校验全部通过。新四类每类 12 个 mask；与旧两类各 12 个合并后，六类各 12 个，summary 的 72 条记录是等类权的 episode-level 算术均值。
2. 六类合并的等类权成立，不等于六类使用同一套 448 控制。新增四类的 448 值来自本脚本同 episode 的 fresh full44832；旧两类的 448 值由旧 full896 结果中记录的 tiling full_dino32 control 提供。旧控制的路径、mask 和 AP 经单独只读复核为 24 / 24 匹配，但没有重新提取，不能把合并 delta 写成六类同脚本重测的 paired estimate。
3. scale_fusion 的 multiscale_concat64 为 0.585361，late_mean64 为 0.581693，差只有 0.003669。thin_scratch 上 late 反高 0.000372，cutpaste 上 concat 高 0.007709；late 是必须保留的强控制，不能把 concat 的小增益写成独立的特征级创新。
4. real_resolution 的关键数据角色和匹配口径目前是合格的：6 类完整 test cache、manifest seed 0 / k2 的两张 train/good memory、test query 排除出 memory、两 edge 同 DINO / distance / dists2map、full 448 pixel AP/AUROC 与 image AUROC。结果未落盘前只能列为待验证入口。
5. 当前深研顺序应先用真实 test resolution-only 诊断决定 full896 是否值得继续，再评估 resolution × storage 的工程折中，最后才考虑多尺度 concat。后者若不能在真实 test 和 late control 上保持清晰优势，不进入论文创新主线。

## 1. full896_extension 的 paired 口径

脚本 probe_full896_extension.py 的 _make_extractor 使用同一冻结 DINOv2 ViT-B/14 权重，分别对同一 rendered image 做 edge 448 与 edge 896 的 BICUBIC + antialias 预处理，得到 32 × 32 与 64 × 64 grid。_nearest_dist / _score_map 两臂共用 raw row L2、归一化特征的 Euclidean 等价距离和 src.utils.dists2map；两臂使用同一原始 1024 × 1024 mask 计算 pixel AP。protocol 明确 memory 只含同类另一张 k2 support 图，query 不进入 memory。

对 outputs/dynamic_fusion/overnight_20260908_full896_extension/RESULTS.json 的 synthetic 行做了只读 grouping：

- 总行数 192，其中 clean 16、synthetic 96（每 episode 两 arm）、nuisance 80；
- synthetic 有 48 个 paired_episode_id 和 48 个 unique mask sha；
- 每个 paired episode 恰有 full44832 与 full896_native64 两行；
- 按 paired_episode_id 比较 query_rel、bank_rel、mask_sha256、grid 和 arm，48 组 mismatch = 0；
- 96 条 synthetic 行的 mask_cache_verified 全为 true，strict_leave_image_memory 全为 true；
- 新四类每类 24 条 arm 行，即 12 个 unique mask；每类每个 family 6 个 mask（2 query × 3 seed）。

新四类的结果是 full44832 0.441014 → full896 0.522257，delta +0.081243。thin_scratch 四类的 delta 均为正，类别均值范围 +0.090636 至 +0.236097；cutpaste 三类为负，分别 -0.007590、-0.022858、-0.008313，tubes 为 +0.044976。按 family 的新四类平均 delta 为 thin_scratch +0.160933、cutpaste +0.001554，说明当前信号主要来自细小 scratch，不能写成所有 defect type 的普遍 resolution gain。


## 2. 旧两类与新增四类的合并权重

代码 _legacy_records 读取旧 full896 RESULTS.json 的 24 条 synthetic full896 行，并取每行已有的 control_full_dino32_ap；它只硬检查 control AP 存在、类别集合和期望行数，不在合并函数中再次验证旧 control 的 query / bank / mask identity。新增记录则由 _pair_rows 先在本脚本内配对 full44832 与 full896，再取 full896 - full44832。最终 _group_summary 对 records 逐条 np.mean。

独立只读核对旧 full896 与旧 tiling full_dino32 control 的 key 为 category + query_rel + kind + seed，结果为：

- 旧 synthetic full896 行 24 条，bracket_black 和 metal_plate 各 12 条；
- control AP、query_rel、bank_rel、mask_sha256 均 24 / 24 匹配；
- 新四类 full896 记录 48 条，四类各 12 条；
- merged records = 72，六类各 12 条，每类两个 family 各 6 条。

所以 SUMMARY.json 中 six_class_merged 的 0.463093 → 0.535253、delta +0.072160 在记录数量上是等类权的。它是 72 个 episode AP 的等权平均，不是把所有像素拼接后的 pooled AP。解释边界仍须保留：旧 24 条的 448 对照是历史 tiling control，新 48 条的 448 对照是本轮同脚本 fresh full44832；合并值适合做候选上下文和类均衡摘要，不适合宣称六类均来自同一轮同脚本 full448 / full896 paired experiment。

此外，full896 的 4096 token attention proxy 为 16,777,216，full44832 为 1,048,576，quadratic proxy 为 16 ×。新四类的正向 AP 不能单独转换为等算力收益。

## 3. scale_fusion 的 late 控制不能省略

scale_fusion/PROTOCOL.json 固定了四个 arm：

- full896_native64；
- dino448_bilinear64；
- multiscale_concat64：两尺度各 row-L2，等权 1 / sqrt(2) 后 concat，再 row-L2；
- late_mean64：两张单尺度 64-grid NN score map 做算术均值，再走共同 map 后处理。

24 个 unique synthetic mask、两个类别、两 query、两 family、三 seed 均在四 arm 同时计算；所有 arm 共用 raw row L2、1 - cosine 等价 NN、dists2map。memory 只用另一张 support 图，未读取 test，也没有从 AP 选参数。

结果中 concat 相对 late 的差值如下：

| 范围 | multiscale_concat64 | late_mean64 | concat - late |
|---|---:|---:|---:|
| synthetic overall | 0.585361 | 0.581693 | +0.003669 |
| thin_scratch | 0.340310 | 0.340682 | -0.000372 |
| cutpaste | 0.830413 | 0.822703 | +0.007709 |
| bracket_black | 0.574028 | 0.571412 | +0.002615 |
| metal_plate | 0.596695 | 0.591973 | +0.004722 |

concat 相对最佳单支 full896 的 +0.024115，late 相对最佳单支的 +0.020447；因此 concat 相对 late 的真实增量只有 +0.003669。这个 late control 具有相同两尺度输入、相同 support 角色、相同 mask 和相同后处理，只把特征级 concat 改成 score-level mean。当前输出已经把二者定位为工程 controls、main_table = false；后续任何“特征融合创新”表格必须同时列 late，并报告 paired per-category / per-family 差值。


## 4. real_resolution 脚本与 protocol 复核

当前出现的 scripts/innovation_overnight_20260908/probe_real_resolution.py 与 real_resolution/PROTOCOL.json 的实现口径如下。

数据角色和覆盖：

- TEST_CACHE 指向 v3_direction_a 的 anomalydino_visual category cache；
- 预期 6 类样本数为 79、77、60、44、97、101，总计 458，其中 normal 176、anomaly 282；
- _load_test_cache 硬检查 sample_ids 数量和唯一性、imgs_masks 为 N × 448 × 448、gt_sp 形状、dataset = mpdd、seed 0、shot 2、branch = anomalydino_visual；
- 每个 sample_id 必须是该类 /test/ 路径，raw image 必须存在；normal mask 必须全零，anomaly mask 必须非空；
- 只读扫描现有六类 cache 得到 79、77、60、44、97、101，全部 sample_id 为 /test/，mask 形状均为 448 × 448，patch_features 为 32 × 32 × 768，ref_patch_features 为 2 × 32 × 32 × 768；
- cache 元数据 dataset_role = development 是 v3 产物的既有标记，不能代替路径角色判断。脚本实际以 /test/ 路径、label / mask 一致性和 raw file 存在性作硬检查；报告应继续称其为 MPDD development diagnostic，而不是独立外部确认。

memory 和 matching：

- 脚本先对 manifest seed 0、shot 2 调用 support_paths 与 assert_fit_ids_are_support；
- 每类恰取两张 train/good support 原图，并分别以 edge 448 / 896 重编码形成 K = 2 bank；不做 LOO；
- 随后按 test cache 的 sample_ids 读取 test raw image，query 不进入 memory，row 中记录 memory_leave_one_out = false、query_in_memory = false；
- 两 edge 共用一个冻结 DINOv2 ViT-B/14、ImageNet normalization、raw row-L2 后的 1 - cosine NN 和相同 dists2map，输出均为 448 × 448 map；
- 脚本只从 evaluate_a1_feature_fusion.py 读取 STRIDE = 8 常量以复现采样诊断，不运行 A1 fusion，也不使用 test 标签调参。

指标和边界：

- full 448 × 448 map 是主 pixel AP / pixel AUROC；
- stride 8 仅是 A1-compatible pixel AP / AUROC 诊断，并记录 fine mask positive pixels、消失计数和比例；
- image score 固定取 448 × 448 map 的 max，报告 image AUROC / image AP；
- 逐类、逐 defect type 与 category macro 输出，AUPRO 不计算；
- protocol 明确 test labels / masks 只作 evaluation，不拟合、不选类别、不选 defect、不选参数；
- 只有两臂覆盖全部 458 张且 6 类 summary 完整时才允许 RESULTS.json 标记 complete。

当前只看到 PARTIAL_RESULTS.json / PER_IMAGE.jsonl，snapshot 状态为 running；PROTOCOL 已记录 loaded_total = 458，尚无最终 RESULTS.json，因此尚无可引用的完整真实 test AP / AUROC 证据。脚本级 protocol 已满足本轮约定；最终审计仍需检查实际输出的 status、loaded counts、completed categories、rows = 916、每类完整性，以及是否出现 partial / OOM 后误标 complete。


## 5. 初步深研排序与可执行门槛

第一优先是 real_resolution。它直接回答 full896 的候选信号是否能从 support-synthetic 延伸到真实 MPDD test，并且不混入 A1 分支。接受结果至少要有 6 类全覆盖、每类 full pixel AP / AUROC 和 image AUROC、per defect type、mask stride8 消失率，以及同 scorer 的 full448 control；任何 partial 都只作为失败或待续记录。

第二优先是 resolution × storage 的工程验证。full896_extension 已显示 scratch 的输入分辨率信号，但 full896 的 token / attention proxy 显著更高；只有在固定 test scorer 下同时报告 FP32 / 压缩 bank 的 AP、AUROC、解码内存和 latency，才可判断是否有实际部署价值。压缩结果不能把接近 full448 的持久化字节误写成等算力。

第三优先是保留 scale fusion 作为候选工程 control。先在 real test 或独立冻结 split 上做 concat 与 late 的 paired per-category 比较；若优势仍只有约 0.0037，或只在 cutpaste 出现，则更合理的结论是“两尺度 score averaging 已解释大部分收益”，不进入特征级创新主线。

full896_extension 本身保留为输入分辨率 / context 候选证据：六类合并 scratch 平均 delta +0.148889，而 cutpaste 为 -0.004568；它值得由 real_resolution 决定是否深挖，但不应把六类合并绝对 delta 与 highfreq 或旧异 scorer 跨表相减。

证据入口：scripts/innovation_overnight_20260908/probe_full896_extension.py、probe_scale_fusion.py、probe_real_resolution.py；experiments/dynamic_fusion/innovation_overnight_20260908/full896_extension、scale_fusion、real_resolution；以及 outputs/dynamic_fusion/overnight_20260908_full896_extension、overnight_20260908_full896、overnight_20260908_tiling、v3_direction_a/features_vitb14_s0_k2/anomalydino_visual。


