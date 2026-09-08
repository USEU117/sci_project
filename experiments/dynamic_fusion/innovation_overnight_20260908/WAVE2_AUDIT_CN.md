# WAVE 2 证据复核（2026-09-08）

本审计只读复核了 `scripts/innovation_overnight_20260908/probe_shift_memory.py`、`scripts/innovation_overnight_20260908/probe_support_selection.py` 及其 `RESULTS.json`、`PROTOCOL.json`、`DECISION_CN.md`。本文件是证据边界记录，不把 support-only 探针写成论文结果。

## 先给结论

- shift-memory 的 native-mask 评价在形状和像素数上是成立的：`16 × 16` score 经过固定 `64 × 64` nearest repeat 后确为 `1024 × 1024`，原始 mask 未下采样，所有 `108` 个唯一 synthetic episode 均有有效 AP（对应 `540` 个方法行）。但这只是每个 score cell 广播到一个 `64 × 64` 常数块，不能称为 native-pixel localization，也不能恢复 cell 内边界。
- 两个 probe 都有 leave-image-out 证据支持：shift 的 `1,500` 行中没有 held-out 路径进入 memory；support-selection 的 `24` 个 fold 都用其余 `3` 张图做选择，facility 路径没有 held-out 路径。support-selection loader 还对 manifest support 和 `/test/` 做了硬检查。shift loader 本身没有再次调用该 manifest 硬检查，但对实际 cache 的 `6` 类、`k 2` refs 做了独立 manifest 核验，全部通过。
- shift 的绝对 nuisance shift 变小不能单独解释为 photometric robustness。向 KNN memory 增加行会使 `1 - max cosine` 逐 query cell 单调不增；这会机械性地收缩分数。equal-budget duplicate-clean 控制排除了“重复行数本身”的影响，却不能排除新增 unique nuisance rows 带来的 max-score 收缩。
- 现有 pooled nuisance-vs-clean AUC 是一个尺度不变的排序指标，`augmented_all5` 从 `0.537370` 降到 `0.522443`，更接近 `0.5`，且 `12 / 12` 个 fold 的 AUC 都下降；这是超过单纯 duplicate 控制的弱支持。但 AUC 不是 defect separation，未给置信区间，也没有进入 shift 的判定门，仍不能支持鲁棒性结论。
- facility 的随机对照是每 fold 对 `C(3,k)` 个组合的精确均值，并非少量随机抽样。各 strategy 都是 `24` fold；facility k 1 相对 uniform 均值 `Δ = -0.004196`，k 2 为 `+0.006264`，但 k 2 只有 `12 / 24` fold 为正，类别和 fold 间波动明显。该结果应继续记为负/不稳健探索信号。

## 1. score 到 native mask 的等价性

shift probe 中 `_score_original_grid()` 先把每个 query 得到的向量分数 reshape 成 `16 × 16`；`_score_to_original()` 使用两次 `np.repeat(..., CELL_SIZE = 64)`。因此输出形状严格为 `16 × 64 = 1024` 的两维数组。小型 CPU 验算得到：输出 `shape = (1024, 1024)`，全部 `256` 个 `64 × 64` block 分别与对应 score cell 相等。

`load_cells()` 还硬检查 `syn_masks.shape = (k, 9, 1024, 1024)`；`_ap()` 直接把 native mask 和 repeat 后的 score map 展平计算 AP。结果中 synthetic 行的 `mask_resolution` 全为 `1024 × 1024`，`score_grid` 全为 `16 × 16`，没有因 grid 不可见而删除 thin-scratch episode。

因此，“native mask 像素数保持”准确；“native 分辨率定位”不准确。每个 `64 × 64` 区块内的像素分数完全相同，thin-scratch 的 AP 仍然可能有效，但其空间排序只在 `16 × 16` cell 粒度发生。support-selection 使用的是 `16 × 16` 到 `1024 × 1024` 的 bilinear resize，kernel 不同；两个 probe 的 AP 不应跨表直接比较。

## 2. held-out 数据角色与 family 诊断

shift 的主路径在每个 held-out image `h` 上先构造 `bank_idx = all indices except h`，再把 `clean[bank_idx]` 和 `nui[bank_idx]` 交给 `_prepare_banks()`。held-out 的 clean、synthetic、nuisance 只作为 query；held-out synthetic mask 只作为评价标签。结果元数据中 `bank_source_indices` / `memory_ref_indices` 均为非 held-out index；对全部 `1,500` 行按路径检查，held-out ref 进入 memory 的次数为 `0`。实际 v14 k 2 cache 的 `12` 个 refs 与 manifest support 集合逐类逐 shot 核验通过，且没有 `/test/` 路径。

support-selection 的 `_evaluate_fold()` 先以 `candidate_indices = [i for i in range(4) if i != held_out]` 构造候选，将仅含这 `3` 张 clean 图的数组传给 `_facility_coverages()` 和 `_strategy_specs()`；选择完成后才读取 held-out clean、nuisance、synthetic query 和 mask。`24` 个 fold 的 candidate pool 均为 `768 = 3 × 256` cells；逐 fold 的 selected source index 和 path 都没有 held-out，loader 也显式调用 `assert_fit_ids_are_support()` 并拒绝 `/test/`。

有一个必须保留的边界：`augmented_loo_family` 在 nuisance 评价时根据 held-out nuisance 的 family index 排除同族 memory，代码把它标成 `family_excluded_for_diagnostic = true`。这没有把 held-out image feature 放入 memory，但使用了 query family 角色，属于预注册的 transductive leave-family-out 诊断；只能报告为诊断，不能当作无路由部署策略。可部署的 `augmented_all5` 不读取 query family。

## 3. score shrink 与尺度不变证据

shift 的主要汇总如下；AUC 是每个 fold 的 cell-pooled nuisance（label 1）对 clean（label 0）分数 AUC，再对 `12` 个 fold 求均值；其余 Δ 相对 `clean_only`。

| method | memory rows | nuisance-clean AUC | Δ AUC | Δ mean abs shift | Δ p 95 abs shift |
|---|---:|---:|---:|---:|---:|
| `clean_only` | `256` | `0.537370` | — | — | — |
| `augmented_all5` | `1,536` | `0.522443` | `-0.014927` | `-0.002178` | `-0.004343` |
| `duplicate_clean_equal_all5` | `1,536` | `0.537371` | `+0.000000` | `≈ 0` | `≈ 0` |
| `augmented_loo_family` | `1,280` | `0.528911` | `-0.008460` | `-0.001527` | `-0.003206` |
| `duplicate_clean_equal_loo` | `1,280` | `0.537368` | `-0.000003` | `≈ 0` | `≈ 0` |

对任意 memory `B` 和增量行 `E`，脚本的分数满足 `score(q, B ∪ E) = 1 - max(cos(q, B), cos(q, E)) ≤ score(q, B)`。所以绝对均值/p 95 shift 的改善本身可能来自 score shrink，而非真正的 invariance。duplicate-clean 在相同行数下几乎不改变分数，说明行数复制本身不是这次变化的来源；但 equal-budget 仍没有消除 unique nuisance rows 造成的单调 max 收缩。

AUC 对严格单调的全局分数缩放不敏感，且 `augmented_all5` 相对 clean 的 AUC 差在 `12 / 12` fold 为负；因此结果保留了一点“nuisance-clean 排序分离变弱”的证据，不是纯粹靠均值变小就能得到的结论。仍有三项限制：AUC 是 cell-pooled 而非 image-level，空间 cell 和同一 held-out image 的 `15` 个 nuisance 变体并不独立；AUC 越接近 `0.5` 只说明 nuisance 与 clean 更难按该分数区分，不等于 defect anomaly separation 变好；当前 G-SHIFT 只门控 absolute shift 和 equal-budget p 95，没有门控 AUC 或 rank-based uncertainty。后续若要主张鲁棒性，至少应预先固定尺度不变的 AUC / rank 指标和独立图级统计，再与 native test 结果配对解释。

## 4. facility 对照的 fold 数与选择隔离

support-selection 的汇总可由结果文件直接复核：6 类 × 4 个 held-out image = `24` 个 fold；每个 strategy 的 `fold_count = 24`。每个 fold 有 3 个 synthetic family × 3 个 seed = `9` 个 held-out query episode，共 `216` 个 unique episode；uniform k 1 / k 2 各在每 fold 穷举 `3` 个组合并取 AP 的精确均值，因此对应 `648` 个组合级 AP 值，不能把它们当作 `648` 个独立样本。

selector 的 facility objective 只对候选池 `3 × 256` clean cells 求 coverage；结果中的 `selected_indices` 是 candidate-local index，`selected_source_indices` / `selected_rel` 映射回原始 support。逐 fold 检查没有 held-out source index 或 held-out path。normal nuisance AUC 和 synthetic mask 读取都发生在选择完成后。`all3` 是 `3` 图 memory 的上限参照，不能与 k 1 / k 2 当作同预算比较。

facility 与 uniform 的 paired fold 差异为：k 1 macro `-0.004196`（`13 / 24` fold 为正、`11 / 24` 为负），k 2 macro `+0.006264`（`12 / 24` 为正、`12 / 24` 为负）。这支持“选择器没有稳定收益”的判定；uniform 是精确组合均值，不是一个带随机 seed 的随机抽样基线。

## 5. 阈值、显著性与非独立样本

shift 的 `G-SYN` 是 synthetic macro ΔAP ≥ `-0.01` 且最差 episode ΔAP ≥ `-0.05`；`G-SHIFT` 要求 mean / p 95 absolute shift 不高于 clean，并在 equal-budget duplicate 上 p 95 不差。它们是候选筛选门，不是零假设检验；没有 p 值、置信区间、预注册的多重比较校正或独立测试集。

shift 的有效单位是 `12` 个 leave-image-out folds（`108` 个 unique synthetic episode 共享同一批 held-out 图和 renderer）；46,000 级别的 cell 数不能当成同样多的独立观测。facility 的 `24` 个 fold 嵌套在 `6` 个类别内，同类别 fold 共享候选池，且每个 held-out image 的 family / seed 变体相关。所有差值应作为 paired descriptive statistics；不能用这些结果声称 statistical significance、稳健泛化或论文创新确认。

## 可复现证据入口

- shift 代码：`scripts/innovation_overnight_20260908/probe_shift_memory.py`
- shift 结果：`experiments/dynamic_fusion/innovation_overnight_20260908/shift_memory/RESULTS.json`
- facility 代码：`scripts/innovation_overnight_20260908/probe_support_selection.py`
- facility 结果：`experiments/dynamic_fusion/innovation_overnight_20260908/support_selection/RESULTS.json`
- 两个协议、逐 fold 路径、memory rows、mask resolution、selection metadata 均保留在相应目录；本次审计只新增本文件。
