# Neighborhood follow-up：既有机制审计（2026-09-08）

状态：`DUPLICATE_NO_RUN`。本文件是本轮唯一新增的 follow-up 产物；没有生成或运行新的 `probe_neighborhood.py`，没有覆盖旧结果，也没有读取 test 图像或标签。

## 直接结论

本轮提议的核心统计量是：对每个 query patch，按 reference image 分开做 NN，得到 3 张 reference 的距离，再取 global minimum、median，并可附加 trimmed mean 或每图 top 2。这个核心已经由历史 CRAM（Cross-Reference Agreement Memory）逐项实现。k4 留一张 heldout 后剩余 3 张 reference 只改变 reference 数量，未改变机制对象；因此按本轮“若历史重复则只审计、不重跑”的规则，停止在审计，不运行新探针。

## 四类历史路线核对

| 历史方向 | 可复核入口 | 已做统计 / 范围 | 审计判断 |
|---|---|---|---|
| normal density / LNDC | `src/industrial_ad/innovation_v2/local_density.py`、`experiments/dynamic_fusion/innovation_v2/FINAL_DECISION.md` | strict LOO 参考 patch 密度 `rho`，再用 A1 距离与 `rho` 的 ratio；k3/k5/k9，shot 1/2/4，含 `global_density_sham` | 已做且归档；属于 density 家族，但不是本轮 per-reference median 的同一公式 |
| LOF | `experiments/dynamic_fusion/innovation_v2/FINAL_DECISION.md:50`、任务书中的停止规则 | 未找到 `LocalOutlierFactor` 或独立 LOF 运行器；历史记录明确禁止继续做 LOF / Isolation Forest / 大范围 density search | 没有可复用的 LOF 结果；不启动被禁止的重复搜索 |
| local reconstruction / LLSE | `scripts/innovation_v10_portfolio/run_r0_explore_llse.py`、`experiments/dynamic_fusion/innovation_v10_portfolio/explore_llse/` | top 8 memory NN 的局部线性重构残差，另有 random-8 locality control；MPDD s0/s1/s2，6 类 | 已做且归档；s0 `mean ΔAP = +0.008896`，s1 `−0.002803`，s2 `+0.001416`，seed 稳健性失败 |
| NN support-consensus / cross-reference agreement | `src/industrial_ad/innovation_v10_portfolio/common.py`、`src/industrial_ad/innovation_v10_portfolio/cram.py`、`scripts/innovation_v10_portfolio/run_r0_cram.py`、`experiments/dynamic_fusion/innovation_v10_portfolio/cram/` | 对每个 reference image 的 patch bank 单独取最小 NN 距离 `d_r`；随后计算 `d_min = min_r d_r`、`d_med = median_r d_r`、MAD 和 `gap = d_med − d_min`；A0/A1/A2 与 med-only、duplicate、shuffled controls；6 类 k2/k4 | 与本轮核心完全重复 |

补充审计：搜索到的 `top2_mean` 位于旧的 branch-fusion strategy，表示分支分数的 top 2，不是每个 reference image 的 top 2 memory NN；旧 spectral / text 代码中的 `trimmed mean` 也不是 reference-image NN 聚合。因此没有一个未做的“同名 top2/trimmed mean”可以把本轮主问题变成独立方向。

## CRAM 的逐代码证据

`common.per_reference_distances(feat_flat, ref)` 将 `ref` 保持为 `[K, H, W, D]`，为每个 reference image 建立独立 FAISS index，并把该图内最小距离写入 `d_r(q)`。`cram.agreement_stats(dr)` 随后直接执行：

```text
d_min = dr.min(axis=-1)
d_med = median(dr, axis=-1)
mad   = median(abs(dr - d_med), axis=-1)
gap   = d_med - d_min
```

`run_r0_cram.py` 的候选已经包含：

- A0：`d_min`，即 pooled global NN，且与冻结 A1 bit-exact；
- A1：`d_min + 0.5 × max(0, d_med − d_min)`，直接使用 reference median 与 global minimum；
- A2：基于 reference MAD 的校准；
- `med_only_control`：单独的 plain median pooling。

历史 CRAM 决策是 `FAIL → ARCHIVED`：A1 median-gap 的 k2/k4 mean `ΔAP` 为 `−0.0008 / −0.0048`，两 shot 合并为 `−0.0028`；正类为 `3 / 6`，最差类为 connector `−0.0291`，mean AUROC loss `−0.0014`。A2 MAD 的两 shot mean `ΔAP` 为 `−0.0076`。shuffled-reference control 也在部分 bracket 类产生正增益，未能证明收益来自真实 reference identity。

这已经覆盖本轮“削弱单一 reference 偶然近邻”的因果假设：global minimum 是 A0，per-reference median 是 A1，gap / MAD 是附加稳定性项，且有 reference attribution controls。仅把 reference 数量改为 heldout 后的 3 张，或把每图 top 1 改成 top 2 后再取 trimmed mean，属于已归档 CRAM 的低阶 order-statistic 变体，不足以满足独立方向条件。

## 运行决定与输入边界

- 不创建 `probe_neighborhood.py`，不创建 `PROTOCOL.json` / `RESULTS.json`，因为这会把已关闭的 CRAM 重新包装成新运行。
- 不读取 `outputs/.../test/`，不使用 query label / mask 做选择，不扫描 median、trim 或 top 2 参数。
- 保留上述历史结果和失败原因，不修改 CRAM、LNDC、LLSE 或旧 A1 文件。

## 可供后续单独立项的区别机制

如果仍要研究“偶然近邻”的根因，建议改成**support-side reciprocal coverage weighting**，而不是再对 query 的 per-reference NN 距离做 order statistic：

1. 只用 memory 中的正常 reference，给每个 memory patch 计算跨其他 reference image 的 reciprocal coverage / mutual-neighbor 稳定度；该量在 support 侧一次性计算，不使用 heldout query、缺陷族或标签。
2. 查询仍使用普通 pooled 1-NN；只根据获胜 memory patch 的 support-side reciprocal 稳定度做固定的可靠性修正。所有 reference image 和 patch 都保留，选择单位仍是 patch reliability，不是删图或 patch coreset。
3. 若日后立项，应预注册 support-only 的 reciprocal 定义、tie-break 和唯一控制：uniform reliability、reference-block shuffle、duplicate-reference invariance；同时与 A1、CRAM `d_min` 做同脚本 paired 对照。

这个候选改变的是 memory patch 的跨图**可复现性 / reciprocal topology**，不是本轮已经重复的 `min / median / trimmed mean` 聚合。它尚未运行，也不应被写成已有结果或新颖算法结论；需要 parent 另行批准独立协议后再做。

