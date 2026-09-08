# 2 × 2 原图裁块重编码探针协议

## 固定范围

- 类别：`bracket_black`、`metal_plate`。
- support：MPDD manifest seed 0、k2；每次 query 只使用同类另一张 support 图作为 memory。
- 读取范围：只读 `train/good` support 原图；不读取 test 图或 test 标签。
- synthetic family 顺序：`thin_scratch`，随后 `cutpaste`；每族预定 seed `0, 1, 2`，seed 使用既有 v14 `_variant_seed(category, support_rel, family, seed)`。
- photometric nuisance：既有 renderer 的 5 族 × 3 固定强度（`R.REF_KEYS` 全部 15 项）。

## 输入机制与对照

使用既有 v14 `export_p1_support_variants.make_extractor("dino", device)`，DINO backbone 冻结，batch 1，torch CPU threads 2，不下载新模型。原图为 1024 × 1024：

1. `full_dino32`：整图 448 edge 重编码，得到 32 × 32，整图 KNN 后按既有 `dists2map` 映射到 1024。
2. `full_map_interp`：同一整图 32 × 32 KNN 距离图，直接线性插值到 1024；不插值 feature。
3. `full_feat_interp64`：整图 32 × 32 feature 双线性插值到 64 × 64 后 KNN，作为 feature-only 插值控制。
4. `tile_reencode64`：原图切成四个不重叠的 512 × 512 crop，各自独立重编码为 32 × 32，再按空间位置拼成 64 × 64；这是本 probe 的输入机制。

## 评价与证据纪律

- 每个 synthetic episode 都使用 renderer 返回的同一张 1024 × 1024 mask，四臂使用同一 mask 和像素口径计算 AP。
- `random_area_ap` 是 mask 正像素比例，作为大样本/随机排序参考；有限样本 AP 是非线性的，因此不把它称作有限样本 AP 的精确期望，也不是拟合得到的分数。
- 正常 FP 阈值为当前留出图 memory 的 LOO 距离图 p99。clean 与 nuisance 统计全图高于阈值的比例；synthetic 只统计 mask 外正常背景。
- 每行保存 `query_rel`、`bank_rel`、memory cell 数、阈值、AP、随机面积基线、FP 范围和实际提取/评分时间，证明 query 图未进入 memory。
- 结果只有 2 类、k2、support synthetic episodes；正差值仅作候选信号，不能写成论文创新已证实。

## 硬截止与复现

脚本在每次整图/裁块提取和 episode 循环前后检查：

```python
datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()
```

到达截止时间后写 `PARTIAL_RESULTS.json` 并停止新增计算。完整运行命令：

```powershell
.venv-anomalyclip\Scripts\python.exe scripts\innovation_overnight_20260908\probe_tiling.py --device cuda:0 --torch-threads 2
```
