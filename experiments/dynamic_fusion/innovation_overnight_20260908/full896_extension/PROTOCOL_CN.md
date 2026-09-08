# Full 896 扩类探针预定协议

## 目的

在已经完成的 `bracket_black`、`metal_plate` 之外，补测 `bracket_brown`、`bracket_white`、`connector`、`tubes` 四类。每个新 episode 同时计算两个冻结 DINO 臂：完整原图 `448 edge → 32 × 32` 的 `full44832` 控制，以及完整原图 `896 edge → 64 × 64` 的 `full896` 候选。这样 AP 差值来自同一脚本、同一输入、同一原图 mask，避免直接拿旧表的绝对值作扩类控制。

## 固定输入与 episode

- 类别：`bracket_brown`、`bracket_white`、`connector`、`tubes`。
- MPDD manifest seed `0`，k2；每次 query 只用同类另一张 support 图建 memory，query 图不进入 memory。
- 每类每张 query 生成 `thin_scratch` 与 `cutpaste`，各使用 renderer seed `0, 1, 2`。因此新增 `4 × 2 × 2 × 3 = 48` 个 unique synthetic mask；连同已完成两类的 24 个，共 72 个 mask episode。
- nuisance 只取预先固定的 5 个现有 photometric key：每个 family 的 index `1`（`exposure_1`、`gamma_1`、`white_balance_1`、`lr_brightness_gradient_1`、`specular_blob_1`）。不根据结果调整强度，不读取 test 图或 test 标签。
- 两臂都使用 renderer 返回的同一 `1024 × 1024` mask，先计算 native grid 的 nearest memory distance，再用同一 `dists2map` 投影回原图尺寸计算 pixel AP。

## memory、缓存与比较

memory 只含另一张 k2 support 图。full44832 与 full896 共用同一个已加载、冻结的 DINOv2 ViT-B/14 权重；每个 episode 使用相同的 rendered image，并分别重编码两次。clean 与 synthetic 两种分辨率的 feature 均缓存，nuisance 在线处理。

新四类结果报告 `full896 AP`、`full44832 AP` 与同 episode 的 `Δ = full896 - full44832`，同时按类别 × 族列出。六类合并只把既有两类 full896 结果与本轮四类结果合并；既有两类控制来源会标明为旧 tiling 表，不能把它解释为本轮重新测得的旧类控制。

正常项不复用旧单图 LOO p99 或 FP 阈值，只报告两臂 score distribution。`random_area_ap` 只表示 mask 正像素 prevalence，是大样本 / 随机排序参考，不是有限样本 AP 的精确期望。

## 资源与停止

- 单 GPU `cuda:0`，RTX 3060 6 GB，batch 1，PyTorch threads 2；单进程运行，预算 45 分钟。
- 先单独做 warmup，再记录正式编码时间；warmup 不计入 episode 时间。显存同时记录 torch allocated / reserved / peak 与整设备 free / total，避免混淆。
- 每次 model extraction 和 episode 前后检查硬截止 `datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()`，即北京时间 2026-09-08 07:00。到点写 partial cache / results 后停止。
- OOM 时记录发生阶段，并只允许显式使用同一 448 / 896 分辨率的 CPU 退路；不临时降低分辨率或挑选结果。

本探针用于扩类质量筛查。任何正向差值都只是候选输入分辨率 / 上下文信号，不能写成论文创新已证实；尤其不把旧两类相对 448 的轻微下降写成 cutpaste 改善。

