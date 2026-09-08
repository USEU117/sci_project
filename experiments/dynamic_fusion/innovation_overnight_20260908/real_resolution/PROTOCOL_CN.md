# 真实 MPDD Full 448 / Full 896 诊断协议

## 目的与边界

这是对冻结 full896 配置的真实 MPDD support-only 诊断，不是独立确认，也不与 A1 融合结果比较。只比较同一冻结 DINOv2 ViT-B/14、同一 k2 memory、同一真实 test image 的两个输入分辨率：`full44832`（448 edge，原生 32 × 32）与 `full89664`（896 edge，原生 64 × 64）。不得用 test 结果修改配置、选择类别、选择缺陷或拟合任何参数。

## 固定数据与 memory

- 6 类全部读取：`bracket_black`、`bracket_brown`、`bracket_white`、`connector`、`metal_plate`、`tubes`。
- manifest seed `0`、k2；每类两张 train/good support 原图全部建 memory，K = 2，不做 LOO。query 是 test image，绝不进入 memory。
- 真实 test 列表、`sample_ids`、448 × 448 cached masks 和 image labels 只读自现有 DINO cache：`outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k2/anomalydino_visual/<category>.npz`。脚本要求每类完整读取该 cache 的所有 sample_ids，并验证 raw image 路径存在与顺序一致。
- 预期完整集合为 458 张：bracket_black 79、bracket_brown 77、bracket_white 60、connector 44、metal_plate 97、tubes 101；normal 176、anomaly 282。任何缺行均只能写 partial，不能冒充 complete。

## 两臂计算

两个 edge 使用相同 ImageNet normalization、同一冻结 DINO 权重、单图 batch 1。full44832 输出 32 × 32 patch grid；full89664 输出 64 × 64 patch grid。两臂均用归一化特征的 memory nearest-neighbor `1 - cosine` 距离（等价于 A1 的 normalized L2 squared distance / 2），再用同一 `dists2map` 映射到 448 × 448。NN 使用有界 torch GPU block，不改变距离定义。

## 评价

- 主指标在完整 448 × 448 map 与 cached 448 × 448 mask 上计算 pixel AP、pixel AUROC；不计算 AUPRO。
- 同时计算与主 A1 `compute_metrics` 相同的固定 `stride = 8` 采样诊断（即 `[::8, ::8]`）的 pixel AP/AUROC，但不调用会额外计算 AUPRO 的函数。
- image score 固定为 map 最大值，报告 image AUROC 与 image AP。每类单独累积和计算，随后报告 6 类 category macro；不把全数据一次性拼接来代替逐类结果。
- 每个 defect type 都和该类全部 `good` normal image 组成一个固定子集，报告两臂 image/pixel 指标及 `n_normal`、`n_anomaly`。每张图片另记录 full/stride8 pixel AP、GT 面积和 image score。
- 对每个异常 mask 同时记录完整像素正数、stride8 正数和 `stride8_mask_disappears`。缺陷 mask 在 stride8 后全为 0 的计数/比例按类、defect type 和宏平均报告；这正是细小异常是否消失的诊断。

## 资源、计时与停止

- 唯一 GPU：RTX 3060 6 GB，`cuda:0`，batch 1，PyTorch threads 2；单进程。
- warmup 独立计时并排除正式 test 统计。formal 每张图分别记录两个 edge 的 encode / score latency；显存字段分开记录 torch allocated/reserved/peak 与整设备 free/total。
- 总预算 60 分钟。06:30 北京硬收尾对应 `datetime(2026, 9, 7, 22, 30, tzinfo=timezone.utc).timestamp()`；最晚 07:00 对应 `datetime(2026, 9, 7, 23, 0, tzinfo=timezone.utc).timestamp()`。每次 extraction、NN block、image episode 和 category 边界检查时间，触发后写 partial 并停止。
- OOM 只记录阶段与资源；允许同一配置的显式 CPU 退路，禁止降低分辨率或按结果挑选样本。只有 6 类完整、两臂均覆盖全部 458 张且所有每类结果存在时，`RESULTS.json` 才可标记 `complete`。

