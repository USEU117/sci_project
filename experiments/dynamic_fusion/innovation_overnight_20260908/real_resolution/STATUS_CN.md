# 真实 MPDD Full 448 / Full 896 运行归档

## 覆盖与运行

- 状态：`complete`；执行会话 ID：`51856`（进程已正常退出）。命令见 [`COMMAND.txt`](COMMAND.txt)。
- 全部读取现有 DINO k2 cache 的 458 个 sample_ids 与 448 × 448 masks：bracket_black 79、bracket_brown 77、bracket_white 60、connector 44、metal_plate 97、tubes 101；其中 normal 176、anomaly 282。
- 每类使用 manifest seed 0 的两张 train/good support 建 K = 2 memory，不做 LOO；query 不进入 memory。916 行逐图记录（458 图 × 2 臂），无 test 反选、无参数拟合、无 AUPRO。
- full44832 与 full89664 使用同一冻结 DINOv2 ViT-B/14、同一 normalized NN `1 - cosine`、同一 `dists2map` 到 448 × 448；只改变输入 edge。总墙钟 `694.111 s`，无 OOM。
- 质量 spot check：`bracket_black` 的 fresh full44832 map 与现有 A1 DINO cache 按同一距离 / map 重算结果一致到数值误差（AP `0.04359639` vs `0.04359606`，AUROC `0.94426233` vs `0.94426231`）。

## 6 类 category macro

主指标为完整 448 × 448 pixel map；stride8 为 A1-compatible `[::8, ::8]` 诊断。

| 臂 | full pixel AP | full pixel AUROC | stride8 AP | stride8 AUROC | image AP | image AUROC |
|---|---:|---:|---:|---:|---:|---:|
| full44832 | 0.317636 | 0.950996 | 0.315611 | 0.952415 | 0.743302 | 0.732330 |
| full89664 | 0.310542 | 0.953672 | 0.311613 | 0.954262 | 0.752245 | 0.754412 |

逐类别完整结果如下；normal 与 anomaly 都在每类分母中。

| 类别 | n (N/A) | full448 AP / image AUROC | full896 AP / image AUROC | stride8 消失 |
|---|---:|---:|---:|---:|
| bracket_black | 79 (32/47) | 0.043596 / 0.462766 | 0.091897 / 0.547872 | 0 / 47 |
| bracket_brown | 77 (26/51) | 0.038210 / 0.452489 | 0.033007 / 0.468326 | 0 / 51 |
| bracket_white | 60 (30/30) | 0.110983 / 0.770000 | 0.038308 / 0.751111 | 6 / 30 |
| connector | 44 (30/14) | 0.263974 / 0.752381 | 0.225372 / 0.828571 | 0 / 14 |
| metal_plate | 97 (26/71) | 0.749756 / 0.998917 | 0.786952 / 0.952329 | 0 / 71 |
| tubes | 101 (32/69) | 0.699297 / 0.957428 | 0.687717 / 0.978261 | 0 / 69 |

## 缺陷类型与细小异常

每个 defect type 都与本类全部 `good` normal image 配对；full pixel AP 和 image AUROC 的两臂细项见 [`SUMMARY.json`](SUMMARY.json)。stride8 后仅 `bracket_white` 有异常 mask 完全消失：`6 / 30 = 20%`，其中 `defective_painting` 为 `4 / 13`、`scratches` 为 `2 / 17`。全 6 类合计为 `6 / 282 = 2.13%`；所有异常 mask 的 stride8 正像素平均仅为完整 mask 的 `1.512%`。

每张图均记录 full / stride8 pixel AP、GT 面积、image score 与 `stride8_mask_disappears`，见 [`PER_IMAGE.jsonl`](../../../outputs/dynamic_fusion/overnight_20260908_real_resolution/PER_IMAGE.jsonl)。

## 时间与显存

- warmup（不计入正式 test）：448 edge `375.476 ms`，896 edge `382.552 ms`；model load `2943.644 ms`。
- 正式平均编码：full44832 `78.137 ms`、full89664 `441.133 ms`；平均 score：`19.228 / 59.081 ms`。
- 正式峰值 torch allocated `503.212 MB`、reserved `631.243 MB`；整设备 free `4699.718 MB`、total `6441.927 MB`。
- 06:30 北京 early-stop 与 07:00 hard deadline 均按显式 UTC datetime 检查；本轮在此前完整结束。

## 结论边界

真实 MPDD 诊断只支持记录 DINO resolution-only 的逐类行为：full896 的 image AUROC macro 上升，但 full pixel AP macro 略低，且 `bracket_white` 的细小 mask 在 stride8 明显消失。它不是独立确认，不与 A1 fusion 比较，也不能据此宣称论文创新已证实。
