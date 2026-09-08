# 真实 MPDD Full 448 / Full 896 诊断结论

状态：`complete`；决策：`REAL_DIAGNOSTIC_COMPLETE; DINO_RESOLUTION_ONLY; no A1 fusion comparison`。

本轮完整读取 6 类 seed 0、k2 的现有 cached sample_ids 与 448 × 448 masks；每类两张 normal support 全部建 K=2 memory，不做 LOO。两臂使用同一冻结 DINOv2 权重、距离和 dists2map，只有输入 edge 不同。

full 448 × 448 pixel AP/AUROC 是主指标；stride8 仅为 A1-compatible 诊断，未计算 AUPRO。image score 固定为 448 × 448 map 的 max。逐类、defect type 与 macro 结果完整保存在 SUMMARY.json。

category macro：
{
 "n_categories": 6,
 "arms": {
  "full44832": {
   "full_pixel": {
    "pixel_ap": 0.31763577025731665,
    "pixel_auroc": 0.9509959161945295
   },
   "stride8": {
    "pixel_ap": 0.3156112986711508,
    "pixel_auroc": 0.9524145993048658
   },
   "image": {
    "image_ap": 0.7433021666974103,
    "image_auroc": 0.7323299517039693
   }
  },
  "full89664": {
   "full_pixel": {
    "pixel_ap": 0.3105420951062044,
    "pixel_auroc": 0.9536722697430443
   },
   "stride8": {
    "pixel_ap": 0.3116133558010428,
    "pixel_auroc": 0.9542621029350012
   },
   "image": {
    "image_ap": 0.7522450719296558,
    "image_auroc": 0.7544118170514262
   }
  }
 },
 "fine_anomaly": {
  "mean_stride8_zero_mask_rate": 0.03333333333333333,
  "mean_stride8_positive_fraction_of_full": 0.01512247409567786,
  "total_anomaly_images": 282,
  "total_stride8_zero_mask_count": 6
 }
}

stride8 细小异常消失计数/比例在每个 category、defect type 和 per-image row 中记录；任何 stride8 指标无效都保留为 null，不用 full-resolution 标签补值。

实际墙钟时间：`694.111 s`。逐图结果见 `outputs/dynamic_fusion/overnight_20260908_real_resolution/PER_IMAGE.jsonl`；partial 也保留。

边界：这是真实 MPDD 的 DINO resolution-only 诊断，不是独立确认；不与 A1 fusion 比较，不用 test 结果改配置，也不据此宣称论文创新已证实。
