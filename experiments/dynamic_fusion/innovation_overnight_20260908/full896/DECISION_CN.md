# Full 896 原生 64 × 64 图像域探针决策

状态：`complete`；决策：`CANDIDATE_NATIVE64_CONTEXT_RESOLUTION_SIGNAL; support-synthetic probe only`。

本 probe 使用同一 MPDD seed 0、k2 support、同一 24 个 synthetic masks、同一 15 项 nuisance 和留一图 memory。full896 是完整 1024 × 1024 图像以 896 edge 通过冻结 DINO 得到的原生 64 × 64；full44832 与 tile64 的 AP 只读自已完成的 tiling 结果。

`random_area_ap` 记录的是 mask 正像素比例，作为大样本 / 随机排序 prevalence 参考；有限样本 AP 非线性，因此不是有限样本 AP 精确期望。正常项不沿用旧单图 LOO p99，只报 score distribution。

full896 与四 crop 的总 token 数都为 4096，但 full-image attention-work proxy 约为四 crop 总和的 4 倍；结果不能解读为等算力比较。

每类每族 full896 delta（完整数值见 SUMMARY.json）：
{
 "bracket_black": {
  "thin_scratch": {
   "n": 6,
   "mean_ap": 0.3137833812558584,
   "mean_delta_vs_full_dino32": 0.12058344201251014,
   "mean_delta_vs_tile_reencode64": -0.01415256655114712
  },
  "cutpaste": {
   "n": 6,
   "mean_ap": 0.7591351570991246,
   "mean_delta_vs_full_dino32": -0.021168153773551263,
   "mean_delta_vs_tile_reencode64": 0.23655104017366865
  }
 },
 "metal_plate": {
  "thin_scratch": {
   "n": 6,
   "mean_ap": 0.36360191602381403,
   "mean_delta_vs_full_dino32": 0.12901922472510316,
   "mean_delta_vs_tile_reencode64": 0.005444876176868054
  },
  "cutpaste": {
   "n": 6,
   "mean_ap": 0.8084639032836641,
   "mean_delta_vs_full_dino32": -0.012456146308381247,
   "mean_delta_vs_tile_reencode64": 0.1049756212176674
  }
 }
}

实际墙钟时间：`177.342 s`。clean / synthetic 原生 feature cache 见 `outputs/dynamic_fusion/overnight_20260908_full896/features/`；逐 episode 结果见 `RESULTS.json` 和 `PER_EPISODE.jsonl`。

边界：本轮只有 2 类、k2、support synthetic episodes；任意正差值都只是候选输入分辨率 / 上下文信号，不能写成论文创新已证实。
