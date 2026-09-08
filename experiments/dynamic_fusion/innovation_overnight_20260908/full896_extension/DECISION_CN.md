# Full 896 扩类探针决策

状态：`complete`；决策：`CANDIDATE_NATIVE64_EXTENSION_SIGNAL; support-synthetic probe only`。

本轮只新增 bracket_brown、bracket_white、connector、tubes。每个 episode 同时计算 full44832 与 full896，使用同一 rendered image、同一 k2 留图 memory、同一 1024 × 1024 原图 mask；没有 tile 臂，也没有重跑已完成两类。

新增四类的 synthetic unique mask 计数为 48；六类合并把既有两类 full896 结果与本轮四类结果合并，共 72 个 mask。六类合并中的旧两类 full44832 数值来自既有 tiling 记录，仅用于上下文；新四类的 delta 均来自本脚本同 episode 两臂。

random_area_ap 是 mask 正像素 prevalence 的大样本 / 随机排序参考，不是有限样本 AP 的精确期望。正常项只报告分数分布，不复用旧单图 LOO p99 或 FP 阈值。

新四类逐类别逐族 `full896 - full44832`（完整数值见 SUMMARY.json）：
{
 "bracket_brown": {
  "thin_scratch": {
   "n": 6,
   "mean_ap_448": 0.16163737903212957,
   "mean_ap_896": 0.29781464933292895,
   "mean_delta_896_minus_448": 0.13617727030079932
  },
  "cutpaste": {
   "n": 6,
   "mean_ap_448": 0.8394978772222664,
   "mean_ap_896": 0.831907759256683,
   "mean_delta_896_minus_448": -0.007590117965583343
  }
 },
 "bracket_white": {
  "thin_scratch": {
   "n": 6,
   "mean_ap_448": 0.1659677723768054,
   "mean_ap_896": 0.25660405000807174,
   "mean_delta_896_minus_448": 0.09063627763126636
  },
  "cutpaste": {
   "n": 6,
   "mean_ap_448": 0.6253498875775844,
   "mean_ap_896": 0.6024920038304372,
   "mean_delta_896_minus_448": -0.022857883747147017
  }
 },
 "connector": {
  "thin_scratch": {
   "n": 6,
   "mean_ap_448": 0.19553531202480912,
   "mean_ap_896": 0.43163271770591866,
   "mean_delta_896_minus_448": 0.2360974056811096
  },
  "cutpaste": {
   "n": 6,
   "mean_ap_448": 0.7018887254886507,
   "mean_ap_896": 0.6935755270650911,
   "mean_delta_896_minus_448": -0.008313198423559542
  }
 },
 "tubes": {
  "thin_scratch": {
   "n": 6,
   "mean_ap_448": 0.16810085102721048,
   "mean_ap_896": 0.3489185897131832,
   "mean_delta_896_minus_448": 0.18081773868597262
  },
  "cutpaste": {
   "n": 6,
   "mean_ap_448": 0.6701334795548979,
   "mean_ap_896": 0.7151099438259855,
   "mean_delta_896_minus_448": 0.04497646427108759
  }
 }
}

六类合并摘要（完整数值见 SUMMARY.json）：
{
 "ap_448": 0.46309310627592776,
 "ap_896": 0.5352532998667301,
 "delta": 0.07216019359080221
}

实际墙钟时间：`334.437 s`。clean / synthetic 双分辨率 feature cache 见 `outputs/dynamic_fusion/overnight_20260908_full896_extension/features/`；逐 episode 结果见 `RESULTS.json` 和 `PER_EPISODE.jsonl`。

边界：这是四类、k2、support-synthetic 短探针。任何正向差值都只能作为候选输入分辨率 / 上下文信号；不要把旧两类 cutpaste 相对 448 的轻微下降写成改善，也不能据此宣称论文创新已证实。
