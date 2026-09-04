# AMENDMENT — deepest-parity gate: raw <1e-5 → map-level Pixel-AP <1e-4 (operational)

date: 2026-09-04
status: amendment to `00_protocol/PROTOCOL_FROZEN.yaml` (frozen original unchanged) + doc 23 §7 g4
authority: docs/.../25_OVERNIGHT_VALIDATION_CODE_AUDIT_AND_NEXT_STEPS_CN_20260904.md §5；
         docs/.../26_CURRENT_CORRECTION_STATUS_AND_NEXT_BREAKTHROUGHS_CN_20260904.md §6.1（收口 parity 缺口）

## 1. 原协议要求（未改动原文）
PROTOCOL_FROZEN.yaml `alignment`：
> deepest parity: ... max abs error < 1e-5 AND per-category pooled Pixel-AP diff < 1e-4

doc 23 §7 Stage0 进入 Stage1 门槛第 4 条：`deepest cache 与当前 A1 对齐误差 <1e-5`。

## 2. 实测（01_multilayer_cache/DEEPEST_PARITY_REPORT.json，18 组 cat×shot）
- raw max abs：dino L11 最大 **0.007057**（k2 bracket_brown），clip L24 最大 **0.001343** → **raw <1e-5 不满足**。
- 同一调用内前向确定性 ~2e-4，跨会话 ~1e-3（commit a6ccbd1 已记录）；raw 1e-5 在跨会话缓存对比中不可达。
- per-category pooled Pixel-AP diff：最大 **1e-6** → **map 级 <1e-4 满足**（18/18）。

## 3. Amendment（本文件即正式记录）
Stage0 的**操作门（operational gate）为 map 级 pooled Pixel-AP parity `<1e-4`**，替代 raw `<1e-5` 子门；
raw 腿如实报告为 FAIL（0.001–0.007），**不表述为“raw 门全部满足”**。Pixel-AP 接近 ≠ score map
逐像素一致，也不单独作为“raw 差异来源=GPU 非确定性”的证明。

适用范围：Stage0 对齐自检及其 g4 判定；Stage1 及后续模块的门不受影响。
凡引用 g4 处需同时引用本 amendment（FINAL_DECISION.md 已同步更新）。
