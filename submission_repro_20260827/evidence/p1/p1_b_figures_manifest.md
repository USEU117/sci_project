# P1-B/R4 定性图 manifest（A1 成功/失败案例）

- Gate R4：从 compact concat/DINO maps + 合法本地原图/GT mask 生成固定成功与失败案例图。
- 选择规则：失败 = P1-B top-1 失败样例（seed0/shot1）；成功 = 该类别异常测试图中 per-image ΔAP 最大者。
- 指标口径：stride=8 Pixel-AP，冻结 evaluator；**不进行参数选择**。
- 原图与 GT mask 不打包（不可再分发）；图中含原图，仅本地 `outputs/p1_b_figures/` 保留。

| dataset | s/k | category | role | sample_id | concat P-AP | dino P-AP | ΔAP | figure SHA256 |
|---|---|---|---|---|---|---:|---:|---:|---|
| mpdd | s0/k1 | metal_plate | success | `metal_plate/test/scratches/026.png` | 0.6585 | 0.4395 | 0.219 | `d8f0362be54e2ff8…` |
| mpdd | s0/k1 | bracket_brown | failure | `bracket_brown/test/parts_mismatch/002.png` | 0.0459 | 0.2436 | -0.1977 | `6cf55df3ec33d1c5…` |
| btad | s0/k1 | 01 | success | `01/test/ko/0026.bmp` | 0.7252 | 0.5789 | 0.1464 | `7a83e5c09cedc300…` |
| visa | s0/k1 | cashew | success | `cashew/Data/Images/Anomaly/085.JPG` | 0.6758 | 0.0734 | 0.6025 | `81be264f79c04d32…` |
| visa | s0/k1 | chewinggum | failure | `chewinggum/Data/Images/Anomaly/037.JPG` | 0.1124 | 0.2769 | -0.1645 | `15a62f1cef6a58c0…` |
| mvtec | s0/k1 | toothbrush | success | `toothbrush/test/defective/004.png` | 0.7444 | 0.4217 | 0.3227 | `8b2ff3f51b62e77b…` |
| mvtec | s0/k1 | leather | failure | `leather/test/poke/016.png` | 0.2345 | 0.9294 | -0.6948 | `6d27560034b0f3b2…` |