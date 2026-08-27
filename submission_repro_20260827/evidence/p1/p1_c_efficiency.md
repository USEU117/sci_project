# P1-C 效率表（A1 / feature-DINO-only / CLIP-only）

- 训练参数：**0**（全部预训练 backbone 冻结；推理时仅构建正常参考记忆库，无训练）
- 单图特征提取墙钟时间（P0-2 smoke 实测，含一次性模型加载，非吞吐）：DINO 10.225 s、CLIP 9.042 s
- 峰值显存（P0-2 smoke 实测）：DINO 374.58 MB、CLIP 2072.81 MB；A1 全流程按两个分支顺序执行，峰值取 max ≈ 2073 MB
- 峰值 RAM：未在 smoke 单独测量
- compact 复现包大小：**186.5 MB**（含 324 个逐图 float16 patch maps 与全部证据）

记忆库（normal memory bank）float32 规模（ref patch 特征，按 dataset×shot 对 3 seeds 取均值）：

| dataset | shot | dino bank (patches) | clip bank (patches) | concat bank (patches) | dino MB | clip MB | concat MB |
|---|---|---:|---:|---:|---:|---:|---:|
| mpdd | 1 | 6 | 6 | 12 | 0.02 | 0.02 | 0.07 |
| mpdd | 2 | 12 | 12 | 24 | 0.04 | 0.04 | 0.15 |
| mpdd | 4 | 24 | 24 | 48 | 0.07 | 0.07 | 0.29 |
| btad | 1 | 3 | 3 | 6 | 0.01 | 0.01 | 0.04 |
| btad | 2 | 6 | 6 | 12 | 0.02 | 0.02 | 0.07 |
| btad | 4 | 12 | 12 | 24 | 0.04 | 0.04 | 0.15 |
| visa | 1 | 12 | 12 | 24 | 0.04 | 0.04 | 0.15 |
| visa | 2 | 24 | 24 | 48 | 0.07 | 0.07 | 0.29 |
| visa | 4 | 48 | 48 | 96 | 0.15 | 0.15 | 0.59 |
| mvtec | 1 | 15 | 15 | 30 | 0.05 | 0.05 | 0.18 |
| mvtec | 2 | 30 | 30 | 60 | 0.09 | 0.09 | 0.37 |
| mvtec | 4 | 60 | 60 | 120 | 0.18 | 0.18 | 0.74 |

注：bank = 正常参考 patch 特征（不含测试特征）；维度 DINO 768、CLIP image-tower 768、concat 1536。