# 路线 D 决策门 D0（A1 之上的动态 headroom）

RunId: `route_d_d0_20260818` · MPDD development · 9 配置（3 seeds × 1/2/4-shot）

- Oracle 定义：逐像素 best-of-branch（concat / feature-DINO-only / CLIP-only），用 GT 在每像素选更优分支。**仅开发诊断，不进入任何融合/路由输入。**
- mean Pixel AP headroom vs A1 = **+0.5807**（要求 ≥ +0.015）
- 逐类 headroom：{'bracket_black': 0.7134, 'bracket_brown': 0.8181, 'bracket_white': 0.9054, 'connector': 0.6676, 'metal_plate': 0.1194, 'tubes': 0.2604}
- ≥ +0.005 的类数：**6/6**（要求 ≥ 4）
- headroom 最大类占比：0.26（要求 < 0.5）
- **Gate D0：PASS → 进入 D1**

## 说明

- Oracle 上限高并不代表无标签可靠性特征能预测『何时修正 A1』；D0 只是第一道检查。
- 未计算『V3.3-clean 逐区域 Oracle』（per-pixel best-of-3 已覆盖 D0 判定所需的主要 headroom）。
