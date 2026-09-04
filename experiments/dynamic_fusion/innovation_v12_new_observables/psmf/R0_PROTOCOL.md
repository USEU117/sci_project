# Doc 22 §7 PSMF R0 协议（预注册，2026-09-04）

全名：Phase-Stable Micro-defect Field。核心：对同一图像做 <patch 的确定性相位平移
（4 相位 (0,0),(0,4),(4,0),(4,4)，448/518 输入上 4px ≈ 0.29 token），逐相位按 A1 配方出图，
反变换回图像坐标后做**逐像素 median**。真实微缺陷固定于图像坐标 → median 保留；
随 token-grid 的 alias 响应在 4 相位下漂移 → median 抑制。

## CPU 预检（已完成，PSMF_PRE_CHECK.json）

缺陷组件面积（GT mask 448 连通域）：
- micro 占比(≤196px=1 token cell)：bracket_white 0.960 / bracket_black 0.660 / bracket_brown 0.543；connector/metal_plate/tubes 0（全为大斑块）
- smallest-25% 面积上限：bracket_black ≤43px、bracket_brown ≤85px、bracket_white ≤2px

→ R0 主类 = **bracket_black, bracket_brown, bracket_white**（微缺陷集）；**metal_plate** 仅作
大缺陷特异性对照（报告 worst，不进 macro）。shot k1。

## 配方（与 a1-arm/CL-RPF 同一积分：7 层 normal-only z 等权均值 → dists2map(448)）
- 相位 p：ref 与 query 同相位重提取（dino@448 3 层 L6/9/11，clip@518 4 层 L6/12/18/24→bilinear 32）
- 逐相位逐层 bank/LOO 支持统计（K=1，32 网格半径 1）→ 逐相位 32-grid z-mean → dists2map 448
- 相位图反变换到图像坐标：np.roll(-dy,-dx)（边界伪差对候选/对照共享，机制门看差值）
- 合并算子：**psmf = 4 相位 median**（主候选）；对照 overlap-average = 4 相位 mean（相同 forward 数）
- 单相位 p=(0,0) 图与既有 a1-arm 逐类 pooled AP 应一致（身份校验）

## 门（macro/whole-cat 用标准 pooled Pixel-AP@56；bin 在 448 全分辨率——smallest-25% 多为 1-85px，
56 网格(8px)采样会让目标缺陷本身不可见，故 bin 门按 doc22 §7 原意用 Pixel-AP 于 448）
| 门 | 定义 | 门限 |
|---|---|---|
| G1 身份 | p=(0,0) pooled AP@56 − a1-arm 逐类值 | ±0.003 内 |
| G2 micro bin | psmf − a1：3 微类 smallest-25% 组件像素集 pooled AP@448 | ≥ +0.015 |
| G3 全体 | psmf − a1：3 微类 macro(pooled@56)；worst 类（含 metal_plate） | ≥ +0.004；≥ −0.010 |
| G4 同算力 | psmf − overlap-average（3 微类 macro） | ≥ +0.004 |
| G5 相位乱序 | psmf − phase-shuffle（每图随机错配相位再 median） | ≥ +0.003 |
| G6 特异性 | metal_plate：psmf − a1 | 报告；|Δ| ≤ 0.010（不伤害），且 ≤ +0.004（不偷涨） |

未过即停止。脚本 `run_r3_psmf_probe.py`；输出 `psmf/PSMF_R0_RESULT.json` + `PSMF_R0_DECISION.md`。
