# Doc 26 §4.3 Detail-Recovery R0 决策（前提门，2026-09-04）

协议：`R0_PROTOCOL.md`（预注册；P1/P2/P3；两小类 bracket_black + metal_plate，shot k1）
恢复算子：**AnyUp multi-backbone**（ICLR'26 Oral，冻结，encoder-agnostic，保留 768-d）
脚本：`scripts/innovation_v12_new_observables/run_r3_ef_recovery_probe.py`
结果：`detail_recovery/RECOVERY_RESULTS.json`
指标：pooled Pixel-AP@56（逐层 normal-only z-map 7 层等权均值 → dists2map(448,σ4)[::8,::8]）
同一配方下 a1 臂 macro（6 类）= 0.309856，与归档 CL-RPF static mean_std 精确一致（配方身份校验通过）。

## 结果（pooled Pixel-AP@56，逐类）

| 类 | a1 (32 匹配后) | bl56 (bilinear 恢复) | au56 (AnyUp 恢复) | au56_w (错配 guide) |
|---|---:|---:|---:|---:|
| bracket_black | 0.005139 | 0.005512 | 0.006493 | 0.005755 |
| metal_plate | 0.961640 | 0.958750 | 0.975835 | 0.941684 |
| **macro(2类)** | 0.483389 | 0.482131 | **0.491164** | 0.473720 |

## 门判定

| 门 | 差值 | 门限 | 结果 |
|---|---|---:|---:|
| P1 前提（匹配前恢复保信息） | au56 − a1 = **+0.0078** | ≥ +0.003 | **PASS** |
| P2 归因（真恢复 ≠ 廉价插值） | au56 − bl56 = **+0.0090** | ≥ +0.003 | **PASS** |
| P3 guide（错配引导应掉点） | au56 − au56_w = **+0.0174** | ≥ +0.003 | **PASS** |

## 结论与下一步

- **前提成立**：把双分支先恢复到共同 56 网格再做 KNN（au56）超过"先匹配后上采样"的 A1 配方
  （macro +0.0078）；且超过同网格 bilinear（+0.0090，metal_plate 尤甚：au56 0.9758 vs bl56 0.9588）
  → 增益来自真实预训练恢复的质量，不是单纯更细网格；错配 RGB guide 掉 −0.0174 → guide 信息真实生效。
- **诚实的边界（不构成追改）**：
  1. 增益由 metal_plate 主导（+0.0142 vs a1）；bracket_black 仅 +0.0014——该类在本单-ref 静态
     配方下基线 0.005 近地板（本就无法定位），恢复挽救不了配方级失效。small-defect 子组门
     （doc26 §4.3 ≥+0.01、FP ≤+5%）必须等到 6 类/真 micro-defect 分类后才算数。
  2. 仅 2 类、shot k1；是前提小门，不是完整机制门。
- 下一步（doc 26 §4.3 主菜）：设计**分支条件化耦合恢复**（共同坐标下双分支互约束，相对
  AnyUp-generic 独立增益 ≥0.003、small-defect ≥0.01、错配分支掉点），即控制 #1–#7 全谱比较。
  此 R0 未声称"耦合创新"，只把"信息在匹配前丢失 + 真恢复可回收"设为已支持的可行前提。
- 基础设施留档：AnyUp ckpt `outputs/external_weights/anyup_multi_backbone.pth`
  (sha256 B6CC407D…，github release 下载，断点续传)；venv torch2.0 需 RMSNorm 等价 shim（内嵌于探针脚本）。

---

## 2026-09-04 v3 更新：6 类扩展 + 耦合诊断 → 本路线归档负

（两小类前提 PASS 后按用户指令扩 6 类验证普适性，再按 doc26 §4.3 在坍缩类上做耦合判定性检验；
协议见 R0_PROTOCOL v2/v3；结果 `RECOVERY_RESULTS.json` + `COUPLED_RESULTS.json`）

### 6 类前提（pooled Pixel-AP@56；au56 独立 AnyUp 恢复先于匹配）

| 类 | a1 | bl56 | au56 | au56−a1 |
|---|---:|---:|---:|---:|
| bracket_black | 0.005139 | 0.005512 | 0.006493 | +0.001354 |
| bracket_brown | 0.034618 | 0.035519 | 0.029657 | −0.004961 |
| bracket_white | 0.038005 | 0.089404 | 0.002868 | −0.035137 |
| connector | 0.162589 | 0.149110 | 0.131700 | −0.030889 |
| metal_plate | 0.961640 | 0.958750 | 0.975835 | +0.014195 |
| tubes | 0.657144 | 0.673063 | 0.518700 | −0.138444 |
| **macro(6)** | 0.309856 | 0.318560 | 0.277542 | **−0.032314** |

P1/P2 macro-6 全 FAIL。au56 在 4/6 类上 ≤ a1，bracket_white(0.003 vs bl56 0.089) 与
tubes(0.519 vs a1 0.657) 严重坍缩；仅 metal_plate 显著为正。**前提不普适。**

### 耦合诊断（坍缩类 bracket_white + tubes；du56 跨分支条件恢复 vs cu56 concat 对照 #6）

| 类 | au56 | du56 | cu56 | FP95(du56/cu56) |
|---|---:|---:|---:|---:|
| bracket_white | 0.002868 | 0.002437 | 0.003184 | 48.5 / 32.2 |
| tubes | 0.518700 | 0.514773 | 0.554982 | 31.2 / 20.7 |
| **macro(2)** | 0.260784 | 0.258605 | 0.279083 | — |

- M1c（du56−au56 ≥+0.003 救回坍缩）：−0.0022 → **FAIL**
- M2（du56−cu56 ≥+0.003，条件化结构≠concat 可用）：−0.0205 → **FAIL**（du56 连整段 concat 同算子都输）
- FP95：du56 > cu56（normal 图高尾更大）→ 条件化不降 FP
- M3（du56_m 错配）未跑：M1c/M2 已败，条件化无可用增益，错配对照无信息（停止规则）

### 判定与结论

- **恢复先于匹配前提不普适**；**doc26 §4.3 的"跨分支互约束抑制虚假/坍缩恢复"机制在该实现下无证据**
  ——坍缩是 AnyUp 学到先验与 MPDD 中深层层级特征（DINOv2-B/14、AnomalyCLIP-L/14 @448/518）
  不匹配的固有现象（bracket_white/tubes 上任何形式的 AnyUp-56 恢复都坍缩，bl56/a1 安全因其无学习先验）。
- 按 R0_PROTOCOL v3 停止规则：**Detail-Recovery（本形态）路线正式归档负**，
  不做 6 类耦合正式门。ledger 更新；脚本/结果/协议留档供复现。
- 教训（记入后续路线）：冻结通用 upsampler 在本资产上只在部分类有益、多数类有害；
  "先恢复再匹配"若要用，必须以**无学习先验或类别内适配**为前置，且不能依赖单一预训练算子。

