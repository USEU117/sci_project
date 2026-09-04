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
