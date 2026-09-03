# S1-HGLC（doc 16 §3）DECISION — 2026-09-03（机器起草）

## 判定：S1-HGLC 像素级融合模块 **不立项（归档）**；图像级文本信号本身为独立正结果

按 doc 16 §3.3 门槛执行（全部 MPDD development s0/k{1,2,4}，外部集冻结未触碰）。

## 1. 做了什么

| 阶段 | 内容 | 结果 |
|---|---|---|
| Export | 冻结 AnomalyCLIP（ViT-L/14@336 @518，`9_12_4_multiscale_visa/epoch_15.pth`，sha 415c5dcb，DAPM20）导出 **CLIP global 图像嵌入**与**异常文本概率 p_abn**（zero-shot，通用 "object" 学习 prompt，类别无关）；swap 方向 sanity：swap 后恰为 1−p（0.1364↔0.8636）✓ | ✅ 6 类，~3min |
| Diag | 四路图像级对比（doc item 2）：A1-max / A1-top1% / DINO CLS / CLIP global / TEXT，Image-AP+AUROC，逐 (cat,shot) | ✅ TEXT 过门 |
| Calib | 全局门控公式 `z_A1 + β·ReLU(p_abn−0.5)·h(z_A1)`，β∈{0.1,0.25,0.5} 固定网格，h∈{z/(1+z), top-q(0.1)}，τ=0.5 先验固定；控制：打乱门控 / 直接乘法 | ❌ 差门槛 0.001 |

## 2. 图像级诊断（过门：TEXT pooled Δ=+0.0249 ≥ +0.010）

pooled Image-AP（每类 shot 均值再取类均；DINO CLS 与 P5A 归档完全一致 0.6769，交叉验证 ✓）：

| 信号 | pooled Image-AP | Δ vs A1-max | 说明 |
|---|---:|---:|---|
| A1-max（冻结 concat 448 max） | 0.7985 | 0 | 基准 |
| A1-top1% | 0.7536 | −0.0449 | 同源图像分备选 |
| DINO CLS @448（1−maxcos, k refs） | 0.6769 | −0.1216 | 与 P5A 归档完全一致（交叉验证 ✓） |
| CLIP global @518（1−maxcos, k refs） | 0.7035 | −0.0950 | 需 refs |
| **TEXT（zero-shot 异常文本概率）** | **0.8234** | **+0.0249** | 无需 refs、类别无关 |

逐类（per-cat shot 均值，详见 S1_HGLC_DIAG.json）：bracket_black +0.068（A1-max 低 shot AUROC 仅 0.43，max-pool 失效而 text 0.68）、
bracket_brown +0.063、connector +0.151（TEXT 0.767 vs A1 0.616）、tubes ≈0；负类：metal_plate −0.058（A1=1.0 饱和）、bracket_white −0.072。

## 3. 像素级校准（未达立项门槛）

- h=z/(1+z)，最优 β=0.5：pooled ΔPixel-AP **+0.0040**（< +0.005 门槛，差 0.001）；worst cat +0.0002（无类退化，满足 ≥ −0.02）
  - 控制：打乱图像级门控 −0.0005（真实门控 +0.0040 为真实但微小信号）；直接乘法 +0.0044（≈加性 z/(1+z)）
- h=top-q(0.1)，最优 β=0.1：pooled −0.0050；worst metal_plate −0.0371 → 不通过
- → doc §3.3 item 4 的 "校准后 Pixel-AP ≥ +0.005" 未满足 → **S1-HGLC 作为像素融合模块不立项（归档）**。
- 亦无法满足 doc "须先证明 S0+S1 > S0"（S0 已于 Wave2 归档）。

## 4. 结论与论文口径建议

1. **图像级跨模态文本信号是真实的、可独立主张的发现**：类别无关、zero-shot、无需 normal refs 的
   AnomalyCLIP 异常文本概率在 MPDD 图像级 AP 上超过 A1 局部记忆图 max-pool 图像分（pooled +0.0249），
   在 connector 与低 shot 配置尤其强（A1 max-pool 在这些配置反而劣于随机）。
   → 可作为论文的**图像级筛查/全局异常证据**叙述（"局部记忆 + 全局语义文本证据"异质互补），
   与"两视觉分支"无关，属真正跨模态增量。
2. **边界诚实**：文本信号不定位（对像素图门控增益仅 +0.0040 < 0.005），不得当作第三张像素图融合
   （与 doc §5.2 旧结论一致：AnomalyCLIP 文本像素图 Pixel-AP 低）。
3. S1-HGLC 完整路线按冻结门槛归档；不进入 Full MPDD / 外部集。

## 5. 证据

- 脚本：`scripts/innovation_v6_dgsafe/run_s1_hglc_export.py`（GPU，.venv-anomalyclip）、
  `run_s1_hglc_diag.py`（.venv-patchcore）、`run_s1_hglc_calib.py`（CPU）
- JSON：`experiments/dynamic_fusion/innovation_v6_dgsafe/s1_hglc/`
  `S1_HGLC_DIAG.json` · `S1_CALIB.json` · `cache/s1_hglc_export_report.json`
- 缓存 npz（gitignore，非入库）：`s1_hglc/cache/{cat}.npz`（CLIP global + text p，label-free）
- GT/标签：仅 evaluator 侧从 sample_id 推导（图像级）与 maps.gt_masks_for（校准段）；MPDD development only。
