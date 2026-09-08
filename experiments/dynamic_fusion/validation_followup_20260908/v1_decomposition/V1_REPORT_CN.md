# V1 逐 type 分解：真实 448 vs 896（DINO-only, seed0 k2, full-pixel mask）

生成：2026-09-08T00:18:45.402058+00:00（UTC）。数据源：overnight real_resolution（complete，458 图 × 2 臂，916 行）。

## 宏对账

| 口径 | full44832 | full89664 | Δ |
|---|---:|---:|---:|
| image-AUROC（category 宏） | 0.732330 | 0.754412 | +0.022082 |
| full-pixel Pixel-AP（category 宏） | 0.317636 | 0.310542 | -0.007094 |
| image-AUROC（defect-type 宏） | 0.732910 | 0.759275 | +0.026365 |

## 逐 (类, defect type)（按 image-AUROC 增益降序）

| group | n | PixAP448 | PixAP896 | dPix | imgAUC448 | imgAUC896 | dimgAUC | D6 关联 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bracket_black/hole | 12 | 0.0068 | 0.0289 | +0.0221 | 0.4141 | 0.7240 | +0.3099 | 是 |
| connector/parts_mismatch | 14 | 0.2640 | 0.2254 | -0.0386 | 0.7524 | 0.8286 | +0.0762 |  |
| bracket_white/defective_painting | 13 | 0.0091 | 0.0089 | -0.0001 | 0.7513 | 0.7949 | +0.0436 | 是 |
| tubes/anomalous | 69 | 0.6993 | 0.6877 | -0.0116 | 0.9574 | 0.9783 | +0.0208 |  |
| bracket_brown/parts_mismatch | 34 | 0.0198 | 0.0135 | -0.0063 | 0.4321 | 0.4480 | +0.0158 | 是 |
| bracket_brown/bend_and_parts_mismatch | 17 | 0.0486 | 0.0435 | -0.0051 | 0.4932 | 0.5090 | +0.0158 | 是 |
| bracket_black/scratches | 35 | 0.0478 | 0.0968 | +0.0490 | 0.4795 | 0.4875 | +0.0080 |  |
| metal_plate/major_rust | 14 | 0.5497 | 0.5709 | +0.0212 | 1.0000 | 1.0000 | +0.0000 |  |
| metal_plate/scratches | 34 | 0.8372 | 0.8709 | +0.0337 | 0.9977 | 0.9762 | -0.0215 |  |
| bracket_white/scratches | 17 | 0.1319 | 0.0404 | -0.0916 | 0.7843 | 0.7176 | -0.0667 |  |
| metal_plate/total_rust | 23 | 0.7206 | 0.7301 | +0.0095 | 1.0000 | 0.8880 | -0.1120 |  |

## 异常图内定位（per-image Pixel-AP，仅 anomaly，剔除正常图影响）

| category | within448 | within896 | Δ |
|---|---:|---:|---:|
| bracket_black | 0.2421 | 0.3553 | +0.1132 |
| bracket_brown | 0.1791 | 0.1327 | -0.0464 |
| bracket_white | 0.2610 | 0.2609 | -0.0000 |
| connector | 0.5078 | 0.4703 | -0.0375 |
| metal_plate | 0.7912 | 0.8379 | +0.0466 |
| tubes | 0.6983 | 0.7065 | +0.0082 |

## 读数要点

1. **image-AUROC 宏观增益不平均**：主要来自 bracket_black/hole（0.414→0.724，+0.31）、connector/parts_mismatch（0.752→0.829）、defective_painting（0.751→0.795）、tubes（0.957→0.978）。
2. **D6 核心缺口（bracket_brown 上下文族）基本未被 896 改善**：parts_mismatch 0.432→0.448、bend 0.493→0.509，仍在近随机（0.45–0.51）。高分辨率不构成上下文装配缺陷的机制。
3. **回归真实存在**：bracket_white/scratches image-AUROC −0.067、Pixel-AP −0.092；connector/parts_mismatch 像素 AP −0.039；bracket_brown 异常图内定位 −0.046、connector −0.038。
4. **两类定位口径分离**：pooled(category 宏) Pixel-AP 448→896 略降（0.3176→0.3105），但异常图内定位（每类 anomaly 图 per-image AP 的类均值再宏）反而升（0.4466→0.4606，bracket_black +0.113、metal_plate +0.047）；pooled 退化集中在 bracket_brown/connector/bracket_white，异常图内退化同样集中在 bracket_brown（−0.046）与 connector（−0.038）。
5. 结论：**896/细节分辨率只对稀疏小面积表面缺陷（hole、划痕类）有稳定收益，对 D6 定位的真缺口（bracket_brown 上下文族）无效，并对部分类造成回归** → 不作为新机制继续；不推翻 A1。
