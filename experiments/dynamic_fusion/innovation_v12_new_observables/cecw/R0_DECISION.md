# V12 CECW — 双编码器顺应功耦合决策（2026-09-03）

协议：`R0_PROTOCOL.json`（预注册：ANoCo 复现默认值 M=64 / Λ_q=1 / r=16 / λ_c=1，正亲和边，A1-distance 退化回退）
脚本：`scripts/innovation_v12_new_observables/run_r1_cecw.py`
范围：MPDD development seed0 × shot {1,2,4} × 6 类，冻结特征（DINOv2-B14 32×32；AnomalyCLIP ViT-L/14@336 对齐 32×32），Pixel-AP @ STRIDE=8 协议。
基线（published ANoCo 机制复现）：ANoCo-DINO / ANoCo-CLIP / ANoCo-A1concat / fixed-mean(r01 平均)。
方法：CECW = 双编码器 ANoCo 能量 + 跨编码器校正冲突项 λ‖P_Dδ_D−P_Cδ_C‖²（P 由 normal support CCA 对齐，r=16），得分=最小联合能量幅值 sqrt(a_D‖δ_D‖²+a_C‖δ_C‖²+λ‖冲突‖²)。

## 结果（mean ΔPixel-AP vs A1，6 类池化后取平均）

| 方法 | k1 | k2 | k4 |
|---|---:|---:|---:|
| ANoCo-DINO | −0.0433 | −0.0479 | −0.0468 |
| ANoCo-CLIP | −0.0514 | −0.0772 | −0.1109 |
| ANoCo-A1concat（最强基线） | −0.0076 | −0.0137 | −0.0055 |
| fixed-mean | −0.0111 | −0.0311 | −0.0205 |
| **CECW** | **−0.0101** | **−0.0229** | **−0.0436** |
| ctrl_shuffled | −0.0101 | −0.0229 | −0.0435 |
| ctrl_noconflict（λ=0） | −0.0101 | −0.0229 | −0.0434 |
| ctrl_qq | −0.0036 | −0.0150 | −0.0418 |
| ctrl_smoothing | −0.0028 | −0.0141 | −0.0442 |
| CECW 对最强基线增益 | −0.0025 | −0.0093 | −0.0381 |
| shuffled 掉点（正确−shuffled） | ≤6e-5 | ≤1e-4 | ≤6e-5 |
| Spearman(CECW,A1) | 0.770 | 0.769 | 0.747 |
| CECW 分类别：positive 类 | 2/6 | 2/6 | 2/6 |
| CECW worst 类 | −0.080 (bracket_white) | −0.099 (bracket_white) | −0.141 (bracket_black) |

## 门判定
- g1 headroom（mean Δ ≥ +0.006 且 ≥4/6 类为正且 worst ≥ −0.010）：三 shot 全 **FAIL**（mean Δ 全负）。
- g2 coupling 独立增益 ≥ +0.003：三 shot 全 **FAIL**（−0.0025/−0.0093/−0.0381）。
- g3 shuffled correspondence 掉点 ≥ 0.003：三 shot 全 **FAIL**（掉点 ≤ 1e-4，coupling 数值惰性）。
- g4 Spearman 重标度否决：未触发（Spearman≈0.75-0.77）；但 cecw≡noconflict≡shuffled（差异 <6e-5）证明 CECW 与“独立优化两分支位移”等价，属位移/权重变换而非耦合信息。

## 结论：CECW **ARCHIVED (FAIL → coupling 无信息，doc 22 §13 立即归档)**
按 doc 22 §13：「若 coupling 对 strongest baseline 独立增益不足 0.003 或 shuffled correspondence 不掉点，立即归档。」
- **不再训练 router；不做任何 ANoCo 族同构参数搜索**（doc 22 §12 禁止）。
- 已发表 ANoCo 机制在本项目冻结特征上（K≤4、32×32 grid）不能胜过 A1 的 1-NN 定位；其强配置需要 DINOv3-L/768 输入/推理增强/10GB 显存，不在本资产范围内。
- 唯一保留的类别级观察（不构成宏观主张）：metal_plate 上 ANoCo-CLIP 与 CECW 三 shot 均为正（+0.03~+0.05），tubes 上 CECW 为正（+0.04~+0.05）；但这些增益被 smoothing/qq 对照复制（metal_plate k1 sm=+0.059>cecw=+0.031），说明其来源是得分场平滑/度加权，而非跨编码器冲突耦合。
- 下一步：按 doc 22 §10/§13 与 v12 ledger，CECW 无信息 → 启动 **NTOF**（P1 第一新主线，normal-only 光照切空间 R0 门）。
