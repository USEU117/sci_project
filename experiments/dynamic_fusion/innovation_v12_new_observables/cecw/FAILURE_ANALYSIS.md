# V12 CECW — 失败分析（2026-09-03）

> CECW（Cross-Encoder Conformity Work，doc 22 §4）在修正实施与全对照下 FAIL。本文分析“为什么 coupling 无信息、为什么 ANoCo 族在本资产上不优于 A1”，并给出后续纪律。

## 1. 机制层失败：跨编码器冲突耦合在数值上惰性

- `cecw ≡ ctrl_noconflict ≡ ctrl_shuffled`（任意 shot 差异 ≤ 6e-5）。冲突项 λ_c‖P_Dδ_D−P_Cδ_C‖² 对联合解没有可测影响。
- 原因：每分支 ANoCo 能量 a_e‖δ_e‖²+2b_eᵀδ_e 中 a_e = D_e + 1（D_e = 正亲和度之和，均值 5-44），耦合以 λ_c/a_e ≤ 0.2 的等效权重进入，且 P 为 r=16 支持 CCA 投影；联合解被刚度项主导 → 最优解几乎就是两个独立 ANoCo 最优。
- 推论：**冲突耦合要产生信息，要么 λ_c 必须远大于 a_e（改变协议含义），要么对齐/冲突需定义在标准化后的更新方向而非原始坐标**。这两条在 R0 前未预注册；现在后验调整违反 doc 22 §11.4，不执行。

## 2. 评分层伪影：work = 度加权能量，在背景主导类上崩坏

- bracket_white/black/brown 大面积平面背景的 patch 度 D_e 很大，`sqrt(a_e)·‖δ_e‖` 型 work 把平坦背景顶高 → 大量 FP。
- CECW worst 类达 −0.080…−0.141；而 ctrl_qq/ctrl_smoothing（对得分场做 4-邻域平滑）在所有 bracket_* 上均优于 CECW 原始 work 图 → CECW 的分不如“得分后平滑”的分。
- metal_plate/tubes 上 CECW 为正（+0.02~+0.05），但 smoothing/qq 对照同样或更高（metal_plate k1 sm=+0.059 vs cecw=+0.031；k4 sm=+0.060 vs cecw=+0.041）→ 增益来源是平滑/重标度，不是跨编码器冲突。

## 3. ANoCo 族基线全部 ≤ A1：机制在本资产上不转移

- ANoCo-DINO/CLIP/concat/fixed-mean 三 shot mean Δ 全 ≤ 0（−0.0055…−0.1109）。非符合度（bipartite Laplacian update 幅值）在 K≤4 正常参考、32×32 冻结特征、无推理增强下不优于 A1 的 1-NN 距离。
- 与已发表 ANoCo 的差异点（配置层，非机制）：其报道配置用 DINOv3-L/16、768×768 输入、25 组配对增强 + 5 参考增强、熵加权聚合、σ=0.8/7 高斯，显存 ≥2-10GB；本项目协议禁止为其引入这些资产（data role / 计算预算 / 禁止按真实 bad 调参）。
- 分类别亮点（不构成宏观主张）：metal_plate ANoCo-CLIP 三 shot +0.044/+0.056/+0.053；该类别恰好是 CLIP 纹理响应最强的类；对 tubes/connector/bracket_white 则是灾难（−0.25~−0.05）。与 MTCOA 的“无专家可跨类泛化”一致。

## 4. 对照有效性
- shuffled correspondence 掉点 ≈ 0 → 不是实现 bug 掩盖了耦合信号，而是耦合本就没有进入解的通道。
- identity/fallback：A1 harness AP 与冻结 A1 完全一致（如 bracket_black k1 0.010484），全 run fallback 比例 0.00。
- Spearman(CECW,A1)≈0.75-0.77：既非 A1 重标度（g4 未触发），也非与 A1 等价——而是与“独立双分支位移的度加权组合”等价。

## 5. 归档与后续纪律
- 归档 CECW 与 ANoCo 族（本协议内）。不再做：同构 ANoCo 超参扫、λ_c/r 后验调、把 smoothing/qq 正类结果包装成 coupling 贡献（doc 22 §12.10）。
- 保留失败资产可复用的部分：A1-distance fallback、robust01 图融合对照、4-邻域平滑对照，在后续 NTOF/PRS 的机制门中作为“简单对照必须被超过”的基线。
- 下一步按 doc 22 §10/§13：CECW 无信息 → **NTOF**（normal-only illumination holdout R0，§3.4 门）。
