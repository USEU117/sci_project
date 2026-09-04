# v12 New-Observables Ledger (doc 22, 2026-09-03)

Authority: `docs/paper_writing_preparation_20260830/22_POST_V11_RESULT_AUDIT_AND_NEW_INNOVATION_ROUTES_CN_20260903.md`.
Status: `NOT_STARTED / RUNNING / PASS / FAIL / ARCHIVED`.
Data roles (doc 22 s11.1): MPDD = exhausted development (R0/candidates only); BTAD/MVTec
consumed; VisA source; Real-IAD/MVTec AD 2 untouched (NOT downloaded).

| 路线 | 状态 | 关键数字 | 门 | 下一步 |
|---|---|---|---|---|
| MTCOA corrected Oracle audit (s0 k1/2/4) | **ARCHIVED** (FAIL → RSR 永久关闭) | mean Δ: k1 +0.0239 / k2 +0.0040 / k4 −0.0129；metal_plate/tubes 各 shot 均 −0.07~−0.13；component-macro: text 32-35% & LLSE 24%（三 shot 一致）；k4 bracket_white 占正 headroom 77% | g2 PASS 三 shot（真实小缺陷互补）；g1/g3/g4 在 k2/k4 FAIL | RSR 永久关闭；类别条件互补留档 → 转向 doc22 NTOF/CECW/PRS |
| CECW (ANoCo baselines + closed-form coupling) | **ARCHIVED** (FAIL → coupling 无信息，doc22 §13) | mean Δ vs A1：cecw −0.010/−0.023/−0.044（k1/2/4）；最强基线 ANoCo-A1concat −0.0076/−0.0137/−0.0055（ANoCo 族全 ≤A1）；coupling 增益 −0.0025/−0.0093/−0.0381；shuffled 掉点 ≤1e-4（cecw≡noconflict≡shuffled）；metal_plate/tubes 类别条件正（被 smoothing/qq 对照复制）；worst −0.080~−0.141 (bracket_*) | g1/g2/g3 三 shot 全 FAIL | 不训练 router；ANoCo 族同构搜索禁止 → 启动 NTOF |
| NTOF normal-only illumination holdout | **ARCHIVED** (FAIL → 转 PRS，doc22 §10/§13) | seed0/k1 全 6 类：g1 FP 比中位 0.934（需 ≤0.75，仅降 ~7%）；g3 true effect 0.060 vs rand 0.032/wrong 0.045/shuf 0.034（差<0.05）；g5 ntof 0.060 < dino-only 0.094；g2 保留 PASS 1.9–2.5；rank{2,3,4}=0.941/0.934/0.928 | g1/g3/g5 FAIL | 不追修（不得用 bad 调 rank/强度）；干预导出管线留作 PRS 基线生成器 |
| PRS / CL-RPF / PSMF / PDMC | PRS **ARCHIVED** (G1 机制门 FAIL → doc22 §5.3/§13)；CL-RPF 见 v12_early_fusion/04_clrpf_probe（G1 三 shot FAIL）；PSMF **ARCHIVED**（R0 FAIL，见 psmf/R0_DECISION.md）；PDMC NOT_STARTED（条件未满足，不启动） | PRS G1 median frac：dino exp 0.173 / gamma 0.264、clip exp 0.072 / gamma 0.236（门 ≥0.80）；方向无关 |corr|≈frac（无反号轴）；mean 响应 V 形（近恒等强度最小）6 类一致 | 未过即停止 | PRS/CL-RPF/PSMF 归档负；PDMC 保持条件性关闭（BC-MCR 已关，重型） |
| Detail-recovery 匹配前恢复（doc26 §4.3，AnyUp multi-backbone） | **ARCHIVED** (6 类前提不普适 + 耦合诊断负 → doc26 §4.3 本形态关闭) | 2 类前提 P1/P2/P3 曾 PASS（au56 0.4912 vs a1 0.4834，metal_plate 主导）；扩 6 类后 macro au56 0.2775 < a1 0.3099 < bl56 0.3186（4/6 类 ≤a1，bracket_white/tubes 坍缩）；坍缩类 du56 0.2586 ≈ au56（M1c FAIL）、< cu56 0.2791（M2 FAIL，条件化连 concat 同算子都不如） | P1/P2 macro-6 FAIL；M1c/M2 FAIL；FP95(du56)>cu56 | 归档负；教训：冻结通用 upsampler 多数类有害（先验与 MPDD 中深层特征不匹配），恢复先于匹配需无学习先验/类别内适配前置 |
| PSMF 相位稳定微缺陷场（doc22 §7） | **ARCHIVED** (R0 FAIL → 机制=多移位平均，doc22 §7.2 预警成真) | G1 身份 PASS（a1(p0) 与归档逐位一致）；macro(3 微类) psmf−a1 −0.0006、psmf−ov −0.0012、psmf−shuf +0.0005；micro bin(448) Δ≈+1e-4 ≪+0.015；唯一增益 metal_plate +0.0053（大缺陷，不微缺陷特异） | G2–G6 全 FAIL | 归档负；不做 Pareto/6 类扩展；4 相位管线留档作"误对齐/错位"类检验基建 |
