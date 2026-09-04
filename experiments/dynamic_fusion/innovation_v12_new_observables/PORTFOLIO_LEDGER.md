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
| PRS / CL-RPF / PSMF / PDMC | PRS **ARCHIVED** (G1 机制门 FAIL → doc22 §5.3/§13)；CL-RPF 见 v12_early_fusion/04_clrpf_probe（G1 三 shot FAIL）；PSMF/PDMC NOT_STARTED | PRS G1 median frac：dino exp 0.173 / gamma 0.264、clip exp 0.072 / gamma 0.236（门 ≥0.80）；方向无关 |corr|≈frac（无反号轴）；mean 响应 V 形（近恒等强度最小）6 类一致 | 未过即停止：无 GPU 导出 | PRS/CL-RPF 归档负；剩余 PSMF/PDMC 依序先过机制门 |
| Detail-recovery 匹配前恢复（doc26 §4.3，AnyUp multi-backbone） | **R0 前提门 PASS**（两小类先行；bracket_black+metal_plate k1） | au56 macro(2类) 0.4912 vs a1 0.4834（+0.0078，P1 ✓）、bl56 0.4821（+0.0090，P2 ✓）、au56_w 错配 guide 0.4737（+0.0174，P3 ✓）；6 类 a1 配方身份 0.309856==归档 mean_std | P1/P2/P3 全过；边界：增益由 metal_plate 主导，bracket_black 基线近地板 | 下一步：分支条件化耦合恢复（控制 #1–7）相对 AnyUp-generic 独立增益门 |
