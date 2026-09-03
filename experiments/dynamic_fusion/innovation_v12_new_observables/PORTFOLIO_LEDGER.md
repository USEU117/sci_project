# v12 New-Observables Ledger (doc 22, 2026-09-03)

Authority: `docs/paper_writing_preparation_20260830/22_POST_V11_RESULT_AUDIT_AND_NEW_INNOVATION_ROUTES_CN_20260903.md`.
Status: `NOT_STARTED / RUNNING / PASS / FAIL / ARCHIVED`.
Data roles (doc 22 s11.1): MPDD = exhausted development (R0/candidates only); BTAD/MVTec
consumed; VisA source; Real-IAD/MVTec AD 2 untouched (NOT downloaded).

| 路线 | 状态 | 关键数字 | 门 | 下一步 |
|---|---|---|---|---|
| MTCOA corrected Oracle audit (s0 k1/2/4) | **ARCHIVED** (FAIL → RSR 永久关闭) | mean Δ: k1 +0.0239 / k2 +0.0040 / k4 −0.0129；metal_plate/tubes 各 shot 均 −0.07~−0.13；component-macro: text 32-35% & LLSE 24%（三 shot 一致）；k4 bracket_white 占正 headroom 77% | g2 PASS 三 shot（真实小缺陷互补）；g1/g3/g4 在 k2/k4 FAIL | RSR 永久关闭；类别条件互补留档 → 转向 doc22 NTOF/CECW/PRS |
| CECW (ANoCo baselines + closed-form coupling) | **ARCHIVED** (FAIL → coupling 无信息，doc22 §13) | mean Δ vs A1：cecw −0.010/−0.023/−0.044（k1/2/4）；最强基线 ANoCo-A1concat −0.0076/−0.0137/−0.0055（ANoCo 族全 ≤A1）；coupling 增益 −0.0025/−0.0093/−0.0381；shuffled 掉点 ≤1e-4（cecw≡noconflict≡shuffled）；metal_plate/tubes 类别条件正（被 smoothing/qq 对照复制）；worst −0.080~−0.141 (bracket_*) | g1/g2/g3 三 shot 全 FAIL | 不训练 router；ANoCo 族同构搜索禁止 → 启动 NTOF |
| NTOF normal-only illumination holdout | NOT_STARTED | — | doc22 s3.4 | CECW 无信息 → 已启动准备 |
| PRS / CL-RPF / PSMF / PDMC | NOT_STARTED | — | doc22 s5/s6/s7/s8 | 依序，均先过机制门 |
