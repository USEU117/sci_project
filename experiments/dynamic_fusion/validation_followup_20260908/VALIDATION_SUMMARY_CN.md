# 夜间交接后续验证总览（2026-09-08）

来源：`docs/OVERNIGHT_HANDOFF_20260908_CN.md` 验证顺序。性质：只读诊断 + 工程 QA；不引入新机制、不改冻结 A1。

## 步骤 1 — 逐 type 分解（V1）
结论：**896/细节分辨率的 image-AUROC 增益不平均且不与 D6 缺口收敛** → 高分辨率不作为 A1 升级机制。
- 增益集中：bracket_black/hole（imgAUC 0.414→0.724）、connector/parts_mismatch（0.752→0.829）、defective_painting（0.751→0.795）、tubes（0.957→0.978）。
- **D6 核心缺口 bracket_brown 上下文族基本未改善**：parts_mismatch 0.432→0.448、bend 0.493→0.509，仍近随机。
- 真实回归：bracket_white/scratches（image −0.067、pixel −0.092）、connector pm 像素 −0.039、bracket_brown/connector 异常图内定位 −0.046/−0.038。
- pooled Pixel-AP 略降（0.3176→0.3105）而异常图内定位微升（0.4466→0.4606）→ 口径分离已量化，退化集中在上下文/连接件类。
- 产物：`v1_decomposition/{V1_JSON.json,V1_REPORT_CN.md}`（`analyze_v1_decomposition.py`）。

## 步骤 2 — 真实 bank 量化 QA（V2）
结论：**FP16 / INT8 持久化存储 roundtrip 在真实 MPDD test（seed0 k2/k4）上全门通过** → 可选工程项，非新颖算法、不改 A1 数字。
- FP16：macro ΔAP ≤ +7.6e-06、worst 类 ≥ −1.4e-06、NN ≥ 99.98%、存储 0.500×、p99 err ≤ 1.3e-05。
- INT8：macro ΔAP |.| ≤ 2.3e-05、worst 类 ≥ −1.7e-04（bracket_white）、NN ≥ 99.53%、存储 0.250×、p99 err ≤ 3.0e-04。
- control 复现冻结宏（k2 0.343698 / k4 0.388328），无 test 拟合/选类。
- 边界：存储压缩 ≠ 运行内存/延迟收益；未运行低精度 runtime。
- 产物：`realbank_quant/{RESULTS_s0_k2.json, RESULTS_s0_k4.json, DECISION_CN.md}`（`probe_realbank_quant.py`）。

## 步骤 3 — 是否立项受限机制
**否。** 判断依据：
1. 若"高分辨率图像级通道"要立项，需其提升集中在 D6 缺口类；实测仅在 hole 收敛，**bracket_brown 上下文族不变（仍 0.45–0.51 近随机）**且 bracket_white/scratches 等真实回归 → 该机制对真正缺口无效、还会伤害部分类。
2. 双尺度 concat ≈ late-mean（差 0.0018，来自 overnight）已否定"融合创新"表述。
3. 为避免按结果挑类别/路由（违反纪律），不进入受控组合设计。

## 总决策：`FOLLOWUP_CLOSE`（A1 仍为唯一冻结方法）
- 分辨率/上下文线索 = 已诊断关闭（部分收益 + 真实回归 + 缺口不收敛）。
- 存储压缩 = 可选工程项，仅在需要持久化 bank 时启用（FP16 2× / INT8 4×），不影响当前冻结清单与论文主表。
- 论文/后续建议：小缺陷评价盲区（stride-8 丢失 6/282 mask）纳入口径说明；D6 的 bracket_brown 上下文缺陷族仍是唯一明确真缺口，需授权新数据/结构先验才另立项。
