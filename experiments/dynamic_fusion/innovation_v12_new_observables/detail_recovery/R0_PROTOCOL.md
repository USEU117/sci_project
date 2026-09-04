# Doc 26 §4.3 Detail-Recovery R0 协议（预注册，2026-09-04）

问题：A1 在匹配前把 CLIP 37×37 缩到 DINO 32×32、并在粗网格上 KNN（"先匹配后上采样"）。
doc 26 §4.3 假说：影响微缺陷定位的信息在**进入匹配之前**就丢失；若先用真实预训练
feature upsampler 把双分支恢复到共同更细网格（56×56）再做同样 KNN，应更好。

## 范围（用户裁定：两小类先行）
- 类别：`bracket_black`, `metal_plate`（MPDD development seed0, shot k1）
- 特征：与静态/A1 参考同一 7 层——dino L{6,9,11} @32、clip L{6,12,18,24} @37（ml_ 缓存）
- 恢复算子：**AnyUp multi-backbone**（ICLR'26 Oral，encoder-agnostic、保留输入 768-d 通道，
  冻结）；ckpt `outputs/external_weights/anyup_multi_backbone.pth`
  sha256 `B6CC407DA8986C7E5C9098E61F7531767A9ACA8FFF20A1BC6C99D488E61AAC59`
  （github release 断点续传下载；venv torch2.0 需 RMSNorm 兼容 shim——已内嵌等价实现）

## 臂（统一配方：逐层 normal-only z-map → 7 层等权均值 → dists2map(448,σ4)[::8,::8] → pooled Pixel-AP@56）
| 臂 | 网格 | 恢复 | 含义 |
|---|---|---|---|
| a1 | 32 | bilinear(clip 37→32) | 现 A1 配方（=CL-RPF 静态 mean_std，k1 macro 已验 0.309856==归档 A1 0.3092 同配方值） |
| bl56 | 56 | bilinear 32/37→56 | 廉价插值的恢复先于匹配 |
| au56 | 56 | AnyUp 32/37→56 | 真实预训练恢复先于匹配 |
| au56_w | 56 | AnyUp + 错误 guide | 对照 #7（内容错配 RGB），仅 P1 过时跑 |

support 统计：K=1 单 ref，逐层 LOO Chebyshev 排除半径按网格比例（32→1、56→2）。
query 与 support 同处理（doc 26 §4.3 要求）。

## 门（2 类 macro；每类 = 该类 pooled Pixel-AP@56）
- P1 前提：`au56 − a1 >= +0.003`（匹配前恢复保住信息）
- P2 归因：`au56 − bl56 >= +0.003`（真恢复 ≠ 廉价插值；报告性）
- P3 guide：`au56 − au56_w >= +0.003`（错配 guide 应掉点）
- 停止规则：P1 不过 → 前提被拒，路线归档负，不做耦合模块、不跑 P3（未过即停止）。
- 附注：本配方下 bracket_black/metal_plate 基线高度不对称（a1 ≈ 0.005 / 0.962），
  逐类报告而非只看 macro。

## 结果位置
`experiments/dynamic_fusion/innovation_v12_new_observables/detail_recovery/RECOVERY_RESULTS.json`
脚本：`scripts/innovation_v12_new_observables/run_r3_ef_recovery_probe.py`

---

## 2026-09-04 v2 扩展：前提门扩 6 类（用户指令）

在两小类 P1/P2/P3 全过后，把 **au56 vs a1 / bl56**（P1/P2）扩到全部 6 类，
验证"匹配前恢复"前提是否普适（bracket_brown、bracket_white、connector、tubes 为新增；
bracket_black、metal_plate 已跑）。判定：macro-6 下 P1/P2 均过，且非仅由单一近天花板/近地板类驱动
（逐类报告）。P3（au56_w 错配 guide）在 macro-6 P1 通过后补跑。
若 6 类普适 → 进入耦合设计（下节）；若 P1 在 macro-6 不过 → 前提线按现有两小类正结果作
"局部非普适"归档，耦合设计仍可在两小类上作诊断，但不扩大投入。

## 2026-09-04 v3：6 类前提结果 + 耦合诊断（用户指令，预注册于 du 前向之前）

**6 类前提（P1/P2 macro-6）FAIL**：au56 macro 0.2775 < a1 0.3099 < bl56 0.3186；
bracket_white(0.003 vs bl56 0.089)/tubes(0.519 vs a1 0.657) 严重坍缩，仅 metal_plate 显著为正。
→ 独立 AnyUp 恢复不普适。

**耦合诊断（在 2 个坍缩类 bracket_white + tubes 上；不是扩投入，是判定性检验）**：
doc26 §4.3 核心声称是"跨分支互约束能抑制虚假/坍缩恢复"。若 du56 能救回 au56 坍缩的类，
说明坍缩可被跨分支条件化稳定 → 机制存在，值得 6 类正式门；若仍坍缩 → 坍缩固有，路线整体归档。

- du56：DINO 层 l 恢复输入 = concat[dino_l；clip_L24@32]（同图），取输出前半为 dino_l^56；
  CLIP 层 l 输入 = concat[clip_l；dino_L11@37]，取前半为 clip_l^56（后半丢弃）。
- cu56（对照 #6）：与 du56 完全相同的前向图，但 KNN 用整段 1536-d concat（不拆、逐对 z → 7 对均值）。
- du56_m：du56 但条件分支特征取**另一张 query 图**（跨分支内容错配）→ 应掉点。
- 门限（macro over bracket_white+tubes）：
  M1c du56−au56 ≥ +0.003（耦合救回坍缩）；M2 du56−cu56 ≥ +0.003（条件化结构≠单纯 concat 可用）；
  M3 du56−du56_m ≥ +0.003；FP95(du56) ≤ 1.05×FP95(cu56)（normal 图 95 分位均值代理）。
- 停止规则：M1c 不过 → 坍缩固有，路线归档负，不做 6 类耦合正式门。
