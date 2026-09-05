# V14 决定性验证 FINAL_DECISION_CN

日期：2026-09-05　依据：`docs/paper_writing_preparation_20260830/28_V13_RESULT_AUDIT_AND_NEXT_DECISION_PLAN_CN_20260905.md`（doc28）
实验目录：`experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/`
未修改任何 V13 文件；未访问新外部数据；未下载模型；未启动未经授权的真实缺陷训练；未改论文主表。

## 0. 结论（一句话）
DNC-I / DNC-C-fixed / 半松弛容量三条"增强机制"在**合法、support-only、图像域**验证下均未通过预注册机制门：**无候选进入真实 MPDD 诊断，无 seed1/2，无任何"融合创新"或"通道适配"主张**。v14 以全部失败归档，后续按 doc28 §8 由用户选择收口 A1（选择 A）或另立可学习匹配路线（选择 B）。

## 1. 阶段与停止原因
| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 审计/单测 | done | DATA_ROLE_AUDIT 18/18（无 /test/ 泄漏）；DNC-C 修复 + 半松弛 OT 单测 18/18 |
| P1-A 缓存导出 | done | dino+clip × k2/k4，6 类，5.9 GB；仅 support 图像渲染 + 冻结提取器重编码 |
| P1-B DNC 机制门 | done → **FAIL/ARCHIVE** | G1 1/3 族；G3 Jaccard 0.9981、冗余降 0.2%；G4 增益 0.0000（见 P1 DECISION.md） |
| P1-C 真实冻结诊断 | skipped | G1/G2 失败 → doc28 §5.3 归档规则，禁止真实运行 |
| P2-A 图像域容量门 | done(k2) → **FAIL**；k4 检查中止 | 三分支缺陷区容量 premium 均低于正常 p95（见 P2 DECISION.md） |
| P2-B 真实 MPDD 容量门 | skipped | §6.1"图像域信号消失即停" |
| P3 seed1/2 | skipped | 无真实胜出者；预注册门未满足 |
| 收口 | done | 本文档 + P1/P2 DECISION.md + RUN_MANIFEST.json |

## 2. doc28 §9 八项审计答复
1. **fit/select IDs 中是否有 /test/**：无。manifest 强制 support-only（`v14_common.assert_fit_ids_are_support` 对任何含 `/test/` 的 ID 硬失败）；DATA_ROLE_AUDIT 18 个 cat×shot 全过。
2. **DNC-C 修复后是否真正改变集合、是否有独立收益**：修复有效（不再是恒等副本）但幅度极小——k2+k4 合并 Jaccard=0.9981、跨分支 mean|corr| 仅降 0.2%（需 ≥10%）；宏 AP 增益 0.0000。**无独立收益，不得称融合创新**。
3. **容量 solver 类型与配置**：半松弛（非 V13 平衡 OT），`Q≠A`、query 行固定 `P·1=a`、anchor 列软 KL 容量（τ 项）。实测：P1/P2-A 用 Q=256、A=256(k2)/768(k4 中止)、grid16（特征源 grid32）、ε=0.05、τ=4.0、ρ uniform；行边缘误差 <1e-5（单测）。未在真实 MPDD 上运行（未过门）。
4. **图像域信号 / 真实总体 / 普通与错配控制**：图像域容量信号缺失（A1 三分支为负：dino −0.028、clip −0.004、concat −0.017 vs 正常 p95 参考）；P1 合成 AP：DNC-I 0.784 ≈ random-mean 0.786 ≈ low_nui 0.785，full 0.800；错配/打乱控制无机制增量。真实总体未测（正确跳过）。
5. **增益归属**：不成立（机制门失败）。DNC-I 相对随机通道集无系统优势；DNC-C 无增益；容量在图像域无信号。
6. **哪些因机制失败停止、哪些因时间/资源未完成**：P1-C、P2-B、P3 因机制门失败而停止（非资源）；P2-A k4 稳健性检查因时间/资源中止（solo CPU 27 分钟未完成 1 类，预计 >2 小时；k2 已决定性失败故中止，不计证据）。
7. **实测成本**：全 CPU（本机 GPU 6GB 被其他任务占用，未使用）。缓存 5.9 GB 磁盘；进程峰值内存：clip 导出 ≈1.4 GB，P1-B/P2-A <0.6 GB。耗时可复现命令见 §4；P2-A 求解以热启动优化实现（初值沿用上一探针的收敛列势，收敛点不变，冷/热差 2.2e-16），否则单类需更久。
8. **是否满足进入 seed/外部确认的预注册门**：否。无通过门候选 → 不申请 seed1/2，不申请外部数据授权。

## 3. 逐类 / 逐 shot 关键数字
P1-B 合成宏 AP（DNC-I vs random-mean vs low_nui；shots 合并 per cat）：
- bracket_black：0.801 / 0.804 / 0.810（Δ −0.003）
- bracket_brown：0.790 / 0.803 / 0.817（Δ −0.013）
- bracket_white：0.740 / 0.802 / 0.817（Δ **−0.062**）
- connector：0.789 / 0.777 / 0.791（Δ +0.012）
- metal_plate：0.795 / 0.786 / 0.769（Δ +0.009）
- tubes：0.749 / 0.748 / 0.723（Δ +0.001）

逐留族（k2 / k4）：cutpaste +0.035 / +0.019；local_erasure −0.036 / −0.055；thin_scratch NaN（32 网格不可评）。
机制控制增量：DNC-I−random 无 ≥0.02 的 2/3 族增益；DNC-C−DNC-I=0.0000；打乱相关后仍 0.0000。
P2-A(k2) spillover/FP：A2/A3 通过（far/ring 与 nuisance 均无抬升，因无信号），A4 面积相关 0.19–0.49；A1/A5 失败。

## 4. 精确可复现命令（.venv-anomalyclip）
```
python scripts/innovation_v14_decisive_validation_20260905/test_p0.py
python scripts/innovation_v14_decisive_validation_20260905/run_p0_audit.py
python scripts/innovation_v14_decisive_validation_20260905/export_p1_support_variants.py --branch dino --shot 2  # (与 --shot 4 / --branch clip 同)
python scripts/innovation_v14_decisive_validation_20260905/run_p1_dnc.py --shots 2 4      # -> P1_dnc_fixed/GATES.json
python scripts/innovation_v14_decisive_validation_20260905/run_p2a_capacity.py --shots 2  # -> P2_soft_capacity/P2A_GATES.json
```
产物：`P1_dnc_fixed/{PROTOCOL,SYNTH_RESULTS,GATES,DECISION}.json/md`、`P2_soft_capacity/{IMAGE_PROBE,P2A_GATES,DECISION}*`、`RUN_MANIFEST.json`。

## 5. 下一步（doc28 §8，需用户拍板）
- **选择 A：收口 A1 论文**——贡献限定为"互补成熟视觉表征在统一 patch memory 下的简单特征级融合"，附完整 single-branch/匹配基线、效率与失败边界；探索作为内部证据，不进主文堆砌。
- **选择 B：另立可学习匹配路线（doc28 §8.2，推荐）**——backbone 冻结，逐分支有界正对角通道权重（身份初始化，保留 A1 旁路）的 support-synthetic 匹配目标度量学习；先在 support 合成 leave-one-family-out 上要求留出排序改善且 normal 路径不坍缩，再谈真实门；成功后作为 A2/新论文协议，不与 zero-training A1 混称同一设置。

两条路都不再复活 N1 JTD、旧 DNC-C、PMC tri；本轮 v14 证据完整归档。
