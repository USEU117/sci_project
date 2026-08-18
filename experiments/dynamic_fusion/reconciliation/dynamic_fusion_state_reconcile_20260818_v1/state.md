# 动态融合状态对账快照（S0）

RunId: `dynamic_fusion_state_reconcile_20260818_v1` · 日期：2026-08-18
用途：只读固定 2026-08-18 真实状态，区分旧历史报告与当前权威状态。不改任何旧产物。

## 1. 运行环境

| 项 | 值 |
|---|---|
| Git commit | `46129a7`（Archive phase1-7 deliverables），main → origin/main |
| Working tree | 干净（0 untracked / 0 modified） |
| GPU | RTX 3060 Laptop 6 GiB，当前 1.2 GiB 占用 / 1% 利用率 → 空闲 |
| 活动进程 | 无 |

## 2. 队列状态

| 队列 | 状态 | 完成 |
|---|---|---|
| PromptAD MVTec 9 组合 | completed | 9/9 |
| A1 VisA 特征导出 | complete | 18/18（0 失败） |
| A1 MVTec 特征导出 | complete | 18/18（0 失败） |

## 3. 冻结包

- manifest：`experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_manifest.json`
- SHA256：`80C5AC9BD85D7F7B0037AFDA222EE1CA3AD084117695BCB606108A1849C8B369`
- 配置：concat w=0.5, pca_dim=0, whiten=0, KNN k=1, stride=8, map=448（vitb14 DINO + AnomalyCLIP）
- 五项泄漏字段：全 false

## 4. 各阶段状态（completed / partial / invalid / optional）

| 组件 | 状态 | 关键数字 |
|---|---|---|
| 一：V3.3 泄漏审计 | completed | 12 报告标记 development-only |
| 二：V3.3-clean | completed | 15/15 测试 |
| 三：MPDD s0/K1 Gate | completed | clean w0.40 +0.0173 |
| 四：local rescue | completed | 13/13 测试；+0.0051 < 固定融合 |
| 五：A1 审计+矩阵+权重+动态 | completed | 9/9 全正，mean ΔAP +0.0486，冻结 w=0.5 |
| 六：正式冻结 | completed | freeze_manifest + METHOD_CARD + REPRODUCE |
| 七：冻结后验证 | completed | MPDD +0.0486 / BTAD K1 +0.0726 / VisA +0.0524 / MVTec +0.0320 |
| GPT 验收对账 | completed | 8 点全映射（ACCEPTANCE_MAPPING.md） |
| 负结果归档 | completed | A2/A2b/A3/V3.4/V3.5 全部只读保留 |

## 5. 数据集角色（权威口径，S2 需写入文档）

- MPDD → `development`
- BTAD → `external_frozen_validation_k1_only`（仅 K1 全覆盖）
- VisA → `in_domain_frozen_validation`（AnomalyCLIP checkpoint 在 VisA 训练过，非独立 holdout）
- MVTec → `external_frozen_validation`
- 注意：`outputs/logs/a1_visa_export_queue/status.json` 与 `a1_mvtec_export_queue/status.json` 内 `dataset_role` 仍写 `holdout`，S2 修正。

## 6. 待办（对应对账总表）

1. S1：冻结 verifier 改严格只读（--create/--verify 互斥）+ 篡改测试
2. S2：修正 VisA 路径（20260817→20260818）与数据集角色；新建权威状态文档
3. S3：统一性能表（matched feature baseline 口径）
4. S4：正式方法包（METHOD_CARD/REPRODUCE 修订）
5. S5：Git 分批归档
6. D0（可选）：仅当必须保留"动态"创新时才计算 A1 之上 headroom

## 7. 结论

- 阶段一~七 + 验收对账全部 completed；唯一 optional 项：新外部数据集验证（需授权）。
- 本快照不覆盖任何旧文件，全部为新增只读记录。
