# 动态融合权威状态（CURRENT STATUS）

更新日期：2026-08-18 · RunId：`current_dynamic_fusion_status_20260818`
机器可读版：[current_dynamic_fusion_status.json](file:///d:/STUDY/My_github/sci_project/docs/current_dynamic_fusion_status.json)
状态快照：`experiments/dynamic_fusion/reconciliation/dynamic_fusion_state_reconcile_20260818_v1/state.json`

本文件是**唯一权威状态**。旧历史报告（阶段七 2026-08-17、V3/V2 各版本）只读保留，不代表当前结论。

---

## 1. 方法（Method）

**名称**：Reference-Conditioned Multimodal Feature Fusion with a Normal Memory Bank（即 A1）

- 分支：DINO `dinov2_vitb14` patch 特征 + AnomalyCLIP `ViT-L/14@336` patch 特征
- 融合：各分支 L2-normalize → CLIP grid 对齐 DINO grid → concat → L2-normalize → KNN(k=1) normal memory bank → distance/2 = 像素异常图
- 冻结配置：`pca_dim=0, whiten=0, dino_weight=0.5, stride=8, map=448`
- memory bank 只由当前 seed/shot 的 K 张正常参考图构建；测试标签/掩码只进 evaluator
- 五项泄漏字段全 `false`；**不是动态路由**（固定融合，权重不随测试图变化）

## 2. 数据集角色（权威口径）

| 数据集 | 角色 | 说明 |
|---|---|---|
| MPDD | `development` | 冻结配置与权重在此开发、矩阵 9/9 全正 |
| BTAD | `external_frozen_validation_k1_only` | 仅 K1 三 seed 全覆盖，K2/4 缺 GPU 特征缓存 |
| VisA | `in_domain_frozen_validation` | AnomalyCLIP checkpoint 在 VisA 训练过，**非独立 holdout** |
| MVTec | `external_frozen_validation` | 冻结后新验证，9/9 全正 |

## 3. 主结果（baseline source 均显式）

| 数据集 | 角色 | 正向配置 | mean ΔAP vs DINO | baseline source |
|---|---|---|---|---|
| MPDD | development | 9/9 | **+0.0486**（vs legacy v2 dino score） | legacy v2 score 缓存 + matched feature-level dino-only KNN |
| MPDD（口径拆分） | development | — | concat-minus-feature-DINO-only **+0.0258** | feature-DINO-only 自身 +0.0227（vs legacy） |
| BTAD | K1 only | 3/3 | **+0.0726** | feature-level dino-only KNN |
| VisA | in-domain | 9/9 | **+0.0524** | feature-level dino-only KNN |
| MVTec | external | 9/9 | **+0.0320** | feature-level dino-only KNN |

要点：
- `+0.0486` 是相对 legacy DINO 分数基线；**纯 concat 贡献看 `+0.0258`**（相对 matched feature-DINO-only）。
- 9/9 是同一测试集上 9 组参考采样的鲁棒性，**不是 9 个独立数据集**，不做伪独立显著性。
- 已归档路线（V3.3-leaky / V3.4 / V3.5 / A2 / A2b / A3）不进入主结果。

## 4. 阶段状态

| 阶段 | 状态 |
|---|---|
| 一：V3.3 泄漏审计 | ✅ completed（12 报告 development-only） |
| 二：V3.3-clean | ✅ completed（15/15 测试） |
| 三：MPDD s0/K1 Gate | ✅ completed（+0.0173，泄漏虚增约 7 倍结论） |
| 四：local rescue | ✅ completed（13/13 测试，安全回退） |
| 五：A1 审计+矩阵+权重+动态 | ✅ completed（9/9 全正，冻结 w=0.5） |
| 六：正式冻结 | ✅ completed（freeze_manifest + METHOD_CARD + REPRODUCE） |
| 七：冻结后验证 | ✅ completed（MPDD/BTAD/VisA/MVTec） |
| GPT 验收对账 | ✅ completed（8 点全映射） |
| S1：只读 verifier | ✅ completed（229 项全过 + 8/8 篡改测试） |
| S2：文档与角色修正 | ✅ completed（本文件 + 60 处 JSON 角色修正 + 链接检查） |
| S3：统一性能表 | ✅ completed（13 行主表 + 36 行逐类，重算 <1e-6 全 PASS） |
| S4：正式方法包 | ✅ completed（METHOD_CARD/REPRODUCE 修订 + 伪代码 + schema + 资源统计） |
| S5：Git 归档 | ✅ completed（分 3 批提交并 push） |
| S6：论文交付 | ⏳ pending（建议下一步） |
| D0：动态 headroom 门 | ✅ passed（MPDD 9 配置逐像素 best-of-3 Oracle，mean headroom +0.5807） |
| D1：可预测性门 | ❌ **failed → 路线 D 永久归档**（LOCO mean AUROC 0.592 < 0.60，特征置乱不下降 0.616 → 无标签特征无法预测修正时机，复现 V3.4 教训） |

## 5. 剩余工作

1. S6 论文交付（7 节结构，全部数字可追溯）——唯一剩余主线任务
2. ~~路线 D~~ → **已归档**（D0 通过但 D1 失败：像素级虽有互补上限，但无标签特征无法预测"何时修正 A1"，动态路由无可靠依据；按设计审查第 12 节第 9 条正式停止扩展）

## 6. 链接检查

详见 `experiments/dynamic_fusion/reconciliation/dynamic_fusion_state_reconcile_20260818_v1/link_check.json`。
