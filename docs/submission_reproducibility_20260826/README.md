# A1 投稿复现包入口（2026-08-26）

本目录是论文投稿前的复现工作入口。当前目标期刊档次为 **SCI 四区**；方法定位为：

> 冻结的 DINOv2 ViT-B/14 与 AnomalyCLIP ViT-L/14@336 双编码器视觉 patch 特征，分别归一化后等权拼接，再以 K 张正常参考图建立 KNN 正常记忆库并输出像素异常图。

它不是动态路由，也没有在最终 A1 推理中显式使用文本特征。动态路线的失败证据仅用于消融、讨论或补充材料。

## 当前完成度（2026-08-27 验收更新）

- **P0-A 已完成**：已建立机器可读审计脚本、版本化证据索引和当前快照。
- **P0-B 已完成**：四数据集、权重、split、结果报告和审计证据可用。
- **CPU 回归已复验**：历史证据 `CPU_REGRESSION_20260826.json` 为 `81 passed`；2026-08-27 对当前 `tests/` 独立复验为 `122 passed in 5.80s`。
- **P0-C 数值重建已完成**：648 个双分支特征 NPZ、36 个配置报告齐全，四数据集重建值均在历史容差内。
- **P0-D smoke 已完成**：实测两个分支均为 768 维，concat 为 1536；旧 1152 记录错误。
- **P0 技术复现包已完成**：compact 包现含 324 个逐图可重放 patch maps、包内独立 CPU 重算脚本 `recompute_tables.py`、`rebuild_manifest_v2.json`、`SOURCE_COMMIT.txt`（`12e1fcf`）与许可证索引；`P0_ACCEPTANCE_AUDIT_20260827.json` 中 `submission_repro_package_complete=true`。公开发布前仍须由作者选择根仓库代码 LICENSE，并把数据集来源补成精确官方 URL 后复核条款。
- **权威验收**：见 `P0_ACCEPTANCE_REVIEW_20260827.md` 和 `P0_ACCEPTANCE_AUDIT_20260827.json`。

## 文件

- `P0_LIVE_AUDIT.json`：2026-08-26 当前机器的只读审计快照。
- `P0_ACCEPTANCE_AUDIT_20260827.json`：加强后的当前验收门禁；区分研究数值重建与可发布复现包。
- `P0_ACCEPTANCE_REVIEW_20260827.md`：人工验收结论、问题和修复顺序。
- `VERSIONED_EVIDENCE.sha256`：本复现入口及核心版本化证据的 SHA256。
- `../PAPER_SUBMISSION_HANDOFF_AND_REPRODUCIBILITY_PLAN_20260826.md`：下一位 AI 的唯一总执行路线。
- `../../scripts/audit_submission_repro_package.py`：重新生成审计快照的脚本。

## 审计命令

```powershell
python scripts\audit_submission_repro_package.py `
  --output docs\submission_reproducibility_20260826\P0_LIVE_AUDIT.json
```

脚本在复现包不完整时返回非零，这是门禁行为，不是脚本故障。当前 P0A–P0I 应全部通过；若返回非零，先检查 maps、source pointer 和 SHA256，不要启动训练或覆盖结果。

## 禁止事项

- 不从 test label / mask 选权重、阈值、类别规则或路由条件。
- 不把 VisA 写成独立外部验证；AnomalyCLIP checkpoint 与 VisA 有训练域关系。
- 不把 3 seeds × 3 shots 当作 9 个独立数据集做显著性检验。
- 不重新开启 V3.3、V3.4、V4 或新动态路由搜索。
- 不把 A1 写成 vision-language fusion；最终用到的是两个视觉编码器的 patch 特征。
