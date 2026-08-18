# REPRODUCE — A1 MPDD 冻结配置复现

RunId: `freeze/a1_mpdd_w05` · 环境 Windows · 冻结 2026-08-17。

## 1. 环境

| 用途 | 解释器 | 关键依赖 |
|---|---|---|
| 特征导出 (DINO) / 评估 (faiss) | `.venv-patchcore\Scripts\python.exe` | torch, faiss, sklearn, skimage, cv2 |
| 特征导出 (CLIP) | `.venv-anomalyclip\Scripts\python.exe` | torch, AnomalyCLIP 源码 |

- DINO checkpoint：`~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- CLIP checkpoint：`methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth`
  （SHA256 见 freeze_manifest.json）

## 2. 数据与清单

- 数据根：`data/mpdd_raw/MPDD`
- manifest：`data/splits/mpdd/manifest.json`（K=1/2/4 × seed 0/1/2 的正常参考路径）

## 3. 命令序列（已冻结，勿改参数）

### 3.1 特征导出（GPU，一次性；本冻结已完成并缓存）
DINO（每 (seed, shot)，K=1 为全量、K=2/4 为 ref-only 复用 k1 测试特征）：
```powershell
# K=1（全量，含测试+参考特征）
.venv-patchcore\Scripts\python.exe scripts\export_anomalydino_mpdd_features.py `
  --manifest data\splits\mpdd\manifest.json --dataset mpdd --data-root data\mpdd_raw\MPDD `
  --output-dir outputs\dynamic_fusion\v3_direction_a\features_vitb14_s{SEED}_k1\anomalydino_visual `
  --seed {SEED} --shot 1 --model-name dinov2_vitb14

# K=2/4（ref-only：复用 k1 测试特征，只重算参考特征）
.venv-patchcore\Scripts\python.exe scripts\export_a1_mpdd_ref_only.py `
  --manifest data\splits\mpdd\manifest.json --dataset mpdd --data-root data\mpdd_raw\MPDD `
  --output-dir outputs\dynamic_fusion\v3_direction_a\features_vitb14_s{SEED}_k{SHOT}\anomalydino_visual `
  --base-cache outputs\dynamic_fusion\v3_direction_a\features_vitb14_s{SEED}_k1\anomalydino_visual `
  --branch dino --seed {SEED} --shot {SHOT}
```
CLIP（同理，branch=clip，加 `--checkpoint methods\AnomalyCLIP-main\checkpoints\9_12_4_multiscale_visa\epoch_15.pth`）：
```powershell
.venv-anomalyclip\Scripts\python.exe scripts\export_a1_mpdd_ref_only.py `
  --manifest data\splits\mpdd\manifest.json --dataset mpdd --data-root data\mpdd_raw\MPDD `
  --output-dir outputs\dynamic_fusion\v3_direction_a\features_s{SEED}_k{SHOT}\anomalyclip_text `
  --base-cache outputs\dynamic_fusion\v3_direction_a\features_s{SEED}_k1\anomalyclip_text `
  --branch clip --seed {SEED} --shot {SHOT} `
  --checkpoint methods\AnomalyCLIP-main\checkpoints\9_12_4_multiscale_visa\epoch_15.pth
```

### 3.2 评估（CPU/faiss，9 配置矩阵）
```powershell
.venv-patchcore\Scripts\python.exe scripts\run_a1_mpdd_matrix.py --validate-only
.venv-patchcore\Scripts\python.exe scripts\run_a1_mpdd_matrix.py
```
（串行、marker 断点；每配置报告写入
`experiments/dynamic_fusion/v3_direction_a/a1_matrix_20260817/seed{SEED}_k{SHOT}/`）

### 3.3 审计
```powershell
.venv-patchcore\Scripts\python.exe scripts\audit_a1_mpdd_matrix.py
```

### 3.4 输入检查（validate-only，一条命令）
```powershell
.venv-anomalyclip\Scripts\python.exe scripts\freeze_a1_mpdd.py --verify
```
（S1 只读全量验证：229 项 hash 检查，缺失/尺寸/hash/额外 npz 均报告，**不写盘**。）

### 3.5 CPU 从冻结缓存重算单配置报告
```powershell
# 任意 (seed, shot) 的 concat 报告，纯 CPU/faiss，复用冻结特征缓存
.venv-patchcore\Scripts\python.exe scripts\evaluate_a1_feature_fusion.py `
  --mode concat --pca-dim 0 --whiten 0 --dino-weight 0.5 `
  --dino-features outputs\dynamic_fusion\v3_direction_a\features_vitb14_s{SEED}_k{SHOT}\anomalydino_visual `
  --clip-features outputs\dynamic_fusion\v3_direction_a\features_s{SEED}_k{SHOT}\anomalyclip_text `
  --baseline-dir outputs\dynamic_fusion\v2_mpdd_predictions\v2_mpdd_s{SEED}_k{SHOT}_full_v1 `
  --output experiments\dynamic_fusion\v3_direction_a\a1_matrix_20260817\seed{SEED}_k{SHOT}\concat_pca0_whiten0_w0.5_report.json
```
（汇总表 `experiments/dynamic_fusion/main_results_20260818/` 可由
`scripts/build_main_results_table.py` 一键重算，内部对每数据集 concat 均值做 <1e-6 重算校验。）

## 4. 目录与产物

| 产物 | 路径 |
|---|---|
| freeze_manifest | `experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_manifest.json` |
| METHOD_CARD | `experiments/dynamic_fusion/freeze/a1_mpdd_w05/METHOD_CARD.md` |
| 只读验证报告 | `experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_verification.{json,md}` |
| 特征缓存 | `outputs/dynamic_fusion/v3_direction_a/features_vitb14_s{seed}_k{shot}/anomalydino_visual/`、`features_s{seed}_k{shot}/anomalyclip_text/` |
| 9 配置报告 | `experiments/dynamic_fusion/v3_direction_a/a1_matrix_20260817/` |
| 矩阵汇总/审计 | `.../a1_matrix_20260817/matrix_summary.json`、`matrix_audit.json` |
| 统一性能表 | `experiments/dynamic_fusion/main_results_20260818/{main_results.csv,main_results.json,main_results.md,per_category_results.csv,metric_definition.md}` |
| 状态对账 | `experiments/dynamic_fusion/reconciliation/dynamic_fusion_state_reconcile_20260818_v1/` |

## 5. 验收标准

- 复现 9 配置 mean ΔAP（vs DINO baseline）= **+0.0486**（±0.0005 内视为一致）。
- 逐配置报告存在且含 6 类；`matrix_audit.json` `all_9_configs_pass=true`。
- `freeze_a1_mpdd.py --verify`（只读）全量通过，且 verify 前后 manifest SHA256 不变。
- 五个泄漏字段全 `false`。
- 统一性能表重算校验 `recompute_all_pass=true`（与 per-config report 误差 <1e-6）。

## 6. 预期产物（冻结配置下）

- MPDD 9 配置 mean fused Pixel AP：0.309~0.405（K1→K4 递增），ΔAP 全正。
- 冻结后验证角色（只读，不得按结果调参）：
  - BTAD（external，K1 only）：3/3 正，mean ΔAP +0.0726。
  - VisA（**in-domain**，checkpoint 在 VisA 训练过）：9/9 正，mean ΔAP +0.0524。
  - MVTec（external）：9/9 正，mean ΔAP +0.0320。
- 若新增数据集验证：**只运行冻结版**，不得按验证结果重选权重/规则。
