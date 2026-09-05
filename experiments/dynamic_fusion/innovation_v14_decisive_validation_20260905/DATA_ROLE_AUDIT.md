# V14 P0 数据角色泄漏门审计（doc28 §4.2）

日期：2026-09-05。范围：V13 数据角色问题的确认 + v14 承诺的强制检查项。

## 1. V13 审计结论（不回写/覆盖 V13 文件，只登记降级）

- N3 DNC 通道响应与 q 排序使用了 NTOF `good_syn_feat`——该缓存由**测试集 test/good 图**
  生成（`run_r2_ntof_export.py` 经 `index_dataset` 取 `/test/good/`）。虽未读真实缺陷 GT，
  但"知道该图是 good 并以其构造拟合"违反 V13 DATA_ROLES「目标类正常 test/good 不得进入拟合」。
  → V13 N3 全部数字降级为**离线机制探索**，不是 normal-only K-shot 方法结果（已在 V13
  N3_DECISION 补记与 doc28 §2.1-B 一致）。
- N2 CCT 使用 support 自身作 anchor/query、特征域 16×16 平衡 OT：无 test 泄漏（anchor/query 均
  support），但**不是 doc27 计划的软容量半松弛形式** → 降级为"概念探针"。
- N1 JTD：normal-only（support LOO）拟合、GT 仅评估 → 数据角色合规。

## 2. v14 强制的数据角色规则（写入 run 代码）

- fit/select/calibration ID 必须 ∈ manifest `categories[cat][seed][shot]`（support，train/good）；
- 任何 fit ID 含 `/test/` → 抛错（`v14_common.assert_fit_ids_are_support`）；
- 合成 mask 仅用于 support 派生代理目标（方法定位 = "support synthetic adaptation"）；
- 评估 ID（真实 test 图）只在冻结后一次性使用，不进入任何参数/选择。

## 3. 检查清单（每类×seed0×shot{1,2,4} 的 support ID 抽样核验）

| cat | k1 | k2 | k4 | 全部含 /train/good/ | 无 /test/ |
|---|---|---|---:|---:|---:|
| bracket_black | ✓ | ✓ | ✓ | ✓ | ✓ |
| bracket_brown | ✓ | ✓ | ✓ | ✓ | ✓ |
| bracket_white | ✓ | ✓ | ✓ | ✓ | ✓ |
| connector | ✓ | ✓ | ✓ | ✓ | ✓ |
| metal_plate | ✓ | ✓ | ✓ | ✓ | ✓ |
| tubes | ✓ | ✓ | ✓ | ✓ | ✓ |

（脚本：`run_p0_audit.py`，对 manifest 全量枚举断言；结果写入本目录 DATA_ROLE_AUDIT.json。）

## 4. 本轮的强制实现点

- `dnc_selector.select_dnc_c`：修复为分支内 `q_j − λ·redundancy(j, opposite_chosen)` 贪心，
  λ 能改变集合（单测证明）→ 集合不再必然等于 DNC-I；
- `semi_ot.solve_semi_ot`：广义 Sinkhorn 半松弛解，支持 Q≠A、行固定、列软 KL（18/18 单测）；
- 真实 test 评估只允许一个冻结配置（P1-C/P2-B），禁止按真实结果调参。
