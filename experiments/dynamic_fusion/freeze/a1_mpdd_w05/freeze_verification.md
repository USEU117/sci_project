# 冻结验证工具只读修复（S1）

RunId：`s1_freeze_verifier_readonly_20260818` · 日期：2026-08-18

## 修复内容（scripts/freeze_a1_mpdd.py）

原缺陷：`--verify` 会先全量重算并**覆盖写入** manifest，再只对 code 子集做比较，
因此既不能证明冻结未变化，也未验证 checkpoints/manifests/evaluators/缓存。

修复后：

- `--create` 与 `--verify` 互斥（argparse required mutually exclusive group），必须显式二选一；
- `--verify` **严格只读**：读取既有 manifest，绝不写盘；
- 全量验证 code / checkpoints / manifests / evaluators / 特征缓存 / baseline 缓存，
  逐项检查 缺失 / 尺寸不一致 / hash 不一致，并扫描缓存目录报告**额外未声明的 .npz**；
- 验证失败返回非零退出码。

## 验收结果

| 项 | 结果 |
|---|---|
| 全量验证（真实冻结包） | **229 项全部通过**，missing/size/hash/extra 全空 |
| `--verify` 前后 manifest SHA256 | 完全相同（`80c5ac9b…8b369`）→ 只读成立 |
| 篡改测试 | 8/8 通过（tests/test_freeze_a1_mpdd.py） |

## 测试覆盖

1. verify 通过且 manifest 不变；
2. 同尺寸篡改内容 → hash mismatch；
3. 删除文件 → missing；
4. 声明尺寸不符 → size mismatch；
5. 缓存目录多出未声明 npz → extra；
6. manifest 损坏（非法 JSON）→ 报 error；
7. `--create --verify` 同传 → 报错退出；
8. manifest 缺失 → 退出码 1。

## 证据

- 工具：[scripts/freeze_a1_mpdd.py](file:///d:/STUDY/My_github/sci_project/scripts/freeze_a1_mpdd.py)
- 测试：[tests/test_freeze_a1_mpdd.py](file:///d:/STUDY/My_github/sci_project/tests/test_freeze_a1_mpdd.py)
- 验证报告：`freeze_verification.json`（本目录）
- 结论：S1 完成，未删除或覆盖任何旧输出。
