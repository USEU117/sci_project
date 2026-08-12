# GPU 全自动任务切换说明

更新时间：2026-07-31

## 当前主调度器

- 程序：`scripts/run_gpu_job_scheduler.ps1`
- 任务配置：`configs/gpu_job_queue.json`
- 当前 PID：`31208`
- 状态目录：`outputs/logs/orchestration/20260731_full_gpu_queue_v2`

主调度器已经启动，不需要在每个任务结束后重新对话或手动运行下一条命令。

## 当前自动任务顺序

1. PromptAD VisA seed 1、1-shot；
2. 第二阶段 VisA seed 0、1-shot 真实正常参考预测与校准；
3. PromptAD VisA seed 1、2-shot；
4. PromptAD VisA seed 1、4-shot；
5. PromptAD VisA seed 2、1-shot；
6. PromptAD VisA seed 2、2-shot；
7. PromptAD VisA seed 2、4-shot。

运行中的调度器已经读取当前任务配置，不要在它运行过程中修改本次队列。

## 自动切换条件

PromptAD 任务只有同时满足以下条件才算完成：

- 24 个 `.complete` 标记；
- 12 个类别预测 NPZ；
- 统一评估报告存在；
- 12 个类别；
- 2,162 个测试样本；
- `validation_errors=0`。

真实参考校准必须为 `status=passed`，明确未使用测试预测和测试标签，并覆盖 12 类。
通过验收后调度器立即进入下一项，不需要用户确认。

## 自动恢复

- 每个任务最多自动尝试两次。
- PromptAD 重试会使用已有标记和 NPZ，跳过已完成部分。
- 真实参考校准重试使用新的 `RunId`，不覆盖第一次失败证据。
- 连续两次仍失败时停止，并把原因写入状态和历史记录。
- 已经完整通过的任务会自动跳过。

如果只有调度器异常退出而 Windows 没有重启，可执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_gpu_job_scheduler.ps1 `
  -Config configs/gpu_job_queue.json `
  -Resume
```

若旧调度器 PID 仍存活，重复实例会被拒绝。

## 状态和日志

```text
outputs/logs/orchestration/20260731_full_gpu_queue_v2/status.json
outputs/logs/orchestration/20260731_full_gpu_queue_v2/history.jsonl
outputs/logs/orchestration/20260731_full_gpu_queue_v2/scheduler.log
```

查看当前状态：

```powershell
Get-Content -Encoding utf8 -Raw `
  outputs/logs/orchestration/20260731_full_gpu_queue_v2/status.json
```

持续查看日志：

```powershell
Get-Content -Encoding utf8 `
  outputs/logs/orchestration/20260731_full_gpu_queue_v2/scheduler.log -Tail 30 -Wait
```

## 已验证的行为

- 能识别已经运行的 PromptAD，不会重复启动。
- 旧的两个临时监督器已经退出；当前只有一个主调度器。
- 重复启动主调度器会被 PID 锁拒绝。
- GPU 任务严格串行。
- PowerShell 单对象 `.Count` 问题已通过强制数组包装修复。
- 真实参考流水线的正式运行参数集、`Tee-Object` 参数组合和 stderr
  处理问题也已修复。v2、v2_retry1、v3、v3_retry1、v4、v4_retry1
  的失败目录和日志均保留，当前正式尝试使用 v5。

## 边界

电脑保持开机时可以全自动切换。Windows 重启会结束 PowerShell 调度器，因此跨重启自动
恢复尚未启用。整晚运行应保持电源连接。
