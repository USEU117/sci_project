# GPU 整晚串行运行计划

更新时间：2026-07-31 00:25（Asia/Shanghai）

## 当前状态

- GPU：RTX 3060 Laptop，显存 6144 MiB。
- 当前显存占用约 4946 MiB，温度约 60°C。
- 当前任务：PromptAD VisA seed 1、1-shot，`pipe_fryum` 分类。
- 当前完成度：22/24 个训练标记，11/12 个合并预测。
- 当前训练进程仍在增加 CPU 时间并更新 checkpoint，没有卡死证据。
- D 盘剩余约 409.8 GB；一个完整 PromptAD 配置约占 3.13 GB，空间充足。

## 时间估算

本机 seed 1、1-shot 的历史中位耗时：

- 单类别分类约 12.6 分钟；
- 单类别分割约 26.7 分钟。

当前 `pcb4` 分割已运行约 27 分钟，后面还剩 `pipe_fryum` 分类、分割、预测合并和统一评估。
预计当前配置还需约 40–70 分钟，较可能在 01:05–01:35 完成。

之后：

- 真实正常参考导出与校准：暂估 15–40 分钟；
- PromptAD seed 1、2-shot：约 7–8 小时；
- 因此 seed 1、2-shot 大约在 08:20–10:15 完成。

该估算来自本机历史标记时间，不是保证时间。数据加载、AUPRO 评估、Windows 后台活动和首次模型加载都会造成波动。

## 自动执行顺序

旧 PID `24720` 和 `38428` 因 PowerShell 单对象 `.Count` 问题退出，失败状态已保留，
当前训练没有受到影响。

统一主调度器 PID `23512` 已取代两个临时队列：

1. 等待并验收 seed 1、1-shot；
2. 运行第二阶段真实正常参考预测与校准；
3. 运行 seed 1、2-shot；
4. 运行 seed 1、4-shot；
5. 运行 seed 2、1-shot；
6. 运行 seed 2、2-shot；
7. 运行 seed 2、4-shot。

每个配置最多自动尝试两次。第二次会利用已有 `.complete`、原始 NPZ 和合并预测断点恢复。
两次仍失败则停止队列并保留日志，不继续传播错误。

## 防止 GPU 冲突

- 任意时刻只允许一个 PromptAD 训练进程。
- 第二阶段真实参考流水线会在发现 PromptAD 时主动拒绝启动。
- 尾队列只有在第一队列进程结束、状态为 `completed` 且 seed 1、2-shot 验收通过后才启动。
- 统一评估和文件合并可能主要使用 CPU，因此短时间 GPU 利用率较低是正常现象。

## 电源与运行条件

Windows 当前“接通电源”状态下自动睡眠和自动休眠均为关闭。
电池模式仍会在约 10 分钟后睡眠，因此整晚运行必须保持电源适配器连接，并避免手动关机或重启。

## 查看状态

统一主调度器：

```powershell
Get-Content -Encoding utf8 -Raw `
  outputs/logs/orchestration/20260731_full_gpu_queue_v2/status.json
```

持续查看调度日志：

```powershell
Get-Content -Encoding utf8 `
  outputs/logs/orchestration/20260731_full_gpu_queue_v2/scheduler.log -Tail 30 -Wait
```

查看 GPU：

```powershell
nvidia-smi -l 5
```

## 局限

- 该队列能处理普通训练失败并进行一次断点重试，但不能跨 Windows 重启自动恢复。
- “保持 GPU 工作”表示持续安排有效实验，不保证 GPU 利用率一直为 100%；数据读取、导出和评估阶段出现低利用率是正常的。
