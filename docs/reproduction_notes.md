# 复现问题记录

## 2026-07-24

- 仓库初始为空，不存在可继承的代码或实验。
- `rg.exe` 在当前沙箱中无法执行，文件盘点改用 PowerShell。
- AnomalyCLIP 官方论文实验硬件为 RTX 3090 24 GB，本机为 RTX 3060 Laptop 6 GB；先验证官方 checkpoint 推理，训练复现另行评估。
- WinCLIP 没有公开的论文作者官方实现；优先使用 AnomalyCLIP 仓库提供的复现，再以 `caoyunkang/WinClip` 交叉核对。

