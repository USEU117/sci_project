# 环境矩阵

原则：每个方法一个独立虚拟环境，先采用官方版本；只有在 Windows/6 GB GPU 下不可运行时才做最小兼容调整。

| Method | Environment | Python | Upstream requirements | Local status |
|---|---|---:|---|---|
| AnomalyCLIP | `.venv-anomalyclip` | 待审查 | PyTorch 2.0.0（官方实验） | 未建立 |
| WinCLIP | 优先复用 AnomalyCLIP 环境 | 待审查 | 待审查 | 未建立 |
| PatchCore | `.venv-patchcore` | 官方称 3.8 | CPU FAISS 首轮 | 未建立 |
| PromptAD | `.venv-promptad` | 3.10 | 官方 `install.sh` | 未建立 |
| AnomalyDINO | `.venv-anomalydino` | 待审查 | 官方 `requirements.txt` | 未建立 |
| ReMP-AD | `.venv-remp-ad` | 待审查 | 待审查 | 未建立 |
| AdaptCLIP | `.venv-adaptclip` | 待审查 | 待审查 | 未建立 |

每次环境建成后导出：

```powershell
python -m pip freeze > outputs/logs/environment/<method>-pip-freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

