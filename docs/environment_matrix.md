# Environment Matrix

Each method uses an isolated environment. The AnomalyCLIP environment has now
been verified on the local RTX 3060 Laptop.

| Method | Environment | Python | Upstream requirements | Local status |
|---|---|---:|---|---|
| AnomalyCLIP | `.venv-anomalyclip` | 3.10 | PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118 | CUDA verified |
| WinCLIP | `.venv-winclip` | 3.10 | PyTorch 2.0.0+cu118, OpenCLIP 2.20.0, LAION ViT-B/16-plus-240 | candle zero-/one-shot smoke verified |
| PatchCore | `.venv-patchcore` | 3.10 | PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118, FAISS CPU 1.7.4, timm 0.6.13 | CUDA/CPU-FAISS smoke verified |
| PromptAD | `.venv-promptad` | 3.10 | official `install.sh` | Not started |
| AnomalyDINO | `.venv-anomalydino` | 3.10 | official `requirements.txt` | Not started |
| ReMP-AD | `.venv-remp-ad` | TBD | official requirements | Not started |
| AdaptCLIP | `.venv-adaptclip` | TBD | official requirements | Not started |

Verified AnomalyCLIP imports:

```text
torch 2.0.0+cu118
torchvision 0.15.1+cu118
CUDA 11.8
torch.cuda.is_available() = True
numpy 1.24.4
scipy 1.9.1
scikit-image 0.20.0
scikit-learn 1.2.2
```

After each environment is created, export:

```powershell
python -m pip freeze > outputs/logs/environment/<method>-pip-freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
