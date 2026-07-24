# Project Status

Updated: 2026-07-25

## Confirmed

- The project root was initially empty; no prior code or Git history was present.
- Python 3.10.11 and Git 2.54.0 are available; Conda is not installed.
- Hardware: NVIDIA RTX 3060 Laptop, 6 GB VRAM; driver 591.86.
- The full phase-one plan is in `PLAN.md`.
- Official sources for AnomalyCLIP, PatchCore, PromptAD, AnomalyDINO, ReMP-AD and AdaptCLIP have been recorded.
- MVTec AD requires the official download form/license flow; VisA is available from AWS Open Data.
- `scripts/verify_system.ps1` has run successfully. PowerShell execution requires:
  `powershell -ExecutionPolicy Bypass -File scripts/verify_system.ps1`.
- `scripts/prepare_splits.py` and `scripts/validate_dataset.py` pass Python 3.10 syntax checks.
- A synthetic directory test confirmed deterministic nested 1/2/4-shot manifest generation.

## Completed

- AnomalyCLIP source archive downloaded and verified:
  commit `3911738c0867544f545a076ad78f3f11d9ecbfdf`,
  ZIP SHA256 `533ED87B6658CDB247D063A249CEFEA54AB81623CB11683C6F02345B9A6CEAFE`.
- VisA official tar downloaded and extracted:
  1,929,840,640 bytes, SHA256
  `2EB8690C803AB37DE0324772964100169EC8BA1FA3F7E94291C9CA673F40F362`.
- AnomalyCLIP environment `.venv-anomalyclip` verified with PyTorch
  `2.0.0+cu118`, torchvision `0.15.1+cu118`, and CUDA available.
- OpenAI CLIP ViT-L/14@336px weight downloaded and verified.
- VisA `meta.json` generated from the official `split_csv/1cls.csv`.
- First complete single-category batch evaluation succeeded on VisA/candle.
- Official 518px four-layer VisA/candle evaluation succeeded without OOM.
- Deterministic nested VisA 1/2/4-shot manifests generated for seeds 0, 1 and 2.

## In progress

- MVTec AD official download and license flow.
- Extending AnomalyCLIP from the candle smoke gate to all VisA categories.

## Not started

- MVTec AD dataset download and validation.
- Full-category AnomalyCLIP reproduction.
- Unified 1/2/4-shot manifests for VisA.

## Constraints and risks

- The official AnomalyCLIP experiments used an RTX 3090 with 24 GB; this machine will
  first run pretrained-checkpoint inference with batch size 1.
- Windows GPU FAISS may be unstable; PatchCore and AnomalyDINO will use CPU FAISS first.
- The AnomalyCLIP environment is isolated in `.venv-anomalyclip`; the system Python
  remains unchanged.
- An incomplete `methods/anomalyclip-main.zip` remains from a timed-out codeload
  attempt; it is not valid source and is not tracked by Git.

## Next action

1. Obtain MVTec AD through its official page.
2. Generate VisA 1/2/4-shot manifests from the extracted official data.
3. Run AnomalyCLIP on all VisA classes at the official 518px setting.
4. Add PatchCore and WinCLIP baseline environments after the AnomalyCLIP gate.
