# Reproduction Notes

## 2026-07-24

- The repository was initially empty.
- `rg.exe` was unavailable in the sandbox, so the inventory used PowerShell.
- Official AnomalyCLIP experiments used an RTX 3090 24 GB; this machine has an
  RTX 3060 Laptop 6 GB. The first milestone is pretrained-checkpoint inference.
- The original WinCLIP implementation is not publicly available from the paper
  authors; use AnomalyCLIP's embedded reproduction first and cross-check with
  `caoyunkang/WinClip`.

## 2026-07-25

- AnomalyCLIP source archive (main, commit
  `3911738c0867544f545a076ad78f3f11d9ecbfdf`) was downloaded and verified.
  ZIP SHA256:
  `533ED87B6658CDB247D063A249CEFEA54AB81623CB11683C6F02345B9A6CEAFE`.
- AnomalyCLIP checkpoints are included in the source archive. VisA evaluation
  uses `checkpoints/9_12_4_multiscale_visa/epoch_15.pth`.
- OpenAI CLIP ViT-L/14@336px weights were downloaded to the code's hardcoded
  cache path and verified with SHA256
  `3035C92B350959924F9F00213499208652FC7EA050643E8B385C2DAC08641F02`.
- The official VisA tar was downloaded and extracted. Size:
  `1929840640` bytes. SHA256:
  `2EB8690C803AB37DE0324772964100169EC8BA1FA3F7E94291C9CA673F40F362`.
- The official VisA `split_csv/1cls.csv` generated `meta.json`: 12 classes,
  962 normal training samples and 1,200 anomalous samples.
- The upstream `test.py` assumes all classes are present in metadata and calls
  `torch.cat([])` for a one-class subset. A minimal local compatibility patch
  skips classes with no predictions; full-class logic is unchanged.
- First complete GPU smoke evaluation: VisA/candle, 336px, feature 24,
  200 test images. Pixel AUROC 98.9, AUPRO 94.2, image AUROC 90.1,
  image AP 91.1.
- Official 518px four-layer VisA/candle evaluation: pixel AUROC 97.6,
  AUPRO 94.5, image AUROC 80.9, image AP 82.6.
- Full VisA official 518px inference completed for all 2,162 test images in
  about 24 minutes without OOM. The upstream all-class AUPRO aggregation was
  stopped after about 51 minutes without writing a log; future runs must pass
  `--dump_predictions` and use `scripts/evaluate_cached.py`.
- Cache validation on VisA/candle succeeded. The 189,930,496-byte NPZ cache
  reproduced image AUROC 80.94%, image AP 82.60%, pixel AUROC 97.58%, pixel AP
  22.47%, and official 200-threshold AUPRO 94.50%.
- The optimized AUPRO implementation extracts connected components once and
  uses sorted pixel scores for every threshold. A deterministic synthetic
  cross-check matched the upstream implementation exactly (absolute
  difference 0.0); the candle result also matches the upstream rounded AUPRO.
- The local AnomalyCLIP compatibility changes are preserved in
  `patches/anomalyclip-test-cache.patch`; third-party source under `methods/`
  remains intentionally untracked.
