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
- Cache-enabled full VisA evaluation completed for all 12 categories and
  2,162 test images. Macro results at the official 518px/four-layer setting:
  image AUROC 81.97%, image AP 85.34%, pixel AUROC 93.93%, pixel AP 18.98%,
  and AUPRO 83.60% (200 thresholds, max FPR 0.30). Per-category values are in
  `outputs/anomalyclip/visa_all_cached_metrics.csv` (the output directory is
  ignored by Git).
- The 12 compressed prediction caches total 2,050,752,485 bytes. The evaluator
  resumes by skipping categories already present in its output CSV.
- PatchCore source HEAD `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` and its ZIP
  SHA256 `17994A589AA979D2981153D57A8B1A7C354A57C803A2607239BF7C79A18B17F4`
  were downloaded. Its isolated environment was verified with CUDA torch
  2.0.0+cu118, torchvision 0.15.1+cu118 and CPU FAISS 1.7.4.
- A VisA/candle adapter was added at `scripts/prepare_patchcore_visa.py`.
  PatchCore's official 224px configuration exceeded local contiguous memory
  (2.69 GiB allocation) after feature extraction. A reduced 128px/256-dim
  smoke completed with `num_workers=0`, CPU FAISS and metrics: instance AUROC
  97.98%, full-pixel AUROC 98.42%, anomaly-pixel AUROC 96.96%. These are
  engineering smoke results, not the official 224px paper setting.
- The same conservative PatchCore configuration was run independently for all
  12 VisA categories. Per-category `results.csv` files are under
  `outputs/patchcore/visa_all_128/`; macro results are instance AUROC 90.87%,
  full-pixel AUROC 97.02%, and anomaly-pixel AUROC 95.89%. This table is an
  engineering comparison only because it uses 128px inputs and 256-dimensional
  target embeddings.
- WinCLIP public reproduction HEAD `a2ee822d77d01fb7beaed54314e61fe34d5027a4`
  was downloaded (ZIP SHA256
  `2FD7FA5426A9D2C66AADF805EA1DBBB04B3C041DBEC9644048FC6156BDE04D0C`).
  Native VisA/candle zero-shot results were Image AUROC 84.97%, Pixel AUROC
  89.86%, Image F1 78.15%, Pixel F1 6.48%; native one-shot results were
  Image AUROC 83.95%, Pixel AUROC 90.74%, Image F1 78.54%, Pixel F1 8.21%.
  The repository runs with `cal_pro=false`, so `p_pro=0` is not a valid
  segmentation metric. Its native k-shot values are 0/1/5/10, not the
  project's unified 1/2/4-shot protocol.
- WinCLIP native zero-shot was then run for all 12 VisA categories using the
  same 240px/LAION ViT-B-16-plus-240 configuration. The per-category logs are
  under `methods/winclip/WinClip-master/outputs/winclip/visa_all_zs/`.
  Macro image AUROC is 66.0883%, pixel AUROC 73.8125%, image F1 74.8358%,
  and pixel F1 8.9192%. These F1 values are repository-native thresholded
  outputs and should not replace AUROC in the main comparison table.
- Native one-shot was run for all 12 VisA categories with one normal support
  image per category. Macro image AUROC is 69.4025%, pixel AUROC 89.9075%,
  image F1 76.2983%, and pixel F1 14.62%. The run is stored under
  `methods/winclip/WinClip-master/outputs/winclip/visa_all_k1/`.
- The original split generator hashed LF text before a Windows text-mode write,
  so the saved digest did not match the bytes on disk. It now writes and hashes
  identical UTF-8 bytes. The regenerated VisA manifest SHA256 is
  `35ecdd88bd3df2c7168a6c234b575ab425180946a80a93ce837d7e350b75e0f2`;
  `scripts/validate_splits.py` checked 252 selected path entries with zero
  errors, and two independent regenerations produced the same hash.
- `scripts/evaluate_unified.py` defines the common NPZ prediction interface.
  Five metric/schema unit tests pass. Its VisA/candle AnomalyCLIP output exactly
  matches the validated cached evaluator: image AUROC 80.94%, image AP 82.60%,
  pixel AUROC 97.58%, pixel AP 22.47%, and AUPRO 94.50%; unified image F1-max
  is 75.63%.
- PatchCore needed two compatibility changes recorded in
  `patches/patchcore-unified-cache.patch`: raw prediction NPZ export and nearest
  neighbor GT-mask resize. The upstream bilinear mask transform produced
  fractional edge labels and then cast them to integers, which disagreed with
  the unified binary-mask rule. After the fix, the upstream rounded pixel AUROC
  and unified result agree.
- PatchCore VisA/candle unified seed-0 results at 128px/256 dimensions are:
  1-shot image/pixel AUROC 83.74%/85.17%, AUPRO 65.79%; 2-shot
  71.57%/89.63%, AUPRO 74.14%; 4-shot 94.97%/94.16%, AUPRO 83.18%.
  The selected normal references are taken only from the common manifest.
- The complete PatchCore VisA unified matrix finished successfully: 12
  categories × 3 shots × 3 seeds. Mean±sample-std over seeds for image AUROC
  is 68.03±1.19%, 72.91±0.47%, 78.68±1.11%; pixel AUROC is 85.96±0.66%,
  90.04±0.21%, 91.95±0.27%; and AUPRO is 50.60±0.29%, 57.96±0.37%,
  62.21±1.32% at 1/2/4-shot. Full per-run and by-shot tables are tracked under
  `experiments/summaries/patchcore_visa_unified/`.
