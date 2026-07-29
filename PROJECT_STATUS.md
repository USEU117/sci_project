# Project Status

Updated: 2026-07-29

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
- Full VisA 518px four-layer inference completed for all 2,162 test images;
  upstream AUPRO aggregation timed out after approximately 51 minutes before
  writing a final log.
- Prediction caching and independent cached evaluation were validated on
  VisA/candle. The optimized 200-threshold evaluator exactly matches the
  upstream AUPRO algorithm on a synthetic cross-check and reproduced the
  official candle metrics: pixel AUROC 97.58% and AUPRO 94.50%.
- Cache-enabled full VisA evaluation completed for all 12 categories and
  2,162 test images. Macro metrics are image AUROC 81.97%, image AP 85.34%,
  pixel AUROC 93.93%, pixel AP 18.98%, and AUPRO 83.60% (official 200
  thresholds).
- The full run produced 12 per-category NPZ caches totaling 2,050,752,485
  bytes; `outputs/anomalyclip/visa_all_cached_metrics.csv` is resumable.
- PatchCore official source and isolated environment are verified. A
  VisA/candle 128px engineering smoke completed with instance AUROC 97.98%,
  full-pixel AUROC 98.42%, and anomaly-pixel AUROC 96.96% using CPU FAISS.
- PatchCore VisA engineering evaluation completed for all 12 categories.
  Macro instance AUROC is 90.87%, full-pixel AUROC 97.02%, and
  anomaly-pixel AUROC 95.89% under the 128px/256-dim local protocol.
- WinCLIP public reproduction source and isolated environment are verified.
  VisA/candle zero-shot and repository-native one-shot smokes completed.
- WinCLIP native zero-shot evaluation completed for all 12 VisA categories at
  240px with LAION ViT-B-16-plus-240. Macro image AUROC is 66.09% and
  pixel AUROC is 73.81% (native `cal_pro=false`; p_pro is not reported).
- WinCLIP native one-shot evaluation completed for all 12 VisA categories.
  Macro image AUROC is 69.40% and pixel AUROC is 89.91%; the native
  thresholded image/pixel F1 means are 76.30% and 14.62%.
- The unified VisA 1/2/4-shot manifest is deterministic and now has a verified
  on-disk SHA256. All 252 selected path entries passed existence, category,
  uniqueness, count and nested-set validation.
- A method-independent NPZ evaluation layer is implemented and its five unit
  tests pass. On the AnomalyCLIP VisA/candle cache it exactly reproduces the
  previously verified image AUROC/AP and pixel AUROC/AP/AUPRO values.
- PatchCore unified VisA/candle Gate A completed for seed 0 at 1/2/4 shots.
  Unified image AUROC is 83.74%/71.57%/94.97%, pixel AUROC is
  85.17%/89.63%/94.16%, and AUPRO is 65.79%/74.14%/83.18%.
- PatchCore unified VisA matrix completed for all 12 categories, shots 1/2/4
  and seeds 0/1/2 (nine full runs, 2,162 test images per run). Across seeds,
  image AUROC mean±std is 68.03±1.19%, 72.91±0.47%, and 78.68±1.11%;
  pixel AUROC is 85.96±0.66%, 90.04±0.21%, and 91.95±0.27%; AUPRO is
  50.60±0.29%, 57.96±0.37%, and 62.21±1.32% for 1/2/4-shot respectively.
- WinCLIP+ now reads the frozen unified manifest, supports 1/2/4-shot, honors
  `--vis false`, and exports the common prediction NPZ schema. VisA/candle
  seed-0 Gate A passed at all three shots. Unified image AUROC is
  84.48%/84.90%/85.09%, pixel AUROC is 90.69%/90.87%/90.90%, and AUPRO is
  85.40%/85.52%/85.36%.
- WinCLIP+ unified VisA matrix completed for all 12 categories, shots 1/2/4
  and seeds 0/1/2 (nine full runs, 2,162 test images per run). Across seeds,
  image AUROC mean±std is 69.96±0.19%, 71.57±0.34%, and 72.58±0.11%;
  pixel AUROC is 89.98±0.24%, 90.50±0.25%, and 90.88±0.08%; AUPRO is
  67.34±0.73%, 68.32±0.75%, and 69.05±0.51% for 1/2/4-shot.
- WinCLIP test features are cached once per VisA category and reused across
  shot/seed configurations. A same-configuration candle rerun matched all
  five NPZ arrays exactly (maximum numeric difference 0).
- The unified evaluator supports two-worker category evaluation; all five
  metric/schema unit tests still pass.
- AnomalyDINO unified VisA matrix completed for all 12 categories, shots
  1/2/4 and seeds 0/1/2. Across seeds, image AUROC mean±std is
  89.40±0.98%, 91.40±0.69% and 92.58±0.24%; pixel AUROC is
  97.97±0.12%, 98.28±0.04% and 98.45±0.04%; AUPRO is
  92.21±0.64%, 93.10±0.36% and 93.69±0.14% for 1/2/4-shot.
- Every AnomalyDINO run contains all 12 categories, 2,162 test samples and
  zero schema validation errors. The reproducible source patch is saved as
  `patches/anomalydino-unified.patch`.
- PromptAD official source is fixed at commit
  `0f86ce0dc1ed59007d51348d8d566aed31360cf9`. Its VisA/candle 1-shot
  classification Gate A completed with image AUROC 92.92%.
- PromptAD's VisA loader now accepts the project dataset path, frozen manifest
  and seed through environment variables. A direct loader check selected the
  exact seed-0 4-shot candle references from the manifest. The reusable gate
  runner is `scripts/run_promptad_gate.ps1`, and the third-party change is
  preserved in `patches/promptad-unified-manifest.patch`.

## In progress

- PromptAD VisA/candle 1-shot segmentation Gate A is running. The next code
  task is replacing the upstream fixed seed files with the frozen manifest
  and exporting the common NPZ schema.
- MVTec AD official download and license flow.

## Not started

- PromptAD Gate A/B/C and its full unified VisA matrix.
- ReMP-AD and AdaptCLIP source/checkpoint audit and Gate A.
- MVTec AD dataset download and validation.
- Full-category AnomalyCLIP reproduction on MVTec.

## Constraints and risks

- The official AnomalyCLIP experiments used an RTX 3090 with 24 GB; this machine will
  first run pretrained-checkpoint inference with batch size 1.
- Windows GPU FAISS may be unstable; PatchCore and AnomalyDINO will use CPU FAISS first.
- PatchCore's upstream mask transform used default bilinear interpolation and
  then cast float masks to integers. Unified runs patch the mask resize to
  nearest-neighbor, matching `configs/protocol.yaml`.
- The AnomalyCLIP environment is isolated in `.venv-anomalyclip`; the system Python
  remains unchanged.
- An incomplete `methods/anomalyclip-main.zip` remains from a timed-out codeload
  attempt; it is not valid source and is not tracked by Git.

## Next action

1. Finish PromptAD segmentation Gate A and record its pixel metrics.
2. Connect PromptAD to the frozen manifest and common NPZ evaluator.
3. Run the PromptAD Gate B/C sequence on VisA.
4. Audit and smoke-test ReMP-AD and AdaptCLIP.
5. When MVTec AD is obtained through the official license flow, generate its
   manifest and extend the validated baselines.
6. Only after the baseline matrix is complete, perform the second-stage
   text/vision complementarity analysis and design dynamic fusion.
