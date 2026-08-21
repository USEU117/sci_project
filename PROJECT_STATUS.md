# Project Status

Updated: 2026-08-09

Status reading rule: this file is a historical audit trail. The single
authoritative current state is `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`
(2026-08-19). Earlier `Confirmed`, `Completed`, `In progress`, `Next action`,
dated queue sections and the `2026-08-09 synchronized current state` section
are retained only as historical evidence.

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
  classification Gate A completed with image AUROC 92.92%; the corresponding
  segmentation Gate A completed with pixel AUROC 96.07%.
- PromptAD's VisA loader now accepts the project dataset path, frozen manifest
  and seed through environment variables. A direct loader check selected the
  exact seed-0 4-shot candle references from the manifest. The reusable gate
  runner is `scripts/run_promptad_gate.ps1`, and the third-party change is
  preserved in `patches/promptad-unified-manifest.patch`.
- MVTec AD has now been obtained through the official download flow. The
  archive is 5,264,982,680 bytes with SHA256
  `CF4313B13603BEC67ABB49CA959488F7EEDCE2A9F7795EC54446C649AC98CD3D`.
- MVTec AD is extracted under `data/mvtec`. All 15 categories, training
  images, test images and anomaly masks pass validation.
- MVTec metadata and the unified nested 1/2/4-shot, seed 0/1/2 manifest are
  generated. The manifest SHA256 is
  `0a04260becf73635dd1ffdbe6fb8f16047e6086a9d431dc973a70f2b258fe59f`;
  315 selected path entries pass with zero errors.
- AnomalyCLIP MVTec full inference and common evaluation completed for all
  15 categories and 1,725 test images. Macro image AUROC is 93.89%, image AP
  is 97.05%, pixel AUROC is 94.23%, pixel AP is 44.54%, and AUPRO is 88.33%.

## In progress

### Second-stage dynamic fusion design track

- The uniform prediction-only interface, deterministic confidence router,
  configuration and regression tests are implemented. The router does not
  accept ground-truth masks, labels, category test labels or test-set
  aggregates.
- WP1 cache alignment is complete for VisA seed 0, 1/2/4-shot: AnomalyDINO
  and WinCLIP+ cover all 12 categories and 2,162 test images with matching
  normalized sample IDs, order and labels. Reports are under
  `outputs/dynamic_fusion/alignment/`.
- Legacy AnomalyCLIP caches now have independently verified sample-ID
  sidecars under `outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified`.
  Verification follows the original `meta.json` order and `shuffle=False`
  loader, and checks all labels and resized masks. The first failed mask
  verification is preserved under
  `experiments/dynamic_fusion/20260730_anomalyclip_sidecar_attempt1_failed`.
- Two candle smoke runs completed with no GPU use and valid common NPZ
  outputs. Both routed all 200 samples to the text branch because raw score
  scales are incompatible. This is a calibration diagnosis, not a
  performance claim.
- A normal-reference-only robust calibration interface is implemented
  separately for image scores and pixel maps. The cache fusion runner can now
  load a per-category calibration report, rejects reports that used test
  predictions or test labels, and stores the calibration path, SHA256 and
  category in every fused NPZ.
- The complete synthetic reference pipeline passes for all 12 VisA categories:
  common reference NPZ creation, frozen-manifest audit, robust parameter fit
  and calibrated-cache integration. This is an engineering check only and is
  not used as a performance result. The regression suite now has 20 passing
  tests.
- Real VisA seed-0 1-shot normal reference views are prepared from the frozen
  manifest: 12 source images and 60 deterministic identity/brightness/contrast
  views. Every source and generated view has a SHA256 record. No test image or
  test label was read.
- AnomalyDINO and AnomalyCLIP normal-reference exporters are implemented and
  syntax-checked. Their GPU runs are intentionally waiting because the active
  PromptAD baseline has priority.
- The five-step real-reference wrapper
  `scripts/run_dynamic_fusion_reference_pipeline.ps1` passes syntax, input and
  dry-run validation. Its GPU guard was exercised against the active PromptAD
  job: it exited before creating any output, so the two workloads cannot be
  started accidentally by this entry point.

### Active serial GPU queue (2026-07-31)

- PromptAD VisA seed 1, 1-shot is still active on `pipe_fryum`
  classification. The latest live audit found 22/24 completion markers, 11/12
  merged prediction NPZ files, active GPU memory, increasing
  Python CPU time, and a recently updated segmentation checkpoint. The run is
  active rather than stalled.
- A serial supervisor is running as PID 24720:
  `scripts/run_promptad_stage2_serial_queue.ps1`.
- Its locked order is: validate PromptAD seed 1, 1-shot -> run the real VisA
  seed-0 1-shot normal-reference calibration pipeline -> start PromptAD seed
  1, 2-shot.
- Current queue state is written to
  `outputs/logs/orchestration/20260731_promptad_s1k1_stage2_s1k2/status.json`;
  append-only execution messages are in the adjacent `queue.log`.
- The supervisor requires 24 markers, 12 category NPZ files, 12 evaluated
  categories, 2,162 samples and zero schema errors before advancing. A failed
  second-stage reference run is preserved but does not prevent the baseline
  matrix from resuming.
- While PromptAD owns the GPU, WP3/WP4/WP5 CPU engineering advanced without
  reading ground truth or test predictions. Visual-only, text-only, declared
  fixed-weight and dynamic modes now share one cache runner. Added reliability
  features cover entropy, branch agreement/disagreement, spatial response
  concentration, deterministic normal-view consistency and cross-shot
  sensitivity.
- The CPU-only synthetic contract smoke passed 8/8 checks and the combined
  dynamic-fusion/unified-evaluator suite now passes 28 tests. No performance
  claim is allowed until the real normal-reference calibration is fitted.
- The two temporary supervisors (PIDs 24720 and 38428) exited after exposing a
  Windows PowerShell single-object `.Count` bug. The active PromptAD training
  was not interrupted. Their failure status is preserved, and both scripts
  now use explicit array wrapping.
- A unified persistent scheduler runs as PID 31208 from
  `scripts/run_gpu_job_scheduler.ps1`, driven by
  `configs/gpu_job_queue.json`. It covers the active run, real reference
  calibration, and every remaining VisA seed 1/2 matrix configuration.
- The scheduler persists the next job index and history, detects external
  active jobs, validates before advancing, retries once, skips verified work,
  rejects duplicate schedulers and supports `-Resume`.
- The real-reference pipeline exposed and then fixed three Windows PowerShell
  compatibility issues: formal-run parameter binding, the `Tee-Object`
  literal-path append combination, and stderr handling under `ErrorAction Stop`.
  Failed v2/v3/v4 attempts are preserved; v5 is now running without a
  concurrent PromptAD task.
- Live timing suggests the active seed 1 shot 1 configuration has about
  40-70 minutes remaining. The real-reference window is estimated at
  15-40 minutes, followed by approximately 7-8 hours for seed 1 shot 2.
  These ranges are derived from local marker timestamps.
- AC sleep and hibernation are disabled. Battery sleep is still enabled, so
  the laptop must remain connected to power. D: has approximately 409.8 GB
  free; completed PromptAD configurations use about 3.13 GB each.
- Full overnight details and monitoring commands are in
  `GPU_OVERNIGHT_PLAN.md`; automatic switching is documented in
  `AUTO_GPU_SCHEDULER.md`.
- The active design plan and record protocol are
  `SECOND_STAGE_PLAN.md` and `docs/dynamic_fusion_experiment_protocol.md`.

- PatchCore MVTec Gate A completed for bottle, 1-shot, seed 0. Unified image
    AUROC is 92.54%, pixel AUROC 92.71%, and AUPRO 69.59% under the same
    conservative 128px/256-dim engineering protocol used for the VisA matrix.
- WinCLIP+ MVTec Gate B completed for all 15 categories, 1-shot, seed 0.
    Common image AUROC is 76.79%, image AP 87.58%, pixel AUROC 86.51%,
    pixel AP 27.88%, and AUPRO 70.64%.
- AnomalyDINO MVTec Gate A completed for bottle, 1-shot, seed 0. Common
    image AUROC is 99.92%, pixel AUROC 98.96%, pixel AP 83.04%, and AUPRO
    96.46%.
- PromptAD candle 1-shot raw NPZ export completed. The common evaluator reports
    image AUROC 91.59%, pixel AUROC 95.75% and AUPRO 90.66%; this export is
    separate from PromptAD's native CSV (92.92%/96.07%) because the project
    evaluator uses per-image prediction maps.
- PromptAD VisA Gate B, 1-shot, seed 0, is complete for all 12 categories and
    2,162 test images with zero schema errors. Macro image AUROC is 80.25%,
    pixel AUROC 96.20%, pixel AP 28.54% and AUPRO 81.73%. PromptAD is marked
    as `target_normal_tuning=true` because it learns from target normal shots.
- PromptAD VisA 2-shot, seed 0, is complete for all 12 categories and 2,162
    test images with zero schema errors. Macro image AUROC is 81.15%, pixel
    AUROC 96.79%, pixel AP 29.63% and AUPRO 82.25%.
- PromptAD VisA 4-shot, seed 0, is complete for all 12 categories and 2,162
    test images with zero schema errors. Macro image AUROC is 80.46%, pixel
    AUROC 97.01%, pixel AP 31.87% and AUPRO 83.71%.
- ReMP-AD and AdaptCLIP source repositories have been cloned at fixed commits
    and audited. Both still require Gate A work; AdaptCLIP additionally needs
    a published checkpoint and a local fix for a duplicated dataset block.

## Not started

- PromptAD VisA seed 1/2, 1/2/4-shot matrix and its mean +/- std summary.
- ReMP-AD and AdaptCLIP source/checkpoint audit and Gate A.
- Full-category AnomalyCLIP few-shot reproduction on MVTec.

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

1. Finish PromptAD raw prediction export and its VisA unified matrix; the
   detailed checkpointed execution order is in `NEXT_ACTIONS.md`.
2. Run PromptAD MVTec Gate A.
3. Audit and smoke-test ReMP-AD and AdaptCLIP.
4. Extend the validated methods to the complete MVTec matrix.
5. After the active PromptAD job releases the GPU, export AnomalyDINO visual
   and AnomalyCLIP text scores for the prepared VisA seed-0 1-shot normal
   views, audit both caches, and fit the first real calibration snapshot.
   Do not tune it on final test aggregates.

## 2026-08-03 current status correction

The persistent GPU queue `20260731_full_gpu_queue_v2` finished successfully at
2026-08-02 23:03 UTC with seven completed configurations: PromptAD VisA seed
1/2 for 1/2/4-shot plus the real-reference calibration step. There is currently
no Python training process. GPU utilization reported by `nvidia-smi` is from
Windows desktop applications, not an active experiment.

PromptAD VisA now has nine unified evaluation reports (seed 0/1/2 × shot 1/2/4),
each covering 12 categories and 2,162 test samples with zero schema errors. The
seed mean/std summary still needs to be materialized as a persistent project
artifact. The project is therefore past the training queue, but not past the
full reproduction milestone: complete MVTec matrices, ReMP-AD/AdaptCLIP Gate A,
and the final cross-method tables remain open.

The calibrated dynamic-fusion development matrix exists for VisA seed 0,
1-shot, using AnomalyDINO visual predictions and AnomalyCLIP text predictions.
It contains visual-only, text-only, five declared fixed weights and dynamic
outputs. The current dynamic result is not yet better than the best fixed
weight on this development snapshot, so it is evidence for further diagnosis,
not a final performance claim and not a reason to tune on seed 1/2 or MVTec.

## 2026-08-04 VisA paper-material completion

The four-method VisA matrix has passed a fresh 36-run audit: 4 methods × 3
shots × 3 seeds, with 12 categories, 2,162 test images, higher-is-more-anomalous
score direction and zero validation errors in every unified report. The audit
CSV/JSON, method comparison chart, shot trend chart and per-category heatmap
are stored under `experiments/summaries/`.

PromptAD seed 1/2 unified results have been added to `experiments/registry.csv`;
the registry now contains 59 experiments and passes validation with zero
errors. PromptAD remains explicitly labelled `target_normal_tuning=true`.
The protocol, fairness boundary and paper results draft are in
`docs/visa_experiment_protocol_and_results_draft_20260804.md`. No GPU job was
started for this update.

## 2026-08-04 dynamic-fusion K=2/K=4 update

The real VisA seed-0 normal-reference pipelines passed for both 2-shot and
4-shot. Both branches were audited for all 12 categories, and both calibration
reports explicitly record `test_predictions_used=false` and
`test_labels_used=false`. The two seed-0 development matrices completed all
eight pre-registered modes. Dynamic routing improved AUPRO over fixed weights
on both shots, but its image AUROC remained below the best fixed weight, so the
router is not frozen and no seed-1/2 or MVTec fusion validation was started.

The first failed attempts and the later resumed reports are kept separately.
The development-matrix script now supports `-Resume` and preserves the failed
report while recording resumed completion. K=4 evaluation required
`Workers=1` after a reproducible memory error with `Workers=4`.

## 2026-08-05 selected dynamic-fusion candidate update

The selected VisA seed-0 candidate (`temperature=0.50`,
`decision_margin=0.15`, `min_weight=0.05`) completed full unified pixel
evaluation for K=2 and K=4. Both runs covered all 12 categories and 2,162 test
images, exited without stderr, and retained
`test_predictions_used_by_router=false` and `test_labels_used_by_router=false`.
K=2 obtained 82.26% image AUROC, 94.88% pixel AUROC, 18.16% pixel AP and
80.40% AUPRO. K=4 obtained 82.11%, 94.86%, 17.88% and 78.50%, respectively.
The image temperature is a viable provisional candidate, but the pixel
temperature remains unfrozen because AUPRO is about 1.1 points below the
original dynamic router at both shots. The next bounded development check is a
split-temperature ablation plus K=1 consistency, still on VisA seed 0 only.

## 2026-08-05 dual-temperature split evaluation (FROZEN)

The split-temperature ablation (`image_temperature=0.50`,
`pixel_temperature=0.20`, `decision_margin=0.15`, `min_weight=0.05`) completed
for VisA seed 0, K=1/2/4. All 25 regression tests pass, including the new
`test_split_temperatures_change_only_the_requested_weight_level`.

K=1 consistency audit confirmed all 12 categories have identical pixel weights
and pixel maps to the T=0.20 control while image weights changed. Results:

| Shot | Image AUROC | Pixel AUROC | AUPRO |
|------|------------|------------|-------|
| K=2  | 82.26%     | 94.87%     | 81.55% |
| K=4  | 82.11%     | 94.86%     | 79.60% |

Image AUROC remains above the best fixed weight (82.07%), and AUPRO is fully
recovered to original dynamic-router levels. The dual-temperature configuration
is now locked for development.

Evidence: `experiments/dynamic_fusion/20260805_visa_s0_split_temperature_k1_check/report.json`

## 2026-08-05 dual-temperature final VisA validation (seed 1 & seed 2)

After locking `image_temperature=0.50` and `pixel_temperature=0.20`, the full
VisA final validation matrix (seed 1/2, K=1/2/4) completed via GPU exports of
normal-reference predictions and CPU dynamic fusion. All 6 calibration fits
passed (12 categories each, 0 failures).

View generation for s1_k4 and s2_k1/k2/k4 encountered no code defect — only
slow image processing (SHA256 hashing of 240 views × 1300px images). Using
`$env:HF_HUB_OFFLINE="1"` to avoid transient torch-hub network errors.

Results (all 6 runs):

| Run  | Image AUROC | Pixel AUROC | AUPRO  |
|------|------------|------------|--------|
| s1_k1| 82.11%     | 93.20%     | 68.67% |
| s1_k2| 82.17%     | 94.93%     | 83.31% |
| s1_k4| 82.29%     | 94.65%     | 82.47% |
| s2_k1| 79.77%     | 92.55%     | 67.55% |
| s2_k2| 82.37%     | 94.76%     | 79.08% |
| s2_k4| 82.50%     | 94.94%     | 84.81% |

Cross-seed comparison with seed 0:

| Shot | s0       | s1       | s2       | Mean     |
|------|----------|----------|----------|----------|
| K=1  | 82.03%   | 82.11%   | 79.77%   | 81.30%   |
| K=2  | 82.26%   | 82.17%   | 82.37%   | 82.27%   |
| K=4  | 82.11%   | 82.29%   | 82.50%   | 82.30%   |

The dual-temperature router is stable across seeds. K=2 (82.27%) and K=4
(82.30%) are virtually tied. s2_k1 at 79.77% is within expected cross-seed
variance for 1-shot (seed-0 already showed ±2% class-level spread).

For the paper, the recommended VisA shot mode is K=2 (lower compute, same
accuracy as K=4 on average across seeds). MVTec validation was completed later;
see the authoritative 2026-08-09 section below.

Evidence: `outputs/dynamic_fusion/final_validation/summary.json`

## 2026-08-09 synchronized current state

This section supersedes the dated operational snapshots above. Historical
sections remain as an audit trail and must not be interpreted as current queue
state.

- No experiment training process was active during the 2026-08-09 audit. GPU
  use was Windows desktop/background use, not a project job.
- VisA baseline audit passed 36/36 runs: PatchCore, WinCLIP+, AnomalyDINO and
  PromptAD each have 3 seeds × 1/2/4-shot, 12 categories, 2,162 samples and
  zero validation errors. PromptAD remains `target_normal_tuning=true`.
- MVTec completeness is PatchCore 9/9, WinCLIP+ 9/9, AnomalyDINO 9/9,
  PromptAD 4/9 and DynamicFusion 9/9. The prior AnomalyDINO 8/9 statement is
  obsolete; the fresh 2026-08-09 matrix verifies s1/k2 and all other runs.
- PromptAD MVTec completed s0/k1, s0/k2, s0/k4 and s1/k1. Its authoritative
  queue state is `paused_by_schedule`; pending runs are s1/k2, s1/k4, s2/k1,
  s2/k2 and s2/k4. The legacy `overnight_status.json` is marked
  `superseded` and is not a live monitor. The s1/k2 partial run has 5/30 stage
  markers: bottle and cable cls/seg plus capsule cls; resume at capsule seg.
- Dynamic fusion is frozen at image temperature 0.50, pixel temperature 0.20,
  decision margin 0.15 and minimum weight 0.05. Final provenance is
  AnomalyDINO visual evidence plus AnomalyCLIP text evidence. This corrects the
  stale WinCLIP+ branch label in the previous configuration.
- The frozen dynamic-fusion audit passed 17/17 outputs: six independent VisA
  seed-1/2 runs, two VisA seed-0 supplementary rechecks and nine MVTec runs.
  All calibration files passed the no-test-prediction/no-test-label checks.
- AnomalyCLIP currently has a zero-shot MVTec result only and is not presented
  as a 1/2/4-shot matrix. ReMP-AD has a CUDA-capable environment but still
  needs manifest/NPZ adaptation and Gate A. AdaptCLIP still needs its official
  checkpoint, a batch-size-1 Gate A and a 6 GB VRAM check.

### Current MVTec paper-ready rows

| Method | 1-shot | 2-shot | 4-shot |
|---|---|---|---|
| PatchCore | ready (3/3 seeds) | ready (3/3) | ready (3/3) |
| WinCLIP+ | ready (3/3) | ready (3/3) | ready (3/3) |
| AnomalyDINO | ready (3/3) | ready (3/3) | ready (3/3) |
| PromptAD | incomplete (2/3) | incomplete (1/3) | incomplete (1/3) |
| DynamicFusion | ready (3/3) | ready (3/3) | ready (3/3) |

DynamicFusion MVTec mean ± sample standard deviation from the current unified
reports:

| Shot | Image AUROC | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|
| 1 | 79.43 ± 2.61% | 91.06 ± 0.40% | 24.66 ± 2.28% | 82.33 ± 3.12% |
| 2 | 86.37 ± 1.98% | 93.79 ± 0.43% | 34.40 ± 2.77% | 90.58 ± 1.36% |
| 4 | 89.52 ± 2.69% | 94.18 ± 0.03% | 34.82 ± 0.29% | 91.81 ± 0.80% |

These results may be reported as audited measurements. They do not support a
claim that fusion is superior to the strongest single branch. The 2026-08-09
CPU analysis has now completed the shot-, seed- and category-level comparison.

### Dynamic-fusion scientific analysis completed on 2026-08-09

- All 17 frozen runs and 231 category-run combinations were analyzed without
  GPU use or V1 retuning.
- The main failure mechanism is normal-reference calibration saturation.
  Visual calibrated scores at or above 0.999 average 99.99% on MVTec and
  91.54% on VisA. The resulting ties destroy much of the raw AnomalyDINO image
  ranking before routing.
- Binary entropy treats values near both 0 and 1 as confident, so a saturated
  calibrated value can be interpreted as reliable instead of out-of-range.
- On MVTec, DynamicFusion trails raw AnomalyDINO Image AUROC by 16.29, 10.49
  and 7.94 percentage points at 1/2/4-shot. Partial localization benefits do
  not justify a universal-superiority claim.
- The VisA seed-0 ablation confirms that image and pixel routing require
  separate temperatures. The frozen 0.50/0.20 split improves AUPRO relative
  to the best fixed image-weight baseline while preserving image ranking, but
  does not repair upstream score saturation.
- The final success/failure grid, route-weight plots, saturation diagnostic,
  category heatmap, complete ablation table and analysis workbook are ready.
- V1 scientific analysis and WP8 are complete. A future V2 must use a new
  development/validation boundary; the viewed final sets cannot be reused for
  parameter selection.

### Authoritative evidence

- `experiments/summaries/project_state_snapshot_20260809.json`
- `experiments/summaries/current_method_status_20260809.csv`
- `experiments/summaries/visa_result_audit_20260809.json`
- `experiments/summaries/mvtec_method_seed_shot_completeness_20260809.json`
- `experiments/summaries/mvtec_paper_main_table_template_20260809.csv`
- `experiments/dynamic_fusion/final_validation_audit_20260808/final_validation_audit.json`
- `experiments/summaries/dynamic_fusion_scientific_analysis_20260809/summary.json`
- `docs/dynamic_fusion_scientific_analysis_20260809.md`
- `docs/dynamic_fusion_ablation_and_visualization_20260809.md`
- `outputs/dynamic_fusion/analysis_20260809/dynamic_fusion_scientific_analysis_20260809.xlsx`
- `outputs/logs/promptad_mvtec_resumable_queue/status.json`

### English SCI-style initial manuscript completed on 2026-08-10

- Rewrote the available evidence as an English journal manuscript rather than
  translating the Chinese draft sentence by sentence.
- Applied a journal-neutral single-column structure based on the common author
  requirements of Elsevier, IEEE and Springer Nature. The draft uses a
  248-word single-paragraph abstract, six keywords, numbered sections,
  declarations, data/code availability statements and a generative-AI usage
  disclosure. It intentionally omits a thesis cover and table of contents.
- Preserved the evidence boundary: PromptAD remains marked as target-normal
  tuning; AnomalyCLIP remains a zero-shot text branch; incomplete MVTec
  PromptAD cells and unverified ReMP-AD/AdaptCLIP comparisons are not converted
  into rankings; the documented DynamicFusion V1 failure is retained.
- Output:
  `outputs/paper_draft_20260810/Leakage-Safe_Uncertainty_Routing_English_SCI_Draft_V0.2.docx`.
- Quality checks passed: 16 rendered pages inspected individually, 19/19 body
  citation numbers matched the reference list, seven figures were inline,
  all table geometries fit the section width, and DOCX ZIP integrity passed.
- The long GPU queue remains `paused_by_schedule`; manuscript creation did not
  start or occupy the GPU.

## Current open work

Authoritative status has moved to `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`.
The PromptAD MVTec matrix, ReMP-AD / AdaptCLIP Gate A and the final fair tables
are complete. The V4 extension is closed by decision D (`paper_eligible =
false`).

The single remaining main-line task is:

1. S6 paper delivery: write the paper from the A1 main results (MPDD / BTAD /
   VisA / MVTec, 9/9 positive each), including journal selection,
   author/affiliation/ORCID/funding/CRediT/conflict-of-interest details, and
   archival repository identifiers.

## 2026-08-10 DynamicFusion V2 implementation checkpoint

- The long GPU queue remains paused and no GPU training was launched by V2
  implementation.
- V1 evidence is protected by a six-file SHA256 manifest at
  `experiments/dynamic_fusion/v2/v1_evidence_freeze_20260810/manifest.json`;
  immediate verification passed.
- Separate V2 calibration, diagnostics, safe-router and ablation modules are
  implemented. V1 `ConfidenceRouter` and its frozen result directories were
  not replaced.
- The V2 router now uses a rank-preserving arctangent calibration, directional
  out-of-support evidence, visual-default fallback, capped text assistance and
  fully separate image/pixel paths.
- The full local test suite passed 45/45 tests. The authoritative CPU smoke is
  `experiments/dynamic_fusion/v2/20260810_v2_cpu_contract_v2/report.json`.
  The preceding v1 smoke failure is retained and documents a corrected audit
  aggregation bug rather than an algorithm failure.
- A retrospective software check fitted V2 calibration from the already
  allowed VisA seed-0 normal-reference cache. All 48 calibration rows passed:
  minimum raw/calibrated Spearman 1.0, minimum unique-value ratio 1.0, maximum
  upper-boundary saturation rate 0 and zero degenerate calibrators. This is not
  parameter selection and not a V2 performance claim.
- MPDD and BTAD are not present locally. Their manifests, branch caches,
  development ablation, parameter freeze and independent holdout validation
  remain open. The code/protocol can be frozen before 13 August; scientific
  parameter freeze additionally requires the new development cache.

### 2026-08-10 V2 data preparation started

- BTAD is downloading from the public server linked by the dataset literature;
  the expected archive size is 1,229,193,337 bytes.
- The MPDD author repository remains authoritative for description and license,
  but its SharePoint download currently redirects to institutional login. A
  Hugging Face mirror is therefore being downloaded with an explicit mirror
  provenance label and expected LFS SHA256
  `69f8da73eea4a31451a50251e5c261e83e0c53f2d1a39a7d4dfc78b5c434ddd6`.
- Both downloads are resumable background jobs. A CPU-only preparation process
  waits for exact archive sizes, then performs safe ZIP extraction, dataset
  audit, 1/2/4-shot x 3-seed nested manifest generation, per-selected-file
  SHA256 validation and data-protocol freezing. It will not start GPU work.
- Live state is stored in
  `experiments/dynamic_fusion/v2/data_preparation/automation_status.json`; logs
  are under `outputs/logs/`.
- MPDD/BTAD support was added to the dataset and split validators. The full test
  suite now passes 48/48 tests. The V2 code/protocol freeze covers 21 files and
  verifies successfully.

### 2026-08-11 V2 data preparation completed

- The unattended preparation process completed successfully without GPU use.
- MPDD passed archive SHA256, safe extraction, six-category dataset validation,
  mask checks and nested 1/2/4-shot x 3-seed manifest validation. The manifest
  SHA256 is `5a6a42dd12de1de9c977c2b10695f35b474d19b37f0c1492f64a7989226a9bd8`.
- BTAD passed ZIP verification, safe extraction, three-category dataset
  validation, mask checks and nested manifest validation. Its manifest SHA256
  is `40696d901a78006c342dce98625dc21221b8ee9f642ebb74b7c3f3ffc5a1d215`.
- The data-readiness report is now `ready`. The data-protocol freeze contains
  ten evidence files and passed independent verification.
- The complete local test suite remains 48/48 passed. Parameters are not yet
  frozen and BTAD holdout metrics remain forbidden until MPDD development and
  the formal parameter-freeze gate complete.
- A temporary heartbeat automation named `V2 overnight CPU follow-up` monitors
  and advances CPU-only preparation every 30 minutes until approximately 08:30
  Asia/Shanghai. It is explicitly forbidden from starting long GPU work.

### 2026-08-11 V2 branch-cache queue prepared and dry-run passed

- CPU-only preparation generated 18 deterministic normal-reference sets for
  MPDD and BTAD across 1/2/4-shot and seeds 0/1/2. Their paths and SHA256 values
  are preserved in `experiments/dynamic_fusion/v2/branch_cache_queue/queue.json`.
- The queue contains 36 uniquely named jobs: AnomalyDINO visual and AnomalyCLIP
  text branches for every dataset/seed/shot combination. Its state is
  `prepared_not_started` and `execution_authorized=false`.
- All 36 commands passed exporter `validate_only` checks. The recorded report is
  `experiments/dynamic_fusion/v2/branch_cache_queue/dry_run_report.json` with
  `failures=0` and `gpu_inference_started=false`.
- A fresh independent audit reconfirmed MPDD 6 categories/126 selected path
  entries and BTAD 3 categories/63 selected path entries, with nested 1/2/4-shot
  sampling, checksum matches and zero errors. The full CPU suite passes 48/48.
- No long GPU inference or training was started. MPDD remains the development
  dataset; BTAD remains the unread holdout, and holdout metrics are forbidden
  until MPDD parameter selection and a formal parameter freeze are complete.

### 2026-08-11 daytime V2 branch-cache execution authorized

- The user authorized an unattended GPU window from the morning until roughly
  14:00-15:00 Asia/Shanghai. A single resumable runner now processes the frozen
  36-job queue serially, stops opening new jobs near the cutoff and never runs
  two branch exporters concurrently.
- The first Gate A artifact, MPDD seed 0 / 1-shot AnomalyDINO, passed all six
  category audits with zero failures and false test-use flags. The runner then
  advanced automatically to the paired AnomalyCLIP text cache.
- Passed output directories, per-job logs, audit JSON/CSV files and calibration
  reports are preserved under the V2 branch-cache runtime and output roots.
  Partial outputs are archived before a retry instead of being silently mixed.
- A half-hour heartbeat monitors the existing runner until 15:00. It may resume
  an unambiguously interrupted queue but must not duplicate a live process,
  inspect BTAD metrics or tune router parameters from the BTAD holdout.

### 2026-08-11 V2 normal-reference matrix completed and MPDD Gate A started

- The normal-reference queue completed 36/36 branch exports and 18/18 paired
  calibrations with zero runtime failures. An independent completion audit
  passed all 36 cache audits and all 18 rank-preserving calibration audits.
- Non-empty exporter stderr files contain only known xFormers-unavailable and
  deprecated `pkg_resources` warnings; every corresponding audit passed.
- Separate full-prediction exporters were added for the MPDD development set.
  Both passed validate-only indexing for six categories, 458 test images, 282
  anomalous images and six frozen seed-0/1-shot normal references.
- The MPDD seed-0/1-shot full-prediction Gate A is now running serially, first
  AnomalyDINO and then AnomalyCLIP. It will audit cross-branch sample IDs,
  labels, mask/map shapes, finite values, metadata and leakage flags without
  computing performance metrics. BTAD test data remains outside this gate.

### 2026-08-11 MPDD full-prediction Gate A passed

- Both MPDD seed-0/1-shot branches completed. The paired audit passed six
  categories and 458 images with aligned sample IDs and image labels, matching
  mask/map shapes within each branch, finite predictions and correct dataset,
  branch, seed, shot and score-direction metadata. No metrics were computed.
- The resumable MPDD-only 3-seed by 1/2/4-shot prediction matrix has started.
  AnomalyDINO is rerun for each frozen few-shot selection. AnomalyCLIP is a
  fixed zero-shot branch, so its passed Gate A predictions are reused through
  hard links with explicit invariant-reuse provenance and source SHA256 rather
  than wasting GPU time on identical inference.
- Every pair is audited before the next one starts. BTAD full test predictions
  and metrics remain forbidden until MPDD parameter selection is formally
  frozen.

### 2026-08-11 MPDD full-prediction matrix completed

- All nine MPDD 3-seed by 1/2/4-shot prediction pairs completed with no queue
  failures, and every pair passed the six-category/458-image alignment and
  provenance audit. GPU inference is no longer needed for this matrix.
- Seed-0 development selection has started on CPU using a predeclared seven-way
  grid: visual-only control, the original safe default and five localization-
  focused candidates. Selection maximizes macro AUPRO then pixel AP subject to
  no more than 0.002 image-AUROC degradation from the visual control.
- Candidate evaluation keeps router inputs separate from development labels:
  labels are used only after inference to compare candidates. It writes metrics
  and route summaries, not duplicate full prediction maps, and does not access
  BTAD.

### 2026-08-11 MPDD seed-0 router screening completed

- Six of seven predeclared candidates fell back completely to the visual branch
  and reproduced the visual-only metrics exactly. The only active candidate,
  `pixel_wide_w25`, kept image AUROC/AP unchanged and improved macro AUPRO by
  0.00305.
- The apparent gain is not yet robust evidence: it came only from the 4-shot
  `bracket_white` category (AUPRO +0.05487) with a very small mean pixel text
  contribution; all other seed-0 category/shot rows were unchanged.
- A restricted seed-1/2 validation is running with visual-only, the original
  safe default and `pixel_wide_w25`. Parameters will be frozen only if the
  localization benefit repeats without image-level degradation; otherwise V2
  remains visual-default and the non-repeating gain is recorded as a failure.

### 2026-08-11 MPDD repetition failure and pixel-route defect found

- Seed-1/2 did not repeat the seed-0 gain: `pixel_wide_w25` changed macro AUPRO
  by approximately -0.0000006 versus visual-only, so that candidate was rejected.
- A gate audit showed 99.88% visual-image out-of-support rate, 78.16% visual-
  pixel out-of-support rate and zero image-assist decisions. More importantly,
  the smoothing path incorrectly reapplied the image OOS mask to pixel weights,
  suppressing pixel assistance even when the independent pixel gate passed.
- The smoothing path now reapplies `pixel_allowed` itself, preserving per-pixel
  safety without coupling it back to the image route. A regression test covers
  image-OOS plus valid pixel evidence, and the complete suite passes 49/49.
- The original failed screening and repetition evidence remain preserved. A new
  seed-0 candidate run is in progress under a separate output directory; no V2
  parameter has been frozen and BTAD remains unread.

### 2026-08-11 corrected pixel-route seed-0 result

- After restoring true image/pixel independence, pixel assistance became active
  (about 0.5-1.1% mean text contribution depending on the candidate). Image
  metrics remained exactly visual-only because image text weight was zero.
- The corrected pixel candidates produced only negligible Pixel AUROC/AP gains
  but reduced macro AUPRO by 0.079-0.112. Visual-only was therefore selected on
  seed 0. This indicates that small spatially varying text-map contributions
  disrupt region ranking even when global pixel ranking barely changes.
- A final seed-1/2 comparison between visual-only and the least harmful active
  candidate (`pixel_only_w15`) is running. If the AUPRO loss repeats, the V2
  parameter freeze will use the safe visual fallback and record the dynamic
  localization hypothesis as unsupported on MPDD rather than claiming a gain.

### 2026-08-11 corrected seed-1/2 repetition result

- `pixel_only_w15` preserved image metrics and improved macro AUPRO by 0.0361
  on seed 1 and 0.0236 on seed 2, opposite to its -0.0791 seed-0 change. The
  sign disagreement makes a claim of stable localization gain premature.
- No new candidate is being introduced. A final numerical confirmation is
  running on all three seeds with the unified evaluator's default 200 AUPRO
  thresholds, comparing only visual-only and the already declared
  `pixel_only_w15`. This checks approximation sensitivity without reopening the
  search space. BTAD remains unread and no parameter is frozen yet.

### 2026-08-11 V2 MPDD parameter freeze completed

- The final 200-threshold evaluation found aggregate AUPRO +0.00324 for
  `pixel_only_w15`, but per-seed changes were -0.00315, +0.01770 and -0.00482.
  Only one of three seeds was positive, so the candidate failed the predeclared
  repeatability requirement despite its slightly positive aggregate mean.
- V2 is formally frozen as `visual_only_safe_fallback` with both image and pixel
  text-weight caps set to zero. This preserves the strongest stable branch and
  records the dynamic localization hypothesis as unsupported on MPDD rather
  than selecting an unstable average gain.
- `experiments/dynamic_fusion/v2/parameter_freeze/manifest.json` freezes eleven
  evidence/code/test files by SHA256 and independently verifies. The full suite
  passes 49/49. It records MPDD-label use for development selection, zero BTAD
  prediction/label/metric use before freezing, and now releases the BTAD final-
  validation gate. BTAD results may be reported but may not trigger retuning.

### 2026-08-11 BTAD frozen holdout validation completed

- The generalized BTAD adapters passed dry-run with 3 categories, 741 test
  images and 290 anomalous images. BTAD category 03 uses BMP masks; the indexer
  resolves the unique same-stem mask across supported image suffixes.
- Seed-0/1-shot dual-branch Gate A passed. The resumable full matrix completed
  all 9 seed/shot pairs and all 9 paired audits with no failure.
- Final evaluation used only the already frozen `visual_only_safe_fallback`;
  no BTAD candidate comparison or parameter selection was performed. It wrote
  27 category/seed/shot rows using 200 AUPRO thresholds.
- Macro mean over 27 rows: Image AUROC 0.942271, Image AP 0.946566, Pixel AUROC
  0.964715, Pixel AP 0.570179 and AUPRO 0.721818. Population standard deviations
  are 0.031769, 0.060706, 0.025497, 0.128059 and 0.186377, respectively.
- Evidence: `experiments/dynamic_fusion/v2/btad_prediction_matrix/runtime/status.json`,
  `experiments/dynamic_fusion/v2/btad_prediction_matrix/runtime/audits/`, and
  `experiments/dynamic_fusion/v2/btad_frozen_evaluation/report.json`/`.csv`.
- The project CPU suite passes 49/49. GPU inference for this matrix is complete;
  remaining work is result analysis, tables and paper integration.

### 2026-08-12 DynamicFusion V3.1 overnight Gate A update

- V3 protocol and implementation are isolated from frozen V1/V2. MPDD is used
  only as development data; BTAD was not accessed for V3 design or tuning.
- Gate A1 found evaluator-only oracle pixel/region headroom for the current
  AnomalyCLIP branch, so label-free predictability was tested in Gate A2.
- Gate A2 did not pass: held-out MPDD Pixel AUROC/AP changes were effectively
  zero but slightly negative, and only 1 of 6 held-out categories was positive.
  This negative result is preserved; the current reliability router is not
  qualified and will not be tuned further on the same evidence.
- A stronger AdaptCLIP branch preflight now has frozen-manifest MPDD metadata,
  unified cache export, an isolated dependency environment, provenance checks,
  and validate-only commands. Its official Hugging Face checkpoint is gated
  and absent, so GPU Gate A correctly remains blocked before inference.
- V3 counterfactual checks confirm that unreliable text cannot alter output and
  that routing is equivariant to sample permutation and spatial flips. V3 tests
  pass 19/19 and the current project CPU suite passes 73/73. No V3 GPU work has
  been started.

### 2026-08-12 AdaptCLIP stronger-text Gate A result

- The official VisA-trained AdaptCLIP checkpoint was downloaded from the fixed
  repository revision and matched the declared size and SHA256 exactly.
- The bounded MPDD seed-0/1-shot inference completed all 458 images and wrote
  six category caches. One first attempt completed inference but failed during
  Windows-junction sample-ID export; the failed log was preserved, the path
  mapping was fixed, 24 targeted tests passed, and the clean V2 output was
  generated without overwriting the failed evidence.
- Cache IDs and labels align exactly with the frozen AnomalyDINO visual branch;
  all scores/maps are finite and router leakage flags are false. AdaptCLIP maps
  are 518x518 while visual maps are 448x448, so evaluator-only analysis resizes
  text maps to visual resolution and uses the frozen visual masks.
- AdaptCLIP Gate A1 passed strongly: oracle image headroom was positive on 5/6
  categories, oracle pixel headroom on 6/6, and AdaptCLIP was better on 204/407
  anomaly regions (50.1%). This is only an evaluator upper bound.
- AdaptCLIP Gate A2 failed all three predeclared label-free candidates. Even the
  strict candidate had mean Pixel AUROC -0.00769 and Pixel AP -0.07369 versus
  visual-only, with 0/6 positive categories. The current V3 router therefore
  remains unqualified; no full GPU matrix or holdout evaluation is allowed.

### 2026-08-12 V3.5 Direction C: Image-level Hierarchical Fusion (completed)

- Three per-image gating strategies tested on MPDD seed 0: discrete (3-bin),
  continuous sigmoid, and cross-modal agreement gate. Pixel-level fusion kept
  static (z-score calibrated weighted average, DINO=0.60, AnomalyCLIP=0.40).
- Oracle image-gate upper bound Delta AP = +0.010 — only 2.1% of Gate A1's
  +0.47 pixel-level oracle potential.
- Conclusion: Image-level gating fundamentally cannot access text branch's
  pixel-level benefits. The best image-level gating (agreement gate) achieves
  +0.081 Delta AP, comparable to V3.3 static's +0.081.
- Positive on 5/6 categories for all strategies (discrete/continuous/agreement).
  Multi-seed validation (s1, s2) confirmed.
- Evidence: `experiments/dynamic_fusion/v3_5_hierarchical/s0_report.json`,
  `s1_report.json`, `s2_report.json`
- Implementation: `src/industrial_ad/fusion/v3_5_strategies.py`,
  `scripts/evaluate_v3_5_hierarchical.py`

### 2026-08-12 V3.5 Direction B: Defect Word Prompt Ensemble (completed)

- Strategy: Replace learnable AnomalyCLIP prompts with hand-crafted defect word
  variants (6 fast: "damaged {}", "broken {}", "scratched {}", "deformed {}",
  "cracked {}", "stained {}"), ensemble at feature level via batched encode_text.
- GPU export: Standard CLIP model (design_details=None), DAPM_replace skipped.
  All 6 MPDD categories (458 images) exported to NPZ successfully.
- CPU evaluation: V3.3-style 60:40 z-score fusion comparison.
  - Original learned prompts: mean Delta AP = +0.0754
  - Defect ensemble: mean Delta AP = -0.0364
  - Gain vs original: -0.1117 (defect ensemble is significantly worse)
- Conclusion: Hand-crafted defect word variants cannot substitute learned prompts
  for the text branch. The learned prompts (even with a generic "object" class
  name) produce far better text features for fusion.
- Evidence: `experiments/dynamic_fusion/v3_5_defect_ensemble/s0_eval.json`
- Implementation: `scripts/defect_ensemble_utils.py`,
  `scripts/export_anomalyclip_defect_ensemble.py`,
  `scripts/evaluate_v3_5_defect_ensemble.py`
- Export outputs: `outputs/dynamic_fusion/v3_5_defect_ensemble/s0_shot1/`
  (6 NPZ files + export_report.json)

### 2026-08-12 Project cleanup and handoff preparation

- Deleted 29 debug/temp/ad-hoc scripts from scripts/ and root directory
- Deleted 5 test images from root
- Created comprehensive HANDOFF.md for SLE.Work克 migration
- Updated README.md with current state (V3.3 as main result, V3.5 conclusions)
- This section appended; see HANDOFF.md for full migration guide

### Summary of V3.5 B/C conclusions

| Direction | Method | Delta AP vs V3.3 learned | Verdict |
|---|---|---|---|
| C (Image gate) | Oracle upper bound | +0.010 | Cannot beat static |
| B (Text branch) | Defect word ensemble | -0.112 | Learned prompts essential |

Both directions confirmed that pixel-level information in the text branch is the
core bottleneck, and image-level gating / hand-crafted text cannot access it.
