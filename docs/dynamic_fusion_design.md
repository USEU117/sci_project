# Dynamic fusion design track

## Scope

The design track may run while baseline reproduction continues. It must not
change frozen dataset manifests, baseline predictions, metric definitions or
existing result rows.

## Inputs

The router receives prediction-only data:

- visual branch image score and pixel anomaly map;
- text-guided branch image score and pixel anomaly map;
- optional image/pixel uncertainty from each branch;
- derived disagreement and consistency features;
- sample identifiers used only for alignment.

Ground-truth masks, image labels, category test labels and aggregate test-set
statistics are absent from the router API and forbidden by configuration.

## Outputs

- fused image anomaly score;
- fused pixel anomaly map;
- visual/text weight at image and optional pixel level;
- route decision: `visual`, `text`, or `weighted_fusion`;
- diagnostic uncertainty and disagreement features.

## Development protocol

1. Use synthetic data for unit, shape and numerical-stability tests.
2. Use only frozen VisA seed-0 prediction caches for design smoke tests and
   design decisions.
3. Do not fit or select router settings using VisA seed 1/2 or any MVTec test
   result.
4. Freeze the router definition and configuration before final validation.
5. After freezing, evaluate once on VisA seed 1/2 and MVTec with the same
   metrics and manifests as the baseline track.

The first implementation is a deterministic uncertainty-weighted router. It is
an engineering baseline, not yet the final paper contribution. Calibration is
an explicit next design task because different branches may emit scores on
different numeric scales.

## Frozen branch mapping

- visual branch: AnomalyDINO frozen predictions;
- text-guided branch: AnomalyCLIP frozen predictions with verified sample-ID
  sidecars;
- development: VisA seed 0 only;
- independent validation: VisA seed 1/2 and MVTec seed 0/1/2 after the design
  and dual-temperature parameters were frozen.

Early alignment and smoke tests used WinCLIP+ as an engineering text-guided
input. It is not the text branch used by the frozen final-validation runs.

## First frozen-cache smoke

The candle smoke aligned 200 AnomalyDINO and WinCLIP+ seed-0 predictions,
resized maps without GPU use, produced a valid common NPZ and passed common
evaluation. All 200 samples routed to the text-guided branch. This is a useful
failure diagnosis rather than a performance result: the two methods emit
scores on different numeric scales, so raw Bernoulli entropy is not a fair
cross-branch uncertainty measure.

No router performance claim will be made until calibration is specified.
Calibration may use source-domain validation predictions or target-normal
reference shots, but not target test images, labels, masks or aggregate test
statistics.

## 2026-07-30 implementation record

- WP1 alignment is complete for VisA seed 0, 1/2/4-shot AnomalyDINO and
  WinCLIP+ caches. The machine-readable reports are under
  `outputs/dynamic_fusion/alignment/`.
- Legacy AnomalyCLIP caches have verified sample-ID sidecars under
  `outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified`. The sidecar
  derivation follows the original `meta.json` order and `shuffle=False` loader,
  and every category passed label and resized-mask checks.
- The first strict sidecar attempt failed because the checker did not reproduce
  the original 0/1-to-0/255 mask conversion. The failed report remains under
  `experiments/dynamic_fusion/20260730_anomalyclip_sidecar_attempt1_failed`.
- The normal-reference-only calibration interface is implemented and covered
  by synthetic tests. No real test-set aggregate has been used to fit
  calibration parameters.
- Two candle smoke runs route all 200 samples to text before calibration. This
  is a known scale-mismatch diagnosis, not a performance result.

## 2026-07-31 CPU-only WP3/WP4/WP5 implementation

- Common controls now cover visual-only, text-only, declared fixed-weight and
  dynamic routing modes.
- Fixed visual-weight candidates are declared before real evaluation:
  `0.0, 0.25, 0.5, 0.75, 1.0`. Test metrics may not be used to select one.
- Reliability features now include binary entropy, image/pixel branch
  agreement and disagreement, spatial response concentration, deterministic
  normal-view consistency and cross-shot sensitivity.
- Internal-layer consistency remains deferred because the frozen common NPZ
  contract does not contain layerwise features.
- A synthetic-only CPU smoke passed 8/8 checks and the regression suite passed
  28 tests. These are engineering checks, not performance evidence.
