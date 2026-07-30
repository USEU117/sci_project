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

## Current branch mapping

- visual branch: AnomalyDINO seed-0 frozen predictions;
- text-guided branch: WinCLIP+ seed-0 frozen predictions.

AnomalyCLIP remains the preferred pure text-guided branch, but its legacy VisA
cache lacks sample identifiers. It will only enter fusion after a reproducible
sample-ID sidecar or regenerated cache is available.

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
