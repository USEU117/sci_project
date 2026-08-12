# DynamicFusion V3.1 development protocol

Status: active development  
Started: 2026-08-12 00:30 Asia/Shanghai

## Research question

Under a target protocol that permits only K normal references (K in 1, 2, 4),
uses no test labels or masks in routing, and uses no test-set aggregate
statistics, determine whether text evidence can selectively rescue visual
failures without materially degrading the visual baseline.

V3.1 separates three concepts that V2 partially conflated:

1. anomaly evidence: distance above the allowed normal-reference evidence;
2. reliability: stability under grouped reference views, prompts and geometry;
3. complementarity: text evidence that is useful beyond the visual branch.

## Data roles

- MPDD: development only. Labels and masks may be used only by offline Gate A
  evaluation after predictions and router features have been produced.
- BTAD: consumed V2 holdout. Forbidden for V3 design, debugging, selection and
  parameter tuning.
- MVTec AD and VisA: historical evidence, not fresh V3 holdouts.
- New holdout: not selected and must not be accessed before V3 parameter freeze.

## Frozen predecessors

- DynamicFusion V1 and V2 code, configurations and artifacts are read-only
  evidence for V3 development.
- V2 `visual_only_safe_fallback` is the mandatory safety control.

## Gate A acceptance logic

Gate A1 asks whether the frozen text branch has repeatable oracle headroom.
Reports must separate image, pixel and region evidence by category, shot and
seed, and must retain visual-only and text-only controls.

Gate A2 asks whether label-free reliability features can identify helpful text
cases. Development labels are evaluator-only. Router contracts must not accept
labels or masks.

No V3 candidate may advance unless:

- sample IDs and map shapes align;
- all inputs are finite and provenance is complete;
- inference uses no test labels, masks or test-set aggregate statistics;
- the full CPU suite passes;
- any claimed gain repeats across at least two of three seeds;
- image performance has no material degradation from the frozen visual control;
- text intervention is non-trivial and its harm rate is reported.

## Stop rules

- Insufficient oracle headroom stops router-weight search and redirects work to
  one stronger text-branch Gate A.
- Ambiguous leakage, invalid provenance or sample misalignment stops the gate.
- A new holdout is never inspected before a formal parameter freeze.

