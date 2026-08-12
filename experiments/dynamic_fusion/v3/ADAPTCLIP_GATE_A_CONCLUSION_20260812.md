# AdaptCLIP V3 Gate A conclusion — 2026-08-12

## Scope

MPDD development only, seed 0 and K=1.  The frozen visual branch is
AnomalyDINO.  AdaptCLIP is the replacement text branch.  This is not a paper
test result and it is not evidence of cross-seed generalization.

## Cache acceptance: passed

The first exporter attempt failed only while converting a junction-resolved
path into a staged-relative ID.  The exporter was repaired to use the physical
MPDD root, and the rerun was written to a new directory without overwriting the
failure evidence.  The acceptance audit verified six categories, 458 images,
global unique IDs, exact IDs/order against the visual cache, finite scores and
maps, valid map/mask dimensions, no traceback/OOM, and all three router leakage
flags false.

Artifacts:

- `adaptclip_gate_a_v2/prediction_cache_acceptance.json`
- `adaptclip_gate_a1_adaptclip/report.json`
- `adaptclip_gate_a2_adaptclip/report.json`

## Gate A1: passed (headroom exists)

The evaluator-only label-informed oracle has image-level headroom in 5/6
categories, pixel-AP headroom in 6/6, and AdaptCLIP is stronger in 204 of 407
annotated anomaly regions (50.12%).  This justified testing a deployable
unlabelled router; it does not itself define one.

## Gate A2: failed (headroom is not predictable)

AdaptCLIP calibration was fitted only from the six selected K=1 normal
reference images.  The frozen visual calibration was already leak-safe.  Three
inference-only selective-rescue settings were evaluated, with labels/masks used
only after fusion for the diagnostic metrics.  The least harmful setting
(`strict`) still yields mean pixel AUROC delta **-0.00769**, mean pixel AP delta
**-0.07369**, and positive AP in **0/6** categories.  It therefore fails the
predeclared rule: AP must be positive, AUROC loss no worse than 0.002, and at
least 3/6 categories must improve.

## Decision

Do not freeze V3 as successful and do not expand AdaptCLIP to additional seeds,
shots, datasets, or a GPU matrix.  The defensible conclusion is narrow: with
the present K=1 normal-reference calibration, the router cannot reliably
identify the cases where the stronger text branch should override the visual
branch.  Preserve this as a negative ablation and return GPU time to the
preplanned baseline matrix after the scheduled window opens.
