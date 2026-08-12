# DynamicFusion V3.1 overnight handoff

Snapshot: 2026-08-12 05:33 Asia/Shanghai

## Completed

- Declared and froze V3 protocol boundaries without changing frozen V1/V2.
- Built separate V3 evidence/reliability contracts, grouped K-shot calibration,
  text eligibility, opportunity detection, region rescue, visual anchoring,
  bounded one-sided residual, independent image/pixel paths, reason codes and
  safe fallback.
- Completed MPDD Gate A1 oracle analysis over 54 category/seed/shot rows.
- Completed MPDD cross-category Gate A2 over all three seeds using the declared
  small grid.
- Added counterfactual, cache-schema, metadata, leakage and provenance audits.
- Prepared one stronger AdaptCLIP branch through the CPU/configuration boundary:
  repository provenance, frozen MPDD seed0/K1 metadata, unified prediction
  export, isolated dependencies, CUDA-overlay preflight, strict checkpoint
  validation and bounded launcher.
- Synchronized runtime status, PROJECT_STATUS, V3 Gate A conclusion and V3 next
  actions. All important artifacts have recorded SHA256 values.

## Scientific conclusion

Gate A1 found oracle headroom, including 802 of 3,663 anomaly regions where the
text branch was better. Gate A2 did not find a deployable label-free selector:
mean held-out Pixel AUROC change was -8.01e-8, mean Pixel AP change was
-7.82e-9, and only one of six held-out categories was positive. The current
AnomalyCLIP route therefore fails Gate A and is not frozen.

## Verification

- V3 tests: 19/19 passed.
- Full project CPU tests: 73/73 passed.
- Leakage/provenance audit: passed.
- V3 GPU inference/training: not used.
- BTAD accessed for V3 design, debugging, selection or tuning: no.

## Preserved failure evidence

- Gate A2 scientific acceptance failure is recorded, not hidden.
- An initial overly broad pytest command collected five external baseline-repo
  tests and failed on their isolated imports. Its log is retained. The correct
  project `tests/` suite then passed 70/70 and later 73/73.
- An unauthenticated official checkpoint probe returned HTTP 401; no partial or
  unverified checkpoint was written.

## Current blocker

The official AdaptCLIP Hugging Face repository requires authenticated gated
access, and this machine has no saved login/token. The required VisA-trained
checkpoint must have SHA256
`777821da141eb57d159acef46868440faf773a2dd0acf5c276ec3f258c27edee`.

After the user completes the one-time login/access step, rerun validate-only.
If it passes, the next experiment is exactly one MPDD seed0/K1, batch-size-1
Gate A with metrics disabled until its 458 samples and six categories pass
cache alignment and leakage audits. Do not start a full GPU matrix.
