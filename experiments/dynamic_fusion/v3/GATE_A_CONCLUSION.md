# DynamicFusion V3.1 Gate A conclusion

Updated: 2026-08-12 04:48 Asia/Shanghai

## Decision

The current AnomalyCLIP-based route does not pass Gate A. Gate A1 shows an
evaluator-only oracle upper bound, but Gate A2 shows that the current label-free
features cannot identify helpful text cases consistently across held-out MPDD
categories. Mean held-out Pixel AUROC/AP changes are effectively zero but
slightly negative, and only one of six held-out categories is positive.

The scientific question remains valid, but the current router is not qualified
and must not be frozen or presented as a successful dynamic-fusion result.

## Stronger text branch boundary

AdaptCLIP repository provenance, frozen-manifest MPDD metadata, unified cache
export, isolated environment and validate-only checks are ready. GPU Gate A is
blocked because the official gated checkpoint is absent. Its expected SHA256 is
`777821da141eb57d159acef46868440faf773a2dd0acf5c276ec3f258c27edee`.

After authenticated download and exact hash verification, only MPDD seed0/K1,
batch size 1, with metrics initially disabled is authorized. Cache alignment
and leakage audits must pass before evaluation. A full GPU matrix is not
authorized, and BTAD/new holdout access remains prohibited before V3 freeze.

## Verification

- V3 tests: 19/19 passed.
- Project CPU tests: 73/73 passed.
- Leakage/provenance audit: passed.
- V3 GPU used: no.
