# DATA_LEAKAGE_AUDIT (V12-EARLY-FUSION Stage 0)

- Only MPDD development seed0 is consumed in Stage 0 probes (R0/candidate selection role).
- Feature export and alignment use NO test labels / masks; masks and gt_sp loaded offline only for
  pooled Pixel-AP evaluation in the probe.
- No hyper-parameter is tuned against Pixel-AP: layer indices are frozen BEFORE export; static concat
  and oracle rules are pre-registered here.
- VisA = potential source/meta-train only; NOT consumed in Stage 0.
- BTAD / MVTec AD: already-consumed diagnostics; NOT retuned. MVTec AD 2: NOT downloaded / labels NOT viewed.
- The multi-layer exporter writes into outputs/dynamic_fusion/v12_early_fusion/... (git-ignored raw caches);
  only manifests/reports and stage0 probe results are committed under experiments/.
- Deepest-layer parity check reproduces the frozen A1 caches; if parity fails the cache set is invalid and
  Stage 0 must not be evaluated.
