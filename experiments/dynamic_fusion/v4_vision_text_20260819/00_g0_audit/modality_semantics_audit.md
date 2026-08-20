# A1 Modality Semantics Audit (G0)

- run_id: `v4_g0_modality_and_source_audit_20260819_v1`
- created_at_utc: `2026-08-19T06:18:55.458697+00:00`
- **A1 modality**: `dual_visual_fixed_fusion`
- A1 freeze verify all_ok: `True` (entries: 229)

## Conclusion

A1's second branch is a CLIP image patch feature (encode_image -> visual ViT), not an explicit text branch. The prompt_learner is loaded but never used during export; encode_text / image-text similarity are never computed. Therefore A1 is a dual-visual fixed fusion, not explicit visual-text fusion.

## Evidence

- [PASS] export script never calls encode_text — encode_text present in export script = False
- [PASS] export script calls model.encode_image(...) — encode_image present = True
- [PASS] prompt learner is loaded but not used in feature extraction — ['    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)', '    prompt_learner.load_state_dict(checkpoint["prompt_learner"])', '    prompt_learner.to(args.device)']
- [PASS] AnomalyCLIP.encode_image delegates to self.visual (vision-only) — encode_image returns self.visual(image.type(...), ...) at AnomalyCLIP.py:478-479
- [PASS] VisionTransformer.forward consumes only pixel tokens (conv1 -> patches) — forward begins `x = self.conv1(x)` (AnomalyCLIP.py:358); no text tensor enters the vision tower.
- [PASS] DAPM_replace swaps in visual self-attention (no text/prompt injection) — DAPM_replace clones in_proj/out_proj into an Attention module and re-assigns visual resblock attn (AnomalyCLIP.py:344-353).
- [PASS] Attention computes q,k,v all from the same visual patch tensor — Attention.forward: qkv = self.qkv(x) -> q,k,v (AnomalyCLIP.py:71-74); no text or prompt tensor present.

## Candidate source lock (summary)

| id | status | license | queueable |
| --- | --- | --- | --- |
| V0_AnomalyDINO | available | Apache-2.0 | True |
| T0_AnomalyCLIP_text | available | MIT | True |
| V1_SubspaceAD_style | not_cloned | unknown (to verify on clone) | True |
| V2_SubspaceAD_official | not_cloned | unknown (to verify on clone) | False |
| V3_FoundAD | not_cloned | unknown (to verify on clone) | False |
| V4_alt_FastRef | blocked_empty_repo | n/a | False |
| T1_ReMP_AD | available | MIT | True |
| T2_AdaptCLIP | available | GPL | False |

