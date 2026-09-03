# S1-HGLC image-level diagnostic (doc 16 s.3.3)

Pooled Image-AP (mean over categories of per-cat mean over shots):
- A1-max : 0.7985
- A1-top1: 0.7536
- DINO CLS: 0.6769  (delta -0.1216)
- CLIP glob: 0.7035  (delta -0.095)
- TEXT: 0.8234  (delta 0.0249)

- best global: text (delta +0.0249)
- gate (any global >= A1 + 0.010): True

Details: S1_HGLC_DIAG.json