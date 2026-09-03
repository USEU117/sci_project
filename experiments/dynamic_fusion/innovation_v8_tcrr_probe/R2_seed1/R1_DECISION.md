# TCRR R1 minimal reranker decision

- macro Pixel-AP gain: +0.036349
- macro Pixel-AUROC gain: +0.002794
- per-shot Pixel-AP gain: {'1': 0.03684561496381592, '2': 0.03743683022844303, '4': 0.034764110151999296}
- per-category Pixel-AP gain: {'bracket_black': 0.026049646169386154, 'bracket_brown': 0.002846775100491472, 'bracket_white': 0.019435363738498135, 'connector': 0.16171986811523573, 'metal_plate': -0.011929512885594185, 'tubes': 0.019970970450499188}
- control gains: {'rotate180': -0.020516223601338163, 'halfroll': -0.05204939205718559}; genuine separation +0.056865
- positive category-shots: 15/18
- gate: PASS — seed1/2 confirmation authorized

- macro_pixel_ap_gain_ge_0005: True
- all_3_shots_positive: True
- positive_category_shots_ge_11: True
- worst_category_gain_ge_minus002: True
- macro_pixel_auroc_loss_ge_minus0002: True
- spatial_control_separation_ge_0003: True