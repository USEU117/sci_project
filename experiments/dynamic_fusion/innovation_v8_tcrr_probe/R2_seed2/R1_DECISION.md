# TCRR R1 minimal reranker decision

- macro Pixel-AP gain: +0.036312
- macro Pixel-AUROC gain: +0.002496
- per-shot Pixel-AP gain: {'1': 0.034842760376883046, '2': 0.04315694801746639, '4': 0.03093634117067109}
- per-category Pixel-AP gain: {'bracket_black': 0.009365103928822735, 'bracket_brown': 0.004589493597192402, 'bracket_white': 0.037995026234584076, 'connector': 0.15999254833576465, 'metal_plate': -0.01513756908866056, 'tubes': 0.021067496122337753}
- control gains: {'rotate180': -0.028894080893862497, 'halfroll': -0.04517973419232515}; genuine separation +0.065206
- positive category-shots: 14/18
- gate: PASS — seed1/2 confirmation authorized

- macro_pixel_ap_gain_ge_0005: True
- all_3_shots_positive: True
- positive_category_shots_ge_11: True
- worst_category_gain_ge_minus002: True
- macro_pixel_auroc_loss_ge_minus0002: True
- spatial_control_separation_ge_0003: True