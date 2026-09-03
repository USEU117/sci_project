# TCRR R1 minimal reranker decision

- macro Pixel-AP gain: +0.032147
- macro Pixel-AUROC gain: +0.002944
- per-shot Pixel-AP gain: {'1': 0.028602426555814353, '2': 0.03420737979280258, '4': 0.03363117530713667}
- per-category Pixel-AP gain: {'bracket_black': 0.008489901405089493, 'bracket_brown': 0.002850199943929356, 'bracket_white': 0.03741437637708494, 'connector': 0.13884210988167356, 'metal_plate': -0.01365612167899847, 'tubes': 0.018941497382728323}
- control gains: {'rotate180': -0.026708285762771393, 'halfroll': -0.041481932208751626}; genuine separation +0.058855
- positive category-shots: 15/18
- gate: PASS — seed1/2 confirmation authorized

- macro_pixel_ap_gain_ge_0005: True
- all_3_shots_positive: True
- positive_category_shots_ge_11: True
- worst_category_gain_ge_minus002: True
- macro_pixel_auroc_loss_ge_minus0002: True
- spatial_control_separation_ge_0003: True