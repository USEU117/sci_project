# Wave-2c distribution-level tail probe (CPU, development-only)

- archived frozen rho(r_sub, delta) = 0.3387 (sanity, reproduced)
- frozen risk-feature attribution (risk_pct vs delta, expect negative): {'u_aug': -0.2157, 'u_layer': -0.3228, 'b_tail': 0.0918}
- frozen B_tail unique values per k (k-purity check): {'1': [1.7047], '2': [2.3026], '4': [2.9444]}

V1 (pooled empirical CDF over all pool residuals):
  rho(U_aug' ,delta)=-0.0217   rho(r_sub2',delta)=-0.0683  gate=False  connector_bottom_q=False
V2 (per-pixel Gaussian):
  rho(U_aug'',delta)=-0.1063   rho(B_tail'',delta)=-0.1063   rho(r_sub2'',delta)=0.6245  gate=True  connector_bottom_q=True
  V2 LOCO rho(r_sub2'',delta): {'bracket_black': 0.5443, 'bracket_brown': 0.6327, 'bracket_white': 0.5179, 'connector': 0.5119, 'metal_plate': 0.7197, 'tubes': 0.71}

Artefact audit (see artefact_diagnostics in the json):
  rho(U_aug'', B_tail'') = -0.9257 (anti-correlated features -> composite rank is noise averaging)
  V2 B_tail'' per-category k-monotonicity: {'bracket_black': {'values': [5.5174, 6.6055, 7.5974], 'strictly_increasing_in_k': True}, 'bracket_brown': {'values': [5.5177, 6.7973, 7.4407], 'strictly_increasing_in_k': True}, 'bracket_white': {'values': [5.5192, 6.8193, 7.6125], 'strictly_increasing_in_k': True}, 'connector': {'values': [5.5307, 6.8425, 7.7666], 'strictly_increasing_in_k': True}, 'metal_plate': {'values': [5.5227, 7.1023, 8.1745], 'strictly_increasing_in_k': True}, 'tubes': {'values': [5.3139, 6.3416, 7.1777], 'strictly_increasing_in_k': True}}
  frozen 2-feature (U_aug+B_tail) control rho = 0.2521 (< 0.40; the jump to 0.62 under V2 comes from the calibration change, not real signal)
  V2 flagged(<=q25, 9/18): ['bracket_brown|1', 'bracket_brown|2', 'bracket_white|1', 'bracket_white|2', 'bracket_white|4', 'connector|1', 'connector|2', 'connector|4', 'tubes|2']
  V2 false positives (flagged but delta>=0): ['bracket_brown|1', 'bracket_brown|2', 'tubes|2']
  V2 distinct r_sub2' values = 7; null random-rank rho: {'mean': -0.006, 'std': 0.239, 'max_1000_draws': 0.635, 'pct_ge_06245': 0.001}

VERDICT: distribution-level recalibration does NOT rescue the Wave-2 gate. V1 (true pooled CDF) destroys the ordering (rho=-0.07). V2's apparent pass (rho=0.62, connector bottom-quartile) is an artefact of the clipped/rank transform: B_tail'' stays a pure k-monotone indicator, U_aug'' and B_tail'' anti-correlate at -0.93, r_sub2' collapses onto only 7 distinct values, and the q25 tie boundary swallows 9/18 configs (connector passes only by sitting ON the 0.3889 tie shared with many non-connector configs). The flag set is unusable: it false-flags bracket_brown|1, bracket_brown|2 and tubes|2 (delta +0.03..+0.11, i.e. SUB is BETTER there) at the same score as connector|2 (delta=-0.181). Even random-rank null draws under this clipped transform reach rho as high as 0.635 (max of 1000 draws), so large rho is attainable without any signal. The only frozen feature with genuine risk direction is U_layer (-0.323), whose B/C grids are not persisted on disk. => Wave-2 negative is ROBUST to the z calibration formula; a full GPU recalibration rerun is NOT warranted before user confirmation.

details: W2C_DIST_TAIL_v1.json