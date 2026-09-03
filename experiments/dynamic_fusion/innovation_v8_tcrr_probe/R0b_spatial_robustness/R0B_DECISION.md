# TCRR R0b spatial robustness decision

- genuine text P90 region AP: 0.3832
- rotate-180 same-image control: 0.2102
- half-roll same-image control: 0.1318
- genuine minus best spatial control: +0.1730
- q=0.95 only gain vs A1: +0.1593 (5/6 cats positive)
- leave-tubes-out gain: +0.0724
- overlap sensitivity gains: 0.01=+0.2305 (6/6 evaluable), 0.1=+0.1463 (6/6 evaluable), 0.25=+0.1361 (5/6 evaluable)
- gate: PASS — minimal R1 authorized

- spatial_drop_ge_003: True
- q095_gain_ge_005: True
- q095_positive_categories_ge_4: True
- leave_tubes_out_gain_ge_005: True
- all_overlap_sensitivity_gain_ge_003: True
- overlap_sensitivity_coverage_ge_5: True