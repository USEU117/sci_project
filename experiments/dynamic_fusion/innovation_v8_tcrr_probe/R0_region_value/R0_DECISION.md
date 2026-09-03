# TCRR R0 region information-value decision

- regions: 23866 (positive 3925, negative 19941)
- A1 region AP: 0.2135
- best text pooling: text_p90, AP 0.3832, gain +0.1697
- shuffled text AP: 0.1952, genuine-minus-shuffle +0.1880
- positive categories: 5/6; top positive gain share 0.620
- gate: FAIL / ARCHIVE

Gate checks:
- gain_ge_005: True
- positive_categories_ge_4: True
- shuffle_drop_ge_003: True
- top_share_le_050: False
- positive_regions_ge_100: True
- negative_regions_ge_100: True

This R0 result is diagnostic only. A PASS authorizes a minimal region reranker; it is not itself a paper contribution.