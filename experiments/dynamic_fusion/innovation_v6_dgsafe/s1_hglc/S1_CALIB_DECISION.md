# S1-HGLC calibration on A1 maps (doc 16 s.3.3 items 3-5)

z_final = z_A1 + beta * ReLU(p_abn - 0.5) * h(z_A1);  beta grid {0.1,0.25,0.5}; h in {z/(1+z), top-q(0.10)}

- h=z1pz: best beta=0.5  pooled dPixel-AP=+0.0040  worst cat=+0.0002 (tubes)  accept=False
    control shuffled-gate: -0.0005  |  control multiplicative: +0.0044
- h=topq: best beta=0.1  pooled dPixel-AP=-0.0050  worst cat=-0.0371 (metal_plate)  accept=False
    control shuffled-gate: -0.0099  |  control multiplicative: +0.0012

Details: S1_CALIB.json
