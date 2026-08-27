# P1-B 失败边界：worst/negative categories 与逐图失败样例

## Negative-gain 类别

- 负增益类别在 36 个配置中出现 **65** 次，去重后 **10** 个 dataset@category：`btad@03、mpdd@bracket_black、mpdd@bracket_brown、mpdd@bracket_white、mvtec@capsule、mvtec@grid、mvtec@hazelnut、mvtec@leather、visa@candle、visa@chewinggum`。

## 每 dataset 的 worst category（跨 9 配置 mean ΔAP 最低）

| dataset | worst category | mean ΔAP | negative configs |
|---|---|---:|---:|
| mpdd | bracket_brown | -0.0051 | 9 |
| btad | 03 | +0.0156 | 2 |
| visa | chewinggum | -0.0386 | 9 |
| mvtec | leather | -0.0428 | 9 |

全部类别（按 mean ΔAP 升序）：

| dataset | category | n_configs | mean ΔAP | negative configs |
|---|---|---:|---:|---:|
| mvtec | leather | 9 | -0.0428 | 9 |
| visa | chewinggum | 9 | -0.0386 | 9 |
| mvtec | hazelnut | 9 | -0.0297 | 9 |
| visa | candle | 9 | -0.0198 | 9 |
| mvtec | capsule | 9 | -0.0161 | 4 |
| mvtec | grid | 9 | -0.0062 | 7 |
| mpdd | bracket_brown | 9 | -0.0051 | 9 |
| mpdd | bracket_black | 9 | -0.0035 | 5 |
| mpdd | bracket_white | 9 | +0.0040 | 2 |
| mvtec | bottle | 9 | +0.0107 | 0 |
| mvtec | transistor | 9 | +0.0117 | 0 |
| btad | 03 | 9 | +0.0156 | 2 |
| visa | capsules | 9 | +0.0199 | 0 |
| btad | 01 | 9 | +0.0245 | 0 |
| mpdd | tubes | 9 | +0.0294 | 0 |
| visa | pcb2 | 9 | +0.0335 | 0 |
| btad | 02 | 9 | +0.0346 | 0 |
| mvtec | cable | 9 | +0.0349 | 0 |
| mpdd | connector | 9 | +0.0386 | 0 |
| visa | macaroni1 | 9 | +0.0410 | 0 |
| mvtec | wood | 9 | +0.0422 | 0 |
| mvtec | metal_nut | 9 | +0.0434 | 0 |
| mvtec | zipper | 9 | +0.0487 | 0 |
| mvtec | tile | 9 | +0.0506 | 0 |
| visa | pcb4 | 9 | +0.0550 | 0 |
| visa | pcb3 | 9 | +0.0589 | 0 |
| mvtec | screw | 9 | +0.0675 | 0 |
| visa | pipe_fryum | 9 | +0.0762 | 0 |
| mvtec | pill | 9 | +0.0769 | 0 |
| mvtec | carpet | 9 | +0.0795 | 0 |
| visa | fryum | 9 | +0.0805 | 0 |
| mpdd | metal_plate | 9 | +0.0917 | 0 |
| visa | macaroni2 | 9 | +0.0925 | 0 |
| mvtec | toothbrush | 9 | +0.1080 | 0 |
| visa | cashew | 9 | +0.1145 | 0 |
| visa | pcb1 | 9 | +0.1148 | 0 |

## 逐图失败样例（每配置 top-5，concat per-image AP 相对 dino 最差）

| dataset | seed | shot | category | sample_id | concat AP | dino AP | ΔAP |
|---|---|---|---|---|---|---:|---:|
| mpdd | 0 | 1 | bracket_white | `bracket_white/test/defective_painting/007.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 0 | 1 | bracket_white | `bracket_white/test/defective_painting/008.png` | 0.5833 | 0.8333 | -0.2500 |
| mpdd | 0 | 1 | bracket_black | `bracket_black/test/hole/003.png` | 0.0962 | 0.3138 | -0.2176 |
| mpdd | 0 | 1 | bracket_black | `bracket_black/test/hole/005.png` | 0.5905 | 0.7986 | -0.2081 |
| mpdd | 0 | 1 | bracket_brown | `bracket_brown/test/parts_mismatch/002.png` | 0.0459 | 0.2436 | -0.1977 |
| mpdd | 0 | 2 | bracket_brown | `bracket_brown/test/parts_mismatch/002.png` | 0.0530 | 1.0000 | -0.9470 |
| mpdd | 0 | 2 | bracket_black | `bracket_black/test/scratches/011.png` | 0.0651 | 0.5988 | -0.5337 |
| mpdd | 0 | 2 | bracket_brown | `bracket_brown/test/parts_mismatch/003.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 0 | 2 | bracket_white | `bracket_white/test/defective_painting/007.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 0 | 2 | bracket_brown | `bracket_brown/test/parts_mismatch/007.png` | 0.3209 | 0.6940 | -0.3731 |
| mpdd | 0 | 4 | bracket_white | `bracket_white/test/defective_painting/007.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 0 | 4 | bracket_brown | `bracket_brown/test/parts_mismatch/014.png` | 0.2801 | 0.7055 | -0.4253 |
| mpdd | 0 | 4 | bracket_brown | `bracket_brown/test/parts_mismatch/007.png` | 0.3245 | 0.7050 | -0.3805 |
| mpdd | 0 | 4 | bracket_brown | `bracket_brown/test/bend_and_parts_mismatch/000.png` | 0.2790 | 0.4270 | -0.1480 |
| mpdd | 0 | 4 | bracket_brown | `bracket_brown/test/parts_mismatch/017.png` | 0.0636 | 0.1978 | -0.1342 |
| mpdd | 1 | 1 | bracket_brown | `bracket_brown/test/parts_mismatch/003.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 1 | 1 | bracket_brown | `bracket_brown/test/parts_mismatch/001.png` | 0.4911 | 0.8262 | -0.3351 |
| mpdd | 1 | 1 | bracket_brown | `bracket_brown/test/parts_mismatch/021.png` | 0.2957 | 0.5909 | -0.2952 |
| mpdd | 1 | 1 | bracket_white | `bracket_white/test/scratches/014.png` | 0.4852 | 0.6833 | -0.1981 |
| mpdd | 1 | 1 | bracket_black | `bracket_black/test/scratches/003.png` | 0.5007 | 0.6791 | -0.1784 |
| mpdd | 1 | 2 | bracket_brown | `bracket_brown/test/parts_mismatch/003.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 1 | 2 | bracket_black | `bracket_black/test/scratches/005.png` | 0.0968 | 0.5523 | -0.4555 |
| mpdd | 1 | 2 | bracket_brown | `bracket_brown/test/parts_mismatch/017.png` | 0.1050 | 0.4910 | -0.3861 |
| mpdd | 1 | 2 | bracket_black | `bracket_black/test/scratches/014.png` | 0.3141 | 0.5401 | -0.2260 |
| mpdd | 1 | 2 | bracket_black | `bracket_black/test/scratches/003.png` | 0.3731 | 0.5985 | -0.2254 |
| mpdd | 1 | 4 | bracket_white | `bracket_white/test/scratches/010.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 1 | 4 | bracket_brown | `bracket_brown/test/parts_mismatch/017.png` | 0.1053 | 0.4940 | -0.3887 |
| mpdd | 1 | 4 | bracket_white | `bracket_white/test/defective_painting/008.png` | 0.4500 | 0.8333 | -0.3833 |
| mpdd | 1 | 4 | bracket_brown | `bracket_brown/test/parts_mismatch/014.png` | 0.4549 | 0.7951 | -0.3402 |
| mpdd | 1 | 4 | bracket_white | `bracket_white/test/defective_painting/012.png` | 0.1855 | 0.5156 | -0.3301 |
| mpdd | 2 | 1 | bracket_brown | `bracket_brown/test/parts_mismatch/002.png` | 0.4167 | 1.0000 | -0.5833 |
| mpdd | 2 | 1 | bracket_white | `bracket_white/test/scratches/006.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 2 | 1 | bracket_brown | `bracket_brown/test/parts_mismatch/014.png` | 0.2860 | 0.5648 | -0.2788 |
| mpdd | 2 | 1 | bracket_brown | `bracket_brown/test/parts_mismatch/017.png` | 0.0694 | 0.2965 | -0.2271 |
| mpdd | 2 | 1 | bracket_white | `bracket_white/test/defective_painting/012.png` | 0.0928 | 0.2917 | -0.1989 |
| mpdd | 2 | 2 | bracket_white | `bracket_white/test/scratches/006.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 2 | 2 | bracket_black | `bracket_black/test/scratches/013.png` | 0.0748 | 0.3353 | -0.2605 |
| mpdd | 2 | 2 | bracket_white | `bracket_white/test/defective_painting/012.png` | 0.0970 | 0.2976 | -0.2006 |
| mpdd | 2 | 2 | bracket_black | `bracket_black/test/scratches/012.png` | 0.1794 | 0.3333 | -0.1539 |
| mpdd | 2 | 2 | bracket_brown | `bracket_brown/test/parts_mismatch/015.png` | 0.2291 | 0.3812 | -0.1521 |
| mpdd | 2 | 4 | bracket_white | `bracket_white/test/defective_painting/000.png` | 0.2500 | 1.0000 | -0.7500 |
| mpdd | 2 | 4 | bracket_brown | `bracket_brown/test/parts_mismatch/003.png` | 0.3333 | 1.0000 | -0.6667 |
| mpdd | 2 | 4 | bracket_brown | `bracket_brown/test/parts_mismatch/008.png` | 0.1688 | 0.7361 | -0.5674 |
| mpdd | 2 | 4 | bracket_white | `bracket_white/test/scratches/006.png` | 0.5000 | 1.0000 | -0.5000 |
| mpdd | 2 | 4 | bracket_white | `bracket_white/test/defective_painting/008.png` | 0.4167 | 0.8333 | -0.4167 |
| btad | 0 | 1 | 02 | `02/test/ko/0101.png` | 0.6357 | 0.8965 | -0.2607 |
| btad | 0 | 1 | 03 | `03/test/ko/0012.bmp` | 0.4475 | 0.6924 | -0.2449 |
| btad | 0 | 1 | 03 | `03/test/ko/0017.bmp` | 0.6226 | 0.8573 | -0.2347 |
| btad | 0 | 1 | 02 | `02/test/ko/0152.png` | 0.1613 | 0.3381 | -0.1768 |
| btad | 0 | 1 | 02 | `02/test/ko/0102.png` | 0.2301 | 0.4022 | -0.1721 |
| btad | 0 | 2 | 02 | `02/test/ko/0113.png` | 0.1533 | 0.7253 | -0.5720 |
| btad | 0 | 2 | 03 | `03/test/ko/0012.bmp` | 0.4012 | 0.7425 | -0.3413 |
| btad | 0 | 2 | 03 | `03/test/ko/0017.bmp` | 0.6270 | 0.8731 | -0.2461 |
| btad | 0 | 2 | 02 | `02/test/ko/0111.png` | 0.2088 | 0.3889 | -0.1801 |
| btad | 0 | 2 | 02 | `02/test/ko/0178.png` | 0.3488 | 0.5144 | -0.1656 |
| btad | 0 | 4 | 02 | `02/test/ko/0113.png` | 0.2850 | 0.7487 | -0.4637 |
| btad | 0 | 4 | 03 | `03/test/ko/0012.bmp` | 0.3989 | 0.7465 | -0.3477 |
| btad | 0 | 4 | 02 | `02/test/ko/0183.png` | 0.3277 | 0.6500 | -0.3223 |
| btad | 0 | 4 | 03 | `03/test/ko/0017.bmp` | 0.6399 | 0.8755 | -0.2356 |
| btad | 0 | 4 | 02 | `02/test/ko/0101.png` | 0.2848 | 0.4179 | -0.1330 |
| btad | 1 | 1 | 02 | `02/test/ko/0161.png` | 0.1904 | 0.6324 | -0.4420 |
| btad | 1 | 1 | 02 | `02/test/ko/0111.png` | 0.3611 | 0.6111 | -0.2500 |
| btad | 1 | 1 | 02 | `02/test/ko/0185.png` | 0.5513 | 0.7953 | -0.2440 |
| btad | 1 | 1 | 02 | `02/test/ko/0126.png` | 0.4111 | 0.6377 | -0.2266 |
| btad | 1 | 1 | 02 | `02/test/ko/0099.png` | 0.7036 | 0.9201 | -0.2164 |
| btad | 1 | 2 | 02 | `02/test/ko/0089.png` | 0.2429 | 1.0000 | -0.7571 |
| btad | 1 | 2 | 02 | `02/test/ko/0169.png` | 0.1639 | 0.6270 | -0.4631 |
| btad | 1 | 2 | 02 | `02/test/ko/0161.png` | 0.3798 | 0.7248 | -0.3450 |
| btad | 1 | 2 | 02 | `02/test/ko/0185.png` | 0.4653 | 0.7536 | -0.2882 |
| btad | 1 | 2 | 02 | `02/test/ko/0152.png` | 0.1759 | 0.4389 | -0.2630 |
| btad | 1 | 4 | 02 | `02/test/ko/0185.png` | 0.3805 | 0.6642 | -0.2837 |
| btad | 1 | 4 | 02 | `02/test/ko/0164.png` | 0.3547 | 0.6325 | -0.2778 |
| btad | 1 | 4 | 01 | `01/test/ko/0033.bmp` | 0.0456 | 0.2890 | -0.2434 |
| btad | 1 | 4 | 02 | `02/test/ko/0187.png` | 0.4021 | 0.6315 | -0.2294 |
| btad | 1 | 4 | 02 | `02/test/ko/0152.png` | 0.2214 | 0.4446 | -0.2232 |
| btad | 2 | 1 | 03 | `03/test/ko/0012.bmp` | 0.1940 | 0.7191 | -0.5251 |
| btad | 2 | 1 | 03 | `03/test/ko/0017.bmp` | 0.2954 | 0.8155 | -0.5201 |
| btad | 2 | 1 | 02 | `02/test/ko/0161.png` | 0.4418 | 0.9029 | -0.4611 |
| btad | 2 | 1 | 02 | `02/test/ko/0098.png` | 0.2256 | 0.5623 | -0.3367 |
| btad | 2 | 1 | 02 | `02/test/ko/0134.png` | 0.3228 | 0.6216 | -0.2988 |
| btad | 2 | 2 | 03 | `03/test/ko/0012.bmp` | 0.1735 | 0.7158 | -0.5423 |
| btad | 2 | 2 | 03 | `03/test/ko/0017.bmp` | 0.2882 | 0.8191 | -0.5310 |
| btad | 2 | 2 | 02 | `02/test/ko/0135.png` | 0.3301 | 0.7292 | -0.3990 |
| btad | 2 | 2 | 02 | `02/test/ko/0095.png` | 0.2074 | 0.5466 | -0.3392 |
| btad | 2 | 2 | 02 | `02/test/ko/0164.png` | 0.5909 | 0.9167 | -0.3258 |
| btad | 2 | 4 | 03 | `03/test/ko/0012.bmp` | 0.1974 | 0.7579 | -0.5605 |
| btad | 2 | 4 | 03 | `03/test/ko/0017.bmp` | 0.5358 | 0.8885 | -0.3527 |
| btad | 2 | 4 | 02 | `02/test/ko/0164.png` | 0.6111 | 0.9167 | -0.3056 |
| btad | 2 | 4 | 02 | `02/test/ko/0125.png` | 0.1305 | 0.4008 | -0.2704 |
| btad | 2 | 4 | 02 | `02/test/ko/0115.png` | 0.3975 | 0.6600 | -0.2626 |
| visa | 0 | 1 | pcb1 | `pcb1/Data/Images/Anomaly/015.JPG` | 0.2615 | 0.9167 | -0.6551 |
| visa | 0 | 1 | capsules | `capsules/Data/Images/Anomaly/005.JPG` | 0.0704 | 0.6429 | -0.5725 |
| visa | 0 | 1 | pcb1 | `pcb1/Data/Images/Anomaly/036.JPG` | 0.1799 | 0.7458 | -0.5659 |
| visa | 0 | 1 | pipe_fryum | `pipe_fryum/Data/Images/Anomaly/061.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 0 | 1 | candle | `candle/Data/Images/Anomaly/027.JPG` | 0.1429 | 0.5000 | -0.3571 |
| visa | 0 | 2 | macaroni2 | `macaroni2/Data/Images/Anomaly/075.JPG` | 0.1429 | 1.0000 | -0.8571 |
| visa | 0 | 2 | macaroni2 | `macaroni2/Data/Images/Anomaly/024.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 0 | 2 | macaroni2 | `macaroni2/Data/Images/Anomaly/040.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 0 | 2 | pcb1 | `pcb1/Data/Images/Anomaly/036.JPG` | 0.3663 | 0.7678 | -0.4015 |
| visa | 0 | 2 | macaroni2 | `macaroni2/Data/Images/Anomaly/060.JPG` | 0.1902 | 0.5833 | -0.3932 |
| visa | 0 | 4 | macaroni1 | `macaroni1/Data/Images/Anomaly/016.JPG` | 0.2083 | 0.8333 | -0.6250 |
| visa | 0 | 4 | macaroni1 | `macaroni1/Data/Images/Anomaly/024.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 0 | 4 | macaroni2 | `macaroni2/Data/Images/Anomaly/024.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 0 | 4 | pcb1 | `pcb1/Data/Images/Anomaly/037.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 0 | 4 | macaroni2 | `macaroni2/Data/Images/Anomaly/060.JPG` | 0.1242 | 0.5033 | -0.3791 |
| visa | 1 | 1 | macaroni1 | `macaroni1/Data/Images/Anomaly/024.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 1 | 1 | macaroni1 | `macaroni1/Data/Images/Anomaly/035.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 1 | 1 | capsules | `capsules/Data/Images/Anomaly/003.JPG` | 0.4524 | 0.9111 | -0.4587 |
| visa | 1 | 1 | pcb1 | `pcb1/Data/Images/Anomaly/076.JPG` | 0.6250 | 1.0000 | -0.3750 |
| visa | 1 | 1 | macaroni1 | `macaroni1/Data/Images/Anomaly/092.JPG` | 0.2000 | 0.5000 | -0.3000 |
| visa | 1 | 2 | capsules | `capsules/Data/Images/Anomaly/021.JPG` | 0.4444 | 1.0000 | -0.5556 |
| visa | 1 | 2 | capsules | `capsules/Data/Images/Anomaly/091.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 1 | 2 | macaroni1 | `macaroni1/Data/Images/Anomaly/024.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 1 | 2 | macaroni1 | `macaroni1/Data/Images/Anomaly/035.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 1 | 2 | capsules | `capsules/Data/Images/Anomaly/005.JPG` | 0.3333 | 0.7500 | -0.4167 |
| visa | 1 | 4 | macaroni1 | `macaroni1/Data/Images/Anomaly/035.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 1 | 4 | pcb3 | `pcb3/Data/Images/Anomaly/058.JPG` | 0.0256 | 0.5149 | -0.4894 |
| visa | 1 | 4 | pcb2 | `pcb2/Data/Images/Anomaly/031.JPG` | 0.2552 | 0.7253 | -0.4702 |
| visa | 1 | 4 | macaroni1 | `macaroni1/Data/Images/Anomaly/054.JPG` | 0.0935 | 0.4917 | -0.3982 |
| visa | 1 | 4 | macaroni2 | `macaroni2/Data/Images/Anomaly/053.JPG` | 0.4500 | 0.8333 | -0.3833 |
| visa | 2 | 1 | pcb1 | `pcb1/Data/Images/Anomaly/076.JPG` | 0.3095 | 1.0000 | -0.6905 |
| visa | 2 | 1 | macaroni1 | `macaroni1/Data/Images/Anomaly/054.JPG` | 0.2368 | 0.9167 | -0.6799 |
| visa | 2 | 1 | candle | `candle/Data/Images/Anomaly/055.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 2 | 1 | capsules | `capsules/Data/Images/Anomaly/037.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 2 | 1 | macaroni1 | `macaroni1/Data/Images/Anomaly/024.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 2 | 2 | pcb1 | `pcb1/Data/Images/Anomaly/076.JPG` | 0.2500 | 1.0000 | -0.7500 |
| visa | 2 | 2 | macaroni1 | `macaroni1/Data/Images/Anomaly/054.JPG` | 0.1139 | 0.7193 | -0.6054 |
| visa | 2 | 2 | candle | `candle/Data/Images/Anomaly/055.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 2 | 2 | macaroni1 | `macaroni1/Data/Images/Anomaly/035.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 2 | 2 | macaroni2 | `macaroni2/Data/Images/Anomaly/024.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 2 | 4 | cashew | `cashew/Data/Images/Anomaly/073.JPG` | 0.3333 | 1.0000 | -0.6667 |
| visa | 2 | 4 | macaroni2 | `macaroni2/Data/Images/Anomaly/075.JPG` | 0.3333 | 1.0000 | -0.6667 |
| visa | 2 | 4 | macaroni1 | `macaroni1/Data/Images/Anomaly/054.JPG` | 0.1087 | 0.7051 | -0.5964 |
| visa | 2 | 4 | candle | `candle/Data/Images/Anomaly/055.JPG` | 0.5000 | 1.0000 | -0.5000 |
| visa | 2 | 4 | capsules | `capsules/Data/Images/Anomaly/034.JPG` | 0.5000 | 1.0000 | -0.5000 |
| mvtec | 0 | 1 | leather | `leather/test/poke/016.png` | 0.2345 | 0.9294 | -0.6948 |
| mvtec | 0 | 1 | leather | `leather/test/poke/004.png` | 0.1431 | 0.8357 | -0.6926 |
| mvtec | 0 | 1 | leather | `leather/test/cut/000.png` | 0.0730 | 0.6979 | -0.6249 |
| mvtec | 0 | 1 | leather | `leather/test/poke/007.png` | 0.0998 | 0.6585 | -0.5587 |
| mvtec | 0 | 1 | leather | `leather/test/poke/008.png` | 0.2616 | 0.7910 | -0.5293 |
| mvtec | 0 | 2 | leather | `leather/test/poke/016.png` | 0.1259 | 0.9260 | -0.8002 |
| mvtec | 0 | 2 | leather | `leather/test/poke/012.png` | 0.1525 | 0.9340 | -0.7815 |
| mvtec | 0 | 2 | leather | `leather/test/cut/012.png` | 0.0903 | 0.7380 | -0.6477 |
| mvtec | 0 | 2 | leather | `leather/test/cut/007.png` | 0.3180 | 0.9575 | -0.6395 |
| mvtec | 0 | 2 | leather | `leather/test/cut/003.png` | 0.1882 | 0.7935 | -0.6053 |
| mvtec | 0 | 4 | leather | `leather/test/poke/016.png` | 0.1157 | 0.8860 | -0.7704 |
| mvtec | 0 | 4 | leather | `leather/test/cut/012.png` | 0.0774 | 0.8202 | -0.7428 |
| mvtec | 0 | 4 | leather | `leather/test/cut/007.png` | 0.2551 | 0.9176 | -0.6625 |
| mvtec | 0 | 4 | leather | `leather/test/poke/012.png` | 0.1597 | 0.7701 | -0.6104 |
| mvtec | 0 | 4 | leather | `leather/test/cut/003.png` | 0.1839 | 0.7828 | -0.5989 |
| mvtec | 1 | 1 | capsule | `capsule/test/faulty_imprint/017.png` | 0.3328 | 0.7036 | -0.3708 |
| mvtec | 1 | 1 | leather | `leather/test/cut/016.png` | 0.2484 | 0.6156 | -0.3672 |
| mvtec | 1 | 1 | leather | `leather/test/glue/015.png` | 0.2792 | 0.6425 | -0.3633 |
| mvtec | 1 | 1 | leather | `leather/test/cut/011.png` | 0.1091 | 0.4479 | -0.3388 |
| mvtec | 1 | 1 | capsule | `capsule/test/scratch/009.png` | 0.4186 | 0.7335 | -0.3149 |
| mvtec | 1 | 2 | leather | `leather/test/poke/012.png` | 0.1320 | 0.7674 | -0.6354 |
| mvtec | 1 | 2 | leather | `leather/test/poke/008.png` | 0.2614 | 0.8816 | -0.6203 |
| mvtec | 1 | 2 | leather | `leather/test/poke/006.png` | 0.2583 | 0.8510 | -0.5926 |
| mvtec | 1 | 2 | leather | `leather/test/cut/003.png` | 0.1736 | 0.7256 | -0.5520 |
| mvtec | 1 | 2 | leather | `leather/test/cut/016.png` | 0.4459 | 0.9025 | -0.4566 |
| mvtec | 1 | 4 | leather | `leather/test/poke/012.png` | 0.1320 | 0.7674 | -0.6354 |
| mvtec | 1 | 4 | leather | `leather/test/poke/008.png` | 0.2614 | 0.8816 | -0.6203 |
| mvtec | 1 | 4 | leather | `leather/test/poke/006.png` | 0.2583 | 0.8510 | -0.5926 |
| mvtec | 1 | 4 | leather | `leather/test/cut/007.png` | 0.1537 | 0.6315 | -0.4777 |
| mvtec | 1 | 4 | leather | `leather/test/cut/016.png` | 0.4329 | 0.8356 | -0.4027 |
| mvtec | 2 | 1 | screw | `screw/test/scratch_head/003.png` | 0.2823 | 0.8225 | -0.5402 |
| mvtec | 2 | 1 | grid | `grid/test/metal_contamination/006.png` | 0.4565 | 0.8413 | -0.3847 |
| mvtec | 2 | 1 | leather | `leather/test/poke/009.png` | 0.1702 | 0.5472 | -0.3770 |
| mvtec | 2 | 1 | capsule | `capsule/test/faulty_imprint/017.png` | 0.2292 | 0.5974 | -0.3682 |
| mvtec | 2 | 1 | leather | `leather/test/poke/010.png` | 0.0807 | 0.4148 | -0.3341 |
| mvtec | 2 | 2 | leather | `leather/test/cut/007.png` | 0.2522 | 0.9210 | -0.6688 |
| mvtec | 2 | 2 | leather | `leather/test/cut/005.png` | 0.1177 | 0.7756 | -0.6579 |
| mvtec | 2 | 2 | leather | `leather/test/poke/012.png` | 0.1187 | 0.7479 | -0.6292 |
| mvtec | 2 | 2 | leather | `leather/test/poke/008.png` | 0.1444 | 0.7699 | -0.6254 |
| mvtec | 2 | 2 | leather | `leather/test/cut/003.png` | 0.1629 | 0.7639 | -0.6010 |
| mvtec | 2 | 4 | leather | `leather/test/cut/007.png` | 0.2522 | 0.9602 | -0.7081 |
| mvtec | 2 | 4 | leather | `leather/test/cut/003.png` | 0.1632 | 0.8108 | -0.6475 |
| mvtec | 2 | 4 | leather | `leather/test/poke/012.png` | 0.1187 | 0.7479 | -0.6292 |
| mvtec | 2 | 4 | leather | `leather/test/poke/008.png` | 0.1444 | 0.7699 | -0.6254 |
| mvtec | 2 | 4 | leather | `leather/test/cut/005.png` | 0.1168 | 0.6664 | -0.5497 |

注：失败样例仅从本地合法数据根按 sample_id 追溯原图；包内不复制不可再分发原图。per-image Pixel-AP 只在 mask 含异常像素的测试图上定义（正常图无正像素，不参与）。