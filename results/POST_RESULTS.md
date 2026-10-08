# Improvement 4 (tuned threshold + TTA) and final ablation

All metrics on the 200 FIVES test images, scored at 512x512 inside the FOV; ± = std over seeds.
Thresholds are tuned on the 120 validation images only. `hires_*` = scored against the 1024x1024 ground truth.

## 1. Prediction settings per model

| Model | Setting | Seeds | Threshold | F1_Dice | Sensitivity | Precision | AUC_ROC | clDice | hires_F1_Dice | hires_clDice |
|---|---|---|---|---|---|---|---|---|---|---|
| CAR-UNet | threshold 0.5 | 3 | 0.5000 ± 0.0000 | 0.8382 ± 0.0031 | 0.8310 ± 0.0126 | 0.8557 ± 0.0151 | 0.9860 ± 0.0014 | 0.8626 ± 0.0057 | 0.8572 ± 0.0125 | 0.8791 ± 0.0101 |
| CAR-UNet | tuned threshold | 3 | 0.4800 ± 0.0917 | 0.8387 ± 0.0035 | 0.8343 ± 0.0074 | 0.8530 ± 0.0033 | 0.9860 ± 0.0014 | 0.8632 ± 0.0065 | n/a | n/a |
| CAR-UNet | TTA | 3 | 0.5000 ± 0.0000 | 0.8313 ± 0.0075 | 0.8214 ± 0.0148 | 0.8521 ± 0.0143 | 0.9860 ± 0.0016 | 0.8602 ± 0.0077 | n/a | n/a |
| CAR-UNet | TTA + tuned threshold | 3 | 0.4333 ± 0.0643 | 0.8326 ± 0.0077 | 0.8348 ± 0.0103 | 0.8401 ± 0.0048 | 0.9860 ± 0.0016 | 0.8622 ± 0.0081 | 0.8586 ± 0.0079 | 0.8849 ± 0.0094 |
| CAR-UNet + DA | threshold 0.5 | 3 | 0.5000 ± 0.0000 | 0.8408 ± 0.0021 | 0.8308 ± 0.0005 | 0.8620 ± 0.0049 | 0.9860 ± 0.0001 | 0.8631 ± 0.0020 | 0.8613 ± 0.0015 | 0.8799 ± 0.0027 |
| CAR-UNet + DA | tuned threshold | 3 | 0.4733 ± 0.0115 | 0.8410 ± 0.0021 | 0.8349 ± 0.0022 | 0.8577 ± 0.0042 | 0.9860 ± 0.0001 | 0.8635 ± 0.0019 | n/a | n/a |
| CAR-UNet + DA | TTA | 3 | 0.5000 ± 0.0000 | 0.8334 ± 0.0016 | 0.8214 ± 0.0040 | 0.8560 ± 0.0011 | 0.9860 ± 0.0006 | 0.8612 ± 0.0022 | n/a | n/a |
| CAR-UNet + DA | TTA + tuned threshold | 3 | 0.4400 ± 0.0346 | 0.8342 ± 0.0014 | 0.8338 ± 0.0059 | 0.8444 ± 0.0074 | 0.9860 ± 0.0006 | 0.8626 ± 0.0018 | 0.8620 ± 0.0019 | 0.8857 ± 0.0012 |
| CAR-UNet + DA + clDice | threshold 0.5 | 3 | 0.5000 ± 0.0000 | 0.8382 ± 0.0050 | 0.8432 ± 0.0071 | 0.8430 ± 0.0048 | 0.9866 ± 0.0007 | 0.8718 ± 0.0041 | 0.8599 ± 0.0030 | 0.8895 ± 0.0043 |
| CAR-UNet + DA + clDice | tuned threshold | 3 | 0.5267 ± 0.0115 | 0.8381 ± 0.0051 | 0.8391 ± 0.0061 | 0.8473 ± 0.0068 | 0.9866 ± 0.0007 | 0.8711 ± 0.0043 | n/a | n/a |
| CAR-UNet + DA + clDice | TTA | 3 | 0.5000 ± 0.0000 | 0.8344 ± 0.0024 | 0.8371 ± 0.0053 | 0.8413 ± 0.0014 | 0.9869 ± 0.0006 | 0.8703 ± 0.0034 | n/a | n/a |
| CAR-UNet + DA + clDice | TTA + tuned threshold | 3 | 0.5133 ± 0.0231 | 0.8342 ± 0.0022 | 0.8347 ± 0.0016 | 0.8435 ± 0.0027 | 0.9869 ± 0.0006 | 0.8698 ± 0.0025 | 0.8629 ± 0.0031 | 0.8905 ± 0.0028 |
| CAR-UNet + DA + clDice @1024 | threshold 0.5 | 3 | 0.5000 ± 0.0000 | 0.8421 ± 0.0028 | 0.8326 ± 0.0069 | 0.8632 ± 0.0104 | 0.9876 ± 0.0000 | 0.8742 ± 0.0011 | 0.8796 ± 0.0015 | 0.8972 ± 0.0013 |
| CAR-UNet + DA + clDice @1024 | tuned threshold | 3 | 0.4533 ± 0.0231 | 0.8430 ± 0.0032 | 0.8464 ± 0.0010 | 0.8504 ± 0.0067 | 0.9876 ± 0.0000 | 0.8744 ± 0.0010 | n/a | n/a |
| CAR-UNet + DA + clDice @1024 | TTA | 3 | 0.5000 ± 0.0000 | 0.8403 ± 0.0013 | 0.8289 ± 0.0095 | 0.8635 ± 0.0076 | 0.9880 ± 0.0003 | 0.8744 ± 0.0006 | n/a | n/a |
| CAR-UNet + DA + clDice @1024 | TTA + tuned threshold | 3 | 0.4333 ± 0.0306 | 0.8418 ± 0.0007 | 0.8489 ± 0.0013 | 0.8453 ± 0.0031 | 0.9880 ± 0.0003 | 0.8750 ± 0.0004 | 0.8816 ± 0.0013 | 0.9016 ± 0.0005 |

## 2. Improvement 4 with the setting chosen on VALIDATION (paired Wilcoxon vs threshold 0.5, per-image Dice averaged over seeds)

- **CAR-UNet** (3 seeds; chosen per seed: threshold 0.5 @ 0.50, tuned threshold @ 0.56, tuned threshold @ 0.38): Dice +0.0006 (better on 158/200 images, p = 3.91e-18); clDice +0.0006 (p = 7.99e-17)
- **CAR-UNet + DA** (3 seeds; chosen per seed: tuned threshold @ 0.48, tuned threshold @ 0.46, tuned threshold @ 0.48): Dice +0.0002 (better on 110/200 images, p = 0.0162); clDice +0.0004 (p = 1.2e-11)
- **CAR-UNet + DA + clDice** (3 seeds; chosen per seed: tuned threshold @ 0.52, tuned threshold @ 0.54, tuned threshold @ 0.52): Dice -0.0000 (better on 115/200 images, p = 0.185); clDice -0.0007 (p = 2.96e-28)
- **CAR-UNet + DA + clDice @1024** (3 seeds; chosen per seed: tuned threshold @ 0.44, TTA + tuned threshold @ 0.44, tuned threshold @ 0.48): Dice +0.0015 (better on 148/200 images, p = 5.13e-17); clDice +0.0009 (p = 6.24e-05)

## 3. Ablation: every step of the improvement chain

| Step | Seeds | F1_Dice | Sensitivity | Precision | AUC_ROC | clDice | hires_F1_Dice |
|---|---|---|---|---|---|---|---|
| U-Net | 3 | 0.8344 ± 0.0034 | 0.8247 ± 0.0078 | 0.8573 ± 0.0047 | 0.9847 ± 0.0008 | 0.8593 ± 0.0026 | n/a |
| CAR-UNet | 3 | 0.8382 ± 0.0031 | 0.8310 ± 0.0126 | 0.8557 ± 0.0151 | 0.9860 ± 0.0014 | 0.8626 ± 0.0057 | 0.8572 ± 0.0125 |
| CAR-UNet + Improvement 4 (setting chosen on validation) | 3 | 0.8387 ± 0.0035 | 0.8343 ± 0.0074 | 0.8530 ± 0.0033 | 0.9860 ± 0.0014 | 0.8632 ± 0.0065 | n/a |
| CAR-UNet + DA | 3 | 0.8408 ± 0.0021 | 0.8308 ± 0.0005 | 0.8620 ± 0.0049 | 0.9860 ± 0.0001 | 0.8631 ± 0.0020 | 0.8613 ± 0.0015 |
| CAR-UNet + DA + Improvement 4 (setting chosen on validation) | 3 | 0.8410 ± 0.0021 | 0.8349 ± 0.0022 | 0.8577 ± 0.0042 | 0.9860 ± 0.0001 | 0.8635 ± 0.0019 | n/a |
| CAR-UNet + DA + clDice | 3 | 0.8382 ± 0.0050 | 0.8432 ± 0.0071 | 0.8430 ± 0.0048 | 0.9866 ± 0.0007 | 0.8718 ± 0.0041 | 0.8599 ± 0.0030 |
| CAR-UNet + DA + clDice + Improvement 4 (setting chosen on validation) | 3 | 0.8381 ± 0.0051 | 0.8391 ± 0.0061 | 0.8473 ± 0.0068 | 0.9866 ± 0.0007 | 0.8711 ± 0.0043 | n/a |
| CAR-UNet + DA + clDice @1024 | 3 | 0.8421 ± 0.0028 | 0.8326 ± 0.0069 | 0.8632 ± 0.0104 | 0.9876 ± 0.0000 | 0.8742 ± 0.0011 | 0.8796 ± 0.0015 |
| CAR-UNet + DA + clDice @1024 + Improvement 4 (setting chosen on validation) | 3 | 0.8436 ± 0.0027 | 0.8465 ± 0.0011 | 0.8516 ± 0.0054 | 0.9878 ± 0.0003 | 0.8751 ± 0.0005 | n/a |
