# FIVES Test Results (all models trained with identical settings)

Metrics are computed per image inside the field of view (threshold 0.5) and averaged over all test images.
With more than one seed, values are mean ± standard deviation across seeds.

## Overall

| Model | Dataset | Seeds | Test images | Parameters | Accuracy | Sensitivity | Specificity | Precision | F1_Dice | IoU | AUC_ROC | AUC_PR | clDice |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | FIVES | 3 | 200 | 31036481 | 0.9723 ± 0.0005 | 0.8247 ± 0.0078 | 0.9852 ± 0.0008 | 0.8573 ± 0.0047 | 0.8344 ± 0.0034 | 0.7246 ± 0.0050 | 0.9847 ± 0.0008 | 0.9174 ± 0.0024 | 0.8593 ± 0.0026 |
| CAR-UNet | FIVES | 3 | 200 | 32435132 | 0.9728 ± 0.0007 | 0.8310 ± 0.0126 | 0.9855 ± 0.0018 | 0.8557 ± 0.0151 | 0.8382 ± 0.0031 | 0.7295 ± 0.0043 | 0.9860 ± 0.0014 | 0.9202 ± 0.0044 | 0.8626 ± 0.0057 |
| CAR-UNet + DA | FIVES | 3 | 200 | 32436406 | 0.9735 ± 0.0005 | 0.8308 ± 0.0005 | 0.9862 ± 0.0005 | 0.8620 ± 0.0049 | 0.8408 ± 0.0021 | 0.7339 ± 0.0034 | 0.9860 ± 0.0001 | 0.9218 ± 0.0012 | 0.8631 ± 0.0020 |
| CAR-UNet + DA + clDice | FIVES | 3 | 200 | 32436406 | 0.9723 ± 0.0009 | 0.8432 ± 0.0071 | 0.9835 ± 0.0004 | 0.8430 ± 0.0048 | 0.8382 ± 0.0050 | 0.7293 ± 0.0077 | 0.9866 ± 0.0007 | 0.9207 ± 0.0032 | 0.8718 ± 0.0041 |
| CAR-UNet + DA + clDice @1024 | FIVES | 1 | 200 | 32436406 | 0.9747 | 0.8272 | 0.9878 | 0.8746 | 0.8453 | 0.7404 | 0.9875 | 0.9263 | 0.8751 |

## Per disease

| Model | Disease | Images | F1_Dice | Sensitivity | Specificity | AUC_ROC | clDice |
|---|---|---|---|---|---|---|---|
| U-Net | AMD | 50 | 0.8630 | 0.8653 | 0.9840 | 0.9890 | 0.8876 |
| CAR-UNet | AMD | 50 | 0.8643 | 0.8710 | 0.9838 | 0.9895 | 0.8884 |
| CAR-UNet + DA | AMD | 50 | 0.8679 | 0.8702 | 0.9849 | 0.9898 | 0.8901 |
| CAR-UNet + DA + clDice | AMD | 50 | 0.8626 | 0.8791 | 0.9820 | 0.9896 | 0.8956 |
| CAR-UNet + DA + clDice @1024 | AMD | 50 | 0.8701 | 0.8637 | 0.9865 | 0.9908 | 0.8989 |
| U-Net | DR | 50 | 0.8550 | 0.8475 | 0.9876 | 0.9894 | 0.8805 |
| CAR-UNet | DR | 50 | 0.8562 | 0.8544 | 0.9872 | 0.9901 | 0.8807 |
| CAR-UNet + DA | DR | 50 | 0.8592 | 0.8522 | 0.9881 | 0.9902 | 0.8819 |
| CAR-UNet + DA + clDice | DR | 50 | 0.8552 | 0.8590 | 0.9863 | 0.9903 | 0.8899 |
| CAR-UNet + DA + clDice @1024 | DR | 50 | 0.8591 | 0.8497 | 0.9885 | 0.9908 | 0.8879 |
| U-Net | Glaucoma | 50 | 0.7887 | 0.7604 | 0.9889 | 0.9761 | 0.8097 |
| CAR-UNet | Glaucoma | 50 | 0.7958 | 0.7764 | 0.9885 | 0.9786 | 0.8178 |
| CAR-UNet + DA | Glaucoma | 50 | 0.7956 | 0.7721 | 0.9892 | 0.9775 | 0.8142 |
| CAR-UNet + DA + clDice | Glaucoma | 50 | 0.7947 | 0.7877 | 0.9868 | 0.9801 | 0.8252 |
| CAR-UNet + DA + clDice @1024 | Glaucoma | 50 | 0.7958 | 0.7577 | 0.9909 | 0.9791 | 0.8236 |
| U-Net | Normal | 50 | 0.8309 | 0.8257 | 0.9802 | 0.9842 | 0.8594 |
| CAR-UNet | Normal | 50 | 0.8363 | 0.8222 | 0.9826 | 0.9860 | 0.8635 |
| CAR-UNet + DA | Normal | 50 | 0.8405 | 0.8286 | 0.9825 | 0.9863 | 0.8662 |
| CAR-UNet + DA + clDice | Normal | 50 | 0.8402 | 0.8471 | 0.9792 | 0.9865 | 0.8767 |
| CAR-UNet + DA + clDice @1024 | Normal | 50 | 0.8561 | 0.8378 | 0.9855 | 0.9895 | 0.8900 |

## Is the difference real?

### CAR-UNet vs U-Net

Paired Wilcoxon signed-rank test on per-image Dice (200 test images, averaged over common seeds [0, 1, 2]):
- mean Dice CAR-UNet − U-Net = +0.0038
- CAR-UNet better on 154 / 200 images
- p = 2.308e-14 (significant at 0.05)
- clDice: mean CAR-UNet − U-Net = +0.0033, better on 114 / 200 images, p = 9.44e-05
- Sensitivity: mean CAR-UNet − U-Net = +0.0063, better on 125 / 200 images, p = 2.045e-06

Per-seed Dice difference (CAR-UNet − U-Net), same seed for both models:
- seed 0: +0.0106, seed 1: -0.0007, seed 2: +0.0014
- mean +0.0038 ± 0.0060; CAR-UNet better in 2 / 3 seeds
- seed-to-seed std of Dice: U-Net 0.0034, CAR-UNet 0.0031

### CAR-UNet + DA vs CAR-UNet

Paired Wilcoxon signed-rank test on per-image Dice (200 test images, averaged over common seeds [0, 1, 2]):
- mean Dice CAR-UNet + DA − CAR-UNet = +0.0026
- CAR-UNet + DA better on 167 / 200 images
- p = 2.136e-19 (significant at 0.05)
- clDice: mean CAR-UNet + DA − CAR-UNet = +0.0005, better on 126 / 200 images, p = 1.662e-05
- Sensitivity: mean CAR-UNet + DA − CAR-UNet = -0.0002, better on 85 / 200 images, p = 0.3468

Per-seed Dice difference (CAR-UNet + DA − CAR-UNet), same seed for both models:
- seed 0: -0.0028, seed 1: +0.0075, seed 2: +0.0031
- mean +0.0026 ± 0.0052; CAR-UNet + DA better in 2 / 3 seeds
- seed-to-seed std of Dice: CAR-UNet 0.0031, CAR-UNet + DA 0.0021

### CAR-UNet + DA + clDice vs CAR-UNet

Paired Wilcoxon signed-rank test on per-image Dice (200 test images, averaged over common seeds [0, 1, 2]):
- mean Dice CAR-UNet + DA + clDice − CAR-UNet = +0.0000
- CAR-UNet + DA + clDice better on 65 / 200 images
- p = 0.01443 (significant at 0.05)
- clDice: mean CAR-UNet + DA + clDice − CAR-UNet = +0.0093, better on 186 / 200 images, p = 8.449e-29
- Sensitivity: mean CAR-UNet + DA + clDice − CAR-UNet = +0.0122, better on 162 / 200 images, p = 6.785e-24

Per-seed Dice difference (CAR-UNet + DA + clDice − CAR-UNet), same seed for both models:
- seed 0: -0.0076, seed 1: +0.0086, seed 2: -0.0009
- mean +0.0000 ± 0.0082; CAR-UNet + DA + clDice better in 1 / 3 seeds
- seed-to-seed std of Dice: CAR-UNet 0.0031, CAR-UNet + DA + clDice 0.0050

### CAR-UNet + DA + clDice vs CAR-UNet + DA

Paired Wilcoxon signed-rank test on per-image Dice (200 test images, averaged over common seeds [0, 1, 2]):
- mean Dice CAR-UNet + DA + clDice − CAR-UNet + DA = -0.0026
- CAR-UNet + DA + clDice better on 40 / 200 images
- p = 1.617e-13 (significant at 0.05)
- clDice: mean CAR-UNet + DA + clDice − CAR-UNet + DA = +0.0088, better on 191 / 200 images, p = 1.602e-32
- Sensitivity: mean CAR-UNet + DA + clDice − CAR-UNet + DA = +0.0124, better on 177 / 200 images, p = 2.054e-27

Per-seed Dice difference (CAR-UNet + DA + clDice − CAR-UNet + DA), same seed for both models:
- seed 0: -0.0049, seed 1: +0.0011, seed 2: -0.0041
- mean -0.0026 ± 0.0032; CAR-UNet + DA + clDice better in 1 / 3 seeds
- seed-to-seed std of Dice: CAR-UNet + DA 0.0021, CAR-UNet + DA + clDice 0.0050

### CAR-UNet + DA + clDice @1024 vs CAR-UNet + DA + clDice

Paired Wilcoxon signed-rank test on per-image Dice (200 test images, averaged over common seeds [0]):
- mean Dice CAR-UNet + DA + clDice @1024 − CAR-UNet + DA + clDice = +0.0118
- CAR-UNet + DA + clDice @1024 better on 178 / 200 images
- p = 1.421e-21 (significant at 0.05)
- clDice: mean CAR-UNet + DA + clDice @1024 − CAR-UNet + DA + clDice = +0.0072, better on 147 / 200 images, p = 7.972e-12
- Sensitivity: mean CAR-UNet + DA + clDice @1024 − CAR-UNet + DA + clDice = -0.0082, better on 52 / 200 images, p = 4.347e-08

Per-seed Dice difference (CAR-UNet + DA + clDice @1024 − CAR-UNet + DA + clDice), same seed for both models:
- seed 0: +0.0118
- mean +0.0118; CAR-UNet + DA + clDice @1024 better in 1 / 1 seeds

### CAR-UNet + DA + clDice @1024 vs CAR-UNet

Paired Wilcoxon signed-rank test on per-image Dice (200 test images, averaged over common seeds [0]):
- mean Dice CAR-UNet + DA + clDice @1024 − CAR-UNet = +0.0041
- CAR-UNet + DA + clDice @1024 better on 147 / 200 images
- p = 3.005e-10 (significant at 0.05)
- clDice: mean CAR-UNet + DA + clDice @1024 − CAR-UNet = +0.0066, better on 155 / 200 images, p = 1.764e-11
- Sensitivity: mean CAR-UNet + DA + clDice @1024 − CAR-UNet = -0.0138, better on 43 / 200 images, p = 6.046e-15

Per-seed Dice difference (CAR-UNet + DA + clDice @1024 − CAR-UNet), same seed for both models:
- seed 0: +0.0041
- mean +0.0041; CAR-UNet + DA + clDice @1024 better in 1 / 1 seeds


## Individual runs

| model | seed | num_test_images | best_epoch | parameters | epochs | train_minutes | Accuracy | Sensitivity | Specificity | Precision | F1_Dice | IoU | AUC_ROC | AUC_PR | clDice |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAR-UNet | 0 | 200 | 28 | 32435132 | 30 | 96.3200 | 0.9729 | 0.8411 | 0.9847 | 0.8493 | 0.8411 | 0.7331 | 0.9871 | 0.9233 | 0.8685 |
| CAR-UNet | 1 | 200 | 13 | 32435132 | 30 | 89.1500 | 0.9721 | 0.8352 | 0.9843 | 0.8448 | 0.8349 | 0.7248 | 0.9845 | 0.9151 | 0.8571 |
| CAR-UNet | 2 | 200 | 22 | 32435132 | 30 | 87.2000 | 0.9735 | 0.8168 | 0.9875 | 0.8729 | 0.8385 | 0.7307 | 0.9865 | 0.9220 | 0.8620 |
| CAR-UNet + DA | 0 | 200 | 24 | 32436406 | 30 | 120.4900 | 0.9729 | 0.8307 | 0.9856 | 0.8563 | 0.8384 | 0.7301 | 0.9860 | 0.9205 | 0.8636 |
| CAR-UNet + DA | 1 | 200 | 20 | 32436406 | 30 | 121.4900 | 0.9738 | 0.8314 | 0.9866 | 0.8649 | 0.8424 | 0.7364 | 0.9860 | 0.9224 | 0.8608 |
| CAR-UNet + DA | 2 | 200 | 17 | 32436406 | 30 | 121.4800 | 0.9737 | 0.8304 | 0.9864 | 0.8647 | 0.8416 | 0.7353 | 0.9858 | 0.9226 | 0.8648 |
| CAR-UNet + DA + clDice | 0 | 200 | 24 | 32436406 | 30 | 123.7600 | 0.9716 | 0.8355 | 0.9835 | 0.8414 | 0.8335 | 0.7224 | 0.9860 | 0.9171 | 0.8679 |
| CAR-UNet + DA + clDice | 1 | 200 | 20 | 32436406 | 30 | 123.6000 | 0.9733 | 0.8495 | 0.9840 | 0.8483 | 0.8435 | 0.7376 | 0.9866 | 0.9230 | 0.8716 |
| CAR-UNet + DA + clDice | 2 | 200 | 27 | 32436406 | 30 | 122.8800 | 0.9720 | 0.8447 | 0.9832 | 0.8392 | 0.8375 | 0.7277 | 0.9873 | 0.9219 | 0.8761 |
| CAR-UNet + DA + clDice @1024 | 0 | 200 | 24 | 32436406 | 30 | 196.8700 | 0.9747 | 0.8272 | 0.9878 | 0.8746 | 0.8453 | 0.7404 | 0.9875 | 0.9263 | 0.8751 |
| U-Net | 0 | 200 | 24 | 31036481 | 30 | 77.9400 | 0.9718 | 0.8166 | 0.9856 | 0.8583 | 0.8305 | 0.7190 | 0.9846 | 0.9160 | 0.8580 |
| U-Net | 1 | 200 | 13 | 31036481 | 30 | 73.7200 | 0.9723 | 0.8321 | 0.9843 | 0.8522 | 0.8356 | 0.7262 | 0.9840 | 0.9161 | 0.8576 |
| U-Net | 2 | 200 | 22 | 31036481 | 30 | 62.6100 | 0.9728 | 0.8254 | 0.9857 | 0.8614 | 0.8370 | 0.7286 | 0.9855 | 0.9201 | 0.8623 |
