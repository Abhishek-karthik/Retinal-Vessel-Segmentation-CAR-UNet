# FIVES Test Results (all models trained with identical settings)

Metrics are computed per image inside the field of view (threshold 0.5) and averaged over all test images.
With more than one seed, values are mean ± standard deviation across seeds.

## Overall

| Model | Dataset | Seeds | Test images | Parameters | Accuracy | Sensitivity | Specificity | Precision | F1_Dice | IoU | AUC_ROC | AUC_PR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | FIVES | 3 | 200 | 31036481 | 0.9723 ± 0.0005 | 0.8247 ± 0.0078 | 0.9852 ± 0.0008 | 0.8573 ± 0.0047 | 0.8344 ± 0.0034 | 0.7246 ± 0.0050 | 0.9847 ± 0.0008 | 0.9174 ± 0.0024 |
| CAR-UNet | FIVES | 3 | 200 | 32435132 | 0.9728 ± 0.0007 | 0.8310 ± 0.0126 | 0.9855 ± 0.0018 | 0.8557 ± 0.0151 | 0.8382 ± 0.0031 | 0.7295 ± 0.0043 | 0.9860 ± 0.0014 | 0.9202 ± 0.0044 |

## Per disease

| Model | Disease | Images | F1_Dice | Sensitivity | Specificity | AUC_ROC |
|---|---|---|---|---|---|---|
| U-Net | AMD | 50 | 0.8630 | 0.8653 | 0.9840 | 0.9890 |
| CAR-UNet | AMD | 50 | 0.8643 | 0.8710 | 0.9838 | 0.9895 |
| U-Net | DR | 50 | 0.8550 | 0.8475 | 0.9876 | 0.9894 |
| CAR-UNet | DR | 50 | 0.8562 | 0.8544 | 0.9872 | 0.9901 |
| U-Net | Glaucoma | 50 | 0.7887 | 0.7604 | 0.9889 | 0.9761 |
| CAR-UNet | Glaucoma | 50 | 0.7958 | 0.7765 | 0.9885 | 0.9786 |
| U-Net | Normal | 50 | 0.8309 | 0.8257 | 0.9802 | 0.9842 |
| CAR-UNet | Normal | 50 | 0.8363 | 0.8222 | 0.9826 | 0.9860 |

## Is the difference real?

Paired Wilcoxon signed-rank test on per-image Dice (200 test images, averaged over seeds):
- mean Dice CAR-UNet − U-Net = +0.0038
- CAR-UNet better on 154 / 200 images
- p = 2.223e-14 (significant at 0.05)

Per-seed Dice difference (CAR-UNet − U-Net), same seed for both models:
- seed 0: +0.0106, seed 1: -0.0007, seed 2: +0.0014
- mean +0.0038 ± 0.0060; CAR-UNet better in 2 / 3 seeds
- seed-to-seed std of Dice: U-Net 0.0034, CAR-UNet 0.0031

## Individual runs

| model | seed | num_test_images | best_epoch | parameters | epochs | train_minutes | Accuracy | Sensitivity | Specificity | Precision | F1_Dice | IoU | AUC_ROC | AUC_PR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAR-UNet | 0 | 200 | 28 | 32435132 | 30 | 96.3200 | 0.9729 | 0.8411 | 0.9847 | 0.8493 | 0.8411 | 0.7331 | 0.9871 | 0.9233 |
| CAR-UNet | 1 | 200 | 13 | 32435132 | 30 | 89.1500 | 0.9721 | 0.8352 | 0.9843 | 0.8448 | 0.8349 | 0.7248 | 0.9845 | 0.9151 |
| CAR-UNet | 2 | 200 | 22 | 32435132 | 30 | 87.2000 | 0.9735 | 0.8168 | 0.9875 | 0.8729 | 0.8385 | 0.7307 | 0.9865 | 0.9220 |
| U-Net | 0 | 200 | 24 | 31036481 | 30 | 77.9400 | 0.9718 | 0.8166 | 0.9856 | 0.8583 | 0.8305 | 0.7190 | 0.9846 | 0.9160 |
| U-Net | 1 | 200 | 13 | 31036481 | 30 | 73.7200 | 0.9723 | 0.8321 | 0.9843 | 0.8522 | 0.8356 | 0.7262 | 0.9840 | 0.9161 |
| U-Net | 2 | 200 | 22 | 31036481 | 30 | 62.6100 | 0.9728 | 0.8254 | 0.9857 | 0.8614 | 0.8370 | 0.7286 | 0.9855 | 0.9201 |
