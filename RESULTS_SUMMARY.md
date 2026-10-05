# Results Summary — U-Net vs CAR-UNet on FIVES

All results: **200 FIVES test images** (50 each AMD, DR, Glaucoma, Normal), metrics per image inside the field of view (threshold 0.5), averaged; ± = standard deviation over **3 training seeds**. Both models trained by `scripts/train_fives.py` with identical settings (30 epochs, batch 8, Adam lr 1e-3, cosine schedule, 0.5·BCE + 0.5·Dice). Generated tables: [`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md).

## 1. Overall

| Model | Accuracy | Sensitivity | Specificity | Precision | Dice | IoU | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| U-Net | 0.9723 ± 0.0005 | 0.8247 ± 0.0078 | 0.9852 ± 0.0008 | 0.8573 ± 0.0047 | 0.8344 ± 0.0034 | 0.7246 ± 0.0050 | 0.9847 ± 0.0008 | 0.9174 ± 0.0024 |
| CAR-UNet | 0.9728 ± 0.0007 | 0.8310 ± 0.0126 | 0.9855 ± 0.0018 | 0.8557 ± 0.0151 | 0.8382 ± 0.0031 | 0.7295 ± 0.0043 | 0.9860 ± 0.0014 | 0.9202 ± 0.0044 |

## 2. Per disease (Dice)

| Disease | U-Net | CAR-UNet | Gain |
|---|---|---|---|
| AMD | 0.8630 | 0.8643 | +0.0013 |
| DR | 0.8550 | 0.8562 | +0.0012 |
| Glaucoma | 0.7887 | 0.7958 | +0.0071 |
| Normal | 0.8309 | 0.8363 | +0.0055 |

## 3. Key findings

1. **CAR-UNet is slightly but consistently better per image**: better on 154 / 200 test images; paired Wilcoxon p = 2.2×10⁻¹⁴; mean gain +0.0038 Dice, +0.0063 sensitivity.
2. **The gain is about the size of run-to-run variation**: per-seed Dice differences +0.0106, −0.0007, +0.0014 (CAR-UNet better in 2 of 3 seeds); seed-to-seed std ≈ 0.003 for both models.
3. **Glaucoma is the hardest group** (Dice ≈ 0.79 vs 0.83–0.86) and benefits most from CAR-UNet.
4. **Remaining errors are mostly missed thin capillaries** (blue in the error maps of `outputs/final_comparison_figure.png`).
5. **Two very low-quality Glaucoma images (122_G, 123_G) fail for both models** (Dice < 0.1). They are kept in the test set.
6. **Cost**: CAR-UNet has 4.5% more parameters (32.4 M vs 31.0 M) and needs ~27% more training time (90.9 vs 71.4 min for 30 epochs on an RTX 4050 laptop GPU).

## 4. Figures

- `outputs/final_comparison_figure.png` — one typical (median-Dice) test image per disease: RGB, ground truth, both predictions, both error maps.
- `outputs/training_curves.png` — training loss and validation Dice per epoch, mean ± std over seeds.

## 5. Earlier results

The first FIVES results (U-Net Dice 0.7744, CAR-UNet 0.7916) are superseded: they were measured on only 50 test images (45 Glaucoma), with a validation set lacking DR and Glaucoma images and with different training settings per model. See `results/legacy_v1/README.md` and REVIEW2.md Section 11.
