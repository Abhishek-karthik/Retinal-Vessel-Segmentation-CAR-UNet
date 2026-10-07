# Improving CAR-UNet for Retinal Vessel Segmentation on FIVES

**Base paper:** Guo et al., *"Channel Attention Residual U-Net for Retinal Vessel Segmentation"*, IEEE ICASSP 2021 ([arXiv:2004.03702](https://arxiv.org/abs/2004.03702))
**Dataset:** FIVES (Jin et al., *Scientific Data*, 2022) — 800 fundus images, AMD / DR / Glaucoma / Normal
**Starting point:** our CAR-UNet implementation on FIVES (see `REVIEW2.md`): Dice 0.8382 ± 0.0031 on all 200 test images, 3 seeds.

This document describes four improvements, each targeting a specific weakness of the paper, and how each one was tested.

---

## 1. Weaknesses of the CAR-UNet paper

| # | Weakness | Evidence |
|---|---|---|
| W1 | **Channel attention only.** MECA learns *which* feature channels matter, but nothing tells the network *where* vessels are. The paper cites CBAM (channel + spatial attention) but uses only the channel part. | Paper §2.2; our error maps: missed thin vessels |
| W2 | **Pixel-wise loss only (BCE).** Vessels are only ~8% of pixels, and nothing in the loss rewards keeping vessels *connected* — breaking a thin vessel costs almost nothing. | Paper §3.2; our error maps: broken capillaries at vessel tips |
| W3 | **Low working resolution.** Designed for small images (DRIVE 565×584). On FIVES we had to downsample 2048×2048 images to 512×512, which erases the finest capillaries. | Our error maps: missed vessels are almost all thin |
| W4 | **Fixed 0.5 decision threshold, single prediction.** No calibration of the threshold, no test-time augmentation. | Paper §3.3 |
| W5 | **Weak evaluation.** Tiny datasets (20 training images), only Accuracy / Sensitivity / Specificity / AUC, a single run, no per-disease analysis, no connectivity measure, no significance tests. | Paper Tables 1–3 |

---

## 2. Improvements

### Improvement 1 — Dual attention (fixes W1)
Every attention module of CAR-UNet (inside all residual blocks and on all skip connections) is replaced by **Dual Attention**:
1. **Channel attention from average- *and* max-pooled descriptors**, passed through a shared 1D convolution and added (the paper's own MECA formulation, Eq. 3);
2. followed by **spatial attention** (Woo et al., CBAM 2018): the channel-wise average and max maps are combined by a 7×7 convolution into a per-pixel weight, so the network learns *where* vessels are.

Cost: +1,274 parameters (32.44 M total, +0.004%); ~39% more time per training step.
Code: `src/car_unet.py` (`attention="dual"`, classes `DualPoolChannelAttention`, `SpatialAttention`, `DualAttention`).

### Improvement 2 — Connectivity-aware loss: BCE + Dice + clDice (fixes W2)
Loss = 0.5·BCE + 0.5·[0.5·Dice + 0.5·clDice], where **clDice** (Shit et al., CVPR 2021) compares the vessel *skeletons* (centrelines) of the prediction and the ground truth using a differentiable soft skeleton. A broken thin vessel removes a whole piece of skeleton, so it is penalised as much as a missing thick vessel.

Sanity test on a synthetic image: cutting a thin vessel (20 pixels) raises the Dice loss by only 0.014 but the clDice loss by 0.062 (4.5×); trimming the end of a thick vessel (40 pixels) costs *less* under clDice (0.024 vs 0.028) because connectivity is preserved.
Code: `src/losses.py` (`SoftClDiceLoss`, `BCEDiceClDiceLoss`).

### Improvement 3 — Higher working resolution: 1024×1024 (fixes W3)
The model is trained and run on 1024×1024 images (created from the 2048×2048 originals, `scripts/resize_fives.py --size 1024`) instead of 512×512. For a fair comparison the 1024 prediction is **downsampled and scored on exactly the same 512×512 ground truth** as every other model; it is additionally scored against the 1024×1024 ground truth (`hires_*` metrics), where thin vessels are better represented. The clDice soft skeleton uses 20 iterations at 1024 (vessels are twice as wide in pixels). Training uses the same number of patches per epoch (equal training compute).
Code: `MODEL_RESOLUTION` in `scripts/train_fives.py`.

### Improvement 4 — Smarter prediction: tuned threshold + test-time augmentation (fixes W4)
No retraining. For each trained model:
- the decision threshold is chosen on the **120 validation images only** (maximum mean Dice over 0.20–0.80), never on the test set;
- **test-time augmentation (TTA):** the prediction is the mean of the predictions on 4 flipped copies of the image (none / horizontal / vertical / both), each flipped back first.
Four settings are reported per model: threshold 0.5, tuned threshold, TTA, TTA + tuned threshold. A built-in check confirms that the "threshold 0.5" setting reproduces the original test results exactly.
Code: `scripts/evaluate_post.py`.

### Stronger evaluation (fixes W5)
Every model is evaluated on all 200 FIVES test images (50 per disease) with Dice, IoU, Sensitivity, Specificity, Precision, AUC-ROC, AUC-PR and **clDice** (connectivity), results per disease, 3 training seeds per model, and paired Wilcoxon tests on the same test images (both per image and per seed).

---

## 3. Experimental protocol

All models share one training script (`scripts/train_fives.py`) and identical settings: disease-stratified split (480 train / 120 validation, 30 per disease), 6 fresh random patches per image per epoch (284→100, flips / rotations / elastic deformation), batch 8, Adam (lr 1e-3, weight decay 1e-5), cosine schedule, 30 epochs, checkpoint = best mean Dice on the 120 full validation images. From 7 Oct 2026 the elastic deformation also uses the seeded random generator, so runs are fully reproducible (before, it was the only unseeded random step; this does not bias comparisons because it was random for all models).

The improvements are added **cumulatively** (ablation): CAR-UNet → + Improvement 1 → + Improvement 2 → + Improvement 3, and Improvement 4 is applied on top of each trained model.
