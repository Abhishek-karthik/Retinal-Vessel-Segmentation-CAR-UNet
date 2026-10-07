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

---

## 4. Results

All numbers: 200 FIVES test images (50 per disease), scored at 512×512 inside the FOV, mean ± std over training seeds. Full tables: `results/RESULTS_TABLE.md` (training runs) and `results/POST_RESULTS.md` (Improvement 4 + ablation).

> **Status (7 Oct 2026):** Improvements 1, 2 and 4 have 3 seeds. Improvement 3 (1024) has **1 seed so far**; seeds 1 and 2 are in progress, and its rows will be updated when they finish.

### 4.1 Ablation

| Step | Seeds | Dice | Sensitivity | Precision | AUC-ROC | clDice |
|---|---|---|---|---|---|---|
| U-Net (reference) | 3 | 0.8344 ± 0.0034 | 0.8247 | 0.8573 | 0.9847 | 0.8593 |
| **CAR-UNet (paper, our implementation)** | 3 | 0.8382 ± 0.0031 | 0.8310 | 0.8557 | 0.9860 | 0.8626 |
| + Improvement 1: Dual Attention | 3 | **0.8408 ± 0.0021** | 0.8308 | **0.8620** | 0.9860 | 0.8631 |
| + Improvement 2: clDice loss | 3 | 0.8382 ± 0.0050 | **0.8432** | 0.8430 | 0.9866 | **0.8718 ± 0.0041** |
| + Improvement 3: 1024 resolution | 1 | **0.8453** | 0.8272 | **0.8746** | **0.9875** | **0.8751** |
| + Improvement 4: tuned threshold (0.44) | 1 | **0.8467** | 0.8453 | 0.8579 | 0.9875 | **0.8754** |

### 4.2 What each improvement achieved (paired tests on the same 200 test images)

| Improvement | Compared with | Dice | clDice | Sensitivity | Images improved (Dice) |
|---|---|---|---|---|---|
| 1. Dual Attention | CAR-UNet (3 seeds) | **+0.0026** (p = 2×10⁻¹⁹) | +0.0005 | −0.0002 | **167 / 200** |
| 2. clDice loss | + DA (3 seeds) | −0.0026 | **+0.0088** (p = 2×10⁻³²) | **+0.0124** | clDice better on **191 / 200** |
| 3. 1024 resolution | same model at 512 (seed 0) | **+0.0118** (p = 1×10⁻²¹) | **+0.0072** | −0.0082 | **178 / 200** |
| 4. Tuned threshold | same model, threshold 0.5 (seed 0) | +0.0014 (p = 3×10⁻¹³) | +0.0003 | **+0.0181** | 148 / 200 |
| **All four** | **CAR-UNet** (seed 0) | **+0.0056** | **+0.0069** | +0.0042 | – |

**Against the 1024×1024 ground truth** (where thin vessels are visible), the 1024 model reaches Dice **0.8809** / clDice **0.8959**, versus 0.8584 / 0.8876 for the same model trained at 512 and 0.8675 / 0.8900 for CAR-UNet (seed 0) — the clearest evidence that higher resolution recovers thin vessels.

### 4.3 Per disease (Dice; 1024 model: seed 0)

| Disease | CAR-UNet | + DA | + DA + clDice | + 1024 |
|---|---|---|---|---|
| AMD | 0.8643 | **0.8679** | 0.8626 | **0.8701** |
| DR | 0.8562 | **0.8592** | 0.8552 | **0.8591** |
| Glaucoma | **0.7958** | 0.7956 | 0.7947 | **0.7958** |
| Normal | 0.8363 | 0.8405 | 0.8402 | **0.8561** |

clDice per disease improves with Improvement 2 in every group (e.g. Glaucoma 0.8178 → 0.8252, Normal 0.8635 → 0.8767).

---

## 5. Discussion

1. **Dual attention** gives a small but very consistent gain (+0.0026 Dice, better on 167/200 images, and the lowest seed-to-seed variation of all models). It mainly removes false vessel detections (precision +0.006).
2. **clDice** does exactly what it was designed for: vessel connectivity improves on 191 of 200 images and sensitivity rises by 1.2 percentage points (more vessel pixels found, especially thin ones). The price is more false detections at the 0.5 threshold, so Dice stays level — a sensitivity/precision trade-off, not a loss of quality (AUC rises slightly).
3. **Higher resolution** gives the largest gain. Thin vessels that disappear at 512×512 are segmented at 1024×1024, and the 1024 model is also much more precise (+0.03). Training takes ~1.6× longer (197 vs ~123 min).
4. **Improvement 4:** the validation-tuned threshold stays close to 0.5 for the 512 models (they are already well calibrated) and helps the more conservative 1024 model (threshold 0.44: +0.018 sensitivity). **Flip TTA does not help** any model: a diagnostic showed that the models segment flipped images slightly worse (up to ~2% Dice for vertical flips on one seed) despite flip augmentation during training, so averaging the flipped predictions lowers Dice. Because the setting is chosen on the validation set, the final pipeline uses the tuned threshold **without** TTA.
5. **Remaining failure cases:** two very low-quality Glaucoma images (122_G, 123_G) still fail for every model (Dice < 0.1); Glaucoma remains the hardest group overall.

## 6. Final pipeline

**CAR-UNet + Dual Attention + clDice loss, trained at 1024×1024, decision threshold tuned on validation.** On the FIVES test set (seed 0): Dice **0.8467**, sensitivity 0.8453, precision 0.8579, AUC 0.9875, clDice **0.8754** — versus the paper's CAR-UNet at Dice 0.8382 / clDice 0.8626 (3-seed mean) and U-Net at 0.8344 / 0.8593.

Figures: `outputs/final_model_comparison.png` (CAR-UNet vs final model, typical image per disease, error maps), `outputs/thin_vessel_zoom.png` (zoom on the region with most thin vessels), `outputs/ablation_comparison.png`, `outputs/ablation_curves.png`.
