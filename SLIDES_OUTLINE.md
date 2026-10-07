# Review Slides — Content Outline

*Slide-by-slide content for the improvement review. Numbers come from `results/RESULTS_TABLE.md` and `results/POST_RESULTS.md`. Rows marked (1 seed) will be updated when the remaining 1024 runs finish.*

---

### Slide 1 — Title
**Improving CAR-UNet for Retinal Vessel Segmentation on the FIVES Dataset**
Base paper: Guo et al., "Channel Attention Residual U-Net for Retinal Vessel Segmentation", IEEE ICASSP 2021
Dataset: FIVES (Jin et al., Scientific Data 2022) — 800 fundus images, 4 disease groups

### Slide 2 — Recap: what we replicated
- CAR-UNet (MECA channel attention + residual blocks + attention on skip connections) on FIVES
- Fair protocol: same training settings for all models, disease-stratified split, all 200 test images, 3 seeds, significance tests
- Starting point: **CAR-UNet Dice 0.8382** vs U-Net 0.8344
- Remaining errors: mostly **missed thin capillaries**; Glaucoma hardest (Dice ≈ 0.80)

### Slide 3 — Weaknesses of the paper (what we set out to fix)
| # | Weakness |
|---|---|
| W1 | Channel attention only — learns *which* features matter, not *where* vessels are |
| W2 | Pixel-wise BCE loss — nothing rewards connected vessels |
| W3 | Designed for small, low-resolution images — thin capillaries lost |
| W4 | Fixed 0.5 threshold, single prediction |
| W5 | Weak evaluation — 20 training images, no Dice / connectivity / per-disease / repeats |

### Slide 4 — Improvement 1: Dual Attention (fixes W1)
- Channel attention from **average + max pooling** (the paper's own MECA equation) + **spatial attention** (CBAM)
- Applied in every residual block and on every skip connection
- Only +1,274 parameters
- **Result (3 seeds): Dice 0.8382 → 0.8408**, better on **167 / 200** images (p = 2×10⁻¹⁹), fewer false vessels

### Slide 5 — Improvement 2: clDice loss (fixes W2)
- Loss = 0.5·BCE + 0.5·(0.5·Dice + 0.5·**clDice**) — clDice compares vessel *skeletons*
- Synthetic test: a broken thin vessel costs **4.5× more** with clDice than with Dice
- **Result (3 seeds): connectivity (clDice) 0.8631 → 0.8718**, better on **191 / 200** images (p = 2×10⁻³²); sensitivity +1.2 points; Dice level

### Slide 6 — Improvement 3: 1024×1024 resolution (fixes W3)
- Train and predict at 1024×1024 (from the 2048×2048 originals) instead of 512×512
- Scored on the **same 512 ground truth** as all other models (fair comparison)
- **Result (1 seed): Dice 0.8335 → 0.8453** vs the same model at 512, better on **178 / 200** images
- Against the 1024 ground truth: **Dice 0.8584 → 0.8809** — thin vessels recovered
- Cost: ~1.6× training time

### Slide 7 — Improvement 4: smarter prediction (fixes W4)
- Threshold tuned on the **validation set only**; flip test-time augmentation (TTA)
- Tuned threshold: small gain (1024 model: threshold 0.44, **Dice 0.8467**, sensitivity +1.8 points)
- **TTA did not help** — diagnostic: models segment flipped images up to ~2% worse, so averaging hurts
- Final choice (made on validation): tuned threshold, no TTA

### Slide 8 — Stronger evaluation (fixes W5)
- 200 test images, 50 per disease; Dice, IoU, sensitivity, specificity, precision, AUC, **clDice**
- 3 seeds per model, paired Wilcoxon tests per image and per seed
- Built-in check that every evaluation reproduces the original numbers

### Slide 9 — Ablation table
| Step | Dice | Sensitivity | Precision | clDice |
|---|---|---|---|---|
| U-Net | 0.8344 | 0.8247 | 0.8573 | 0.8593 |
| CAR-UNet (paper) | 0.8382 | 0.8310 | 0.8557 | 0.8626 |
| + Dual Attention | 0.8408 | 0.8308 | 0.8620 | 0.8631 |
| + clDice loss | 0.8382 | 0.8432 | 0.8430 | 0.8718 |
| + 1024 resolution (1 seed) | 0.8453 | 0.8272 | 0.8746 | 0.8751 |
| + tuned threshold (1 seed) | **0.8467** | 0.8453 | 0.8579 | **0.8754** |

### Slide 10 — Visual results
- `outputs/final_model_comparison.png` — CAR-UNet vs final model, typical image per disease, error maps
- `outputs/thin_vessel_zoom.png` — zoom on the region with most thin vessels
- `outputs/ablation_curves.png` — validation Dice / sensitivity during training

### Slide 11 — Per disease
| Disease | CAR-UNet | Final (1024, 1 seed) |
|---|---|---|
| AMD | 0.8643 | 0.8701 |
| DR | 0.8562 | 0.8591 |
| Glaucoma | 0.7958 | 0.7958 |
| Normal | 0.8363 | 0.8561 |

### Slide 12 — Conclusions & limitations
- Every weakness addressed; final model: **Dice 0.8467, clDice 0.8754** vs paper's CAR-UNet 0.8382 / 0.8626
- Largest gain from higher resolution; clDice gives the best connectivity; dual attention the most consistent gain
- Honest negative result: flip TTA does not help these models
- Limitations: two very low-quality images still fail; Glaucoma remains hardest; 1024 model costs ~1.6× training time
