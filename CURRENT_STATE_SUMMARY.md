# Current Codebase State Summary (Review 1 Baseline)
**Project:** Automated Retinal Blood Vessel Segmentation  
**Baseline Approach:** Ronneberger et al. (2015) U-Net on DRIVE Dataset  
**Target Upgrade:** Channel Attention Residual U-Net (CAR-UNet) on FIVES Dataset  

---

## 1. Preprocessing Pipeline (`src/preprocessing.py`)
The existing pipeline implements classical image enhancement specifically optimized for retinal fundus images:
1. **Green Channel Extraction:** Blood vessels exhibit maximum optical contrast and hemoglobin absorption in the green channel (~540–575 nm) compared to red and blue channels.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Local contrast enhancement with `clip_limit=2.0` and `tile_grid_size=(8, 8)` to highlight thin capillaries while preventing background noise amplification.
3. **Bilateral Filter Noise Reduction:** Edge-preserving smoothing with diameter `d=5`, `sigmaColor=75`, and `sigmaSpace=75` to eliminate high-frequency camera noise while preserving sharp vessel margins.
4. **Intensity Normalization:** Rescaling pixel values from uint8 `[0, 255]` to float32 `[0.0, 1.0]`.

---

## 2. Patch Extraction & Data Augmentation (`src/dataset.py`)
- **Dataset Splitting:** The 20 DRIVE training images are split at the image level into **16 training images** and **4 validation images** to strictly prevent spatial data leakage across patches.
- **Valid Convolution Patch Geometry:**
  - Input patch size: `284 × 284` pixels.
  - Output ground truth mask size: `100 × 100` pixels (centered with a 92-pixel margin on all four sides).
  - Mirror padding (margin = 92 px) is applied to original images to enable valid patch sampling near the retinal borders.
- **FOV-Guided Sampling:** Sampling coordinates are restricted to centers lying within the binary Field of View (FOV) mask.
- **Data Augmentations:**
  - Random horizontal flips ($p=0.5$)
  - Random vertical flips ($p=0.5$)
  - Random 90-degree rotations ($k \in \{0, 1, 2, 3\}$)
  - Elastic deformations ($\alpha=30.0, \sigma=4.0$) using Gaussian displacement fields and spline interpolation as specified in Ronneberger et al. (2015).

---

## 3. U-Net Architecture (`src/unet_model.py`)
- **Framework:** **PyTorch**
- **Convolution Style:** **Valid convolutions (`padding=0`)** matching original Ronneberger et al. (2015).
- **Encoder (Contracting Path):**
  - Layer 1: $1 \to 64$ filters ($284 \to 280$), MaxPool ($280 \to 140$)
  - Layer 2: $64 \to 128$ filters ($140 \to 136$), MaxPool ($136 \to 68$)
  - Layer 3: $128 \to 256$ filters ($68 \to 64$), MaxPool ($64 \to 32$)
  - Layer 4: $256 \to 512$ filters ($32 \to 28$), MaxPool ($28 \to 14$)
- **Bottleneck:**
  - $512 \to 1024$ filters ($14 \to 10$) with Dropout ($p=0.5$)
- **Decoder (Expansive Path):**
  - 4 Transposed Convolutions ($2 \times 2$, stride 2)
  - Center-crop of encoder feature maps to match decoder dimensions for concatenation
  - DoubleConvs reducing channels progressively ($1024 \to 512 \to 256 \to 128 \to 64$)
- **Output Layer:** $1 \times 1$ Conv ($64 \to 1$) followed by Sigmoid activation producing pixel-wise probabilities in $[0, 1]$.

---

## 4. Training Engine & Loss Functions (`src/losses.py`, `scripts/train.py`)
- **Loss Function:** Combined BCE + Dice Loss:
  $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
  where Dice Loss handles the severe class imbalance (vessels occupy only ~10–15% of pixels).
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$).
- **Scheduler:** `CosineAnnealingLR` over the training duration.
- **Model Checkpointing:** Best checkpoint saved based on validation F1/Dice score to `outputs/saved_models/best_unet.pth`.

---

## 5. Inference & Evaluation (`src/utils.py`, `src/metrics.py`, `scripts/evaluate.py`)
- **Overlap-Tile Inference:** Mirror-pads input image and stitches non-overlapping $388 \times 388$ predicted tiles from $572 \times 572$ input patches (or $100 \times 100$ from $284 \times 284$).
- **FOV Masking:** Non-retinal pixels outside the circular FOV mask are masked out ($=0$).
- **Computed Metrics:**
  - Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
  - Sensitivity (Recall): $\frac{TP}{TP + FN}$
  - Specificity: $\frac{TN}{TN + FP}$
  - Precision: $\frac{TP}{TP + FP}$
  - F1 / Dice Score: $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$
  - IoU (Jaccard Index): $\frac{TP}{TP + FP + FN}$
  - AUC-ROC & AUC-PR (Area under ROC and PR curves)
- **Baseline DRIVE Results (Review 1):**
  - Validation Accuracy: **93.99%**
  - Validation Sensitivity: **77.61%**
  - Validation Specificity: **96.28%**
  - Validation F1 / Dice: **75.88%**
  - Validation AUC-ROC: **95.65%**

---

## 6. Framework & Upgrade Compatibility Requirements
- **Framework:** PyTorch (`torch`, `torchvision`).
- **Upgrade Requirements:**
  - All new models (CAR-UNet) must be implemented in PyTorch using the same loss and metric interfaces.
  - Baseline DRIVE pipeline and results must remain untouched.
  - New FIVES dataset pipeline will scale up training data by $\sim 20\times$ (800 fundus images).
