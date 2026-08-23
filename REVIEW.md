# Automated Retinal Blood Vessel Segmentation Using U-Net
### Complete Project Review — Medical Image Processing (Review 1)
### Based on: Ronneberger et al. (2015) — "U-Net: Convolutional Networks for Biomedical Image Segmentation"

---

## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [Dataset: DRIVE](#2-dataset-drive)
   - 2.1 Dataset Description
   - 2.2 Image Characteristics
   - 2.3 Folder Structure & File Formats
   - 2.4 Dataset Splits & Data Leakage Prevention
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
   - 3.1 Green Channel Extraction
   - 3.2 CLAHE — Contrast Limited Adaptive Histogram Equalization
   - 3.3 Bilateral Filter — Edge-Preserving Denoising
   - 3.4 Normalization
4. [Data Augmentation & Elastic Deformation](#4-data-augmentation--elastic-deformation)
   - 4.1 Why Augmentation?
   - 4.2 Flips & Rotations
   - 4.3 Elastic Deformations (Ronneberger et al. 2015)
5. [Patch Extraction Strategy](#5-patch-extraction-strategy)
   - 5.1 Why Patches?
   - 5.2 Valid Convolution Math — Why 284×284 → 100×100?
   - 5.3 FOV-Guided Random Patch Sampling
   - 5.4 Mirror-Padded Borders
6. [U-Net Architecture (Ronneberger et al. 2015)](#6-u-net-architecture-ronneberger-et-al-2015)
   - 6.1 Architecture Overview
   - 6.2 Valid Convolutions (padding=0)
   - 6.3 Contracting Path (Encoder)
   - 6.4 Bottleneck
   - 6.5 Expansive Path (Decoder)
   - 6.6 Skip Connections — Crop & Concatenate
   - 6.7 Output Layer
   - 6.8 Batch Normalization — Design Decision
   - 6.9 Parameter Count
7. [Loss Functions](#7-loss-functions)
   - 7.1 Class Imbalance Problem
   - 7.2 Dice Loss
   - 7.3 Binary Cross Entropy (BCE) Loss
   - 7.4 Combined BCE + Dice Loss
8. [Training Engine](#8-training-engine)
   - 8.1 Optimizer — Adam
   - 8.2 Learning Rate Scheduler — CosineAnnealingLR
   - 8.3 Model Checkpointing
   - 8.4 Training Loop
9. [Inference: Overlap-Tile Strategy](#9-inference-overlap-tile-strategy)
   - 9.1 The Full-Image Inference Problem
   - 9.2 Mirror Padding
   - 9.3 Tile Sliding & Stitching
   - 9.4 FOV Masking
10. [Evaluation Metrics](#10-evaluation-metrics)
    - 10.1 Confusion Matrix Components
    - 10.2 Accuracy, Sensitivity, Specificity, Precision
    - 10.3 F1 / Dice Score
    - 10.4 IoU (Jaccard Index)
    - 10.5 AUC-ROC and AUC-PR
    - 10.6 FOV-Restricted Evaluation
11. [Results](#11-results)
12. [Code Deep Dive](#12-code-deep-dive)
    - 12.1 src/preprocessing.py
    - 12.2 src/dataset.py
    - 12.3 src/unet_model.py
    - 12.4 src/losses.py
    - 12.5 src/metrics.py
    - 12.6 src/utils.py
    - 12.7 scripts/train.py
    - 12.8 scripts/evaluate.py
    - 12.9 scripts/setup_dataset.py
    - 12.10 scripts/test_dataset_and_model.py
13. [Project Folder Structure](#13-project-folder-structure)
14. [Execution Order & Commands](#14-execution-order--commands)
15. [Design Decisions Summary](#15-design-decisions-summary)
16. [Output Artifacts](#16-output-artifacts)

---

## 1. Problem Statement & Motivation

**Retinal blood vessel segmentation** is the task of automatically identifying and delineating the vascular network inside a retinal fundus image. This is a fundamental step in the computer-aided diagnosis (CAD) of ophthalmic diseases including:

- **Diabetic Retinopathy** — abnormal vessel growth (neovascularization)
- **Glaucoma** — changes in disc-vessel ratio
- **Hypertension** — arteriovenous nicking and vessel diameter changes
- **Age-related Macular Degeneration (AMD)** — vessel abnormalities near the macula

**The core challenge** is that blood vessels are very thin, tortuous, and occupy only ~10–15% of the total retinal image area. This extreme class imbalance (vessels vs. background), combined with noise, illumination variation, and structures like the optic disc and lesions, makes accurate segmentation non-trivial.

**Our approach** implements the original **U-Net (Ronneberger et al., 2015)** — a fully convolutional encoder-decoder network designed specifically for biomedical image segmentation with limited training data. The key innovations of this paper are:

1. Symmetric encoder-decoder with skip connections
2. Valid convolutions (no padding) for exact spatial correspondence
3. Overlap-tile inference strategy for arbitrarily large images
4. Elastic deformation augmentation to overcome limited labeled data

---

## 2. Dataset: DRIVE

### 2.1 Dataset Description

**DRIVE** (Digital Retinal Images for Vessel Extraction) is the standard benchmark dataset for retinal vessel segmentation. It was introduced by Staal et al. (2004) and is published at: https://drive.grand-challenge.org/

**Key facts:**
- 40 fundus photographs from a diabetic retinopathy screening program in the Netherlands
- 20 training images with expert ground truth vessel annotations
- 20 test images (held out — no public ground truth)
- Images were acquired using a Canon CR5 camera at 45-degree Field of View (FOV)

### 2.2 Image Characteristics

| Property | Detail |
|---|---|
| Image Size | 565 × 584 pixels |
| Color Space | RGB (24-bit) |
| File Format | .tif (TIFF) |
| Field of View | ~45 degrees circular FOV |
| Vessel Coverage | ~10–15% of pixels inside FOV |
| FOV mask format | .gif (binary circular mask) |
| Ground Truth format | .gif (binary vessel annotation) |

The **FOV mask** is critical. The fundus image contains a circular retinal region embedded in a black square frame. All evaluation metrics must be restricted to pixels **inside this circular FOV** to avoid biasing metrics by the trivially-black background surrounding the circle.

### 2.3 Folder Structure & File Formats

```
data/DRIVE/
├── training/
│   ├── images/          20 fundus .tif images (21_training.tif to 40_training.tif)
│   ├── 1st_manual/      20 binary vessel GT annotations (.gif)
│   └── mask/            20 binary circular FOV masks (.gif)
└── test/
    ├── images/          20 fundus .tif images (01_test.tif to 20_test.tif)
    └── mask/            20 binary circular FOV masks (.gif)
```

> **Important**: The test/ folder does NOT have a 1st_manual/ directory. Ground truth annotations for the test set are withheld by the benchmark. We treat test/mask purely as the Field of View mask.

### 2.4 Dataset Splits & Data Leakage Prevention

We perform an **image-level train/validation split** on the 20 labeled training images:

| Split | Images | Count |
|---|---|---|
| Training | 21_training.tif to 36_training.tif | 16 images |
| Validation | 37_training.tif to 40_training.tif | 4 images |
| Test (held-out) | 01_test.tif to 20_test.tif | 20 images |

**Why image-level split matters (Data Leakage):**

Patches extracted from the same image are spatially correlated — they share similar vessel patterns, illumination, and noise characteristics. If we extracted patches from all 20 images and **then** randomly split into train/val, patches from the same image could appear in both training and validation sets. This is **data leakage** — the model would have effectively "seen" the validation images during training, producing falsely optimistic metrics.

By splitting at the **image level** first, we guarantee:
- Validation set images are completely unseen during training
- No spatial correlation between train and val samples
- Metrics on the val set are a true estimate of generalization

---

## 3. Preprocessing Pipeline

Raw retinal images are RGB photographs taken with a fundus camera. The classical preprocessing pipeline transforms raw images into a normalized, enhanced grayscale representation suitable for neural network input.

**Full pipeline:** RGB Input → Green Channel → CLAHE → Bilateral Filter → Normalize [0,1]

### 3.1 Green Channel Extraction

**Why the green channel?**

Retinal blood vessels appear **darkest** in the green channel of the RGB fundus image. This is due to the optical absorption properties of hemoglobin:
- Hemoglobin has peak absorption at ~540–575 nm (green light)
- This creates maximum vessel-to-background contrast in the green channel
- The red and blue channels carry less vessel-specific information and more illumination noise

```python
def extract_green_channel(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image[:, :, 1]  # Index 1 = Green channel in RGB
    return image
```

Returns a 2D array (H, W) of uint8 pixel values from the green channel.

### 3.2 CLAHE — Contrast Limited Adaptive Histogram Equalization

**Standard Histogram Equalization** redistributes pixel intensity values to achieve a flat histogram. The problem with global HE on retinal images: it can over-amplify noise in dark regions while over-saturating bright regions.

**Adaptive HE (AHE)** divides the image into small tiles and computes separate histograms per tile. However, in uniform regions, AHE can amplify noise dramatically.

**CLAHE** adds a **clip limit** to AHE. The histogram is clipped at the clip limit before redistribution — histogram counts above the limit are redistributed uniformly across all intensity levels. This prevents excessive noise amplification while enhancing local contrast.

**Parameters used:**
- `clip_limit = 2.0` — Maximum slope of the CDF. Higher = more contrast boost (but more noise)
- `tile_grid_size = (8, 8)` — Image divided into 8×8 = 64 tiles for local processing

**Why this matters for vessels:** Thin capillaries can be invisible in raw images due to poor local contrast. CLAHE enhances the local contrast specifically where vessels are present.

```python
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)  # Input must be uint8 single-channel
```

### 3.3 Bilateral Filter — Edge-Preserving Denoising

**Why not just Gaussian blur?** Standard Gaussian blur reduces noise but blurs vessel boundaries, reducing sharpness of thin vessels.

**Bilateral Filter** is an **edge-preserving** smoothing filter that considers two similarity measures:
1. **Spatial proximity** (like Gaussian) — nearby pixels have more weight
2. **Intensity similarity** — pixels with similar intensity have more weight

At an edge (strong intensity discontinuity), pixels across the edge get very low weights due to the intensity distance term. The filter reduces noise within uniform regions while preserving sharp edges.

**Parameters:**
- `d = 5` — neighborhood diameter
- `sigmaColor = 75` — pixels with intensity difference > 75 get very low weight
- `sigmaSpace = 75` — spatial Gaussian sigma

```python
return cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
```

### 3.4 Normalization

```python
def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0
```

Converts uint8 [0, 255] to float32 [0.0, 1.0]. Essential for stable neural network gradient propagation.

---

## 4. Data Augmentation & Elastic Deformation

### 4.1 Why Augmentation?

The DRIVE training set has only 20 labeled images. A 31M parameter network can easily overfit to this. Data augmentation artificially increases the effective size and diversity by applying random transformations.

**Critical rule:** Transformations must be applied **identically** to both the input image patch and the corresponding ground truth mask patch.

### 4.2 Flips & Rotations

| Augmentation | Trigger | Code |
|---|---|---|
| Horizontal Flip | prob > 0.5 | `np.fliplr(img_p)`, `np.fliplr(mask_p)` |
| Vertical Flip | prob > 0.5 | `np.flipud(img_p)`, `np.flipud(mask_p)` |
| 90° Rotations (k=1,2,3) | Random k ∈ {0,1,2,3} | `np.rot90(img_p, k)`, `np.rot90(mask_p, k)` |

These augmentations are **label-preserving** — a horizontally flipped vessel is still a vessel. They expose the model to all orientations of vessel branching patterns.

### 4.3 Elastic Deformations (Ronneberger et al. 2015)

The paper states: "For microscopic images, deformations in the specimen are the most common variation and realistic deformations can be simulated efficiently."

**Elastic deformation algorithm:**

1. **Generate random displacement fields** `dx` and `dy`:
   - Draw uniform random values in [-1, 1] at image resolution
   - Apply Gaussian smoothing (sigma) to make spatially correlated and smooth
   - Scale by alpha to control displacement magnitude

2. **Create coordinate grids** for original image positions

3. **Add displacement to coordinates:** `new_x = x + dx`, `new_y = y + dy`

4. **Interpolate pixel values** at new (fractional) coordinates using `map_coordinates`:
   - `order=1` (bilinear interpolation) for the image — smooth interpolation
   - `order=0` (nearest-neighbor) for the mask — preserves binary labels

**Parameters:**
- `alpha = 30.0` — max displacement magnitude (~30 pixels)
- `sigma = 4.0` — smoothness of displacement (local, not global deformations)

**Critical implementation detail — shape mismatch:**

Input patch is (284, 284) but target mask is (100, 100). Displacement field is generated at (284, 284). For the mask, we use a center-sliced region of the displacement field:

```python
my = (img_shape[0] - mask_shape[0]) // 2  # = 92
mx = (img_shape[1] - mask_shape[1]) // 2  # = 92
dx_mask = dx[my : my + mask_shape[0], mx : mx + mask_shape[1]]
dy_mask = dy[my : my + mask_shape[0], mx : mx + mask_shape[1]]
```

This ensures mask is deformed by the **same spatial field** as the center region of the input patch, maintaining perfect label correspondence.

---

## 5. Patch Extraction Strategy

### 5.1 Why Patches?

Full DRIVE images are 565×584 pixels. A full-resolution 31M-parameter U-Net would require enormous memory. Patch-based training extracts smaller regions, trains on batches of those, and uses Overlap-Tile inference to reconstruct full-image predictions.

### 5.2 Valid Convolution Math — Why 284×284 → 100×100?

This is the most important geometric detail in the implementation.

**Each 3×3 convolution with padding=0** reduces spatial dimensions by 2 (1 pixel lost per side).

**Each DoubleConv block** applies two such convolutions: reduces by **4 pixels total**.

**Network structure:**
- 5 DoubleConv blocks in encoder + bottleneck × 4px = 20px
- 4 DoubleConv blocks in decoder × 4px = 16px
- **Total reduction = 36px on each side = 184px diameter**

**Therefore:**
```
Output size = Input size - 184
For 284x284 input: output = 284 - 184 = 100x100 ✓
For 572x572 input: output = 572 - 184 = 388x388 ✓ (original paper)
```

**Margin calculation:**
- `margin = (284 - 100) / 2 = 92 pixels`

The U-Net needs 92 pixels of context around each output region. The 100×100 output represents the **center region** of the 284×284 input.

### 5.3 FOV-Guided Random Patch Sampling

Not all locations are equally informative. We bias toward vessel-rich FOV regions:

```python
# Find all valid FOV pixel coordinates
valid_y, valid_x = np.where(padded_fov[margin:h-margin, margin:w-margin] > 0)
valid_y += margin; valid_x += margin  # Shift to padded coordinates

# Sample patch centers
indices = np.random.choice(len(valid_y), size=num_patches, replace=True)
ys = valid_y[indices] - (in_patch_size // 2)  # Top-left of 284x284 window
xs = valid_x[indices] - (in_patch_size // 2)
ys = np.clip(ys, 0, h_pad - in_patch_size)    # Keep within bounds
```

This concentrates training patches on the clinically relevant retinal region.

### 5.4 Mirror-Padded Borders

Before patch extraction, the image is mirror-padded by margin=92 pixels:

```python
padded_image = np.pad(image, ((margin, margin), (margin, margin)), mode='reflect')
padded_mask  = np.pad(mask,  ((margin, margin), (margin, margin)), mode='reflect')
padded_fov   = np.pad(fov,   ((margin, margin), (margin, margin)), mode='constant', constant_values=0)
```

**reflect** mirrors pixel values at the boundary — more natural than zero-padding, consistent with inference mirror-padding. FOV uses `constant=0` because we don't assume FOV continues beyond image boundary.

---

## 6. U-Net Architecture (Ronneberger et al. 2015)

### 6.1 Architecture Overview

The U-Net is a fully convolutional encoder-decoder shaped like the letter "U":

```
Input (284x284)
    |
[enc1: 1→64 ch, 284→280]  ────────────────────────────── crop → cat → [dec1: 128→64, 104→100]
    | pool (280→140)                                                            | up (52→104)
[enc2: 64→128, 140→136]  ─────────────────────────── crop → cat → [dec2: 256→128, 200→196]
    | pool (136→68)                                                             | up (100→200)
[enc3: 128→256, 68→64]  ─────────────────────── crop → cat → [dec3: 512→256, 104→100]
    | pool (64→32)                                                              | up (52→104)
[enc4: 256→512, 32→28]  ─────────── crop → cat → [dec4: 1024→512, 56→52]
    | pool (28→14)                                        | up (28→56)
    |                                                     |
[bottleneck: 512→1024, 14→10] + Dropout(0.5)
    |
    └──────────────────────────────────────────────────────────────────┘
                                                           |
                                                   [out 1x1 conv] → sigmoid
                                                   Output (100x100)
```

### 6.2 Valid Convolutions (padding=0)

**Same padding (padding=1):** Pads with zeros before each conv to maintain spatial dimensions. Output pixels at the border are computed partly from artificial zeros.

**Valid padding (padding=0):** No padding added. Only computes outputs where the filter **fully overlaps** with real input data. Border pixels of the output are guaranteed to be based entirely on real input.

**Consequence:** Input must be larger than output. The network "consumes" 92 pixels of context per side.

### 6.3 Contracting Path (Encoder)

Each encoder level:
1. `DoubleConv`: two Conv2d(3×3, padding=0) → BatchNorm → ReLU blocks
2. `MaxPool2d(2, 2)`: downsamples spatial dimensions by 2

| Layer | Channels | Size (input 284) |
|---|---|---|
| enc1 (DoubleConv) | 1→64 | 284→280 |
| pool1 | 64 | 280→140 |
| enc2 (DoubleConv) | 64→128 | 140→136 |
| pool2 | 128 | 136→68 |
| enc3 (DoubleConv) | 128→256 | 68→64 |
| pool3 | 256 | 64→32 |
| enc4 (DoubleConv) | 256→512 | 32→28 |
| pool4 | 512 | 28→14 |
| bottleneck | 512→1024 | 14→10 |

**MaxPooling**: Selects maximum activation in each 2×2 region. Achieves translation invariance, increases receptive field, and enables hierarchical feature learning.

**DoubleConv code:**
```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

- `bias=False`: BatchNorm already has learnable bias term (beta), making Conv bias redundant
- `ReLU(inplace=True)`: modifies tensor in-place to save memory allocation

### 6.4 Bottleneck

```python
self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)  # 512→1024
self.dropout = nn.Dropout(0.5)
b = self.dropout(self.bottleneck(self.pool4(e4)))
```

- Maximum abstraction at minimum spatial resolution (10×10 with 1024 channels)
- **Dropout(0.5)**: randomly zeros 50% of activations during training
  - Prevents co-adaptation of neurons
  - Acts as ensemble of different sub-networks
  - Improves generalization, critical with only 16 training images
  - Automatically disabled during `model.eval()`

### 6.5 Expansive Path (Decoder)

Each decoder level:
1. `ConvTranspose2d(2×2, stride=2)` — learned upsampling (doubles spatial dimensions, halves channels)
2. Crop + concatenate skip connection from corresponding encoder level
3. `DoubleConv` — refines features at this resolution

**Transposed Convolution** (ConvTranspose2d): Learnable upsampling, unlike bilinear which has no parameters. The 2×2 kernel with stride=2 doubles spatial dimensions through the inverse operation of convolution.

| Layer | Channels In | Size In | Size Out |
|---|---|---|---|
| up4 (ConvT) + cat(e4_cropped) | 1024+512=1536 | 10→20 | 20 |
| dec4 (DoubleConv) | 1024→512 | 20 | 16 |
| up3 (ConvT) + cat(e3_cropped) | 512+256=768 | 16→32 | 32 |
| dec3 (DoubleConv) | 512→256 | 32 | 28 |
| up2 (ConvT) + cat(e2_cropped) | 256+128=384 | 28→56 | 56 |
| dec2 (DoubleConv) | 256→128 | 56 | 52 |
| up1 (ConvT) + cat(e1_cropped) | 128+64=192 | 52→104 | 104 |
| dec1 (DoubleConv) | 128→64 | 104 | 100 |
| out_conv (1×1) | 64→1 | 100 | 100 |

### 6.6 Skip Connections — Crop & Concatenate

**Why skip connections?** During downsampling, spatial resolution is reduced 16×. The decoder must recover fine vessel details. Without skip connections, only coarse bottleneck information is available — precise thin-vessel localization is impossible.

**The shape mismatch problem with valid convolutions:**

Encoder feature maps are **larger** than corresponding decoder feature maps after upsampling, because valid convolutions on both sides cause more shrinkage in the encoder than is recovered by upsampling.

**Solution: `crop_tensor()` — center crop**

```python
def crop_tensor(enc_tensor, target_tensor):
    _, _, H_enc, W_enc = enc_tensor.shape
    _, _, H_tgt, W_tgt = target_tensor.shape
    delta_H = (H_enc - H_tgt) // 2
    delta_W = (W_enc - W_tgt) // 2
    return enc_tensor[:, :, delta_H : delta_H + H_tgt, delta_W : delta_W + W_tgt]
```

Discards the outer border of the encoder tensor symmetrically to match decoder spatial dimensions.

**Skip connection in forward pass:**
```python
u4 = self.up4(b)                              # Upsample bottleneck
e4_cropped = crop_tensor(e4, u4)              # Center-crop encoder feature map
d4 = self.dec4(torch.cat([u4, e4_cropped], dim=1))  # Concatenate channel-wise
```

`torch.cat(..., dim=1)` concatenates along the channel dimension, doubling channel count. The subsequent DoubleConv processes this to refine and reduce.

### 6.7 Output Layer

```python
self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
self.sigmoid = nn.Sigmoid()
```

- **1×1 convolution**: per-pixel linear projection from 64 channels to 1 channel (binary logit). Does not change spatial dimensions.
- **Sigmoid**: maps logits to probabilities in [0, 1]. Each pixel output = P(vessel | pixel).

### 6.8 Batch Normalization — Design Decision

The original Ronneberger et al. (2015) paper **did not use Batch Normalization**. We retained it as a deliberate design enhancement.

**Batch Normalization:** For each mini-batch, normalizes the input to each layer to have zero mean and unit variance using batch statistics, then applies learned scale (gamma) and shift (beta):

```
y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

**Why we kept it:**

1. **Training stability on CPU**: Without BatchNorm, training a 31M-parameter network from a small dataset with lr=1e-3 risks exploding/vanishing gradients. BatchNorm normalizes activations, keeping them in a stable range.
2. **Faster convergence**: Allows higher learning rates by reducing internal covariate shift.
3. **Implicit regularization**: Batch statistics differ slightly from population statistics during training — adds controlled noise.
4. **Not architecturally significant**: BatchNorm doesn't change the U-Net's fundamental encoder-decoder structure, skip connections, valid convolutions, or inference strategy.

During `model.eval()`, PyTorch BatchNorm uses accumulated running mean/variance (not batch statistics).

### 6.9 Parameter Count

With `base_filters=64` (31,036,481 total trainable parameters):

| Component | Approx Parameters |
|---|---|
| Encoder (enc1–enc4) | ~6.3M |
| Bottleneck | ~18.9M |
| Decoder (dec1–dec4) | ~5.8M |
| ConvTranspose layers | ~0.04M |
| Output 1×1 | ~65 |
| **Total** | **~31.0M** |

The bottleneck dominates: 512→1024 channels at 14×14 spatial = two DoubleConv blocks with 1024 channels each.

---

## 7. Loss Functions

### 7.1 Class Imbalance Problem

In retinal images, blood vessels occupy approximately **10–15% of FOV pixels**. The remaining ~85–90% is background. With standard BCE alone, the model can achieve ~87% accuracy by predicting "background" for every pixel — a trivial degenerate solution. We need a loss that forces the model to attend to the minority class.

### 7.2 Dice Loss

The **Dice coefficient** measures overlap between predicted and ground-truth binary segmentations:

```
Dice = 2 * |P ∩ G| / (|P| + |G|) = 2 * TP / (2*TP + FP + FN)
Dice Loss = 1 - Dice
```

**Why Dice handles imbalance:** If the model predicts all background (all zeros), the intersection with sparse ground truth is ~0, giving Dice Loss ≈ 1 — maximum loss. The model is forced to detect vessels.

**Soft Dice** (used in training):
```python
pred_flat = pred.view(-1)      # Continuous probabilities in [0,1]
target_flat = target.view(-1)  # Binary ground truth
intersection = (pred_flat * target_flat).sum()  # Soft intersection
dice = (2.0 * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
return 1.0 - dice
```

Continuous predictions allow proper gradient flow (gradients of binary tensors would be ~0 everywhere). The `smooth=1.0` Laplace term prevents division by zero when both are empty.

### 7.3 Binary Cross Entropy (BCE) Loss

```
BCE = -(y * log(p) + (1-y) * log(1-p))
```

where y is ground truth (0 or 1) and p is predicted probability. BCE provides strong per-pixel gradient signal but doesn't directly optimize for overlap. Used as a complementary term.

### 7.4 Combined BCE + Dice Loss

```python
class CombinedBCEDiceLoss(nn.Module):
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)    # Pixel-level accuracy
        dice_loss = self.dice(pred, target)  # Overlap quality
        return 0.5 * bce_loss + 0.5 * dice_loss
```

- **BCE**: stable per-pixel gradient signal (good for optimizer)
- **Dice**: directly optimizes evaluation metric (handles imbalance)
- **50-50 weighting**: common default effective for vessel segmentation

Note: `nn.BCELoss()` requires post-sigmoid probabilities. Our model applies sigmoid before output, which is correct.

---

## 8. Training Engine

### 8.1 Optimizer — Adam

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

**Adam (Adaptive Moment Estimation)** maintains per-parameter adaptive learning rates using first and second moment estimates of gradients. Adapts learning rate for each weight based on its gradient history. Generally robust and good default for medical imaging.

`weight_decay=1e-5`: L2 regularization that adds a penalty proportional to parameter magnitudes (prevents weights from growing too large).

### 8.2 Learning Rate Scheduler — CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
```

The learning rate follows a cosine curve from initial LR down toward 0 over T_max epochs:
```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))
```

- Early epochs: high LR → fast convergence
- Late epochs: low LR → fine-grained adjustment, avoids oscillation near minima
- Smoother than StepLR (no sudden drops)

### 8.3 Model Checkpointing

```python
if val_dice > best_val_dice:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "history": history
    }, checkpoint_path)
```

Only the **best** model (by validation Dice) is saved. Prevents saving overfitted later-epoch models. Full state saved enables training resumption.

### 8.4 Training Loop

**Train epoch** (model.train() mode):
```python
optimizer.zero_grad()    # Clear accumulated gradients
preds = model(imgs)      # Forward pass: (B, 1, 100, 100)
loss = criterion(preds, masks)
loss.backward()          # Backprop: compute gradients
optimizer.step()         # Update weights
```

**Val epoch** (model.eval() mode):
```python
with torch.no_grad():    # No gradient computation (saves memory + time)
    preds = model(imgs)
    loss = criterion(preds, masks)
```

`model.train()` and `model.eval()` switch BatchNorm (batch stats vs. running stats) and Dropout (active vs. disabled).

**Defaults:**
- epochs = 5, batch_size = 4, lr = 1e-3
- in_patch_size = 284, patches_per_img = 10, base_filters = 64
- Total training samples: 16 images × 10 patches = 160 patches
- Training time: ~13 minutes on CPU for 5 epochs

---

## 9. Inference: Overlap-Tile Strategy

### 9.1 The Full-Image Inference Problem

Our model accepts 284×284 input and produces 100×100 output. A full DRIVE image is 565×584 — much larger than a single input patch.

**Naive approach (wrong):** Resize full image to 284×284 → run inference → resize output back. Discards spatial resolution and fine vessel detail.

**Correct: Overlap-Tile Strategy (Ronneberger 2015)** — tile the full image with non-overlapping 100×100 output tiles, each predicted from a surrounding 284×284 input tile (92px context on all sides).

### 9.2 Mirror Padding

For output pixels near the image border, the 92-pixel input context window extends beyond the image boundary.

**Solution:** Mirror-pad the full image by the margin before tiling:

```python
padded_img = np.pad(prep_img,
                    ((margin, margin + pad_h_extra),
                     (margin, margin + pad_w_extra)),
                    mode='reflect')
```

`mode='reflect'` mirrors pixel values at the boundary. `pad_h_extra` and `pad_w_extra` ensure the tiled 100×100 output blocks perfectly cover the full image dimensions:

```python
pad_h_extra = (out_patch_size - (h % out_patch_size)) % out_patch_size
```

### 9.3 Tile Sliding & Stitching

```python
# Collect all input tiles (step by output size = 100px)
for y in range(0, target_h, out_patch_size):
    for x in range(0, target_w, out_patch_size):
        p = padded_img[y : y + in_patch_size, x : x + in_patch_size]  # 284x284
        patch_list.append(p)
        coord_list.append((y, x))

# Batch inference through model
patch_tensors = torch.from_numpy(np.array(patch_list)).unsqueeze(1).float()
predictions = []
with torch.no_grad():
    for batch in chunks(patch_tensors, batch_size=4):
        out = model(batch)           # (4, 1, 100, 100)
        predictions.append(out.cpu().numpy())

# Place predictions non-overlappingly into output map
for (y, x), pred_patch in zip(coord_list, predictions):
    prob_map[y : y + out_patch_size, x : x + out_patch_size] = pred_patch
```

### 9.4 FOV Masking

```python
prob_map = prob_map[:h, :w]    # Crop to original dimensions
prob_map = prob_map * fov_mask  # Zero out non-retinal background
```

Elementwise multiplication with the binary FOV mask sets all predictions outside the circular retinal field to 0.

---

## 10. Evaluation Metrics

All metrics are computed **exclusively within the Field of View** (pixels where fov_mask > 0.5). Outside the FOV is trivially zero — including it would inflate specificity and accuracy unfairly.

### 10.1 Confusion Matrix Components

Threshold: pred_prob >= 0.5 → binary prediction

| | **Pred: Vessel** | **Pred: Background** |
|---|---|---|
| **GT: Vessel** | TP (True Positive) | FN (False Negative) |
| **GT: Background** | FP (False Positive) | TN (True Negative) |

### 10.2 Accuracy, Sensitivity, Specificity, Precision

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correct pixel classification rate |
| **Sensitivity** | TP/(TP+FN) | Fraction of vessels correctly detected (Recall/TPR) |
| **Specificity** | TN/(TN+FP) | Fraction of background correctly classified (TNR) |
| **Precision** | TP/(TP+FP) | Fraction of vessel predictions that are correct (PPV) |

**Achieved:** Accuracy=93.99%, Sensitivity=77.61%, Specificity=96.28%, Precision=74.43%

### 10.3 F1 / Dice Score

```
F1 = 2 * TP / (2*TP + FP + FN) = 2 * Precision * Sensitivity / (Precision + Sensitivity)
```

**Harmonic mean** of Precision and Sensitivity — primary segmentation quality metric. Balances recall (detecting all vessels) with precision (not over-predicting). Directly measures overlap.

**Achieved: 75.88%**

### 10.4 IoU (Jaccard Index)

```
IoU = TP / (TP + FP + FN)
```

Ratio of intersection to union of predicted and ground-truth vessel regions. Stricter than Dice.

**Relationship:** `IoU = Dice / (2 - Dice)` — IoU is always lower than Dice.

**Achieved: 61.14%**

### 10.5 AUC-ROC and AUC-PR

**AUC-ROC:** Area Under the ROC curve (TPR vs. FPR at all thresholds). Measures discriminatory ability threshold-independently. AUC=0.5 is random, AUC=1.0 is perfect.

**AUC-PR:** Area Under the Precision-Recall curve. More informative than AUC-ROC for highly imbalanced datasets — focuses on the minority class and is not inflated by large TN counts.

Both use continuous probability values (not thresholded binary predictions), capturing the full quality of the probability map.

**Achieved:** AUC-ROC=95.65%, AUC-PR=84.44%

### 10.6 FOV-Restricted Evaluation Code

```python
fov_indices = np.where(fov_mask > 0.5)   # (y_array, x_array) of valid pixels
y_true = ground_truth[fov_indices].flatten().astype(int)
y_prob = pred_prob[fov_indices].flatten()
y_pred = (y_prob >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
```

`labels=[0, 1]` ensures consistent ordering even if one class is absent. `1e-8` epsilon prevents division by zero.

---

## 11. Results

### Validation Benchmark (4 held-out labeled images, full-image overlap-tile evaluation)

| Image | Accuracy | Sensitivity | Specificity | Precision | F1/Dice | IoU | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| 37_training.tif | 0.9341 | 0.7931 | 0.9546 | 0.7174 | 0.7533 | 0.6043 | 0.9472 | 0.8434 |
| 38_training.tif | 0.9417 | 0.7364 | 0.9713 | 0.7869 | 0.7608 | 0.6140 | 0.9593 | 0.8500 |
| 39_training.tif | 0.9382 | 0.7596 | 0.9636 | 0.7484 | 0.7540 | 0.6051 | 0.9554 | 0.8327 |
| 40_training.tif | 0.9455 | 0.8152 | 0.9616 | 0.7244 | 0.7671 | 0.6222 | 0.9641 | 0.8516 |
| **AVERAGE** | **0.9399** | **0.7761** | **0.9628** | **0.7443** | **0.7588** | **0.6114** | **0.9565** | **0.8444** |

### Training Summary

| Parameter | Value |
|---|---|
| Best validation Dice (patch-level) | 0.7631 |
| Full-image validation Dice | 0.7588 |
| Total training time | ~13.24 minutes on CPU |
| Training epochs | 5 |
| Model parameters | 31,036,481 |

### Context — DRIVE State-of-the-Art

| Method | AUC-ROC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Staal et al. 2004 (Classical) | 0.952 | 0.718 | 0.977 | 0.944 |
| Soares et al. 2006 (Classical) | 0.961 | 0.733 | 0.978 | 0.946 |
| Ronneberger U-Net (full training) | ~0.979 | ~0.770 | ~0.981 | ~0.953 |
| **Our 5-epoch CPU implementation** | **0.957** | **0.776** | **0.963** | **0.940** |

Our results are competitive with pre-deep-learning classical methods, considering only 5 training epochs on CPU.

---

## 12. Code Deep Dive

### 12.1 `src/preprocessing.py`

**File purpose:** Classical image preprocessing pipeline.

```python
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
```

---

**`extract_green_channel(image)`**
```python
if len(image.shape) == 3 and image.shape[2] == 3:
    return image[:, :, 1]  # Green = index 1 in RGB
return image  # Safety fallback for already-grayscale input
```

---

**`apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8))`**
```python
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
return clahe.apply(image)  # Input/output: uint8 (H,W) array
```

---

**`apply_noise_reduction(image, method='bilateral')`**
```python
# method='bilateral' → cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
# method='median'    → cv2.medianBlur(image, ksize=3)
# method='gaussian'  → cv2.GaussianBlur(image, (3,3), 0)
```

---

**`apply_elastic_transform(image, mask, alpha=30.0, sigma=4.0)`**

Step-by-step:
```python
# 1. Generate smooth random displacement fields
dx = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
dy = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

# 2. Create coordinate grids (indexing='ij' = row-major: y first, x second)
y_img, x_img = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), indexing='ij')

# 3. Add displacement to get new coordinates
indices_img = np.reshape(y_img + dy, (-1, 1)), np.reshape(x_img + dx, (-1, 1))

# 4. Bilinear interpolation at new coordinates (order=1)
distorted_image = map_coordinates(image, indices_img, order=1, mode='reflect').reshape(img_shape)

# 5. For mask: slice center region of displacement field (handles 284→100 mismatch)
my = (img_shape[0] - mask_shape[0]) // 2  # = 92
mx = (img_shape[1] - mask_shape[1]) // 2  # = 92
dx_mask = dx[my : my + mask_shape[0], mx : mx + mask_shape[1]]
dy_mask = dy[my : my + mask_shape[0], mx : mx + mask_shape[1]]

# 6. Nearest-neighbor for mask (order=0) — preserves binary labels without blending
distorted_mask = map_coordinates(mask, indices_mask, order=0, mode='reflect').reshape(mask_shape)
```

---

**`preprocess_image(image)`**
```python
green    = extract_green_channel(image)
clahe    = apply_clahe(green, clip_limit=2.0, tile_grid_size=(8, 8))
denoised = apply_noise_reduction(clahe, method='bilateral')
normalized = normalize_image(denoised)
return normalized  # float32 (H, W) in [0.0, 1.0]
```

---

### 12.2 `src/dataset.py`

**File purpose:** File path loading, image-level train/val splitting, patch extraction, PyTorch Dataset.

---

**`load_drive_image_pairs(data_dir, mode, val_ratio=0.2)`**

```python
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))  # sorted = deterministic
val_count = int(len(img_paths) * val_ratio)  # = 4

if mode == 'train':
    img_paths = img_paths[:-val_count]       # Indices 0-15 (images 21-36)
else:  # 'val'
    img_paths = img_paths[-val_count:]       # Indices 16-19 (images 37-40)
```

Returns list of `(img_path, manual_path, mask_path)` tuples. For test mode, manual_path = None.

---

**`extract_random_patches(image, mask, fov_mask, in_patch_size=284, num_patches=50)`**

```python
out_patch_size = in_patch_size - 184  # 100
margin = 92  # (284 - 100) // 2

# Mirror-pad image and ground truth (reflect = natural boundary continuation)
padded_image = np.pad(image, ((margin, margin), (margin, margin)), mode='reflect')
padded_mask  = np.pad(mask,  ((margin, margin), (margin, margin)), mode='reflect')
padded_fov   = np.pad(fov_mask, ((margin, margin), (margin, margin)),
                      mode='constant', constant_values=0)

# Get all valid FOV pixel coordinates in original (unpadded) region
valid_y, valid_x = np.where(padded_fov[margin:h_pad-margin, margin:w_pad-margin] > 0)
valid_y += margin; valid_x += margin  # Adjust to padded coordinate space

# Sample num_patches centers from valid FOV pixels (with replacement)
indices = np.random.choice(len(valid_y), size=num_patches, replace=True)
ys = valid_y[indices] - (in_patch_size // 2)  # Convert center → top-left of 284x284
xs = valid_x[indices] - (in_patch_size // 2)
ys = np.clip(ys, 0, h_pad - in_patch_size)    # Keep within padded bounds
xs = np.clip(xs, 0, w_pad - in_patch_size)

for y, x in zip(ys, xs):
    img_p  = padded_image[y : y + in_patch_size,  x : x + in_patch_size]      # Full 284x284
    mask_p = padded_mask [y + margin : y + margin + out_patch_size,            # Center 100x100
                          x + margin : x + margin + out_patch_size]
```

---

**`RetinalPatchDataset.__init__`**

All patches pre-extracted and stored in memory as numpy arrays:
```python
for img_path, manual_path, mask_path in image_triplets:
    raw_rgb = np.array(Image.open(img_path))
    prep_img = preprocess_image(raw_rgb)
    manual_gt = np.array(Image.open(manual_path)).astype(np.float32) / 255.0
    fov_mask  = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
    
    manual_gt = (manual_gt > 0.5).astype(np.float32)  # Binarize (threshold at 0.5)
    fov_mask  = (fov_mask  > 0.5).astype(np.float32)  # Binarize FOV
    
    imgs, masks = extract_random_patches(prep_img, manual_gt, fov_mask, ...)

self.images = np.concatenate(all_img_patches, axis=0)   # (N, 284, 284)
self.masks  = np.concatenate(all_mask_patches, axis=0)  # (N, 100, 100)
```

---

**`RetinalPatchDataset.__getitem__`**

```python
img_p  = self.images[idx].copy()   # .copy() prevents in-place modification of stored array
mask_p = self.masks[idx].copy()

# Augmentation (only in train mode)
if self.augment and self.mode == 'train':
    if np.random.rand() > 0.5: img_p = np.fliplr(img_p); mask_p = np.fliplr(mask_p)
    if np.random.rand() > 0.5: img_p = np.flipud(img_p); mask_p = np.flipud(mask_p)
    k = np.random.choice([0, 1, 2, 3])
    if k > 0: img_p = np.rot90(img_p, k); mask_p = np.rot90(mask_p, k)
    if np.random.rand() > 0.5:
        img_p, mask_p = apply_elastic_transform(img_p, mask_p, alpha=30.0, sigma=4.0)

# np.rot90 returns a non-contiguous view; PyTorch requires contiguous memory
img_p  = np.ascontiguousarray(img_p)
mask_p = np.ascontiguousarray(mask_p)

# Add channel dim: (H,W) → (1,H,W)
tensor_img  = torch.from_numpy(img_p).unsqueeze(0).float()   # (1, 284, 284)
tensor_mask = torch.from_numpy(mask_p).unsqueeze(0).float()  # (1, 100, 100)
return tensor_img, tensor_mask
```

---

### 12.3 `src/unet_model.py`

**`crop_tensor(enc_tensor, target_tensor)`**

```python
delta_H = (H_enc - H_tgt) // 2  # Integer division = equal margins
delta_W = (W_enc - W_tgt) // 2
return enc_tensor[:, :, delta_H : delta_H + H_tgt, delta_W : delta_W + W_tgt]
# Keeps all batches (:) and channels (:), crops spatial dimensions to target size
```

**`UNet.forward`**

```python
# Encoder: save feature maps at each level for skip connections
e1 = self.enc1(x)                     # (B, 64, 280, 280)
e2 = self.enc2(self.pool1(e1))        # (B, 128, 136, 136)
e3 = self.enc3(self.pool2(e2))        # (B, 256, 64, 64)
e4 = self.enc4(self.pool3(e3))        # (B, 512, 28, 28)

b  = self.dropout(self.bottleneck(self.pool4(e4)))  # (B, 1024, 10, 10)

# Decoder: upsample → crop skip connection → concatenate → refine
u4 = self.up4(b)                      # (B, 512, 20, 20)
e4_cropped = crop_tensor(e4, u4)      # (B, 512, 20, 20)
d4 = self.dec4(torch.cat([u4, e4_cropped], dim=1))  # cat: 1024ch, out: (B, 512, 16, 16)

u3 = self.up3(d4)                     # (B, 256, 32, 32)
e3_cropped = crop_tensor(e3, u3)      # (B, 256, 32, 32)
d3 = self.dec3(torch.cat([u3, e3_cropped], dim=1))  # (B, 256, 28, 28)

u2 = self.up2(d3)                     # (B, 128, 56, 56)
e2_cropped = crop_tensor(e2, u2)      # (B, 128, 56, 56)
d2 = self.dec2(torch.cat([u2, e2_cropped], dim=1))  # (B, 128, 52, 52)

u1 = self.up1(d2)                     # (B, 64, 104, 104)
e1_cropped = crop_tensor(e1, u1)      # (B, 64, 104, 104)
d1 = self.dec1(torch.cat([u1, e1_cropped], dim=1))  # (B, 64, 100, 100)

logits = self.out_conv(d1)            # (B, 1, 100, 100)
output = self.sigmoid(logits)         # (B, 1, 100, 100), values in [0,1]
return output
```

---

### 12.4 `src/losses.py`

**`DiceLoss.forward`**
```python
pred_flat   = pred.view(-1)    # Flatten all dims to 1D vector
target_flat = target.view(-1)
intersection = (pred_flat * target_flat).sum()  # Soft intersection Σ(pred_i * target_i)
dice_score = (2.0 * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
return 1.0 - dice_score        # Minimize this → maximize Dice
```

**`CombinedBCEDiceLoss.forward`**
```python
bce_loss  = self.bce(pred, target)   # nn.BCELoss — requires pred in [0,1] (post-sigmoid)
dice_loss = self.dice(pred, target)
return 0.5 * bce_loss + 0.5 * dice_loss
```

---

### 12.5 `src/metrics.py`

**`compute_fov_metrics(pred_prob, ground_truth, fov_mask, threshold=0.5)`**

```python
fov_indices = np.where(fov_mask > 0.5)       # Extract FOV pixel indices
y_true = ground_truth[fov_indices].flatten().astype(int)
y_prob = pred_prob[fov_indices].flatten()
y_pred = (y_prob >= threshold).astype(int)

# Confusion matrix with explicit labels=[0,1] for safe ravel ordering
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
sensitivity = tp / (tp + fn + 1e-8)   # Recall
specificity = tn / (tn + fp + 1e-8)
precision   = tp / (tp + fp + 1e-8)
f1_dice     = 2 * tp / (2 * tp + fp + fn + 1e-8)
iou         = tp / (tp + fp + fn + 1e-8)

# AUC metrics on continuous probabilities (threshold-independent)
auc_roc = roc_auc_score(y_true, y_prob)
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
auc_pr  = auc(rec_curve, prec_curve)
```

---

### 12.6 `src/utils.py`

**`predict_full_image(model, prep_img, fov_mask, in_patch_size=284, device=cpu)`**

Full overlap-tile inference:
```python
out_patch_size = in_patch_size - 184   # 100
margin = 92                            # (284 - 100) // 2

# Extra padding to make image dimensions divisible by out_patch_size
pad_h_extra = (out_patch_size - (h % out_patch_size)) % out_patch_size
pad_w_extra = (out_patch_size - (w % out_patch_size)) % out_patch_size

# Mirror-pad full image (reflection = natural boundary continuation)
padded_img = np.pad(prep_img, ((margin, margin + pad_h_extra),
                                (margin, margin + pad_w_extra)), mode='reflect')

# Collect all tiles (stride = out_patch_size = non-overlapping outputs)
for y in range(0, target_h, out_patch_size):
    for x in range(0, target_w, out_patch_size):
        p = padded_img[y : y + in_patch_size, x : x + in_patch_size]  # 284x284 input tile
        patch_list.append(p)
        coord_list.append((y, x))

# Stack and run in mini-batches
patch_tensors = torch.from_numpy(np.array(patch_list)).unsqueeze(1).float()  # (N,1,284,284)
with torch.no_grad():
    for i in range(0, len(patch_tensors), 4):
        batch = patch_tensors[i:i+4].to(device)
        out = model(batch)               # (4, 1, 100, 100)
        predictions.append(out.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)[:, 0, :, :]  # (N, 100, 100) — remove channel dim

# Place each 100x100 prediction at its position in the output map
for (y, x), pred_patch in zip(coord_list, predictions):
    prob_map[y : y + out_patch_size, x : x + out_patch_size] = pred_patch

prob_map = prob_map[:h, :w] * fov_mask  # Crop to original size, mask background
return prob_map  # float32 (H, W) probability map in [0, 1]
```

**`save_prediction_figure(..., ground_truth=None)`**

With ground_truth (validation — 5 panels):
1. Raw RGB fundus
2. Ground truth binary mask (gray)
3. Predicted probability heatmap (magma colormap)
4. Binary prediction (threshold 0.5)
5. Error map: Green=TP, Red=FP, Blue=FN

Without ground_truth (test — 3 panels):
1. Raw RGB fundus
2. Vessel probability heatmap
3. Binary vessel mask

---

### 12.7 `scripts/train.py`

**Argparse CLI defaults:**
```python
--epochs          5    (use more for better convergence)
--batch_size      4
--lr              1e-3
--in_patch_size   284
--patches_per_img 10
--base_filters    64
```

**Setup:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = RetinalPatchDataset(data_dir, mode='train', ..., augment=True)
val_dataset   = RetinalPatchDataset(data_dir, mode='val',   ..., augment=False)
# Note: val uses patches_per_img=20 to cover more of each validation image
```

**Training curves saved at end:**
- Plot 1 (left): Train Loss vs Val Loss (BCE+Dice) over epochs
- Plot 2 (right): Val Dice vs Val Accuracy over epochs
- Saved to `outputs/visualizations/training_curves.png`

---

### 12.8 `scripts/evaluate.py`

**Flow:**
1. Load `best_unet.pth` with `map_location=device` (GPU→CPU compatible)
2. For each validation image (37-40): preprocess → overlap-tile predict → compute FOV metrics → save 5-panel figure
3. Build Pandas DataFrame of per-image metrics → append AVERAGE row → save CSV
4. For each test image (01-20): preprocess → overlap-tile predict → save 3-panel figure (no GT available)

```python
# Pandas CSV creation
df_val = pd.DataFrame(val_results)
cols = ["filename", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Dice", "IoU", "AUC_ROC", "AUC_PR"]
df_val = df_val[cols]
avg_row = df_val.mean(numeric_only=True).to_dict()
avg_row["filename"] = "AVERAGE"
df_val = pd.concat([df_val, pd.DataFrame([avg_row])], ignore_index=True)
df_val.to_csv(report_csv_path, index=False)
```

---

### 12.9 `scripts/setup_dataset.py`

Run first to verify dataset integrity:
```python
assert len(train_images) == 20  # 21_training.tif to 40_training.tif
assert len(train_manuals) == 20  # Expert GT annotations
assert len(train_masks)   == 20  # FOV masks
assert len(test_images)   == 20  # 01_test.tif to 20_test.tif
assert len(test_masks)    == 20  # FOV masks only (no GT)
```

---

### 12.10 `scripts/test_dataset_and_model.py`

Integration sanity tests (run after setup, before training):
1. `RetinalPatchDataset` creates patches with shape `(1, 572, 572)` / `(1, 388, 388)` — confirms valid conv math
2. `UNet(base_filters=64)` accepts `(2,1,572,572)` input → produces `(2,1,388,388)` output
3. `CombinedBCEDiceLoss` computes without error
4. `compute_fov_metrics` returns all expected metric keys

Also saves `dataset_patches_sample.png` — visual verification of input/target patch pairs.

---

## 13. Project Folder Structure

```
MIP/
├── data/
│   └── DRIVE/
│       ├── training/
│       │   ├── images/          21_training.tif to 40_training.tif
│       │   ├── 1st_manual/      21_manual1.gif to 40_manual1.gif (expert GT)
│       │   └── mask/            21_training_mask.gif to 40_training_mask.gif (FOV)
│       └── test/
│           ├── images/          01_test.tif to 20_test.tif
│           └── mask/            01_test_mask.gif to 20_test_mask.gif (FOV)
│
├── src/                         Core library (importable as 'src')
│   ├── __init__.py
│   ├── preprocessing.py         Classical image pipeline (CLAHE + Bilateral)
│   ├── dataset.py               DRIVE loading + PyTorch Dataset + patch extraction
│   ├── unet_model.py            U-Net architecture (valid conv + crop skip)
│   ├── losses.py                DiceLoss + CombinedBCEDiceLoss
│   ├── metrics.py               FOV-restricted segmentation metrics
│   └── utils.py                 Overlap-tile inference + prediction visualization
│
├── scripts/                     Executable pipeline scripts
│   ├── setup_dataset.py         Step 0: Dataset integrity verification
│   ├── test_preprocessing.py    Optional: Preprocessing visualization
│   ├── test_dataset_and_model.py Step 1: Integration sanity tests
│   ├── train.py                 Step 2: Full training engine
│   ├── evaluate.py              Step 3: Evaluation + test predictions
│   └── create_notebook.py       Step 4: Generates main_notebook.ipynb
│
├── outputs/
│   ├── metrics_report.csv       Per-image + AVERAGE validation metrics
│   ├── saved_models/
│   │   └── best_unet.pth        Best model checkpoint (~93MB)
│   └── visualizations/
│       ├── training_curves.png
│       ├── preprocessing_sample.png
│       ├── dataset_patches_sample.png
│       ├── val_pred_37_training.png   5-panel (Raw+GT+Heatmap+Binary+ErrorMap)
│       ├── val_pred_38_training.png
│       ├── val_pred_39_training.png
│       ├── val_pred_40_training.png
│       ├── test_pred_01_test.png      3-panel (Raw+Heatmap+Binary)
│       └── ... test_pred_02 to 20
│
├── main_notebook.ipynb          Fully executed presentation notebook (~1.8MB)
├── requirements.txt             Python dependencies
├── research paper_baseline.pdf  Ronneberger et al. (2015)
├── project plan.pdf             Faculty project plan
└── REVIEW.md                    This document
```

---

## 14. Execution Order & Commands

All commands run from the project root (MIP/ directory):

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1: Verify Dataset
```bash
python scripts/setup_dataset.py
```
Expected: "Dataset verification SUCCESSFUL!" with counts of 20/20/20 for training.

### Step 2: Integration Sanity Test (Optional)
```bash
python scripts/test_dataset_and_model.py
```
Confirms dataset → model → loss → metrics pipeline is correct before committing to training.

### Step 3: Train
```bash
python scripts/train.py --epochs 5
# For better convergence (if time permits):
python scripts/train.py --epochs 50 --patches_per_img 50
```
Saves `best_unet.pth` and `training_curves.png`.

### Step 4: Evaluate
```bash
python scripts/evaluate.py
```
Saves `metrics_report.csv` + all validation and test prediction figures.

### Step 5: Generate & Execute Notebook
```bash
python scripts/create_notebook.py
python -m nbconvert --to notebook --execute --inplace main_notebook.ipynb
```
Produces fully executed `main_notebook.ipynb`.

---

## 15. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| **Conv padding** | Valid (padding=0) | Strict paper compliance; no border artifacts |
| **Skip connections** | Center crop + concatenate | Required for valid conv spatial mismatch |
| **Base filters** | 64 → 31M params | Paper-exact architecture |
| **Batch Normalization** | Retained | CPU training stability; paper was GPU-trained |
| **Dropout** | 0.5 at bottleneck | Regularization; prevents overfit on 16 images |
| **Augmentation** | Flips + Rotations + Elastic | Paper-specified; max effective dataset size |
| **Elastic alpha** | 30.0 px | Paper specification |
| **Elastic sigma** | 4.0 | Paper specification (local deformations) |
| **Input patch** | 284×284 → 100×100 output | Valid conv math: out = in - 184 |
| **Loss function** | BCE + Dice (50-50) | Handles class imbalance + pixel accuracy |
| **Optimizer** | Adam (lr=1e-3, wd=1e-5) | Robust, adaptive, standard choice |
| **LR scheduler** | CosineAnnealingLR | Smooth decay from 1e-3 to near 0 |
| **Train/val split** | Image-level (16/4) | Prevents data leakage from correlated patches |
| **Inference** | Overlap-tile + mirror padding | Paper-specified; full-res without resize |
| **Evaluation restriction** | Inside FOV mask only | Excludes trivial black background pixels |
| **Primary metric** | Dice / F1 | Handles imbalance; standard segmentation metric |
| **Threshold** | 0.5 | Standard default for binary classification |

---

## 16. Output Artifacts

| Artifact | Location | Description |
|---|---|---|
| `best_unet.pth` | `outputs/saved_models/` | Best model (by val Dice), ~93MB |
| `metrics_report.csv` | `outputs/` | Per-image + AVERAGE validation metrics CSV |
| `training_curves.png` | `outputs/visualizations/` | Loss and Dice progression over 5 epochs |
| `preprocessing_sample.png` | `outputs/visualizations/` | 4-stage pipeline visualization |
| `dataset_patches_sample.png` | `outputs/visualizations/` | Input 284×284 + target 100×100 pairs |
| `val_pred_37/38/39/40_training.png` | `outputs/visualizations/` | 5-panel: Raw+GT+Heatmap+Binary+ErrorMap |
| `test_pred_01-20_test.png` | `outputs/visualizations/` | 3-panel: Raw+Heatmap+Binary (no GT) |
| `main_notebook.ipynb` | Root | Fully executed presentation notebook, ~1.8MB |

---

*This document provides a complete technical reference for the Automated Retinal Blood Vessel Segmentation project (Review 1), implementing the Ronneberger et al. (2015) baseline U-Net on the DRIVE dataset.*
