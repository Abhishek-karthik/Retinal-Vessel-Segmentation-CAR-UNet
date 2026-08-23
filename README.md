# Retinal Blood Vessel Segmentation using CAR-UNet on the FIVES Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning pipeline for automated retinal blood vessel segmentation in fundus photography. This repository upgrades the classical **Ronneberger U-Net (2015)** on the **DRIVE dataset** to **Channel Attention Residual U-Net (CAR-UNet)** on the multi-cohort **FIVES dataset (800 images)**, based on [Guo et al. (IEEE 2021)](https://arxiv.org/abs/2004.03702).

---

## 📊 Consolidated 3-Way Benchmark Comparison

Evaluated strictly inside the **Field of View (FOV)** on held-out test sets:

| Model | Dataset | Training Scale | Accuracy | Sensitivity (Recall) | Specificity | F1 / Dice Score | AUC-ROC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **U-Net (Baseline)** | DRIVE | 16 train / 4 val | **93.99%** | **77.61%** | **96.28%** | **75.88%** | **95.65%** |
| **U-Net (Dataset Scale)** | FIVES | 480 train / 120 val | **97.56%** | **72.88%** | **99.05%** | **77.44%** | **97.47%** |
| **CAR-UNet (Full Upgrade ★)** | FIVES | 480 train / 120 val | **97.56%** | **76.92%** | **98.81%** | **79.16%** | **97.59%** |

---

## 🌟 Key Architectural Innovations

```
Input Patch (284x284)
  │
  ├──> [Encoder Level 1: CADRB (64)] ─── MaxPool ──> [MECA Skip Attention 1]
  │         │                                                    │
  ├──> [Encoder Level 2: CADRB (128)] ── MaxPool ──> [MECA Skip Attention 2]
  │         │                                                    │
  ├──> [Encoder Level 3: CADRB (256)] ── MaxPool ──> [MECA Skip Attention 3]
  │         │                                                    │
  ├──> [Encoder Level 4: CADRB (512)] ── MaxPool ──> [MECA Skip Attention 4]
  │         │                                                    │
  └──> [Bottleneck: CADRB (1024) + Dropout 0.5]                  │
            │                                                    │
       ConvTranspose ────────────────── Concat & Crop <──────────┘
            │
       [Decoder Levels 4 to 1: CADRB]
            │
       1x1 Conv + Sigmoid ──> Output Vessel Mask (100x100)
```

1. **Modified Efficient Channel Attention (MECA):** Fast, non-dimensionality-reducing channel recalibration using 1D adaptive convolutions:
   $$k = \psi(C) = \left| \frac{\log_2(C) + b}{\gamma} \right|_{\text{odd}}$$
2. **Channel Attention Double Residual Blocks (CADRB):** Identity shortcut mappings embedded with channel attention to prevent vanishing gradients and preserve microcapillary edges.
3. **Bridge Attention on Skip Connections:** Recalibrates encoder feature maps prior to decoder concatenation to suppress background artifacts.
4. **Ronneberger Overlap-Tile Strategy:** Valid convolutions (`padding=0`) with mirror-padded sliding window inference for seamless full-image reconstruction without seamline artifacts.

---

## 📁 Repository Structure

```
.
├── Review2_Complete_CAR_UNet_Pipeline.ipynb  # All-in-one standalone interactive notebook
├── Review2_CAR_UNet_FIVES.ipynb              # Modular Review 2 demonstration notebook
├── main_notebook.ipynb                       # Original Review 1 DRIVE baseline notebook
├── REVIEW2.md                                # Comprehensive Review 2 technical report
├── RESULTS_SUMMARY.md                        # Quantitative & qualitative findings
├── CURRENT_STATE_SUMMARY.md                  # Review 1 audit & baseline summary
├── requirements.txt                          # Python dependencies
│
├── src/                                      # Core library modules
│   ├── car_unet.py                           # CAR-UNet architecture (MECA + CADRB)
│   ├── unet_model.py                         # Baseline Ronneberger U-Net model
│   ├── preprocessing_fives.py                # Green channel, CLAHE, bilateral filter, FOV mask
│   ├── data_loader_fives.py                  # Patch extraction & elastic deformation
│   ├── losses.py                             # Combined BCE + Dice loss
│   ├── metrics.py                            # True FOV medical evaluation metrics
│   └── utils.py                              # Overlap-Tile inference engine
│
├── scripts/                                  # Execution engines
│   ├── train_car_unet_fives.py               # CAR-UNet training & evaluation engine
│   ├── train_unet_fives.py                   # Baseline U-Net training engine
│   ├── compare_results.py                    # 3-Way benchmark comparison generator
│   └── resize_fives.py                       # FIVES dataset preparation script
│
├── results/                                  # Benchmark metrics
│   ├── comparison_table.csv                  # Consolidated metrics CSV
│   ├── car_unet_fives_metrics.json           # CAR-UNet evaluation metrics
│   └── unet_fives_metrics.json               # Scaled U-Net evaluation metrics
│
└── outputs/                                  # Visual figures & predictions
    ├── final_comparison_figure.png           # 5-Column side-by-side comparison figure
    ├── fives_dataset_check.png               # FIVES dataset verification panel
    └── fives_preprocessing_examples/         # 6-Panel optical preprocessing examples
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Abhishek-karthik/Retinal-Vessel-Segmentation-CAR-UNet.git
cd Retinal-Vessel-Segmentation-CAR-UNet
pip install -r requirements.txt
```

### 2. Run the Interactive Jupyter Notebook
Open and run **`Review2_Complete_CAR_UNet_Pipeline.ipynb`** in VS Code, Jupyter Lab, or Google Colab for an end-to-end interactive demonstration.

### 3. Run via Command Line
```bash
# Train CAR-UNet on FIVES
python scripts/train_car_unet_fives.py --epochs 15 --patches_per_img 6 --batch_size 16 --test_images 50

# Run 3-Way Comparison & Generate Figures
python scripts/compare_results.py
```

---

## 📚 References
1. **Guo, C. et al.** *"Channel Attention Residual U-Net for Retinal Vessel Segmentation."* IEEE, 2021. [arXiv:2004.03702](https://arxiv.org/abs/2004.03702)
2. **Jin, K. et al.** *"FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation."* Nature Scientific Data, 2022. [DOI: 10.1038/s41597-022-01564-3](https://doi.org/10.1038/s41597-022-01564-3)
3. **Ronneberger, O. et al.** *"U-Net: Convolutional Networks for Biomedical Image Segmentation."* MICCAI, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
