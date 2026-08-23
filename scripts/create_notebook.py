import nbformat as nbf
import os

def create_main_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# 22AIE437 Medical Image Processing - Project Review 1
## Automated Retinal Blood Vessel Segmentation using Classical Image Processing & U-Net

**Author / Team**: Medical Image Processing Project Group  
**Baseline Research Paper**: Ronneberger et al. (2015) *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*  
**Domain Reference**: Liskowski & Krawiec (2016) *"Segmenting Retinal Blood Vessels with Deep Neural Networks"*  
**Dataset**: DRIVE (Digital Retinal Images for Vessel Extraction)  

---

### Project Overview & Technical Implementation Specifications
1. **Classical Image Preprocessing**: Green Channel extraction + CLAHE Contrast Boost + Bilateral Denoising + Normalization.
2. **Patch Extraction & Elastic Deformation**: $572 \times 572$ input patches $\to$ $388 \times 388$ output target masks with Elastic Deformation ($\alpha=30, \sigma=4$), Flips, and 90-degree Rotations.
3. **Original Ronneberger U-Net Model Architecture**:
   - 4-level Contracting Encoder $\to$ Bottleneck $\rightarrow$ 4-level Expansive Decoder
   - **Valid Convolutions (`padding=0`)** with center-cropped skip connections
   - **64 Base Filters** (64 $\to$ 128 $\to$ 256 $\to$ 512 $\to$ 1024 channels, ~31 million parameters)
   - **Batch Normalization** added for CPU training convergence stability
4. **Optimization**: Combined BCE + Dice Loss ($L = L_{BCE} + L_{Dice}$) with Adam Optimizer and Cosine Annealing Learning Rate.
5. **Ronneberger Overlap-Tile Inference**: 92-pixel mirror-padded border tiling for full-resolution prediction.
6. **Evaluation**: Metrics evaluated strictly inside FOV (Field of View) mask on 4 validation images & held-out test predictions on 20 test images.
"""))

    # Cell 1: Setup & Environment
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add project root to sys path
sys.path.append(os.path.abspath("."))
from src.preprocessing import preprocess_image, extract_green_channel, apply_clahe, apply_noise_reduction, apply_elastic_transform
"""))

    # Markdown: Section 1
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Classical Image Preprocessing Pipeline
Retinal fundus images have uneven lighting and low vessel contrast. 
- **Green Channel**: Blood vessels have maximum absorption/contrast.
- **CLAHE**: Enhances local contrast of faint capillaries without amplifying background noise.
- **Bilateral Filter**: Smooths background speckle noise while preserving sharp vessel boundaries.
"""))

    # Cell 2: Preprocessing Visualization
    cells.append(nbf.v4.new_code_cell("""raw_img_path = "data/DRIVE/training/images/21_training.tif"
raw_rgb = np.array(Image.open(raw_img_path))

green = extract_green_channel(raw_rgb)
clahe = apply_clahe(green)
denoised = apply_noise_reduction(clahe)
final_prep = preprocess_image(raw_rgb)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(raw_rgb)
axes[0].set_title("1. Raw RGB Fundus")
axes[0].axis('off')

axes[1].imshow(green, cmap='gray')
axes[1].set_title("2. Green Channel")
axes[1].axis('off')

axes[2].imshow(clahe, cmap='gray')
axes[2].set_title("3. CLAHE Contrast Boost")
axes[2].axis('off')

axes[3].imshow(final_prep, cmap='gray')
axes[3].set_title("4. Denoised & Normalized")
axes[3].axis('off')

plt.tight_layout()
plt.show()
"""))

    # Markdown: Section 2
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. U-Net Architecture & Training Progress
Original U-Net architecture (Ronneberger et al. 2015) using **Valid Convolutions (`padding=0`)**, **Center-Cropped Skip Connections**, **64 Base Filters**, and **Elastic Deformation Data Augmentation**.
"""))

    # Cell 3: Training Curves Plot
    cells.append(nbf.v4.new_code_cell("""curve_img_path = "outputs/visualizations/training_curves.png"
if os.path.exists(curve_img_path):
    img = Image.open(curve_img_path)
    plt.figure(figsize=(14, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Ronneberger U-Net Loss & Validation Metrics Progression Curves", fontsize=14)
    plt.show()
else:
    print("Training curves not found. Run scripts/train.py first.")
"""))

    # Markdown: Section 3
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Quantitative Benchmark Metrics Report
Metrics evaluated strictly inside the FOV (Field of View) mask on the Validation Set:
"""))

    # Cell 4: Display Metrics CSV
    cells.append(nbf.v4.new_code_cell("""csv_path = "outputs/metrics_report.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    display(df.style.highlight_max(axis=0, color='lightgreen'))
else:
    print("Metrics report not found. Run scripts/evaluate.py first.")
"""))

    # Markdown: Section 4
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Visual Segmentation Results (Validation & Held-out Test Sets)
Visualizing side-by-side: Raw Fundus | Ground Truth GT | Vessel Probability Heatmap | Binary Segmentation Mask | Error Map
"""))

    # Cell 5: Display Prediction Visualizations
    cells.append(nbf.v4.new_code_cell("""val_viz_path = "outputs/visualizations/val_pred_37_training.png"
test_viz_path = "outputs/visualizations/test_pred_01_test.png"

if os.path.exists(val_viz_path):
    plt.figure(figsize=(18, 5))
    plt.imshow(Image.open(val_viz_path))
    plt.axis('off')
    plt.title("Validation Set Segmentation Result (Image 37_training.tif)", fontsize=14)
    plt.show()

if os.path.exists(test_viz_path):
    plt.figure(figsize=(15, 5))
    plt.imshow(Image.open(test_viz_path))
    plt.axis('off')
    plt.title("Held-out Test Set Prediction Result (Image 01_test.tif)", fontsize=14)
    plt.show()
"""))

    nb.cells = cells
    
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main_notebook.ipynb"))
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Created main_notebook.ipynb at: {out_path}")

if __name__ == "__main__":
    create_main_notebook()

