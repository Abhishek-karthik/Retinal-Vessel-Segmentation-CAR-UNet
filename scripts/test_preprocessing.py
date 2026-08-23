import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import extract_green_channel, apply_clahe, apply_noise_reduction, preprocess_image

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_paths = sorted(glob.glob(os.path.join(base_dir, "data", "DRIVE", "training", "images", "*.tif")))
    output_dir = os.path.join(base_dir, "outputs", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    if not image_paths:
        print("Error: No training images found in data/DRIVE/training/images/")
        return

    # Select 2 sample images
    sample_paths = image_paths[:2]
    
    fig, axes = plt.subplots(len(sample_paths), 4, figsize=(16, 8))
    if len(sample_paths) == 1:
        axes = [axes]
        
    for i, path in enumerate(sample_paths):
        # Load image (OpenCV loads BGR, PIL handles TIF reliably)
        pil_img = Image.open(path)
        rgb_img = np.array(pil_img)
        
        green = extract_green_channel(rgb_img)
        clahe = apply_clahe(green)
        denoised = apply_noise_reduction(clahe)
        final_norm = preprocess_image(rgb_img)
        
        fname = os.path.basename(path)
        
        axes[i][0].imshow(rgb_img)
        axes[i][0].set_title(f"Raw RGB ({fname})", fontsize=12)
        axes[i][0].axis("off")
        
        axes[i][1].imshow(green, cmap="gray")
        axes[i][1].set_title("1. Green Channel", fontsize=12)
        axes[i][1].axis("off")
        
        axes[i][2].imshow(clahe, cmap="gray")
        axes[i][2].set_title("2. CLAHE Contrast Boost", fontsize=12)
        axes[i][2].axis("off")
        
        axes[i][3].imshow(final_norm, cmap="gray")
        axes[i][3].set_title("3. Denoised & Normalized", fontsize=12)
        axes[i][3].axis("off")
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "preprocessing_sample.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Preprocessing visualization saved to: {save_path}")
    print("Task 2 Classical Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
