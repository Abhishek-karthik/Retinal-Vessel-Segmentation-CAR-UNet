import os
import cv2
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    Extracts the green channel from an RGB fundus image.
    Retinal blood vessels have maximum optical absorption/contrast in the green channel.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image[:, :, 1]
    return image

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to boost local contrast
    without amplifying uniform background noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_noise_reduction(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Applies edge-preserving noise reduction (Bilateral filter or Median filter).
    """
    if method == 'bilateral':
        # d=5, sigmaColor=75, sigmaSpace=75 keeps sharp vessel borders
        return cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
    elif method == 'median':
        return cv2.medianBlur(image, ksize=3)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (3, 3), 0)
    return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values from [0, 255] integer range to [0.0, 1.0] float32 range.
    """
    return image.astype(np.float32) / 255.0

def apply_elastic_transform(image: np.ndarray, mask: np.ndarray, alpha: float = 30.0, sigma: float = 4.0, random_state=None) -> tuple:
    """
    Applies elastic deformation to image and mask pairs as described in Ronneberger et al. (2015).
    Generates random smooth displacement fields using Gaussian filtering.
    Supports mismatched shapes (e.g. 284x284 input image and 100x100 target mask).
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    img_shape = image.shape
    mask_shape = mask.shape
    
    dx = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*img_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    y_img, x_img = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), indexing='ij')
    indices_img = np.reshape(y_img + dy, (-1, 1)), np.reshape(x_img + dx, (-1, 1))
    distorted_image = map_coordinates(image, indices_img, order=1, mode='reflect').reshape(img_shape)
    
    if img_shape == mask_shape:
        indices_mask = indices_img
    else:
        my = (img_shape[0] - mask_shape[0]) // 2
        mx = (img_shape[1] - mask_shape[1]) // 2
        
        y_mask, x_mask = np.meshgrid(np.arange(mask_shape[0]), np.arange(mask_shape[1]), indexing='ij')
        dx_mask = dx[my : my + mask_shape[0], mx : mx + mask_shape[1]]
        dy_mask = dy[my : my + mask_shape[0], mx : mx + mask_shape[1]]
        
        indices_mask = np.reshape(y_mask + dy_mask, (-1, 1)), np.reshape(x_mask + dx_mask, (-1, 1))

    distorted_mask = map_coordinates(mask, indices_mask, order=0, mode='reflect').reshape(mask_shape)
    return distorted_image, distorted_mask

def preprocess_image(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Full classical preprocessing pipeline for retinal fundus images (FIVES dataset):
    1. Green channel extraction
    2. CLAHE contrast enhancement
    3. Bilateral noise reduction
    4. Normalization [0.0, 1.0]
    
    Returns:
        np.ndarray: Preprocessed float32 image with shape (H, W) and values in [0.0, 1.0]
    """
    green = extract_green_channel(image)
    clahe = apply_clahe(green, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    denoised = apply_noise_reduction(clahe, method='bilateral')
    normalized = normalize_image(denoised)
    return normalized

def generate_fov_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Generates a binary Field of View (FOV) mask for fundus images by thresholding
    and morphological cleanup. Returns binary mask (H, W) with {0.0, 1.0}.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 128).astype(np.float32)

def run_fives_preprocessing_examples():
    """
    Visualizes before/after preprocessing on sample FIVES images and saves figures.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "FIVES_resized", "train")
    output_dir = os.path.join(base_dir, "outputs", "fives_preprocessing_examples")
    os.makedirs(output_dir, exist_ok=True)
    
    img_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "masks", "*.png")))
    
    if not img_paths:
        print(f"Error: No images found in {data_dir}/images. Ensure resize_fives.py has completed!")
        return
        
    samples_to_show = [0, len(img_paths)//4, len(img_paths)//2, 3*len(img_paths)//4, len(img_paths)-1]
    
    print(f"Generating preprocessing examples for {len(samples_to_show)} FIVES images...")
    for idx in samples_to_show:
        img_p = img_paths[idx]
        mask_p = mask_paths[idx]
        fname = os.path.basename(img_p)
        
        raw_rgb = np.array(Image.open(img_p))
        green = extract_green_channel(raw_rgb)
        clahe = apply_clahe(green, clip_limit=2.0, tile_grid_size=(8, 8))
        denoised = apply_noise_reduction(clahe, method='bilateral')
        preprocessed = normalize_image(denoised)
        gt_mask = np.array(Image.open(mask_p))
        
        fig, axes = plt.subplots(1, 6, figsize=(22, 4))
        
        axes[0].imshow(raw_rgb)
        axes[0].set_title(f"1. Raw RGB\n({fname})", fontsize=11)
        axes[0].axis('off')
        
        axes[1].imshow(green, cmap='gray')
        axes[1].set_title("2. Green Channel", fontsize=11)
        axes[1].axis('off')
        
        axes[2].imshow(clahe, cmap='gray')
        axes[2].set_title("3. CLAHE Enhanced\n(clip=2.0, grid=8x8)", fontsize=11)
        axes[2].axis('off')
        
        axes[3].imshow(denoised, cmap='gray')
        axes[3].set_title("4. Bilateral Denoised\n(d=5, sigma=75)", fontsize=11)
        axes[3].axis('off')
        
        axes[4].imshow(preprocessed, cmap='gray')
        axes[4].set_title("5. Normalized [0, 1]", fontsize=11)
        axes[4].axis('off')
        
        axes[5].imshow(gt_mask, cmap='gray')
        axes[5].set_title("6. Ground Truth GT", fontsize=11)
        axes[5].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"preprocessing_step_{fname}")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
    print("All preprocessing example visualizations saved successfully!")

if __name__ == "__main__":
    run_fives_preprocessing_examples()
