import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    Extracts the green channel from an RGB fundus image.
    Retinal blood vessels have maximum absorption/contrast in the green channel.
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
    Supports mismatched shapes (e.g. 572x572 input image and 388x388 target mask).
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
    Full classical preprocessing pipeline for retinal fundus images:
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

