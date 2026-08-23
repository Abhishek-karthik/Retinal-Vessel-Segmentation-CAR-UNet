import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.preprocessing import preprocess_image, apply_elastic_transform

def load_drive_image_pairs(data_dir: str, mode: str = 'train', val_ratio: float = 0.2):
    """
    Loads image, ground truth mask, and FOV mask paths from DRIVE dataset.
    Performs image-level train/val split on the labeled training dataset.
    """
    if mode in ['train', 'val']:
        img_dir = os.path.join(data_dir, "training", "images")
        manual_dir = os.path.join(data_dir, "training", "1st_manual")
        mask_dir = os.path.join(data_dir, "training", "mask")
        
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        manual_paths = sorted(glob.glob(os.path.join(manual_dir, "*.gif")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.gif")))
        
        # Image-level split (e.g., 16 train, 4 val)
        val_count = int(len(img_paths) * val_ratio)
        if mode == 'train':
            img_paths = img_paths[:-val_count]
            manual_paths = manual_paths[:-val_count]
            mask_paths = mask_paths[:-val_count]
        else: # 'val'
            img_paths = img_paths[-val_count:]
            manual_paths = manual_paths[-val_count:]
            mask_paths = mask_paths[-val_count:]
            
        return list(zip(img_paths, manual_paths, mask_paths))
        
    elif mode == 'test':
        img_dir = os.path.join(data_dir, "test", "images")
        mask_dir = os.path.join(data_dir, "test", "mask")
        
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.gif")))
        
        return list(zip(img_paths, [None]*len(img_paths), mask_paths))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def extract_random_patches(image: np.ndarray, mask: np.ndarray, fov_mask: np.ndarray, in_patch_size: int = 284, num_patches: int = 50):
    """
    Extracts random sub-patches for valid convolutions (Ronneberger et al., 2015).
    Input patch size: 284x284
    Target output mask size: 100x100 (centered inside input patch, 92 margin on each side)
    """
    out_patch_size = in_patch_size - 184
    margin = 92
    
    # Mirror-pad image and ground truth by margin to allow patch extraction across borders
    padded_image = np.pad(image, ((margin, margin), (margin, margin)), mode='reflect')
    padded_mask = np.pad(mask, ((margin, margin), (margin, margin)), mode='reflect')
    padded_fov = np.pad(fov_mask, ((margin, margin), (margin, margin)), mode='constant', constant_values=0)
    
    h_pad, w_pad = padded_image.shape
    
    # Extract valid centers inside FOV mask
    valid_y, valid_x = np.where(padded_fov[margin : h_pad - margin, margin : w_pad - margin] > 0)
    valid_y += margin
    valid_x += margin
    
    if len(valid_y) == 0:
        # Fallback to random selection if FOV is small
        max_y = h_pad - in_patch_size
        max_x = w_pad - in_patch_size
        ys = np.random.randint(0, max(1, max_y + 1), size=num_patches)
        xs = np.random.randint(0, max(1, max_x + 1), size=num_patches)
    else:
        indices = np.random.choice(len(valid_y), size=num_patches, replace=True)
        # Convert FOV centers to top-left corner coordinates for input patches
        ys = valid_y[indices] - (in_patch_size // 2)
        xs = valid_x[indices] - (in_patch_size // 2)
        
        # Clamp to valid image bounds
        ys = np.clip(ys, 0, h_pad - in_patch_size)
        xs = np.clip(xs, 0, w_pad - in_patch_size)
        
    patches_img = []
    patches_mask = []
    
    for y, x in zip(ys, xs):
        img_p = padded_image[y : y + in_patch_size, x : x + in_patch_size]
        mask_p = padded_mask[y + margin : y + margin + out_patch_size, x + margin : x + margin + out_patch_size]
        
        patches_img.append(img_p)
        patches_mask.append(mask_p)
        
    return np.array(patches_img), np.array(patches_mask)

class RetinalPatchDataset(Dataset):
    """
    PyTorch Dataset serving retinal vessel image & ground truth patches for valid convolution U-Net.
    """
    def __init__(self, data_dir: str, mode: str = 'train', in_patch_size: int = 284, patches_per_img: int = 50, augment: bool = True):
        super().__init__()
        self.mode = mode
        self.in_patch_size = in_patch_size
        self.out_patch_size = in_patch_size - 184
        self.augment = augment
        
        image_triplets = load_drive_image_pairs(data_dir, mode=mode)
        
        all_img_patches = []
        all_mask_patches = []
        
        for img_path, manual_path, mask_path in image_triplets:
            raw_rgb = np.array(Image.open(img_path))
            prep_img = preprocess_image(raw_rgb)
            
            manual_gt = np.array(Image.open(manual_path)).astype(np.float32) / 255.0 if manual_path else np.zeros_like(prep_img)
            fov_mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
            
            # Threshold ground truth binary mask
            manual_gt = (manual_gt > 0.5).astype(np.float32)
            fov_mask = (fov_mask > 0.5).astype(np.float32)
            
            imgs, masks = extract_random_patches(prep_img, manual_gt, fov_mask, in_patch_size=in_patch_size, num_patches=patches_per_img)
            all_img_patches.append(imgs)
            all_mask_patches.append(masks)
            
        self.images = np.concatenate(all_img_patches, axis=0) # Shape: (N, in_patch_size, in_patch_size)
        self.masks = np.concatenate(all_mask_patches, axis=0)  # Shape: (N, out_patch_size, out_patch_size)

        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_p = self.images[idx].copy()
        mask_p = self.masks[idx].copy()
        
        # Apply data augmentations if enabled (flips, 90-deg rotations, elastic deformation)
        if self.augment and self.mode == 'train':
            if np.random.rand() > 0.5:
                img_p = np.fliplr(img_p)
                mask_p = np.fliplr(mask_p)
            if np.random.rand() > 0.5:
                img_p = np.flipud(img_p)
                mask_p = np.flipud(mask_p)
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                img_p = np.rot90(img_p, k)
                mask_p = np.rot90(mask_p, k)
                
            # Elastic Deformation (Ronneberger et al., 2015)
            if np.random.rand() > 0.5:
                img_p, mask_p = apply_elastic_transform(img_p, mask_p, alpha=30.0, sigma=4.0)
                
        img_p = np.ascontiguousarray(img_p)
        mask_p = np.ascontiguousarray(mask_p)
        
        # Add channel dimension: (1, H, W)
        tensor_img = torch.from_numpy(img_p).unsqueeze(0).float()
        tensor_mask = torch.from_numpy(mask_p).unsqueeze(0).float()
        
        return tensor_img, tensor_mask

