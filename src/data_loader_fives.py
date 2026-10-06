import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from PIL import Image
import cv2

from src.preprocessing_fives import preprocess_image, preprocess_image_uint8, apply_elastic_transform, generate_fov_mask

DISEASE_NAMES = {"A": "AMD", "D": "DR", "G": "Glaucoma", "N": "Normal"}

def get_disease_code(path: str) -> str:
    """
    Returns the FIVES disease code from a filename such as '12_A.png' -> 'A'
    (A = AMD, D = Diabetic Retinopathy, G = Glaucoma, N = Normal).
    """
    return os.path.splitext(os.path.basename(path))[0].split("_")[-1][0].upper()

def _sorted_pairs(img_dir: str, mask_dir: str):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    assert len(img_paths) == len(mask_paths), f"Mismatch: {len(img_paths)} imgs vs {len(mask_paths)} masks"
    for img_p, mask_p in zip(img_paths, mask_paths):
        assert os.path.basename(img_p) == os.path.basename(mask_p), f"Unpaired files: {img_p} vs {mask_p}"
    return list(zip(img_paths, mask_paths))

def load_fives_image_pairs(data_dir: str, mode: str = 'train', val_ratio: float = 0.2, split_seed: int = 42):
    """
    Loads image and ground truth mask paths from FIVES dataset (resized to 512x512).
    Performs an image-level, disease-stratified train/val split on the 600 training images
    (prevents data leakage and keeps AMD / DR / Glaucoma / Normal equally represented in validation).

    Args:
        data_dir: Path to FIVES or FIVES_resized directory (e.g. data/FIVES_resized)
        mode: 'train', 'val', or 'test'
        val_ratio: Fraction of each disease group used for validation (default: 0.2 -> 480 train / 120 val, 30 val per disease)
        split_seed: Seed for the train/val split. Kept fixed so every model and seed sees the same split.

    Returns:
        list of tuples: (image_path, mask_path)
    """
    if mode in ['train', 'val']:
        pairs = _sorted_pairs(os.path.join(data_dir, "train", "images"), os.path.join(data_dir, "train", "masks"))

        rng = np.random.RandomState(split_seed)
        train_pairs, val_pairs = [], []
        for code in sorted(set(get_disease_code(p[0]) for p in pairs)):
            group = [p for p in pairs if get_disease_code(p[0]) == code]
            order = rng.permutation(len(group))
            val_count = int(round(len(group) * val_ratio))
            val_pairs += [group[i] for i in order[:val_count]]
            train_pairs += [group[i] for i in order[val_count:]]

        return sorted(train_pairs) if mode == 'train' else sorted(val_pairs)

    elif mode == 'test':
        return _sorted_pairs(os.path.join(data_dir, "test", "images"), os.path.join(data_dir, "test", "masks"))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def extract_random_patches(image: np.ndarray, mask: np.ndarray, fov_mask: np.ndarray, in_patch_size: int = 284, num_patches: int = 10):
    """
    Extracts random sub-patches for valid convolutions (Ronneberger et al., 2015).
    Input patch size: 284x284
    Target output mask size: 100x100 (centered inside input patch, 92 margin on each side)
    """
    out_patch_size = in_patch_size - 184 # 100
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
        max_y = h_pad - in_patch_size
        max_x = w_pad - in_patch_size
        ys = np.random.randint(0, max(1, max_y + 1), size=num_patches)
        xs = np.random.randint(0, max(1, max_x + 1), size=num_patches)
    else:
        indices = np.random.choice(len(valid_y), size=num_patches, replace=True)
        ys = valid_y[indices] - (in_patch_size // 2)
        xs = valid_x[indices] - (in_patch_size // 2)
        
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

class FIVESPatchDataset(Dataset):
    """
    PyTorch Dataset serving retinal vessel image & ground truth patches for FIVES dataset.
    Extracts valid convolution patches (284x284 -> 100x100) with data augmentation.
    Preprocessed images are cached in memory; call resample() at the start of every epoch
    to draw a fresh set of random patch locations.
    """
    def __init__(self, data_dir: str, mode: str = 'train', in_patch_size: int = 284, patches_per_img: int = 10, augment: bool = True, image_pairs=None):
        super().__init__()
        self.mode = mode
        self.in_patch_size = in_patch_size
        self.out_patch_size = in_patch_size - 184
        self.patches_per_img = patches_per_img
        self.augment = augment

        if image_pairs is None:
            image_pairs = load_fives_image_pairs(data_dir, mode=mode)

        self.prep_imgs, self.gt_masks, self.fov_masks = [], [], []
        print(f"Loading {mode.upper()} dataset ({len(image_pairs)} images, {patches_per_img} patches/image)...")
        for img_path, mask_path in image_pairs:
            raw_rgb = cv2.imread(img_path)
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
            self.prep_imgs.append(preprocess_image_uint8(raw_rgb))  # cached as uint8 (4x less memory)

            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.gt_masks.append((raw_mask > 128).astype(np.uint8))
            self.fov_masks.append(generate_fov_mask(raw_rgb).astype(np.uint8))

        self.resample()

    def resample(self):
        """Draws a new random set of FOV-centred patches from every cached image."""
        all_img_patches = []
        all_mask_patches = []
        for prep_img, manual_gt, fov_mask in zip(self.prep_imgs, self.gt_masks, self.fov_masks):
            imgs, masks = extract_random_patches(prep_img, manual_gt, fov_mask, in_patch_size=self.in_patch_size, num_patches=self.patches_per_img)
            all_img_patches.append(imgs)
            all_mask_patches.append(masks)

        # Convert to float32 exactly as preprocess_image / the float masks would give
        self.images = np.concatenate(all_img_patches, axis=0).astype(np.float32) / 255.0 # (N, in_patch_size, in_patch_size)
        self.masks = np.concatenate(all_mask_patches, axis=0).astype(np.float32)          # (N, out_patch_size, out_patch_size)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_p = self.images[idx].copy()
        mask_p = self.masks[idx].copy()
        
        if self.augment and self.mode == 'train':
            if np.random.rand() < 0.35:
                img_p, mask_p = apply_elastic_transform(img_p, mask_p, alpha=25.0, sigma=4.0, random_state=np.random)  # seeded global RNG
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
                
        img_p = np.ascontiguousarray(img_p)
        mask_p = np.ascontiguousarray(mask_p)
        
        # Add channel dimension: (1, H, W)
        tensor_img = torch.from_numpy(img_p).unsqueeze(0).float()
        tensor_mask = torch.from_numpy(mask_p).unsqueeze(0).float()
        
        return tensor_img, tensor_mask

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "FIVES_resized")
    
    train_pairs = load_fives_image_pairs(data_dir, mode='train')
    val_pairs = load_fives_image_pairs(data_dir, mode='val')
    test_pairs = load_fives_image_pairs(data_dir, mode='test')
    
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images:   {len(val_pairs)}")
    print(f"Test images:  {len(test_pairs)}")
