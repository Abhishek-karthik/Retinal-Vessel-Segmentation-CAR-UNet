import os
import shutil
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_single_pair(args):
    src_img_path, src_mask_path, dest_orig_img, dest_orig_mask, dest_resized_img, dest_resized_mask, target_size = args
    
    # 1. Copy original files to data/FIVES/ if not already present
    if not os.path.exists(dest_orig_img):
        shutil.copyfile(src_img_path, dest_orig_img)
    if not os.path.exists(dest_orig_mask):
        shutil.copyfile(src_mask_path, dest_orig_mask)
        
    # 2. Resize to 512x512 using OpenCV (extremely fast C++ SIMD)
    if not os.path.exists(dest_resized_img) or not os.path.exists(dest_resized_mask):
        img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None and mask is not None:
            # INTER_AREA / INTER_LINEAR is best for downsampling
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            # INTER_NEAREST preserves exact binary values {0, 255}
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(dest_resized_img, img_resized)
            cv2.imwrite(dest_resized_mask, mask_resized)
    return True

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    raw_root = os.path.join(base_dir, "data", "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation", "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation")
    
    fives_dir = os.path.join(base_dir, "data", "FIVES")
    resized_dir = os.path.join(base_dir, "data", "FIVES_resized")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    target_size = (512, 512) # (width, height)
    num_workers = min(16, (os.cpu_count() or 4) * 2)
    
    print("=" * 60)
    print(f"FAST MULTITHREADED FIVES PREPARATION & RESIZING ({num_workers} WORKERS)")
    print(f"Raw source:  {raw_root}")
    print(f"Target size: {target_size}")
    print("=" * 60)
    
    splits = [("train", "train"), ("test", "test")]
    
    for split_name, raw_split in splits:
        raw_img_dir = os.path.join(raw_root, raw_split, "Original")
        raw_gt_dir = os.path.join(raw_root, raw_split, "Ground truth")
        
        fives_img_dir = os.path.join(fives_dir, split_name, "images")
        fives_mask_dir = os.path.join(fives_dir, split_name, "masks")
        os.makedirs(fives_img_dir, exist_ok=True)
        os.makedirs(fives_mask_dir, exist_ok=True)
        
        resized_img_dir = os.path.join(resized_dir, split_name, "images")
        resized_mask_dir = os.path.join(resized_dir, split_name, "masks")
        os.makedirs(resized_img_dir, exist_ok=True)
        os.makedirs(resized_mask_dir, exist_ok=True)
        
        img_files = sorted([f for f in os.listdir(raw_img_dir) if f.endswith(".png")])
        print(f"\nProcessing {split_name.upper()} split ({len(img_files)} images)...")
        
        tasks = []
        for fname in img_files:
            src_img_path = os.path.join(raw_img_dir, fname)
            src_mask_path = os.path.join(raw_gt_dir, fname)
            dest_orig_img = os.path.join(fives_img_dir, fname)
            dest_orig_mask = os.path.join(fives_mask_dir, fname)
            dest_resized_img = os.path.join(resized_img_dir, fname)
            dest_resized_mask = os.path.join(resized_mask_dir, fname)
            
            tasks.append((src_img_path, src_mask_path, dest_orig_img, dest_orig_mask, dest_resized_img, dest_resized_mask, target_size))
            
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_single_pair, tasks), total=len(tasks), desc=f"Resizing {split_name}"))
            
    # Verification
    print("\n" + "=" * 60)
    print("VERIFYING FIVES DATASET INTEGRITY")
    print("=" * 60)
    
    train_imgs = sorted(glob.glob(os.path.join(resized_dir, "train", "images", "*.png")))
    train_masks = sorted(glob.glob(os.path.join(resized_dir, "train", "masks", "*.png")))
    test_imgs = sorted(glob.glob(os.path.join(resized_dir, "test", "images", "*.png")))
    test_masks = sorted(glob.glob(os.path.join(resized_dir, "test", "masks", "*.png")))
    
    print(f"Resized Train Images: {len(train_imgs)}")
    print(f"Resized Train Masks:  {len(train_masks)}")
    print(f"Resized Test Images:  {len(test_imgs)}")
    print(f"Resized Test Masks:   {len(test_masks)}")
    
    assert len(train_imgs) == len(train_masks) == 600, f"Expected 600 train pairs, got {len(train_imgs)} images and {len(train_masks)} masks!"
    assert len(test_imgs) == len(test_masks) == 200, f"Expected 200 test pairs, got {len(test_imgs)} images and {len(test_masks)} masks!"
    
    for img_p, mask_p in zip(train_imgs, train_masks):
        assert os.path.basename(img_p) == os.path.basename(mask_p), f"Filename mismatch: {img_p} vs {mask_p}"
    for img_p, mask_p in zip(test_imgs, test_masks):
        assert os.path.basename(img_p) == os.path.basename(mask_p), f"Filename mismatch: {img_p} vs {mask_p}"
        
    print(">>> All 800 image-mask pairs match perfectly 1-to-1 with no missing files!")
    
    # Generate 3 sample visualization verification figure
    print("\nGenerating verification panel with 3 sample pairs...")
    sample_indices = [0, len(train_imgs) // 2, len(train_imgs) - 1]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, idx in enumerate(sample_indices):
        img_p = train_imgs[idx]
        mask_p = train_masks[idx]
        fname = os.path.basename(img_p)
        
        bgr = cv2.imread(img_p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        # Overlay
        overlay = rgb.copy()
        mask_bool = mask > 128
        overlay[mask_bool] = [0, 255, 0] # Green vessels
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Train Image: {fname}\n(512x512 RGB)", fontsize=11)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Ground Truth Mask: {fname}\n(512x512 Binary)", fontsize=11)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Vessel Overlay (Green)", fontsize=11)
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    check_fig_path = os.path.join(outputs_dir, "fives_dataset_check.png")
    plt.savefig(check_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved dataset verification figure to: {check_fig_path}")
    print("=" * 60)
    print("STEP 2 COMPLETED AND VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
