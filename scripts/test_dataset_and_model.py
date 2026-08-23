import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import RetinalPatchDataset
from src.unet_model import UNet
from src.losses import CombinedBCEDiceLoss
from src.metrics import compute_fov_metrics

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "DRIVE")
    output_dir = os.path.join(base_dir, "outputs", "visualizations")
    
    print("=" * 60)
    print("VALID CONVOLUTION U-NET INTEGRATION SANITY TEST")
    print("=" * 60)
    
    # 1. Test PyTorch Dataset
    print("\n[1/4] Testing RetinalPatchDataset (Valid Convolutions + Elastic Transform)...")
    train_dataset = RetinalPatchDataset(data_dir=data_dir, mode='train', in_patch_size=572, patches_per_img=10, augment=True)
    val_dataset = RetinalPatchDataset(data_dir=data_dir, mode='val', in_patch_size=572, patches_per_img=5, augment=False)
    
    print(f"  -> Train Dataset Total Patches: {len(train_dataset)}")
    print(f"  -> Val Dataset Total Patches  : {len(val_dataset)}")
    
    img_patch, mask_patch = train_dataset[0]
    print(f"  -> Sample Image Patch Shape   : {img_patch.shape} (dtype={img_patch.dtype}, min={img_patch.min():.2f}, max={img_patch.max():.2f})")
    print(f"  -> Sample Mask Patch Shape    : {mask_patch.shape} (dtype={mask_patch.dtype}, min={mask_patch.min():.2f}, max={mask_patch.max():.2f})")
    
    # Save a visualization of sample patches
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        img_p, mask_p = train_dataset[i * 2]
        axes[0, i].imshow(img_p[0].numpy(), cmap='gray')
        axes[0, i].set_title(f"Patch {i+1} Input (572x572)")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_p[0].numpy(), cmap='gray')
        axes[1, i].set_title(f"Patch {i+1} Target (388x388)")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    patch_viz_path = os.path.join(output_dir, "dataset_patches_sample.png")
    plt.savefig(patch_viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved patch visualization to: {patch_viz_path}")
    
    # 2. Test U-Net Model (Ronneberger 2015 exact base_filters=64)
    print("\n[2/4] Testing Valid Convolution U-Net Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    
    dummy_input = torch.randn(2, 1, 572, 572).to(device)
    dummy_output = model(dummy_input)
    print(f"  -> Model Device: {device}")
    print(f"  -> Input Tensor Shape : {dummy_input.shape}")
    print(f"  -> Output Tensor Shape: {dummy_output.shape}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Total Trainable Parameters: {num_params:,}")
    
    # 3. Test Loss Function
    print("\n[3/4] Testing Combined BCE + Dice Loss...")
    criterion = CombinedBCEDiceLoss()
    dummy_target = torch.randint(0, 2, (2, 1, 388, 388)).float().to(device)
    loss = criterion(dummy_output, dummy_target)
    print(f"  -> Dummy Batch Loss: {loss.item():.4f}")
    
    # 4. Test Metrics Calculation
    print("\n[4/4] Testing FOV Metrics Calculation...")
    dummy_pred_np = dummy_output[0, 0].detach().cpu().numpy()
    dummy_gt_np = dummy_target[0, 0].detach().cpu().numpy()
    dummy_fov_np = np.ones_like(dummy_gt_np)
    
    metrics = compute_fov_metrics(dummy_pred_np, dummy_gt_np, dummy_fov_np)
    print("  -> Metrics computed successfully:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"      {k:12s}: {v:.4f}")
            
    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

