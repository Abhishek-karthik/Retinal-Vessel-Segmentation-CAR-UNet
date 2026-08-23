import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import RetinalPatchDataset
from src.unet_model import UNet
from src.losses import CombinedBCEDiceLoss
from src.metrics import compute_fov_metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(dataloader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Validating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            
            running_loss += loss.item() * imgs.size(0)
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_masks = np.concatenate(all_masks, axis=0).flatten()
    fov_mask = np.ones_like(all_masks)
    
    metrics = compute_fov_metrics(all_preds, all_masks, fov_mask)
    metrics["val_loss"] = val_loss
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train Baseline U-Net on DRIVE Retinal Vessel Dataset (Ronneberger et al., 2015)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--in_patch_size", type=int, default=284, help="Input patch size")
    parser.add_argument("--patches_per_img", type=int, default=10, help="Patches per image")
    parser.add_argument("--base_filters", type=int, default=64, help="U-Net base filters")
    args = parser.parse_args()


    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "DRIVE")
    model_save_dir = os.path.join(base_dir, "outputs", "saved_models")
    viz_dir = os.path.join(base_dir, "outputs", "visualizations")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("RONNEBERGER U-NET TRAINING ENGINE (VALID CONVOLUTIONS & ELASTIC DEFORMATION)")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Filters: {args.base_filters}")
    print("=" * 60)
    
    # Data Loaders
    print("Loading Training Dataset (16 Images)...")
    train_dataset = RetinalPatchDataset(data_dir=data_dir, mode='train', in_patch_size=args.in_patch_size, patches_per_img=args.patches_per_img, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print("Loading Validation Dataset (4 Images)...")
    val_dataset = RetinalPatchDataset(data_dir=data_dir, mode='val', in_patch_size=args.in_patch_size, patches_per_img=20, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Total Train Samples: {len(train_dataset):,} | Total Val Samples: {len(val_dataset):,}")
    
    # Model, Optimizer, Loss
    model = UNet(in_channels=1, out_channels=1, base_filters=args.base_filters).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_acc": [], "val_auc": []}
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        val_loss = val_metrics["val_loss"]
        val_dice = val_metrics["F1_Dice"]
        val_acc = val_metrics["Accuracy"]
        val_auc = val_metrics["AUC_ROC"]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save Best Model Checkpoint based on Validation Dice Score
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint_path = os.path.join(model_save_dir, "best_unet.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "history": history
            }, checkpoint_path)
            print(f"  >>> Checkpoint Saved! Best Val Dice Score: {best_val_dice:.4f}")
            
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"TRAINING COMPLETED IN {total_time/60:.2f} MINUTES! Best Val Dice: {best_val_dice:.4f}")
    print("=" * 60)
    
    # Plot & Save Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(range(1, args.epochs + 1), history["val_loss"], label="Val Loss", color="red", linestyle="--", linewidth=2)
    plt.title("Loss Progression (BCE + Dice)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), history["val_dice"], label="Val Dice Score", color="green", linewidth=2)
    plt.plot(range(1, args.epochs + 1), history["val_acc"], label="Val Accuracy", color="orange", linestyle="--", linewidth=2)
    plt.title("Validation Metrics Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    curve_path = os.path.join(viz_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves plot to: {curve_path}")

if __name__ == "__main__":
    main()

