import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from src.data_loader_fives import FIVESPatchDataset, load_fives_image_pairs
from src.unet_model import UNet
from src.losses import CombinedBCEDiceLoss
from src.metrics import compute_fov_metrics
from src.preprocessing_fives import preprocess_image, generate_fov_mask
from src.utils import predict_full_image, save_prediction_figure

torch.backends.cudnn.benchmark = True

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_val_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for imgs, masks in dataloader:
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

def evaluate_full_test_set(model, test_pairs, device, output_pred_dir, max_eval_images=20, max_save_viz=5):
    """
    Evaluates model on test set using Overlap-Tile inference (Ronneberger et al., 2015)
    and computes full benchmark metrics inside FOV.
    """
    os.makedirs(output_pred_dir, exist_ok=True)
    model.eval()
    
    eval_subset = test_pairs[:max_eval_images] if max_eval_images else test_pairs
    all_results = []
    print(f"\nEvaluating on {len(eval_subset)} Full-Resolution Held-Out Test Images...", flush=True)
    
    for i, (img_path, mask_path) in enumerate(eval_subset):
        fname = os.path.basename(img_path)
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        prep_img = preprocess_image(raw_rgb)
        
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (raw_mask > 128).astype(np.float32)
        fov_mask = generate_fov_mask(raw_rgb)
        
        pred_prob = predict_full_image(model, prep_img, fov_mask, in_patch_size=284, device=device)
        
        metrics = compute_fov_metrics(pred_prob, gt_mask, fov_mask)
        metrics["filename"] = fname
        all_results.append(metrics)
        
        if i < max_save_viz:
            save_path = os.path.join(output_pred_dir, f"pred_{fname}")
            save_prediction_figure(raw_rgb, pred_prob, fov_mask, save_path, ground_truth=gt_mask, filename=fname)
            
        if (i + 1) % 5 == 0 or (i + 1) == len(eval_subset):
            print(f"  Test Eval Progress: [{i+1}/{len(eval_subset)}] images processed", flush=True)
            
    df = pd.DataFrame(all_results)
    numeric_cols = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Dice", "IoU", "AUC_ROC", "AUC_PR"]
    summary_metrics = df[numeric_cols].mean().to_dict()
    return summary_metrics, df

def main():
    parser = argparse.ArgumentParser(description="Train Baseline U-Net on FIVES Dataset (Review 2 Dataset Upgrade)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patches_per_img", type=int, default=2, help="Patches per image for training")
    parser.add_argument("--base_filters", type=int, default=64, help="U-Net base filters")
    parser.add_argument("--test_images", type=int, default=20, help="Number of test images to evaluate")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "FIVES_resized")
    saved_models_dir = os.path.join(base_dir, "outputs", "saved_models")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    viz_dir = os.path.join(base_dir, "outputs", "unet_fives_predictions")
    
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print("BASELINE U-NET TRAINING ENGINE ON FIVES DATASET", flush=True)
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}", flush=True)
    print("=" * 60, flush=True)

    # 1. Datasets & DataLoaders
    train_dataset = FIVESPatchDataset(data_dir=data_dir, mode='train', in_patch_size=284, patches_per_img=args.patches_per_img, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = FIVESPatchDataset(data_dir=data_dir, mode='val', in_patch_size=284, patches_per_img=2, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    test_pairs = load_fives_image_pairs(data_dir, mode='test')

    print(f"Total Train Patches: {len(train_dataset):,} | Total Val Patches: {len(val_dataset):,} | Test Images: {len(test_pairs)}", flush=True)

    # 2. Model, Loss, Optimizer
    model = UNet(in_channels=1, out_channels=1, base_filters=args.base_filters).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_acc": [], "val_auc": []}

    start_time = time.time()
    best_checkpoint_path = os.path.join(saved_models_dir, "unet_fives.pt")
    models_checkpoint_path = os.path.join(models_dir, "unet_fives.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_val_epoch(model, val_loader, criterion, device)
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
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}", flush=True)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            state_dict_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "history": history
            }
            torch.save(state_dict_payload, best_checkpoint_path)
            torch.save(state_dict_payload, models_checkpoint_path)
            print(f"  >>> Checkpoint Saved to {best_checkpoint_path} (Best Val Dice: {best_val_dice:.4f})", flush=True)

    total_time = time.time() - start_time
    print("=" * 60, flush=True)
    print(f"TRAINING COMPLETE IN {total_time/60:.2f} MINUTES! Best Val Dice: {best_val_dice:.4f}", flush=True)
    print("=" * 60, flush=True)

    # 3. Load Best Model and Run Full Evaluation on Held-out Test Set
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_metrics, df_test = evaluate_full_test_set(model, test_pairs, device, viz_dir, max_eval_images=args.test_images, max_save_viz=5)
    
    # Save Metrics
    metrics_json_path = os.path.join(results_dir, "unet_fives_metrics.json")
    metrics_csv_path = os.path.join(results_dir, "unet_fives_metrics.csv")
    
    with open(metrics_json_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    df_test.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved test metrics to {metrics_json_path} and {metrics_csv_path}", flush=True)
    print("\n--- U-NET ON FIVES FINAL TEST BENCHMARK RESULTS ---", flush=True)
    for k, v in test_metrics.items():
        print(f"{k:>15}: {v:.4f}", flush=True)

if __name__ == "__main__":
    main()
