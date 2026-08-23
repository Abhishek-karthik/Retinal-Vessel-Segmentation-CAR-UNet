import os
import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_image
from src.dataset import load_drive_image_pairs
from src.unet_model import UNet
from src.metrics import compute_fov_metrics
from src.utils import predict_full_image, save_prediction_figure

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "DRIVE")
    checkpoint_path = os.path.join(base_dir, "outputs", "saved_models", "best_unet.pth")
    viz_dir = os.path.join(base_dir, "outputs", "visualizations")
    report_csv_path = os.path.join(base_dir, "outputs", "metrics_report.csv")
    
    os.makedirs(viz_dir, exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Run scripts/train.py first!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("RONNEBERGER U-NET EVALUATION & OVERLAP-TILE PREDICTION ENGINE")
    print(f"Device: {device} | Checkpoint: {checkpoint_path}")
    print("=" * 60)

    # 1. Load Trained Model
    model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Successfully loaded trained checkpoint from epoch {checkpoint.get('epoch', 'N/A')} (Best Val Dice: {checkpoint.get('best_val_dice', 0.0):.4f})")

    # 2. Evaluate on Validation Set (with Ground Truth)
    print("\n[1/2] Evaluating on Validation Images (Ground Truth Benchmark)...")
    val_triplets = load_drive_image_pairs(data_dir, mode='val')
    
    val_results = []
    
    for img_path, manual_path, mask_path in tqdm(val_triplets, desc="Val Evaluation"):
        fname = os.path.basename(img_path)
        raw_rgb = np.array(Image.open(img_path))
        prep_img = preprocess_image(raw_rgb)
        
        gt_mask = np.array(Image.open(manual_path)).astype(np.float32) / 255.0
        fov_mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
        
        gt_mask = (gt_mask > 0.5).astype(np.float32)
        fov_mask = (fov_mask > 0.5).astype(np.float32)
        
        # Predict full image via Ronneberger Overlap-Tile Strategy
        pred_prob = predict_full_image(model, prep_img, fov_mask, in_patch_size=284, device=device)
        
        # Compute metrics inside FOV
        metrics = compute_fov_metrics(pred_prob, gt_mask, fov_mask)
        metrics["filename"] = fname
        val_results.append(metrics)
        
        # Save visualization
        save_path = os.path.join(viz_dir, f"val_pred_{fname.replace('.tif', '.png')}")
        save_prediction_figure(raw_rgb, pred_prob, fov_mask, save_path, ground_truth=gt_mask, filename=fname)

    df_val = pd.DataFrame(val_results)
    cols = ["filename", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Dice", "IoU", "AUC_ROC", "AUC_PR"]
    df_val = df_val[cols]
    
    # Calculate Average Metrics
    avg_row = df_val.mean(numeric_only=True).to_dict()
    avg_row["filename"] = "AVERAGE"
    df_val = pd.concat([df_val, pd.DataFrame([avg_row])], ignore_index=True)
    
    df_val.to_csv(report_csv_path, index=False)
    print(f"Validation metrics report saved to: {report_csv_path}")
    print("\n--- VALIDATION BENCHMARK METRICS SUMMARY ---")
    print(df_val.to_string(index=False))

    # 3. Generate Predictions on Held-out Test Set (20 Images)
    print("\n[2/2] Generating Predictions for Held-out Test Images (20 Images)...")
    test_triplets = load_drive_image_pairs(data_dir, mode='test')
    
    for img_path, _, mask_path in tqdm(test_triplets, desc="Test Predictions"):
        fname = os.path.basename(img_path)
        raw_rgb = np.array(Image.open(img_path))
        prep_img = preprocess_image(raw_rgb)
        fov_mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
        fov_mask = (fov_mask > 0.5).astype(np.float32)
        
        pred_prob = predict_full_image(model, prep_img, fov_mask, in_patch_size=284, device=device)
        
        save_path = os.path.join(viz_dir, f"test_pred_{fname.replace('.tif', '.png')}")
        save_prediction_figure(raw_rgb, pred_prob, fov_mask, save_path, filename=fname)
        
    print("=" * 60)
    print("ALL EVALUATIONS AND TEST PREDICTIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

