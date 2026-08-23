import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def predict_full_image(model: torch.nn.Module, prep_img: np.ndarray, fov_mask: np.ndarray, in_patch_size: int = 572, device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    Performs full image vessel segmentation inference using Ronneberger et al. (2015) Overlap-Tile Strategy.
    Mirror-pads the image border by margin (92 px), slides input tile window of size 572x572, and places predicted
    388x388 output tiles into the full resolution map.
    """
    model.eval()
    h, w = prep_img.shape
    out_patch_size = in_patch_size - 184 # 388
    margin = 92 # (572 - 388) // 2
    
    # Calculate required padded dimensions so tiles completely cover h and w
    pad_h_extra = (out_patch_size - (h % out_patch_size)) % out_patch_size
    pad_w_extra = (out_patch_size - (w % out_patch_size)) % out_patch_size
    
    target_h = h + pad_h_extra
    target_w = w + pad_w_extra
    
    # Mirror-pad image: base margin + extra extension for full tiling
    padded_img = np.pad(prep_img, ((margin, margin + pad_h_extra), (margin, margin + pad_w_extra)), mode='reflect')
    
    prob_map = np.zeros((target_h, target_w), dtype=np.float32)
    
    patch_list = []
    coord_list = []
    
    for y in range(0, target_h, out_patch_size):
        for x in range(0, target_w, out_patch_size):
            p = padded_img[y : y + in_patch_size, x : x + in_patch_size]
            patch_list.append(p)
            coord_list.append((y, x))
            
    batch_size = 4
    patch_tensors = torch.from_numpy(np.array(patch_list)).unsqueeze(1).float() # Shape: (N, 1, 572, 572)
    
    predictions = []
    with torch.no_grad():
        for i in range(0, len(patch_tensors), batch_size):
            batch = patch_tensors[i:i+batch_size].to(device)
            out = model(batch)
            predictions.append(out.cpu().numpy())
            
    predictions = np.concatenate(predictions, axis=0)[:, 0, :, :] # Shape: (N, 388, 388)
    
    # Place non-overlapping predicted 388x388 tiles directly into final probability map
    for (y, x), pred_patch in zip(coord_list, predictions):
        prob_map[y : y + out_patch_size, x : x + out_patch_size] = pred_patch
        
    # Crop back to original dimensions (h, w)
    prob_map = prob_map[:h, :w]
    
    # Mask out non-retinal background outside FOV
    prob_map = prob_map * fov_mask
    return prob_map


def save_prediction_figure(rgb_img: np.ndarray, pred_prob: np.ndarray, fov_mask: np.ndarray, save_path: str, ground_truth: np.ndarray = None, filename: str = ""):
    """
    Saves visual comparison report figure for full image prediction.
    """
    pred_bin = (pred_prob >= 0.5).astype(np.float32) * fov_mask
    
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"1. Raw Fundus\n({filename})", fontsize=11)
        axes[0].axis('off')
        
        axes[1].imshow(ground_truth, cmap='gray')
        axes[1].set_title("2. Ground Truth GT", fontsize=11)
        axes[1].axis('off')
        
        axes[2].imshow(pred_prob, cmap='magma')
        axes[2].set_title("3. Prob Heatmap", fontsize=11)
        axes[2].axis('off')
        
        axes[3].imshow(pred_bin, cmap='gray')
        axes[3].set_title("4. Binary Pred (>0.5)", fontsize=11)
        axes[3].axis('off')
        
        # Error Map: Green = True Positive, Red = False Positive, Blue = False Negative
        error_map = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
        tp_mask = (ground_truth > 0.5) & (pred_bin > 0.5) & (fov_mask > 0.5)
        fp_mask = (ground_truth <= 0.5) & (pred_bin > 0.5) & (fov_mask > 0.5)
        fn_mask = (ground_truth > 0.5) & (pred_bin <= 0.5) & (fov_mask > 0.5)
        
        error_map[tp_mask] = [0, 255, 0]   # Green = TP
        error_map[fp_mask] = [255, 0, 0]   # Red = FP
        error_map[fn_mask] = [0, 0, 255]   # Blue = FN
        
        axes[4].imshow(error_map)
        axes[4].set_title("5. Error Map\n(G:TP, R:FP, B:FN)", fontsize=11)
        axes[4].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"1. Raw Fundus\n({filename})", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(pred_prob, cmap='magma')
        axes[1].set_title("2. Vessel Probability Heatmap", fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(pred_bin, cmap='gray')
        axes[2].set_title("3. Predicted Vessel Mask", fontsize=12)
        axes[2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
