import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

def compute_fov_metrics(pred_prob: np.ndarray, ground_truth: np.ndarray, fov_mask: np.ndarray, threshold: float = 0.5):
    """
    Computes medical image evaluation metrics strictly inside the Field of View (FOV) mask.
    
    Args:
        pred_prob: Predicted continuous probability map (H, W) in range [0, 1]
        ground_truth: Binary ground truth vessel mask (H, W) in {0, 1}
        fov_mask: Binary FOV mask (H, W) in {0, 1}
        threshold: Binarization decision threshold (default: 0.5)
        
    Returns:
        dict: Evaluation metrics (Accuracy, Sensitivity, Specificity, Precision, F1/Dice, IoU, AUC-ROC, AUC-PR)
    """
    # Extract only valid pixels inside the FOV
    fov_indices = np.where(fov_mask > 0.5)
    
    y_true = ground_truth[fov_indices].flatten().astype(int)
    y_prob = pred_prob[fov_indices].flatten()
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)   # Recall / TPR
    specificity = tn / (tn + fp + 1e-8)   # TNR
    precision = tp / (tp + fp + 1e-8)     # PPV
    f1_dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Area Under ROC Curve
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.0
        
    # Area Under Precision-Recall Curve
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(rec_curve, prec_curve)
    except Exception:
        auc_pr = 0.0
        
    return {
        "Accuracy": float(accuracy),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1_Dice": float(f1_dice),
        "IoU": float(iou),
        "AUC_ROC": float(auc_roc),
        "AUC_PR": float(auc_pr),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }
