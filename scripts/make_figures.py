"""
Generates the report figures from the finished runs in results/runs/.

    outputs/final_comparison_figure.png  one test image per disease (AMD, DR, Glaucoma, Normal):
                                         RGB | ground truth | U-Net | CAR-UNet | U-Net errors | CAR-UNet errors
    outputs/training_curves.png          training loss and validation Dice per epoch (mean over seeds)

The example image for each disease is the one with the MEDIAN CAR-UNet Dice in that group,
so the figure shows typical cases rather than hand-picked best ones.
"""
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # must be imported before numpy on this environment (DLL load order)
import glob
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.unet_model import UNet
from src.car_unet import CARUNet
from src.preprocessing_fives import preprocess_image, generate_fov_mask
from src.utils import predict_full_image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(BASE_DIR, "results", "runs")
TEST_DIR = os.path.join(BASE_DIR, "data", "FIVES_resized", "test")
MODELS = [("unet", "U-Net", UNet, "tab:blue"), ("car_unet", "CAR-UNet", CARUNet, "tab:green")]


def error_map(pred_bin, gt, fov):
    """Green = correct vessel (TP), red = false vessel (FP), blue = missed vessel (FN)."""
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    inside = fov > 0.5
    out[(gt > 0.5) & (pred_bin > 0.5) & inside] = [0, 200, 0]
    out[(gt <= 0.5) & (pred_bin > 0.5) & inside] = [230, 0, 0]
    out[(gt > 0.5) & (pred_bin <= 0.5) & inside] = [0, 90, 255]
    return out


def comparison_figure(seed, device):
    per_image = {key: pd.read_csv(os.path.join(RUNS_DIR, f"{key}_seed{seed}", "test_metrics_per_image.csv")).set_index("filename")
                 for key, *_ in MODELS}
    car = per_image["car_unet"]
    examples = []
    for disease, group in car.groupby("disease"):
        group = group.sort_values("F1_Dice")
        examples.append((disease, group.index[len(group) // 2]))

    models = {}
    for key, label, cls, _ in MODELS:
        model = cls(in_channels=1, out_channels=1, base_filters=64).to(device)
        ckpt = torch.load(os.path.join(BASE_DIR, "models", f"{key}_seed{seed}.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        models[key] = model.eval()

    fig, axes = plt.subplots(len(examples), 6, figsize=(24, 5.0 * len(examples)))
    for row, (disease, fname) in enumerate(examples):
        rgb = cv2.cvtColor(cv2.imread(os.path.join(TEST_DIR, "images", fname)), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(os.path.join(TEST_DIR, "masks", fname), cv2.IMREAD_GRAYSCALE) > 128).astype(np.float32)
        fov = generate_fov_mask(rgb)
        prep = preprocess_image(rgb)
        preds = {key: (predict_full_image(models[key], prep, fov, in_patch_size=284, device=device, batch_size=16) >= 0.5).astype(np.float32) * fov
                 for key in models}

        panels = [(rgb, f"{disease} — {fname}\nRGB fundus"), (gt, "Ground truth")]
        panels += [(preds[k], f"{lbl}\nDice {per_image[k].loc[fname, 'F1_Dice']:.3f} | Sens {per_image[k].loc[fname, 'Sensitivity']:.3f}") for k, lbl, *_ in MODELS]
        panels += [(error_map(preds[k], gt, fov), f"{lbl} errors\ngreen TP · red FP · blue FN") for k, lbl, *_ in MODELS]
        for col, (img, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

    fig.suptitle(f"FIVES test set — typical case per disease (median CAR-UNet Dice), seed {seed}", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.5)
    path = os.path.join(BASE_DIR, "outputs", "final_comparison_figure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path} (examples: {examples})")


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for key, label, _, color in MODELS:
        histories = []
        for path in sorted(glob.glob(os.path.join(RUNS_DIR, f"{key}_seed*", "history.json"))):
            if os.path.basename(os.path.dirname(path)).split("_seed")[-1].isdigit():
                with open(path) as f:
                    histories.append(json.load(f))
        if not histories:
            continue
        n = min(len(h["epoch"]) for h in histories)
        epochs = np.arange(1, n + 1)
        for ax, metric in zip(axes, ["train_loss", "val_dice"]):
            values = np.array([h[metric][:n] for h in histories])
            mean, std = values.mean(0), values.std(0)
            ax.plot(epochs, mean, color=color, label=f"{label} ({len(histories)} seed{'s' if len(histories) > 1 else ''})")
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
    axes[0].set(title="Training loss (0.5·BCE + 0.5·Dice)", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation Dice (120 full images, 30 per disease)", xlabel="Epoch", ylabel="Dice")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "outputs", "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Which seed's models to show in the comparison figure")
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU (e.g. while the GPU is busy training)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    training_curves()
    comparison_figure(args.seed, device)


if __name__ == "__main__":
    main()
