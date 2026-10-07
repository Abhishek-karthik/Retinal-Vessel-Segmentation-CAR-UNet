"""
Generates the report figures from the finished runs in results/runs/.

    <output>                 one test image per disease (AMD, DR, Glaucoma, Normal):
                             RGB | ground truth | each model's prediction | each model's error map
    <zoom_output>            the same examples, zoomed on the 128x128 region with the most thin vessels
    <curves_output>          training curves (mean ± std over seeds)

The example image for each disease is the one with the MEDIAN Dice of the selection model in that group,
and the zoom region is chosen from the ground truth only (most thin-vessel pixels), so nothing is hand-picked.

    python scripts/make_figures.py                                    # U-Net vs CAR-UNet (default)
    python scripts/make_figures.py --models car_unet car_unet_da_cldice_1024 --setting tta_thr \\
        --output final_model_comparison.png --zoom_output thin_vessel_zoom.png --curves_output improvement_curves.png
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
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.train_fives import MODEL_SPECS, MODEL_RESOLUTION, EVAL_RESOLUTION, prepare_eval_item, predict_item

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(BASE_DIR, "results", "runs")
TEST_DIR = os.path.join(BASE_DIR, "data", "FIVES_resized", "test")
COLORS = {"unet": "tab:blue", "car_unet": "tab:green", "car_unet_da": "tab:orange",
          "car_unet_da_cldice": "tab:red", "car_unet_da_cldice_1024": "tab:purple"}
FLIPS = [(), (1,), (0,), (0, 1)]
ZOOM = 128


def error_map(pred_bin, gt, fov):
    """Green = correct vessel (TP), red = false vessel (FP), blue = missed vessel (FN)."""
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    inside = fov > 0.5
    out[(gt > 0.5) & (pred_bin > 0.5) & inside] = [0, 200, 0]
    out[(gt <= 0.5) & (pred_bin > 0.5) & inside] = [230, 0, 0]
    out[(gt > 0.5) & (pred_bin <= 0.5) & inside] = [0, 90, 255]
    return out


def thin_vessel_window(gt):
    """Top-left corner of the ZOOMxZOOM window containing the most thin-vessel pixels (vessel radius <= 1.5 px)."""
    thin = ((gt > 0.5) & (distance_transform_edt(gt > 0.5) <= 1.5)).astype(np.float32)
    counts = cv2.boxFilter(thin, -1, (ZOOM, ZOOM), normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    counts = counts[: gt.shape[0] - ZOOM + 1, : gt.shape[1] - ZOOM + 1]
    y, x = np.unravel_index(np.argmax(counts), counts.shape)
    return int(y), int(x)


def load_model(key, seed, device):
    _, cls, kwargs, _ = MODEL_SPECS[key]
    model = cls(in_channels=1, out_channels=1, base_filters=64, **kwargs).to(device)
    ckpt = torch.load(os.path.join(BASE_DIR, "models", f"{key}_seed{seed}.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def run_metrics(key, seed, setting):
    """Per-image metrics table and decision threshold for one run and prediction setting."""
    run_dir = os.path.join(RUNS_DIR, f"{key}_seed{seed}")
    if setting == "base":
        return pd.read_csv(os.path.join(run_dir, "test_metrics_per_image.csv")).set_index("filename"), 0.5
    post = json.load(open(os.path.join(run_dir, "post", "summary.json")))
    df = pd.read_csv(os.path.join(run_dir, "post", f"test_metrics_{setting}.csv")).set_index("filename")
    return df, post["settings"][setting]["threshold"]


def comparison_figures(keys, seed, setting, select_key, output, zoom_output, device):
    labels = {k: MODEL_SPECS[k][0] for k in keys}
    metrics, thresholds = {}, {}
    for k in keys:
        metrics[k], thresholds[k] = run_metrics(k, seed, setting)
    examples = []
    for disease, group in metrics[select_key].groupby("disease"):
        group = group.sort_values("F1_Dice")
        examples.append((disease, group.index[len(group) // 2]))
    models = {k: load_model(k, seed, device) for k in keys}

    n_cols = 2 + 2 * len(keys)
    fig, axes = plt.subplots(len(examples), n_cols, figsize=(4 * n_cols, 5.0 * len(examples)))
    zfig, zaxes = plt.subplots(len(examples), n_cols, figsize=(3.2 * n_cols, 3.6 * len(examples)))
    for row, (disease, fname) in enumerate(examples):
        img_path, mask_path = os.path.join(TEST_DIR, "images", fname), os.path.join(TEST_DIR, "masks", fname)
        preds = {}
        for k in keys:
            item = prepare_eval_item(img_path, mask_path, MODEL_RESOLUTION.get(k, EVAL_RESOLUTION), keep_raw=True)
            flips = FLIPS if setting.startswith("tta") else [()]
            prob = np.mean([predict_item(models[k], item, device, 16, flip_axes=f)[0] for f in flips], axis=0)
            preds[k] = (prob >= thresholds[k]).astype(np.float32) * item["fov"]
        rgb, gt, fov = item["raw"], item["gt"].astype(np.float32), item["fov"].astype(np.float32)
        y, x = thin_vessel_window(gt)

        panels = [(rgb, f"{disease} — {fname}\nRGB fundus"), (gt, "Ground truth")]
        panels += [(preds[k], f"{labels[k]}\nDice {metrics[k].loc[fname, 'F1_Dice']:.3f} | Sens {metrics[k].loc[fname, 'Sensitivity']:.3f}") for k in keys]
        panels += [(error_map(preds[k], gt, fov), f"{labels[k]} errors\ngreen TP · red FP · blue FN") for k in keys]
        for col, (img, title) in enumerate(panels):
            for ax_grid, crop, size in [(axes, False, 11), (zaxes, True, 9)]:
                ax = ax_grid[row, col]
                shown = img[y:y + ZOOM, x:x + ZOOM] if crop else img
                ax.imshow(shown, cmap="gray" if shown.ndim == 2 else None, interpolation="nearest")
                ax.set_title(title, fontsize=size)
                ax.axis("off")
            if col == 0:  # mark the zoom window on the full image
                axes[row, 0].add_patch(plt.Rectangle((x, y), ZOOM, ZOOM, fill=False, edgecolor="yellow", linewidth=1.5))

    set_name = {"base": "threshold 0.5", "thr": "tuned threshold", "tta": "TTA", "tta_thr": "TTA + tuned threshold"}[setting]
    title = f"FIVES test set — typical case per disease (median {labels[select_key]} Dice), seed {seed}, {set_name}"
    for f, out, extra in [(fig, output, ""), (zfig, zoom_output, f" — zoom on the {ZOOM}x{ZOOM} region with most thin vessels (yellow box)")]:
        f.suptitle(title + extra, fontsize=14, fontweight="bold", y=0.995)
        f.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.5)
        path = os.path.join(BASE_DIR, "outputs", out)
        f.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(f)
        print(f"Saved {path}")
    print(f"Examples: {examples}")


def training_curves(keys, output):
    same_loss = len({MODEL_SPECS[k][3] for k in keys}) == 1
    panels = [("train_loss", "Training loss"), ("val_dice", "Validation Dice")] if same_loss else \
             [("val_dice", "Validation Dice"), ("val_sensitivity", "Validation sensitivity")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for key in keys:
        histories = []
        for path in sorted(glob.glob(os.path.join(RUNS_DIR, f"{key}_seed*", "history.json"))):
            if os.path.basename(os.path.dirname(path))[len(key) + 5:].isdigit():  # exact model key, no tags
                with open(path) as f:
                    histories.append(json.load(f))
        if not histories:
            continue
        n = min(len(h["epoch"]) for h in histories)
        epochs = np.arange(1, n + 1)
        for ax, (metric, _) in zip(axes, panels):
            values = np.array([h[metric][:n] for h in histories])
            mean, std = values.mean(0), values.std(0)
            label = f"{MODEL_SPECS[key][0]} ({len(histories)} seed{'s' if len(histories) > 1 else ''})"
            ax.plot(epochs, mean, color=COLORS.get(key), label=label)
            ax.fill_between(epochs, mean - std, mean + std, color=COLORS.get(key), alpha=0.15)
    for ax, (_, name) in zip(axes, panels):
        suffix = " (0.5·BCE + 0.5·Dice)" if name == "Training loss" else " (120 full images, 30 per disease)"
        ax.set(title=name + suffix, xlabel="Epoch", ylabel=name.split()[-1])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "outputs", output)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["unet", "car_unet"])
    parser.add_argument("--seed", type=int, default=0, help="Which seed's models to show in the comparison figure")
    parser.add_argument("--setting", default="base", choices=["base", "thr", "tta", "tta_thr"])
    parser.add_argument("--select_by", default=None, help="Model whose median Dice picks the examples (default: CAR-UNet if shown, else last model)")
    parser.add_argument("--output", default="final_comparison_figure.png")
    parser.add_argument("--zoom_output", default="final_comparison_zoom.png")
    parser.add_argument("--curves_output", default="training_curves.png")
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU (e.g. while the GPU is busy training)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    select_key = args.select_by or ("car_unet" if "car_unet" in args.models else args.models[-1])
    training_curves(args.models, args.curves_output)
    comparison_figures(args.models, args.seed, args.setting, select_key, args.output, args.zoom_output, device)


if __name__ == "__main__":
    main()
