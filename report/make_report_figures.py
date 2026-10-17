"""
Generates the report figures that differ from the slide versions:
  dual_attention_block.png        larger text, readable at column width
  replicated_samples_all.png      all four disease groups (the slides show three for space)
  compare_all_diseases.png        all four disease groups
The remaining report figures are copies of presentation/figures/* (pipeline.png with its title cropped).

Run from the project root:  python report/make_report_figures.py
"""
import os
import sys
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath("."))
import torch  # import before numpy on this machine
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

from scripts.train_fives import MODEL_SPECS, prepare_eval_item, predict_item
from scripts.summarize_post import selected_setting
from src.metrics import compute_fov_metrics, compute_cldice

OUT = os.path.join("report", "figures")
os.makedirs(OUT, exist_ok=True)
RUNS = "results/runs"
BLUE, ORANGE = "#4C72B0", "#DD8452"
DISEASE = {"A": "AMD", "D": "DR", "G": "Glaucoma", "N": "Normal"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold"})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved", name)


def load_model(key, seed):
    _, cls, kwargs, _ = MODEL_SPECS[key]
    m = cls(in_channels=1, out_channels=1, base_filters=64, **kwargs).to(device)
    m.load_state_dict(torch.load(f"models/{key}_seed{seed}.pt", map_location=device, weights_only=False)["model_state_dict"])
    return m.eval()


def error_map(pred, gt, fov):
    out = np.full((*gt.shape, 3), 0, np.uint8)
    inside = fov > 0.5
    out[(gt > 0.5) & (pred > 0) & inside] = [0, 200, 0]
    out[(gt <= 0.5) & (pred > 0) & inside] = [230, 0, 0]
    out[(gt > 0.5) & (pred == 0) & inside] = [0, 110, 255]
    return out


def thin_window(gt, size=128):
    thin = ((gt > 0.5) & (distance_transform_edt(gt > 0.5) <= 1.5)).astype(np.float32)
    c = cv2.boxFilter(thin, -1, (size, size), normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    c = c[:gt.shape[0] - size + 1, :gt.shape[1] - size + 1]
    return np.unravel_index(np.argmax(c), c.shape)


# 0. Dual attention block diagram (drawn larger than the slide version so it stays readable at column width) ----
steps = [("Input feature map $F$\n($C \\times H \\times W$)", "#8C8C8C"),
         ("Channel attention\navg + max pooling, shared 1D conv\n$\\rightarrow$ one weight per channel", BLUE),
         ("Spatial attention\nmean and max over channels", ORANGE),
         ("$7 \\times 7$ conv + sigmoid\n$\\rightarrow$ one weight per pixel", ORANGE),
         ("Output $F''$\nre-weighted features", "#55A868")]
fig, ax = plt.subplots(figsize=(4.6, 6.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(steps))
for i, (text, col) in enumerate(steps):
    y = len(steps) - 1 - i
    ax.add_patch(FancyBboxPatch((0.03, y + 0.14), 0.94, 0.72, boxstyle="round,pad=0.01,rounding_size=0.04",
                                facecolor=col, edgecolor="black", lw=1.2))
    ax.text(0.5, y + 0.5, text, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y - 0.14), xytext=(0.5, y + 0.14),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
ax.axis("off")
save("dual_attention_block.png")

# 0b. Preprocessing steps, 2 x 3 grid with readable panel titles ---------------------------------------------
from PIL import Image
from src.preprocessing_fives import extract_green_channel, apply_clahe, apply_noise_reduction, normalize_image
fname = "100_A.png"
raw_rgb = np.array(Image.open(f"data/FIVES_resized/train/images/{fname}"))
green = extract_green_channel(raw_rgb)
clahe = apply_clahe(green, clip_limit=2.0, tile_grid_size=(8, 8))
denoised = apply_noise_reduction(clahe, method="bilateral")
panels = [(raw_rgb, f"(a) RGB input ({fname})"), (green, "(b) Green channel"), (clahe, "(c) CLAHE (clip 2.0, 8×8 tiles)"),
          (denoised, "(d) Bilateral filter (d = 5, σ = 75)"), (normalize_image(denoised), "(e) Normalised to [0, 1]"),
          (np.array(Image.open(f"data/FIVES_resized/train/masks/{fname}")), "(f) Expert ground truth")]
fig, ax = plt.subplots(2, 3, figsize=(12, 8.6))
for a, (img, title) in zip(ax.ravel(), panels):
    a.imshow(img, cmap=None if img.ndim == 3 else "gray")
    a.set_title(title, fontsize=13)
    a.axis("off")
plt.tight_layout()
save("preprocessing.png")

# ------------------------------------------------- examples: median baseline Dice per disease (seed 0, not hand-picked)
per_image = pd.read_csv(f"{RUNS}/car_unet_seed0/test_metrics_per_image.csv")
per_image["disease"] = per_image["filename"].str.extract(r"_([ADGN])\.png$")[0].map(DISEASE)
EXAMPLES = [(d, g.sort_values("F1_Dice")["filename"].iloc[len(g) // 2]) for d, g in per_image.groupby("disease")]
print("median examples:", EXAMPLES)

# ------------------------------------------------------------------------------------------- live predictions
base_model, impr_model = load_model("car_unet", 0), load_model("car_unet_da_cldice_1024", 0)
post = json.load(open(f"{RUNS}/car_unet_da_cldice_1024_seed0/post/summary.json"))
sel = selected_setting(post)
THR = post["settings"][sel]["threshold"]
FLIPS = [(), (1,), (0,), (0, 1)] if sel.startswith("tta") else [()]
R = {}
for disease, f in EXAMPLES:
    ip, mp = f"data/FIVES_resized/test/images/{f}", f"data/FIVES_resized/test/masks/{f}"
    it512, it1024 = prepare_eval_item(ip, mp, 512, keep_raw=True), prepare_eval_item(ip, mp, 1024)
    with torch.no_grad():
        pb, _ = predict_item(base_model, it512, device, 36)
        pi = np.mean([predict_item(impr_model, it1024, device, 36, flip_axes=a)[0] for a in FLIPS], axis=0)
    gt, fov = it512["gt"].astype(np.float32), it512["fov"].astype(np.float32)
    res = {"rgb": it512["raw"], "gt": gt, "fov": fov}
    for name, prob, thr in [("base", pb, 0.5), ("impr", pi, THR)]:
        m = compute_fov_metrics(prob, gt, fov, threshold=thr)
        b = ((prob >= thr) * fov).astype(np.uint8)
        res[name] = b
        res[name + "_m"] = {"Dice": m["F1_Dice"], "IoU": m["IoU"], "clDice": compute_cldice(b.astype(np.float32), gt, fov)}
    R[f] = res
    print(f"{disease:9s} {f:10s} base Dice {res['base_m']['Dice']:.4f}  improved Dice {res['impr_m']['Dice']:.4f}")
print(f"improved model setting: {sel}, threshold {THR}")

# 1. Replicated model, one median example per disease (Input | Ground truth | Predicted mask) ------------------
fig, ax = plt.subplots(3, 4, figsize=(13.5, 10.6))
for c, (disease, f) in enumerate(EXAMPLES):
    r, m = R[f], R[f]["base_m"]
    for row, (img, title) in enumerate([(r["rgb"], f"{disease} ({f})\nInput"), (r["gt"], "Ground truth"),
                                         (r["base"], f"Predicted (CAR-UNet)\nDice {m['Dice']:.3f}  IoU {m['IoU']:.3f}")]):
        ax[row, c].imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax[row, c].set_title(title, fontsize=12)
        ax[row, c].axis("off")
plt.tight_layout()
save("replicated_samples_all.png")

# 2. Baseline vs improved, full outputs + zoomed error maps, all four diseases --------------------------------
fig, ax = plt.subplots(4, 6, figsize=(19, 14.4))
for row, (disease, f) in enumerate(EXAMPLES):
    r = R[f]
    y, x = thin_window(r["gt"])
    s = np.s_[y:y + 128, x:x + 128]
    panels = [(r["rgb"], f"Input — {disease}", "black"), (r["gt"], "Ground truth", "black"),
              (r["base"], f"Baseline\nDice {r['base_m']['Dice']:.3f} | clDice {r['base_m']['clDice']:.3f}", BLUE),
              (r["impr"], f"Improved\nDice {r['impr_m']['Dice']:.3f} | clDice {r['impr_m']['clDice']:.3f}", ORANGE),
              (error_map(r["base"], r["gt"], r["fov"])[s], "Baseline errors (zoom)", BLUE),
              (error_map(r["impr"], r["gt"], r["fov"])[s], "Improved errors (zoom)", ORANGE)]
    for c, (img, title, col) in enumerate(panels):
        ax[row, c].imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest" if c >= 4 else "antialiased")
        ax[row, c].set_title(title, fontsize=14, color=col)
        ax[row, c].axis("off")
    ax[row, 0].add_patch(plt.Rectangle((x, y), 128, 128, fill=False, edgecolor="yellow", lw=2))
fig.legend(handles=[Patch(color="#00C800", label="correct vessel"), Patch(color="#E60000", label="false vessel"),
                    Patch(color="#006EFF", label="missed vessel"), Patch(fill=False, edgecolor="#D4B000", label="zoom region (yellow box)")],
           loc="lower center", ncol=4, fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.015))
plt.tight_layout(rect=(0, 0.03, 1, 1), h_pad=2.6)
save("compare_all_diseases.png")
