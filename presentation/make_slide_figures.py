"""
Generates every figure used in the Beamer presentation (presentation/figures/).
Run from the project root:  python presentation/make_slide_figures.py
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

from scripts.train_fives import MODEL_SPECS, prepare_eval_item, predict_item
from scripts.summarize_post import selected_setting
from src.metrics import compute_fov_metrics, compute_cldice
from src.losses import DiceLoss, SoftClDiceLoss

OUT = os.path.join("presentation", "figures")
os.makedirs(OUT, exist_ok=True)
RUNS, SEEDS = "results/runs", [0, 1, 2]
BLUE, ORANGE, GREEN, GRAY, PURPLE, RED, NAVY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C", "#8172B2", "#C44E52", "#1F3A5F"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold"})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save(name, facecolor="white"):
    plt.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", facecolor=facecolor)
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


# ------------------------------------------------------------------------------------------- live predictions
base_model, impr_model = load_model("car_unet", 0), load_model("car_unet_da_cldice_1024", 0)
post = json.load(open(f"{RUNS}/car_unet_da_cldice_1024_seed0/post/summary.json"))
sel = selected_setting(post)
THR = post["settings"][sel]["threshold"]
FLIPS = [(), (1,), (0,), (0, 1)] if sel.startswith("tta") else [()]
EXAMPLES = [("AMD", "31_A.png"), ("DR", "72_D.png"), ("Glaucoma", "104_G.png"), ("Normal", "156_N.png")]
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
    R[f] = (disease, res)
print(f"improved model setting: {sel}, threshold {THR}")

# 1. Replicated model: 3 samples (Input | Ground truth | Predicted mask) -------------------------------------
three = EXAMPLES[:3]
fig, ax = plt.subplots(3, 3, figsize=(10.5, 10.5))
for c, (disease, f) in enumerate(three):
    _, r = R[f]
    m = r["base_m"]
    for row, (img, title) in enumerate([(r["rgb"], f"{disease} ({f})\nInput"), (r["gt"], "Ground truth"),
                                         (r["base"], f"Predicted (CAR-UNet)\nDice {m['Dice']:.3f}  IoU {m['IoU']:.3f}")]):
        ax[row, c].imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax[row, c].set_title(title, fontsize=12)
        ax[row, c].axis("off")
plt.tight_layout()
save("replicated_samples.png")

# 2. Side-by-side: full images (Input | GT | Baseline | Improved) -------------------------------------------
fig, ax = plt.subplots(3, 4, figsize=(15, 11.6))
for row, (disease, f) in enumerate([EXAMPLES[0], EXAMPLES[1], EXAMPLES[3]]):
    _, r = R[f]
    panels = [(r["rgb"], f"Input — {disease}"), (r["gt"], "Ground truth"),
              (r["base"], f"Baseline CAR-UNet\nDice {r['base_m']['Dice']:.3f} | clDice {r['base_m']['clDice']:.3f}"),
              (r["impr"], f"Improved model\nDice {r['impr_m']['Dice']:.3f} | clDice {r['impr_m']['clDice']:.3f}")]
    for c, (img, title) in enumerate(panels):
        ax[row, c].imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax[row, c].set_title(title, fontsize=13, color=BLUE if c == 2 else (ORANGE if c == 3 else "black"))
        ax[row, c].axis("off")
plt.tight_layout()
save("compare_full.png")

# 3. Side-by-side: zoom on thin vessels with error maps -------------------------------------------------------
fig, ax = plt.subplots(3, 5, figsize=(16, 10.4))
for row, (disease, f) in enumerate([EXAMPLES[0], EXAMPLES[1], EXAMPLES[3]]):
    _, r = R[f]
    y, x = thin_window(r["gt"])
    s = np.s_[y:y + 128, x:x + 128]
    panels = [(r["gt"][s], f"{disease}: ground truth"), (r["base"][s], "Baseline"), (r["impr"][s], "Improved"),
              (error_map(r["base"], r["gt"], r["fov"])[s], "Baseline errors"), (error_map(r["impr"], r["gt"], r["fov"])[s], "Improved errors")]
    for c, (img, title) in enumerate(panels):
        ax[row, c].imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
        ax[row, c].set_title(title, fontsize=13, color=BLUE if c in (1, 3) else (ORANGE if c in (2, 4) else "black"))
        ax[row, c].axis("off")
fig.legend(handles=[Patch(color="#00C800", label="correct vessel"), Patch(color="#E60000", label="false vessel"), Patch(color="#006EFF", label="missed vessel")],
           loc="lower center", ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.03, 1, 1))
save("compare_zoom.png")

# 3b. Combined side-by-side figure (full outputs + zoomed error maps) for the single qualitative slide -----------
fig, ax = plt.subplots(3, 6, figsize=(19, 11.0))
for row, (disease, f) in enumerate([EXAMPLES[0], EXAMPLES[1], EXAMPLES[3]]):
    _, r = R[f]
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
           loc="lower center", ncol=4, fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.035, 1, 1), h_pad=2.6)
save("compare_combined.png")

# 4. Evidence for each modification ---------------------------------------------------------------------------
cap = {}
h = impr_model.enc1.meca.spatial.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
it = prepare_eval_item("data/FIVES_resized/test/images/104_G.png", "data/FIVES_resized/test/masks/104_G.png", 1024)
patch = it["prep"].astype(np.float32)[300:584, 330:614] / 255.0
with torch.no_grad():
    impr_model(torch.from_numpy(patch)[None, None].to(device))
    xx, sp = cap["x"], impr_model.enc1.meca.spatial
    att = torch.sigmoid(sp.conv(torch.cat([xx.mean(1, keepdim=True), xx.amax(1, keepdim=True)], 1)))[0, 0].cpu().numpy()
h.remove()
fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
ax[0].imshow(patch[2:-2, 2:-2], cmap="gray"); ax[0].set_title("Input patch")
ax[1].imshow(att, cmap="magma"); ax[1].set_title("Spatial attention map")
for a in ax: a.axis("off")
plt.tight_layout(); save("ev_attention.png")

gt = np.zeros((100, 100), np.float32); cv2.line(gt, (5, 50), (95, 50), 1, 5); cv2.line(gt, (50, 5), (50, 95), 1, 1)
cut = gt.copy(); cut[20:40, 49:52] = 0
trim = gt.copy(); trim[48:53, 5:13] = 0
T = lambda a: torch.from_numpy(a)[None, None]
dl, cl = DiceLoss(), SoftClDiceLoss()
vals = [(dl(T(cut), T(gt)).item(), cl(T(cut), T(gt)).item()), (dl(T(trim), T(gt)).item(), cl(T(trim), T(gt)).item())]
fig, ax = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax[0].imshow(cut, cmap="gray"); ax[0].set_title("Thin vessel cut (20 px)"); ax[0].axis("off")
xs = np.arange(2)
ax[1].bar(xs - 0.2, [v[0] for v in vals], 0.4, label="Dice loss", color=BLUE)
ax[1].bar(xs + 0.2, [v[1] for v in vals], 0.4, label="clDice loss", color=ORANGE)
ax[1].set_xticks(xs); ax[1].set_xticklabels(["thin vessel\ncut", "thick vessel\ntrimmed"]); ax[1].legend(fontsize=11)
ax[1].set_title("Loss increase"); ax[1].grid(alpha=0.3, axis="y")
plt.tight_layout(); save("ev_cldice.png")

g512 = cv2.imread("data/FIVES_resized/test/masks/151_N.png", 0) > 128
g1024 = cv2.imread("data/FIVES_1024/test/masks/151_N.png", 0) > 128
y, x, s = 330, 80, 96
fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
ax[0].imshow(g512[y:y + s, x:x + s], cmap="gray", interpolation="nearest"); ax[0].set_title("Ground truth at 512")
ax[1].imshow(g1024[2 * y:2 * (y + s), 2 * x:2 * (x + s)], cmap="gray", interpolation="nearest"); ax[1].set_title("Ground truth at 1024")
for a in ax: a.axis("off")
plt.tight_layout(); save("ev_resolution.png")

# 5. Ablation chart -------------------------------------------------------------------------------------------
def per_image(key, seed, setting="base"):
    p = f"{RUNS}/{key}_seed{seed}/" + ("test_metrics_per_image.csv" if setting == "base" else f"post/test_metrics_{setting}.csv")
    return pd.read_csv(p)


def improved(seed):
    po = json.load(open(f"{RUNS}/car_unet_da_cldice_1024_seed{seed}/post/summary.json"))
    return per_image("car_unet_da_cldice_1024", seed, selected_setting(po))


steps = [("Baseline\nCAR-UNet", lambda s: per_image("car_unet", s)), ("+ Dual\nattention", lambda s: per_image("car_unet_da", s)),
         ("+ clDice\nloss", lambda s: per_image("car_unet_da_cldice", s)), ("+ 1024\nresolution", lambda s: per_image("car_unet_da_cldice_1024", s)),
         ("+ tuned\nthreshold", improved)]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for a, m, title in zip(ax, ["F1_Dice", "clDice"], ["Dice", "clDice (connectivity)"]):
    v = [[f(s)[m].mean() for s in SEEDS] for _, f in steps]
    means, stds = [np.mean(x) for x in v], [np.std(x, ddof=1) for x in v]
    a.bar(range(5), means, yerr=stds, capsize=5, color=[BLUE, "#8FA9D6", "#8FA9D6", "#8FA9D6", ORANGE])
    for k, mv in enumerate(means):
        a.text(k, mv + max(stds) + 0.0008, f"{mv:.4f}", ha="center", fontsize=11)
    a.set_xticks(range(5)); a.set_xticklabels([n for n, _ in steps], fontsize=11)
    a.set_ylim(min(means) - 0.012, max(means) + 0.008); a.set_title(title); a.grid(alpha=0.3, axis="y")
plt.tight_layout(); save("ablation.png")

# 6. Change per metric ----------------------------------------------------------------------------------------
M = [("Accuracy", "Accuracy"), ("Sensitivity", "Sensitivity"), ("Specificity", "Specificity"), ("Precision", "Precision"),
     ("Dice", "F1_Dice"), ("IoU", "IoU"), ("AUC-ROC", "AUC_ROC"), ("AUC-PR", "AUC_PR"), ("clDice", "clDice")]
ch = [100 * (np.mean([improved(s)[k].mean() for s in SEEDS]) - np.mean([per_image("car_unet", s)[k].mean() for s in SEEDS])) for _, k in M]
hi_b = np.mean([json.load(open(f"{RUNS}/car_unet_seed{s}/post/summary.json"))["settings"]["base"]["overall"]["hires_F1_Dice"] for s in SEEDS])
hi_i = np.mean([json.load(open(f"{RUNS}/car_unet_da_cldice_1024_seed{s}/post/summary.json"))["settings"]["base"]["overall"]["hires_F1_Dice"] for s in SEEDS])
names = [n for n, _ in M] + ["Dice @1024 GT"]
ch.append(100 * (hi_i - hi_b))
order = np.argsort(ch)
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.barh([names[k] for k in order], [ch[k] for k in order], color=[GREEN if ch[k] > 0 else RED for k in order])
for p, k in enumerate(order):
    ax.text(ch[k] + (0.05 if ch[k] >= 0 else -0.05), p, f"{ch[k]:+.2f}", va="center", ha="left" if ch[k] >= 0 else "right", fontsize=11)
ax.axvline(0, color="black", lw=0.8); ax.set_xlim(min(ch) - 0.5, max(ch) + 0.6)
ax.set_xlabel("change vs baseline (percentage points)"); ax.set_title("Improved vs baseline (3 seeds)"); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); save("change_per_metric.png")

# 7. Training log "screenshot" (real lines from results/logs/baselines.log) -----------------------------------
lines = open("results/logs/baselines.log", encoding="utf-8", errors="replace").read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("[run] car_unet_seed0"))
end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("[run]"))   # stop at the next run
block = lines[start:end]
head = [l for l in block if l.startswith(("CAR-UNet on FIVES", "Parameters", "Train images"))]
epochs = [l for l in block if "Epoch [" in l]
tail = [l for l in block if l.startswith(("Training finished", "Loaded best"))]
test = [l.strip() for l in block if l.strip().startswith(("Sensitivity:", "F1_Dice:", "IoU:", "AUC_ROC:"))]
shown = head + epochs[:3] + ["   ...   (epochs 4-26)"] + epochs[-4:] + tail + ["--- test set (200 images) ---"] + test
shown = [" ".join(l.split()) if "Epoch [" in l else l for l in shown]
fig, ax = plt.subplots(figsize=(13, 0.34 * len(shown) + 0.8))
fig.patch.set_facecolor("#1E1E1E"); ax.set_facecolor("#1E1E1E"); ax.axis("off")
ax.text(0.01, 0.985, "$ python scripts/train_fives.py --model car_unet --seed 0", color="#7EE787", family="monospace", fontsize=11.5,
        va="top", transform=ax.transAxes)
for k, l in enumerate(shown):
    color = "#F2CC60" if l.startswith(("Loaded best", "Training finished", "F1_Dice", "IoU", "Sensitivity", "AUC_ROC")) else \
            ("#7FB8F0" if l.startswith("---") else "#E6E6E6")
    ax.text(0.01, 0.985 - (k + 1.4) / (len(shown) + 1.8), l[:120], color=color, family="monospace", fontsize=10.5, va="top", transform=ax.transAxes)
save("training_log.png", facecolor="#1E1E1E")

# 8. Architecture diagrams (baseline and modified) ------------------------------------------------------------
def box(ax, x, y, w, h, text, color, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, color="black", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4, linestyle=ls))


def architecture(modified, name):
    fig, ax = plt.subplots(figsize=(13, 6.4)); ax.set_xlim(0, 12); ax.set_ylim(-0.5, 7.1); ax.axis("off")
    blk = "CADRB" if True else ""
    enc = [("1→64", 5.4), ("64→128", 4.3), ("128→256", 3.2), ("256→512", 2.1)]
    dec = [("1024→512", 2.1), ("512→256", 3.2), ("256→128", 4.3), ("128→64", 5.4)]
    att_label = "Dual attention\n(channel + spatial)" if modified else "Channel attention\n(MECA)"
    att_fc, att_ec = ("#FBE3D6", ORANGE) if modified else ("#E8E3F5", PURPLE)
    for i, (t, y) in enumerate(enc):
        x = 0.6 + i * 0.9
        box(ax, x, y, 1.55, 0.78, f"{blk}\n{t}", BLUE, fs=9)
        if i < 3:
            arrow(ax, x + 0.75, y, x + 1.55, y - 0.32, color=RED)
    box(ax, 4.6, 0.55, 2.6, 0.9, f"{blk} bottleneck\n512→1024 + Dropout", NAVY, fs=9)
    arrow(ax, 3.85, 2.1, 4.85, 1.45, color=RED)
    for i, (t, y) in enumerate(dec):
        box(ax, 7.4 + i * 0.9, y, 1.55, 0.78, f"{blk}\n{t}", "#C9733F" if modified else ORANGE, fs=9)
    arrow(ax, 6.95, 1.45, 7.7, 2.1, color=GREEN)
    for i in range(3):
        arrow(ax, 7.4 + i * 0.9 + 0.77, dec[i][1] + 0.78, 7.4 + (i + 1) * 0.9 + 0.77, dec[i + 1][1], color=GREEN)
    for i in range(4):
        y = enc[i][1] + 0.39
        xs, xe = 0.6 + i * 0.9 + 1.55, 7.4 + (3 - i) * 0.9
        ax.add_patch(FancyArrowPatch((xs, y), (xe, y), arrowstyle="-|>", mutation_scale=10, color=att_ec, lw=1.3, linestyle="--"))
        ax.add_patch(FancyBboxPatch(((xs + xe) / 2 - 0.75, y - 0.27), 1.5, 0.54, boxstyle="round,pad=0.02", fc=att_fc, ec=att_ec, lw=1.0))
        ax.text((xs + xe) / 2, y, att_label, fontsize=8, ha="center", va="center", color="black")
    inp = "1024×1024 image\n→ 284×284 patches" if modified else "512×512 image\n→ 284×284 patches"
    box(ax, 0.5, 6.4, 1.9, 0.6, inp, GREEN if modified else GRAY, fs=8.5)
    arrow(ax, 1.38, 6.4, 1.38, 6.18)
    outp = "sigmoid → threshold\ntuned on validation" if modified else "sigmoid →\nthreshold 0.5"
    box(ax, 9.7, 6.4, 2.1, 0.6, outp, GREEN if modified else GRAY, fs=8.5)
    arrow(ax, 10.55, 6.18, 10.55, 6.4)
    loss = "Loss: BCE + Dice + clDice" if modified else "Loss: BCE + Dice"
    ax.text(6.0, 6.75, loss, ha="center", fontsize=11, color=GREEN if modified else "black", fontweight="bold")
    note = "Inside each CADRB: conv-BN-ReLU → conv-BN → " + ("Dual attention" if modified else "MECA") + " → + shortcut → ReLU"
    ax.text(6.0, -0.35, note, ha="center", fontsize=9.5, style="italic")
    if modified:
        ax.text(6.0, 6.35, "changes vs baseline shown in orange / green", ha="center", fontsize=9, color=ORANGE)
    save(name)


architecture(False, "arch_baseline.png")
architecture(True, "arch_modified.png")
print("all slide figures done")
