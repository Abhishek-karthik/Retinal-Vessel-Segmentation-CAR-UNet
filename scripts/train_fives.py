"""
Unified training & evaluation engine for retinal vessel segmentation on FIVES.

Both the baseline U-Net and CAR-UNet are trained by this ONE script so that every
setting (epochs, patches, optimiser, scheduler, loss, data split, seed) is identical
and the comparison between models is fair.

Examples:
    python scripts/train_fives.py --model unet     --seed 0
    python scripts/train_fives.py --model car_unet --seed 0
    python scripts/train_fives.py --model car_unet --seed 0 --resume      # continue after interruption
    python scripts/train_fives.py --model car_unet --seed 0 --eval_only   # re-score saved best model
"""
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # must be imported before numpy on this environment (DLL load order)
from torch.utils.data import DataLoader
import argparse
import json
import random
import shutil
import time
import numpy as np
import pandas as pd
import cv2

from src.data_loader_fives import FIVESPatchDataset, load_fives_image_pairs, get_disease_code, DISEASE_NAMES
from src.unet_model import UNet
from src.car_unet import CARUNet
from src.losses import CombinedBCEDiceLoss, BCEDiceClDiceLoss
from src.metrics import compute_fov_metrics, compute_cldice
from src.preprocessing_fives import preprocess_image, preprocess_image_uint8, generate_fov_mask
from src.utils import predict_full_image, save_prediction_figure

# name -> (label, model class, model kwargs, loss)
MODEL_SPECS = {
    "unet":               ("U-Net",                     UNet,    {},                     "bce_dice"),
    "car_unet":           ("CAR-UNet",                  CARUNet, {"attention": "meca"},  "bce_dice"),
    "car_unet_da":        ("CAR-UNet + DA",             CARUNet, {"attention": "dual"},  "bce_dice"),          # Improvement 1
    "car_unet_da_cldice": ("CAR-UNet + DA + clDice",    CARUNet, {"attention": "dual"},  "bce_dice_cldice"),   # Improvements 1 + 2
    "car_unet_da_cldice_1024": ("CAR-UNet + DA + clDice @1024", CARUNet, {"attention": "dual"}, "bce_dice_cldice_1024"),  # + Improvement 3
}
# Working resolution of each model (default 512). Every model is SCORED on the same 512x512 ground truth;
# higher-resolution predictions are downsampled first, and additionally scored against their own ground truth (hires_*).
MODEL_RESOLUTION = {"car_unet_da_cldice_1024": 1024}
EVAL_RESOLUTION = 512
MODELS = {k: v[1] for k, v in MODEL_SPECS.items()}
MODEL_LABELS = {k: v[0] for k, v in MODEL_SPECS.items()}
LOSS_NAMES = {"bce_dice": "0.5*BCE + 0.5*Dice", "bce_dice_cldice": "0.5*BCE + 0.5*(0.5*Dice + 0.5*clDice)",
              "bce_dice_cldice_1024": "0.5*BCE + 0.5*(0.5*Dice + 0.5*clDice), 20 skeleton iterations (vessels are 2x wider at 1024)"}
HIRES_COLS = ["hires_F1_Dice", "hires_Sensitivity", "hires_Precision", "hires_clDice"]
METRIC_COLS = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Dice", "IoU", "AUC_ROC", "AUC_PR", "clDice"]


def make_loss(name):
    if name == "bce_dice":
        return CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    if name == "bce_dice_cldice":
        return BCEDiceClDiceLoss(alpha=0.5)
    if name == "bce_dice_cldice_1024":
        return BCEDiceClDiceLoss(alpha=0.5, iterations=20)
    raise ValueError(name)


def data_dir_for(base_dir, resolution):
    return os.path.join(base_dir, "data", "FIVES_resized" if resolution == 512 else f"FIVES_{resolution}")
IN_PATCH_SIZE = 284


def set_seed(seed: int):
    """Seeds every random number generator used in training (Python, NumPy, PyTorch CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_eval_image(img_path, mask_path):
    raw_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gt_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.float32)
    return raw_rgb, preprocess_image(raw_rgb), gt_mask, generate_fov_mask(raw_rgb)


def prepare_eval_item(img_path, mask_path, model_res=512, keep_raw=False):
    """
    Loads one evaluation image (paths point to the 512x512 copy) in a compact uint8 form:
      prep / fov_in : model input + FOV at the model's working resolution
      gt / fov      : 512x512 ground truth + FOV used for the official metrics
      gt_hi / fov_hi: ground truth + FOV at the model resolution (only when model_res != 512)
    """
    raw = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
    fov = generate_fov_mask(raw).astype(np.uint8)
    item = {"gt": gt, "fov": fov}
    if model_res == EVAL_RESOLUTION:
        item.update(prep=preprocess_image_uint8(raw), fov_in=fov)
    else:
        hi_dir = f"FIVES_{model_res}"
        hi_img = img_path.replace("FIVES_resized", hi_dir)
        hi_raw = cv2.cvtColor(cv2.imread(hi_img), cv2.COLOR_BGR2RGB)
        fov_hi = generate_fov_mask(hi_raw).astype(np.uint8)
        gt_hi = (cv2.imread(mask_path.replace("FIVES_resized", hi_dir), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
        item.update(prep=preprocess_image_uint8(hi_raw), fov_in=fov_hi, gt_hi=gt_hi, fov_hi=fov_hi)
    if keep_raw:
        item["raw"] = raw
    return item


def predict_item(model, item, device, inference_batch, flip_axes=()):
    """
    Overlap-Tile prediction at the model resolution (optionally on a flipped copy, flipped back afterwards).
    Returns (probability map at 512x512 masked by the 512 FOV, probability map at model resolution).
    """
    prep = item["prep"].astype(np.float32) / 255.0   # identical to preprocess_image()
    fov_in = item["fov_in"].astype(np.float32)
    if flip_axes:
        prep, fov_in = np.flip(prep, axis=flip_axes).copy(), np.flip(fov_in, axis=flip_axes).copy()
    prob = predict_full_image(model, prep, fov_in, in_patch_size=IN_PATCH_SIZE, device=device, batch_size=inference_batch)
    if flip_axes:
        prob = np.flip(prob, axis=flip_axes).copy()
    if prob.shape == item["gt"].shape:
        return prob, prob
    prob_eval = cv2.resize(prob, item["gt"].shape[::-1], interpolation=cv2.INTER_AREA) * item["fov"]
    return prob_eval, prob


def hires_metrics(prob, gt, fov, threshold=0.5):
    """Dice / Sensitivity / Precision / clDice against a higher-resolution ground truth (no AUC: too slow at 1024)."""
    inside = fov > 0.5
    pred, g = prob[inside] >= threshold, gt[inside] > 0.5
    tp, fp, fn = np.sum(pred & g), np.sum(pred & ~g), np.sum(~pred & g)
    return {"hires_F1_Dice": float(2 * tp / (2 * tp + fp + fn + 1e-8)), "hires_Sensitivity": float(tp / (tp + fn + 1e-8)),
            "hires_Precision": float(tp / (tp + fp + 1e-8)),
            "hires_clDice": compute_cldice((prob >= threshold).astype(np.float32), gt, fov)}


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)


def validate_full_images(model, val_items, device, inference_batch):
    """
    Validation with the same full-image Overlap-Tile protocol as the test set (scored at 512x512).
    Returns the mean per-image Dice, Sensitivity and Specificity inside the FOV (threshold 0.5).
    """
    dices, sens, specs = [], [], []
    for item in val_items:
        prob, _ = predict_item(model, item, device, inference_batch)
        gt_mask, fov_mask = item["gt"], item["fov"]
        inside = fov_mask > 0.5
        pred = prob[inside] >= 0.5
        gt = gt_mask[inside] > 0.5
        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)
        tn = np.sum(~pred & ~gt)
        dices.append(2 * tp / (2 * tp + fp + fn + 1e-8))
        sens.append(tp / (tp + fn + 1e-8))
        specs.append(tn / (tn + fp + 1e-8))
    return float(np.mean(dices)), float(np.mean(sens)), float(np.mean(specs))


def evaluate_test_set(model, test_pairs, device, viz_dir, inference_batch, viz_per_disease=1, model_res=512):
    """Scores EVERY test image with Overlap-Tile inference; metrics are computed inside the FOV only."""
    os.makedirs(viz_dir, exist_ok=True)
    model.eval()
    rows = []
    saved_viz = {}
    print(f"\nEvaluating on {len(test_pairs)} held-out test images...", flush=True)
    for i, (img_path, mask_path) in enumerate(test_pairs):
        fname = os.path.basename(img_path)
        code = get_disease_code(fname)
        item = prepare_eval_item(img_path, mask_path, model_res, keep_raw=True)
        raw_rgb, gt_mask, fov_mask = item["raw"], item["gt"].astype(np.float32), item["fov"].astype(np.float32)
        prob, prob_model = predict_item(model, item, device, inference_batch)

        metrics = compute_fov_metrics(prob, gt_mask, fov_mask)
        if model_res != EVAL_RESOLUTION:
            metrics.update(hires_metrics(prob_model, item["gt_hi"], item["fov_hi"]))
        metrics["filename"] = fname
        metrics["clDice"] = compute_cldice((prob >= 0.5).astype(np.float32), gt_mask, fov_mask)
        metrics["disease"] = DISEASE_NAMES[code]
        rows.append(metrics)

        if saved_viz.get(code, 0) < viz_per_disease:
            save_prediction_figure(raw_rgb, prob, fov_mask, os.path.join(viz_dir, f"pred_{fname}"), ground_truth=gt_mask, filename=fname)
            saved_viz[code] = saved_viz.get(code, 0) + 1

        if (i + 1) % 25 == 0 or (i + 1) == len(test_pairs):
            print(f"  Test progress: [{i+1}/{len(test_pairs)}]", flush=True)

    df = pd.DataFrame(rows)
    cols = METRIC_COLS + [c for c in HIRES_COLS if c in df]
    summary = {
        "overall": df[cols].mean().to_dict(),
        "overall_std": df[cols].std().to_dict(),
        "per_disease": {d: g[cols].mean().to_dict() for d, g in df.groupby("disease")},
        "num_test_images": int(len(df)),
        "images_per_disease": df["disease"].value_counts().sort_index().to_dict(),
    }
    return summary, df


def main():
    parser = argparse.ArgumentParser(description="Train / evaluate U-Net or CAR-UNet on FIVES with identical settings")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=int, default=0, help="Training seed (weights init, patch sampling, augmentation)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patches_per_img", type=int, default=6, help="Fresh random patches drawn per training image every epoch")
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--inference_batch", type=int, default=16, help="Tiles per forward pass during full-image inference")
    parser.add_argument("--resume", action="store_true", help="Continue training from the last saved epoch")
    parser.add_argument("--eval_only", action="store_true", help="Skip training; evaluate the saved best checkpoint")
    parser.add_argument("--limit_images", type=int, default=0, help="Smoke test only: use N images per split (0 = all)")
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for the run name (e.g. 'timing')")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (smoke tests while the GPU is busy)")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = data_dir_for(base_dir, EVAL_RESOLUTION)          # 512 copy: validation / test scoring
    model_res = MODEL_RESOLUTION.get(args.model, EVAL_RESOLUTION)
    train_dir = data_dir_for(base_dir, model_res)               # training images at the model resolution
    run_name = f"{args.model}_seed{args.seed}" + ("_smoke" if args.limit_images else "") + (f"_{args.tag}" if args.tag else "")
    ckpt_dir = os.path.join(base_dir, "outputs", "saved_models")
    run_dir = os.path.join(base_dir, "results", "runs", run_name)
    viz_dir = os.path.join(base_dir, "outputs", "predictions", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    last_ckpt = os.path.join(ckpt_dir, f"{run_name}_last.pt")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    config = {k: v for k, v in vars(args).items() if k not in ("resume", "eval_only")}
    config.update({"in_patch_size": IN_PATCH_SIZE, "out_patch_size": IN_PATCH_SIZE - 184, "loss": LOSS_NAMES[MODEL_SPECS[args.model][3]], "resolution": model_res, "scored_at": EVAL_RESOLUTION, "attention": MODEL_SPECS[args.model][2].get("attention", "none"),
                   "optimizer": "Adam", "scheduler": "CosineAnnealingLR(T_max=epochs)", "split": "disease-stratified 480/120, split_seed=42",
                   "selection": "best mean per-image validation Dice (full images, FOV, threshold 0.5)", "device": str(device)})

    print("=" * 70, flush=True)
    print(f"{MODEL_LABELS[args.model]} on FIVES | run: {run_name} | device: {device}", flush=True)
    print(json.dumps(config, indent=2), flush=True)
    print("=" * 70, flush=True)

    model = MODELS[args.model](in_channels=1, out_channels=1, base_filters=args.base_filters, **MODEL_SPECS[args.model][2]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}", flush=True)
    criterion = make_loss(MODEL_SPECS[args.model][3])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_pairs = load_fives_image_pairs(train_dir, mode="train")
    assert [os.path.basename(p[0]) for p in train_pairs] == [os.path.basename(p[0]) for p in load_fives_image_pairs(data_dir, mode="train")], \
        "Train split differs between resolutions"
    val_pairs = load_fives_image_pairs(data_dir, mode="val")
    test_pairs = load_fives_image_pairs(data_dir, mode="test")
    if args.limit_images:
        train_pairs, val_pairs, test_pairs = train_pairs[:args.limit_images], val_pairs[:args.limit_images], test_pairs[:args.limit_images]

    history = {"epoch": [], "train_loss": [], "val_dice": [], "val_sensitivity": [], "val_specificity": [], "lr": [], "epoch_seconds": []}
    best_val_dice, best_epoch, start_epoch, train_seconds = -1.0, 0, 1, 0.0

    if args.resume and os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history, best_val_dice, best_epoch = ckpt["history"], ckpt["best_val_dice"], ckpt["best_epoch"]
        train_seconds = ckpt.get("train_seconds", 0.0)
        start_epoch = ckpt["epoch"] + 1
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
        print(f"Resumed from epoch {ckpt['epoch']} (best val Dice {best_val_dice:.4f} at epoch {best_epoch})", flush=True)

    if not args.eval_only and start_epoch <= args.epochs:
        train_dataset = FIVESPatchDataset(train_dir, mode="train", in_patch_size=IN_PATCH_SIZE, patches_per_img=args.patches_per_img, augment=True, image_pairs=train_pairs)
        val_images = [prepare_eval_item(i, m, model_res) for i, m in val_pairs]
        print(f"Train images: {len(train_pairs)} ({len(train_dataset)} patches/epoch) | Val images: {len(val_pairs)} | Test images: {len(test_pairs)}", flush=True)

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            if epoch > 1:
                train_dataset.resample()
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            model.eval()
            val_dice, val_sens, val_spec = validate_full_images(model, val_images, device, args.inference_batch)
            scheduler.step()
            elapsed = time.time() - t0
            train_seconds += elapsed

            for k, v in zip(history, [epoch, train_loss, val_dice, val_sens, val_spec, lr, elapsed]):
                history[k].append(v)
            improved = val_dice > best_val_dice
            if improved:
                best_val_dice, best_epoch = val_dice, epoch
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_dice": best_val_dice,
                            "config": config, "history": history}, best_ckpt)
            print(f"Epoch [{epoch:02d}/{args.epochs}] {elapsed:6.1f}s | lr {lr:.2e} | train loss {train_loss:.4f} | "
                  f"val Dice {val_dice:.4f} Sens {val_sens:.4f} Spec {val_spec:.4f}" + ("  <- best" if improved else ""), flush=True)

            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(), "history": history, "best_val_dice": best_val_dice,
                        "best_epoch": best_epoch, "train_seconds": train_seconds,
                        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, last_ckpt)

        print(f"Training finished in {train_seconds/60:.1f} min. Best val Dice {best_val_dice:.4f} at epoch {best_epoch}.", flush=True)

    if not os.path.exists(best_ckpt):
        sys.exit(f"No checkpoint found at {best_ckpt}. Train the model first.")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    history = ckpt.get("history", history) if args.eval_only else history
    best_epoch, best_val_dice = ckpt["epoch"], ckpt["best_val_dice"]
    if args.eval_only and os.path.exists(last_ckpt):
        train_seconds = torch.load(last_ckpt, map_location="cpu", weights_only=False).get("train_seconds", train_seconds)
    print(f"Loaded best checkpoint: epoch {best_epoch}, val Dice {best_val_dice:.4f}", flush=True)

    t0 = time.time()
    summary, df = evaluate_test_set(model, test_pairs, device, viz_dir, args.inference_batch, model_res=model_res)
    summary.update({"model": MODEL_LABELS[args.model], "run": run_name, "seed": args.seed, "parameters": num_params,
                    "best_epoch": best_epoch, "best_val_dice": best_val_dice, "train_minutes": round(train_seconds / 60, 2),
                    "test_seconds_per_image": round((time.time() - t0) / len(test_pairs), 3), "config": config})

    df.to_csv(os.path.join(run_dir, "test_metrics_per_image.csv"), index=False)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if not args.eval_only:
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    if not args.limit_images and not args.tag:
        shutil.copy(best_ckpt, os.path.join(base_dir, "models", f"{run_name}.pt"))

    print(f"\n--- {MODEL_LABELS[args.model]} (seed {args.seed}) on {summary['num_test_images']} FIVES test images ---", flush=True)
    for k in METRIC_COLS:
        print(f"{k:>12}: {summary['overall'][k]:.4f} +/- {summary['overall_std'][k]:.4f}", flush=True)
    print("Per-disease Dice: " + ", ".join(f"{d} {m['F1_Dice']:.4f}" for d, m in summary["per_disease"].items()), flush=True)
    print(f"Saved results to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
