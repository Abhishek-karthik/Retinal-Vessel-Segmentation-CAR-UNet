"""
Improvement 4: smarter prediction without retraining.

For each trained run this script evaluates four prediction settings on the FIVES test set:
    base         threshold 0.5, single prediction            (must reproduce the original test results)
    thr          threshold tuned on the VALIDATION set, single prediction
    tta          threshold 0.5, test-time augmentation (mean of 4 flips: none / horizontal / vertical / both)
    tta_thr      TTA + threshold tuned on the validation set with TTA

The threshold is chosen ONLY on the 120 validation images (maximum mean per-image Dice inside the FOV);
the test set is never used for any choice.

All settings are scored on the 512x512 ground truth (as in training). When data/FIVES_1024 exists, the 'base' and
'tta_thr' settings are ALSO scored against the 1024x1024 ground truth (hires_* columns): 512 models are upsampled
(bilinear), 1024 models are used natively. This shows whether higher resolution recovers thin vessels.

Results: results/runs/<run>/post/summary.json and per-image CSVs per setting.

    python scripts/evaluate_post.py --runs car_unet_seed0 car_unet_da_cldice_seed1
    python scripts/evaluate_post.py --models car_unet car_unet_da car_unet_da_cldice --seeds 0 1 2
"""
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # must be imported before numpy on this environment (DLL load order)
import argparse
import json
import time
import numpy as np
import pandas as pd

import cv2
from scripts.train_fives import (MODEL_SPECS, MODEL_RESOLUTION, METRIC_COLS, HIRES_COLS, EVAL_RESOLUTION,
                                 prepare_eval_item, predict_item, hires_metrics)
from src.data_loader_fives import load_fives_image_pairs, get_disease_code, DISEASE_NAMES
from src.metrics import compute_fov_metrics, compute_cldice

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "FIVES_resized")
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.02), 2)
FLIPS = [(), (1,), (0,), (0, 1)]  # axes to flip: none, horizontal, vertical, both
HIRES_RES = 1024
HIRES_DIR = os.path.join(BASE_DIR, "data", f"FIVES_{HIRES_RES}")


def predict_with_flips(model, item, device, batch):
    """
    Returns ((single, TTA) at 512 for the official metrics, (single, TTA) at model resolution).
    TTA = mean of the predictions on the 4 flipped copies, each flipped back before averaging.
    """
    evals, natives = zip(*[predict_item(model, item, device, batch, flip_axes=axes) for axes in FLIPS])
    return (evals[0], np.mean(evals, axis=0)), (natives[0], np.mean(natives, axis=0))


def to_hires(prob_native, item_hi):
    """Probability map at 1024x1024 for the hi-res ground truth (bilinear upsampling for 512 models)."""
    if prob_native.shape == item_hi["gt_hi"].shape:
        return prob_native
    return cv2.resize(prob_native, item_hi["gt_hi"].shape[::-1], interpolation=cv2.INTER_LINEAR) * item_hi["fov_hi"]


def dice_at_thresholds(prob, gt, fov):
    inside = fov > 0.5
    p, g = prob[inside], gt[inside] > 0.5
    out = []
    for t in THRESHOLDS:
        pred = p >= t
        tp = np.sum(pred & g)
        out.append(2 * tp / (pred.sum() + g.sum() + 1e-8))
    return np.array(out)


def evaluate_run(run, device, batch, limit):
    model_key = run.rsplit("_seed", 1)[0]
    label, cls, kwargs, _ = MODEL_SPECS[model_key]
    model_res = MODEL_RESOLUTION.get(model_key, EVAL_RESOLUTION)
    use_hires = os.path.isdir(HIRES_DIR)
    model = cls(in_channels=1, out_channels=1, base_filters=64, **kwargs).to(device)
    ckpt = torch.load(os.path.join(BASE_DIR, "models", f"{run}.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_pairs = load_fives_image_pairs(DATA_DIR, mode="val")
    test_pairs = load_fives_image_pairs(DATA_DIR, mode="test")
    if limit:
        val_pairs, test_pairs = val_pairs[:limit], test_pairs[:limit]

    # 1. Choose thresholds on the validation set only.
    t0 = time.time()
    val_dice = {"single": [], "tta": []}
    for img_path, mask_path in val_pairs:
        item = prepare_eval_item(img_path, mask_path, model_res)
        gt, fov = item["gt"], item["fov"]
        (single, tta), _ = predict_with_flips(model, item, device, batch)
        val_dice["single"].append(dice_at_thresholds(single, gt, fov))
        val_dice["tta"].append(dice_at_thresholds(tta, gt, fov))
    curves = {k: np.mean(v, axis=0) for k, v in val_dice.items()}
    best_thr = {k: float(THRESHOLDS[int(np.argmax(c))]) for k, c in curves.items()}
    print(f"  validation: best threshold single {best_thr['single']:.2f} (Dice {curves['single'].max():.4f} vs "
          f"{curves['single'][list(THRESHOLDS).index(0.5)]:.4f} at 0.5) | TTA {best_thr['tta']:.2f} "
          f"(Dice {curves['tta'].max():.4f}) [{time.time() - t0:.0f}s]", flush=True)

    # 2. Score the test set under the four settings.
    settings = {"base": ("single", 0.5), "thr": ("single", best_thr["single"]),
                "tta": ("tta", 0.5), "tta_thr": ("tta", best_thr["tta"])}
    rows = {s: [] for s in settings}
    for i, (img_path, mask_path) in enumerate(test_pairs):
        fname = os.path.basename(img_path)
        item = prepare_eval_item(img_path, mask_path, model_res)
        gt, fov = item["gt"].astype(np.float32), item["fov"].astype(np.float32)
        evals, natives = predict_with_flips(model, item, device, batch)
        probs, probs_native = dict(zip(["single", "tta"], evals)), dict(zip(["single", "tta"], natives))
        item_hi = item if model_res == HIRES_RES else (prepare_eval_item(img_path, mask_path, HIRES_RES) if use_hires else None)
        for s, (kind, thr) in settings.items():
            m = compute_fov_metrics(probs[kind], gt, fov, threshold=thr)
            m["clDice"] = compute_cldice((probs[kind] >= thr).astype(np.float32), gt, fov)
            if item_hi is not None and s in ("base", "tta_thr"):
                m.update(hires_metrics(to_hires(probs_native[kind], item_hi), item_hi["gt_hi"], item_hi["fov_hi"], threshold=thr))
            m["filename"], m["disease"] = fname, DISEASE_NAMES[get_disease_code(fname)]
            rows[s].append(m)
        if (i + 1) % 50 == 0 or (i + 1) == len(test_pairs):
            print(f"  test progress [{i + 1}/{len(test_pairs)}]", flush=True)

    out_dir = os.path.join(BASE_DIR, "results", "runs", run, "post" + ("_smoke" if limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    summary = {"run": run, "model": label, "model_resolution": model_res, "thresholds_tuned_on": f"{len(val_pairs)} validation images",
               "best_threshold": best_thr,
               "val_dice_curve": {k: dict(zip(map(str, THRESHOLDS), map(float, c))) for k, c in curves.items()},
               "settings": {}}
    for s, (kind, thr) in settings.items():
        df = pd.DataFrame(rows[s])
        df.to_csv(os.path.join(out_dir, f"test_metrics_{s}.csv"), index=False)
        cols = METRIC_COLS + [c for c in HIRES_COLS if c in df]
        summary["settings"][s] = {"prediction": kind, "threshold": thr, "overall": df[cols].mean().to_dict(),
                                  "per_disease": {d: g[cols].mean().to_dict() for d, g in df.groupby("disease")}}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 3. Sanity check: the 'base' setting must reproduce the original evaluation of this run.
    orig = pd.read_csv(os.path.join(BASE_DIR, "results", "runs", run, "test_metrics_per_image.csv")).set_index("filename")
    new = pd.DataFrame(rows["base"]).set_index("filename")
    max_diff = (new["F1_Dice"] - orig.loc[new.index, "F1_Dice"]).abs().max()
    status = "OK" if max_diff < 1e-3 else "MISMATCH"
    print(f"  sanity check vs original evaluation: max |Dice difference| = {max_diff:.2e} -> {status}", flush=True)
    for s in settings:
        o = summary["settings"][s]["overall"]
        hi = f" | hi-res Dice {o['hires_F1_Dice']:.4f} clDice {o['hires_clDice']:.4f}" if "hires_F1_Dice" in o else ""
        print(f"  {s:8s} thr {settings[s][1]:.2f}: Dice {o['F1_Dice']:.4f} | Sens {o['Sensitivity']:.4f} | "
              f"Prec {o['Precision']:.4f} | clDice {o['clDice']:.4f}{hi}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--inference_batch", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_done", action="store_true", help="Skip runs that already have post/summary.json")
    parser.add_argument("--limit_images", type=int, default=0, help="Smoke test only: N val and N test images")
    args = parser.parse_args()

    runs = list(args.runs) + [f"{m}_seed{s}" for s in args.seeds for m in args.models]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    failed = []
    for run in runs:
        if not os.path.exists(os.path.join(BASE_DIR, "models", f"{run}.pt")):
            print(f"[skip] {run}: no trained model yet", flush=True)
            continue
        if args.skip_done and not args.limit_images and os.path.exists(os.path.join(BASE_DIR, "results", "runs", run, "post", "summary.json")):
            print(f"[skip] {run}: already evaluated", flush=True)
            continue
        print(f"[post] {run} ({device})", flush=True)
        if evaluate_run(run, device, args.inference_batch, args.limit_images) != "OK":
            failed.append(run)
    if failed:
        sys.exit(f"Sanity check failed for: {failed}")


if __name__ == "__main__":
    main()
