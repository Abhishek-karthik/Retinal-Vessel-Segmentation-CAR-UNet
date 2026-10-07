"""
Builds the Improvement 4 tables (threshold tuning + test-time augmentation) and the final ablation table.

Inputs : results/runs/<run>/post/summary.json and test_metrics_<setting>.csv (from evaluate_post.py)
         results/runs/<run>/summary.json (main training runs)
Outputs: results/POST_RESULTS.md, results/post_settings_table.csv, results/ablation_table.csv

Smoke-test and tagged runs are ignored.
"""
import os
import re
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(BASE_DIR, "results", "runs")
OUT_DIR = os.path.join(BASE_DIR, "results")
RUN_PATTERN = re.compile(r"^[a-z0-9_]+?_seed\d+$")
SETTINGS = ["base", "thr", "tta", "tta_thr"]
SETTING_NAMES = {"base": "threshold 0.5", "thr": "tuned threshold", "tta": "TTA", "tta_thr": "TTA + tuned threshold"}
CHAIN = ["U-Net", "CAR-UNet", "CAR-UNet + DA", "CAR-UNet + DA + clDice", "CAR-UNet + DA + clDice @1024"]
COLS = ["F1_Dice", "Sensitivity", "Precision", "AUC_ROC", "clDice"]
HI = ["hires_F1_Dice", "hires_clDice"]


def to_markdown(df):
    cells = [[f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row] for row in df.itertuples(index=False)]
    header = [str(c) for c in df.columns]
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in cells])


def ms(values):
    values = [v for v in values if not pd.isna(v)]
    if not values:
        return "n/a"
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def selected_setting(post):
    """The prediction setting with the best VALIDATION Dice (the test set is never used for this choice)."""
    single, tta = post["val_dice_curve"]["single"], post["val_dice_curve"]["tta"]
    val = {"base": single["0.5"], "thr": max(single.values()), "tta": tta["0.5"], "tta_thr": max(tta.values())}
    return max(SETTINGS, key=lambda st: (round(val[st], 6), -SETTINGS.index(st)))  # ties -> simpler setting


def main():
    main_runs, post_runs = {}, {}
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        name = os.path.basename(run_dir)
        if not RUN_PATTERN.match(name):
            continue
        if os.path.exists(os.path.join(run_dir, "summary.json")):
            main_runs[name] = json.load(open(os.path.join(run_dir, "summary.json")))
        if os.path.exists(os.path.join(run_dir, "post", "summary.json")):
            post_runs[name] = json.load(open(os.path.join(run_dir, "post", "summary.json")))

    lines = ["# Improvement 4 (tuned threshold + TTA) and final ablation", "",
             "All metrics on the 200 FIVES test images, scored at 512x512 inside the FOV; ± = std over seeds.",
             "Thresholds are tuned on the 120 validation images only. `hires_*` = scored against the 1024x1024 ground truth.", ""]

    # 1. Effect of each prediction setting, per model.
    rows = []
    by_model = {}
    for run, s in post_runs.items():
        by_model.setdefault(s["model"], []).append((run, s))
    order = [m for m in CHAIN if m in by_model] + sorted(set(by_model) - set(CHAIN))
    for model in order:
        runs = by_model[model]
        for st in SETTINGS:
            row = {"Model": model, "Setting": SETTING_NAMES[st], "Seeds": len(runs),
                   "Threshold": ms([s["settings"][st]["threshold"] for _, s in runs])}
            for c in COLS + HI:
                vals = [s["settings"][st]["overall"].get(c, np.nan) for _, s in runs]
                row[c] = ms(vals)
            rows.append(row)
    if rows:
        settings_df = pd.DataFrame(rows)
        settings_df.to_csv(os.path.join(OUT_DIR, "post_settings_table.csv"), index=False)
        lines += ["## 1. Prediction settings per model", "", to_markdown(settings_df), ""]

        # 2. Paired test: validation-selected setting vs threshold 0.5, same model, same images.
        lines += ["## 2. Improvement 4 with the setting chosen on VALIDATION (paired Wilcoxon vs threshold 0.5, "
                  "per-image Dice averaged over seeds)", ""]
        for model in order:
            per_seed, chosen = [], []
            for run, post in by_model[model]:
                sel = selected_setting(post)
                chosen.append(f"{SETTING_NAMES[sel]} @ {post['settings'][sel]['threshold']:.2f}")
                base = pd.read_csv(os.path.join(RUNS_DIR, run, "post", "test_metrics_base.csv")).set_index("filename")
                best = pd.read_csv(os.path.join(RUNS_DIR, run, "post", f"test_metrics_{sel}.csv")).set_index("filename")
                per_seed.append((base["F1_Dice"], best["F1_Dice"], base["clDice"], best["clDice"]))
            b_, t_, bc, tc = [pd.concat([x[i] for x in per_seed], axis=1).mean(axis=1) for i in range(4)]
            if np.allclose(t_, b_):
                lines.append(f"- **{model}** ({len(per_seed)} seeds): validation chose threshold 0.5 (no change) — chosen: {chosen}")
                continue
            _, p = wilcoxon(t_, b_)
            _, pc = wilcoxon(tc, bc)
            lines.append(f"- **{model}** ({len(per_seed)} seeds; chosen per seed: {', '.join(chosen)}): Dice {t_.mean() - b_.mean():+.4f} "
                         f"(better on {int((t_ > b_).sum())}/{len(t_)} images, p = {p:.3g}); clDice {tc.mean() - bc.mean():+.4f} (p = {pc:.3g})")
        lines.append("")

    # 3. Final ablation table: each step of the improvement chain (+ Improvement 4 with the validation-selected setting).
    abl = []
    for model in CHAIN:
        runs = {r: s for r, s in main_runs.items() if s["model"] == model}
        if not runs:
            continue
        row = {"Step": model, "Seeds": len(runs)}
        for c in COLS:
            row[c] = ms([s["overall"].get(c, np.nan) for s in runs.values()])
        posts = [post_runs[r] for r in runs if r in post_runs]
        row["hires_F1_Dice"] = ms([p["settings"]["base"]["overall"].get("hires_F1_Dice", np.nan) for p in posts]) if posts else "n/a"
        abl.append(row)
        if posts:
            prow = {"Step": f"{model} + Improvement 4 (setting chosen on validation)", "Seeds": len(posts)}
            for c in COLS + ["hires_F1_Dice"]:
                vals = [p["settings"][selected_setting(p)]["overall"].get(c, np.nan) for p in posts]
                prow[c] = "n/a" if any(pd.isna(v) for v in vals) else ms(vals)  # never report a partial-seed mean
            abl.append(prow)
    if abl:
        abl_df = pd.DataFrame(abl)
        abl_df.to_csv(os.path.join(OUT_DIR, "ablation_table.csv"), index=False)
        lines += ["## 3. Ablation: every step of the improvement chain", "", to_markdown(abl_df), ""]

    with open(os.path.join(OUT_DIR, "POST_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
