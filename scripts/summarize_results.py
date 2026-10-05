"""
Builds the final comparison tables from every finished run in results/runs/.

Outputs:
    results/comparison_table.csv   overall test metrics per model, mean +/- std across seeds
    results/per_disease_table.csv  Dice / Sensitivity per disease group per model
    results/RESULTS_TABLE.md       the same tables in Markdown, plus a paired significance test

Smoke-test and tagged runs (folder names containing '_smoke' or a tag) are ignored.
"""
import os
import sys
import glob
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.environ.get("FIVES_RUNS_DIR", os.path.join(BASE_DIR, "results", "runs"))
OUT_DIR = os.environ.get("FIVES_OUT_DIR", os.path.join(BASE_DIR, "results"))
METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Dice", "IoU", "AUC_ROC", "AUC_PR"]
MODEL_ORDER = ["U-Net", "CAR-UNet"]
RUN_PATTERN = re.compile(r"^(unet|car_unet)_seed\d+$")


def load_runs():
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        name = os.path.basename(run_dir)
        summary_path = os.path.join(run_dir, "summary.json")
        per_image_path = os.path.join(run_dir, "test_metrics_per_image.csv")
        if not RUN_PATTERN.match(name) or not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        per_image = pd.read_csv(per_image_path)
        per_image["model"], per_image["seed"] = summary["model"], summary["seed"]
        runs.append((summary, per_image))
    return runs


def to_markdown(df):
    """Minimal Markdown table writer (avoids an extra 'tabulate' dependency)."""
    cells = [[f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row] for row in df.itertuples(index=False)]
    header = [str(c) for c in df.columns]
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in cells])


def fmt(mean, std, n):
    return f"{mean:.4f} ± {std:.4f}" if n > 1 else f"{mean:.4f}"


def main():
    runs = load_runs()
    if not runs:
        sys.exit(f"No finished runs found in {RUNS_DIR}")

    # One row per run (seed): the mean over all test images of that run.
    run_rows = pd.DataFrame([{"model": s["model"], "seed": s["seed"], "num_test_images": s["num_test_images"],
                              "best_epoch": s["best_epoch"], "parameters": s["parameters"],
                              "epochs": s["config"]["epochs"], "train_minutes": s["train_minutes"],
                              **s["overall"]} for s, _ in runs])
    if run_rows["num_test_images"].nunique() != 1:
        sys.exit("Runs were evaluated on different numbers of test images; re-evaluate before comparing.")
    if run_rows["epochs"].nunique() != 1:
        print("WARNING: runs used different epoch counts:", run_rows.groupby("model")["epochs"].unique().to_dict())

    per_image = pd.concat([p for _, p in runs], ignore_index=True)
    models = [m for m in MODEL_ORDER if m in set(run_rows["model"])]

    # Overall table: mean +/- std across seeds.
    overall_rows, overall_md = [], []
    for m in models:
        g = run_rows[run_rows["model"] == m]
        row = {"Model": m, "Dataset": "FIVES", "Seeds": len(g), "Test images": int(g["num_test_images"].iloc[0]),
               "Parameters": int(g["parameters"].iloc[0])}
        md = dict(row)
        for k in METRICS:
            row[k] = g[k].mean()
            row[f"{k}_std"] = g[k].std(ddof=1) if len(g) > 1 else 0.0
            md[k] = fmt(row[k], row[f"{k}_std"], len(g))
        overall_rows.append(row)
        overall_md.append(md)
    pd.DataFrame(overall_rows).to_csv(os.path.join(OUT_DIR, "comparison_table.csv"), index=False)

    # Per-disease table: average over seeds of the per-disease means.
    disease_rows = []
    for (m, d), g in per_image.groupby(["model", "disease"]):
        seed_means = g.groupby("seed")[["F1_Dice", "Sensitivity", "Specificity", "AUC_ROC"]].mean()
        disease_rows.append({"Model": m, "Disease": d, "Images": int(len(g) / g["seed"].nunique()),
                             **{k: seed_means[k].mean() for k in seed_means.columns}})
    disease_df = pd.DataFrame(disease_rows)
    disease_df["Model"] = pd.Categorical(disease_df["Model"], models)
    disease_df = disease_df.sort_values(["Disease", "Model"])
    disease_df.to_csv(os.path.join(OUT_DIR, "per_disease_table.csv"), index=False)

    # Paired test on the same test images: per-image Dice averaged over seeds.
    sig_lines = []
    if len(models) == 2:
        dice = per_image.groupby(["model", "filename"])["F1_Dice"].mean().unstack(0).dropna()
        a, b = dice[models[0]], dice[models[1]]
        stat, p = wilcoxon(b, a)
        sig_lines = [f"Paired Wilcoxon signed-rank test on per-image Dice ({len(dice)} test images, averaged over seeds):",
                     f"- mean Dice {models[1]} − {models[0]} = {np.mean(b - a):+.4f}",
                     f"- {models[1]} better on {int((b > a).sum())} / {len(dice)} images",
                     f"- p = {p:.4g} ({'significant' if p < 0.05 else 'NOT significant'} at 0.05)"]
        # Seed-level view: the image-level test above ignores run-to-run (seed) variation.
        seed_dice = run_rows.pivot(index="seed", columns="model", values="F1_Dice").dropna()
        if len(seed_dice) > 1:
            diff = seed_dice[models[1]] - seed_dice[models[0]]
            sig_lines += ["", f"Per-seed Dice difference ({models[1]} − {models[0]}), same seed for both models:",
                          "- " + ", ".join(f"seed {s}: {d:+.4f}" for s, d in diff.items()),
                          f"- mean {diff.mean():+.4f} ± {diff.std(ddof=1):.4f}; {models[1]} better in {int((diff > 0).sum())} / {len(diff)} seeds",
                          f"- seed-to-seed std of Dice: {models[0]} {seed_dice[models[0]].std(ddof=1):.4f}, {models[1]} {seed_dice[models[1]].std(ddof=1):.4f}"]

    lines = ["# FIVES Test Results (all models trained with identical settings)", "",
             "Metrics are computed per image inside the field of view (threshold 0.5) and averaged over all test images.",
             "With more than one seed, values are mean ± standard deviation across seeds.", "",
             "## Overall", "", to_markdown(pd.DataFrame(overall_md)), "",
             "## Per disease", "", to_markdown(disease_df), ""]
    if sig_lines:
        lines += ["## Is the difference real?", ""] + sig_lines + [""]
    lines += ["## Individual runs", "", to_markdown(run_rows.sort_values(["model", "seed"])), ""]
    with open(os.path.join(OUT_DIR, "RESULTS_TABLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
