# Legacy results (v1) — kept for reference only, do not report

These were the first FIVES results. They are superseded by `results/runs/` and
`results/RESULTS_TABLE.md` because the evaluation had these problems:

1. **Only 50 of 200 test images were scored**, and because filenames were sorted as text the
   50 images were 45 Glaucoma, 4 AMD, 1 DR, 0 Normal — not a multi-disease test.
2. **Validation set was Normal (68) + AMD (52) only** (last 120 filenames), so the "best"
   checkpoint was chosen without any DR or Glaucoma images.
3. **Training settings were not guaranteed identical** between U-Net and CAR-UNet
   (separate scripts with different defaults: 5 vs 15 epochs, 2 vs 6 patches/image).
4. **No fixed seed** and a single run per model, so differences could be random noise.
5. Training patches were cut once and reused every epoch.
6. The DRIVE row in `comparison_table.csv` compared different datasets (4 DRIVE validation
   images vs FIVES test images).
