# Agent Task Instructions: Upgrade Retinal Vessel Segmentation Project
## From DRIVE + U-Net → FIVES + CAR-UNet

**Context for the agent:** There is an existing project repository at
`https://github.com/Abhishek-Ramesh-19/Medical-Image-processing-Project`
containing a working U-Net-based retinal blood vessel segmentation pipeline trained
on the DRIVE dataset (folders: `data/DRIVE`, `scripts/`, `src/`, `main_notebook.ipynb`,
`requirements.txt`, `research paper_baseline.pdf`).

**Goal:** Upgrade this project by (1) replacing the DRIVE dataset with the larger
FIVES dataset, and (2) upgrading the model from plain U-Net to CAR-UNet
(Channel Attention Residual U-Net), based on a real IEEE-published paper.
The original U-Net + DRIVE pipeline must be preserved and NOT deleted — it
serves as the "Review 1 baseline" for comparison against the new "Review 2" upgrade.

Do not skip steps. Complete them in order. After each step, verify the output
before moving to the next step.

---

## Reference Materials (use these exact sources — do not substitute)

| Item | Name | Link |
|---|---|---|
| Baseline paper (Review 1, already done) | U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015) | https://arxiv.org/abs/1505.04597 |
| Upgrade paper (Review 2, to implement) | Channel Attention Residual U-Net for Retinal Vessel Segmentation (Guo et al.) | IEEE Xplore: https://ieeexplore.ieee.org/document/9414282/ <br> arXiv (free full text): https://arxiv.org/abs/2004.03702 |
| Upgrade paper's official code | CAR-UNet official repo | https://github.com/clguo/CAR-UNet |
| Old dataset (used in Review 1) | DRIVE | https://drive.grand-challenge.org/ |
| New dataset (to use for Review 2) | FIVES | Paper: https://doi.org/10.1038/s41597-022-01564-3 <br> Download: https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169 |
| Existing project repo | Medical-Image-processing-Project | https://github.com/Abhishek-Ramesh-19/Medical-Image-processing-Project |

---

## STEP 1 — Clone and inspect the existing repository

1. Clone the repo:
   ```bash
   git clone https://github.com/Abhishek-Ramesh-19/Medical-Image-processing-Project.git
   cd Medical-Image-processing-Project
   ```
2. Open and read `main_notebook.ipynb`, everything inside `src/`, and everything
   inside `scripts/`. Produce a short written summary covering:
   - What preprocessing steps are currently implemented (e.g., green channel
     extraction, CLAHE, noise removal, normalization)
   - How patches/training samples are generated from the DRIVE images
   - The exact U-Net architecture currently used (number of layers, filter
     sizes, activation functions, loss function, optimizer)
   - How the model is trained, evaluated, and which metrics are currently
     computed (accuracy, sensitivity, specificity, AUC, Dice, etc.)
3. Read `requirements.txt` and note the exact framework in use (TensorFlow/Keras
   or PyTorch) and versions — this determines how CAR-UNet must be implemented
   later, since the framework must match the rest of the codebase.
4. Do NOT modify anything yet. This step is read-only reconnaissance.

**Deliverable for this step:** A short markdown summary of the current codebase
(save it as `CURRENT_STATE_SUMMARY.md` in the repo root).

---

## STEP 2 — Download and prepare the FIVES dataset

1. Download the FIVES dataset from the Figshare link above (or, if direct
   download is not possible in this environment, document the manual download
   steps needed and flag this back to the user).
2. Create a new folder: `data/FIVES/`
3. Organize it to mirror the existing `data/DRIVE/` folder structure as closely
   as possible, i.e.:
   ```
   data/FIVES/
   ├── train/
   │   ├── images/
   │   └── masks/
   └── test/
       ├── images/
       └── masks/
   ```
4. FIVES images are high resolution (2048×2048). Write a preprocessing script
   `scripts/resize_fives.py` that resizes all images and masks to a consistent
   working resolution (recommend 512×512 or 768×768 — choose based on
   available GPU memory) and saves them into `data/FIVES_resized/` so the
   original high-res files are preserved separately.
5. Verify the new dataset: print the count of train/test images and masks,
   confirm image-mask pairs match by filename, and display 3 sample
   image-mask pairs side by side to confirm correctness. Save this
   verification output as `outputs/fives_dataset_check.png`.

**Do NOT delete or overwrite `data/DRIVE/`.** It remains the Review 1 dataset.

---

## STEP 3 — Port the existing preprocessing pipeline to FIVES

1. Copy the current preprocessing logic (green channel extraction, CLAHE
   enhancement, noise removal/filtering, normalization) from `src/` into a new
   file `src/preprocessing_fives.py`.
2. Update file paths so it reads from `data/FIVES_resized/` instead of
   `data/DRIVE/`.
3. Keep the preprocessing logic itself unchanged — the point of this step is
   only to repoint it at the new dataset, not to redesign the preprocessing.
4. Run the updated preprocessing on a few sample FIVES images and save
   before/after comparison images to `outputs/fives_preprocessing_examples/`
   so the effect of preprocessing can be visually verified.

---

## STEP 4 — Update patch extraction / data loading for FIVES

1. Copy the current patch-extraction / data-loading code into
   `src/data_loader_fives.py`.
2. Update it to read from the preprocessed FIVES images instead of DRIVE.
3. Since FIVES has ~20x more images than DRIVE, adjust patch count per image
   downward if needed to keep total training patch count reasonable and
   avoid excessive training time — document whatever value is chosen and why.
4. Keep the train/validation split logic consistent with how it was done for
   DRIVE (same split ratio).
5. Print final counts: total training patches, validation patches, test images.

---

## STEP 5 — Re-run the EXISTING U-Net model on FIVES (updated baseline)

1. Using the current U-Net model code (unchanged architecture), train it on
   the new FIVES data pipeline created in Steps 3–4.
2. Save this trained model as `models/unet_fives.h5` (or `.pt` depending on
   framework).
3. Evaluate using the exact same metrics already used for the DRIVE baseline:
   Accuracy, Sensitivity, Specificity, F1/Dice score, AUC-ROC.
4. Save results to `results/unet_fives_metrics.json` (or `.csv`).
5. Save 5 example predicted vessel masks next to their ground truth as
   `outputs/unet_fives_predictions/`.

**Purpose:** this proves whether the bigger dataset alone improves results,
independent of any architecture change. Keep the DRIVE-trained model and its
metrics untouched for comparison.

---

## STEP 6 — Implement CAR-UNet (the paper upgrade)

1. Study the official CAR-UNet repository (https://github.com/clguo/CAR-UNet)
   and the arXiv paper (https://arxiv.org/abs/2004.03702) to understand two
   specific architectural additions on top of plain U-Net:
   - **Modified Efficient Channel Attention (MECA):** applied to the skip
     connections between encoder and decoder, instead of directly
     concatenating encoder features into the decoder path
   - **Channel Attention Double Residual Block (CADRB):** replaces the plain
     convolutional blocks in the encoder/decoder with residual blocks that
     incorporate channel attention
2. Implement this as a new file `src/car_unet.py`, written in the SAME
   framework already used in the existing repo (match Step 1's findings —
   do not introduce a second framework).
3. Reuse as much of the existing U-Net code structure as possible (e.g., same
   input/output shapes, same loss function options) so the two models are
   directly comparable — only the internal blocks should differ according to
   the paper's design.
4. Add a short docstring at the top of `car_unet.py` citing the paper:
   `Guo, C. et al. "Channel Attention Residual U-Net for Retinal Vessel
   Segmentation." IEEE, 2021. https://arxiv.org/abs/2004.03702`
5. Do a quick sanity check: pass a dummy batch of the correct input shape
   through the model and confirm the output shape matches (should be same
   H×W as input, 1 channel, values in [0,1] after sigmoid).

---

## STEP 7 — Train CAR-UNet on FIVES (Review 2 model)

1. Using the identical data pipeline from Steps 3–4 (same train/val/test
   split as the Step 5 U-Net-on-FIVES run — this is critical for a fair
   comparison), train the new CAR-UNet model.
2. Use the same optimizer, learning rate schedule, and number of epochs as
   the Step 5 baseline unless the CAR-UNet paper specifies different
   hyperparameters — if so, follow the paper's settings and note the
   difference.
3. Save the trained model as `models/car_unet_fives.h5` (or `.pt`).
4. Evaluate with the same metrics: Accuracy, Sensitivity, Specificity,
   F1/Dice, AUC-ROC. Save to `results/car_unet_fives_metrics.json`.
5. Save 5 example predicted masks (same test images used in Step 5, for
   direct visual comparison) as `outputs/car_unet_fives_predictions/`.

---

## STEP 8 — Build the comparison report

1. Create a single script `scripts/compare_results.py` that loads all three
   metrics files:
   - Original U-Net on DRIVE (Review 1 baseline — already exists in the repo)
   - U-Net on FIVES (Step 5)
   - CAR-UNet on FIVES (Step 7)
2. Output a single comparison table (as both a printed table and a saved
   `results/comparison_table.csv`) with columns:
   `Model | Dataset | Accuracy | Sensitivity | Specificity | Dice | AUC-ROC`
3. Generate a side-by-side visual figure showing, for 3 sample test images:
   original image → ground truth mask → U-Net(DRIVE) prediction →
   U-Net(FIVES) prediction → CAR-UNet(FIVES) prediction. Save as
   `outputs/final_comparison_figure.png`.
4. Write a short markdown summary `RESULTS_SUMMARY.md` explaining, in plain
   language, whether performance improved from (a) DRIVE→FIVES and (b)
   U-Net→CAR-UNet, using the actual numbers produced.

---

## STEP 9 — Update project documentation

1. Update the existing `REVIEW.md` file (or create `REVIEW2.md` if it should
   stay separate) to include:
   - Statement that the dataset was upgraded from DRIVE (40 images) to FIVES
     (800 images) to address data scarcity, citing the FIVES paper
     (https://doi.org/10.1038/s41597-022-01564-3)
   - Statement that the model was upgraded from plain U-Net to CAR-UNet,
     citing the CAR-UNet paper (https://arxiv.org/abs/2004.03702 /
     IEEE: https://ieeexplore.ieee.org/document/9414282/)
   - The final comparison table from Step 8
   - The final comparison figure from Step 8
2. Update `requirements.txt` if any new libraries were needed for FIVES
   handling or CAR-UNet implementation.
3. Do not remove the original `research paper_baseline.pdf` — add a new file
   `research_paper_upgrade.pdf` (the CAR-UNet paper PDF) alongside it.

---

## Final Checklist (confirm all before reporting completion)

- [ ] `data/DRIVE/` untouched, original U-Net results preserved
- [ ] `data/FIVES/` and `data/FIVES_resized/` created and verified
- [ ] Preprocessing ported to FIVES and visually verified
- [ ] U-Net retrained on FIVES with metrics saved
- [ ] CAR-UNet implemented, matching the paper's architecture description
- [ ] CAR-UNet trained on FIVES with metrics saved (same test split as U-Net/FIVES run)
- [ ] Comparison table and figure generated
- [ ] REVIEW.md / REVIEW2.md updated with citations and results
- [ ] requirements.txt updated if needed
- [ ] Nothing from the original Review 1 work was deleted or overwritten
