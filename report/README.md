# Project report (Elsevier CAS single-column format)

Final project report for 22AIE437 Medical Image Processing, written with the Elsevier CAS single-column LaTeX template (`cas-sc`), as required by section 2A of the course guidelines.

| File | Content |
|---|---|
| `report.pdf` | Compiled report (13 pages) |
| `report.tex` | LaTeX source |
| `refs.bib` | BibTeX references (style `cas-model2-names`, author–year) |
| `figures/` | All figures used in the report |
| `make_report_figures.py` | Regenerates the report-only figures (dual attention diagram, four-disease prediction figures) |

## Before submission: fill in the group details

Every red `\fillme{...}` placeholder in `report.tex` must be replaced:

- the group number or name (title-page footnote and running footer `\shortauthors`)
- the names of members 2–4 (`\author[1]{...}`)
- the roll numbers of all members (title-page footnote)
- what each member did (*CRediT authorship contribution statement*, near the end)

Search `report.tex` for `fillme` to find all of them. Once they are filled in, the `\newcommand{\fillme}` line can stay as it is.

## Compiling

**Overleaf (easiest):** create a new project from the "Elsevier CAS LaTeX Single Column Template", then upload `report.tex`, `refs.bib` and the `figures/` folder, set `report.tex` as the main document and compile. Overleaf runs BibTeX automatically.

**Locally (TeX Live / MiKTeX):** the class `cas-sc` and the style `cas-model2-names` are in the `els-cas-templates` package.

```
pdflatex report
bibtex report
pdflatex report
pdflatex report
```

**Tectonic:** `tectonic report.tex` (downloads the needed packages on first use).

The single `Overfull \hbox` warning at `\maketitle` comes from the CAS template itself when natbib is loaded. The box is empty and does not change the output.

## Regenerating the figures

Most figures are copies of `presentation/figures/` (made by `presentation/make_slide_figures.py`). The pipeline figure is the same diagram with its title cropped. The report-only figures are regenerated from the project root with:

```
python report/make_report_figures.py
```

This loads the trained seed-0 checkpoints in `models/` and selects the example images automatically: the test image with the median baseline Dice in each disease group.

## Where the numbers come from

All numbers in the report are taken from `results/RESULTS_TABLE.md`, `results/POST_RESULTS.md` and the per-image CSV files in `results/runs/`, i.e. three training seeds per model evaluated on all 200 FIVES test images.
