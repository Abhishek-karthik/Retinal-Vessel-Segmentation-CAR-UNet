# Project report (Elsevier CAS single-column format)

Final project report for 22AIE437 Medical Image Processing, written with the Elsevier CAS single-column LaTeX template (`cas-sc`), as required by section 2A of the course guidelines.

| File | Content |
|---|---|
| `report.pdf` | Compiled report (14 pages) |
| `report.tex` | LaTeX source |
| `refs.bib` | BibTeX references (style `cas-model2-names`, author–year) |
| `figures/` | All figures used in the report |
| `make_report_figures.py` | Regenerates the report-only figures (dual attention diagram, preprocessing steps, four-disease prediction figures) |

## Authors (group members)

| Name | Roll number |
|---|---|
| Abhishek Karthik J | CB.SC.U4AIE23010 |
| Abhishek Ramesh | CB.SC.U4AIE23011 |
| Krish S | CB.SC.U4AIE23040 |
| Theerth Krish S | CB.SC.U4AIE23071 |

All four members contributed equally (see the CRediT statement in the report).

The text follows the course guidelines (section 2A) and Dr. Vipin Venugopal's guide *Writing a Good Journal or Conference Paper* (github.com/VipinBioLab/Research-paper-writing-guide). Every reference was checked against Crossref.

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
