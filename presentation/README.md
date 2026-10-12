# Final presentation (LaTeX Beamer)

| File | What it is |
|---|---|
| `slides.tex` | Beamer source of the presentation (16:9, Madrid theme) |
| `refs.bib` | BibTeX references |
| `slides.pdf` | Compiled slides |
| `figures/` | All figures used in the slides |
| `make_slide_figures.py` | Regenerates the figures from the trained models and results (`python presentation/make_slide_figures.py` from the project root) |

## Before submitting — fill in the red placeholders
Search `slides.tex` for `\fillme{` and replace:
- group number / name (title slide and footer: `\author[Group \fillme{No.}]`)
- names and roll numbers of all members (title slide)
- each member's contribution (conclusion slide)

## Compile
- **Overleaf:** upload the whole `presentation/` folder, set the main file to `slides.tex`, compiler pdfLaTeX (or XeLaTeX) — BibTeX runs automatically.
- **Local (TeX Live / MiKTeX):** `pdflatex slides && bibtex slides && pdflatex slides && pdflatex slides`
