# Course Report (LaTeX)

This folder contains an ACM WebSci-style LaTeX report draft for the course project.

Recommended compile command:

```powershell
latexmk -pdf main.tex
```

If `latexmk` is unavailable, try:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The report uses local figures copied from `output/` into `report/figures/`.
