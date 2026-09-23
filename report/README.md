# Report

IEEE-format, double-column report for the smart-surveillance project.

| File | What it is |
|---|---|
| `main.tex` | the report source (IEEEtran, `conference` mode) |
| `refs.bib` | 22 IEEE-style references |
| `figures/` | every figure the report includes, as vector PDF |
| `smart-surveillance-report.pdf` | the compiled 10-page output |

## Building

**On Overleaf** — upload this whole folder, set the compiler to **pdfLaTeX** and the
main document to `main.tex`. Overleaf runs the bibliography pass automatically.

**Locally** — with any TeX distribution:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

or, with [tectonic](https://tectonic-typesetting.github.io/) (no TeX install required):

```bash
tectonic -X compile main.tex --outdir build
```

## Regenerating the figures

The figures are produced by the project itself, not drawn by hand:

```bash
python scripts/make_diagrams.py     # Figs 1, 2, 5 — pipeline, flowchart, interdependence
python experiments/run_all.py       # Figs 3, 4 — notch spectrum, latency breakdown
cp results/figures/*.pdf report/figures/
```

`make_diagrams.py` reads the decision thresholds from `surveillance/pipeline.py` and the
measured couplings from `results/tables/`, so the diagrams cannot drift from the code or
from the numbers the experiments produced.

## Notes on the format

- The document is exactly **10 pages**, double column, US Letter — the IEEE conference
  standard that `IEEEtran` produces by default.
- The author block at the top of `main.tex` carries the four team members with their
  student IDs and the repository link. Add the Overleaf link there before submitting.
- Every number quoted in the text comes from `results/tables/`; run
  `python scripts/summarize_results.py` to print them all side by side.
