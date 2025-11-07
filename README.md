# Raman PCA & Maps Documentation Hub

This top-level README links the three notebook-based guides that document the Raman processing and visualization tools in this repository.

## Documentation map

- [`PCA_rodrigo/PCA_V9_bin_orig_README.ipynb`](PCA_rodrigo/PCA_V9_bin_orig_README.ipynb)
  - Explains how `PCA_V9_bin_orig.py` loads spectra, runs the combined PCA canvas, and exports metrics.
  - Details the centroid math, signed σ distances, and the Excel/TXT artefacts generated from each run.
- [`Raman Maps and Hebex Graphene/Hebex_Graphene_README.ipynb`]("Raman Maps and Hebex Graphene/Hebex_Graphene_README.ipynb")
  - Covers the Hebex graphene analysis script, including band windows, ratio thresholds, FWHM detection, and the figures saved to disk.
- [`Raman Maps and Hebex Graphene/maps-graphene_README.ipynb`]("Raman Maps and Hebex Graphene/maps-graphene_README.ipynb")
  - Documents the `g4.py` Raman map viewer: smoothing, SNR gating, band maxima, intensity ratios, and the six-panel visualization layout.

## How to read the notebooks

1. Open them directly on GitHub (they render via nbviewer) or inside VS Code / Jupyter Lab for the best experience.
2. Each notebook is pure Markdown—no execution required—so you can safely commit them as documentation.
3. If you need PDF or HTML versions, export through Jupyter (`File → Export Notebook As`).

## Quick navigation tips

- Use the GitHub file explorer to jump between the notebooks without cloning.
- When working locally, pin these notebooks in VS Code’s “Working Files” list for faster access.
- For MkDocs/Material builds, you can copy the Markdown cells into the `docs/` folder if you prefer a static site.

Feel free to extend this README with additional guides (e.g., SNV pipeline, GUI walkthroughs) as new documentation becomes available.