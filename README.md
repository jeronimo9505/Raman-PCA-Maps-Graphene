# Raman PCA & Maps Documentation Hub

This top-level README links the three notebook-based guides that document the Raman processing and visualization tools in this repository.

## Documentation 


- [`PCA_rodrigo/PCA_V9_bin_orig_README.ipynb`](PCA_rodrigo/PCA_V9_bin_orig_README.ipynb)
  - Provides a notebook walkthrough of the PCA analysis workflow and deliverables.
  - Explains how `PCA_V9_bin_orig.py` loads spectra, runs the combined PCA canvas, and exports metrics.
  - Details the centroid math, signed σ distances, and the Excel/TXT artefacts generated from each run.
- [`Raman Maps and Hebex Graphene/Hebex_Graphene_README.ipynb`](Raman%20Maps%20and%20Hebex%20Graphene/Hebex_Graphene_README.ipynb)
  - Serves as the comprehensive reference for the Hebex graphene characterisation pipeline.
  - Breaks down the `Hebex_Graphene.py` pipeline step by step: CLI options, input assumptions, and the band windows used for D, G, and 2D features.
  - Details peak metrics (intensity, position, area), FWHM computation, and how intensity ratios drive monolayer/damage thresholds.
  - Summarises every figure exported (hexbin correlation, FWHM histogram, quadrant spectra, composite panel) with file names.
- [`Raman Maps and Hebex Graphene/maps-graphene_README.ipynb`](Raman%20Maps%20and%20Hebex%20Graphene/maps-graphene_README.ipynb)
  - Acts as the user guide for the map viewer used to explore Raman datasets interactively.
  - Documents the `g4.py` Raman map viewer: TSV ingestion, Savitzky–Golay smoothing, SNR gating, and band maxima extraction.
  - Explains the ratio formulas with safeguards, colour scales used across the six subplots, and interaction shortcuts (click-to-view spectrum, copy-to-clipboard).
  - Notes implementation constraints (map orientation flip, masked pixels, parameter tweaks) for adapting the script to new datasets.
## How to read the notebooks

1. Open them directly on GitHub (they render via nbviewer) or inside VS Code / Jupyter Lab for the best experience.
2. Each notebook is pure Markdown—no execution required—so you can safely commit them as documentation.
3. If you need PDF or HTML versions, export through Jupyter (`File → Export Notebook As`).

## Quick navigation tips

- Use the GitHub file explorer to jump between the notebooks without cloning.
- When working locally, pin these notebooks in VS Code’s “Working Files” list for faster access.
- For MkDocs/Material builds, you can copy the Markdown cells into the `docs/` folder if you prefer a static site.


Feel free to extend this README with additional guides (e.g., SNV pipeline, GUI walkthroughs) as new documentation becomes available.

