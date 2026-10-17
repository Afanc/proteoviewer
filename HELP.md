# ProteoViewer Help / FAQ

This page provides a concise reference for interpreting results in ProteoViewer.

---

## Display

If the layout does not fit your screen, adjust browser zoom:

- Zoom in/out: `Ctrl +` / `Ctrl -`
- Reset: `Ctrl 0`

---

## Tabs Overview

- **Overview**
  - Identification counts
  - CV and rMAD distributions
  - PCA / MDS visualizations
  - Volcano plot and data exploration

- **Preprocessing**
  - Quality control
  - Effects of filtering, normalization and imputation

- **Analysis**
  - Log2FC distribution
  - p-value and q-value distributions
  - Hierarchical clustering

---

## Plots (interactive)

All plots are interactive (Plotly):

- Zoom / pan / select points
- Hover for details
- Export as **PNG** using the camera icon (first top right)

Avoid screenshots - use the built-in export :)

---

## Metrics

### CV (Coefficient of Variation)
Computed on **linear scale**:

CV = standard deviation / mean

---

### rMAD (relative Median Absolute Deviation)
Robust alternative to CV:

rMAD = MAD / median

where MAD = median(|x - median(x)|)

Less sensitive to outliers than CV.

---

## PCA / MDS

Standard dimensionality reduction methods for sample relationships.

Only available PCA axis pairs are offered. If the result contains fewer than
two components, the 2D plot shows an informative empty state.

- PCA: https://en.wikipedia.org/wiki/Principal_component_analysis
- MDS: https://en.wikipedia.org/wiki/Multidimensional_scaling


---
## Volcano Plot

Volcano plots display differential expression results for a selected contrast.

- **Log2FC**: effect size between conditions
- **q-value**: empirical Bayes–moderated, multiple-testing corrected p-value

### Volcano types

- **Proteomics / Peptidomics**: standard volcano
- **Phosphoproteomics**:
  - **Phospho (model)**: producer-selected adjusted or unadjusted fallback statistics;
    detail shows the inference used when provenance is available.
  - **Phospho (raw)**: raw phospho signal
  - **Flowthrough**: protein-level (covariate) signal

---

### Coloring

Points can be colored by:

- **Significance** (default)
- **nrSC** (see below)
- **Average intensity**
- **Average IBAQ** (if available, see below)

---

### Filters

#### Minimum / condition
Minimum number of **observed (non-imputed) values per condition** required (in at least one condition of the selected contrast).

---

#### Consistent precursors / peptides
At least **N peptides/precursors present in all samples of at least one condition**:

- Proteomics: peptides
- Peptidomics: precursors

Ensures the protein signal is supported by multiple independent measurements.

---

#### Max nrSC misalignment
Maximum allowed nrSC misalignment per protein, keeps proteins with nrSC misalignment ≤ threshold (see below).

---

### Cohort Inspector

Define and explore groups of proteins.

Search by:
- FASTA headers
- Gene names
- UniProt IDs
- Regex
- Input file

Used for highlighting and export.

---

### Data Views (right panel)

These affect the **barplot and detailed views**, not the volcano itself:

- **Raw**: original intensities
- **Log-only**: log-transformed, before normalization
- **Final / Normalized**: processed intensities used for statistics
- **Spectral Counts**: number of quantified precursors, when available.
  Missing evidence hides this option; phospho detail shows “Not available”, not zero.

Normalization distributions retain valid zero and negative log values. 

---

### Barplot & Detailed View

Per-sample intensities for the selected protein.

Values depend on the selected data view.

---

### Peptide Trends

Peptide trends are **centered per peptide** (not absolute intensity values)

---

### INDEX

Each protein/feature has an **INDEX** that corresponds to rows in exported `.xlsx` tables.

---

### iBAQ (Intensity-Based Absolute Quantification)

Approximate absolute protein abundance based on peptide intensities.

Reference:
Schwanhäusser et al., *Nature*, 2011
https://doi.org/10.1038/nature10098

---

### nrSC (normalized relative Spectral Counts)

Computed per contrast (A vs B):

nrSC = (A - B) / (A + B)

- A = sum of spectral counts in condition A
- B = sum of spectral counts in condition B

Properties:
- Range: [-1, 1], reflects consistency of detection
- Expected to have same sign as Log2FC, useful as a sanity check

Important:
- Not a statistical metric
- Strict filtering may discard valid signal

#### nrSC misalignment

Computed from nrSC and Log2FC:

nrSC_misalignment = 1 − sign(Log2FC) * nrSC

Properties:
- Range: [0, 2]
- Low values: consistent signal (intensity and spectral support agree)
- High values: potential inconsistencies (e.g. low support, noise, or single hits in one condition)

Important:
- Sensitive to low-effect sizes (small Log2FC may flip sign)

---

### Export behavior (priority)

When exporting from the volcano:

1. **Single clicked datapoint** → export that protein
2. **Cohort defined** → export cohort
3. **Lasso / box selection** → export selected proteins

Phospho CSVs retain the existing `ADJUSTED_*` columns for standard model results
and add `COVARIATE_ADJUSTED_<CONTRAST>` when provided by ProteoFlux (`False`
indicates fallback inference). The statistics themselves are unchanged.

### Kinase methods and exports

KSEA displays mean shift and Z-score. PhosX displays signed weighted KS and
Activity score; shared components identify the producer method in titles,
details and legends. Kinase CSVs include `METHOD`. PhosX selection columns are
`KS_SCORE`, `ACTIVITY_SCORE` and `PHOSX_DIRECTION`; its substrate CSV uses
`PHOSX_ACTIVITY_SCORE`. KSEA retains its previous column names.

### PELSA concentrations

Dose grouping, sorting and curve labels preserve the loaded numeric
concentrations. "Control" follows the producer's control designation, including
nonzero/negative control sentinels.

---

## Contact

Proteomics Core Facility  
Biozentrum – University of Basel  
dariush.mollet@unibas.ch
