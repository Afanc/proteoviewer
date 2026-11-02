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

### Protein Coverage: Peptide SD and Fold-Change Disagreement

For ordinary proteomics, rectangles span the peptide coordinates exported by
ProteoFlux. Both diagnostics use **log2 of positive, finite raw peptide
intensities**, restricted to the selected contrast. They do not use imputed
protein values, and the Raw/Log-only/Processed selector does not change them.

**Peptide SD** is pooled within-condition variation. For each peptide, subtract
its mean within each condition, then calculate:

`SD = sqrt(sum((log2_intensity - condition_mean)^2) / (N_valid - K_valid))`

`N_valid` counts finite log2 measurements; `K_valid` counts conditions with at
least one such measurement. A difference in average abundance between the
conditions therefore does not inflate SD. A condition with only one observation
contributes no residual degrees of freedom. If `N_valid <= K_valid`, SD is NA;
with sufficient constant measurements, SD is zero. The axis increases upward,
with 5% display padding below zero and above the upper bound (at least 1).
White minor gridlines mark half-unit steps.

Peptides with valid exported coordinates remain visible even when SD cannot be
calculated, including peptides with no usable measurements in the selected
contrast. They appear in a separate **grey NA row**, with `SD: NA (insufficient
observations)` in hover. If all SDs are unavailable, the plot shows just that
coverage row and an NA axis label. NA is not zero variance. Missing or ambiguous
coordinates still cannot be drawn.

**Fold-change disagreement** (colorbar: **Disagr.**) measures agreement with
other peptides from the same protein in the selected contrast. For `A_vs_B`:

`peptide log2FC = mean(log2 intensity in A) - mean(log2 intensity in B)`

`reference log2FC = median(log2FC of OTHER eligible peptide sequences)`

`disagreement = abs(peptide log2FC - reference log2FC)`

Each peptide must have **at least two positive, finite observations in each
condition** to have an eligible log2FC. The reference needs **at least two other
eligible peptide sequences**, excluding the target sequence. Calculate each
peptide's FC first, then take the median of those FCs; do not subtract medians of
condition abundances. Missing or nonpositive intensities are never imputed.
ProteoFlux's exported peptide sequence identities already combine charge states;
Viewer retains those identities, including modifications retained by the producer.
Missing or duplicate sequence identities are left unscored rather than aggregated.
Other eligible peptides contribute even if their coordinates cannot be drawn.
This calculation does not use exported protein abundance or rerun protein rollup.

Disagreement is in **log2 units**, with zero meaning the same estimated FC as
the reference. The sequential color scale is fixed across proteins at **0–2**;
the **≥2** endpoint caps color only. Hover reports uncapped disagreement, peptide
log2FC and reference log2FC, alongside SD and observed sample counts. **Grey means
unavailable (NA)** because the target or reference is ineligible; it does not mean
zero disagreement. The observed `n / N` is a total across the contrast and does
not replace the per-condition eligibility check.

This is a local agreement diagnostic, **not a validated PRM score or statistical
evidence of equivalence**. Estimates can be uncertain with small sample counts,
even when the minimum requirements are met. When the protein barely changes,
small disagreement provides limited information about whether a peptide tracks
protein changes. Disagreement can also reflect real biology, including different
proteoforms or modifications; it does not by itself establish a bad measurement.

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
