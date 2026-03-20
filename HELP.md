# ProteoViewer Help / FAQ

This page provides a concise reference for interpreting results in ProteoViewer.

---

## Tabs Overview

- **Overview**
  - Identification counts
  - CV and rMAD distributions
  - PCA / MDS visualizations
  - Volcano plot

- **Preprocessing**
  - Quality control before/after filtering
  - Effects of filtering and normalization

- **Analysis**
  - Log2FC distribution
  - p-value and q-value distributions
  - Hierarchical clustering

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
  - **Phospho (adj.)**: covariate-adjusted phospho signal
  - **Phospho (raw)**: raw phospho signal
  - **Flowthrough**: protein-level (covariate) signal

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

### Coloring

Points can be colored by:

- **Significance** (default)
- **nrSC**
- **Average intensity**
- **Average IBAQ** (if available)

---

## Data Views (right panel)

These affect the **barplot and detailed views**, not the volcano itself:

- **Raw**: original intensities
- **Log-only**: log-transformed, before normalization
- **Final / Normalized**: processed intensities used for statistics
- **Spectral Counts**: number of quantified precursors

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

---

### Export behavior (priority)

When exporting from the volcano:

1. **Single clicked datapoint** → export that protein
2. **Cohort defined** → export cohort
3. **Lasso / box selection** → export selected proteins

---

## Contact

Proteomics Core Facility  
Biozentrum – University of Basel  
dariush.mollet@unibas.ch
