intro_preprocessing_text="""

#### Normalization

Normalization is performed in **two steps** to reduce systematic variation:

1. **Log₂ Transformation:**
   Converts the typically right-skewed intensity values into a more symmetric distribution and stabilizes variance across the dynamic range.

2. **Scaling:**
   Aligns samples for comparability using one of several methods:
   - Median equalization
   - Linear fitting (local or global)
   - LOESS fitting (local or global)


"""
log_transform_text="""
Proteomics intensity values span several orders of magnitude and are typically right-skewed.
Log2 transformation compresses this range, reduces skewness, and makes the data more symmetric.
It also stabilizes variance and allows fold changes to be interpreted as additive differences.

"""

before_after_normalization_violins="""These violin plots show the distribution of log2-transformed intensities before and after normalization, grouped by condition. After normalization, the distributions should become more centered and aligned across conditions. The dashed horizontal line marks the median intensity, which should stabilize if normalization was successful. Deviations from this pattern may indicate remaining batch effects or incomplete correction.
Here, violin plots display log2 intensities for individual samples before and after normalization. This view is useful for identifying sample-specific anomalies. After normalization, distributions should be comparable across samples. Samples with unusually broad distributions or shifted medians may be outliers or poorly normalized and should be reviewed carefully."""

normalization_metrics="""Both CV and rMAD are measures of variability:

    CV (Coefficient of Variation) = (Standard Deviation / Mean)

    rMAD (Relative Median Absolute Deviation) = (MAD / Median)
    where MAD is the median of absolute deviations from the median.

These metrics are computed across replicates within each condition. They quantify the consistency of protein intensities. In a well-normalized dataset:

    CV and rMAD should not increase after normalization.

    High CV/rMAD may reflect technical noise or biological heterogeneity.

We include both metrics because:

    CV is sensitive to outliers and skewed distributions.

    rMAD is more robust to outliers and provides a complementary view of spread.

By comparing CV and rMAD before and after normalization, you can assess whether normalization preserved biological signals while reducing technical variance."""
