"""Small accessors for producer labels and provenance; no scientific analysis."""
import pandas as pd


def require_unique_labels(labels, name):
    index = pd.Index(labels)
    if not index.is_unique or index.hasnans:
        raise ValueError(f"{name} must have unique, non-missing identifiers")
    return index


def covariate_adjusted(adata, contrast):
    """Return per-feature producer provenance, or None for legacy files."""
    values = adata.varm.get("covariate_adjusted")
    if values is None:
        return None
    features = require_unique_labels(adata.var_names, "Feature axis")
    contrasts = require_unique_labels(adata.uns["contrast_names"], "Contrast axis")
    frame = pd.DataFrame(values, index=features, columns=contrasts)
    return frame[contrast].astype("boolean")


def covariate_inference_labels(adata, contrast):
    provenance = covariate_adjusted(adata, contrast)
    if provenance is None:
        return pd.Series("FT-adjusted (legacy; provenance unavailable)", index=adata.var_names)
    return provenance.map({True: "FT-adjusted", False: "Unadjusted (fallback)"}).fillna("Unknown provenance")


def kinase_method_labels(method="ksea"):
    """Vocabulary for shared kinase UI/exports; missing legacy method is KSEA."""
    method = str(method).strip().lower()
    if method == "phosx":
        return dict(method="PhosX", effect="KS score", score="Activity score",
                    effect_axis="Signed weighted KS score", effect_column="KS_SCORE",
                    score_column="ACTIVITY_SCORE", substrate_score_column="PHOSX_ACTIVITY_SCORE",
                    direction_column="PHOSX_DIRECTION")
    if method == "ksea":
        return dict(method="KSEA", effect="Mean shift", score="Z-score",
                    effect_axis="Mean substrate shift vs background (log2FC)", effect_column="MEAN_SHIFT",
                    score_column="ACTIVITY_Z_SCORE", substrate_score_column="KSEA_Z_SCORE",
                    direction_column="KSEA_DIRECTION")
    return dict(method=method, effect="Effect", score="Activity score",
                effect_axis="Effect", effect_column="EFFECT", score_column="ACTIVITY_SCORE",
                substrate_score_column="ACTIVITY_SCORE", direction_column="DIRECTION")
