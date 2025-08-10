# components/de_plots.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple
from functools import lru_cache
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from components.plot_utils import plot_cluster_heatmap_plotly

from utils import log_time

def _get_contrast_index(adata, contrast: str) -> int:
    names = adata.uns.get("contrast_names")
    if names is None:
        # fallback: infer from varm shape
        n = adata.varm["log2fc"].shape[1]
        names = [f"C{i}" for i in range(n)]
    try:
        return list(names).index(contrast)
    except ValueError:
        return 0

def residual_variance_hist(adata) -> go.Figure:
    res_var = adata.uns.get("residual_variance")
    fig = px.histogram(
        x=np.asarray(res_var).ravel(),
        nbins=100,
        labels={"x": "Residual variance", "y": "Count"},
    )
    fig.update_layout(
        title=dict(text="Distribution of Variance of Residuals", x=0.5),
        yaxis_type="log",
        width=600,
        height=400,
    )
    fig.update_yaxes(
         dtick=1,
         exponentformat="power",
         showexponent="all",
    )
    fig.update_traces(hoverinfo='skip',
                      hovertemplate=None,
                      marker_line_width=1,
                      marker_line_color="black")
    return fig

def log2fc_histogram(adata, contrast: str) -> go.Figure:
    idx = _get_contrast_index(adata, contrast)
    vals = np.asarray(adata.varm["log2fc"][:, idx]).ravel()
    fig = px.histogram(
        x=vals, nbins=100,
        labels={"x": "log₂FC", "y": "Count"},
    )
    fig.update_layout(
        title=dict(text=f"log₂FC Distribution — {contrast}", x=0.5),
        width=600,
        height=400,
    )

    fig.update_traces(hoverinfo='skip', hovertemplate=None, marker_line_width=1, marker_line_color="black")
    return fig

def stat_histogram(adata, stat: str, contrast: str) -> go.Figure:
    """
    Overlay hist of raw vs eBayes for a given stat ('p' or 'q').
    """
    idx = _get_contrast_index(adata, contrast)
    if stat == "p":
        raw = np.asarray(adata.varm["p"][:, idx]).ravel()
        eb  = np.asarray(adata.varm["p_ebayes"][:, idx]).ravel()
        title = f"P-values — {contrast}"
        xlab  = "p"
    else:
        raw = np.asarray(adata.varm["q"][:, idx]).ravel()
        eb  = np.asarray(adata.varm["q_ebayes"][:, idx]).ravel()
        title = f"q-values — {contrast}"
        xlab  = "q"

    df = pd.DataFrame({xlab: np.concatenate([raw, eb]),
                       "source": (["Raw"] * raw.size) + (["eBayes"] * eb.size)})
    fig = px.histogram(
        df, x=xlab, color="source",
        nbins=50, barmode="overlay", opacity=0.55,
        category_orders={"source": ["Raw", "eBayes"]},
    )
    fig.update_layout(title=dict(text=title, x=0.5))
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(range=[0, 1])  # p/q in [0,1]
    fig.update_traces(hoverinfo='skip', hovertemplate=None, marker_line_width=1, marker_line_color="black")
    fig.update_layout(legend_title_text="")
    return fig

def stat_shrinkage_scatter(adata, stat: str, contrast: str, max_points: int = 50000) -> go.Figure:
    """
    X: raw, Y: eBayes. Uses Scattergl for speed; subsamples if huge.
    """
    idx = _get_contrast_index(adata, contrast)
    if stat == "p":
        x = np.asarray(adata.varm["p"][:, idx]).ravel()
        y = np.asarray(adata.varm["p_ebayes"][:, idx]).ravel()
        title = f"P-value shrinkage — {contrast}"
        xlab, ylab = "p (raw)", "p (eBayes)"
        line = [0, 1]
    else:
        x = np.asarray(adata.varm["q"][:, idx]).ravel()
        y = np.asarray(adata.varm["q_ebayes"][:, idx]).ravel()
        title = f"q-value shrinkage — {contrast}"
        xlab, ylab = "q (raw)", "q (eBayes)"
        line = [0, 1]

    n = x.size
    if n > max_points:
        idxs = np.random.choice(n, size=max_points, replace=False)
        x = x[idxs]; y = y[idxs]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="markers",
        marker=dict(size=5, opacity=0.5),
        name="proteins",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=line, y=line, mode="lines", line=dict(dash="dash", width=1.5, color="black"),
        name="y=x", showlegend=False
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=xlab, yaxis_title=ylab,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig

@log_time("Plotting Hierarchical Clustering")
def plot_h_clustering_heatmap(adata):
    # 1) Build (genes × samples) centered DataFrame
    mat = adata.X.copy()               # samples × genes
    # if you’ve stored the centered matrix in a layer, swap adata.X -> adata.layers["centered"]
    df_z = pd.DataFrame(
        adata.layers['centered'].T,
        index=adata.var_names,
        columns=adata.obs_names
    )

    # 2) Prepare labels & colors
    if "GENE_NAMES" in adata.var.columns:
        y_labels = adata.var["GENE_NAMES"].reindex(df_z.index).tolist()
    else:
        y_labels = df_z.index.tolist()

    cond_ser = adata.obs["CONDITION"].reindex(df_z.columns)

    # 3) Grab your precomputed linkages
    sample_linkage  = adata.uns["sample_linkage"]
    feature_linkage = adata.uns["feature_linkage"]

    # 4) Draw the heatmap using those linkages
    fig = plot_cluster_heatmap_plotly(
        data=df_z,
        y_labels=y_labels,
        cond_series=cond_ser,
        colorscale="RdBu",
        title="Clustergram of deviation to the mean",
        sample_linkage=sample_linkage,
        feature_linkage=feature_linkage,
    )
    return fig
