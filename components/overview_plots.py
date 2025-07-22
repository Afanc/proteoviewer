import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from components.plot_utils import plot_stacked_proteins_by_category, plot_violins, compute_metric_by_condition, get_color_map, plot_cluster_heatmap_plotly, plot_volcanoes

from utils import logger, log_time


@log_time("Plotting barplot proteins per sample")
def plot_barplot_proteins_per_sample(
    adata,
    matrix_key: str = "normalized",
    bar_color: str = "teal",
    title: str = "Proteins Detected per Sample",
    width: int = 900,
    height: int = 500,
) -> go.Figure:
    """
    Count proteins per sample and draw as a bar plot using the generic helper.
    """
    # call the generic bar helper

    fig = plot_stacked_proteins_by_category(adata)
    return fig

@log_time("Plotting violins metrics per sample")
def plot_violin_cv_rmad_per_condition(
    adata,
    matrix_key: str = "normalized",
    title: str = "%CV / rMAD per Condition",
    width: int = 900,
    height: int = 800,
) -> list[go.Figure]:

    samples = adata.obs.index.tolist()
    conditions = adata.obs['CONDITION']

    # Vectorized + safe computation
    cv_dict = compute_metric_by_condition(adata, metric="CV")
    rmad_dict = compute_metric_by_condition(adata, metric="rMAD")

    # 3) draw grouped violins

    labels = list(cv_dict.keys())
    color_map = get_color_map(labels,
                              palette=px.colors.qualitative.Plotly,
                              anchor="Total",
                              anchor_color="gray")

    cv_fig = plot_violins(
                 data=cv_dict,
                 colors=color_map,
                 title="%CV per Condition",
                 width=width,
                 height=height,
                 x_title="Condition",
                 y_title="%CV",
                 showlegend=False,
                 )

    # enforce that x-axis uses exactly this array, not an alphabetical sort:
    rmad_fig = plot_violins(
                   data=rmad_dict,
                   colors=color_map,
                   title="%rMAD per Condition",
                   width=width,
                   height=height,
                   x_title="Condition",
                   y_title="%rMAD",
                   showlegend=False,
                   )
    return [cv_fig, rmad_fig]


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

    # so much faster than plotly. maybe plotly overkill.
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = sns.clustermap(df_z, method="ward", metric="euclidean", figsize=(12,10), cmap="RdBu_r")
    return fig.figure
    #fig = plot_cluster_heatmap_plotly(
    #    data=df_z,
    #    y_labels=y_labels,
    #    cond_series=cond_ser,
    #    colorscale="RdBu",
    #    title="Clustergram of All Samples",
    #    sample_linkage=sample_linkage,
    #    feature_linkage=feature_linkage,
    #)
    #return fig


def plot_h_clustering_heatmap_old(im):

    meta = im.dfs["protein_metadata"].to_pandas()
    df = pd.DataFrame(im.matrices["imputed"],
                      index=meta["INDEX"].tolist(),
                      columns=im.columns)

    df_z = df.sub(df.mean(axis=1), axis=0)

    gene_map = meta.set_index("INDEX")["GENE_NAMES"]
    y_labels = gene_map.reindex(df_z.index).tolist()

    cond_ser = (
        im.dfs["condition_pivot"]
          .to_pandas()
          .set_index("Sample")["CONDITION"]
    )

    fig = plot_cluster_heatmap_plotly(
        data=df_z,
        y_labels=y_labels,
        method="ward",
        metric="euclidean",
        colorscale="RdBu",
        cond_series=cond_ser.reindex(df_z.columns),
        title="Clustergram of All Samples"
    )
    return fig

@log_time("Plotting Volcano Plots")
def plot_volcanoes_wrapper(
    state,
    sign_threshold: float = 0.05,
    width: int = 900,
    height: int = 600,
    show_measured: bool = True,
    show_imp_cond1:  bool = True,
    show_imp_cond2:  bool = True,
    highlight: str = None,
    contrast: str = None,
) -> go.Figure:
    # simply forward the SessionState + args into the pure util
    fig =  plot_volcanoes(
        state=state,               # if `im` is your SessionState
        contrast=contrast,
        sign_threshold=sign_threshold,
        width=width,
        height=height,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        highlight=highlight,
    )

    return fig
