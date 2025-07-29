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
                   #width=width,
                   #height=height,
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

    # --- seaborn version
    #import seaborn as sns
    #import matplotlib.pyplot as plt
    #fig = sns.clustermap(df_z, method="ward", metric="euclidean", figsize=(12,10), cmap="RdBu_r")
    #return fig.figure
    # ---

    # --- plotly version
    fig = plot_cluster_heatmap_plotly(
        data=df_z,
        y_labels=y_labels,
        cond_series=cond_ser,
        colorscale="RdBu",
        title="Clustergram of All Samples",
        sample_linkage=sample_linkage,
        feature_linkage=feature_linkage,
    )
    return fig
    # ---


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
    color_by: str = None,
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
        color_by=color_by,
    )

    return fig

def plot_intensity_by_protein(state, contrast, protein, layer):
    ad = state.adata

    # pick your normalized layer (fallback to .X)
    proc_data = ad.X
    intensity_scale = "Log Intensity"
    if layer.value == "Raw":
        proc_data = ad.layers.get('raw')
        intensity_scale = "Intensity"
    elif layer.value == "Log-Normalized":
        proc_data = ad.layers.get('lognorm')

    # find column index by GENE_NAMES
    try:
        col = list(ad.var["GENE_NAMES"]).index(protein)
    except ValueError:
        return px.bar(pd.DataFrame({"x":[],"y":[]}))

    # extract processed and raw values for this protein
    y_vals = proc_data[:, col].A1 if hasattr(proc_data, "A1") else proc_data[:, col]
    raw_layer = ad.layers.get('raw', ad.X)
    raw_vals = raw_layer[:, col].A1 if hasattr(raw_layer, "A1") else raw_layer[:, col]
    imputed_mask = np.isnan(raw_vals)

    # build DataFrame
    df = pd.DataFrame({
        "sample": ad.obs_names,
        "condition": ad.obs["CONDITION"],
        "intensity": y_vals,
        "imputed": imputed_mask
    })

    # filter by contrast
    grp1, grp2 = contrast.split("_vs_")
    df = df[df["condition"].isin([grp1, grp2])]

    # color mapping
    conditions = sorted(ad.obs["CONDITION"].unique())
    color_map = get_color_map(conditions, palette=px.colors.qualitative.Plotly)

    # create initial bar chart
    fig = px.bar(
        df,
        x="sample",
        y="intensity",
        color="condition",
        pattern_shape="imputed",
        pattern_shape_map={False: "", True: "/"},
        labels={"intensity": f"{intensity_scale}", "sample": "Sample"},
        color_discrete_map=color_map,
    )
    fig.update_traces(
        marker_pattern_fillmode="overlay",
        marker_pattern_size=6,
        marker_pattern_solidity=0.3
    )

    # adjust legend: show each condition once, hide imputed from condition traces
    for trace in fig.data:
        trace.showlegend = False

    # 2) re-add one dummy bar per condition
    for cond in [grp1, grp2]:
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=cond,
            marker_color=color_map[cond],
            showlegend=True
        ))

    # add a custom legend entry for imputed pattern
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        name="Imputed",
        marker_color="white",
        marker_pattern_shape="/",
        marker_pattern_fillmode="overlay",
        marker_pattern_size=6,
        marker_pattern_solidity=0.3,
        showlegend=True
    ))

    fig.update_layout(
        margin={"t":40,"b":40,"l":40,"r":40},
        title="",
        legend_title_text="Conditions",
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )
    return fig

def get_protein_info(state, contrast, protein, layer):
    ad   = state.adata
    names = ad.var["GENE_NAMES"].astype(str)
    mask  = names == protein
    uniprot_id = ad.var_names[mask][0]

    layer_data = ad.X
    if layer.value == "Raw":
        layer_data = ad.layers.get('raw')
    elif layer.value == "Log-Normalized":
        layer_data = ad.layers.get('lognorm')

    # log2FC & q-value
    df_fc = pd.DataFrame(
        ad.varm["log2fc"],
        index=ad.var_names,
        columns=ad.uns["contrast_names"],
    )
    df_q  = pd.DataFrame(
        ad.varm.get("q_ebayes", ad.varm["q"]),
        index=ad.var_names,
        columns=ad.uns["contrast_names"],
    )

    logfc = df_fc.loc[uniprot_id, contrast]
    qval  = df_q.loc[uniprot_id, contrast]

    mat   = layer_data.toarray() if hasattr(layer_data, "toarray") else layer_data
    # pick samples for this contrast
    grp1, grp2 = contrast.split("_vs_")
    idx1 = ad.obs["CONDITION"] == grp1
    idx2 = ad.obs["CONDITION"] == grp2
    # compute mean across all samples
    col_idx = list(ad.var["GENE_NAMES"]).index(protein)
    vals = mat[np.logical_or(idx1, idx2), col_idx]
    avg_int = float(np.nanmean(vals))

    protein_info = {
        'uniprot_id': uniprot_id,
        'qval': qval,
        'logfc': logfc,
        'avg_int': avg_int,
        'index': col_idx,
    }
    return protein_info
