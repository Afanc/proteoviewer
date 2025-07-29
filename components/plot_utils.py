"""
Low-level Plotly utilities for ProteoFlux panel app.
"""

import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
from typing import List, Sequence, Optional, Dict, Tuple, Union, Literal
from anndata import AnnData
import scanpy as sc
from utils import logger, log_time
from functools import lru_cache

def get_color_map(
    labels: List[str],
    palette: List[str] = None,
    anchor: str = "Total",
    anchor_color: str = "black",
) -> Dict[str,str]:
    """
    Returns a stable mapping label→color.

    - anchor: if present in labels, gets anchor_color.
    - the remaining labels (sorted) get colors from palette in order.
    """
    palette = palette or px.colors.qualitative.Plotly
    out: Dict[str,str] = {}
    # assign anchor first (so it's always the same)
    if anchor in labels:
        out[anchor] = anchor_color
    # assign the rest in sorted order
    #others = sorted(l for l in labels if l != anchor)
    others = [l for l in labels if l != anchor]
    for i, lbl in enumerate(others):
        out[lbl] = palette[i % len(palette)]
    return out

def categorize_proteins_by_run_count(df: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame of shape (samples × proteins), returns a Series
    indexed by protein with one of four categories:
      - “complete”: seen in every sample
      - “unique”  : seen in exactly one sample
      - “sparse”  : seen in ≤50% of the samples (but >1)
      - “shared”  : seen in >50% but <100% of the samples
    """
    # count, for each protein (i.e. each column), how many non‐NA values it has
    run_counts = df.notna().sum(axis=0)
    total_runs = df.shape[0]
    half_runs  = total_runs * 0.5

    def _cat(n):
        if n == total_runs: return "complete"
        if n == 1:          return "unique"
        if n <= half_runs:  return "sparse"
        return "shared"

    return run_counts.map(_cat)


def plot_stacked_proteins_by_category(
    adata,
    matrix_key: str = "normalized",
    category_colors: Dict[str,str] = None,
    width: int = 1200,
    height: int = 500,
    title: str = "Protein IDs by Sample and Category",
) -> go.Figure:

    samples = adata.obs.index.tolist()
    protein_ids  = adata.var.index.tolist()

    # now df is samples × proteins, no transpose needed
    df = pd.DataFrame(adata.layers['raw'],
                      index=samples,
                      columns=protein_ids)


    # 1) categorize each protein
    prot_cat = categorize_proteins_by_run_count(df)

    # 2) build a (sample × category) table of counts
    cats  = ["complete","shared","sparse","unique"]
    pivot = pd.DataFrame(
        {cat: df.loc[:, prot_cat[prot_cat==cat].index]
                 .notna().sum(axis=1)
         for cat in cats},
        index=samples
    )
    sample_conditions = adata.obs["CONDITION"]
    unique_conds      = sorted(sample_conditions.unique().tolist())
    cond_palette      = px.colors.qualitative.Plotly
    cond_color_map    = get_color_map(unique_conds, px.colors.qualitative.Plotly)
    edge_colors       = [cond_color_map[sample_conditions[s]] for s in samples]

    # default fills
    if category_colors is None:
        category_colors = {
            "complete": "darkgray",
            "shared"  : "lightgray",
            "sparse"  : "white",
            "unique"  : "red",
        }

    fig = go.Figure()

    # 5) actual bar traces (no legend entries)
    for cat in cats:
        fig.add_trace(go.Bar(
            x=samples, y=pivot[cat],
            marker_color=category_colors[cat],
            marker_line_color=edge_colors,
            marker_line_width=2,
            showlegend=False,
            hovertemplate=(
                f"Category: {cat.capitalize()}<br>"+
                "Count: %{y}<extra></extra>"
            ),
        ))

    # 6) total annotations
    totals = pivot.sum(axis=1)
    for s, tot in totals.items():
        fig.add_annotation(
            x=s, y=tot,
            text=str(tot),
            showarrow=False,
            yanchor="bottom"
        )

    # 7) Protein Category legend (left)
    for i, cat in enumerate(cats):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=cat.capitalize(),
            marker=dict(
                symbol="square",
                size=12,
                color=category_colors[cat],
                line=dict(color="black", width=2),
            ),
            legend="legend1",
            legendgrouptitle_text="Protein Category",
        ))

    # 8) Sample Condition legend (right)
    for j, cond in enumerate(unique_conds):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=cond,
            marker=dict(
                symbol="square",
                size=12,
                color="white",
                line=dict(color=cond_color_map[cond], width=2),
            ),
            legend="legend2",
            legendgrouptitle_text="Sample Condition",
        ))

    # 9) layout with two legends
    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.5),
        xaxis_title="Sample",
        yaxis_title="Number of Proteins",
        template="plotly_white",
        width=width, height=height,
        # first legend (categories) on the left
        legend=dict(
            x=1.40, y=1,
            xanchor="right", yanchor="top",
            bordercolor="black", borderwidth=1
        ),
        # second legend (conditions) on the right
        legend2=dict(
            x=1.20, y=1,
            xanchor="right", yanchor="top",
            bordercolor="black", borderwidth=1
        ),
        legend_itemclick=False,
        legend_itemdoubleclick=False,
        #margin=dict(l=120, r=120),
    )
    fig.update_xaxes(tickangle=45)
    return fig

def plot_bar_plotly(
    x: Sequence,
    y: Sequence,
    colors: Union[str, Sequence[str]] = "steelblue",
    orientation: str = "v",             # "v" or "h"
    width: int = 900,
    height: int = 500,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    template: str = "plotly_white",
    **bar_kwargs,                       # passed to go.Bar
) -> go.Figure:
    """
    Generic bar plot.

    Parameters
    ----------
    x : sequence of category labels
    y : sequence of numeric values
    colors : single color or sequence matching len(x)
    orientation : 'v' or 'h'
    **bar_kwargs : any go.Bar args (e.g. opacity, marker_line)
    """
    fig = go.Figure(go.Bar(
        x=x if orientation=="v" else y,
        y=y if orientation=="v" else x,
        marker_color=colors,
        orientation=orientation,
        **bar_kwargs
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template=template,
        width=width,
        height=height,
        barmode="group",
    )
    return fig

def compute_metric_by_condition(
    adata: AnnData,
    layer: str = "normalized",
    cond_key: str = "CONDITION",
    metric: Literal["CV", "rMAD"] = "CV",
) -> Dict[str, np.ndarray]:
    """
    Returns a dict mapping:
      - "Total" → array of length = n_proteins
      - each condition → same‐length array
    computed **per protein** across samples *without ever transposing*.
    """
    # 1) samples × proteins matrix
    df = pd.DataFrame(
        adata.X,
        index=adata.obs_names,      # samples
        columns=adata.var_names,    # proteins
    )
    conditions = adata.obs[cond_key]

    # 2) choose the right function (operate on axis=0 for proteins)
    def _cv(x: np.ndarray) -> np.ndarray:
        """
        Geometric CV on log2-normalized data (samples × proteins),
        returned as a percentage.
        """
        linear_values = 2**x

        std = np.nanstd(linear_values, axis=0)
        mean = np.clip(np.nanmean(linear_values, axis=0), 1e-6, None)

        cv = 100*std/mean

        return cv

    def _rmad(x: np.ndarray) -> np.ndarray:
        linear = 2 ** x

        med = np.nanmedian(linear, axis=0)
        mad = np.nanmedian(np.abs(linear - med), axis=0)

        rmad = 100 * mad / np.clip(med, 1e-6, None)

        return rmad

    compute_fn = _cv if metric == "CV" else _rmad

    out: Dict[str, np.ndarray] = {}
    # Global (“Total”)
    out["Total"] = compute_fn(df.values)

    # Per‐condition
    for cond, subdf in df.groupby(conditions, axis=0):
        out[cond] = compute_fn(subdf.values)

    return out

def compute_cv_by_condition(
    df_mat: pd.DataFrame,
    cond_map: pd.Series
) -> Dict[str, np.ndarray]:
    cvs: Dict[str, np.ndarray] = {}
    # 0) global CV across all samples
    mean_all = df_mat.mean(axis=1)
    sd_all   = df_mat.std(axis=1, ddof=1)
    cvs["Total"] = (sd_all / mean_all * 100).values

    # 1) per‐condition CV
    for cond in cond_map.unique():
        cols = cond_map[cond_map == cond].index
        sub  = df_mat[cols]
        mean = sub.mean(axis=1)
        sd   = sub.std(axis=1, ddof=1)
        cvs[cond] = (sd / mean * 100).values

    return cvs

def plot_violins(
    data: Dict[str, Sequence],
    colors: Dict[str,str] = None,
    title: str = None,
    width: int = 900,
    height: int = 500,
    y_title: str = None,
    x_title: str = None,
    showlegend: bool = True,
) -> go.Figure:
    """
    Draw one full violin per key in `data`.
    - data: {label: array_of_values}
    - colors: optional mapping label→color (falls back to Plotly palette)
    """
    labels = list(data.keys())

    # default palette if not provided
    default_colors = px.colors.qualitative.Plotly
    color_map = colors or dict(zip(labels, default_colors))

    fig = go.Figure()
    for lbl in labels:
        arr = np.asarray(data[lbl])
        fig.add_trace(go.Violin(
            x=[lbl]*len(arr),
            y=arr,
            name=lbl,
            legendgroup=lbl,
            line_color=color_map[lbl],
            opacity=0.6,
            width=0.7,
            box_visible=True,
            meanline_visible=True,
            points=False,
            hoverinfo="skip",
        ))

    y_all = np.concatenate(list(data.values()))
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    offset = (y_max - y_min) * 0.1
    for lbl in labels:
        med = np.nanmedian(data[lbl])
        fig.add_annotation(
            x=lbl, y=med+offset,
            text=f"{med:.2f}",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="black", weight="bold", size=11)
        )

    fig.update_layout(
        violinmode="group",
        template="plotly_white",
        title=dict(text=title, x=0.5) or "",
        autosize=True,
        width=width, height=height,
        xaxis=dict(title=x_title or "", showgrid=True),
        yaxis=dict(title=y_title or "", showgrid=True),
        showlegend=showlegend,
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    return fig


@log_time("Plotting PCA")
def plot_pca_2d(
    adata: AnnData,
    pc: tuple[int,int] = (1,2),
    color_key: str = "CONDITION",
    colors: dict[str,str] = None,
    title: str = "PCA",
    width: int = 900,
    height: int = 500,
    annotate: bool = False,
) -> go.Figure:
    """
    2D PCA scatter of samples, colored by adata.obs[color_key].
    """
    # build DataFrame
    pcs = adata.obsm["X_pca"][:, [pc[0]-1, pc[1]-1]]
    df = pd.DataFrame(pcs, columns=[f"PC{pc[0]}", f"PC{pc[1]}"],
                      index=adata.obs_names)
    df[color_key] = adata.obs[color_key].values

    # pick colors
    levels = sorted(df[color_key].unique().tolist())
    palette = get_color_map(levels, px.colors.qualitative.Plotly)

    fig = go.Figure()
    for lv in levels:
        sub = df[df[color_key] == lv]
        fig.add_trace(go.Scatter(
            x=sub[f"PC{pc[0]}"],
            y=sub[f"PC{pc[1]}"],
            mode="markers+text" if annotate else "markers",
            text=sub.index.to_list(),
            textposition="top center",
            name=lv,
            marker=dict(color=palette[lv], size=8, line=dict(width=1, color="black")),
            hovertemplate = (f"Sample: %{{text}}")
        ))
    # axis labels with explained variance
    var = adata.uns["pca"]["variance_ratio"]
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=width, height=height,
        template="plotly_white",
        xaxis=dict(title=f"PC{pc[0]} ({var[pc[0]-1]*100:.1f}% var)"),
        yaxis=dict(title=f"PC{pc[1]} ({var[pc[1]-1]*100:.1f}% var)"),
    )
    fig.update_xaxes(showline=True, mirror=True, linecolor="black")
    fig.update_yaxes(showline=True, mirror=True, linecolor="black")
    return fig


@log_time("UMAP")
def plot_umap_2d(
    adata: AnnData,
    color_key: str = "CONDITION",
    colors: dict[str,str] = None,
    title: str = "UMAP",
    width: int = 900,
    height: int = 500,
    annotate: bool = False,
) -> go.Figure:
    """
    2D UMAP scatter of samples, colored by adata.obs[color_key].
    """
    # ensure neighbors + UMAP are present
    df = pd.DataFrame(
        adata.obsm["X_umap"][:, :2],
        columns=["UMAP1","UMAP2"],
        index=adata.obs_names
    )
    df[color_key] = adata.obs[color_key].values

    levels = sorted(df[color_key].unique().tolist())
    palette = get_color_map(levels, px.colors.qualitative.Plotly)

    fig = go.Figure()
    for lv in levels:
        sub = df[df[color_key] == lv]
        fig.add_trace(go.Scatter(
            x=sub["UMAP1"],
            y=sub["UMAP2"],
            mode="markers+text" if annotate else "markers",
            text=sub.index.to_list(),
            textposition="top center",
            name=lv,
            marker=dict(color=palette[lv], size=8, line=dict(width=1, color="black")),
            hovertemplate="Sample: %{text}")
        )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=width, height=height,
        template="plotly_white",
        xaxis=dict(title="UMAP1"),
        yaxis=dict(title="UMAP2"),
    )
    fig.update_xaxes(showline=True, mirror=True, linecolor="black")
    fig.update_yaxes(showline=True, mirror=True, linecolor="black")
    return fig

# tried to use this for caching, doesn't work !
def _compute_orders_and_dendros(
    mat_bytes: bytes,
    shape: tuple,
    row_labels: tuple,
    col_labels: tuple,
    method: str,
    metric: str
):
    """
    mat_bytes : raw bytes of your data matrix, dtype float64
    shape     : (n_rows, n_cols) of that matrix
    """
    n_rows, n_cols = shape
    arr = np.frombuffer(mat_bytes, dtype=np.float64).reshape(n_rows, n_cols)

    # SciPy linkage exactly once
    row_link = sch.linkage(arr,       method=method, metric=metric)
    col_link = sch.linkage(arr.T,     method=method, metric=metric)

    # Produce the two dendrogram objects
    dcol = ff.create_dendrogram(
        arr.T,
        orientation="bottom",
        labels=list(col_labels),       # ← pass your sample names
        linkagefun=lambda _: col_link
    )
    # tag the col‐dendrogram traces so they draw in the right subplot
    for tr in dcol.data:
        tr["yaxis"] = "y2"

    drow = ff.create_dendrogram(
        arr,
        orientation="right",
        labels=list(row_labels),       # ← pass your protein IDs
        linkagefun=lambda _: row_link
    )
    # similarly tag the row‐dendrogram
    for tr in drow.data:
        tr["xaxis"] = "x2"

    return dcol, drow

def plot_cluster_heatmap_plotly(
    data: pd.DataFrame,
    y_labels: Optional[pd.DataFrame] = None,
    cond_series: Optional[pd.Series] = None,
    method: str = "ward",
    metric: str = "euclidean",
    colorscale: str = "vlag",
    width: int = 800,
    height: int = 800,
    title: str = "Hierarchical Clustering Heatmap",
    sample_linkage: np.ndarray = None,
    feature_linkage: np.ndarray = None,
) -> go.Figure:
    """
    Produces a “clustergram” of sample–sample distances:
      - top & left dendrograms via ff.create_dendrogram
      - heatmap = distance matrix (pdist → squareform), reordered
      - no hoverinfo for speed
      - nice axis styling exactly like the Plotly example
    """
    # 1) Clean up: fill any NaNs, but keep all rows
    df = data

    # 2) Column dendrogram (samples)
    #dendro_col = ff.create_dendrogram(
    #    df.values.T,
    #    orientation="bottom",
    #    labels=list(df.columns),
    #    linkagefun=lambda x: sch.linkage(x, method=method, metric=metric)
    #)
    dendro_col = ff.create_dendrogram(
        df.values.T,
        orientation="bottom",
        labels=list(df.columns),
        linkagefun=(
            lambda x: sample_linkage
        ) if sample_linkage is not None else
        (lambda x: sch.linkage(x, method=method, metric=metric))
    )
    for trace in dendro_col.data:
        trace["yaxis"] = "y2"

    # 3) Row dendrogram (proteins)
    #dendro_row = ff.create_dendrogram(
    #    df.values,
    #    orientation="right",
    #    labels=list(df.index),
    #    linkagefun=lambda x: sch.linkage(x, method=method, metric=metric)
    #)
    dendro_row = ff.create_dendrogram(
        df.values,
        orientation="right",
        labels=list(df.index),
        linkagefun=(
            lambda x: feature_linkage
        ) if feature_linkage is not None else
        (lambda x: sch.linkage(x, method=method, metric=metric))
    )

    for trace in dendro_row.data:
        trace["xaxis"] = "x2"
        dendro_col.add_trace(trace)

    fig = dendro_col  # start from the col‐dendrogram figure

    # 4) Extract the leaf order by label
    col_leaves = fig.layout["xaxis"]["ticktext"]
    row_leaves = dendro_row.layout["yaxis"]["ticktext"]
    col_order = [df.columns.get_loc(lbl) for lbl in col_leaves]
    row_order = [df.index.get_loc(lbl)   for lbl in row_leaves]

    # 5) Reorder the DataFrame
    df = df.iloc[row_order, :].iloc[:, col_order]

    # 6) Add the heatmap of the raw (or z-scored) values
    min_val, max_val = np.nanmin(df.values), np.nanmax(df.values)
    abs_max = max(abs(min_val), abs(max_val))
    zmin, zmax = -abs_max, abs_max

    heat = go.Heatmap(
        z=df.values,
        x=fig.layout.xaxis["tickvals"],
        y=dendro_row.layout.yaxis["tickvals"],
        colorscale=colorscale,
        reversescale=True,
        zmid=0,
        zmin=zmin,
        zmax=zmax,
        #hoverinfo="none",
        hoverinfo="text",
        hovertemplate=
            "Protein: %{y}<br>"
            "Sample: %{x}<br>"
            "Value: %{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Value"),
    )
    fig.add_trace(heat)

    # gene names
    if y_labels is not None:
        # get the protein IDs in the clustered order
        row_leaves = dendro_row.layout["yaxis"]["ticktext"]
        # map each protein ID → gene name
        ticktexts = [y_labels[df.index.get_loc(pid)] for pid in row_leaves]
        # overwrite the main y-axis (heatmap) ticks
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=dendro_row.layout["yaxis"]["tickvals"],
                ticktext=ticktexts
            )
        )

    levels  = pd.Categorical(cond_series).categories
    cmap    = get_color_map(levels, px.colors.qualitative.Plotly)
    colours = [cmap[cond_series[s]] for s in df.columns]

    # inject coloured tick labels via simple HTML <span>
    fig.update_xaxes(
        tickvals = fig.layout.xaxis.tickvals,
        ticktext = [
            f"<span style='color:{col};'>{lbl}</span>"
            for lbl,col in zip(df.columns, colours)
        ]
    )

    for lvl in levels:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=cmap[lvl], size=10),
            name=str(lvl),
            showlegend=True,
            hoverinfo="none"
        ))

    for tr in fig.data:
        # only show legend for our dummy condition traces
        if not (tr.type == "scatter" and tr.name in levels):
            tr.showlegend = False

    # position the legend to the right
    fig.update_layout(legend=dict(
        title="Condition",
        orientation="v",
        x=-0.0, y=1,
        xanchor="left", yanchor="top"
    ))

    # 7) Tidy up layout
    fig.update_layout({
        "width": width, "height": height,
        "showlegend": True,
        "hovermode": "closest",
        "title": {"text": title, "x": 0.5},
    })
    fig.update_layout(xaxis={"domain":[0.15,1], "mirror":False,
                             "showgrid":False, "showline":False,
                             "zeroline":False, "ticks":""})
    fig.update_layout(xaxis2={"domain":[0,0.15], "mirror":False,
                              "showgrid":False, "showline":False,
                              "zeroline":False, "showticklabels":False,
                              "ticks":""})
    fig.update_layout(yaxis={"domain":[0,0.85], "mirror":False,
                             "showgrid":False, "showline":False,
                             "zeroline":False, "showticklabels":False,
                             "ticks":""})
    fig.update_layout(yaxis2={"domain":[0.825,0.975], "mirror":False,
                              "showgrid":False, "showline":False,
                              "zeroline":False, "showticklabels":False,
                              "ticks":""})

    return fig

@log_time("Plotting Volcano Plots")
def plot_volcanoes(
    state,
    contrast: str,
    sign_threshold: float = 0.05,
    width: int = 900,
    height: int = 600,
    show_measured: bool = True,
    show_imp_cond1: bool = True,
    show_imp_cond2: bool = True,
    highlight: str = None,
    color_by: str = "SNR",
) -> go.Figure:
    """
    Single‐contrast volcano with separate toggles for:
      - measured in both
      - imputed in condition1
      - imputed in condition2
    """
    adata = state.adata
    genes = np.array(adata.var["GENE_NAMES"])

    # build opacity mask: highlight == full, others faded; no highlight => all full
    genes = np.array(adata.var["GENE_NAMES"])

    if highlight and highlight in genes:
        is_high = (genes == highlight)
        base_opacity = np.where(is_high, 1.0, 0.1)
        # stash index for annotation
        high_idx = np.where(is_high)[0][0]
    else:
        base_opacity = np.ones(len(genes))
        high_idx = None

    # prepare data
    df_fc = pd.DataFrame(
        adata.varm["log2fc"],
        index=adata.var_names,
        columns=adata.uns["contrast_names"]
    )
    df_q  = pd.DataFrame(
        adata.varm.get("q_ebayes", adata.varm["q"]),
        index=adata.var_names,
        columns=adata.uns["contrast_names"]
    )
    x = df_fc[contrast]
    y = -np.log10(df_q[contrast])

    # missingness masks
    miss = pd.DataFrame(adata.uns["missingness"])
    grp1, grp2 = contrast.split("_vs_")
    a = miss[grp1].values >= 1.0
    b = miss[grp2].values >= 1.0
    measured_mask = (~a & ~b)
    imp1_mask     = (a & ~b)
    imp2_mask     = (b & ~a)

    # significance coloring
    sig_up   = (df_q[contrast] < sign_threshold) & (x > 0)
    sig_down = (df_q[contrast] < sign_threshold) & (x < 0)

    # --- prepare color arrays/scales based on color_by ---
    if color_by == "Significance":
        # discrete red/blue/gray
        color_vals = np.where(sig_up, "red",
                      np.where(sig_down, "blue", "gray"))
        colorscale = None
        colorbar   = None

    elif color_by == "Avg Expression":
        # placeholder: average of some expression layer (e.g. lognorm)
        expr_layer = adata.X
        mat = expr_layer.toarray() if hasattr(expr_layer, "toarray") else expr_layer
        idx1 = adata.obs["CONDITION"] == grp1
        idx2 = adata.obs["CONDITION"] == grp2
        mean1 = mat[idx1, :].mean(axis=0)
        mean2 = mat[idx2, :].mean(axis=0)
        avg_expr = pd.Series((mean1 + mean2) / 2, index=adata.var_names)
        color_vals = avg_expr
        colorscale = "thermal"
        colorbar = dict(title="Mean log expr", len=0.5)

    else:
        raise ValueError(f"Unknown color_by mode: {color_by!r}")

    # helper to build a trace
    def add_group_trace(mask, name, symbol):
        trace_kwargs = dict(
            x=x[mask], y=y[mask],
            mode="markers",
            marker=dict(
                symbol=symbol,
                size=6,
                opacity=base_opacity[mask],
            ),
            name=name,
            text=adata.var["GENE_NAMES"][mask],
            hovertemplate="Gene: %{text}<br>log2FC: %{x:.2f}<br>-log10(q): %{y:.2f}<extra></extra>"
        )
        # continuous coloring
        add_colorbar = False
        if color_by != "Significance" and name == "Observed in both":
            add_colorbar = True
        if color_by != "significance":
            trace_kwargs["marker"].update(
                color=color_vals[mask],
                colorscale=colorscale,
                showscale = True if add_colorbar else False,
                colorbar=colorbar if add_colorbar else None,
            )
        else:
            trace_kwargs["marker"]["color"] = color_vals[mask]

        fig.add_trace(go.Scattergl(**trace_kwargs))

    fig = go.Figure()

    if show_measured:   add_group_trace(measured_mask, "Observed in both", "circle")
    if show_imp_cond1:  add_group_trace(imp1_mask, f"Imputed in {grp1}", "triangle-up")
    if show_imp_cond2:  add_group_trace(imp2_mask, f"Imputed in {grp2}", "triangle-down")

    # threshold & axes with padding
    thr_y = -np.log10(sign_threshold)
    all_x = np.concatenate([
        x[measured_mask] if show_measured else np.array([]),
        x[imp1_mask]     if show_imp_cond1 else np.array([]),
        x[imp2_mask]     if show_imp_cond2 else np.array([]),
    ])
    all_y = np.concatenate([
        y[measured_mask] if show_measured else np.array([]),
        y[imp1_mask]     if show_imp_cond1 else np.array([]),
        y[imp2_mask]     if show_imp_cond2 else np.array([]),
    ])
    if all_x.size == 0:
        # no points to show → default safe view
        xmin, xmax = -1.0, 1.0
        ymin, ymax = 0.0, 1.0
    else:
        xmin, xmax = float(all_x.min()), float(all_x.max())
        ymin, ymax = float(all_y.min()), float(all_y.max())

    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05

    # annotations
    up   = int(((x > 0) & (df_q[contrast] < sign_threshold) & np.isin(x, all_x)).sum())
    down = int(((x < 0) & (df_q[contrast] < sign_threshold) & np.isin(x, all_x)).sum())
    rest = int(len(all_x) - up - down)

    # mask to know whether to show annotation
    visible_mask = np.zeros_like(measured_mask)
    if show_measured:
        visible_mask |= measured_mask
    if show_imp_cond1:
        visible_mask |= imp1_mask
    if show_imp_cond2:
        visible_mask |= imp2_mask

    arrow_ann = None
    if high_idx is not None and visible_mask[high_idx]:
        sign = 1 if x[high_idx] >= 0 else -1
        xh = x.values[high_idx]
        yh = y.values[high_idx]
        arrow_ann = dict(
            x=xh+sign*0.05, y=yh+0.05,
            ax=xh+sign*0.5, ay=yh+0.5,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=genes[high_idx],
        )

    annos = [
        dict(x=0.05, y=0.95, xref="paper", yref="paper",
             text=f"<b>{down}</b>", bgcolor="blue", font=dict(color="white"), showarrow=False),
        dict(x=0.4875, y=0.95, xref="paper", yref="paper",
             text=f"<b>{rest}</b>", bgcolor="lightgrey", font=dict(color="black"), showarrow=False),
        dict(x=0.95, y=0.95, xref="paper", yref="paper",
             text=f"<b>{up}</b>", bgcolor="red", font=dict(color="white"), showarrow=False),
    ]
    if arrow_ann:
        annos.append(arrow_ann)

    fig.update_layout(
        margin=dict(
            l=60,
            r=120,
            t=60,
            b=60,
            autoexpand=False   # <- disable legends/colorbars pushing the plot area
        ),
        annotations=annos,
        title=dict(text=f"{contrast}",
                        x=0.5),
        showlegend=False,
        shapes=[
            dict(type="line",
                 x0=xmin-pad_x, x1=xmax+pad_x,
                 y0=thr_y,      y1=thr_y,
                 line=dict(color="black", dash="dash")),
            dict(type="line",
                 x0=0,           x1=0,
                 y0=ymin-pad_y,  y1=ymax+pad_y,
                 line=dict(color="black", dash="dash")),
        ],
        xaxis=dict(title="log2 Fold Change", autorange=True),
        yaxis=dict(title="-log10(q-value)", autorange=True),
        width=width, height=height,
    )

    return fig


def plot_histogram_plotly(
    df: Optional[pd.DataFrame] = None,
    value_col: str = "Intensity",
    group_col: str = "Normalization",
    labels: List[str] = ["Before", "After"],
    colors: List[str] = ["blue", "red"],
    nbins: int = 50,
    stat: str = "probability",           # "count" or "probability"
    log_x: bool = True,                  # whether to log-transform
    log_base: int = 10,                   # 2 or 10
    opacity: Union[float, Sequence[float]] = 0.5,
    x_range: Optional[Tuple[float,float]] = None,
    y_range: Optional[Tuple[float,float]] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    width: int = 900,
    height: int = 500,
    title: Optional[str] = None,
    data: Optional[Dict[str, Sequence]] = None,
) -> go.Figure:
    """
    Generic overlaid histogram for one or more distributions.
    Supports optional log-base-2 or log-base-10 transform with correct tick labels.
    """
    # ------------------------------------------------
    # 1) Prepare raw arrays
    # ------------------------------------------------
    if data is not None:
        labels = list(data.keys())
        values = {lbl: np.asarray(data[lbl]) for lbl in labels}
    else:
        if df is None:
            raise ValueError("Need either 'data' or 'df'")
        df2 = df.copy()
        if log_x:
            df2 = df2[df2[value_col] > 0]
            df2["_val"] = np.log(df2[value_col]) / np.log(log_base)
        else:
            df2["_val"] = df2[value_col]
        values = {
            lbl: df2.loc[df2[group_col] == lbl, "_val"].values
            for lbl in labels
        }

    # ------------------------------------------------
    # 2) Compute bins on the transformed scale
    # ------------------------------------------------
    all_vals = np.concatenate(list(values.values()))
    mn = x_range[0] if (log_x and x_range) else (np.min(all_vals))
    mx = x_range[1] if (log_x and x_range) else (np.max(all_vals))
    bins = np.linspace(mn, mx, nbins + 1)

    # ------------------------------------------------
    # 3) Draw overlaid bars
    # ------------------------------------------------
    fig = go.Figure()
    ops = (list(opacity) if isinstance(opacity, (list,tuple,np.ndarray))
           else [opacity]*len(labels))

    for i, lbl in enumerate(labels):
        arr = values[lbl]
        counts, edges = np.histogram(arr, bins=bins)
        if stat == "probability":
            counts = counts / counts.sum()
        mids   = 0.5*(edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        fig.add_trace(go.Bar(
            x=mids, y=counts, width=widths,
            name=lbl,
            marker_color=colors[i] if i < len(colors) else None,
            opacity=ops[i],
            hoverinfo="skip",
        ))

    # ------------------------------------------------
    # 4) Axis formatting
    # ------------------------------------------------
    if log_x:
        # tick positions are integers on the log-scale
        lo = int(np.floor(bins[0]))
        hi = int(np.ceil (bins[-1]))
        units = list(range(lo, hi+1))
        # labels like 2ⁿ or 10ⁿ
        if log_base == 10:
            ticktext = [f"10<sup>{u}</sup>" for u in units]
            xlabel   = x_title or f"log₁₀({value_col})"
        else:
            ticktext = [f"2<sup>{u}</sup>"  for u in units]
            xlabel   = x_title or f"log₂({value_col})"
        fig.update_xaxes(
            type="linear",
            autorange=False,
            range=[lo, hi],
            tickmode="array",
            tickvals=units,
            ticktext=ticktext,
            showgrid=True,
            title_text=xlabel,
        )
        fig.update_yaxes(
            title_text=y_title or stat.title(),
            showgrid=True,
        )
    else:
        fig.update_xaxes(
            range=x_range,
            title_text=x_title or value_col,
            showgrid=True,
        )
        fig.update_yaxes(
            range=y_range,
            title_text=y_title or stat.title(),
            showgrid=True,
        )

    # ------------------------------------------------
    # 5) Final layout
    # ------------------------------------------------
    fig.update_layout(
        title=dict(text=title or "", x=0.5),
        barmode="overlay",
        template="plotly_white",
        width=width, height=height,
    )

    return fig

def add_violin_traces(
    fig: go.Figure,
    df,
    x: str,
    y: str,
    color: str,
    name_suffix: str,
    row: int,
    col: int,
    showlegend: bool = False,
    opacity: float = 0.7,
    width: float = 0.8,
) -> None:
    """
    Add violin traces to the given (row, col) subplot of a Figure.

    Parameters
    ----------
    fig : go.Figure
        Figure created with make_subplots.
    df : pd.DataFrame
        Long-format DataFrame containing columns [x, y].
    x : str
        Column name for categorical axis.
    y : str
        Column name for numeric values.
    color : str
        Color for the violin outline.
    name_suffix : str
        Legend group identifier (if showlegend=True).
    row : int
        Subplot row index (1-based).
    col : int
        Subplot column index (1-based).
    showlegend : bool
        Whether to add a legend entry for these traces.
    opacity : float
        Fill opacity of the violin.
    width : float
        Width of each violin in category units.
    """
    cats = df[x].unique()
    for cat in cats:
        sub = df[df[x] == cat]
        fig.add_trace(
            go.Violin(
                x=sub[x],
                y=sub[y],
                showlegend=showlegend,
                legendgroup=name_suffix,
                scalegroup=cat,
                line_color=color,
                opacity=opacity,
                width=width,

                # internal quartile+median lines only (transparent box)
                box_visible=True,
                box_fillcolor="rgba(0,0,0,0)",
                box_line_color="black",
                box_line_width=1,
                meanline_visible=False,

                # no outlier markers
                points=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

