"""
Low-level Plotly utilities for ProteoFlux panel app.
"""

import copy
import numpy as np
import pandas as pd
import warnings
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
from typing import List, Sequence, Optional, Dict, Tuple, Union, Literal
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from utils import logger, log_time
from functools import lru_cache

def _abbr(s, head=12, tail=6):
    return s if len(s) <= head+tail+1 else f"{s[:head]}…{s[-tail:]}"

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

def color_ticks_by_condition(fig, samples, cond_series, cmap):
    """
    Color x/y tick labels by condition.
    - samples: list[str] in the same order as fig's axis (corr.columns/index)
    - cond_series: pd.Series indexed by sample name -> condition label
    - cmap: dict {condition -> color}
    """
    colors = [cmap[str(cond_series.loc[s])] for s in samples]
    short = [_abbr(s, head=6) for s in samples]

    tickvals = list(range(len(samples)))
    ticktext = [f"<span style='color:{c}'>{s}</span>" for s, c in zip(short, colors)]

    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=30)
    fig.update_yaxes(tickmode="array", tickvals=tickvals,
                     ticktext=[f"<span style='color:{c}'>{s}</span>" for s, c in zip(short, colors)])
    return fig

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
    sort_by: Literal["condition","sample"] = "sample",   # <-- NEW
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

    # --- NEW: decide x order without touching colors ---
    if sort_by == "condition":
        # group by condition (A,B,C...), then sort samples by name within each
        ordered_samples = []
        for cond in unique_conds:  # alphabetical condition order
            group = [s for s in samples if sample_conditions[s] == cond]
            ordered_samples.extend(sorted(group))
    elif sort_by == "sample":
        ordered_samples = sorted(samples)
    else:
        ordered_samples = samples[:]

    pivot=pivot.reindex(index=ordered_samples)

    fig = go.Figure()

    # 5) actual bar traces (no legend entries)
    for cat in cats:
        for cond in unique_conds:
            xs = [s for s in ordered_samples if sample_conditions[s] == cond]  # <-- use ordered
            if not xs:
                continue
            ys = pivot.loc[xs, cat].tolist()
            fig.add_trace(go.Bar(
                x=xs,
                y=ys,
                marker_color=category_colors[cat],
                marker_line_color=[cond_color_map[cond]] * len(xs),
                marker_line_width=2,
                showlegend=False,                    # controlled by dummy legend items below
                legendgroup=f"COND::{cond}",         # <-- link to condition legend item
                customdata=np.array(samples),  # FULL names
                hovertemplate=(f"Category: {cat.capitalize()}<br>"
                               "Sample: %{customdata}<br>"
                               "Count: %{y}<extra></extra>"),
            ))

    # 6) total annotations
    totals = pivot.sum(axis=1)

    # padding above the bar tops so the label sits just above the stack
    y_pad = max(1.0, 0.02 * float(totals.max()))  # 2% of max height

    # build annotations (one per sample), rotated 45°
    rot_annos = []
    for s in ordered_samples:
        v = float(totals[s])
        rot_annos.append(dict(
            x=s, y=v + y_pad,
            xref="x", yref="y",
            text=f"{int(round(v))}",
            showarrow=False,
            textangle=-30,
            xanchor="center",     # so it leans away from the bar
            yanchor="bottom",
            font=dict(color="black", size=11),
        ))

    # append to any annotations you already added (e.g., the categories row)
    existing = list(fig.layout.annotations) if fig.layout.annotations else []
    existing.extend(rot_annos)
    fig.update_layout(annotations=existing)

    # 7) Horizontal "Categories" legend row (inert) with a boxed background
    cat_labels = ["Complete","Shared","Sparse","Unique"]
    cat_keys   = ["complete","shared","sparse","unique"]

    # background box (legend-like)
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.57, x1=0.97,          # row span (tweak if needed)
        y0=1.14, y1=1.25,          # height of the box above the plot
        line=dict(color="black", width=1),
        fillcolor="white",
        layer="above"
    )

    # lay out swatches + labels horizontally inside the box
    x0 = 0.59                      # start x (inside the box)
    dx = 0.10                      # spacing between items
    y  = 1.19                      # baseline y (centered in the box)
    sw = 0.02                      # swatch width in paper coords
    sh = 0.018                     # swatch half-height

    for i,(lab,key) in enumerate(zip(cat_labels, cat_keys)):
        xi = x0 + i*dx
        # swatch with black border
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=xi, x1=xi+sw,
            y0=y-sh, y1=y+sh,
            line=dict(color="black", width=2),
            fillcolor=category_colors[key],
            layer="above"
        )
        # label
        fig.add_annotation(
            xref="paper", yref="paper",
            x=xi+sw+0.01, y=y,
            text=f"{lab}",
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=12, color="black")
        )

    # 8) Conditions legend
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
            showlegend=True,
            legendgroup=f"COND::{cond}",            # links to bar+annotation traces
            legendgrouptitle_text=("Condition" if j == 0 else None),
            hoverinfo="skip",
        ))

    # 9) Layout: single (main) legend on the right; single‑click only
    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.4),
        xaxis_title="Sample",
        yaxis_title="Number of Proteins",
        template="plotly_white",
        margin=dict(t=150, r=160, l=60, b=60),      # top margin for the horiz row
        legend=dict(
            x=1.15, y=1.19,
            xanchor="right", yanchor="top",
            bordercolor="black", borderwidth=1,
            orientation="v",
            groupclick="togglegroup",               # single-click toggles the group
            itemclick=False,
            itemdoubleclick=False,
        ),
    )

    fig.update_layout(
        meta=dict(
            ordered_samples=ordered_samples,
            sample2cond={s: str(sample_conditions[s]) for s in ordered_samples},
            conditions=unique_conds,
        )
    )
    tickvals = ordered_samples  # categorical axis: use the category names
    ticktext = [
        f"<span style='color:{cond_color_map[sample_conditions[s]]}'>{_abbr(s)}</span>"
        for s in ordered_samples
    ]
    fig.update_xaxes(
        tickangle=30,
        categoryorder="array",
        categoryarray=ordered_samples,
        tickvals=ordered_samples,
        ticktext=ticktext,
    )
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
        title=dict(text=title, x=0.5),
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
    cond_key: str = "CONDITION",
    metric: Literal["CV", "rMAD"] = "CV",
    layer: str = None,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict mapping:
      - "Total" → array of length = n_proteins
      - each condition → same‐length array
    computed **per protein** across samples *without ever transposing*.
    """
    # 1) samples × proteins matrix
    data = adata.X
    if layer is not None:
        data = adata.layers[layer]

    df = pd.DataFrame(
        data,
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
        with warnings.catch_warnings():
            # filterwarnings lets us match on the exact message
            warnings.filterwarnings(
                "ignore",
                message="Mean of empty slice",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Degrees of freedom <= 0 for slice",
                category=RuntimeWarning,
            )
            with np.errstate(invalid='ignore', divide='ignore'):
                linear = 2 ** x
                std    = np.nanstd(linear, axis=0)
                mean   = np.clip(np.nanmean(linear, axis=0), 1e-6, None)
                return 100 * std / mean

    def _rmad(x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            with np.errstate(invalid='ignore', divide='ignore'):
                linear = 2 ** x
                med    = np.nanmedian(linear, axis=0)
                mad    = np.nanmedian(np.abs(linear - med), axis=0)
                return 100 * mad / np.clip(med, 1e-6, None)

    compute_fn = _cv if metric == "CV" else _rmad

    out: Dict[str, np.ndarray] = {}
    # Global (“Total”)
    out["Total"] = compute_fn(df.values)

    # Per‐condition
    #for cond, subdf in df.groupby(conditions, axis=0):
    for cond, subdf in df.groupby(by=conditions, observed=False):
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
        legend=dict(
            title_text=" Condition",
            bordercolor="black",
            borderwidth=1,
            x=1.02, y=1,
            xanchor="left", yanchor="top"
        )
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
        legend=dict(
            title_text=" Condition",
            bordercolor="black",
            borderwidth=1,
            x=1.02, y=1,
            xanchor="left", yanchor="top"
        )
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
    import scipy.cluster.hierarchy as sch
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
    width: int = 1700,
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
    import scipy.cluster.hierarchy as sch
    # 1) Clean up: fill any NaNs, but keep all rows
    df = data

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

    fig = dendro_col

    # Add the heatmap of the raw (or z-scored) values
    min_val, max_val = np.nanmin(df.values), np.nanmax(df.values)

    if (min_val < 0) and (max_val > 0):
        # centered data → symmetric diverging scale
        abs_max = max(abs(min_val), abs(max_val))
        zmin, zmax = -abs_max, abs_max
        zmid = 0
        rev  = True
    else:
        # nonnegative (intensities) -> natural range, no zero centering
        zmin, zmax = float(min_val), float(max_val)
        zmid = None
        rev  = False

    # Build per-cell labels
    # Use gene names if provided, otherwise fall back to df.index
    if y_labels is not None:
        _ylab_map = pd.Series(y_labels, index=df.index)
        row_disp = [_ylab_map.get(r, r) for r in df.index]
    else:
        row_disp = list(df.index)
    col_disp = list(df.columns)
    # customdata shape: (n_rows, n_cols, 2) -> [protein_label, sample_label]
    _cd = np.empty((df.shape[0], df.shape[1], 2), dtype=object)
    for i, r in enumerate(row_disp):
        _cd[i, :, 0] = r
    for j, c in enumerate(col_disp):
        _cd[:, j, 1] = c

    #heat = go.Heatmapgl(
    heat = go.Heatmap(
        z=df.values,
        x=fig.layout.xaxis["tickvals"],
        y=dendro_row.layout.yaxis["tickvals"],
        customdata=_cd,
        colorscale=colorscale,
        reversescale=True,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        hoverinfo="text",
        hovertemplate=(
            "Gene Name: %{customdata[0]}<br>"
            "Sample: %{customdata[1]}<br>"
            "Value: %{z:.2f}<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(title="Value"),
    )
    fig.add_trace(heat)

    levels  = pd.Categorical(cond_series).categories
    cmap    = get_color_map(levels, px.colors.qualitative.Plotly)
    colours = [cmap[cond_series[s]] for s in df.columns]

    short = [_abbr(s) for s in df.columns]
    # inject coloured tick labels via simple HTML <span>
    fig.update_xaxes(
        tickvals = fig.layout.xaxis.tickvals,
        ticktext = [
            f"<span style='color:{col};'>{lbl}</span>"
            #for lbl,col in zip(df.columns, colours)
            for lbl,col in zip(short, colours)
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
        #"width": width, "height": height,
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

@log_time("Plotting Missing Values Heatmap (Plotly)")
def plot_binary_cluster_heatmap_plotly(
    adata: AnnData,
    cond_key: str = "Condition",
    layer: str = "normalized",
    title: str = "Clustered Missing Values Heatmap",
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """
    Interactive Plotly heatmap of binary missingness, reordered by
    precomputed sample/feature linkages in adata.uns, with a color strip
    and legend for conditions.
    """

    # 1) build missingness DataFrame (features × samples)
    missing = np.isnan(adata.layers[layer]).astype(int)  # samples × features
    df = pd.DataFrame(
        missing.T,                                      # features × samples
        index=adata.var_names,
        columns=adata.obs_names
    )

    # 2) reorder according to .uns
    feat_order   = adata.uns['missing_feature_order']
    sample_order = adata.uns['missing_sample_order']
    df = df.reindex(index=feat_order, columns=sample_order)

    # 3) make a condition→color mapping & color-strip
    levels = adata.obs[cond_key].unique().tolist()
    cmap_cond = get_color_map(sorted(levels), palette=None, anchor=None)
    cond_ser  = adata.obs[cond_key].reindex(sample_order)
    x_colors  = [cmap_cond[c] for c in cond_ser]

    # 4) draw the heatmap
    fig = px.imshow(
        df,
        color_continuous_scale=[[0, '#f0f0f0'], [1, '#e34a33']],
        zmin=0, zmax=1,
        aspect="auto",
        labels=dict(x="", y="", color="Missing"),
        width=width, height=height,
    )

    fig.update_traces(showscale=False)
    #fig.update_layout(coloraxis_showscale=False)

    # 5) inject colored tick labels on x-axis
    short = [_abbr(s) for s in sample_order]
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(sample_order))),
        ticktext=[
            f"<span style='color:{col};'>{lbl}</span>"
            for lbl, col in zip(short, x_colors)
        ],
        tickangle=30,
        side="bottom",
    )

    # 6) tidy y-axis
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(feat_order))),
        ticktext=feat_order,
        ticks="",
        showgrid=False,
    )
    fig.update_yaxes(showticklabels=False, showgrid=False)

    # 7) add binary legend (Present vs Missing)
    # Present = 0 = grey, Missing = 1 = red
    for label, color in [("Present", "#f0f0f0"), ("Missing", "#e34a33")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color, size=10),
            name=label,
            showlegend=True,
            hoverinfo="skip",
        ))

    # hide any non‐legend traces (i.e. the heatmap trace)
    for tr in fig.data:
        if tr.type == 'heatmap':
            tr.showlegend = False

    # 8) finalize layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        margin=dict(l=50, r=200, t=50, b=100),
        legend=dict(
            title=cond_key,
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top"
        ),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        hovermode="closest",
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )

    return fig

def plot_volcanoes(
    state,
    contrast: str,
    data_type: str = "default",
    sign_threshold: float = 0.05,
    width: int = 900,
    height: int = 900,
    show_measured: bool = True,
    show_imp_cond1: bool = True,
    show_imp_cond2: bool = True,
    min_nonimp_per_cond: int = 0,
    min_precursors: int = 1,
    highlight: str = None,
    highlight_group: Optional[Sequence[str]] = None,
    color_by: str = "Significance",
) -> go.Figure:
    """
    Volcano plot with phospho-aware hover + highlighting.

    - Hover shows: Site (if available) + Gene + log2FC + -log10(q).
    - Click behavior is preserved: we keep 'text' = gene so the Overview tab
      still writes the gene back into the search box.
    - Highlighting accepts gene, UniProt, or site labels.
    """
    adata = state.adata

    has_cov = bool(adata.uns["has_covariate"])

    # ---- labels & ids (robust to phospho) ----
    genes   = np.array(adata.var.get("GENE_NAMES", pd.Series(index=adata.var_names, dtype=str)).astype(str))
    protids = np.array(adata.var_names, dtype=str)

    proteomics_mode = True if adata.uns["analysis"]["analysis_type"] == "DIA" else False
    if proteomics_mode:
        sites = genes
    else:
        sites = adata.var_names.astype(str).to_numpy()

    # ---- single highlight (accept gene, UniProt, or site) ----
    def _match_one(token: str) -> np.ndarray:
        if not token:
            return np.zeros(len(genes), dtype=bool)
        t = str(token)
        return (genes == t) | (protids == t) | (sites == t)

    is_high = _match_one(highlight)
    high_idx = int(np.flatnonzero(is_high)[0]) if is_high.any() else None

    # ---- cohort highlight (accepts a list of gene/UniProt/site) ----
    if highlight_group:
        group = set(map(str, highlight_group))
        in_group = np.array(
            [(g in group) or (p in group) or (s in group) for g, p, s in zip(genes, protids, sites)],
            dtype=bool
        )
    else:
        in_group = np.zeros(len(genes), dtype=bool)

    # ---- opacity & size style (unchanged logic, just uses new masks) ----
    base_opacity = np.ones(len(genes), dtype=float)
    any_single   = bool(is_high.any())
    any_group    = bool(in_group.any())

    if any_single and not any_group:
        base_opacity = np.where(is_high, 1.0, 0.08)
    elif any_group and not any_single:
        base_opacity = np.where(in_group, 1.0, 0.05)
    elif any_group and any_single:
        tmp = np.where(in_group, 0.2, 0.05)
        base_opacity = np.where(is_high, 1.0, tmp)

    base_size = np.full(len(genes), 6.0, dtype=float)
    if any_single:
        base_size = np.where(is_high, base_size * 1.05, base_size)
    if any_group:
        base_size = np.where(in_group, base_size * 1.05, base_size)

    # ---- data prep (same stats pipeline) ----
    if data_type == "default" or has_cov == False:
        df_fc = pd.DataFrame(
            adata.varm["log2fc"], index=adata.var_names, columns=adata.uns["contrast_names"]
        )
        df_q  = pd.DataFrame(
            adata.varm["q_ebayes"], index=adata.var_names, columns=adata.uns["contrast_names"]
        )
    elif data_type == "phospho":
        df_fc = pd.DataFrame(
            adata.varm["raw_log2fc"], index=adata.var_names, columns=adata.uns["contrast_names"]
        )
        df_q  = pd.DataFrame(
            adata.varm["raw_q_ebayes"], index=adata.var_names, columns=adata.uns["contrast_names"]
        )
    elif data_type == "flowthrough":
        # flowthrough volcano: use FT-adjusted stats (experiment-wide covariate)
        df_fc = pd.DataFrame(
            adata.varm["ft_log2fc"], index=adata.var_names, columns=adata.uns["contrast_names"]
        )
        df_q  = pd.DataFrame(
            adata.varm["ft_q_ebayes"], index=adata.var_names, columns=adata.uns["contrast_names"]
        )


    x = df_fc[contrast]
    y = -np.log10(df_q[contrast])

    miss = pd.DataFrame(adata.uns["missingness"])
    grp1, grp2 = contrast.split("_vs_")
    # number of replicates in each condition (can differ)
    n1 = int((adata.obs["CONDITION"] == grp1).sum())
    n2 = int((adata.obs["CONDITION"] == grp2).sum())

    # fully missing in condition if count == number of replicates
    a = miss[grp1].to_numpy() >= n1   # fully missing in grp1
    b = miss[grp2].to_numpy() >= n2   # fully missing in grp2
    measured_mask = (~a & ~b)
    imp1_mask     = (a & ~b)
    imp2_mask     = (b & ~a)

    # --- NEW: filter by experiment-wide min precursors ---
    # Uses PG.NrOfPrecursorsIdentified (experiment-wide) harmonized as PRECURSORS_EXP.
    # Any feature with PRECURSORS_EXP < min_precursors is hidden from all groups.
    if data_type == "phospho":
        arr = np.nanmax(np.asarray(adata.layers["spectral_counts"], dtype=float), axis=0)
        prec = pd.Series(arr, index=adata.var_names, name="PRECURSORS_EXP")
    else:
        prec = adata.var.get("PRECURSORS_EXP")

    prec = pd.to_numeric(prec, errors="coerce").fillna(0)
    keep = (prec.values >= int(min_precursors))
    measured_mask &= keep
    imp1_mask     &= keep
    imp2_mask     &= keep

    # --- NEW: per-condition non-imputed counts (from raw intensities) ---
    # We treat "measurement" as a non-NaN value in the raw layer.
    # For fully imputed points, group toggles still control visibility regardless of this threshold.
    mat_raw = adata.layers.get("raw", adata.X)
    mat = mat_raw.toarray() if hasattr(mat_raw, "toarray") else mat_raw  # (samples × features)
    idx1 = (adata.obs["CONDITION"] == grp1).to_numpy()
    idx2 = (adata.obs["CONDITION"] == grp2).to_numpy()
    # counts per feature (axis=0 over samples)
    n1 = np.sum(~np.isnan(mat[idx1, :]), axis=0).astype(int)
    n2 = np.sum(~np.isnan(mat[idx2, :]), axis=0).astype(int)
    thr = int(min_nonimp_per_cond or 0)
    # Apply threshold:
    #  - For "Observed in both": require both sides to meet the threshold
    #  - For "Imputed in grp1": require >=thr on grp2 (the measured side)
    #  - For "Imputed in grp2": require >=thr on grp1 (the measured side)
    meets_both = (n1 >= thr) & (n2 >= thr)
    meets_2    = (n2 >= thr)
    meets_1    = (n1 >= thr)
    measured_mask &= meets_both
    imp1_mask     &= meets_2
    imp2_mask     &= meets_1

    sig_up   = (df_q[contrast] < sign_threshold) & (x > 0)
    sig_down = (df_q[contrast] < sign_threshold) & (x < 0)

    colorscale = None
    colorbar   = None
    if color_by == "Significance":
        color_vals = np.where(sig_up, "red", np.where(sig_down, "blue", "gray"))
    elif color_by == "Avg Intensity":
        expr_layer = adata.X
        mat = expr_layer.toarray() if hasattr(expr_layer, "toarray") else expr_layer
        idx1 = adata.obs["CONDITION"] == grp1
        idx2 = adata.obs["CONDITION"] == grp2
        mean1 = mat[idx1, :].mean(axis=0)
        mean2 = mat[idx2, :].mean(axis=0)
        avg_expr = pd.Series((mean1 + mean2) / 2, index=adata.var_names)
        color_vals = avg_expr
        colorscale = "thermal"
        colorbar = dict(title="Avg Int", len=0.5)
    elif color_by == "Avg IBAQ" and "ibaq" in adata.layers:
        ibaq = adata.layers["ibaq"]
        avg_ibaq = np.log10(np.nanmean(ibaq+1, axis=0))
        color_vals = avg_ibaq
        colorbar = dict(title="Avg log(IBAQ)", len=0.5)
    elif color_by == "Raw LogFC":
        raw = adata.varm["raw_log2fc"]
        df_raw = pd.DataFrame(raw, index=adata.var_names, columns=adata.uns["contrast_names"])

        # Align to plotting order
        color_vals = df_raw[contrast].reindex(protids).to_numpy()

        colorscale = "thermal"
        colorbar = dict(title="Raw log₂ FC", len=0.5)
    elif color_by == "Adj. LogFC":
        raw = adata.varm["log2fc"]
        df_raw = pd.DataFrame(raw, index=adata.var_names, columns=adata.uns["contrast_names"])

        # Align to plotting order
        color_vals = df_raw[contrast].reindex(protids).to_numpy()

        colorscale = "thermal"
        colorbar = dict(title="Adj. log₂ FC", len=0.5)

    elif color_by == "FT LogFC":
        raw = adata.varm["ft_log2fc"]
        df_raw = pd.DataFrame(raw, index=adata.var_names, columns=adata.uns["contrast_names"])

        # Align to plotting order
        color_vals = df_raw[contrast].reindex(protids).to_numpy()

        colorscale = "thermal"
        colorbar = dict(title="FT log₂ FC", len=0.5)

    else:
        raise ValueError(f"Unknown color_by mode: {color_by!r}")

    fig = go.Figure()

    use_coloraxis = (color_by != "Significance")
    if use_coloraxis:
        v = np.asarray(color_vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            vmin, vmax, cmid = 0.0, 1.0, None
        elif "LogFC" in color_by:  # center FC scales on 0
            vmax = float(np.nanmax(np.abs(v)))
            vmin, vmax, cmid = -vmax, vmax, 0.0
        else:
            vmin, vmax, cmid = float(np.nanmin(v)), float(np.nanmax(v)), None

        # Define coloraxis in layout BEFORE traces so the colorbar space is reserved immediately
        fig.update_layout(
            coloraxis=dict(
                colorscale=colorscale,
                colorbar=(colorbar or dict(title=color_by, len=0.5)),
                cmin=vmin, cmax=vmax, cmid=cmid,
                showscale=True,
            ),
            coloraxis_showscale=True,
        )

    def add_group_trace(mask, name, symbol):
        # NOTE: keep 'text' = gene for click → search to stay identical
        trace_kwargs = dict(
            x=x[mask],
            y=y[mask],
            mode="markers",
            marker=dict(
                symbol=symbol,
                size=base_size[mask],
                opacity=base_opacity[mask],
                line=dict(width=0),
            ),
            name=name,
            text=(genes[mask] if proteomics_mode else sites[mask]),
            customdata=(None if proteomics_mode else np.c_[sites[mask], genes[mask], protids[mask]]),
            hovertemplate=(
                ("Gene: %{text}<br>"
                 "log2FC: %{x:.2f}<br>"
                 "-log10(q): %{y:.2f}<extra></extra>")
                if proteomics_mode else
                ("Phosphosite: %{customdata[0]}<br>"
                 "Gene: %{customdata[1]}<br>"
                 "log2FC: %{x:.2f}<br>"
                 "-log10(q): %{y:.2f}<extra></extra>")
            ),
        )
        #add_colorbar = (color_by != "Significance" and name == "Observed in both")
        add_colorbar = (color_by != "Significance")
        if color_by != "Significance":
            trace_kwargs["marker"].update(
                color=color_vals[mask],
                colorscale=colorscale,
                coloraxis="coloraxis",
                #showscale=True if add_colorbar else False,
                #colorbar=colorbar if add_colorbar else None,
            )
        else:
            trace_kwargs["marker"]["color"] = color_vals[mask]
        fig.add_trace(go.Scattergl(**trace_kwargs))

    if show_measured:   add_group_trace(measured_mask, "Observed in both", "circle")
    if show_imp_cond1:  add_group_trace(imp1_mask, f"Imputed in {grp1}", "triangle-up")
    if show_imp_cond2:  add_group_trace(imp2_mask, f"Imputed in {grp2}", "triangle-down")

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
        xmin, xmax = -1.0, 1.0
        ymin, ymax = 0.0, 1.0
    else:
        xmin, xmax = float(all_x.min()), float(all_x.max())
        ymin, ymax = float(all_y.min()), float(all_y.max())
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05

    visible_mask = np.zeros_like(measured_mask)
    if show_measured:  visible_mask |= measured_mask
    if show_imp_cond1: visible_mask |= imp1_mask
    if show_imp_cond2: visible_mask |= imp2_mask

    arrow_ann = None
    #print(visible_mask[high_idx])
    #print(show_measured, show_imp_cond1, show_imp_cond2)
    if (high_idx is not None) and visible_mask[high_idx]:
        sign = 1 if x.iloc[high_idx] >= 0 else -1
        xh = float(x.iloc[high_idx]); yh = float(y.iloc[high_idx])
        arrow_ann = dict(
            x=xh+sign*0.05, y=yh+0.05,
            ax=xh+sign*0.5, ay=yh+0.5,
            xref="x", yref="y", axref="x", ayref="y",
            text=str(genes[high_idx]),  # keep gene tag for consistency
        )

    up   = int(((x > 0) & (df_q[contrast] < sign_threshold) & np.isin(x, all_x)).sum())
    down = int(((x < 0) & (df_q[contrast] < sign_threshold) & np.isin(x, all_x)).sum())
    rest = int(len(all_x) - up - down)

    annos = [
        dict(x=0.02,  y=0.98, xref="paper", yref="paper", opacity=0.7,
             text=f"<b>{down}</b>", bgcolor="blue",  font=dict(color="white"), showarrow=False),
        dict(x=0.500, y=0.98, xref="paper", yref="paper",
             text=f"<b>{rest}</b>", bgcolor="lightgrey", font=dict(color="black"), showarrow=False),
        dict(x=0.98,  y=0.98, xref="paper", yref="paper", opacity=0.7,
             text=f"<b>{up}</b>", bgcolor="red",   font=dict(color="white"), showarrow=False),
    ]
    if arrow_ann: annos.append(arrow_ann)

    fig.update_layout(
        margin=dict(l=60, r=120, t=60, b=60, autoexpand=False),
        annotations=annos,
        title=dict(text=f"{contrast}", x=0.5),
        showlegend=False,
        shapes=[
            dict(type="line", x0=xmin-pad_x, x1=xmax+pad_x, y0=thr_y, y1=thr_y, line=dict(color="black", dash="dash")),
            dict(type="line", x0=0, x1=0, y0=ymin-pad_y, y1=ymax+pad_y, line=dict(color="black", dash="dash")),
        ],
        xaxis=dict(title="log2 Fold Change", autorange=True),
        yaxis=dict(title="-log10(q-value)",  autorange=True),
        height=height,
        #width=width,
    )
    return fig

def plot_histogram_plotly(
    data: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str],
    nbins: int = 50,
    stat: str = "probability",
    log_base: int = 10,
    opacity: float = 0.5,
    width: int = 900,
    height: int = 500,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Overlaid log-scaled histogram of multiple distributions.

    Parameters
    ----------
    data : list of sequences
        Each sequence of raw intensity values (must be positive).
    labels : list of str
        Labels corresponding to each distribution.
    colors : list of str
        Colors for each label.
    nbins : int
        Number of histogram bins.
    stat : str
        "count" or "probability".
    log_base : int
        Base for log-transform (2 or 10).
    opacity : float
        Bar opacity.
    width, height : int
        Figure dimensions.
    title : str
        Plot title.
    """
    # 1) Log-transform and filter
    transformed: Dict[str, np.ndarray] = {}
    for lbl, seq in zip(labels, data):
        arr = np.asarray(seq)
        arr = arr[arr > 0]
        transformed[lbl] = np.log(arr) / np.log(log_base)

    # 2) Compute shared bins
    all_vals = np.concatenate(list(transformed.values()))
    bins = np.linspace(all_vals.min(), all_vals.max(), nbins + 1)

    # 3) Build figure
    fig = go.Figure()
    for lbl, color in zip(labels, colors):
        arr = transformed[lbl]
        counts, edges = np.histogram(arr, bins=bins)
        if stat == "probability":
            counts = counts / counts.sum()
        mids = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]
        fig.add_trace(go.Bar(
            x=mids,
            y=counts,
            width=widths,
            name=lbl,
            marker_color=color,
            opacity=opacity,
            hoverinfo="skip",
        ))

    # 4) Format log-x axis
    lo, hi = int(np.floor(bins[0])), int(np.ceil(bins[-1]))
    ticks = list(range(lo, hi + 1))
    ticktext = [f"10<sup>{t}</sup>" for t in ticks]
    fig.update_xaxes(
        type="linear",
        autorange=False,
        range=[0, hi],
        tickmode="array",
        tickvals=ticks,
        ticktext=ticktext,
        showgrid=True,
        title_text=f"log₁₀(Value)",
    )
    fig.update_yaxes(
        title_text=stat.title(),
        showgrid=True,
    )

    # 5) Final layout
    fig.update_layout(
        title=dict(text=title or "", x=0.5),
        barmode="overlay",
        template="plotly_white",
        width=width,
        height=height,
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
                name="",
            ),
            row=row,
            col=col,
        )
