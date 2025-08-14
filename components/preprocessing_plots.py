import warnings
import numpy as np
import pandas as pd
import panel as pn
from anndata import AnnData
from typing import Tuple, List, Optional, Dict
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from components.plot_utils import plot_histogram_plotly, add_violin_traces, compute_metric_by_condition, get_color_map, plot_bar_plotly, plot_binary_cluster_heatmap_plotly, color_ticks_by_condition
from regression_normalization import compute_reference_values, compute_MA, fit_regression
from utils import logger, log_time

def prepare_long_df(
    df: pd.DataFrame,
    label: str,
    label_col: str,
    condition_mapping: pd.DataFrame = None,
    sample_col: str = "Sample",
    intensity_col: str = "Intensity",
) -> pd.DataFrame:
    """ Convert a wide-format DataFrame (samples as columns) to long-format, filter out non-positive intensities, optionally merge with a condition mapping, and add a processing label.
    Args:
        df (pd.DataFrame): Wide-format DataFrame.
        label (str): The label to assign (e.g., "Imputed" or "Before/After").
        label_col (str): The column name for the label (e.g., "Imputation" or "Normalization").
        condition_mapping (pd.DataFrame, optional): Mapping to merge on 'Sample'.

    Returns:
        pd.DataFrame: Long-format DataFrame with 'Sample', 'Intensity', and the label_col.
    """
    df_long = df.melt(var_name=sample_col, value_name=intensity_col)
    df_long = df_long[df_long[intensity_col] > 0].copy()
    if condition_mapping is not None:
        df_long = df_long.merge(condition_mapping, on=sample_col, how="left")
    df_long[label_col] = label
    return df_long

@log_time("Converting to long df")
@pn.cache
def get_intensity_long_df(
    adata: AnnData,
    before_key: str = "lognorm",
    after_key:  str = "normalized",
    cond_map:    pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build one long-format DataFrame combining Before/After intensities from an AnnData.layers.

    Columns: [Sample, Intensity, Normalization, Condition].

    - adata: AnnData object with layers containing numeric matrices, and obs containing a 'CONDITION' column.
    - before_key/after_key: names of layers in adata.layers (e.g. 'raw', 'lognorm').
    - cond_map: optional pandas DataFrame with columns ['Sample', 'Condition'].
                If None, pulled from adata.obs.
    """
    # 1) prepare condition mapping
    if cond_map is None:
        cond_map = (
            adata.obs
                 .reset_index()
                 .rename(columns={adata.obs.index.name or 'index': 'Sample', 'CONDITION': 'Condition'})
                 .loc[:, ['Sample', 'Condition']]
        )

    # Helper to melt and tag
    def _prep(key: str, label: str) -> pd.DataFrame:
        # construct wide DataFrame: features × samples, so columns=sample names
        mat = adata.layers[key]
        df_wide = pd.DataFrame(
            mat.T,
            index=adata.var_names,
            columns=adata.obs_names
        )
        # reuse existing prepare_long_df function
        return prepare_long_df(
            df=df_wide,
            label=label,
            label_col="Normalization",
            condition_mapping=cond_map
        )

    # 2) melt & tag Before
    df_before = _prep(before_key, label="Before")
    # 3) melt & tag After
    df_after  = _prep(after_key,  label="After")

    # 4) concatenate and drop any NA
    return pd.concat([df_before, df_after], ignore_index=True).dropna()

@log_time("Plotting violins before/after norm")
def plot_violin_by_group_go(
    df_long:       pd.DataFrame,
    group_col:     str,
    colors:        Tuple[str, str] = ("blue","red"),
    subplot_titles:Tuple[str, str] = ("Before","After"),
    title:         Optional[str] = None,
    width:         int = 900,
    height:        int = 500,
    opacity:       float = 0.7,
    key: str = "Intensity",
):
    """
    Generic side-by-side violin plot.

    - df_long: output of get_intensity_long_df()
    - group_col: which column to group on ("Condition" or "Sample")
    - colors: (before_color, after_color)
    - subplot_titles: titles for the two panels
    - title: overall figure title
    """
    df_b = df_long[df_long["Normalization"]=="Before"]
    df_a = df_long[df_long["Normalization"]=="After"]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=False,
        subplot_titles=list(subplot_titles),
        horizontal_spacing=0.08,
    )

    # add violins
    add_violin_traces(fig, df_b, x=group_col, y=key,
                      color=colors[0], name_suffix="Before",
                      row=1, col=1, showlegend=False, opacity=opacity)
    add_violin_traces(fig, df_a, x=group_col, y=key,
                      color=colors[1], name_suffix="After",
                      row=1, col=2, showlegend=False, opacity=opacity)

    # draw median line across each panel
    for col_id, df_sub in [(1, df_b), (2, df_a)]:
        med = df_sub[key].median()
        fig.add_hline(
            y=med,
            row=1, col=col_id,
            line=dict(color="gray", dash="dash", width=1),
            layer="below"
        )

    # add vertical grid‐lines at each category center
    cats = df_b[group_col].unique()
    for i, _ in enumerate(cats):
        for xref in ("x1", "x2"):
            fig.add_shape(
                type="line",
                xref=xref, yref="paper",
                x0=i, x1=i,
                y0=0, y1=1,
                line=dict(color="gray", dash="dot", width=1),
                layer="above",
            )

    # finalize layout
    fig.update_layout(
        title=dict(text=title, x=0.5) if title else {},
        width=width,
        height=height,
        template="plotly_white",
        violinmode="group",
        showlegend=False
    )
    # draw boxed axes
    for rid, cid in [(1,1),(1,2)]:
        fig.update_xaxes(showline=True, mirror=True, linecolor="black",
                         row=rid, col=cid, title_text=group_col)
        fig.update_yaxes(showline=True, mirror=True, linecolor="black",
                         row=rid, col=cid, title_text=key)
    fig.update_yaxes(row=1, col=2, side="right")

    return fig

@log_time("Plotting filtered data histograms")
def plot_filter_histograms(adata: AnnData) -> Dict[str, go.Figure]:
    """
    Return three separate histograms (Q-value, PEP, run evidence count)
    of everything *before* filtering, each with its own threshold line.
    """
    flt = adata.uns["preprocessing"]["filtering"]

    # Define each metric in one place:
    metrics = [
        {
            "key":       "qvalue",
            "label":     "Q-value",
            "threshold": flt["qvalue"]["threshold"],
            "removed":   flt["qvalue"]["number_dropped"],
            "kept":      flt["qvalue"]["number_kept"],
            "values":    np.asarray(flt["qvalue"]["raw_values"]),
        },
        {
            "key":       "pep",
            "label":     "PEP",
            "threshold": flt["pep"]["threshold"],
            "removed":   flt["pep"]["number_dropped"],
            "kept":      flt["pep"]["number_kept"],
            "values":    np.asarray(flt["pep"]["raw_values"]),
        },
        {
            "key":       "run_evidence_count",
            "label":     "Run Evidence Count",
            "threshold": flt["rec"]["threshold"],
            "removed":   flt["rec"]["number_dropped"],
            "kept":      flt["rec"]["number_kept"],
            "values":    np.asarray(flt["rec"]["raw_values"]),
        },
    ]

    # thousands separators
    fmt = lambda n: f"{n:,}".replace(",", "'")

    figs: Dict[str, go.Figure] = {}
    for m in metrics:
        vals = m["values"][np.isfinite(m["values"])]
        kept, removed = m["kept"], m["removed"]

        # Two‐line title: metric name on line 1, details on line 2
        title_md = (
            f"{m['label']}<br>"
            f"Cutoff: {m['threshold']}, kept: {fmt(kept)}, dropped: {fmt(removed)}"
        )

        fig = go.Figure(go.Histogram(
            x=vals,
            nbinsx=80,
            marker_line_color="black",
            marker_line_width=1,
            showlegend=False,
            hoverinfo="none",
        ))
        fig.add_vline(
            x=m["threshold"],
            line=dict(color="red", dash="dash"),
        )
        fig.update_layout(
            title=dict(text=title_md, x=0.5),
            xaxis_title=m["label"],
            yaxis_title="Count",
            yaxis_type="log",
            template="plotly_white",
            width=600,
            height=400,
        )
        fig.update_yaxes(
            dtick=1,
            exponentformat="power",
            showexponent="all",
        )

        figs[m["key"]] = fig

    return figs

@log_time("Plotting dynamic range")
def plot_dynamic_range(
    adata: AnnData,
) -> go.Figure:
    """
    One point per protein: aggregated intensity (sum/mean) across all samples,
    sorted descending, plotted on log10 scale vs. rank.
    """
    mat = adata.layers['raw']
    vals = np.asarray(mat.mean(axis=0)).ravel()

    # 2) filter out zero / non-finite, take log10
    mask = np.isfinite(vals) & (vals > 0)
    vals = vals[mask]
    logvals = np.log10(vals)

    # 3) sort descending
    order = np.argsort(-logvals)
    logvals = logvals[order]

    # optional protein names for hover
    prots = adata.var_names.values
    prots = prots[mask][order]
    gene_names = adata.var["GENE_NAMES"].values[mask][order]

    # 4) build scatter
    fig = go.Figure(go.Scatter(
        x=np.arange(1, len(logvals) + 1),
        y=logvals,
        mode="markers",
        marker=dict(size=6, opacity=0.6),
        hovertemplate=(
            "Rank %{x}<br>"
            "Protein: %{customdata[0]}<br>"
            "Gene: %{customdata[1]}<br>"
            "log₁₀(abundance): %{y:.2f}<extra></extra>"
        ),
        customdata=np.stack([prots, gene_names], axis=-1),
    ))

    fig.update_layout(
        title=dict(text=f"Dynamic Range across all samples", x=0.5),
        xaxis_title="Protein rank",
        yaxis_title="log₁₀(abundance)",
        template="plotly_white",
        width=800,
        height=500,
    )
    return fig

@log_time("Plotting histogram before/after log")
def plot_intensities_histogram(adata: AnnData) -> go.Figure:
    """
    Plot overlaid Before/After intensity histograms directly from adata.layers.
    """
    # 1) pull and filter
    raw_vals     = adata.layers["raw"].ravel()
    lognorm_vals = adata.layers["lognorm"].ravel()
    # only positives
    raw_vals     = raw_vals[raw_vals     > 0]
    lognorm_vals = lognorm_vals[lognorm_vals > 0]

    # 2) plot
    return plot_histogram_plotly(
        data=[raw_vals, lognorm_vals],
        labels=["Before","After"],
        colors=["blue","red"],
        nbins=50,
        stat="probability",
        #log_x=True,
        opacity=0.5,
        width=900,
        height=500,
        title="Distribution of intensities"
    )

@log_time("Plotting violins before/after by condition")
def plot_violin_intensity_by_condition(adata) -> go.Figure:
    """
    Side-by-side violins grouped by Condition.
    """
    df = get_intensity_long_df(adata, cond_map=None)
    return plot_violin_by_group_go(
        df_long=df,
        #key="CV",
        group_col="Condition",
        colors=("blue","red"),
        subplot_titles=("Before Normalization","After Normalization"),
        title="Distribution by Condition",
    )


@log_time("Plotting violins before/after by sample")
def plot_violin_intensity_by_sample(adata) -> go.Figure:
    """
    Side-by-side violins grouped by Sample.
    """
    df = get_intensity_long_df(adata, cond_map=None)
    return plot_violin_by_group_go(
        df_long=df,
        group_col="Sample",
        colors=("blue","red"),
        subplot_titles=("Before Normalization","After Normalization"),
        title="Distribution by Sample",
        #width=max(1500, len(im.columns)*20),
        height=600,
    )

def compute_before_after_metrics(
    adata,
    metric: str,
    cond_key: str = "CONDITION",
    before_layer: str = "lognorm",
    after_layer: str = "normalized"
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Returns two dicts mapping:
      - "Total" → array of length = n_proteins
      - each condition → same-length array
    for both the 'Before' (raw) and 'After' (normalized) matrices,
    in one pass each.
    """
    # 1) pick the same metric functions as your original compute_metric_by_condition :contentReference[oaicite:2]{index=2}
    def _cv(x: np.ndarray) -> np.ndarray:
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

    # 2) pull out the raw arrays
    raw_arr  = adata.layers[before_layer]
    norm_arr = adata.layers[after_layer]

    # 3) compute global (“Total”) metrics once each
    before_total = compute_fn(raw_arr)
    after_total  = compute_fn(norm_arr)

    # 4) group‐by-condition
    conds = adata.obs[cond_key].values
    unique_conds = sorted(np.unique(conds).tolist())

    # preserve insertion order: Total → sorted(conds)
    before_metrics = {"Total": before_total}
    after_metrics  = {"Total": after_total}

    for cond in unique_conds:
        mask = (conds == cond)
        before_metrics[cond] = compute_fn(raw_arr[mask, :])
        after_metrics[cond]  = compute_fn(norm_arr[mask, :])

    return before_metrics, after_metrics


def plot_grouped_violin_metric_by_condition(
    adata,
    metric: str,
    colors=('blue', 'red'),
    title=None,
    width=900,
    height=600,
) -> go.Figure:
    # 1) get both dicts in one pass
    before_metrics, after_metrics = compute_before_after_metrics(
        adata, metric=metric
    )

    # 2) preserve the same x-ordering as before: keys in insertion order
    conditions = list(before_metrics.keys())  # ["Total", *sorted(conds)]
    idx_map    = {cond: i for i, cond in enumerate(conditions)}
    offset     = 0.2

    # 3) build the two violin traces (Before vs After)
    fig = go.Figure()
    for norm_label, metrics_dict, color in zip(
        ["Before", "After"],
        [before_metrics, after_metrics],
        colors
    ):
        x_vals = []
        y_vals = []
        shift  = -offset if norm_label == "Before" else +offset

        for cond in conditions:
            arr = metrics_dict[cond]
            x0  = idx_map[cond] + shift
            x_vals.extend([x0] * arr.size)
            y_vals.extend(arr.tolist())

        fig.add_trace(go.Violin(
            x=x_vals,
            y=y_vals,
            name=norm_label,
            legendgroup=norm_label,
            line_color=color,
            box_visible=True,
            box_line_color="black",
            box_line_width=1,
            opacity=0.6,
            width=offset * 1.8,
            meanline_visible=True,
            points=False,
            spanmode="hard",
        ))

    # 4) add the same vertical separators
    for i in idx_map.values():
        fig.add_vline(
            x=i,
            line=dict(color="gray", dash="dot", width=1),
            layer="above"
        )

    # 5) finalize layout exactly as before
    title_text = title or f"{metric} per Condition"
    fig.update_layout(
        template="plotly_white",
        violinmode="overlay",
        title=dict(text=title_text, x=0.5),
        width=width,
        height=height,
        xaxis=dict(
            title="Condition",
            tickmode="array",
            tickvals=list(idx_map.values()),
            ticktext=conditions,
        ),
        yaxis=dict(title=metric, showgrid=True),
        showlegend=True,
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    return fig

@log_time("Plotting violins cv by condition")
def plot_cv_by_condition(adata: AnnData) -> go.Figure:
    """
    Simplified: compute %CV per protein per Condition, then call the existing
    violin helper. No more aggregate_metrics or record‑loops.
    """
    return plot_grouped_violin_metric_by_condition(
        adata=adata,
        metric="CV",
        colors=("blue", "red"),
        title="Coefficient of Variation (CV) per Condition",
        width=900,
        height=600,
    )


@log_time("Plotting MA plots")
def plot_ma(
    adata: AnnData,
    sample: str,
    scale: Optional[str] = "global",
    span: float = 0.9
) -> Tuple[go.Figure, go.Figure]:
    """
    Returns (fig_before, fig_after) MA‐plots for a single sample:
     - before = lognorm layer
     - after  = normalized layer
    Performs a LOESS fit on (M,A) and overlays it.
    """
    # 1) extract matrices
    raw = adata.layers.get("lognorm", adata.X).T
    norm = adata.layers.get("normalized").T

    raw_mat  = raw.A if hasattr(raw, "A") else raw
    norm_mat = norm.A if hasattr(norm, "A") else norm

    # 2) build reference from raw
    conds = adata.obs["CONDITION"].tolist()
    ref_mat = compute_reference_values(raw_mat, scale, conds)

    # 3) find sample index
    samples = list(adata.obs_names)
    if sample not in samples:
        raise ValueError(f"Sample {sample!r} not found in adata.obs_names")
    j = samples.index(sample)

    # 4) compute (M,A)
    M_before, A_before = compute_MA(raw_mat[:, j],  ref_mat[:, j])
    M_after,  A_after  = compute_MA(norm_mat[:, j], ref_mat[:, j])

    # 5) fit LOESS to each
    _, preds_before = fit_regression(M_before, A_before, "loess", span)
    _, preds_after  = fit_regression(M_after,  A_after,  "loess", span)

    def _make_fig(M, A, preds, title, color):
        # density‐based coloring if lots of points
        mask = np.isfinite(A) & np.isfinite(M)
        Mg, Ag = M[mask], A[mask]

        density = None
        if Mg.size > 2000 and Ag.std()>1e-6 and Mg.std()>1e-6:
            xy      = np.vstack([Mg, Ag])
            density = gaussian_kde(xy)(xy)

        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=Mg, y=Ag,
            mode="markers",
            marker=dict(
                color=density if density is not None else color,
                colorscale="Viridis" if density is not None else None,
                showscale=(density is not None),
                opacity=0.3,
                size=5,
                line_width=0
            ),
            name="Data",
            hoverinfo="skip"
        ))

        # LOESS fit line
        fit_x = Mg
        fit_y = preds[mask]
        order = np.argsort(fit_x)
        fig.add_trace(go.Scatter(
            x=fit_x[order], y=fit_y[order],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="LOESS fit",
            hoverinfo="skip"
        ))

        # horizontal zero line
        fig.add_hline(y=0, line=dict(color="gray", dash="dot"))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="M (mean intensity)",
            yaxis_title="A (log‐ratio)",
            template="plotly_white",
            width=600, height=500,
            showlegend=False
        )
        return fig

    fig_before = _make_fig(
        M_before, A_before, preds_before,
        title=f"MA before normalization<br><sub>{sample}</sub>",
        color="blue"
    )
    fig_after = _make_fig(
        M_after, A_after, preds_after,
        title=f"MA after normalization<br><sub>{sample}</sub>",
        color="red"
    )
    return fig_before, fig_after


@log_time("Plotting violins rmad by condition")
def plot_rmad_by_condition(adata: AnnData) -> go.Figure:
    """
    Simplified: compute %CV per protein per Condition, then call the existing
    violin helper. No more aggregate_metrics or record‑loops.
    """
    return plot_grouped_violin_metric_by_condition(
        adata=adata,
        metric="rMAD",
        colors=("blue", "red"),
        title="Relative Median Absolute Deviation (rMAD) per Condition",
        width=900,
        height=600,
    )
@log_time("Plotting Missing Values barplots")
def plot_mv_barplots(adata: AnnData):
    # 1) Get the raw array and sample/condition metadata
    arr = adata.layers['normalized']
    sample_names = adata.obs_names.tolist()
    conds = adata.obs['CONDITION'].values

    # 2) Compute missing per sample
    mask = np.isnan(arr)
    mvi_per_sample = mask.sum(axis=1).astype(int)

    # 3) Compute missing per condition
    unique_conds, inv = np.unique(conds, return_inverse=True)  # already ascending
    mvi_per_condition_vals = np.bincount(inv, weights=mvi_per_sample).astype(int)
    cond_levels = unique_conds
    mvi_desc = mvi_per_condition_vals

    # 4) Totals & rate
    total_imputed     = int(mvi_per_condition_vals.sum())
    total_entries     = arr.size
    total_non_imputed = total_entries - total_imputed
    rate = total_imputed / total_entries

    # 5) Assemble x/y for the condition barplot
    cond_labels = ["Total (Non-Missing)", "Total (Missing)"] + cond_levels.tolist()
    cond_y      = [total_non_imputed, total_imputed] + mvi_desc.tolist()

    # 6) Stable color map based on **sorted(conditions)** (not display order)
    palette = px.colors.qualitative.Plotly

    levels_sorted = unique_conds.tolist()       # np.unique already sorted ascending
    base_map = {lvl: palette[i % len(palette)] for i, lvl in enumerate(levels_sorted)}

    cond_color_map = {
        "Total (Non-Missing)": "#888888",
        "Total (Missing)"    : "#CCCCCC",
        **base_map,                                   # ← colors now consistent app-wide
    }

    # 7) Plot missing-by-condition (log-y)
    fig_cond = plot_bar_plotly(
        x       = cond_labels,
        y       = cond_y,
        colors  = [cond_color_map[c] for c in cond_labels],
        title   = f"Missing Values (ratio = {100*rate:.2f}%)",
        x_title = "Condition",
        y_title = "Count",
        width   = 800,
        height  = 400,
    )
    fig_cond.update_yaxes(type="log", dtick=1, exponentformat="power")

    # 8) Plot missing-by-sample
    sample_colors = [cond_color_map[c] for c in conds]
    fig_samp = go.Figure()

    # keep a stable x order
    order = sample_names
    conds_arr = np.asarray(conds)

    # use your stable palette mapping
    levels_sorted = sorted(set(conds_arr.astype(str)))
    for cond in levels_sorted:
        m = (conds_arr == cond)
        if not np.any(m):
            continue
        fig_samp.add_trace(go.Bar(
            x=np.array(sample_names)[m],
            y=mvi_per_sample[m],
            name=cond,
            marker=dict(color=cond_color_map[cond]),
            hovertemplate="Condition: " + cond + "<br>Sample: %{x}<br>Missing: %{y}<extra></extra>",
            showlegend=True,
        ))

    fig_samp.update_layout(
        title=dict(text="Missing Values per Sample", x=0.5),
        xaxis_title="Sample",
        yaxis_title="Count",
        width=1200,
        height=400,
        barmode="group",  # bars won’t overlap since x’s differ, but keeps spacing tidy
        legend=dict(
            title=" Conditions",
            orientation="h",
            x=0.7, xanchor="left",
            y=1.05, yanchor="bottom",
            bordercolor="black", borderwidth=1,
        ),
        # allow interactive legend toggling
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
    )

    # preserve sample order on the categorical axis
    fig_samp.update_xaxes(categoryorder="array", categoryarray=order)

    return fig_cond, fig_samp
@log_time("Plotting Missing Values barplots")
def plot_mv_barplots_old(adata: AnnData):
    # 1) Get the raw array and sample/condition metadata
    arr = adata.layers['normalized']           # shape: (n_samples × n_proteins)
    sample_names = adata.obs_names.tolist()    # same order as arr’s rows
    conds = adata.obs['CONDITION'].values      # array of length n_samples

    # 2) Compute missing per sample
    mask = np.isnan(arr)
    mvi_per_sample = mask.sum(axis=1).astype(int)   # 1D int array

    # 3) Compute missing per condition
    #    a) find unique conditions (sorted ascending) + inverse mapping
    unique_conds, inv = np.unique(conds, return_inverse=True)
    #    b) sum up sample‐wise missing counts into bins
    mvi_per_condition_vals = np.bincount(inv, weights=mvi_per_sample).astype(int)
    #    c) to match your old `cond_levels = .index.sort_values(ascending=False)`
    cond_levels = unique_conds[::-1]               # descending alphabetical
    mvi_desc = mvi_per_condition_vals[::-1]        # align with cond_levels

    # 4) Totals & rate
    total_imputed     = int(mvi_per_condition_vals.sum())
    total_entries     = arr.size
    total_non_imputed = total_entries - total_imputed
    rate = total_imputed / total_entries

    # 5) Assemble x/y for the **condition** barplot
    cond_labels = ["Total (Non-Missing)", "Total (Missing)"] + cond_levels.tolist()
    cond_y      = [total_non_imputed, total_imputed] + mvi_desc.tolist()

    # 6) Build a stable color map exactly as before
    palette = px.colors.qualitative.Plotly
    cond_color_map = {
        "Total (Non-Missing)": "#888888",
        "Total (Missing)"    : "#CCCCCC",
    }
    for i, lbl in enumerate(cond_levels):
        cond_color_map[lbl] = palette[i % len(palette)]

    # 7) Plot missing‐by‐condition (log‐y)
    fig_cond = plot_bar_plotly(
        x       = cond_labels,
        y       = cond_y,
        colors  = [cond_color_map[c] for c in cond_labels],
        title   = f"Missing Values (ratio = {100*rate:.2f}%)",
        x_title = "Condition",
        y_title = "Count",
        width   = 800,
        height  = 400,
    )
    fig_cond.update_yaxes(type="log", dtick=1, exponentformat="power")

    # 8) Plot missing‐by‐sample
    sample_colors = [cond_color_map[c] for c in conds]
    fig_samp = plot_bar_plotly(
        x       = sample_names,
        y       = mvi_per_sample.tolist(),
        colors  = sample_colors,
        title   = "Missing Values per Sample",
        x_title = "Sample",
        y_title = "Count",
        width   = 1200,
        height  = 400,
    )
    # hide the main trace from the legend & attach condition info for hover
    bar = fig_samp.data[0]
    bar.showlegend = False
    bar.customdata = conds.tolist()
    bar.hovertemplate = (
        "Condition: %{customdata}<br>"
        "Sample: %{x}<br>"
        "Missing: %{y}<extra></extra>"
    )

    # 9) Dummy‐scatter legend entries for each condition
    for cond, color in cond_color_map.items():
        if cond.startswith("Total"):  # skip the two totals
            continue
        fig_samp.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", size=10, color=color),
            name=cond,
            showlegend=True,
        ))
    fig_samp.update_layout(
        legend=dict(
            title=" Conditions",
            orientation="h",
            x=0.7, xanchor="left",
            y=1.02, yanchor="bottom",
            bordercolor="black", borderwidth=1,
        ),
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )

    return fig_cond, fig_samp

@log_time("Plotting Missing Values Heatmaps")
def plot_mv_heatmaps(adata:AnnData):
    df_raw     = pd.DataFrame(
        adata.layers['normalized'],
        index=adata.obs_names,    # samples
        columns=adata.var_names   # proteins
    )
    df_missing = df_raw.isna().astype(int).T

    fig_binary = plot_binary_cluster_heatmap_plotly(
        adata=adata,
        cond_key="CONDITION",
        layer="normalized",
        title="Clustered Missing Values Heatmap",
        width=900,
        height=900,
    )

    # 3b) Correlation heatmap (plotly) of sample‐sample missingness
    corr = df_missing.corr()

    samples = list(corr.columns)  # same as index
    cond_ser = adata.obs["CONDITION"].astype(str).reindex(samples)

    # reuse the same mapping you already use elsewhere
    cmap_cond = get_color_map(sorted(cond_ser.unique()), palette=None)

    fig_corr = px.imshow(
        corr,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        labels=dict(x="", y="", color="Corr"),
    )
    fig_corr.update_layout(
        title=dict(text="Missing Values Correlation Heatmap", x=0.5),
        coloraxis_colorbar=dict(x=0.90),
    ),
    color_ticks_by_condition(fig_corr, samples, cond_ser, cmap_cond)

    return (fig_binary, fig_corr)

@log_time("Plotting Imputation Distribution by Condition")
def plot_grouped_violin_imputation_by_condition(
        adata,
        colors=('blue','red'),
        title="Distribution of Imputed Values by Condition",
        width=900,
        height=600,
    ) -> go.Figure:

    # 1) pull arrays and condition info
    raw_arr   = adata.layers['raw']                                     # (n_samples × n_proteins)
    norm_arr  = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    cond_series = adata.obs['CONDITION']
    conditions  = cond_series.unique().tolist()                         # preserves original order
    idx_map     = {cond: i for i, cond in enumerate(conditions)}
    offset      = 0.2

    # 2) flatten everything once
    mask      = np.isnan(raw_arr)                                       # boolean (n×p)
    mask_flat = mask.ravel()                                            # 1D length=n*p, row-major
    norm_flat = norm_arr.ravel()
    cond_flat = np.repeat(cond_series.values, raw_arr.shape[1])         # repeats each sample’s cond p times

    # 3) split measured vs imputed entries
    meas_vals  = norm_flat[~mask_flat]
    meas_conds = cond_flat[~mask_flat]
    imp_vals   = norm_flat[mask_flat]
    imp_conds  = cond_flat[mask_flat]

    # 4) assemble the exact same long‐form DataFrame
    df = pd.DataFrame({
        'Condition':   np.concatenate([meas_conds, imp_conds]),
        'Normalization': ['Measured'] * len(meas_vals) + ['Imputed'] * len(imp_vals),
        'Value':         np.concatenate([meas_vals, imp_vals])
    })

    # 5) now the plotting loop is untouched
    fig = go.Figure()
    for norm_label, color in zip(['Measured','Imputed'], colors):
        sub = df[df['Normalization'] == norm_label]
        x_vals = sub['Condition'].map(idx_map) + (
            -offset if norm_label == 'Measured' else +offset
        )
        fig.add_trace(go.Violin(
            x=x_vals,
            y=sub['Value'],
            name=norm_label,
            legendgroup=norm_label,
            line_color=color,
            box_visible=True,
            box_line_color='black',
            box_line_width=1,
            meanline_visible=True,
            opacity=0.6,
            points=False,
            width=offset * 1.8,
            spanmode='hard',
            scalemode='width',
        ))

    # 6) vertical separators and styling (identical to original)
    for i in idx_map.values():
        fig.add_vline(
            x=i,
            line=dict(color='gray', dash='dot', width=1),
            layer='above'
        )
    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(
        template='plotly_white',
        violinmode='overlay',
        title=dict(text=title, x=0.5),
        width=width, height=height,
        xaxis=dict(
            title='Condition',
            tickmode='array',
            tickvals=list(idx_map.values()),
            ticktext=conditions,
        ),
        yaxis=dict(title='Intensity', showgrid=True),
        showlegend=True,
    )
    fig.update_xaxes(showline=True, mirror=True, linecolor='black')
    fig.update_yaxes(showline=True, mirror=True, linecolor='black')

    return fig

@log_time("Plotting Imputation Distribution by Sample (optimized)")
def plot_grouped_violin_imputation_by_sample(
    adata,
    colors=('blue','red'),
    title="Distribution of Imputed Values by Sample",
    width=900,
    height=600,
) -> go.Figure:

    # 1) pull raw + final intensities
    raw_arr  = adata.layers['raw']                                             # shape: (n_samples × n_features)
    norm_arr = (
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    )

    # 2) prepare a flat mask, flat values, and flat sample labels
    n_samples, n_feats = raw_arr.shape
    mask_flat   = np.isnan(raw_arr).ravel()                                     # length = n_samples*n_feats
    norm_flat   = norm_arr.ravel()
    samples     = adata.obs_names.tolist()                                      # length = n_samples
    sample_flat = np.repeat(samples, n_feats)                                   # length = n_samples*n_feats

    # 3) split measured vs imputed
    meas_vals   = norm_flat[~mask_flat]
    meas_samps  = sample_flat[~mask_flat]
    imp_vals    = norm_flat[ mask_flat]
    imp_samps   = sample_flat[ mask_flat]

    # 4) assemble the **same** long‐form DataFrame your loop makes :contentReference[oaicite:0]{index=0}
    df = pd.DataFrame({
        'Sample'       : np.concatenate([meas_samps, imp_samps]),
        'Normalization': ['Measured'] * len(meas_vals) + ['Imputed'] * len(imp_vals),
        'Value'        : np.concatenate([meas_vals, imp_vals])
    })

    # 5) reuse your exact plotting code below—unchanged—
    idx_map    = {s: i for i, s in enumerate(samples)}
    offset     = 0.2
    fig        = go.Figure()
    first_seen = {'Measured': False, 'Imputed': False}

    for sample in samples:
        for norm_label, color in zip(['Measured','Imputed'], colors):
            sub = df[(df.Sample == sample) & (df.Normalization == norm_label)]
            if sub.empty:
                continue
            x0 = idx_map[sample] + (-offset if norm_label=='Measured' else +offset)
            fig.add_trace(go.Violin(
                x=[x0] * len(sub),
                y=sub['Value'],
                name=norm_label,
                legendgroup=norm_label,
                showlegend=not first_seen[norm_label],
                scalegroup=sample,            # per‐sample scaling
                line_color=color,
                box_visible=True,
                box_line_color='black',
                box_line_width=1,
                meanline_visible=True,
                opacity=0.6,
                points=False,
                width=offset * 1.8,
                spanmode='hard',
                scalemode='width',
            ))
            first_seen[norm_label] = True

    # separators & styling (identical)
    for i in idx_map.values():
        fig.add_vline(
            x=i, line=dict(color='gray', dash='dot', width=1), layer='above'
        )
    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(
        template='plotly_white',
        violinmode='overlay',
        title=dict(text=title, x=0.5),
        width=width, height=height,
        xaxis=dict(
            title='Sample',
            tickmode='array',
            tickvals=list(idx_map.values()),
            ticktext=samples,
            tickangle=30,
            showline=True, mirror=True, linecolor='black',
        ),
        yaxis=dict(
            title='Intensity',
            showgrid=True, showline=True, mirror=True, linecolor='black'
        ),
        showlegend=True,
    )

    return fig

@log_time("Plotting Imputation CV and rMAD by Condition")
def plot_grouped_violin_imputation_metrics_by_condition(
    adata,
    metrics=('CV', 'rMAD'),
    colors=('blue', 'red'),
    width=900,
    height=600,
) -> tuple[go.Figure, go.Figure]:

    def make_fig(metric: str) -> go.Figure:
        # carve out measured vs imputed
        raw  = adata.layers['raw']
        norm = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        dm = compute_metric_by_condition(
            AnnData(X=np.where(np.isnan(raw), np.nan, norm),
                   obs=adata.obs.copy(), var=adata.var.copy()),
            metric=metric
        )
        di = compute_metric_by_condition(
            AnnData(X=np.where(~np.isnan(raw), np.nan, norm),
                   obs=adata.obs.copy(), var=adata.var.copy()),
            metric=metric
        )

        # force "Total" first
        conditions = ['Total'] + [c for c in dm.keys() if c != 'Total']
        idx_map    = {c:i for i,c in enumerate(conditions)}
        offset     = 0.2

        # long-form DF
        rows = []
        for cond in conditions:
            rows += [(cond, 'Measured', v) for v in dm.get(cond, []) if not np.isnan(v)]
            rows += [(cond, 'Imputed',  v) for v in di.get(cond, []) if not np.isnan(v)]
        df = pd.DataFrame(rows, columns=['Condition','Normalization','Value'])

        # build fig
        fig = go.Figure()
        first_seen = {'Measured': False, 'Imputed': False}

        for cond in conditions:
            for norm_label, color in zip(['Measured','Imputed'], colors):
                sub = df[(df.Condition==cond) & (df.Normalization==norm_label)]
                if sub.empty:
                    continue
                x0 = idx_map[cond] + (-offset if norm_label=='Measured' else +offset)
                fig.add_trace(go.Violin(
                    x=[x0] * len(sub),
                    y=sub['Value'],
                    name=norm_label,
                    legendgroup=norm_label,
                    showlegend=not first_seen[norm_label],
                    scalegroup=str(cond),
                    line_color=color,
                    box_visible=True,
                    box_line_color='black',
                    box_line_width=1,
                    meanline_visible=True,
                    opacity=0.6,
                    points=False,
                    width=offset * 1.8,
                    spanmode='hard',
                    scalemode='width',
                ))
                first_seen[norm_label] = True

        # separators
        for i in idx_map.values():
            fig.add_vline(x=i, line=dict(color='gray', dash='dot', width=1), layer='above')

        # styling
        fig.update_traces(box_visible=True, meanline_visible=True)
        fig.update_layout(
            template='plotly_white',
            violinmode='overlay',
            title=dict(text=f"{metric} Distribution by Condition", x=0.5),
            width=width, height=height,
            xaxis=dict(
                title='Condition',
                tickmode='array',
                tickvals=list(idx_map.values()),
                ticktext=conditions,
                showline=True, mirror=True, linecolor='black'#, tickangle=30
            ),
            yaxis=dict(title=metric, showgrid=True, showline=True, mirror=True, linecolor='black'),
            showlegend=True,
        )
        return fig

    fig_cv  = make_fig('CV')
    fig_rmad = make_fig('rMAD')
    return fig_rmad, fig_cv

def plot_line_density_by_sample(
    adata,
    before_layer: str = "lognorm",
    after_layer:  str = "normalized",
    bins: int = 200,
) -> go.Figure:
    """
    Same distribution as the violins, but as per‑sample density *lines*.
    - X axis: intensity
    - Y axis: probability density (top of the distribution)
    - One line per sample, dashed, colors from Prism palette.
    - Legend toggling acts on both panels.
    """
    samples = list(adata.obs_names)
    palette = px.colors.qualitative.Prism
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(samples)}

    def _densities(layer: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        arr = adata.layers[layer]                 # shape: samples × proteins
        # keep finite & positive values for range
        vals = arr[np.isfinite(arr) & (arr > 0)]
        if vals.size == 0:
            # fallback to 0..1 to avoid NaNs
            x = np.linspace(0, 1, bins)
            return x, {s: np.zeros_like(x) for s in samples}

        x = np.linspace(vals.min(), vals.max(), bins)
        edges = np.linspace(vals.min(), vals.max(), bins + 1)

        ys: Dict[str, np.ndarray] = {}
        for i, s in enumerate(samples):
            v = arr[i, :]
            v = v[np.isfinite(v) & (v > 0)]
            if v.size == 0:
                y = np.zeros(bins)
            else:
                # histogram density as a quick, robust estimator
                y, _ = np.histogram(v, bins=edges, density=True)
                # tiny smoothing to de‑jag (5‑tap kernel)
                k = np.array([1, 2, 3, 2, 1], dtype=float)
                k /= k.sum()
                y = np.convolve(y, k, mode="same")
            ys[s] = y
        return x, ys

    xb, yb = _densities(before_layer)
    xa, ya = _densities(after_layer)

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=False,
        subplot_titles=["Before Normalization", "After Normalization"],
        horizontal_spacing=0.08,
    )

    # LEFT: Before  |  RIGHT: After
    for s in samples:
        # show legend only on the left panel; legendgroup couples both
        fig.add_trace(go.Scatter(
            x=xb, y=yb[s],
            mode="lines",
            name=str(s),
            legendgroup=str(s),
            showlegend=True,
            line=dict(dash="dash", width=1, color=color_map[s]),
            hovertemplate="Sample: %{fullData.name}<br>Intensity: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=xa, y=ya[s],
            mode="lines",
            name=str(s),
            legendgroup=str(s),
            showlegend=False,              # kept in sync via legendgroup
            line=dict(dash="dash", width=1, color=color_map[s]),
            hovertemplate="Sample: %{fullData.name}<br>Intensity: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        ), row=1, col=2)

    # layout: compact legend; clicking a sample toggles both panels
    fig.update_layout(
        template="plotly_white",
        width=900, height=600,
        title=dict(text="Distribution by Sample", x=0.5),
        legend=dict(
            y=1.0, yanchor="top",
            x=1.02, xanchor="left",
            orientation="v",
            itemwidth=60,
            bordercolor="black", borderwidth=1,
            groupclick="togglegroup",
        ),
        #margin=dict(l=60, r=140, t=60, b=60),
    )
    for c in (1, 2):
        fig.update_xaxes(title_text="Intensity", showline=True, linecolor="black", mirror=True, row=1, col=c)
        fig.update_yaxes(title_text="Density",   showline=True, linecolor="black", mirror=True, row=1, col=c)
    return fig

