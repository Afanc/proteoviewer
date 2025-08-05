import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Tuple, List, Optional, Dict
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from components.plot_utils import plot_histogram_plotly, add_violin_traces, compute_metric_by_condition, get_color_map, plot_bar_plotly, plot_binary_cluster_heatmap_plotly
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

def plot_filter_histograms_old(adata: AnnData):
    """
    Return three separate histograms (Q‑value, PEP, run evidence count)
    of everything *before* filtering, each with its own threshold line.
    """
    # 1) thresholds
    flt = adata.uns["preprocessing"]["filtering"]
    thresholds = {
        "qvalue":             flt["qvalue"]["threshold"],
        "pep":                flt["pep"]["threshold"],
        "run_evidence_count": flt["rec"]["threshold"],
    }

    # 2) raw arrays
    # Q‑values
    kept_q = flt.get('qvalue', {}).get("number_kept")
    removed_q = flt.get('qvalue', {}).get("number_dropped")

    # PEP
    kept_p = flt.get('pep', {}).get("number_kept")
    removed_p = flt.get('pep', {}).get("number_dropped")

    # Run‑evidence count
    kept_re = flt.get('rec', {}).get("number_kept")
    removed_re = flt.get('rec', {}).get("number_dropped")

    arrs = {
        "qvalue": flt.get('qvalue', {}).get("raw_values"),
        "pep": flt.get('pep', {}).get("raw_values"),
        "run_evidence_count": flt.get('rec', {}).get("raw_values"),
    }

    ## how many got filtered out, how many we kept
    filtered_kept_len = {
        "qvalue":            (removed_q,  kept_q),
        "pep":               (removed_p,  kept_p),
        "run_evidence_count":(removed_re, kept_re),
    }

    # definitions
    metrics = [
        ("qvalue",             "Q‑value",             thresholds["qvalue"]),
        ("pep",                "PEP",                 thresholds["pep"]),
        ("run_evidence_count", "Run Evidence Count",  thresholds["run_evidence_count"]),
    ]

    figs: Dict[str, Figure] = {}
    for key, label, thr in metrics:
        vals = arrs[key]
        vals = vals[np.isfinite(vals)]

        removed = filtered_kept_len[key][0]
        kept = filtered_kept_len[key][1]
        title_text = f"{label} (Cutoff {thr}, kept: {kept}/{kept+removed})"

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=80,
            name=label,
            marker_line_color="black",
            marker_line_width=1,
            showlegend=False,
            hoverinfo='none',
        ))
        fig.add_vline(
            x=thr,
            line=dict(color="red", dash="dash"),
        )
        fig.update_layout(
            title=dict(text=title_text, x=0.5),
            xaxis_title=label,
            yaxis_title="Count",
            yaxis_type="log",
            template="plotly_white",
            width=600,
            height=400,
        )
        fig.update_yaxes(
            dtick=1,
            exponentformat='power',
            showexponent='all',
        )

        figs[key] = fig

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

def plot_grouped_violin_metric_by_condition(
    adata,
    metric: str,
    colors=('blue', 'red'),
    title=None,
    width=900,
    height=600,
) -> go.Figure:
    raw  = compute_metric_by_condition(adata,
               layer="lognorm",
               cond_key="CONDITION",
               metric=metric,
           )   # → Dict[Condition, np.ndarray]

    norm = compute_metric_by_condition(adata,
               layer="normalized",
               cond_key="CONDITION",
               metric=metric,
           )

    # 2) vectorized reshape to long form
    df_before = (
        pd.DataFrame(raw)
          .melt(var_name="Condition", value_name="Value")
          .assign(Normalization="Before")
    )
    df_after = (
        pd.DataFrame(norm)
          .melt(var_name="Condition", value_name="Value")
          .assign(Normalization="After")
    )

    # 3) stitch together & drop the “Total” aggregate
    dfm = pd.concat([df_before, df_after], ignore_index=True)

    # 2) map each condition to an integer slot
    conditions = dfm["Condition"].unique()
    idx_map    = {cond:i for i,cond in enumerate(conditions)}
    offset     = 0.2

    # 3) build the violin traces
    fig = go.Figure()
    for norm, color in zip(["Before", "After"], colors):
        sub = dfm[dfm["Normalization"] == norm]
        # numeric x with ±offset
        x_vals = sub["Condition"].map(idx_map) + ( -offset if norm=="Before" else +offset )
        fig.add_trace(go.Violin(
            x=x_vals,
            y=sub["Value"],
            name=norm,
            legendgroup=norm,
            line_color=color,
            box_visible=True,
            box_line_color="black",
            box_line_width=1,
            opacity=0.6,
            width=offset * 1.8,   # half-width ≈ offset
            meanline_visible=True,
            points=False,
            #hoverinfo="skip",
            spanmode="hard",
        ))


    for i in list(idx_map.values()):
        fig.add_vline(
            x=i,
            line=dict(color="gray", dash="dot", width=1),
            layer="above",
        )

    # 4) finalize layout with one tick per condition
    title_text=f"{metric} per Condition"
    fig.update_layout(
        template="plotly_white",
        violinmode="overlay",  # works with our manual offsets
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

    # 6) draw boxed axes
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

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

    df_raw = pd.DataFrame(
        adata.layers['normalized'],
        index=adata.obs.index,       # samples
        columns=adata.var_names      # proteins
    )

    # --- 2) Count missing per sample & per condition
    mvi_per_sample = df_raw.isna().sum(axis=1)      # Series: Sample → count
    conds = adata.obs['CONDITION']
    mvi_per_condition = mvi_per_sample.groupby(conds).sum()  # Series: Condition → count

    #  total mvi
    total_imputed     = int(mvi_per_condition.sum())        # what will be imputed
    total_entries     = df_raw.size                         # all values
    total_non_imputed = total_entries - total_imputed       # left untouched
    rate = total_imputed/total_entries

    total_mvi = int(mvi_per_condition.sum())

#    # --- 3) Get a stable color map for conditions
    cond_levels = mvi_per_condition.index.sort_values(ascending=False)

    cond_labels = ["Total (Non-Missing)", "Total (Missing)"] + cond_levels.tolist()

    # manual color map: gray for the two totals, then Plotly palette
    palette = px.colors.qualitative.Plotly
    cond_color_map = {
        "Total (Non-Missing)": "#888888",
        "Total (Missing)":     "#CCCCCC"
    }
    for i, lbl in enumerate(cond_levels):
        cond_color_map[lbl] = palette[i % len(palette)]


    # append total for conditions
    cond_x = cond_labels
    cond_y = [
        total_non_imputed,
        total_imputed,
        *[int(mvi_per_condition.loc[c]) for c in cond_levels]
    ]

    condition_title = f"Missing Values (ratio = {100*rate:.2f}%)"

    # --- 4) Plotly figures
    fig_cond = plot_bar_plotly(
        x=cond_x,
        y=cond_y,
        colors=[cond_color_map[c] for c in cond_x],
        title=condition_title,
        x_title="Condition",
        y_title="Count",
        width=800, height=400
    )
    fig_cond.update_yaxes(type="log", dtick=1, exponentformat="power")

    fig_samp = plot_bar_plotly(
        x=list(mvi_per_sample.index),
        y=list(mvi_per_sample.values),
        colors=[cond_color_map[conds[s]] for s in mvi_per_sample.index],
        title="Missing Values per Sample",
        x_title="Sample",
        y_title="Count",
        width=1200, height=400
    )

    # weird behaviour with trace 0, correct
    bar = fig_samp.data[0]

    # 2) Hide it from the legend
    bar.showlegend = False

    # 3) Attach a list of conditions matching each bar
    bar.customdata = [conds[s] for s in mvi_per_sample.index]

    # 4) New hovertemplate: show the condition, sample, and count
    bar.hovertemplate = (
        "Condition: %{customdata}<br>"
        "Sample: %{x}<br>"
        "Missing: %{y}<extra></extra>"
    )

    # legend
    for cond, color in cond_color_map.items():
        # skip Total if you’ve added it
        if cond in ["Total (Missing)", "Total (Non-Missing)"]:
            continue
        fig_samp.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", size=10, color=color),
            name=cond,
            showlegend=True
        ))

    # position the legend as you like
    fig_samp.update_layout(
        legend=dict(
            title="Conditions",
            orientation="h",
            x=0.7, xanchor="left",
            y=1.02, yanchor="bottom"
        )
    )

    return (fig_cond, fig_samp)

@log_time("Plotting Missing Values Heatmaps")
def plot_mv_heatmaps(adata:AnnData):
    df_raw     = pd.DataFrame(
        adata.layers['normalized'],
        index=adata.obs_names,    # samples
        columns=adata.var_names   # proteins
    )
    df_missing = df_raw.isna().astype(int).T

    ## 2) Prepare a color‐series for your samples (columns) keyed to CONDITION
    #cond_to_color = get_color_map(
    #    adata.obs['CONDITION'].unique().tolist(),
    #    palette=None,    # your default Plotly qualitative
    #    anchor=None
    #)
    #col_colors = adata.obs['CONDITION'].map(cond_to_color)
    #col_colors.name = "Condition"

    fig_binary = plot_binary_cluster_heatmap_plotly(
        adata=adata,
        cond_key="CONDITION",
        layer="normalized",
        title="Clustered Missing Values Heatmap",
        width=900,
        height=900,
    )
    ## 3a) Clustered heatmap (matplotlib/seaborn) via your helper
    #g = plot_cluster_heatmap_plt(
    #    data=df_missing,
    #    col_colors=col_colors,
    #    title="Clustered Missing Values",
    #    legend_mapping=cond_to_color,
    #    row_cluster=False,
    #    col_cluster=True,
    #)

    # 3b) Correlation heatmap (plotly) of sample‐sample missingness
    corr = df_missing.corr()
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

    return (fig_binary, fig_corr)

@log_time("Plotting Imputation Distribution by Condition")
def plot_grouped_violin_imputation_by_condition(
        adata,
        colors=('blue','red'),
        title="Distribution of Imputed Values by Condition",
        width=900,
        height=600,
    ) -> go.Figure:

    raw = adata.layers['raw']
    # final intensities (with imputation filled in)
    norm = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    cond_series = adata.obs['CONDITION']
    conditions = cond_series.unique().tolist()
    idx_map    = {cond:i for i,cond in enumerate(conditions)}
    offset     = 0.2

    # build long rows
    rows = []
    for sample_idx, sample in enumerate(adata.obs_names):
        cond = cond_series.iloc[sample_idx]
        mask_imp = np.isnan(raw[sample_idx, :])
        # Measured = final norm where raw was not NaN
        meas = norm[sample_idx, :][~mask_imp]
        # Imputed = final norm where raw was NaN
        impt = norm[sample_idx, :][mask_imp]
        rows += [(cond, 'Measured', v) for v in meas]
        rows += [(cond, 'Imputed',  v) for v in impt]

    df = pd.DataFrame(rows, columns=['Condition','Normalization','Value'])

    # now exactly your grouped‐violin recipe:
    fig = go.Figure()
    for norm_label, color in zip(['Measured','Imputed'], colors):
        sub = df[df['Normalization']==norm_label]
        x_vals = sub['Condition'].map(idx_map) + ( -offset if norm_label=='Measured' else +offset )
        fig.add_trace(go.Violin(
            x=x_vals,
            y=sub['Value'],
            name=norm_label,
            legendgroup=norm_label,
            line_color=color,
            box_visible=True,
            box_line_color='black',
            box_line_width=1,
            opacity=0.6,
            width=offset*1.8,
            meanline_visible=True,
            points=False,
            spanmode='hard',
            scalemode='width',
            scalegroup="all_same",
        ))

    # vertical separators
    for i in idx_map.values():
        fig.add_vline(x=i, line=dict(color='gray',dash='dot',width=1),layer='above')

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

@log_time("Plotting Imputation Distribution by Sample")
def plot_grouped_violin_imputation_by_sample(
    adata,
    colors=('blue','red'),
    title="Distribution of Imputed Values by Sample",
    width=900,
    height=600,
) -> go.Figure:

    # 1) extract raw vs final intensities
    raw  = adata.layers['raw']
    norm = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    # 2) build long‐form DataFrame
    rows = []
    samples = adata.obs_names.tolist()
    for i, sample in enumerate(samples):
        mask_imp = np.isnan(raw[i, :])
        meas = norm[i, :][~mask_imp]
        impt = norm[i, :][mask_imp]
        rows += [(sample, 'Measured', v) for v in meas]
        rows += [(sample, 'Imputed',  v) for v in impt]
    df = pd.DataFrame(rows, columns=['Sample','Normalization','Value'])

    # 3) prepare numeric mapping + offset
    idx_map = {s: i for i, s in enumerate(samples)}
    offset  = 0.2

    # 4) plot with manual offsets and per‐sample scalegroup
    fig = go.Figure()
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

    # 5) vertical separators
    for i in idx_map.values():
        fig.add_vline(
            x=i,
            line=dict(color='gray', dash='dot', width=1),
            layer='above'
        )

    # 6) styling & layout
    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(
        template='plotly_white',
        violinmode='overlay',
        title=dict(text=title, x=0.5),
        width=width,
        height=height,
        xaxis=dict(
            title='Sample',
            tickmode='array',
            tickvals=list(idx_map.values()),
            ticktext=samples,
            tickangle=30,
            showline=True,
            mirror=True,
            linecolor='black',
        ),
        yaxis=dict(title='Intensity', showgrid=True, showline=True, mirror=True, linecolor='black'),
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

