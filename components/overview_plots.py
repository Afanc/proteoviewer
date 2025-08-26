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
    sort_by: str = "sample",
) -> go.Figure:
    """
    Count proteins per sample and draw as a bar plot using the generic helper.
    """
    # call the generic bar helper

    fig = plot_stacked_proteins_by_category(adata, sort_by=sort_by)
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


@log_time("Plotting Volcano Plots")
def plot_volcanoes_wrapper(
    state,
    sign_threshold: float = 0.05,
    width: int = 900,
    height: int = 900,
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
        #width=width,
        height=height,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        highlight=highlight,
        color_by=color_by,
    )

    return fig

@log_time("Plotting Protein Barplots")
def plot_intensity_by_protein(state, contrast, protein, layer):
    ad = state.adata

    # pick your normalized layer (fallback to .X)
    proc_data = ad.X
    intensity_scale = "Log Intensity"
    if layer.value == "Raw":
        proc_data = ad.layers.get('raw')
        intensity_scale = "Intensity"
    elif layer.value == "Log (pre-norm)":
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

    # fix sample order
    names1 = ad.obs_names[ad.obs["CONDITION"] == grp1].tolist()
    names2 = ad.obs_names[ad.obs["CONDITION"] == grp2].tolist()
    sample_order = sorted(names1) + sorted(names2)

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
    fig.update_xaxes(categoryorder="array", categoryarray=sample_order)
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
        margin={"t":40,"b":40,"l":60,"r":60},
        title=dict(text="Protein Expression", x=0.5),
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
        ad.varm["q_ebayes"],
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

@log_time("Plotting Peptide Trends (centered)")
def plot_peptide_trends_centered(adata, uniprot_id: str, contrast: str) -> go.Figure:
    # 1) pull & slice peptide matrices
    pep = adata.uns["peptides"]
    X_all   = np.asarray(pep["centered"], dtype=float)     # (rows x samples)
    rows    = list(pep["rows"])
    prot_ix = np.asarray(pep["protein_index"], dtype=str)  # per-row protein id
    seqs    = np.asarray(pep["peptide_seq"], dtype=str)    # per-row peptide sequence
    cols    = list(map(str, pep["cols"]))                  # sample names in matrix

    # keep only peptides belonging to this UniProt id (no group handling here)
    keep = (prot_ix == str(uniprot_id))
    X = X_all[keep, :]
    seqs = seqs[keep]

    # 2) align columns to obs order, then filter to the contrast’s samples
    obs_order = list(map(str, adata.obs_names))
    if cols != obs_order:
        pos = {c: i for i, c in enumerate(cols)}
        idx = [pos[c] for c in obs_order if c in pos]
        X   = X[:, idx]
        cols = [obs_order[i] for i in range(len(obs_order)) if obs_order[i] in pos]

    grp1, grp2 = contrast.split("_vs_")
    conds = adata.obs.loc[cols, "CONDITION"].astype(str).to_numpy()
    mask  = np.isin(conds, [grp1, grp2])
    X = X[:, mask]
    sample_labels = np.array(cols)[mask]
    cond_labels   = conds[mask]

    #order
    order1 = sorted([s for s, c in zip(sample_labels, cond_labels) if c == grp1])
    order2 = sorted([s for s, c in zip(sample_labels, cond_labels) if c == grp2])
    new_labels = order1 + order2

    pos = {s: i for i, s in enumerate(sample_labels)}
    col_idx = [pos[s] for s in new_labels]

    X = X[:, col_idx]
    sample_labels = np.array(new_labels)
    cond_labels = np.array([grp1] * len(order1) + [grp2] * len(order2))

    # 3) build figure: one line per peptide, marker color by condition
    fig = go.Figure()
    palette = px.colors.qualitative.Prism
    uniq = sorted(pd.unique(seqs).tolist())
    line_cmap = {s: palette[i % len(palette)] for i, s in enumerate(uniq)}

    cond_cmap = get_color_map([grp1, grp2], palette=px.colors.qualitative.Plotly)

    def _truncate_name(name, max_len=10):
        return name if len(name) <= max_len else name[:max_len - 1] + "…"

    for y, seq in zip(X, seqs):
        if np.isnan(y).all():
            continue
        valid = ~np.isnan(y)
        idx = np.flatnonzero(valid)
        singleton_sizes = np.zeros_like(y, dtype=float)
        if idx.size:
            # split into contiguous runs
            breaks = np.where(np.diff(idx) != 1)[0] + 1
            runs = np.split(idx, breaks)
            for run in runs:
                if run.size == 1:
                    singleton_sizes[run[0]] = 7.0  # only this point gets a marker

        fig.add_trace(go.Scatter(
            x=sample_labels,
            y=y,
            mode="lines+markers",
            name=_truncate_name(seq),
            line=dict(color=line_cmap[seq], width=2, dash="dash"),
            marker=dict(size=singleton_sizes, color=line_cmap[seq]),
            customdata=np.c_[np.full_like(y, seq, dtype=object), cond_labels],
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Sample: %{x}<br>"
                          "Cond: %{customdata[1]}<br>"
                          "Value: %{y:.3f}<extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="Peptide trends", x=0.5),
        xaxis_title="Sample",
        yaxis_title="Intensity / mean",
        margin=dict(l=60, r=40, t=40, b=50),
        legend_title_text="Peptide",
    )
    return fig

