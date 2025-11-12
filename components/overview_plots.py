import re
from anndata import AnnData
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Set
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from components.plot_utils import plot_stacked_proteins_by_category, plot_violins, compute_metric_by_condition, get_color_map, plot_cluster_heatmap_plotly, plot_volcanoes

from utils import logger, log_time

def _shorten_labels(labels, head=6, tail=4, sep="…"):
    """
    Shorten strings like 'VERY_LONG_SAMPLE_NAME' -> 'VERY_L…ME'.
    Keeps ordering and hover info intact (we only change tick text).
    """
    def _short(s: str) -> str:
        s = str(s)
        if len(s) <= head + tail + 1:
            return s
        return f"{s[:head]}{sep}{s[-tail:]}"
    # Make labels unique if shortening collides
    out = [_short(s) for s in labels]
    if len(set(out)) < len(out):
        seen = {}
        for i, s in enumerate(out):
            if s in seen:
                seen[s] += 1
                out[i] = f"{s}_{seen[s]}"  # minimal disambiguation
            else:
                seen[s] = 0
    return out

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
    data_type: str = "default",
    width: int = 900,
    height: int = 900,
    show_measured: bool = True,
    show_imp_cond1:  bool = True,
    show_imp_cond2:  bool = True,
    min_nonimp_per_cond: int=0,
    min_precursors: int=1,
    highlight: str = None,
    highlight_group: str = None,
    color_by: str = None,
    contrast: str = None,
) -> go.Figure:
    # simply forward the SessionState + args into the pure util
    fig =  plot_volcanoes(
        state=state,               # if `im` is your SessionState
        contrast=contrast,
        data_type=data_type,
        sign_threshold=sign_threshold,
        height=height,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        min_nonimp_per_cond=min_nonimp_per_cond,
        min_precursors=min_precursors,
        highlight=highlight,
        highlight_group=highlight_group,
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
    elif layer.value == "Log-only":
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

    # Average iBAQ across samples
    ibaq_layer = ad.layers.get("ibaq")
    avg_ibaq = None
    if ibaq_layer is not None:
        ibaq_mat = ibaq_layer.toarray() if hasattr(ibaq_layer, "toarray") else np.asarray(ibaq_layer)
        col_vals = ibaq_mat[:, col_idx]
        avg_ibaq_val = np.nanmean(col_vals.astype(float))
        avg_ibaq = float(avg_ibaq_val) if np.isfinite(avg_ibaq_val) else None

    protein_info = {
        'uniprot_id': uniprot_id,
        'qval': qval,
        'logfc': logfc,
        'avg_int': avg_int,
        'index': col_idx,
        'avg_ibaq': avg_ibaq
    }
    return protein_info

def _compile_user_pattern(pat: Optional[str]) -> Optional[re.Pattern]:
    """
    Accepts simple wildcards (*, ?) or full regex. Case-insensitive.
    If no wildcard/regex tokens are present, auto-wrap as contains (.*....*).
    """
    if not pat:
        return None
    raw = pat.strip()
    if not raw:
        return None

    # Heuristic: if user typed regex-ish tokens, treat as regex
    looks_regex = bool(re.search(r"[.\[\]\(\)\{\}\|\+\^\$]", raw))
    if looks_regex:
        return re.compile(raw, re.IGNORECASE)

    # Treat * and ? as wildcards; escape everything else
    escaped = re.escape(raw).replace(r"\*", ".*").replace(r"\?", ".")
    if "*" not in raw and "?" not in raw:
        # no explicit wildcard → "contains"
        escaped = f".*{escaped}.*"
    return re.compile(escaped, re.IGNORECASE)


def resolve_pattern_to_uniprot_ids(adata, field: str, pattern: Optional[str]) -> Set[str]:
    """
    Resolve a free-text/regex pattern into a set of UniProt IDs (adata.var_names),
    searching in one of:
      - "FASTA headers"  → PROTEIN_DESCRIPTIONS (fallback: FASTA_HEADERS)
      - "Gene names"     → GENE_NAMES (split on ; , whitespace)
      - "UniProt IDs"    → adata.var_names
    """
    rx = _compile_user_pattern(pattern)
    if rx is None:
        return set()

    ids = np.array(adata.var_names, dtype=str)
    var = adata.var

    if field == "FASTA headers":
        col = "FASTA_HEADERS"
        if col is None:
            return set()
        texts = var[col].astype(str).to_numpy()
        mask = np.fromiter((bool(rx.search(t)) for t in texts), dtype=bool, count=len(texts))
        return set(ids[mask])

    elif field == "Gene names":
        if "GENE_NAMES" not in var.columns:
            return set()
        gn = var["GENE_NAMES"].astype(str).to_numpy()
        splitter = re.compile(r"[;,\s]+")
        def any_match(s: str) -> bool:
            return any(rx.search(p) for p in splitter.split(s) if p)
        mask = np.fromiter((any_match(s) for s in gn), dtype=bool, count=len(gn))
        return set(ids[mask])

    else:  # "UniProt IDs"
        mask = np.fromiter((bool(rx.search(u)) for u in ids), dtype=bool, count=len(ids))
        return set(ids[mask])

def resolve_exact_list_to_uniprot_ids(adata, field: str, items: List[str] | Set[str]) -> Set[str]:
    """
    Map a *list of exact identifiers* to UniProt IDs (adata.var_names), interpreting
    the list according to `field`:
      - "FASTA headers": exact match against FASTA_HEADERS
      - "Gene names":    exact match against any token of GENE_NAMES split on ; , whitespace
      - "UniProt IDs":   exact match against adata.var_names
    Returns a set of UniProt IDs (strings).
    """
    if not items:
        return set()
    items = set(map(str, items))

    ids = np.array(adata.var_names, dtype=str)
    var = adata.var

    if field == "FASTA headers":
        col = "FASTA_HEADERS"
        if col not in var.columns:
            return set()
        texts = var[col].astype(str).to_numpy()
        mask = np.isin(texts, list(items))
        return set(ids[mask])

    elif field == "Gene names":
        if "GENE_NAMES" not in var.columns:
            return set()
        gn = var["GENE_NAMES"].astype(str).to_numpy()
        splitter = re.compile(r"[;,\s]+")
        def matches_any_token(s: str) -> bool:
            return any((tok in items) for tok in splitter.split(s) if tok)
        mask = np.fromiter((matches_any_token(s) for s in gn), dtype=bool, count=len(gn))
        return set(ids[mask])

    else:  # "UniProt IDs"
        return set(u for u in ids if u in items)

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

@log_time("Plotting Cohort vs Non-cohort Violins")
def plot_group_violin_for_volcano(
    state,
    contrast: str,
    highlight_group: list[str] | set[str],
    show_measured: bool,
    show_imp_cond1: bool,
    show_imp_cond2: bool,
    width: int = 1200,
    height: int = 300,
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Horizontal violins of log2FC for cohort vs non-cohort, using the same
    visible subset as the volcano (Observed/Imputed toggles).
    """
    ad = state.adata
    # log2FC & q-value
    df_fc = pd.DataFrame(ad.varm["log2fc"], index=ad.var_names, columns=ad.uns["contrast_names"])
    df_q  = pd.DataFrame(ad.varm["q_ebayes"], index=ad.var_names, columns=ad.uns["contrast_names"])
    x = df_fc[contrast]  # log2FC

    # Visible subset masks (mirror volcano)
    miss = pd.DataFrame(ad.uns["missingness"])
    grp1, grp2 = contrast.split("_vs_")
    a = miss[grp1].values >= 1.0
    b = miss[grp2].values >= 1.0
    measured_mask = (~a & ~b)
    imp1_mask     = (a & ~b)
    imp2_mask     = (b & ~a)

    visible = np.zeros_like(measured_mask, dtype=bool)
    if show_measured:  visible |= measured_mask
    if show_imp_cond1: visible |= imp1_mask
    if show_imp_cond2: visible |= imp2_mask

    # Cohort membership by UniProt or Gene
    ids = np.array(ad.var_names, dtype=str)
    genes = np.array(ad.var["GENE_NAMES"].astype(str))
    group = set(map(str, (highlight_group or [])))
    in_group = np.array([(pid in group) or (g in group) for pid, g in zip(ids, genes)], dtype=bool)

    # Two distributions (only visible points)
    cohort_vals = x[visible & in_group].to_numpy(dtype=float)
    rest_vals   = x[visible & ~in_group].to_numpy(dtype=float)

    # Build a clean, tiny figure if no cohort
    if cohort_vals.size == 0 and rest_vals.size == 0:
        fig = go.Figure()
        fig.update_layout(width=width, height=height, template="plotly_white",
                          margin=dict(l=60, r=40, t=10, b=40))
        return fig

    # Shared x-range like volcano
    all_vals = np.concatenate([cohort_vals, rest_vals]) if rest_vals.size else cohort_vals
    xmin, xmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = max(0.05 * (xmax - xmin), 0.2)
    xmin -= pad; xmax += pad

    # Colors
    labels = ["Cohort", "Non-cohort"]
    cmap = get_color_map(labels, palette=px.colors.qualitative.Plotly, anchor="Cohort", anchor_color="#6c5ce7")
    def _v(arr, name):
        colors = {"Cohort": "#6c5ce7", "Non-cohort": "#2a9d8f"}
        return go.Violin(
            y=[name]*len(arr),  # horizontal violin (category on y, values on x)
            x=arr,
            orientation="h",
            name=name,
            line_color=colors[name],
            line_width=1,
            width=0.45,                # thinner vertical footprint
            opacity=0.65,
            box_visible=True,
            meanline_visible=True,
            points=False,
            hoverinfo="skip",          # ← disable hover
            showlegend=False,
        )

    fig = go.Figure()
    if cohort_vals.size:
        fig.add_trace(_v(cohort_vals, "Cohort"))
    if rest_vals.size:
        fig.add_trace(_v(rest_vals, "Non-cohort"))

    # Median / SD annotations to the right of each violin
    def _stats(arr):  # robust to nans
        med = float(np.nanmedian(arr)) if arr.size else float("nan")
        sd  = float(np.nanstd(arr, ddof=1)) if np.sum(~np.isnan(arr)) > 1 else float("nan")
        return med, sd
    annos = []
    for name, arr in (("Cohort", cohort_vals), ("Non-cohort", rest_vals)):
        if arr.size:
            med, sd = _stats(arr)
            annos.append(dict(
                x=med, y=name, xref="x", yref="y",
                text=f"median: {med:.2f} • sd: {sd:.2f}",
                showarrow=False,
                xanchor="center", yanchor="bottom",
                yshift=12,
                font=dict(size=11, color="black"),
            ))

    xr = [xmin, xmax] if x_range is None else list(x_range)
    fig.update_layout(
        template="plotly_white",
        width=width, height=height,
        margin=dict(t=30, b=0, l=70, r=130),
        title="",
        xaxis=dict(title="", range=xr, zeroline=True, zerolinecolor="black"),
        yaxis=dict(title="", side="right"),
        violinmode="group",
        violingap=0.00,
        violingroupgap=0.0,
        showlegend=False,
    )
    fig.update_layout(annotations=annos)
    return fig

@log_time("Plotting Phosphosite Barplot")
def plot_intensity_by_site(state, contrast, site_id: str, layer_choice: str) -> go.Figure:
    ad = state.adata
    # layer mapping (phospho)
    if layer_choice == "Raw":
        proc = ad.layers.get("raw", ad.X)
        raw_for_mask = proc
        y_label = "Intensity"
    elif layer_choice == "Log-only":
        proc = ad.layers.get("lognorm", ad.X)
        raw_for_mask = ad.layers.get("raw", None)
        y_label = "Log Intensity"
    else:  # "Final"
        proc = ad.X
        raw_for_mask = ad.layers.get("raw", None)
        y_label = "Log Intensity"

    # column index by var_names (phosphosite ids)
    try:
        col = list(map(str, ad.var_names)).index(str(site_id))
    except ValueError:
        return px.bar(pd.DataFrame({"x": [], "y": []}))

    y_vals = proc[:, col].A1 if hasattr(proc, "A1") else proc[:, col]
    if raw_for_mask is not None:
        raw_vals = raw_for_mask[:, col].A1 if hasattr(raw_for_mask, "A1") else raw_for_mask[:, col]
        imputed_mask = np.isnan(raw_vals)
    else:
        imputed_mask = np.zeros_like(y_vals, dtype=bool)

    df = pd.DataFrame({
        "sample": ad.obs_names,
        "condition": ad.obs["CONDITION"],
        "intensity": y_vals,
        "imputed": imputed_mask,
    })
    grp1, grp2 = contrast.split("_vs_")
    df = df[df["condition"].isin([grp1, grp2])]
    names1 = ad.obs_names[ad.obs["CONDITION"] == grp1].tolist()
    names2 = ad.obs_names[ad.obs["CONDITION"] == grp2].tolist()
    order = sorted(names1) + sorted(names2)
    colors = get_color_map(sorted(ad.obs["CONDITION"].unique()), px.colors.qualitative.Plotly)

    fig = px.bar(
        df, x="sample", y="intensity", color="condition",
        pattern_shape="imputed",
        pattern_shape_map={False: "", True: "/"},
        labels={"intensity": y_label, "sample": "Sample"},
        color_discrete_map=colors,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)

    fig.update_xaxes(
        tickvals=order,                      # full sample ids (the categories)
        ticktext=_shorten_labels(order),     # what we display on the axis
    )

    fig.update_traces(marker_pattern_fillmode="overlay", marker_pattern_size=6, marker_pattern_solidity=0.3)
    # hide autoshow legends on traces, add clean legend entries
    for tr in fig.data: tr.showlegend = False
    for cond in [grp1, grp2]:
        fig.add_trace(go.Bar(x=[None], y=[None], name=cond, marker_color=colors[cond], showlegend=True))
    if imputed_mask.any():
        fig.add_trace(go.Bar(
            x=[None], y=[None], name="Imputed",
            marker_color="white", marker_pattern_shape="/",
            marker_pattern_fillmode="overlay", marker_pattern_size=6, marker_pattern_solidity=0.3,
            showlegend=True
        ))
    fig.update_layout(
        margin={"t": 40, "b": 40, "l": 60, "r": 60},
        title=dict(text="Phospho Intensity", x=0.5),
        legend_title_text="Conditions",
        legend_itemclick=False, legend_itemdoubleclick=False,
    )
    return fig

@log_time("Plotting Covariate Barplot")
def plot_covariate_by_site(state, contrast: str, site_id: str, layer_choice: str) -> go.Figure:
    ad = state.adata

    # Non-phospho runs or no covariate: return empty bar to keep layout stable
    #if "covariate" not in ad.layers:
    #    print("oops")
    #    return px.bar(pd.DataFrame({"x": [], "y": []}))

    # Use layers_map if present, else fall back to the layers that actually exist.

    cov_map = {
        "Raw":        "raw_covariate",
        "Log-only":   "lognorm_covariate",
        "Normalized": "normalizade_covariate",
        "Processed":  "processed_covariate",
        "Centered":   "centered_covariate",               # shown only if present
        "Residuals":  "residuals_covariate",              # legacy alias
    }
    layer_key = cov_map.get(layer_choice, "covariate")
    # choose label based on raw vs log scale

    y_label = "Intensity" if layer_key == (cov_map.get("Raw") or "__none__") else "Log Intensity"

    # pick column (= feature/site) safely
    try:
        col = list(map(str, ad.var_names)).index(str(site_id))
    except ValueError:
        return px.bar(pd.DataFrame({"x": [], "y": []}))

    # pull selected layer; handle dense/sparse
    proc = ad.layers.get(layer_key, ad.layers["residuals_covariate"])
    y_vals = proc[:, col]

    # imputation pattern: derive from global 'raw' if covariate-raw not stored
    raw_for_mask_layer = cov_map.get("Raw")
    rawL = ad.layers.get(raw_for_mask_layer) if raw_for_mask_layer else None
    if rawL is None:
        rawL = ad.layers.get("raw")  # fallback: global raw
    if rawL is not None:
        raw_arr = rawL.toarray() if hasattr(rawL, "toarray") else rawL
        imputed_mask = np.isnan(raw_arr[:, col])
    else:
        imputed_mask = np.isnan(y_vals)

    # condition column shim
    cond_col = "CONDITION" if "CONDITION" in ad.obs.columns else ("Condition" if "Condition" in ad.obs.columns else None)
    if cond_col is None:
        return px.bar(pd.DataFrame({"x": [], "y": []}))
    cond = ad.obs[cond_col]

    # restrict to the two groups of the contrast
    try:
        grp1, grp2 = str(contrast).split("_vs_")
    except Exception:
        # if contrast name is unexpected, just show all
        grp1 = grp2 = None

    df = pd.DataFrame({
        "sample": ad.obs_names,
        "condition": cond.values,
        "intensity": y_vals,
        "imputed": imputed_mask,
    })

    if grp1 and grp2:
        df = df[df["condition"].isin([grp1, grp2])]
        # order samples by condition, preserving index order within group
        left  = [s for s in ad.obs_names if ad.obs.loc[s, cond_col] == grp1]
        right = [s for s in ad.obs_names if ad.obs.loc[s, cond_col] == grp2]
        order = left + right
    else:
        order = list(ad.obs_names)

    # colors
    cond_levels = sorted(pd.unique(df["condition"]))
    colors = get_color_map(sorted(ad.obs["CONDITION"].unique()), px.colors.qualitative.Plotly)

    # plot
    fig = px.bar(
        df, x="sample", y="intensity", color="condition",
        pattern_shape="imputed",
        pattern_shape_map={False: "", True: "/"},
        labels={"intensity": y_label, "sample": "Sample"},
        color_discrete_map=colors,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_xaxes(
        tickvals=order,                      # full sample ids (the categories)
        ticktext=_shorten_labels(order),     # what we display on the axis
    )

    fig.update_traces(marker_pattern_fillmode="overlay", marker_pattern_size=6, marker_pattern_solidity=0.3)

    # clean legend (one entry per condition, plus 'Imputed' if applicable)
    for tr in fig.data:
        tr.showlegend = False
    for c in cond_levels:
        fig.add_trace(go.Bar(x=[None], y=[None], name=c, marker_color=colors[c], showlegend=True))
    if bool(np.any(df["imputed"].values)):
        fig.add_trace(go.Bar(
            x=[None], y=[None], name="Imputed",
            marker_color="white", marker_pattern_shape="/",
            marker_pattern_fillmode="overlay", marker_pattern_size=6, marker_pattern_solidity=0.3,
            showlegend=True
        ))

    fig.update_layout(
        margin={"t": 40, "b": 40, "l": 60, "r": 60},
        title=dict(text="Flowthrough Intensity", x=0.5),
        legend_title_text="Conditions",
        legend_itemclick=False, legend_itemdoubleclick=False,
    )
    return fig

def _get_site_stats(adata: AnnData, contrast: str, site_id: str, which: str = "adjusted") -> dict:
    """
    Pull phospho / covariate stats that were persisted by the exporter.
    - which: 'adjusted' (default), 'raw', 'covariate'
    Returns {logfc, qval, pval}; missing → np.nan.
    """
    import numpy as np
    site = str(site_id)
    lim = adata.uns.get("limma", {})
    if which == "covariate":
        block = lim.get("covariate", {})
        l2fc = block.get("log2fc", {}).get(contrast, {})
        qval = block.get("q_ebayes", {}).get(contrast, {})
        pval = block.get("p_ebayes", {}).get(contrast, {})
    elif which == "raw":
        block = lim.get("phospho", {})
        l2fc = block.get("raw_log2fc", {}).get(contrast, {})
        qval = block.get("raw_q_ebayes", {}).get(contrast, {})
        pval = block.get("raw_p_ebayes", {}).get(contrast, {})
    else:  # adjusted
        block = lim.get("phospho", {})
        l2fc = block.get("log2fc", {}).get(contrast, {})
        qval = block.get("q_ebayes", {}).get(contrast, {})
        pval = block.get("p_ebayes", {}).get(contrast, {})
    return dict(
        logfc=float(l2fc.get(site, np.nan)),
        qval=float(qval.get(site, np.nan)),
        pval=float(pval.get(site, np.nan)),
    )

