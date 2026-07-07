import re
from anndata import AnnData
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Set
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from components.plot_utils import (
    plot_stacked_proteins_by_category,
    plot_violins,
    compute_metric_by_condition,
    get_color_map,
    plot_cluster_heatmap_plotly,
    plot_volcanoes,
    get_volcano_classification_masks,
)
from components.domain_annotations import fetch_interpro_representative_domains

from utils.utils import logger, log_time

PELSA_LOCAL_STABILITY_COLORSCALE = [
    [0.00, "#73d055"],  # threshold entry: viridis-like yellow-green
    [0.35, "#2a788e"],  # blue-teal
    [0.70, "#355f8d"],  # blue
    [1.00, "#440154"],  # dark purple
]
PELSA_LOCAL_STABILITY_LOW_COLOR = "#9e9e9e"

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

def _resolve_protein_col_idx(ad, protein: str) -> int:
    """
    Resolve a clicked/searched protein token to a single column index in ad.var.

    Resolution order:
      1) exact var_names match
      2) exact GENE_NAMES match
      3) token match within grouped var_names  (split on ; , whitespace)
      4) token match within grouped GENE_NAMES (split on ; , whitespace)

    Raises KeyError with a clear message if no match is found.
    """
    token = str(protein).strip()
    if not token:
        raise KeyError("Empty protein token.")

    ids = ad.var_names.astype(str)

    # 1) exact UniProt / var_names match
    try:
        return int(ids.get_loc(token))
    except KeyError:
        pass

    if "GENE_NAMES" in ad.var.columns:
        names = ad.var["GENE_NAMES"].astype(str)

        # 2) exact gene-group match
        exact = np.flatnonzero(names.to_numpy() == token)
        if len(exact):
            return int(exact[0])

        splitter = re.compile(r"[;,\s]+")

        def _has_token(s: str) -> bool:
            return token in [x for x in splitter.split(str(s)) if x]

        # 3) tokenized var_names match
        id_hits = np.flatnonzero(
            np.fromiter((_has_token(s) for s in ids.to_numpy()), dtype=bool, count=ad.n_vars)
        )
        if len(id_hits):
            return int(id_hits[0])

        # 4) tokenized gene-group match
        gene_hits = np.flatnonzero(
            np.fromiter((_has_token(s) for s in names.to_numpy()), dtype=bool, count=ad.n_vars)
        )
        if len(gene_hits):
            return int(gene_hits[0])

    raise KeyError(f"Protein not found in overview detail lookup: {token}")

@log_time("Plotting barplot proteins per sample")
def plot_barplot_proteins_per_sample(
    adata,
    matrix_key: str = "normalized",
    bar_color: str = "teal",
    title: str = "Protein IDs by Sample and Category",
    width: int = 900,
    height: int = 500,
    sort_by: str = "sample",
    group_key: str = "CONDITION",
    group_label: str = "Condition",
) -> go.Figure:
    """
    Count proteins per sample and draw as a bar plot using the generic helper.
    """
    # call the generic bar helper

    fig = plot_stacked_proteins_by_category(
        adata,
        sort_by=sort_by,
        title=title,
        group_key=group_key,
        group_label=group_label,
    )

    return fig

@log_time("Plotting violins metrics per sample")
def plot_violin_cv_rmad_per_condition(
    adata,
    matrix_key: str = "normalized",
    title: str = "%CV / rMAD per Condition",
    width: int = 900,
    height: int = 800,
    group_key: str = "CONDITION",
    group_label: str = "Condition",
) -> list[go.Figure]:

    if group_key not in adata.obs.columns:
        raise KeyError(f"Missing adata.obs[{group_key!r}] required for metric grouping.")

    cv_dict = compute_metric_by_condition(adata, cond_key=group_key, metric="CV")
    rmad_dict = compute_metric_by_condition(adata, cond_key=group_key, metric="rMAD")

    # draw grouped violins
    labels = list(cv_dict.keys())
    color_map = get_color_map(labels,
                              palette=px.colors.qualitative.Plotly,
                              anchor="Total",
                              anchor_color="gray")

    cv_fig = plot_violins(
                 data=cv_dict,
                 colors=color_map,
                 title=f"%CV per {group_label}",
                 width=width,
                 height=height,
                 x_title=group_label,
                 y_title="%CV",
                 showlegend=False,
                 )

    # enforce that x-axis uses exactly this array, not an alphabetical sort:
    rmad_fig = plot_violins(
                   data=rmad_dict,
                   colors=color_map,
                   title=f"%rMAD per {group_label}",
                   x_title=group_label,
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
    min_nonimp_ft_per_cond: int=0,
    min_precursors: int=1,
    min_nrsc_alignment: float=1.50,
    highlight: str = None,
    highlight_group: str = None,
    color_by: str = None,
    contrast: str = None,
) -> go.Figure:
    # simply forward the SessionState + args into the pure util
    fig =  plot_volcanoes(
        state=state,
        contrast=contrast,
        data_type=data_type,
        sign_threshold=sign_threshold,
        height=height,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        min_nonimp_per_cond=min_nonimp_per_cond,
        min_nonimp_ft_per_cond=min_nonimp_ft_per_cond,
        min_precursors=min_precursors,
        min_nrsc_alignment=min_nrsc_alignment,
        highlight=highlight,
        highlight_group=highlight_group,
        color_by=color_by,
    )

    return fig

@log_time("Plotting Protein Barplots")
def plot_intensity_by_protein(state, contrast, protein, layer):
    ad = state.adata

    mode = str(ad.uns.get("preprocessing", {}).get("analysis_type", "")).lower()
    proteomics_mode = (mode in {"dia", "dda", "proteomics"})
    title_txt = "Protein Expression"
    if not proteomics_mode:
        title_txt = "Peptide Expression"

    if proteomics_mode:
        col = _resolve_protein_col_idx(ad, str(protein))
    else:
        col = list(map(str, ad.var_names)).index(str(protein))

    # pick normalized layer (fallback to .X)
    proc_data = ad.X
    intensity_scale = "Log Intensity"
    is_spectral = False

    if layer.value == "Raw":
        proc_data = ad.layers.get('raw')
        intensity_scale = "Intensity"
    elif layer.value == "Log (pre-norm)":
        proc_data = ad.layers.get('lognorm')
    elif layer.value == "Spectral Counts":
        proc_data = ad.layers.get('spectral_counts')
        intensity_scale = "Spectral Counts"
        is_spectral = True

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
        "imputed": (np.zeros_like(y_vals, dtype=bool) if is_spectral else imputed_mask),
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

    # Shorten sample names on x-axis (display only; data untouched)
    fig.update_xaxes(
        tickvals=sample_order,
        ticktext=_shorten_labels(sample_order),
    )

    # adjust legend: show each condition once, hide imputed from condition traces
    for trace in fig.data:
        trace.showlegend = False

    # re-add one dummy bar per condition
    for cond in [grp1, grp2]:
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=cond,
            marker_color=color_map[cond],
            showlegend=True
        ))

    # add a custom legend entry for imputed pattern
    if not is_spectral:
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
        title=dict(text=title_txt, x=0.5),
        legend_title_text="Conditions",
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )
    return fig

def get_protein_info(state, contrast, protein, layer):
    ad   = state.adata
    mode = str(ad.uns.get("preprocessing", {}).get("analysis_type", "")).lower()
    proteomics_mode = (mode in {"dia", "dda", "proteomics"})

    if proteomics_mode:
        col_idx = _resolve_protein_col_idx(ad, str(protein))
        uniprot_id = str(ad.var_names[col_idx])
        gene_names = (
            str(ad.var["GENE_NAMES"].iloc[col_idx])
            if "GENE_NAMES" in ad.var.columns
            else ""
        )
    else:
        uniprot_id = str(protein)
        col_idx = list(map(str, ad.var_names)).index(uniprot_id)
        gene_names = (
            str(ad.var["GENE_NAMES"].iloc[col_idx])
            if "GENE_NAMES" in ad.var.columns
            else ""
        )

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
        'gene_names': gene_names,
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
        # Default behavior (proteo/peptido): exact match on var_names
        # Phospho behavior: var_names are typically "<UNIPROT>|<SITE>" (e.g. Q96JB3|S584).
        # In that case, allow an uploaded UniProt accession to match ALL sites for that protein.

        # normalize items a tiny bit (isoforms)
        items_norm = set()
        for x in items:
            s = str(x).strip()
            if not s:
                continue
            items_norm.add(s)
            items_norm.add(s.split("-", 1)[0])  # Q03169-2 -> Q03169

        ids_arr = np.array(ids, dtype=str)

        # if any var_name looks phospho-like, match by prefix before '|'
        if np.any(np.char.find(ids_arr, "|") >= 0):
            parents = np.array([u.split("|", 1)[0] for u in ids_arr], dtype=str)
            mask = np.isin(ids_arr, list(items_norm)) | np.isin(parents, list(items_norm))
            return set(ids_arr[mask])

        return set(u for u in ids_arr if u in items_norm)

@log_time("Plotting Peptide Trends (centered)")
def plot_peptide_trends_centered(adata, uniprot_id: str, contrast: str) -> go.Figure:
    ## pull & slice matrices
    analysis_type = str(adata.uns.get("preprocessing", {}).get("analysis_type", "")).lower()
    proteomics_mode = analysis_type in {"dia", "dda", "proteomics"}
    src_key = "peptides" if proteomics_mode else "precursors"

    if src_key not in adata.uns:
        raise KeyError(
            f"Expected adata.uns['{src_key}'] for analysis_type='{analysis_type}', "
            f"but it is missing. Available keys: {sorted(list(adata.uns.keys()))}"
        )

    block = adata.uns[src_key]

    X_all   = np.asarray(block["centered"], dtype=float)     # (rows x samples)
    rows    = list(block["rows"])
    prot_ix = np.asarray(block["protein_index"], dtype=str)  # per-row protein id
    cols    = list(map(str, block["cols"]))                  # sample names in matrix

    if src_key == "precursors":
        seqs   = np.asarray(block["peptide_seq"], dtype=str)
        chg    = np.asarray(block["charge"], dtype=str)
        labels = np.array([f"{s}/+{c}" if not str(c).startswith("+") else f"{s}/{c}" for s, c in zip(seqs, chg)], dtype=object)
        title  = "Precursor trends"
        legend_name = "Precursor"
    else:
        seqs   = np.asarray(block["peptide_seq"], dtype=str)
        labels = seqs
        title  = "Peptide trends"
        legend_name = "Peptide"

    # keep only peptides belonging to this UniProt id (no group handling here)
    keep = (prot_ix == str(uniprot_id))
    X = X_all[keep, :]
    seqs = seqs[keep]
    labels = labels[keep]
    if src_key == "precursors":
        chg = chg[keep]

    # align columns to obs order, then filter to the contrast’s samples
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

    # build figure: one line per peptide, marker color by condition
    fig = go.Figure()
    palette = px.colors.qualitative.Prism
    uniq = sorted(pd.unique(labels).tolist())
    line_cmap = {s: palette[i % len(palette)] for i, s in enumerate(uniq)}

    cond_cmap = get_color_map([grp1, grp2], palette=px.colors.qualitative.Plotly)

    def _truncate_name(name, max_len=10):
        return name if len(name) <= max_len else name[:max_len - 1] + "…"

    for y, lab in zip(X, labels):
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
            name=_truncate_name(str(lab)),
            line=dict(color=line_cmap[lab], width=2, dash="dash"),
            marker=dict(size=singleton_sizes, color=line_cmap[lab]),
            customdata=np.c_[np.full_like(y, lab, dtype=object), cond_labels],
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Sample: %{x}<br>"
                          "Cond: %{customdata[1]}<br>"
                          "Value: %{y:.3f}<extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        #title=dict(text="Peptide trends", x=0.5),
        title=dict(text=title, x=0.5),
        xaxis_title="Sample",
        yaxis_title="Intensity / mean",
        margin=dict(l=60, r=40, t=40, b=50),
        legend_title_text=legend_name,
        shapes=[
            dict(
                type="line",
                xref="paper", yref="y",
                x0=0, x1=1,
                y0=1, y1=1,
                line=dict(
                    color="black",
                    width=1,
                    dash="dot"
                ),
            ),
        ],
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(sample_labels),
        ticktext=_shorten_labels(sample_labels),
    )
    return fig

@log_time("Plotting Cohort vs Non-cohort Violins")
def plot_group_violin_for_volcano(
    state,
    contrast: str,
    min_nonimp_per_cond: int,
    min_consistent_peptides: int,
    highlight_group: list[str] | set[str],
    show_measured: bool,
    show_imp_cond1: bool,
    show_imp_cond2: bool,
    width: int = 1200,
    height: int = 300,
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    ad = state.adata

    df_fc = pd.DataFrame(ad.varm["log2fc"], index=ad.var_names, columns=ad.uns["contrast_names"])
    grp1, grp2 = contrast.split("_vs_")

    # Get classification masks from shared helper
    measured, imp1, imp2 = get_volcano_classification_masks(
        ad,
        contrast=contrast,
        min_nonimp_per_cond=min_nonimp_per_cond,
        min_consistent_peptides=min_consistent_peptides,
    )
    visible = np.zeros_like(measured, dtype=bool)
    if show_measured:  visible |= measured
    if show_imp_cond1: visible |= imp1
    if show_imp_cond2: visible |= imp2

    # Now align x (log2FCs)
    x = df_fc.loc[ad.var_names, contrast]

    # Determine cohort membership by UniProt or Gene Name
    ids = np.array(ad.var_names, dtype=str)
    genes = np.array(ad.var["GENE_NAMES"].astype(str))
    group = set(map(str, (highlight_group or [])))
    in_group = np.array([(pid in group) or (g in group) for pid, g in zip(ids, genes)], dtype=bool)

    # Filter values
    cohort_vals = x[visible & in_group].to_numpy(dtype=float)
    rest_vals   = x[visible & ~in_group].to_numpy(dtype=float)

    # Empty plot if both sides are empty
    if cohort_vals.size == 0 and rest_vals.size == 0:
        fig = go.Figure()
        fig.update_layout(width=width, height=height, template="plotly_white",
                          margin=dict(l=60, r=40, t=10, b=40))
        return fig

    # X-axis range
    all_vals = np.concatenate([cohort_vals, rest_vals]) if rest_vals.size else cohort_vals
    xmin, xmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = max(0.05 * (xmax - xmin), 0.2)
    xmin -= pad; xmax += pad
    xr = [xmin, xmax] if x_range is None else list(x_range)

    # Colors
    labels = ["Cohort", "Non-cohort"]
    cmap = get_color_map(labels, palette=px.colors.qualitative.Plotly, anchor="Cohort", anchor_color="#6c5ce7")
    def _v(arr, name):
        colors = {"Cohort": "#6c5ce7", "Non-cohort": "#2a9d8f"}
        return go.Violin(
            y=[name]*len(arr),
            x=arr,
            orientation="h",
            name=name,
            line_color=colors[name],
            line_width=1,
            width=0.45,
            opacity=0.65,
            box_visible=True,
            meanline_visible=True,
            points=False,
            hoverinfo="skip",
            showlegend=False,
        )

    # Plot
    fig = go.Figure()
    if cohort_vals.size:
        fig.add_trace(_v(cohort_vals, "Cohort"))
    if rest_vals.size:
        fig.add_trace(_v(rest_vals, "Non-cohort"))

    # Median / SD annotations
    def _stats(arr):
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
        annotations=annos,
    )
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

def _pelsa_results_df(adata) -> pd.DataFrame:
    df = adata.uns.get("pelsa", {}).get("curve_results")
    if df is None:
        raise KeyError("Missing adata.uns['pelsa']['curve_results']")
    return df.copy()


def _pelsa_points_df(adata) -> pd.DataFrame:
    df = adata.uns.get("pelsa", {}).get("curve_points")
    if df is None:
        raise KeyError("Missing adata.uns['pelsa']['curve_points']")
    return df.copy()


def _pelsa_four_pl(x, pec50, slope, front, back):
    x = np.asarray(x, dtype=float)
    return back + (front - back) / (1.0 + np.power(10.0, slope * (x + pec50)))

def _pelsa_x_tick_labels(sub: pd.DataFrame) -> tuple[list[float], list[str]]:
    tick_df = sub[["concentration", "log10_concentration"]].copy()
    tick_df["concentration"] = pd.to_numeric(tick_df["concentration"], errors="coerce")
    tick_df["log10_concentration"] = pd.to_numeric(
        tick_df["log10_concentration"],
        errors="coerce",
    )
    tick_df = tick_df.dropna().drop_duplicates().sort_values("log10_concentration")

    tickvals = tick_df["log10_concentration"].to_numpy(dtype=float).tolist()
    ticktext = [
        "Control" if conc == 0 else f"{conc:g}"
        for conc in tick_df["concentration"].to_numpy(dtype=float)
    ]
    return tickvals, ticktext


def plot_pelsa_volcano(
    state,
    highlight: str = None,
    highlight_group=None,
    color_by: str = "Significance",
    sign_threshold: float = 0.05,
    hide_zero_neglog10_q: bool = False,
    ec50_range: tuple[float, float] | None = None,
    width: int = 900,
    height: int = 900,
) -> go.Figure:
    ad = state.adata
    res = _pelsa_results_df(ad)

    x = pd.to_numeric(res["curve_fold_change_log2"], errors="coerce")
    q = pd.to_numeric(res["curve_q_value"], errors="coerce")
    y = pd.to_numeric(res["curve_neglog10_q"], errors="coerce")
    pec50 = pd.to_numeric(res.get("pec50", np.nan), errors="coerce")
    log10_ec50 = -pec50
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ec50 = np.power(10.0, log10_ec50.to_numpy(dtype=float))
    ec50[~np.isfinite(ec50)] = np.nan


    ids = res["peptide_id"].astype(str).to_numpy()
    genes = (
        ad.var["GENE_NAMES"].astype(str).reindex(ids).fillna("").to_numpy()
        if "GENE_NAMES" in ad.var.columns
        else np.array([""] * len(ids), dtype=object)
    )

    fit_success = res["fit_success"].astype(bool).to_numpy()
    finite = fit_success & np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if hide_zero_neglog10_q:
        finite &= y.to_numpy(dtype=float) != 0.0

    mask = finite
    if ec50_range is not None:
        lo, hi = map(float, ec50_range)
        if lo > hi:
            lo, hi = hi, lo
        mask &= np.isfinite(ec50)
        mask &= ec50 >= lo
        mask &= ec50 <= hi

    sig = q.to_numpy(dtype=float) < float(sign_threshold)
    color_mode = str(color_by or "Significance")
    if color_mode == "Significance":
        color_vals = np.where(sig & (x.to_numpy() > 0), "red", np.where(sig & (x.to_numpy() < 0), "blue", "gray"))
        marker_color_kwargs = dict(color=color_vals[mask])
    elif color_mode == "EC50":
        marker_color_kwargs = dict(
            color=log10_ec50[mask],
            colorscale="Viridis",
            colorbar=dict(title="log<sub>10</sub> EC50"),
            showscale=True,
        )
    elif color_mode == "Avg Control Int.":
        pelsa = ad.uns.get("pelsa", {}) or {}
        concentration_col = str(pelsa.get("concentration_column", "")).strip()
        if not concentration_col or concentration_col not in ad.obs.columns:
            raise KeyError(
                "PELSA volcano control-intensity coloring requires "
                "adata.uns['pelsa']['concentration_column'] to point to an obs column."
            )

        control_concentration = pd.to_numeric(
            pelsa.get("control_concentration", 0),
            errors="coerce",
        )
        if not np.isfinite(control_concentration):
            control_concentration = 0.0

        obs_conc = pd.to_numeric(ad.obs[concentration_col], errors="coerce").to_numpy(dtype=float)
        control_mask = np.isclose(
            obs_conc,
            float(control_concentration),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=False,
        )
        if not np.any(control_mask):
            raise ValueError(
                "PELSA volcano control-intensity coloring found no samples at "
                f"control concentration={float(control_concentration):g}."
            )

        raw_layer = ad.layers.get("raw", ad.X)
        raw_mat = raw_layer.toarray() if hasattr(raw_layer, "toarray") else np.asarray(raw_layer)
        avg_control = np.nanmean(raw_mat[control_mask, :], axis=0)
        avg_control = pd.Series(avg_control, index=ad.var_names).reindex(ids).to_numpy(dtype=float)

        marker_color_kwargs = dict(
            color=np.log10(np.clip(avg_control, 0.0, None) + 1.0)[mask],
            colorscale="Viridis",
            colorbar=dict(title="log<sub>10</sub> Int"),
            showscale=True,
        )
    else:
        raise ValueError(f"Unsupported PELSA volcano color mode: {color_mode!r}")

    token = str(highlight or "").strip()
    is_high = np.zeros(len(ids), dtype=bool)
    if token:
        is_high = (ids == token) | (genes == token)

    group = set(map(str, highlight_group or []))
    in_group = np.array([(pid in group) or (g in group) for pid, g in zip(ids, genes)], dtype=bool)

    opacity = np.ones(len(ids), dtype=float)
    if is_high.any() and not in_group.any():
        opacity = np.where(is_high, 1.0, 0.08)
    elif in_group.any() and not is_high.any():
        opacity = np.where(in_group, 1.0, 0.05)
    elif in_group.any() and is_high.any():
        opacity = np.where(is_high, 1.0, np.where(in_group, 0.2, 0.05))

    size = np.full(len(ids), 6.0, dtype=float)
    size = np.where(is_high | in_group, 7.5, size)

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=x[mask],
        y=y[mask],
        mode="markers",
        marker=dict(
            size=size[mask],
            **marker_color_kwargs,
            opacity=opacity[mask],
            line=dict(width=0),
        ),
        text=ids[mask],
        customdata=np.c_[
            ids[mask],
            genes[mask],
            log10_ec50[mask].to_numpy(),
            ec50[mask],
        ],
        hovertemplate=(
            "Peptide: %{customdata[0]}<br>"
            "Gene: %{customdata[1]}<br>"
            "Curve range log₂: %{x:.3f}<br>"
            "-log10(q): %{y:.2f}<br>"
            "log10 EC50: %{customdata[2]:.3g}<br>"
            "EC50: %{customdata[3]:.3g}<br>"
        ),
        name="",
    ))

    thr_y = -np.log10(sign_threshold)

    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    xmin, xmax = (float(np.nanmin(xv)), float(np.nanmax(xv))) if xv.size else (-1.0, 1.0)
    ymax = float(np.nanmax(yv)) if yv.size else 1.0
    xpad = max((xmax - xmin) * 0.05, 0.2)

    # sign marker
    up = int(np.sum(mask & sig & (x.to_numpy(dtype=float) > 0)))
    down = int(np.sum(mask & sig & (x.to_numpy(dtype=float) < 0)))
    rest = int(np.sum(mask) - up - down)

    annos = [
        dict(x=0.02, y=0.98, xref="paper", yref="paper", opacity=0.7,
             text=f"<b>{down}</b>", bgcolor="blue", font=dict(color="white"), showarrow=False),
        dict(x=0.500, y=0.98, xref="paper", yref="paper",
             text=f"<b>{rest}</b>", bgcolor="lightgrey", font=dict(color="black"), showarrow=False),
        dict(x=0.98, y=0.98, xref="paper", yref="paper", opacity=0.7,
             text=f"<b>{up}</b>", bgcolor="red", font=dict(color="white"), showarrow=False),
    ]

    fig.update_layout(
        title=dict(text="PELSA Volcano", x=0.5),
        annotations=annos,
        height=height,
        margin=dict(l=60, r=120, t=60, b=60, autoexpand=False),
        showlegend=False,
        shapes=[
            dict(type="line", x0=xmin - xpad, x1=xmax + xpad, y0=thr_y, y1=thr_y,
                 line=dict(color="black", dash="dash")),
            dict(type="line", x0=0, x1=0, y0=0, y1=ymax,
                 line=dict(color="black", dash="dash")),
        ],
        xaxis=dict(title="Curve range log₂ ratio to control"),
        yaxis=dict(title="-log10(curve q-value)"),
    )
    return fig


def get_pelsa_info(state, peptide_id: str) -> dict:
    ad = state.adata
    peptide_id = str(peptide_id)
    res = _pelsa_results_df(ad).set_index("peptide_id")
    if peptide_id not in res.index:
        raise KeyError(f"PELSA peptide not found: {peptide_id}")

    def _ec50_from_pec50(value) -> float:
        pec50 = pd.to_numeric(value, errors="coerce")
        if not np.isfinite(pec50):
            return float("nan")
        # Avoid inf from pathological/unconstrained fits.
        exponent = -float(pec50)
        if exponent > 308 or exponent < -308:
            return float("nan")
        return float(10.0 ** exponent)

    row = res.loc[peptide_id]
    idx = list(map(str, ad.var_names)).index(peptide_id)

    return {
        "peptide_id": peptide_id,
        "index": idx,
        "gene_names": str(ad.var["GENE_NAMES"].astype(str).iloc[idx]) if "GENE_NAMES" in ad.var.columns else "",
        "protein": str(ad.var["FASTA_HEADERS"].astype(str).iloc[idx]) if "FASTA_HEADERS" in ad.var.columns else "",
        "parent_protein": str(ad.var["PARENT_PROTEIN"].astype(str).iloc[idx]) if "PARENT_PROTEIN" in ad.var.columns else "",
        "qval": float(row.get("curve_q_value", np.nan)),
        "pval": float(row.get("curve_p_value", np.nan)),
        "f_value": float(row.get("curve_f_value", np.nan)),
        "range_log2": float(row.get("curve_fold_change_log2", np.nan)),
        "rmse": float(row.get("rmse", np.nan)),
        "normalized_rmse": float(row.get("normalized_rmse", np.nan)),
        "r2": float(row.get("r2", np.nan)),
        "pec50": float(row.get("pec50", np.nan)),
        "ec50": _ec50_from_pec50(row.get("pec50", np.nan)),
        "pec50_ci_width_norm": float(row.get("pEC50_ci_width_norm", np.nan)),
        "pec50_ci_low": float(row.get("pEC50_ci_low", np.nan)),
        "pec50_ci_high": float(row.get("pEC50_ci_high", np.nan)),
        "slope": float(row.get("slope", np.nan)),
        "front": float(row.get("front", np.nan)),
        "back": float(row.get("back", np.nan)),
        "pEC50_inside_range": bool(row.get("pEC50_inside_range", False)),
    }


def plot_pelsa_curve(state, peptide_id: str, width: int = 800, height: int = 400) -> go.Figure:
    ad = state.adata
    peptide_id = str(peptide_id)

    pts = _pelsa_points_df(ad)
    sub = pts[pts["peptide_id"].astype(str) == peptide_id].copy()

    res = _pelsa_results_df(ad).set_index("peptide_id")
    if peptide_id not in res.index or sub.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=height, title="PELSA curve")
        return fig

    row = res.loc[peptide_id]

    x_vals = pd.to_numeric(sub["log10_concentration"], errors="coerce").to_numpy(dtype=float)
    finite_x = np.isfinite(x_vals)
    if not np.any(finite_x):
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=height, title="PELSA curve")
        return fig

    x_min = float(np.nanmin(x_vals[finite_x]))
    x_max = float(np.nanmax(x_vals[finite_x]))
    x_grid = np.linspace(x_min, x_max, 100)

    y_grid = _pelsa_four_pl(
        x_grid,
        float(row["pec50"]),
        float(row["slope"]),
        float(row["front"]),
        float(row["back"]),
    )

    fig = go.Figure()

    if "replicate" in sub.columns:
        sub["replicate_group"] = sub["replicate"].astype(str)
    elif "REPLICATE" in ad.obs.columns:
        sample_to_rep = ad.obs["REPLICATE"].astype(str).to_dict()
        sub["replicate_group"] = sub["sample"].map(sample_to_rep).astype(str)
    else:
        raise KeyError("Missing replicate information: expected curve_points['replicate'] or adata.obs['REPLICATE'].")

    reps = sorted(sub["replicate_group"].dropna().astype(str).unique().tolist())
    cmap = get_color_map(reps, palette=px.colors.qualitative.Plotly)

    for rep, g in sub.groupby("replicate_group", sort=False):
        g = g.copy()
        g["plot_ratio"] = pd.to_numeric(g["ratio"], errors="coerce")
        if "log2_ratio" in g.columns:
            g["plot_log2_ratio"] = pd.to_numeric(g["log2_ratio"], errors="coerce")
        else:
            g["plot_log2_ratio"] = np.log2(g["plot_ratio"])

        fig.add_trace(go.Scatter(
            x=g["log10_concentration"],
            y=g["plot_log2_ratio"],
            mode="markers",
            name=str(rep),
            marker=dict(size=8, color=cmap.get(str(rep), "gray"), line=dict(width=1, color="black")),
            customdata=np.c_[g["sample"], g["replicate_group"], g["plot_ratio"]],
            hovertemplate=(
                "Sample: %{customdata[0]}<br>"
                "Replicate: %{customdata[1]}<br>"
                "log10 conc: %{x:.3f}<br>"
                "Ratio: %{customdata[2]:.3f}<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_grid,
        mode="lines",
        name="4PL fit",
        line=dict(color="black", width=2),
        customdata=np.c_[np.exp2(y_grid)],
        hovertemplate=(
            "log10 conc: %{x:.3f}<br>"
            "Fitted log₂ ratio: %{y:.3f}<br>"
            "Fitted ratio: %{customdata[0]:.3f}<extra></extra>"
        ),
    ))

    fig.add_hline(y=0.0, line_dash="dot", line_color="black")

    tickvals, ticktext = _pelsa_x_tick_labels(sub)

    fig.update_layout(
        title=dict(text="PELSA titration curve", x=0.5),
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(l=60, r=40, t=50, b=60),
        xaxis_title="Concentration",
        yaxis_title="log₂ ratio to control",
        legend_title_text="Replicate",
    )
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    return fig


def _pelsa_parent_protein(adata, peptide_id: str) -> str:
    if "PARENT_PROTEIN" not in adata.var.columns:
        return ""
    try:
        parent = adata.var.loc[str(peptide_id), "PARENT_PROTEIN"]
    except KeyError:
        raise KeyError(f"PELSA peptide not found in adata.var: {peptide_id}")
    parent = str(parent).strip()
    if parent.lower() in {"", "nan", "none"}:
        return ""
    return parent


def get_pelsa_sister_peptides(state, peptide_id: str, sign_threshold: float = 0.05) -> pd.DataFrame:
    """Return same-parent PELSA peptides with curve metrics for the detail table."""
    ad = state.adata
    peptide_id = str(peptide_id)
    parent = _pelsa_parent_protein(ad, peptide_id)
    if not parent:
        return pd.DataFrame(columns=["peptide_id", "range_log2", "qval", "current"])

    res = _pelsa_results_df(ad).set_index("peptide_id")
    mask = ad.var["PARENT_PROTEIN"].astype(str) == parent
    siblings = ad.var_names[mask].astype(str).tolist()
    siblings = [p for p in siblings if p in res.index]
    if not siblings:
        return pd.DataFrame(columns=["peptide_id", "range_log2", "qval", "current"])

    out = pd.DataFrame(index=siblings)
    out["peptide_id"] = siblings
    out["range_log2"] = pd.to_numeric(res.loc[siblings, "curve_fold_change_log2"], errors="coerce").to_numpy(dtype=float)
    out["qval"] = pd.to_numeric(res.loc[siblings, "curve_q_value"], errors="coerce").to_numpy(dtype=float)
    pec50 = pd.to_numeric(res.loc[siblings, "pec50"], errors="coerce").to_numpy(dtype=float)
    exponent = -pec50
    ec50 = np.full_like(exponent, np.nan, dtype=float)
    ok = np.isfinite(exponent) & (exponent <= 308) & (exponent >= -308)
    ec50[ok] = np.power(10.0, exponent[ok])
    out["ec50"] = ec50
    out["current"] = out["peptide_id"].astype(str) == peptide_id
    out["__abs_range__"] = np.abs(out["range_log2"].to_numpy(dtype=float))
    out = out.sort_values(["__abs_range__", "qval", "peptide_id"], ascending=[False, True, True], kind="mergesort")
    return out.drop(columns=["__abs_range__"]).reset_index(drop=True)


def _empty_pelsa_profile(height: int, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=55, r=20, t=45, b=45),
        title=dict(text="Local stability profile", x=0.5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=message,
            showarrow=False,
            font=dict(size=12, color="black"),
        )],
    )
    return fig

def _pelsa_qvalue_significance_score(qvals: pd.Series) -> np.ndarray:
    """
    Convert q-values to a finite -log10(q) significance score for plotting.

    q <= 0 is not a valid q-value but can appear after numerical underflow.
    For display, clamp it to one order of magnitude above the largest finite
    positive score so it remains visibly high without crashing the color scale.
    """
    q = pd.to_numeric(qvals, errors="coerce").to_numpy(dtype=float)
    score = np.full(q.shape, np.nan, dtype=float)

    positive = np.isfinite(q) & (q > 0)
    score[positive] = -np.log10(q[positive])

    finite_score = score[np.isfinite(score)]
    replacement = float(np.nanmax(finite_score) + 1.0) if finite_score.size else 1.0
    score[np.isfinite(q) & (q <= 0)] = replacement
    score[~np.isfinite(score)] = 0.0

    return score


def _pelsa_local_stability_color_spec(
    scores: np.ndarray,
    *,
    sign_threshold: float,
    global_cmax: float,
) -> tuple[list[str], list[list[float | str]], float, float]:
    """Return per-peptide colors and the matching thresholded global colorscale."""
    scores = np.asarray(scores, dtype=float)
    sign_score = -np.log10(float(sign_threshold))
    cmax = max(float(global_cmax), sign_score)

    if cmax <= sign_score:
        colorscale = [
            [0.0, PELSA_LOCAL_STABILITY_LOW_COLOR],
            [1.0, PELSA_LOCAL_STABILITY_LOW_COLOR],
        ]
    else:
        threshold_pos = float(np.clip(sign_score / cmax, 0.0, 0.98))
        span = 1.0 - threshold_pos

        # Start fading out of gray slightly before q=0.05.
        # This makes q values just above 0.05 visually "near-threshold" rather
        # than indistinguishable from completely non-significant peptides.
        pre_threshold = min(0.05, threshold_pos * 0.20)
        gray_end = max(0.0, threshold_pos - pre_threshold)

        entry_pos = min(threshold_pos + span * 0.10, threshold_pos + span * 0.30)

        colorscale = [
            [0.0, PELSA_LOCAL_STABILITY_LOW_COLOR],
            [gray_end, PELSA_LOCAL_STABILITY_LOW_COLOR],
            [threshold_pos, "#8fb86a"],  # gray-green transition at q=0.05
            [entry_pos, PELSA_LOCAL_STABILITY_COLORSCALE[0][1]],
            [threshold_pos + span * 0.42, PELSA_LOCAL_STABILITY_COLORSCALE[1][1]],
            [threshold_pos + span * 0.72, PELSA_LOCAL_STABILITY_COLORSCALE[2][1]],
            [1.0, PELSA_LOCAL_STABILITY_COLORSCALE[3][1]],
        ]
        #colorscale = [
        #    [0.0, PELSA_LOCAL_STABILITY_LOW_COLOR],
        #    [threshold_pos, PELSA_LOCAL_STABILITY_LOW_COLOR],
        #    [min(threshold_pos + 1e-6, 1.0), PELSA_LOCAL_STABILITY_COLORSCALE[0][1]],
        #    [threshold_pos + span * 0.35, PELSA_LOCAL_STABILITY_COLORSCALE[1][1]],
        #    [threshold_pos + span * 0.70, PELSA_LOCAL_STABILITY_COLORSCALE[2][1]],
        #    [1.0, PELSA_LOCAL_STABILITY_COLORSCALE[3][1]],
        #]

    norm = np.clip(scores / cmax, 0.0, 1.0)
    colors = px.colors.sample_colorscale(colorscale, norm)
    return colors, colorscale, 0.0, cmax

def _pelsa_global_significance_cmax(adata, sign_threshold: float) -> float:
    """Dataset-level -log10(q) max for consistent local-stability coloring."""
    res = _pelsa_results_df(adata)
    if "curve_q_value" not in res.columns:
        raise KeyError("Missing curve_results['curve_q_value'] required for PELSA local-stability colors.")

    scores = _pelsa_qvalue_significance_score(res["curve_q_value"])
    scores = scores[np.isfinite(scores)]
    sign_score = -np.log10(float(sign_threshold))

    if scores.size == 0:
        return sign_score
    return max(sign_score, float(np.nanmax(scores)))


def _add_pelsa_domain_strip(
    fig: go.Figure,
    *,
    domains,
    protein_len: int,
    y_center: float,
    halfheight: float,
) -> None:
    """
    Add a compact InterPro representative-domain strip to the bottom of the
    local stability profile.

    Rectangles provide the visual layer; transparent line traces provide hover.
    """
    palette = (
        px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Plotly
    )

    y0 = y_center - halfheight
    y1 = y_center + halfheight

    # Empty/background strip. This keeps the visual layout stable and makes
    # "no representative/Pfam domains" look intentional rather than broken.
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0,
        x1=max(float(protein_len), 1.0),
        y0=y0,
        y1=y1,
        fillcolor="rgba(230,230,230,0.55)",
        line=dict(color="white", width=1),
        layer="below",
    )

    # Thin visual separator between peptide profile and domain strip.
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=0,
        x1=max(float(protein_len), 1.0),
        y0=y1 + halfheight * 0.65,
        y1=y1 + halfheight * 0.65,
        line=dict(color="white", width=2),
        layer="above",
    )

    if not domains:
        fig.add_annotation(
            x=max(float(protein_len), 1.0) * 0.5,
            y=y_center,
            xref="x",
            yref="y",
            text="No representative domains",
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(size=9, color="#777"),
        )

    for i, dom in enumerate(domains):
        start = max(1.0, float(dom.start))
        end = min(float(protein_len), float(dom.end))
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue

        color = palette[i % len(palette)]
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=start,
            x1=end,
            y0=y0,
            y1=y1,
            fillcolor=color,
            opacity=0.85,
            line=dict(color="white", width=1),
            layer="below",
        )

        n_hit = max(6, int(np.ceil((end - start) / 8.0)))
        hit_x = np.linspace(start, end, n_hit)
        hit_y = np.full(n_hit, y_center, dtype=float)
        customdata = np.array([[
            dom.name,
            dom.accession,
            dom.source,
            start,
            end,
        ]] * n_hit, dtype=object)

        fig.add_trace(go.Scatter(
            x=hit_x,
            y=hit_y,
            mode="lines+markers",
            line=dict(color="rgba(0,0,0,0)", width=max(10, halfheight * 18)),
            marker=dict(size=max(8, halfheight * 10), color="rgba(0,0,0,0.001)"),
            customdata=customdata,
            hovertemplate=(
                "Domain: %{customdata[0]}<br>"
                "Accession: %{customdata[1]}<br>"
                "Source: %{customdata[2]}<br>"
                "Position: %{customdata[3]:.0f}–%{customdata[4]:.0f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_annotation(
        x=0,
        y=y_center,
        xref="x",
        yref="y",
        text="Domains",
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        xshift=-6,
        font=dict(size=10, color="#555"),
    )


def plot_pelsa_local_stability_profile(
    state,
    peptide_id: str,
    sign_threshold: float = 0.05,
    width: int = 430,
    height: int = 260,
) -> go.Figure:
    """
    Plot same-parent PELSA peptide ranges along the estimated protein length.

    Optional metadata behavior:
    - If the exported position/length columns are absent, return an empty profile.
    - If they are present but malformed for the selected parent, raise ValueError.
    """
    ad = state.adata
    peptide_id = str(peptide_id)
    required = {"PARENT_PROTEIN", "PEPTIDE_START", "PEPTIDE_END", "PROTEIN_LENGTH_ESTIMATE_AA"}
    missing = sorted(required - set(ad.var.columns))
    if missing:
        return _empty_pelsa_profile(height, "Local stability metadata not available")

    parent = _pelsa_parent_protein(ad, peptide_id)
    if not parent:
        return _empty_pelsa_profile(height, "Parent protein not available")

    res = _pelsa_results_df(ad).set_index("peptide_id")
    mask = ad.var["PARENT_PROTEIN"].astype(str) == parent
    siblings = ad.var.loc[mask].copy()
    sibling_ids = siblings.index.astype(str).tolist()
    sibling_ids = [p for p in sibling_ids if p in res.index]
    if not sibling_ids:
        return _empty_pelsa_profile(height, "No sister peptide curve results")

    siblings = siblings.loc[sibling_ids].copy()
    for col in ["PEPTIDE_START", "PEPTIDE_END", "PROTEIN_LENGTH_ESTIMATE_AA"]:
        siblings[col] = pd.to_numeric(siblings[col], errors="coerce")

    bad = siblings[["PEPTIDE_START", "PEPTIDE_END"]].isna().any(axis=1)
    bad |= siblings["PEPTIDE_END"] < siblings["PEPTIDE_START"]
    if bad.any():
        examples = siblings.index[bad].astype(str).tolist()[:10]
        raise ValueError(
            "Invalid PELSA peptide-position metadata for local stability profile. "
            f"Examples={examples}"
        )

    length_vals = siblings["PROTEIN_LENGTH_ESTIMATE_AA"].dropna().astype(float)
    if length_vals.empty:
        return _empty_pelsa_profile(height, "Estimated protein length not available")
    protein_len = int(round(float(length_vals.max())))
    if protein_len <= 0:
        raise ValueError(f"Invalid PROTEIN_LENGTH_ESTIMATE_AA for parent {parent!r}: {protein_len}")

    domain_result = fetch_interpro_representative_domains(parent)
    domains = list(domain_result.domains or [])
    if domain_result.protein_length is not None and int(domain_result.protein_length) > 0:
        # Prefer real InterPro/UniProt sequence length when available, but never
        # shrink below observed peptide coordinates.
        protein_len = max(protein_len, int(domain_result.protein_length))

    plot_df = pd.DataFrame(index=sibling_ids)
    plot_df["start"] = siblings["PEPTIDE_START"].astype(float).to_numpy()
    plot_df["end"] = siblings["PEPTIDE_END"].astype(float).to_numpy()
    plot_df["range_log2"] = pd.to_numeric(res.loc[sibling_ids, "curve_fold_change_log2"], errors="coerce").to_numpy(dtype=float)
    plot_df["qval"] = pd.to_numeric(res.loc[sibling_ids, "curve_q_value"], errors="coerce").to_numpy(dtype=float)
    plot_df["gene"] = (
        ad.var.loc[sibling_ids, "GENE_NAMES"].astype(str).to_numpy()
        if "GENE_NAMES" in ad.var.columns
        else np.array([""] * len(sibling_ids), dtype=object)
    )
    plot_df["peptide_id"] = sibling_ids
    plot_df = plot_df[np.isfinite(plot_df["range_log2"].to_numpy(dtype=float))]
    if plot_df.empty:
        return _empty_pelsa_profile(height, "No finite local stability values")

    score = _pelsa_qvalue_significance_score(plot_df["qval"])
    global_cmax = _pelsa_global_significance_cmax(ad, sign_threshold)
    colors, colorscale, cmin, cmax = _pelsa_local_stability_color_spec(
        score,
        sign_threshold=sign_threshold,
        global_cmax=global_cmax,
    )

    fig = go.Figure()

    # Invisible marker trace used only to expose a continuous colorbar.
    tick_max = int(np.ceil(cmax))
    tickvals = list(range(0, tick_max + 1))
    ticktext = [str(x) for x in tickvals]
    if tickvals:
        ticktext[-1] = f"≥{tickvals[-1]}"

    fig.add_trace(go.Scatter(
        x=(plot_df["start"].to_numpy(dtype=float) + plot_df["end"].to_numpy(dtype=float)) / 2.0,
        y=plot_df["range_log2"].to_numpy(dtype=float),
        mode="markers",
        marker=dict(
            size=0.1,
            opacity=0.0,
            color=score,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title="-log<sub>10</sub>(q)",
                len=1.0,
                thickness=7,
                ticks="outside",
                dtick=1,
                ticklen=10,
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
    ))

    current = plot_df["peptide_id"].astype(str).to_numpy() == peptide_id
    current_ids = set(plot_df.loc[current, "peptide_id"].astype(str))

    yvals = plot_df["range_log2"].to_numpy(dtype=float)
    ymax = max(1.0, float(np.nanmax(np.abs(yvals))) * 1.15)
    peptide_halfheight = max(0.025 * ymax, 0.08)

    protein_len = max(protein_len, int(np.ceil(float(plot_df["end"].max()))))

    domain_halfheight = peptide_halfheight
    domain_y = -ymax - 2.25 * peptide_halfheight
    y_min = domain_y - 1.65 * domain_halfheight


    for (_, row), color in zip(plot_df.iterrows(), colors):
        peptide = str(row["peptide_id"])
        is_current = peptide in current_ids

        start = float(row["start"])
        end = float(row["end"])
        y = float(row["range_log2"])
        if end <= start:
            raise ValueError(
                "Invalid PELSA peptide-position metadata for local stability profile. "
                f"Peptide={peptide!r}, start={start}, end={end}."
            )

        q_score = -np.log10(row["qval"]) if row["qval"] > 0 else np.nan
        customdata = np.array([[
            peptide,
            row["gene"],
            row["qval"],
            q_score,
            start,
            end,
        ]] * 5, dtype=object)

        hovertemplate = (
            "Peptide: %{customdata[0]}<br>"
            "Gene: %{customdata[1]}<br>"
            "Position: %{customdata[4]:.0f}–%{customdata[5]:.0f}<br>"
            "Range log₂: " + f"{y:.3f}" + "<br>"
            "q-value: %{customdata[2]:.3e}<br>"
            "-log10(q): %{customdata[3]:.2f}<extra></extra>"
        )

        # Visual peptide block. Hover/click is handled by the transparent
        # hitbox trace below because Plotly fill-hover does not reliably
        # preserve customdata/hovertemplate.
        fig.add_trace(go.Scatter(
            x=[start, end, end, start, start],
            y=[
                y - peptide_halfheight,
                y - peptide_halfheight,
                y + peptide_halfheight,
                y + peptide_halfheight,
                y - peptide_halfheight,
            ],
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(color="red" if is_current else color, width=2 if is_current else 0),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Interaction hitbox. Sample points across the peptide interval so
        # hover/click works over the full segment instead of only at vertices.
        n_hit = max(6, int(np.ceil((end - start) / 4.0)))
        hit_x = np.linspace(start, end, n_hit)
        hit_y = np.full(n_hit, y, dtype=float)
        hit_customdata = np.array([customdata[0]] * n_hit, dtype=object)

        fig.add_trace(go.Scatter(
            x=hit_x,
            y=hit_y,
            mode="lines+markers",
            line=dict(color="rgba(0,0,0,0)", width=14),
            marker=dict(size=14, color="rgba(0,0,0,0.001)"),
            customdata=hit_customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
        ))

    _add_pelsa_domain_strip(
        fig,
        domains=domains,
        protein_len=protein_len,
        y_center=domain_y,
        halfheight=domain_halfheight,
    )

    title_parent = parent.split(";", 1)[0]
    fig.update_layout(
        title=dict(text=f"Local stability profile", x=0.5),
        height=height,
        width=width,
        margin=dict(l=55, r=20, t=50, b=45),
        xaxis=dict(title="Protein position (aa)", range=[0, max(protein_len, float(plot_df["end"].max()))]),
        yaxis=dict(title="Range log₂", range=[y_min, ymax], zeroline=True, zerolinecolor="black"),
    )
    return fig

