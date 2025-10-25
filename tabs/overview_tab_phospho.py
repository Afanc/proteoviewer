# overview_tab_phospho.py (refactored)
from __future__ import annotations

import os
import re
import textwrap
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models.widgets.tables import NumberFormatter
from session_state import SessionState

from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_covariate_by_site,
    plot_group_violin_for_volcano,
    plot_intensity_by_site,
    plot_peptide_trends_centered,  # kept import parity
    plot_violin_cv_rmad_per_condition,
    plot_volcanoes_wrapper,
    resolve_exact_list_to_uniprot_ids,  # kept import parity
    resolve_pattern_to_uniprot_ids,
    get_protein_info,  # used for UID helper
)
from components.plot_utils import plot_pca_2d, plot_umap_2d
from components.string_links import get_string_link
from components.texts import intro_preprocessing_text, log_transform_text  # keep parity
from layout_utils import (
    FRAME_STYLES,
    make_hr,
    make_vr,
)
from utils import log_time, logger


# ---------- Small utilities (safe to hoist later into helpers modules) ----------

def _fmt_files_list(paths: Iterable[str], max_items: int = 6) -> list[str]:
    paths = list(paths or [])
    if not paths:
        return []
    head = [f"  - {os.path.basename(str(p))}" for p in paths[:max_items]]
    rest = len(paths) - max_items
    if rest > 0:
        head.append(f"  - … (+{rest} more)")
    return head


def _pn_only(site_id: str) -> str:
    m = re.search(r"\|(p\d+)", str(site_id))
    return m.group(1) if m else ""


def _peptide_from_site(site_id: str) -> str:
    return str(site_id).split("|", 1)[0]


def _short_mid(s: str, head: int = 12, tail: int = 6, sep: str = "…") -> str:
    s = str(s)
    return s if len(s) <= head + tail + 1 else f"{s[:head]}{sep}{s[-tail:]}"


@lru_cache(maxsize=4096)
def _cached_string_link(uniprot_id: str) -> str:
    try:
        return get_string_link(uniprot_id) or ""
    except Exception:
        return ""


def _first_token(x: str) -> str:
    if isinstance(x, str) and ";" in x:
        return x.split(";", 1)[0].strip()
    return x


def _fmt_int(v) -> str:
    v = int(v)
    return f"{v:,}"


def _fmt_ibaq(v) -> str:
    v = float(v)
    if v >= 1_000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.2e}"


def _fmt_step(step: dict, default_thr: str) -> tuple[str, str]:
    if not step or step.get("skipped"):
        return "Skipped", "n/a"
    return f"{step.get('number_dropped', 0):,} PSM removed", str(step.get("threshold", default_thr))


def _pep_dir_symbol(step: dict) -> str:
    if not step:
        return "≤"
    d = str(step.get("direction", "")).lower()
    return "≥" if d.startswith("greater") else "≤"


# ---------- Cached accessors bound to a specific AnnData ----------

def _make_adata_views(adata):
    var_index = pd.Index(adata.var_names, name="INDEX")
    contrast_names = tuple(adata.uns["contrast_names"])

    def _df_opt(varm_key):
        """Return DataFrame if varm_key exists, else None."""
        arr = adata.varm.get(varm_key, None)
        if arr is None:
            return None
        return pd.DataFrame(arr, index=var_index, columns=contrast_names)
    # Required (always written by limma)
    df_log2fc = _df_opt("log2fc");  assert df_log2fc is not None, "Missing varm['log2fc']"
    df_q      = _df_opt("q_ebayes"); assert df_q      is not None, "Missing varm['q_ebayes']"
    # Optional: fall back to adjusted if raw_* are absent (no covariate run)
    df_raw_fc = _df_opt("raw_log2fc")
    if df_raw_fc is None:
        df_raw_fc = df_log2fc
    df_raw_q = _df_opt("raw_q_ebayes")
    if df_raw_q is None:
        df_raw_q = df_q

    # Optional FT/covariate outputs
    df_ft_q     = _df_opt("ft_q_ebayes")
    df_ft_fc    = _df_opt("ft_log2fc")
    df_cov_part = _df_opt("cov_part")
#def _make_adata_views(adata):
#    """
#    Create fast, cached views and lambdas bound to this AnnData snapshot.
#    All returned callables close over `adata` to avoid re-locating arrays.
#    """
#    contrast_names = tuple(adata.uns["contrast_names"])
#    var_index = adata.var_names
#
#    # varm matrices (DataFrame-backed for convenient .loc)
#    def _df(varm_key: str, fallback=None) -> pd.DataFrame:
#        arr = adata.varm.get(varm_key, fallback)
#        if arr is None:
#            raise KeyError(f"Missing adata.varm['{varm_key}']")
#        return pd.DataFrame(arr, index=var_index, columns=contrast_names)

#    df_log2fc = _df("log2fc")
#    df_q = _df("q_ebayes")
#    df_raw_q = _df("raw_q_ebayes")
#    df_raw_fc = _df("raw_log2fc", fallback=np.zeros_like(df_log2fc.values))
#    df_ft_q = _df("ft_q_ebayes")
#    df_ft_fc = _df("ft_log2fc")
#    df_cov_part = _df("cov_part", fallback=np.full_like(df_log2fc.values, np.nan))

    # layers frequently queried
    spectral_counts = pd.DataFrame(
        adata.layers["spectral_counts"], index=adata.obs_names, columns=var_index
    )

    # optional layer
    locprob = adata.layers.get("locprob", None)
    df_locprob = None
    if locprob is not None:
        df_locprob = pd.DataFrame(locprob, index=adata.obs_names, columns=var_index)

    # fast column getters
    var_cols = adata.var.columns

    def vcol(name: str) -> Optional[pd.Series]:
        return adata.var[name] if name in var_cols else None

    # cached indexing for site → integer position
    @lru_cache(maxsize=65536)
    def site_idx(site_id: str) -> int:
        return int(var_index.get_loc(site_id))

    return {
        "contrast_names": contrast_names,
        "df_log2fc": df_log2fc,
        "df_q": df_q,
        "df_raw_q": df_raw_q,
        "df_raw_fc": df_raw_fc,
        "df_ft_q": df_ft_q,
        "df_ft_fc": df_ft_fc,
        "df_cov_part": df_cov_part,
        "spectral_counts": spectral_counts,
        "df_locprob": df_locprob,
        "vcol": vcol,
        "site_idx": site_idx,
    }


def _build_pipeline_summary(adata) -> str:
    preproc = adata.uns.get("preprocessing", {})
    analysis = adata.uns.get("analysis", {})

    filtering = preproc.get("filtering", {})
    normalization = preproc.get("normalization", {})
    imputation = preproc.get("imputation", {})

    num_samples = len(adata.obs.index.unique())
    num_conditions = len(adata.obs["CONDITION"].unique())
    num_contrasts = int(num_conditions * (num_conditions - 1) / 2)

    analysis_type = preproc.get("analysis_type", "DIA")
    ebayes_method = analysis.get("ebayes_method", "limma")
    input_layout = preproc.get("input_layout", "")
    quant_method = preproc.get("quantification_method", "sum")

    cont_step = filtering.get("cont", {})
    q_step = filtering.get("qvalue", {})
    pep_step = filtering.get("pep", {})
    rec_step = filtering.get("rec", {})

    cont_txt, _ = _fmt_step(cont_step, "n/a")
    q_txt, q_thr_txt = _fmt_step(q_step, "n/a")
    pep_txt, pep_thr_txt = _fmt_step(pep_step, "n/a")
    rec_txt, rec_thr_txt = _fmt_step(rec_step, "n/a")
    pep_op = _pep_dir_symbol(pep_step)

    contaminants_files = [os.path.basename(p) for p in cont_step.get("files", [])]

    # Normalization methods
    norm_methods = normalization.get("method", []).tolist()
    if "loess" in norm_methods:
        loess_span = normalization.get("loess_span")
        i = norm_methods.index("loess")
        norm_methods[i] = f"loess (loess_span={loess_span})"
    if "median_equalization_by_tag" in norm_methods:
        tags = normalization.get("reference_tag")
        tags = [tags] if isinstance(tags, str) else tags
        tag_matches = normalization.get("tag_matches")
        i = norm_methods.index("median_equalization_by_tag")
        norm_methods[i] = f"median_equalization_by_tag (tags={tags}, matches={tag_matches})"
    norm_methods_str = ", ".join(norm_methods)

    # Imputation description
    imp_method = str(imputation.get("method", ""))
    extras = []
    if "knn" in imp_method:
        if "knn_k" in imputation:
            extras.append(f"k={imputation['knn_k']}")
        if "tnknn" in imp_method and "knn_tn_perc" in imputation:
            extras.append(f"tn_perc={imputation['knn_tn_perc']}")
    if "rf" in imp_method and "rf_max_iter" in imputation:
        extras.append(f"rf_max_iter={imputation['rf_max_iter']}")
    if "lc_conmed" in imp_method and "lc_conmed_lod_k" in imputation:
        extras.append(f"lod_k={imputation['lc_conmed_lod_k']}")
    if extras:
        imp_method = f"{imp_method} ({', '.join(extras)})"

    return textwrap.dedent(
        f"""
        **Analysis Type**: {analysis_type}

        {num_samples} Samples - {num_conditions} Conditions - {num_contrasts} Contrasts

        **Input Layout**: {input_layout}

        **Pipeline steps**
        - **Quantification**: {quant_method}
        - **Filtering**:
            - Contaminants ({', '.join(contaminants_files)}): {cont_txt}
            - q-value ≤ {q_thr_txt}: {q_txt}
            - PEP {pep_op} {pep_thr_txt}: {pep_txt}
            - Min. run evidence count = {rec_thr_txt}: {rec_txt}
        - **Normalization**: {norm_methods_str}
        - **Imputation**: {imp_method}
        - **Differential expression**: eBayes via {ebayes_method}
        """
    ).strip()


def _ensure_gene(state: SessionState, token: str | None) -> str | None:
    """
    Map `token` to a GENE_NAMES entry when possible.
    Falls back to UID→gene when token matches a var_name.
    """
    if not token:
        return None
    ad = state.adata
    names = ad.var["GENE_NAMES"].astype(str)
    t = str(token)

    # fast path: exact gene
    # set() once for speed; memoize across calls using lru_cache if needed
    if not hasattr(_ensure_gene, "_geneset"):
        setattr(_ensure_gene, "_geneset", set(names))
    if t in getattr(_ensure_gene, "_geneset"):
        return t

    try:
        idx = ad.var_names.get_loc(t)
        gene = names.iloc[idx]
        return gene.split(";", 1)[0].strip() if isinstance(gene, str) else str(gene)
    except KeyError:
        return t


# ---------- Main Tab ----------

@log_time("Preparing Overview Tab")
def overview_tab_phospho(state: SessionState):
    """
    Overview Tab:
      - Info on samples and filtering, config
      - Hists of Ids, CVs
      - PCA/UMAP
      - Volcanoes + detail panes
    """
    adata = state.adata
    views = _make_adata_views(adata)
    has_cov = bool(adata.uns["has_covariate"])
    contrast_names = list(views["contrast_names"])

    # ---------- Summary / Intro ----------
    summary_md = _build_pipeline_summary(adata)
    summary_pane = pn.pane.Markdown(
        summary_md,
        sizing_mode="stretch_width",
        margin=(-10, 0, 0, 20),
        styles={"line-height": "1.4em", "word-break": "break-word", "overflow-wrap": "anywhere", "min-width": "0"},
    )

    id_sort_toggle = pn.widgets.RadioButtonGroup(
        name="Order",
        options=["By condition", "By sample"],
        value="By condition",
        button_type="default",
        sizing_mode="fixed",
        width=170,
        margin=(20, 0, 0, 20),
        styles={"z-index": "10"},
    )

    def _sort_arg(mode: str) -> str:
        return "condition" if mode == "By condition" else "sample"

    hist_ID_dmap = pn.bind(
        plot_barplot_proteins_per_sample,
        adata=adata,
        sort_by=pn.bind(_sort_arg, id_sort_toggle),
    )

    hist_plot_pane = pn.pane.Plotly(
        hist_ID_dmap,
        height=500,
        margin=(-20, 20, 0, -190),
        styles={"flex": "1"},
    )

    intro_pane = pn.Row(
        pn.Column(pn.pane.Markdown("##   Summary"), summary_pane, styles={"flex": "0.32", "min-width": "0"}),
        make_vr(),
        pn.Spacer(width=20),
        id_sort_toggle,
        hist_plot_pane,
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={"border-radius": "15px", "box-shadow": "3px 3px 5px #bcbcbc", "width": "98vw"},
    )

    # ---------- Metrics ----------
    cv_fig, rmad_fig = plot_violin_cv_rmad_per_condition(adata)
    rmad_pane = pn.pane.Plotly(rmad_fig, height=500, sizing_mode="stretch_width",
                               styles={"flex": "1"}, config={'responsive': True}, margin=(0, 0, 0, -100))
    cv_pane = pn.pane.Plotly(cv_fig, height=500, sizing_mode="stretch_width",
                             styles={"flex": "1"}, config={'responsive': True})

    metrics_pane = pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex": "0.1", "z-index": "10"}),
        rmad_pane,
        pn.Spacer(width=25),
        make_vr(),
        pn.Spacer(width=25),
        cv_pane,
        pn.Spacer(width=50),
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={"border-radius": "15px", "box-shadow": "3px 3px 5px #bcbcbc", "width": "98vw"},
    )

    # ---------- Clustering ----------
    pca_pane = pn.pane.Plotly(
        plot_pca_2d(adata),
        height=500,
        sizing_mode="stretch_width",
        styles={"flex": "1"},
        margin=(0, 0, 0, -100),
    )
    umap_pane = pn.pane.Plotly(
        plot_umap_2d(adata),
        height=500,
        sizing_mode="stretch_width",
        styles={"flex": "1"},
    )
    clustering_pane = pn.Row(
        pn.pane.Markdown("##   Clustering", styles={"flex": "0.1", "z-index": "10"}),
        pca_pane,
        make_vr(),
        pn.Spacer(width=60),
        umap_pane,
        make_vr(),
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={"border-radius": "15px", "box-shadow": "3px 3px 5px #bcbcbc", "width": "98vw"},
    )

    # ---------- Volcanoes ----------
    contrasts = contrast_names
    contrast_sel = pn.widgets.Select(name="Contrast", options=contrasts, value=contrasts[0], width=180)

    # which volcano to plot
    volcano_src_opts = (["Phospho (raw)"] if not has_cov
                        else ["Phospho (raw)", "Phospho (adj.)", "Flowthrough"])
    volcano_src_sel = pn.widgets.Select(name="Volcano source", options=volcano_src_opts, value=volcano_src_opts[0], width=130)

    def _volcano_dtype(label: str) -> str:
        return {
            "Phospho (raw)"      : "phospho",   # raw_* columns
            "Phospho (adj.)" : "default",   # adjusted columns
            "Flowthrough"   : "flowthrough",  # ft_* columns
        }[label]

    show_measured = pn.widgets.Checkbox(name="Observed in Both", value=True)
    show_imp_cond1 = pn.widgets.Checkbox(name="", value=True)
    show_imp_cond2 = pn.widgets.Checkbox(name="", value=True)

    color_options = ["Significance"] + (["Raw LogFC", "Adj. LogFC", "FT LogFC"] if has_cov else [])
    color_by = pn.widgets.Select(name="Color by", options=color_options, value=color_options[0], width=120)

    def _update_toggle_labels(_=None):
        grp1, grp2 = contrast_sel.value.split("_vs_")
        show_imp_cond1.name = f"▲ Fully Imputed in {grp1}"
        show_imp_cond2.name = f"▼ Fully Imputed in {grp2}"

    _update_toggle_labels()
    contrast_sel.param.watch(_update_toggle_labels, "value")

    num_rep = max(adata.obs["REPLICATE"])
    min_meas_options = {f"≥{i}": i for i in range(1,num_rep+1)}
    min_meas_sel = pn.widgets.Select(name="Min / condition",
                                     options=list(min_meas_options.keys()), value="≥1", width=80)

    def _min_meas_value(label: str) -> int:
        return min_meas_options[label]

    # Min numb. precursors options
    min_prec_options = {f"≥{i}": i for i in range(1, 6)}
    min_prec_sel = pn.widgets.Select(
        name="Precursors",
        options=list(min_prec_options.keys()),
        value="≥1",
        width=80,
    )

    def _min_prec_value(label: str) -> int:
        return min_prec_options[label]


    search_input = pn.widgets.AutocompleteInput(
        name="Search Phosphosite",
        options=list(adata.var_names),
        width=200,
        case_sensitive=False,
    )
    clear_search = pn.widgets.Button(name="Clear", width=80)
    clear_search.on_click(lambda _e: setattr(search_input, "value", ""))

    search_field_sel = pn.widgets.Select(
        name="Search Field",
        options=["FASTA headers", "Gene names", "UniProt IDs"],
        value="FASTA headers",
        width=130,
        styles={"z-index": "10"},
        margin=(2, 0, 0, 0),
    )

    search_input_group = pn.widgets.AutocompleteInput(
        name="Search Gene/Protein",
        placeholder="Gene name or Uniprot ID",
        options=list(set(adata.var.get("GENE_NAMES", pd.Series(dtype=str)))) +
                list(set(adata.var.get("PARENT_PROTEIN", pd.Series(dtype=str)))),
        case_sensitive=False,
        width=180,
        styles={"z-index": "10"},
        margin=(2, 0, 0, 0),
    )

    def _group_ids(pattern, field):
        try:
            return sorted(resolve_pattern_to_uniprot_ids(adata, field, pattern))
        except Exception:
            return []

    group_ids_dmap = pn.bind(_group_ids, search_input_group, search_field_sel)

    def _fmt_status(ids_pat, pat_text):
        if not (pat_text and str(pat_text).strip()):
            return ""
        n = len(ids_pat or [])
        label = "match" if n == 1 else "matches"
        return f"**{n} {label}**"

    status_md = pn.bind(_fmt_status, group_ids_dmap, search_input_group)
    status_pane = pn.pane.Markdown(status_md, margin=(-10, 0, 0, 0), align="center")

    def _has_query(pat_text):
        return bool((pat_text or "").strip())

    status_pane.visible = pn.bind(_has_query, search_input_group)

    clear_all = pn.widgets.Button(name="Clear", width=80)
    clear_all.on_click(lambda _e: setattr(search_input_group, "value", ""))

    volcano_dmap = pn.bind(
        plot_volcanoes_wrapper,
        state=state,
        contrast=contrast_sel,
        #data_type="phospho",
        #data_type=pn.bind(_volcano_dtype, volcano_src_sel),
        data_type=(lambda: "phospho") if not has_cov else pn.bind(_volcano_dtype, volcano_src_sel),
        color_by=color_by,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        min_nonimp_per_cond=pn.bind(_min_meas_value, min_meas_sel),
        min_precursors=pn.bind(_min_prec_value, min_prec_sel),
        highlight=search_input,
        highlight_group=group_ids_dmap,
        sign_threshold=0.05,
        width=None,
        height=950,
    )

    def _on_volcano_click(event):
        click = event.new
        if click and click.get("points"):
            gene = click["points"][0]["text"]
            search_input.value = gene

    def _with_uirevision(fig, contrast):
        fig.update_layout(uirevision=f"volcano-{contrast}")
        return fig

    volcano_dmap_wrapped = pn.bind(_with_uirevision, volcano_dmap, contrast_sel)

    volcano_plot = pn.pane.Plotly(
        volcano_dmap_wrapped,
        height=950,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        config={"responsive": True},
        styles={"border-radius": "8px", "box-shadow": "3px 3px 5px #bcbcbc", "flex": "1"},
    )
    volcano_plot.param.watch(_on_volcano_click, "click_data")

    # ---------- Cohort violin ----------
    def _cohort_violin(ids, contrast, sm, s1, s2):
        if not ids:
            return pn.Spacer(height=0)
        fig = plot_group_violin_for_volcano(
            state=state,
            contrast=contrast,
            highlight_group=ids,
            show_measured=sm,
            show_imp_cond1=s1,
            show_imp_cond2=s2,
            width=1200,
            height=100,
        )
        return pn.pane.Plotly(
            fig,
            height=150,
            margin=(-10, 0, 10, 20),
            sizing_mode="stretch_width",
            config={"responsive": True},
            styles={"border-radius": "8px", "box-shadow": "3px 3px 5px #bcbcbc"},
        )

    cohort_violin_view = pn.bind(
        _cohort_violin, group_ids_dmap, contrast_sel, show_measured, show_imp_cond1, show_imp_cond2
    )

    # ---------- Detail panes (cards + bars + peptide table) ----------
    layers_phos = ["Processed", "Log-only", "Raw"]
    layers_cov = ["Processed", "Log-only", "Raw"]
    layers_phos_sel = pn.widgets.Select(name="Phospho View", options=layers_phos, value=layers_phos[0],
                                        width=100, margin=(20, 0, 0, 0))
    layers_cov_sel = pn.widgets.Select(name="Flowthrough View", options=layers_cov, value=layers_cov[0],
                                       width=100, margin=(20, 0, 0, 0), visible=has_cov)

    def _toggle_layers_visibility(event):
        visible = bool(event.new)
        layers_phos_sel.visible = visible
        layers_cov_sel.visible = visible

    search_input.param.watch(_toggle_layers_visibility, "value")
    layers_phos_sel.visible = bool(search_input.value)
    layers_cov_sel.visible = bool(search_input.value)

    main_info_holder = pn.Column()
    cov_info_holder = pn.Column(visible=has_cov)
    pep_table_holder = pn.Column()
    main_bar_holder = pn.Column()
    cov_bar_holder = pn.Column(visible=has_cov)

    detail_panel = pn.Row(
        pn.Column(
            main_info_holder,
            pn.Spacer(height=20),
            pn.Row(cov_info_holder, pn.Spacer(width=20, visible=has_cov), pep_table_holder, sizing_mode="fixed", width=800),
            pn.Spacer(height=45),
            main_bar_holder,
            pn.Spacer(height=20),
            cov_bar_holder,
            sizing_mode="fixed",
            width=840,
        ),
        margin=(0, 0, 0, 0),
        sizing_mode="fixed",
        styles={"margin-left": "auto"},
    )

    bokeh_doc = pn.state.curdoc

    def _render_main_info():
        site_id = search_input.value
        if not site_id:
            return pn.Spacer(width=800, height=170)
        try:
            idx = views["site_idx"](site_id)
        except KeyError:
            return pn.Spacer(width=800, height=170)

        df_q = views["df_q"]
        df_fc = views["df_log2fc"]
        df_raw_q = views["df_raw_q"]
        df_raw_fc = views["df_raw_fc"]
        df_cov_part = views["df_cov_part"]
        df_locprob = views["df_locprob"]
        vcol = views["vcol"]

        contrast = contrast_sel.value
        q_adj = float(df_q.loc[site_id, contrast])
        lfc_adj = float(df_fc.loc[site_id, contrast])
        q_raw = float(df_raw_q.loc[site_id, contrast])
        lfc_raw = float(df_raw_fc.loc[site_id, contrast])
        cov_piece = float(df_cov_part.loc[site_id, contrast]) if df_cov_part is not None else float("nan")

        loc_score = "n/a"
        if df_locprob is not None:
            col = df_locprob.columns[idx]
            loc_score = float(np.nanmax(df_locprob[col].values))

        parent_pep = str(vcol("PARENT_PEPTIDE_ID").iloc[idx]) if vcol("PARENT_PEPTIDE_ID") is not None else "n/a"
        parent_prot = str(vcol("PARENT_PROTEIN").iloc[idx]) if vcol("PARENT_PROTEIN") is not None else "n/a"

        # Precursors (site, mean spectral counts)
        pep_num_prec = int(np.nanmax(views["spectral_counts"].loc[:, site_id]))

        uid_for_link = _first_token(parent_prot) or ""
        string_link = _cached_string_link(uid_for_link)

        Number = pn.indicators.Number
        raw_q_ind = Number(name="Raw q-value", value=q_raw, format="{value:.3e}", default_color="red",
                           font_size="12pt", styles={"flex": "1"})
        raw_lfc_ind = Number(name="Raw log₂ FC", value=lfc_raw, format="{value:.3f}", default_color="red",
                             font_size="12pt", styles={"flex": "1"})
        loc_ind = Number(name="Max(Loc. Score)", value=loc_score, format="{value:.3f}", default_color="gray",
                         font_size="12pt", styles={"flex": "1"})
        if has_cov:
            q_ind = Number(name="FT-adj. q-value", value=q_adj, format="{value:.3e}", default_color="purple",
                           font_size="12pt", styles={"flex": "1"})
            lfc_ind = Number(name="FT-adj. log₂ FC", value=lfc_adj, format="{value:.3f}", default_color="purple",
                             font_size="12pt", styles={"flex": "1"})
            covp_ind = Number(name="Covariate part", value=cov_piece, format="{value:.3f}", default_color="gray",
                          font_size="12pt", styles={"flex": "1"})

        pretty_id = re.sub(r"(\w+)\|(p\d+)", r"\1 | \2", site_id)
        header = pn.Row(
            pn.pane.Markdown(f"**Phosphosite**: {pretty_id}",
                             styles={"font-size": "16px", "padding": "0", "line-height": "0px"}),
            sizing_mode="stretch_width",
            height=50,
            styles={
                "display": "flex",
                "background": "#f9f9f9",
                "margin": "0px",
                "padding": "0px",
                "border-bottom": "1px solid #ddd",
            },
        )

        footer_left = pn.pane.HTML(
            f"<span style='font-size: 12px;'>Precursors (site): <b>{pep_num_prec}</b></span>"
        )
        footer_right = pn.pane.HTML(
            (
                "<span style='font-size: 12px;'>"
                f"🔗 <a href='https://www.uniprot.org/uniprotkb/{uid_for_link}/entry' "
                "target='_blank' rel='noopener'>UniProt Entry</a>"
                " &nbsp;|&nbsp; "
                f"<a href='{string_link}' target='_blank' rel='noopener'>STRING Entry</a>"
                "</span>"
            )
        )
        footer_links = pn.Row(
            footer_left,
            sizing_mode="stretch_width",
            styles={"justify-content": "space-between", "padding": "2px 8px 4px 0px", "margin-top": "-6px"},
        )

        card_style = {
            "background": "#f9f9f9",
            "align-items": "center",
            "border-radius": "8px",
            "text-align": "center",
            "padding": "5px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "justify-content": "space-evenly",
        }

        #card = pn.Card(
        #    header,
        #    pn.Row(raw_q_ind, raw_lfc_ind, loc_ind, sizing_mode="stretch_width"),
        #    make_hr(),
        #    #pn.Row(q_ind, lfc_ind, covp_ind, sizing_mode="stretch_width"),
        #    #make_hr(),
        #    footer_links,
        #    width=800,
        #    styles=card_style,
        #    collapsible=False,
        #    hide_header=True,
        #)
        rows = [header,
                pn.Row(raw_q_ind, raw_lfc_ind, loc_ind, sizing_mode="stretch_width"),
                make_hr()]
        # Add FT-adjusted row only if covariate outputs exist
        if has_cov:
            rows += [pn.Row(q_ind, lfc_ind, covp_ind, sizing_mode="stretch_width"),
                     make_hr()]
#        rows = [header,
#                pn.Row(q_ind, lfc_ind, covp_ind, sizing_mode="stretch_width"),
#                make_hr()]
#        if has_cov:
#            rows += [pn.Row(q_ind, lfc_ind, covp_ind, sizing_mode="stretch_width"),
#                     make_hr()]
        card = pn.Card(
            header,
            *rows,
            footer_links,
            width=800,
            styles=card_style,
            collapsible=False,
            hide_header=True,
        )

        return card

    def _render_cov_info():
        if not has_cov:
            return pn.Spacer(width=800, height=120)
        site_id = search_input.value
        if not site_id:
            return pn.Spacer(width=800, height=120)
        try:
            idx = views["site_idx"](site_id)
        except KeyError:
            return pn.Spacer(width=800, height=120)

        df_ft_q = df_ft_qc = df_cov_part = None
        if has_cov:
            df_ft_q = views["df_ft_q"]
            df_ft_fc = views["df_ft_fc"]
            df_cov_part = views["df_cov_part"]
        vcol = views["vcol"]

        contrast = contrast_sel.value
        q_ft = float(df_ft_q.loc[site_id, contrast])
        lfc_ft = float(df_ft_fc.loc[site_id, contrast])

        cov_piece = float(df_cov_part.loc[site_id, contrast]) if df_cov_part is not None else float("nan")

        gene = str(vcol("GENE_NAMES").iloc[idx]) if vcol("GENE_NAMES") is not None else ""
        parent_prot = str(vcol("PARENT_PROTEIN").iloc[idx]) if vcol("PARENT_PROTEIN") is not None else "n/a"
        precursors = vcol("PRECURSORS_EXP")
        precursors_val = int(precursors.iloc[idx]) if precursors is not None and pd.notna(precursors.iloc[idx]) else "n/a"

        uid_for_link = _first_token(parent_prot) or ""
        string_link = _cached_string_link(uid_for_link)

        Number = pn.indicators.Number
        ft_q_ind = Number(name="FT q-value", value=q_ft, format="{value:.3e}", default_color="gray",
                          font_size="12pt", styles={"flex": "1"})
        ft_lfc_ind = Number(name="FT log₂ FC", value=lfc_ft, format="{value:.3f}", default_color="gray",
                            font_size="12pt", styles={"flex": "1"})
        covp_ind = Number(name="Covariate part", value=cov_piece, format="{value:.3f}",
                          default_color="purple", font_size="12pt", styles={"flex": "1"})

        header = pn.pane.Markdown("**Flowthrough Analysis**",
                                  styles={"font-size": "16px", "padding": "0", "line-height": "0px"})

        left_bits = []
        if precursors_val != "n/a":
            left_bits.append(f"Precursors (protein): <b>{precursors_val}</b>")

        if gene:
            left_bits.append(f"Gene(s): <b>{_first_token(gene)}</b>")
        if parent_prot and parent_prot != "n/a":
            left_bits.append(f"Protein: <b>{_first_token(parent_prot)}</b>")

        footer_left = pn.pane.HTML(
            "<span style='font-size: 12px;'>" + " &nbsp;|&nbsp; ".join(left_bits) + "</span>"
        )

        footer_right = pn.pane.HTML(
            (
                "<span style='font-size: 12px;'>"
                f"🔗 <a href='https://www.uniprot.org/uniprotkb/{uid_for_link}/entry' "
                "target='_blank' rel='noopener'>UniProt Entry</a>"
                " &nbsp;|&nbsp; "
                f"<a href='{string_link}' target='_blank' rel='noopener'>STRING Entry</a>"
                "</span>"
            ),
            styles={"text-align": "right", "width": "98%"},
            margin=(0, 0, 5, 0),
        )

        card_style = {
            "background": "#f9f9f9",
            "align-items": "center",
            "border-radius": "8px",
            "text-align": "center",
            "padding": "5px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "justify-content": "space-evenly",
        }

        card = pn.Card(
            header,
            make_hr(),
            pn.Row(ft_q_ind, ft_lfc_ind, sizing_mode="stretch_width"),
            make_hr(),
            footer_left,
            make_hr(),
            footer_right,
            width=450,
            styles=card_style,
            collapsible=False,
            hide_header=True,
        )
        return card

    def _render_peptide_table():
        site_id = str(search_input.value or "")
        if not site_id:
            return pn.Spacer(width=300, height=120)

        try:
            idx = views["site_idx"](site_id)
        except KeyError:
            return pn.Spacer(width=300, height=120)

        vcol = views["vcol"]
        parent_pep = str(vcol("PARENT_PEPTIDE_ID").iloc[idx])
        if not parent_pep:
            return pn.Spacer(width=300, height=120)

        contrast = contrast_sel.value
        df_raw_fc = views["df_raw_fc"]
        df_raw_q = views["df_raw_q"]

        mask = (adata.var["PARENT_PEPTIDE_ID"].astype(str) == parent_pep)
        siblings = adata.var_names[mask]
        if len(siblings) == 0:
            return pn.Spacer(width=300, height=120)

        df = pd.DataFrame(index=siblings)
        df["Site"] = [_pn_only(s) for s in siblings]
        df["Log2FC"] = df_raw_fc.loc[siblings, contrast].astype(float).values
        df["Q"] = df_raw_q.loc[siblings, contrast].astype(float).values
        df["site_id"] = df.index
        df["__pnum__"] = df["Site"].str.extract(r"p(\d+)").astype(int)
        df = df.sort_values("__pnum__", kind="mergesort").copy()

        sig = df["Q"] < 0.05
        colors = np.where((df["Log2FC"] > 0) & sig, "red",
                          np.where((df["Log2FC"] < 0) & sig, "blue", "gray"))
        df["Site_html"] = [f"<span style='color:{c}'>{pn_}</span>" for c, pn_ in zip(colors, df["Site"])]
        df["Q_sci"] = [f"{x:.2e}" if np.isfinite(x) else "NA" for x in df["Q"]]

        disp = (
            df[["Site_html", "Log2FC", "Q_sci", "site_id"]]
            .reset_index(drop=True)
            .rename(columns={"Site_html": "Site", "Q_sci": "Q-value"})
        )

        # highlight current row
        def _highlight_current(row):
            return ['background-color: rgba(255,235,59,0.35)']*len(row) if row["site_id"] == site_id else ['']*len(row)

        def _color_site_col(col):
            # `col` is the 'Site' Series in display order; use precomputed row_colors
            return [f'color: {c}' for c in row_colors]

        styled = (disp.style
            .apply(_highlight_current, axis=1)
            .apply(_color_site_col, subset=["Site"])
            .format({"Log2FC": "{:.2f}"})  # .2f as requested
        )


        # Compact height with scrolling after 3 rows
        row_h, header_h = 32, 30
        nrows = len(disp)
        visible = max(min(int(nrows), 3), 1)
        table_h = row_h * (3 if nrows > 3 else visible) + header_h

        tbl = pn.widgets.Tabulator(
            styled,
            formatters={"Site": {"type": "html"}, "Log2FC": NumberFormatter(format="0.000")},
            hidden_columns=["site_id"],
            selectable=1,
            show_index=False,
            layout="fit_columns",
            disabled=True,
            height=table_h,
            width=300,
            widths={"Site": 55},
            configuration={
                "rowHeight": row_h,
                "columnHeaderVertAlign": "bottom",
                "movableColumns": False,
                "columnDefaults": {"editor": False, "headerSort": False},
            },
            styles={"flex": "0"},
            margin=(8, 8, 8, 8),
        )

        def _on_select(event):
            sel = event.new
            if sel:
                i = sel[0]
                picked = disp.iloc[i]["site_id"]
                search_input.value = str(picked)

        tbl.param.watch(_on_select, "selection")

        header = pn.pane.Markdown("**Adjacent Sites**",
                                  styles={"font-size": "16px", "padding": "0", "line-height": "0px"})

        card = pn.Card(
            header, make_hr(), tbl,
            width=330, collapsible=False, hide_header=True,
            styles={"background": "#f9f9f9", "border-radius": "8px",
                    "box-shadow": "3px 3px 5px #bcbcbc", "padding": "5px"},
        )
        return card

    def _render_bar_phos():
        return plot_intensity_by_site(state, contrast_sel.value, str(search_input.value), layers_phos_sel.value)

    def _render_bar_cov():
        return plot_covariate_by_site(state, contrast_sel.value, str(search_input.value), layers_cov_sel.value)

    # Update orchestration
    def _update_info(_=None):
        main_info_holder[:] = [_render_main_info()]
        cov_info_holder[:] = [_render_cov_info()]

    def _update_main_bar(_=None):
        site = search_input.value
        if not site:
            main_bar_holder[:] = [pn.Spacer(width=800, height=500, margin=(-30, 0, 0, 0))]
            return
        main_bar_holder.loading = True
        try:
            main_bar_holder[:] = [pn.pane.Plotly(_render_bar_phos(), width=800, height=250,
                                                 margin=(-30, 0, 0, 0),
                                                 styles={"border-radius": "8px",
                                                         "box-shadow": "3px 3px 5px #bcbcbc"})]
        finally:
            main_bar_holder.loading = False

    def _update_cov_bar(_=None):
        if not has_cov:
            cov_bar_holder[:] = [pn.Spacer(width=0, height=0)]
            return
        site = search_input.value
        if not site:
            cov_bar_holder[:] = [pn.Spacer(width=800, height=500, margin=(-30, 0, 0, 0))]
            return
        cov_bar_holder.loading = True
        try:
            cov_bar_holder[:] = [pn.pane.Plotly(_render_bar_cov(), width=800, height=190, margin=(0, 0, 0, 0),
                                                styles={"border-radius": "8px",
                                                        "box-shadow": "3px 3px 5px #bcbcbc"})]
        finally:
            cov_bar_holder.loading = False

    def _update_peptide_table(_=None):
        if not search_input.value:
            pep_table_holder[:] = [pn.Spacer(width=330, height=120)]
            return
        pep_table_holder.loading = True
        try:
            pep_table_holder[:] = [_render_peptide_table()]
        finally:
            pep_table_holder.loading = False

    # Wire events
    search_input.param.watch(lambda _e: (_update_info(), _update_peptide_table(), _update_main_bar(), _update_cov_bar()),
                             "value")
    contrast_sel.param.watch(lambda _e: (_update_info(), _update_peptide_table(), _update_main_bar(), _update_cov_bar()),
                             "value")
    layers_phos_sel.param.watch(lambda _e: _update_main_bar(), "value")
    layers_cov_sel.param.watch(lambda _e: _update_cov_bar(), "value")

    bokeh_doc.add_next_tick_callback(lambda: (_update_info(), _update_main_bar(), _update_cov_bar(), _update_peptide_table()))

    volcano_and_detail = pn.Row(
        pn.Column(
            volcano_plot,
            cohort_violin_view,
            sizing_mode="stretch_both",
            styles={"flex": "1", "min-width": "600px"},
        ),
        pn.Spacer(width=30),
        detail_panel,
        sizing_mode="stretch_width",
        styles={"align-items": "stretch"},
        margin=(20, 0, 0, 0),
    )

    volcano_pane = pn.Column(
        pn.pane.Markdown("##   Volcano plots"),
        pn.Row(
            contrast_sel,
            pn.Spacer(width=10),
            volcano_src_sel,
            pn.Spacer(width=10),
            color_by,
            pn.Spacer(width=10),
            make_vr(),
            pn.Spacer(width=10),
            pn.Column(show_measured, show_imp_cond1, show_imp_cond2, margin=(-5, 0, 0, 0)),
            pn.Spacer(width=10),
            #min_meas_sel,
            pn.Column(
                min_meas_sel,
                min_prec_sel,
                margin=(-30,0,0,0),
            ),

            pn.Spacer(width=10),
            make_vr(),
            pn.Spacer(width=20),
            pn.Column(search_field_sel),
            pn.Spacer(width=10),
            pn.Column(search_input_group),
            pn.Column(pn.Row(clear_all, margin=(13, 0, 0, 0)), status_pane),
            pn.Spacer(width=10),
            make_vr(),
            pn.Spacer(width=10),
            search_input,
            pn.Row(clear_search, margin=(15, 0, 0, 0)),
            pn.Spacer(width=0),
            pn.Row(layers_phos_sel, margin=(-17, 0, 0, 0)),
            pn.Spacer(width=10),
            pn.Row(layers_cov_sel, margin=(-17, 0, 0, 0)),
            sizing_mode="fixed",
            width=300,
            height=70,
        ),
        volcano_and_detail,
        height=1110,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={"border-radius": "15px", "box-shadow": "3px 3px 5px #bcbcbc", "width": "98vw"},
    )
    volcano_pane.height = pn.bind(lambda ids: 1250 if ids else 1110, group_ids_dmap)

    # ---------- Final layout ----------
    layout = pn.Column(
        pn.Spacer(height=10),
        intro_pane,
        pn.Spacer(height=30),
        metrics_pane,
        pn.Spacer(height=30),
        clustering_pane,
        pn.Spacer(height=30),
        volcano_pane,
        pn.Spacer(height=30),
        sizing_mode="stretch_both",
        styles=FRAME_STYLES,
    )
    return layout

