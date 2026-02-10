from __future__ import annotations

import os
import re
import warnings
import textwrap
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models.widgets.tables import NumberFormatter
from utils.session_state import SessionState

from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_covariate_by_site,
    plot_group_violin_for_volcano,
    plot_intensity_by_site,
    plot_peptide_trends_centered,
    plot_violin_cv_rmad_per_condition,
    plot_volcanoes_wrapper,
    resolve_exact_list_to_uniprot_ids,
    resolve_pattern_to_uniprot_ids,
    get_protein_info,
)
from components.selection_export import (
    SelectionExportSpec,
    make_volcano_selection_downloader,
    make_adjacent_sites_csv_callback
)
from tabs.overview_shared import (
    make_id_sort_toggle,
    sort_arg,
    make_intro_pane,
    make_min_meas_select,
    make_min_precursor_select,
    make_toggle_label_updater,
    make_cohort_inspector_widgets,
    make_metrics_pane,
    make_clustering_pane,
    fmt_files_list,
    bind_uirevision,
    wire_cohort_export_updates,
)
from components.plot_utils import plot_pca_2d, plot_umap_2d, plot_mds_2d
from components.string_links import get_string_link
from components.texts import intro_preprocessing_text, log_transform_text
from utils.layout_utils import (
    plotly_section,
    make_vr,
    make_hr,
    make_section,
    make_row,
    FRAME_STYLES,
    FRAME_STYLES_TALL,
    FRAME_STYLES_SHORT
)
from utils.utils import log_time, logger


# Small utilities

def _pn_only(site_id: str) -> str:
    s = str(site_id)
    m = re.search(r"\|((?:p|[STY])\d+)", s)
    return m.group(1) if m else ""

def _site_token_after_pipe(site_id: str) -> str:
    s = str(site_id)
    return s.split("|", 1)[1] if "|" in s else s

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


# Cached accessors bound to a specific AnnData 

def _make_adata_views(adata):
    var_index = pd.Index(adata.var_names, name="INDEX")
    contrast_names = tuple(adata.uns["contrast_names"])

    def _df_opt(varm_key):
        """Return DataFrame if varm_key exists, else None."""
        arr = adata.varm.get(varm_key, None)
        if arr is None:
            return None
        return pd.DataFrame(arr, index=var_index, columns=contrast_names)
    # Required 
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

    # cached indexing for site -> integer position
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

    analysis_type = preproc.get("analysis_type", "phospho")
    phospho_cfg = preproc.get("phospho", {}) or {}
    multisite_mode = str(phospho_cfg.get("multisite_collapse_policy", "explode") or "explode")
    ebayes_method = analysis.get("ebayes_method", "limma")
    input_layout = preproc.get("input_layout", "")
    quant_method = preproc.get("quantification_method", "sum")

    pf_version = adata.uns['proteoflux'].get("pf_version", 0.0)

    cont_step = filtering.get("cont", {})
    q_step = filtering.get("qvalue", {})
    pep_step = filtering.get("pep", {})
    prec_step = filtering.get("prec", {})
    censor_step= filtering.get("censor", {})
    loc_step = filtering.get("loc", {})

    cont_txt, _ = _fmt_step(cont_step, "n/a")
    q_txt, q_thr_txt = _fmt_step(q_step, "n/a")
    pep_txt, pep_thr_txt = _fmt_step(pep_step, "n/a")
    pep_op = _pep_dir_symbol(pep_step)
    prec_txt, prec_thr_txt = _fmt_step(prec_step, "n/a")
    censor_txt, censor_thr_txt= _fmt_step(censor_step,  "n/a")
    loc_txt, loc_thr_txt = _fmt_step(loc_step, "n/a")
    loc_mode = str(loc_step.get("mode", "N/A")).replace("filter_", "")

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
        lc_conmed_lod_k = preproc.get("imputation").get("lc_conmed_lod_k", "NA")
        lc_conmed_min_obs = preproc.get("imputation").get("lc_conmed_in_min_obs", "1")
        extras.append(f"lod_k={lc_conmed_lod_k}")
        extras.append(f"min_obs={lc_conmed_min_obs}")
    if extras:
        imp_method = f"{imp_method} ({', '.join(extras)})"

    return textwrap.dedent(
        f"""
        **Analysis Type**: {analysis_type}

        {num_samples} Samples - {num_conditions} Conditions - {num_contrasts} Contrasts

        **Input Layout**: {input_layout}

        **Pipeline steps**
        - **Quantification**: {quant_method}
        - **Multiphosphorylated policy**: {multisite_mode}
        - **Filtering**:
            - Contaminants ({', '.join(contaminants_files)}): {cont_txt}
            - q-value ≤ {q_thr_txt}: {q_txt}
            - PEP {pep_op} {pep_thr_txt}: {pep_txt}
            - Min. run evidence count = {prec_thr_txt}: {prec_txt}
            - Left Censoring ≤ {censor_thr_txt}: {censor_txt}
            - Phospho Localization Score ({loc_mode}, thr={loc_thr_txt}): {loc_txt}
        - **Normalization**: {norm_methods_str}
        - **Imputation**: {imp_method}
        - **Differential expression**: eBayes via {ebayes_method}

        **Proteoflux Version** {pf_version}
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


# Main Tab 

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
    preproc_cfg = adata.uns["preprocessing"]
    analysis_type  = preproc_cfg.get("analysis_type", "DIA")
    phospho_cfg = preproc_cfg.get("phospho", {}) or {}
    multisite_mode = str(phospho_cfg.get("multisite_collapse_policy", "explode") or "explode")

    views = _make_adata_views(adata)
    has_cov = bool(adata.uns["has_covariate"])
    contrast_names = list(views["contrast_names"])

    # Summary / Intro 
    summary_md = _build_pipeline_summary(adata)
    summary_pane = pn.pane.Markdown(
        summary_md,
        sizing_mode="stretch_width",
        margin=(-10, 0, 0, 20),
        styles={"line-height": "1.4em", "word-break": "break-word", "overflow-wrap": "anywhere", "min-width": "0"},
    )

    id_sort_toggle = make_id_sort_toggle(margin=(20, 0, 0, 20), width=170)

    hist_ID_dmap = pn.bind(
        plot_barplot_proteins_per_sample,
        adata=adata,
        sort_by=pn.bind(sort_arg, id_sort_toggle),
        title="Phosphosites by Sample and Category"
    )

    hist_plot_pane = pn.pane.Plotly(
        hist_ID_dmap,
        height=500,
        margin=(-20, 20, 0, -190),
        styles={"flex": "1"},
    )

    intro_pane = make_intro_pane(
        summary_pane=summary_pane,
        id_sort_toggle=id_sort_toggle,
        hist_plot_pane=hist_plot_pane,
        hist_plot_margin=(-20, 20, 0, -190),
        height=530,
    )

    # Metrics + Clustering (shared builders; figures are unchanged)
    cv_fig, rmad_fig = plot_violin_cv_rmad_per_condition(adata)
    metrics_pane = make_metrics_pane(cv_fig=cv_fig, rmad_fig=rmad_fig)

    clustering_pane = make_clustering_pane(
        adata=state.adata,
        plot_pca_2d=plot_pca_2d,
        plot_mds_2d=plot_mds_2d,
        plot_umap_2d=plot_umap_2d,
    )

    # Volcanoes 
    contrasts = contrast_names
    contrast_sel = pn.widgets.Select(name="Contrast", options=contrasts, value=contrasts[0], width=180)

    # which volcano to plot
    volcano_src_opts = (["Phospho (raw)"] if not has_cov
                        else ["Phospho (adj.)", "Phospho (raw)", "Flowthrough"])
    volcano_src_sel = pn.widgets.Select(name="Volcano source", options=volcano_src_opts, value=volcano_src_opts[0], width=130)

    def _volcano_dtype(label: str) -> str:
        return {
            "Phospho (raw)"      : "phospho",
            "Phospho (adj.)" : "default",
            "Flowthrough"   : "flowthrough",
        }[label]

    show_measured = pn.widgets.Checkbox(name="Observed in Both", value=True)
    show_imp_cond1 = pn.widgets.Checkbox(name="", value=True)
    show_imp_cond2 = pn.widgets.Checkbox(name="", value=True)

    color_options = ["Significance"] + (["Raw LogFC", "Adj. LogFC", "FT LogFC"] if has_cov else [])
    color_by = pn.widgets.Select(name="Color by", options=color_options, value=color_options[0], width=120)

    make_toggle_label_updater(
        contrast_sel=contrast_sel,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
    )

    min_meas_sel, _min_meas_value = make_min_meas_select(
        adata=adata,
        contrast_sel=contrast_sel,
        allow_zero=False,
        name="Min / condition",
        width=80,
        default_value_label=None,  # preserve original: ≥min(reps)
    )

    # Flowthrough non-imputed datapoints per condition filter (only meaningful when has_cov)
    ft_min_meas_sel, _min_meas_ft_value = make_min_meas_select(
        adata=adata,
        contrast_sel=contrast_sel,
        allow_zero=True,            # FT can be 0
        name="Min / FT condition",
        width=90,
        default_value_label=None,   # preserve original: default to ≥max(reps)
    )
    ft_min_meas_sel.disabled = (not has_cov)


    # Min numb. precursors options
    max_prec_options = 6 if analysis_type == "DIA" else 4
    min_prec_title = "pep" if analysis_type == "DIA" else "prec"
    min_prec_sel, _min_prec_value = make_min_precursor_select(
        max_prec_options=max_prec_options,
        title_token=min_prec_title,
        width=80,
        default_label="≥0",
    )

    search_input = pn.widgets.AutocompleteInput(
        name="Search Phosphosite",
        options=list(adata.var_names),
        width=200,
        case_sensitive=False,
    )
    clear_search = pn.widgets.Button(name="Clear", width=80)
    clear_search.on_click(lambda _e: setattr(search_input, "value", ""))

    def _update_ft_filter_enable(_=None) -> None:
        # FT filter is a no-op in raw volcano mode; disable it so UI matches semantics
        if not has_cov:
            ft_min_meas_sel.disabled = True
            return
        ft_min_meas_sel.disabled = (volcano_src_sel.value == "Phospho (raw)")

    volcano_src_sel.param.watch(_update_ft_filter_enable, "value")
    _update_ft_filter_enable()

    # If the user clears the phosphosite selection, we must also clear the export "click" state.
    # Plotly click_data is sticky and does not reliably emit an "empty" event.
    def _on_site_cleared(event) -> None:
        if (event.new or "") == "":
            _on_volcano_click_data({})

    # Ensure FASTA headers are available under a stable name for the resolver.
    # Non-phospho uses a "FASTA headers" field option; phospho datasets may or may not have FASTA headers.
    # We support common column variants and keep behavior "no matches" if absent.
    if "FASTA_HEADERS" not in adata.var.columns:
        if "FASTA_HEADER" in adata.var.columns:
            adata.var["FASTA_HEADERS"] = adata.var["FASTA_HEADER"]
        elif "FASTA" in adata.var.columns:
            adata.var["FASTA_HEADERS"] = adata.var["FASTA"]
        else:
            # Don't spam logs; just warn once per session.
            warnings.warn("Phospho overview: FASTA headers not found in adata.var; pattern search on FASTA will return 0 matches.")
            adata.var["FASTA_HEADERS"] = ""

    (
        search_field_sel,
        search_input_group,
        file_holder,
        clear_all,
        status_pane,
        group_ids_selected,
        _file_text,
        cohort_filename,
    ) = make_cohort_inspector_widgets(
        adata=adata,
        search_field_options=["FASTA headers", "Gene names", "UniProt IDs"],
        search_field_default="Gene names",
        pattern_placeholder="ECOLI or ^gene[0-9]$",
        status_margin=(-10, 0, 0, 0),
        clear_btn_width=80,
        file_btn_width=200,
        pattern_width=200,
        field_width=130,
        field_margin=(2, 0, 0, 0),
    )

    volcano_dmap = pn.bind(
        plot_volcanoes_wrapper,
        state=state,
        contrast=contrast_sel,
        data_type=(lambda: "phospho") if not has_cov else pn.bind(_volcano_dtype, volcano_src_sel),
        color_by=color_by,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        min_nonimp_per_cond=pn.bind(_min_meas_value, min_meas_sel),
        min_nonimp_ft_per_cond=(0 if (not has_cov) else pn.bind(_min_meas_ft_value, ft_min_meas_sel)),
        min_precursors=pn.bind(_min_prec_value, min_prec_sel),
        highlight=search_input,
        highlight_group=group_ids_selected,
        sign_threshold=0.05,
        width=None,
        height=900,
    )

    def _on_volcano_click(event):
        click = event.new
        if click and click.get("points"):
            gene = click["points"][0]["text"]
            search_input.value = gene

    volcano_dmap_wrapped = bind_uirevision(volcano_dmap, contrast_sel, prefix="volcano")

    volcano_plot = pn.pane.Plotly(
        volcano_dmap_wrapped,
        height=1150,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        config={"responsive": True},
        styles={"border-radius": "8px", "box-shadow": "3px 3px 5px #bcbcbc", "flex": "1"},
    )
    volcano_plot.param.watch(_on_volcano_click, "click_data")

    # selection download 
    download_selection, _on_volcano_selected_data, _on_volcano_click_data, _on_cohort_ids = make_volcano_selection_downloader(
        state=state,
        contrast_getter=lambda: str(contrast_sel.value),
        spec=SelectionExportSpec(
            filename="proteoflux_selection.csv",
            label="Download selection",
            uniprot_var_col="PARENT_PROTEIN",
            id_col_name="PHOSPHOSITE_ID",
        ),
    )
    volcano_plot.param.watch(lambda e: _on_volcano_selected_data(e.new), "selected_data")
    volcano_plot.param.watch(lambda e: _on_volcano_click_data(e.new), "click_data")

    # Cohort changes must update the export state immediately (priority: click > cohort > lasso).
    wire_cohort_export_updates(
        group_ids_selected=group_ids_selected,
        on_cohort_ids=_on_cohort_ids,
        search_input_group=search_input_group,
        file_text_widget=_file_text,
        clear_btn=clear_all,
        search_field_sel=search_field_sel,
    )

    # Cohort Violin View (same semantics as non-phospho overview)
    def _cohort_violin(ids, contrast, sm, s1, s2, min_nonimp_per_cond, min_consistent_peptides):
        if not ids:
            return pn.Spacer(height=0)  # collapses cleanly when no cohort
        fig = plot_group_violin_for_volcano(
            state=state,
            contrast=contrast,
            min_nonimp_per_cond=min_nonimp_per_cond,
            min_consistent_peptides=min_consistent_peptides,
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
        _cohort_violin,
        group_ids_selected,
        contrast_sel,
        show_measured,
        show_imp_cond1,
        show_imp_cond2,
        min_nonimp_per_cond=pn.bind(_min_meas_value, min_meas_sel),
        min_consistent_peptides=pn.bind(_min_prec_value, min_prec_sel),
    )

    # Clear exporter click state when phosphosite is cleared.
    search_input.param.watch(_on_site_cleared, "value")

    # Detail panes 
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
    prec_trend_holder = pn.Column()

    detail_panel = pn.Row(
        pn.Column(
            main_info_holder,
            pn.Spacer(height=20),
            pn.Row(cov_info_holder, pn.Spacer(width=20, visible=has_cov), pep_table_holder, width=800),
            pn.Spacer(height=45),
            main_bar_holder,
            pn.Spacer(height=20),
            cov_bar_holder,
            pn.Spacer(height=20),
            prec_trend_holder,
            width=840,
        ),
        margin=(0, 0, 0, 0),
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
            "<span style='font-size: 12px;'>"
            f"Precursors (site): <b>{pep_num_prec}</b>"
            f" &nbsp;|&nbsp; Phospho index: <b>{idx + 1}</b>"
            "</span>"
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

        rows = [header,
                pn.Row(raw_q_ind, raw_lfc_ind, loc_ind, sizing_mode="stretch_width"),
                make_hr()]
        # Add FT-adjusted row only if covariate outputs exist
        if has_cov:
            rows += [pn.Row(q_ind, lfc_ind, covp_ind, sizing_mode="stretch_width"),
                     make_hr()]

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
        parent_prot = str(vcol("PARENT_PROTEIN").iloc[idx])
        if not parent_prot:
            return pn.Spacer(width=300, height=120)

        contrast = contrast_sel.value
        # Adjacent sites should show adjusted stats when FT is present.
        # For "no-FT" contrasts, adjusted stats are already set to raw upstream.
        if has_cov:
            df_fc = views["df_log2fc"]
            df_q  = views["df_q"]
            hdr_suffix = " (adj.)"
        else:
            df_fc = views["df_raw_fc"]
            df_q  = views["df_raw_q"]
            hdr_suffix = ""

        mask = (adata.var["PARENT_PROTEIN"].astype(str) == parent_prot)
        siblings = adata.var_names[mask]
        if len(siblings) == 0:
            return pn.Spacer(width=300, height=120)

        df = pd.DataFrame(index=siblings)
        df["Site"] = [_site_token_after_pipe(s) for s in siblings]
        df["Log2FC"] = df_fc.loc[siblings, contrast].astype(float).values
        df["Q"]      = df_q.loc[siblings, contrast].astype(float).values

        df["site_id"] = df.index
        # Sort by the first numeric position encountered (works for "S42" and "S42,S53").
        df["__pnum__"] = pd.to_numeric(df["Site"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        if df["__pnum__"].isna().any():
            bad = df.loc[df["__pnum__"].isna(), "site_id"].astype(str).tolist()[:10]
            raise ValueError(
                "Phospho site parsing failed for Adjacent Sites table. "
                f"Expected '|p<INT>' or '|[STY]<INT>'. Examples={bad}"
            )
        df["__pnum__"] = df["__pnum__"].astype(int)
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
            .format({"Log2FC": "{:.2f}"})
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
            widths={"Site": 75 if multisite_mode == "keep" else 55},
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

        download_adjacent = pn.widgets.FileDownload(
            label="Download",
            callback=make_adjacent_sites_csv_callback(
                state=state,
                contrast_getter=lambda: str(contrast_sel.value),
                siblings_getter=lambda: list(siblings),
            ),
            filename="proteoflux_adjacent_sites.csv",
            button_type="success",
            visible=True,
            margin=(5, 10, 0, 0),
        )

        header_label = "**Adjacent Sites**" if multisite_mode != "keep" else "**Adjacent Phosphopeptides**"
        header = pn.Row(
                    pn.pane.Markdown(header_label,
                                  styles={"font-size": "16px", "padding": "0", "line-height": "0px"}),
                    pn.Spacer(sizing_mode="stretch_width"),
                    download_adjacent,
                    sizing_mode="stretch_width")

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

    def _render_precursor_trends():
        site_id = str(search_input.value or "")
        if not site_id:
            return pn.Spacer(width=800, height=200)
        fig = plot_peptide_trends_centered(state.adata, site_id, contrast_sel.value)
        return pn.pane.Plotly(
            fig,
            height=180,
            width=800,
            margin=(0, 0, 0, 0),
            styles={"border-radius": "8px", "box-shadow": "3px 3px 5px #bcbcbc"},
        )

    # Update orchestration
    def _update_info(_=None):
        main_info_holder[:] = [_render_main_info()]
        cov_info_holder[:] = [_render_cov_info()]

    def _update_main_bar(_=None):
        site = search_input.value
        if not site:
            main_bar_holder[:] = [pn.Spacer(width=800, height=170, margin=(-30, 0, 0, 0))]
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
            cov_bar_holder[:] = [pn.Spacer(width=800, height=100, margin=(-30, 0, 0, 0))]
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
            pep_table_holder[:] = [pn.Spacer(width=100, height=120)]
            return
        pep_table_holder.loading = True
        try:
            pep_table_holder[:] = [_render_peptide_table()]
        finally:
            pep_table_holder.loading = False

    def _update_precursor_trends(_=None):
        if not search_input.value:
            prec_trend_holder[:] = [pn.Spacer(width=800, height=100)]
            return
        prec_trend_holder.loading = True
        try:
            prec_trend_holder[:] = [_render_precursor_trends()]
        finally:
            prec_trend_holder.loading = False

    # Wire events
    search_input.param.watch(lambda _e: (_update_info(), _update_peptide_table(), _update_main_bar(), _update_cov_bar(), _update_precursor_trends()),
                             "value")
    contrast_sel.param.watch(lambda _e: (_update_info(), _update_peptide_table(), _update_main_bar(), _update_cov_bar(), _update_precursor_trends()),
                             "value")
    layers_phos_sel.param.watch(lambda _e: _update_main_bar(), "value")
    layers_cov_sel.param.watch(lambda _e: _update_cov_bar(), "value")

    bokeh_doc.add_next_tick_callback(lambda: (_update_info(), _update_main_bar(), _update_cov_bar(), _update_peptide_table(), _update_precursor_trends()))

    volcano_and_detail = pn.Row(
        pn.Column(
            volcano_plot,
            cohort_violin_view,
            sizing_mode="stretch_width",
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
            pn.Column(
                min_meas_sel,
                min_prec_sel,
                margin=(-30,0,0,0),
            ),
            pn.Spacer(width=10),
            make_vr(),
            pn.Spacer(width=10),
            ft_min_meas_sel,
            pn.Spacer(width=10),
            make_vr(),
            pn.Spacer(width=20),
            pn.Column(
                pn.pane.Markdown("**Cohort Inspector**", align="start", margin=(-20,0,0,10)),
                search_field_sel,
            ),
            pn.Spacer(width=10),
            pn.Column(
                search_input_group,
                file_holder,
                margin=(-25,0,0,0)
            ),
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
            pn.Spacer(width=20),
            download_selection,
            width=300,
            height=70,
        ),
        volcano_and_detail,
        height=1310,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={"border-radius": "15px", "box-shadow": "3px 3px 5px #bcbcbc", "width": "98vw"},
    )


    # Expand container when cohort violin is visible (prevents clipping)
    volcano_pane.height = pn.bind(lambda ids: 1460 if ids else 1310, group_ids_selected)

    # Final layout 
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
        sizing_mode="stretch_width",
        styles=FRAME_STYLES_TALL,
    )

    return layout

