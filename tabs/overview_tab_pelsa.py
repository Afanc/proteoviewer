import os
import re
import numpy as np
import pandas as pd
import panel as pn
import textwrap
from functools import lru_cache
from bokeh.models.widgets.tables import NumberFormatter
from utils.session_state import SessionState
from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_pelsa_volcano,
    plot_pelsa_curve,
    get_pelsa_info,
    plot_peptide_trends_centered,
    plot_group_violin_for_volcano,
    get_pelsa_sister_peptides,
    plot_pelsa_local_stability_profile,
)
from components.selection_export import (
    make_volcano_selection_downloader,
    SelectionExportSpec
)
from components.plot_utils import plot_pca_2d, plot_umap_2d, plot_mds_2d
from components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from components.string_links import get_string_link
from tabs.overview_shared import (
    make_id_sort_toggle,
    sort_arg,
    fmt_files_list,
    make_intro_pane,
    make_min_meas_select,
    make_min_precursor_select,
    make_toggle_label_updater,
    make_cohort_inspector_widgets,
    make_metrics_pane,
    make_clustering_pane,
    bind_uirevision,
    wire_cohort_export_updates,
    filter_feature_ids_to_visible_volcano,
)
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
from utils.utils import logger, log_time


def _fmt_tags_list(tags):
    """Format tags as 'a, b, c' (no brackets/quotes)."""
    if tags is None:
        return ""
    if isinstance(tags, (list, tuple)):
        return ", ".join(str(t) for t in tags)
    return str(tags)

@log_time("Preparing Overview Tab")
def overview_tab_pelsa(state: SessionState):
    """
    Overview Tab
     - Info on samples and filtering, config
     - Hists of Ids, CVs
     - PCAs, UMAP
     - Hierch. clustering
     - Volcanoes
    """
    # Plots
    adata = state.adata

    pelsa_uns = adata.uns.get("pelsa", {})
    if not pelsa_uns:
        raise KeyError("Missing adata.uns['pelsa']; PELSA overview requires PELSA export metadata.")

    concentration_col = str(pelsa_uns.get("concentration_column", "")).strip()
    concentration_group_key = "Concentration"
    conc_values = pd.to_numeric(adata.obs[concentration_col], errors="raise")
    adata.obs[concentration_group_key] = conc_values.map(lambda x: f"{x:g}").astype(str)

    # Normalize protein/gene names
    def _build_search_options(ad):
        ids = list(map(str, ad.var_names))

        if "GENE_NAMES" not in ad.var.columns:
            return ids

        genes = ad.var["GENE_NAMES"].astype(str).tolist()

        counts = {}
        for g in genes:
            g = str(g).strip()
            if not g or g.lower() == "nan":
                continue
            counts[g] = counts.get(g, 0) + 1

        opts = []
        seen = set()

        for gene, uid in zip(genes, ids):
            gene = str(gene).strip()
            uid = str(uid).strip()

            if gene and gene.lower() != "nan":
                label = f"{gene} | {uid}" if counts.get(gene, 0) > 1 else gene
                if label not in seen:
                    opts.append(label)
                    seen.add(label)

        for uid in ids:
            if uid not in seen:
                opts.append(uid)
                seen.add(uid)

        return opts

    def _normalize_search_token(token: str) -> str:
        s = str(token or "").strip()
        if not s:
            return s
        if " | " in s:
            return s.rsplit(" | ", 1)[1].strip()
        return s

    ## Config Pane
    # Texts
    preproc_cfg = adata.uns["preprocessing"]
    analysis_cfg = adata.uns["analysis"]
    filtering      = preproc_cfg.get("filtering", {})
    normalization  = preproc_cfg.get("normalization", {})
    imputation     = preproc_cfg.get("imputation", {})
    analysis_type  = preproc_cfg.get("analysis_type", "")
    proteomics_mode = analysis_type in {"dia", "dda", "proteomics"}
    peptidomics_mode = analysis_type in {"peptido", "peptidomics"}
    phospho_mode = analysis_type in {"phospho", "phosphoproteomics"}
    ebayes_method  = analysis_cfg.get("ebayes_method", "limma")
    batch_cols = analysis_cfg.get("batch_effect_columns", None)
    input_layout  = preproc_cfg.get("input_layout", "")

    num_samples = len(adata.obs.index.unique())
    num_replicates = len(adata.obs["REPLICATE"].astype(str).unique())
    num_concentrations = len(adata.obs[concentration_group_key].unique())

    quant_method = preproc_cfg.get("quantification", {}).get("method", "sum")
    if quant_method == "directlfq":
        min_nonan = preproc_cfg.get("quantification").get("directlfq_min_nonan", 1)
        quant_method += f", min nonan={min_nonan}"

    pf_version = adata.uns['proteoflux'].get("pf_version", 0.0)

    flt_cfg = adata.uns.get("preprocessing", {}).get("filtering", [])

    def _fmt_step(step: dict, name: str, default_thr: str) -> tuple[str, str]:
        if not step:
            return "Skipped", "n/a"
        if step.get("skipped"):
            # our preprocessing metadata sets 'skipped': True when the column is absent
            return "Skipped", "n/a"
        return f"{step.get('number_dropped', 0):,} PSM removed", str(step.get("threshold", default_thr))

    cont_step = flt_cfg.get("cont", {})
    q_step    = flt_cfg.get("qvalue", {})
    pep_step  = flt_cfg.get("pep", {})
    prec_step  = flt_cfg.get("prec", {})
    censor_step= flt_cfg.get("censor", {})

    cont_txt, _          = _fmt_step(cont_step, "cont", "n/a")
    q_txt,    q_thr_txt  = _fmt_step(q_step,    "qvalue", "n/a")
    pep_txt,  pep_thr_txt= _fmt_step(pep_step,  "pep", "n/a")
    pep_op = "≥" if flt_cfg.get("pep").get("direction").startswith("greater") else "≤"
    prec_txt,  prec_thr_txt= _fmt_step(prec_step,  "prec", "n/a")
    censor_txt, censor_thr_txt= _fmt_step(censor_step,  "censor", "n/a")

    contaminants_files = [os.path.basename(p) for p in flt_cfg.get('cont', {}).get('files', [])]

    # Norm condensation
    norm_methods = normalization.get("method", []).tolist()
    if "loess" in norm_methods:
        loess_span = preproc_cfg.get("normalization").get("loess_span")
        norm_methods += f" (loess_span={loess_span})"
    if "median_equalization_by_tag" in norm_methods:
        tags = preproc_cfg.get("normalization").get("reference_tag")
        if isinstance(tags, str):
            tags = [tags]
        tag_matches = preproc_cfg.get("normalization").get("tag_matches")
        median_index = norm_methods.index("median_equalization_by_tag")
        norm_methods[median_index] += " " + f"(tags={tags}, matches={tag_matches})  "

    norm_methods = ", ".join(norm_methods)

    # Imputation condensation
    imp_method = imputation.get("method", "")
    extras = []
    if "knn" in imp_method:
        extras = []
        if "knn_k" in imputation:
            extras.append(f"k={imputation['knn_k']}")
        if "tnknn" in imp_method and "knn_tn_perc" in imputation:
            extras.append(f"tn_perc={imputation['knn_tn_perc']}")
        if extras:
            imp_method += " (" + ", ".join(extras) + ")"

    if "rf" in imp_method:
        rf_max_iter = preproc_cfg.get("imputation").get("rf_max_iter")
        imp_method += f", rf_max_iter={rf_max_iter}"

    if "lc_conmed" in imp_method and "lc_conmed_lod_k" in imputation:
        lc_conmed_lod_k = preproc_cfg.get("imputation").get("lc_conmed_lod_k", "NA")
        lc_conmed_min_obs = preproc_cfg.get("imputation").get("lc_conmed_in_min_obs", "1")
        extras.append(f"lod_k={lc_conmed_lod_k}")
        extras.append(f"min_obs={lc_conmed_min_obs}")
    if extras:
        imp_method = f"{imp_method} ({', '.join(extras)})"

    # format batch info (single line, appended to DE line)
    batch_txt = ""
    if batch_cols:
        if isinstance(batch_cols, (list, tuple)):
            batch_txt = f" (batch effect columns: {', '.join(map(str, batch_cols))})"
        else:
            batch_txt = f" (batch effect column: {batch_cols})"

    # build a single Markdown string
    summary_md = textwrap.dedent(f"""

        **Analysis Type**: {analysis_type}

        {num_samples} Samples - {num_replicates} Replicates - {num_concentrations} Concentrations

        **Input Layout**: {input_layout}

        **Pipeline steps**
        - **Filtering**:
            - Contaminants ({', '.join(contaminants_files)}): {cont_txt}
            - q-value ≤ {q_thr_txt}: {q_txt}
            - PEP {pep_op} {pep_thr_txt}: {pep_txt}
            - Min. run evidence count = {prec_thr_txt}: {prec_txt}
            - Left Censoring ≤ {censor_thr_txt}: {censor_txt}
        - **Quantification**: {quant_method}
        - **Normalization**: {norm_methods}
        - **Imputation**: {imp_method}
        - **Curve fitting**: {analysis_cfg.get("analysis_method", "pelsa_curve_fit")}

        **Proteoflux Version** {pf_version}
    """).strip()

    # intro_pane:
    summary_pane = pn.pane.Markdown(summary_md,
        sizing_mode="stretch_width",
        margin=(-10, 0, 0, 20),
        styles={
            "line-height":"1.4em",
            "word-break": "break-word",
            "overflow-wrap": "anywhere",
            "min-width": "0",
        }
    )

    id_sort_toggle = pn.widgets.RadioButtonGroup(
        name="Order",
        options=["By concentration", "By sample"],
        value="By concentration",
        button_type="default",
        width=190,
        margin=(20, 0, 0, 20),
        styles={"z-index": "10"},
    )

    def _pelsa_sort_arg(mode: str) -> str:
        return "group" if mode == "By concentration" else "sample"

    barplot_title = "Protein IDs by Sample and Category"
    if peptidomics_mode:
        barplot_title = "Peptide IDs by Sample and Category"
    if phospho_mode:
        barplot_title = "Phosphosites by Sample and Category"

    hist_ID_dmap = pn.bind(
        plot_barplot_proteins_per_sample,
        adata=adata,
        sort_by=pn.bind(_pelsa_sort_arg, id_sort_toggle),
        title=barplot_title,
        group_key=concentration_group_key,
        group_label="Concentration",
    )

    hist_plot_pane = pn.pane.Plotly(hist_ID_dmap,
                       height=500,
                       margin=(0, 20, 0, -190),
                       styles={"flex":"1",
                              }
    )

    intro_pane = make_intro_pane(
        summary_pane=summary_pane,
        id_sort_toggle=id_sort_toggle,
        hist_plot_pane=hist_plot_pane,
        hist_plot_margin=(0, 20, 0, -190),
        height=530,
    )

    # Metrics + Clustering
    cv_fig, rmad_fig = plot_violin_cv_rmad_per_condition(
        adata,
        group_key=concentration_group_key,
        group_label="Concentration",
    )

    metrics_pane = make_metrics_pane(cv_fig=cv_fig, rmad_fig=rmad_fig)

    clustering_pane = make_clustering_pane(
        adata=state.adata,
        plot_pca_2d=plot_pca_2d,
        plot_mds_2d=plot_mds_2d,
        plot_umap_2d=plot_umap_2d,
        color_key=concentration_group_key,
    )

    ## Volcanoes
    # Contrast selector
    contrasts = list(map(str, state.adata.uns.get("contrast_names", ["pelsa_curve_fit"])))

    contrast_sel = pn.widgets.Select(
        name="Contrast",
        options=contrasts,
        value=contrasts[0],
        width=250,
        visible=False,
    )

    show_measured  = pn.widgets.Checkbox(name="Observed in Both", value=True)
    show_imp_cond1 = pn.widgets.Checkbox(name=f"", value=True)
    show_imp_cond2 = pn.widgets.Checkbox(name=f"", value=True)

    # Color selector
    color_options = ["Significance"]
    if "nrsc" in state.adata.varm:
        color_options.append("Norm. rel. SC")
    color_options.append("Avg Intensity")
    if "ibaq" in state.adata.layers:
        color_options.append("Avg IBAQ")


    color_by = pn.widgets.Select(
        name="Color by",
        options=color_options,
        value=color_options[0],
        width=150,
    )

    #make_toggle_label_updater(
    #    contrast_sel=contrast_sel,
    #    show_imp_cond1=show_imp_cond1,
    #    show_imp_cond2=show_imp_cond2,
    #)

    #min_meas_sel, _min_meas_value = make_min_meas_select(
    #    adata=adata,
    #    contrast_sel=contrast_sel,
    #    allow_zero=False,
    #    name="Min / condition",
    #    width=80,
    #    default_value_label=None,  # preserve original: ≥min(reps)
    #)

    # Min numb. precursors options
    max_prec_options = 6 if proteomics_mode else 4
    min_prec_title = "pep" if proteomics_mode else "prec"
    min_prec_sel, _min_prec_value = make_min_precursor_select(
        max_prec_options=max_prec_options,
        title_token=min_prec_title,
        width=80,
        default_label="≥0",
    )
    nrsc_alignment_sel = pn.widgets.FloatSlider(
        name="Max nrSC misalign.",
        start=0.0,
        end=2.0,
        step=0.05,
        value=1.50,
        width=130,
        bar_color="blue",
        visible=("nrsc_misalignment" in state.adata.varm),
    )

    curve_results = pd.DataFrame(pelsa_uns["curve_results"])

    def _slider_end(col: str, default: float) -> float:
        if col not in curve_results.columns:
            return default
        vals = pd.to_numeric(curve_results[col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            return default
        return max(default, float(vals.quantile(0.99)))

    #max_nrmse_sel = pn.widgets.FloatSlider(
    #    name="Max nRMSE",
    #    start=0.0,
    #    end=_slider_end("normalized_rmse", 2.0),
    #    step=0.05,
    #    value=_slider_end("normalized_rmse", 2.0),
    #    width=130,
    #    bar_color="gray",
    #)

    #max_pec50_ci_sel = pn.widgets.FloatSlider(
    #    name="Max pEC50 CI/range",
    #    start=0.0,
    #    end=_slider_end("pEC50_ci_width_norm", 3.0),
    #    step=0.05,
    #    value=_slider_end("pEC50_ci_width_norm", 3.0),
    #    width=150,
    #    bar_color="gray",
    #)
    hide_zero_q_sel = pn.widgets.Checkbox(
        name="Hide -log10(q)=0",
        value=True,
    )

    search_input_name = "Search Protein/Gene"
    placeholder_txt="Gene name or UniProt ID"
    options_list = _build_search_options(state.adata)
    #options_list=list(state.adata.var["GENE_NAMES"]) + list(state.adata.var_names)
    if peptidomics_mode:
        search_input_name = "Search Peptide"
        placeholder_txt = "Peptide Sequence"
        options_list=list(state.adata.var_names)

    search_input = pn.widgets.AutocompleteInput(
        name=search_input_name,
        options=options_list,
        placeholder=placeholder_txt,
        width=200,
        case_sensitive=False,
    )

    clear_search = pn.widgets.Button(name="Clear", width=80)
    clear_search.on_click(lambda event: setattr(search_input, "value", ""))

    # If the user clears the protein selection, we must also clear the export "click" state.
    # Plotly click_data is sticky and does not reliably emit an "empty" event.
    def _on_search_cleared(event) -> None:
        if (event.new or "") == "":
            _on_volcano_click_data({})

    search_input.param.watch(_on_search_cleared, "value")

    search_field_sel = pn.widgets.Select(
        name="Search Field",
        options=["FASTA headers", "Gene names", "UniProt IDs"],
        value="FASTA headers",
        width=130,
        styles={"z-index": "10"},
        margin=(2,0,0,-2),
    )

    search_input_group = pn.widgets.TextInput(
        name="Pattern or File", placeholder="ECOLI or ^gene[0-9]$",
        width=200, styles={"z-index": "10"}
    )

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
        adata=state.adata,
        search_field_options=["FASTA headers", "Gene names", "UniProt IDs"],
        search_field_default="FASTA headers",
        pattern_placeholder="*_ECOLI+ or ^gene[0-9]$",
        status_margin=(-10, 0, 0, 0),
        clear_btn_width=90,
        file_btn_width=200,
        pattern_width=200,
        field_width=130,
        field_margin=(2, 0, 0, -2),
    )

    volcano_dmap = pn.bind(
        plot_pelsa_volcano,
        state=state,
        highlight=pn.bind(_normalize_search_token, search_input),
        highlight_group=group_ids_selected,
        sign_threshold=0.05,
        hide_zero_neglog10_q=hide_zero_q_sel,
        #max_normalized_rmse=max_nrmse_sel,
        #max_pec50_ci_width_norm=max_pec50_ci_sel,
        width=None,
        height=900,
    )

    def _on_volcano_click(event):
        click = event.new
        if click and click.get("points"):
            pt = click["points"][0]
            cd = pt.get("customdata") or []

            search_input.value = str(cd[0] if isinstance(cd, (list, tuple)) and len(cd) else pt.get("text", ""))

    #volcano_dmap_wrapped = bind_uirevision(volcano_dmap, contrast_sel, prefix="volcano")
    volcano_dmap_wrapped = volcano_dmap

    volcano_plot = pn.pane.Plotly(
        volcano_dmap_wrapped,
        height=1150,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        config={'responsive': True},
        styles={
            'border-radius':  '8px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'flex': '1',
        }
    )
    volcano_plot.param.watch(_on_volcano_click, "click_data")


    # selection download
    download_selection, _on_volcano_selected_data, _on_volcano_click_data, _on_cohort_ids = make_volcano_selection_downloader(
        state=state,
        contrast_getter=lambda: str(contrast_sel.value),
        spec=SelectionExportSpec(
            filename="proteoflux_selection.csv",
            label="Download selection",
        ),
    )
    volcano_plot.param.watch(lambda e: _on_volcano_selected_data(e.new), "selected_data")
    volcano_plot.param.watch(lambda e: _on_volcano_click_data(e.new), "click_data")

    # Cohort changes must update the export state immediately (priority: click > cohort > lasso).
    wire_cohort_export_updates(
        group_ids_selected=group_ids_selected,
        on_cohort_ids=_on_cohort_ids,
        search_input_group=search_input_group,
        transform_ids=lambda ids: list(ids or []),
        file_text_widget=_file_text,
        clear_btn=clear_all,
        search_field_sel=search_field_sel,
    )

    # Cohort Violin View
    #def _cohort_violin(ids, contrast, sm, s1, s2, min_nonimp_per_cond, min_consistent_peptides):
    #    if not ids:
    #        return pn.Spacer(height=0)  # collapses cleanly when no cohort
    #    fig = plot_group_violin_for_volcano(
    #        state=state,
    #        contrast=contrast,
    #        min_nonimp_per_cond=min_nonimp_per_cond,
    #        min_consistent_peptides=min_consistent_peptides,
    #        highlight_group=ids,
    #        show_measured=sm,
    #        show_imp_cond1=s1,
    #        show_imp_cond2=s2,
    #        width=1200,
    #        height=100,
    #    )
    #    return pn.pane.Plotly(
    #        fig,
    #        height=150,
    #        margin=(-10, 0, 10, 20),
    #        sizing_mode="stretch_width",
    #        config={'responsive': True},
    #        styles={
    #            'border-radius':  '8px',
    #            'box-shadow':     '3px 3px 5px #bcbcbc',
    #        }
    #    )

    ## Bind reactivity via pn.bind (don’t pass bind objects into @depends)
    #cohort_violin_view = pn.bind(
    #    _cohort_violin,
    #    group_ids_selected,
    #    contrast_sel,
    #    show_measured,
    #    show_imp_cond1,
    #    show_imp_cond2,
    #    min_nonimp_per_cond=pn.bind(_min_meas_value, min_meas_sel),
    #    min_consistent_peptides=pn.bind(_min_prec_value, min_prec_sel),
    #)

    ## bind a detail‐plot function to the same contrast & search_input
    layers = ["Final", "Log-only", "Raw", "Spectral Counts"]

    layers_sel = pn.widgets.Select(
        name="Protein Data View",
        options=layers,
        value=layers[0],
        width=130,
        margin=(20, 0, 0, 20),
    )

    def _toggle_layers_visibility(event):
        layers_sel.visible = bool(event.new)

    search_input.param.watch(_toggle_layers_visibility, "value")
    layers_sel.visible = False

    @lru_cache(maxsize=4096)
    def _cached_string_link(uniprot_id: str) -> str:
        # Cache ONLY the URL string (safe). If the call fails, return "".
        try:
            return get_string_link(uniprot_id) or ""
        except Exception:
            return ""

    @pn.depends(protein=search_input)
    def info_card(protein):
        if not protein:
            return pn.Spacer(width=800, height=170)

        key = _normalize_search_token(protein)
        info = get_pelsa_info(state, key)

        Number = pn.indicators.Number
        metrics_row_items = [
            Number(name="q-value", value=info["qval"], format="{value:.3e}", default_color="red", font_size="12pt", styles={"flex": "1"}),
            Number(name="Range log₂", value=info["range_log2"], format="{value:.3f}", default_color="red", font_size="14pt", styles={"flex": "1"}),
            #Number(name="RMSE", value=info["rmse"], format="{value:.3f}", default_color="dimgray", font_size="14pt", styles={"flex": "1"}),
            #Number(name="nRMSE", value=info["normalized_rmse"], format="{value:.3f}", default_color="dimgray", font_size="14pt", styles={"flex": "1"}),
            #Number(name="RMSE log₂", value=info["rmse"], format="{value:.3f}", default_color="dimgray", font_size="14pt", styles={"flex": "1"}),
            #Number(name="nRMSE log₂", value=info["normalized_rmse"], format="{value:.3f}", default_color="dimgray", font_size="14pt", styles={"flex": "1"}),
            Number(name="R²", value=info["r2"], format="{value:.3f}", default_color="dimgray", font_size="14pt", styles={"flex": "1"}),
            #Number(name="pEC50", value=info["pec50"], format="{value:.3f}", default_color="dimgray", font_size="14pt", styles={"flex": "1"}),
            #Number(name="pEC50 CI/range", value=info["pec50_ci_width_norm"], format="{value:.3f}", default_color="dimgray", font_size="12pt", styles={"flex": "1"}),
        ]

        header = pn.Row(
            pn.pane.Markdown(f"**Peptide**: {info['peptide_id']}", styles={"font-size": "16px"}),
            sizing_mode="stretch_width",
            height=50,
            styles={
                "display": "flex",
                "background": "#f9f9f9",
                "padding": "0px",
                "border-bottom": "1px solid #ddd",
            },
        )

        footer_left = pn.pane.HTML(
            "<span style='font-size: 12px;'>"
            f"Gene: <b>{info['gene_names']}</b>"
            f" &nbsp;|&nbsp; p: <b>{info['pval']:.3e}</b>"
            #f" &nbsp;|&nbsp; pEC50 95% CI: <b>{info['pec50_ci_low']:.3f}–{info['pec50_ci_high']:.3f}</b>"
            #f" &nbsp;|&nbsp; pEC50 in range: <b>{info['pEC50_inside_range']}</b>"
            "</span>"
        )

        hr = pn.Spacer(height=1, sizing_mode="stretch_width", styles={"background": "#ccc", "margin": "6px 0"})

        return pn.Card(
            header,
            pn.Row(*metrics_row_items, sizing_mode="stretch_width"),
            hr,
            footer_left,
            width=800,
            styles={
                "background": "#f9f9f9",
                "align-items": "center",
                "border-radius": "8px",
                "text-align": "center",
                "padding": "5px",
                "box-shadow": "3px 3px 5px #bcbcbc",
                "justify-content": "space-evenly",
            },
            collapsible=False,
            hide_header=True,
        )

    @pn.depends(protein=search_input)
    def pelsa_curve_view(protein):
        if not protein:
            return pn.Spacer(width=800, height=500, margin=(-30, 0, 0, 0))

        key = _normalize_search_token(protein)
        fig = plot_pelsa_curve(state, key, width=800, height=350)

        return pn.pane.Plotly(
            fig,
            width=800,
            height=350,
            margin=(-30, 0, 0, 0),
            styles={
                "border-radius": "8px",
                "box-shadow": "3px 3px 5px #bcbcbc",
            },
        )

    info_holder = pn.Column()
    bar_holder  = pn.Column()
    pep_holder  = pn.Column()

    detail_panel = pn.Row(
        pn.Column(
            info_holder,
            pn.Spacer(height=50),
            bar_holder,
            pn.Spacer(height=20),
            pep_holder,
            width=840,
        ),
        margin=(0, 0, 0, 0),
        styles={
            "margin-left": "auto",
        }
    )

    bokeh_doc = pn.state.curdoc  # for next-tick scheduling

    def _current_uniprot_id():
        token = search_input.value
        if not token:
            return None
        key = _normalize_search_token(token)
        info = get_protein_info(state, contrast_sel.value, key, layers_sel)
        return info["uniprot_id"]

    def _render_info():
        # pass current values explicitly (protein, contrast)
        return info_card(search_input.value)

    def _render_bar():
        # pass (protein, contrast, layer) explicitly
        return pelsa_curve_view(search_input.value)

    def _render_sister_peptide_table(peptide):
        if not peptide:
            return pn.Spacer(width=380, height=260)

        key = _normalize_search_token(peptide)
        df = get_pelsa_sister_peptides(state, key)
        if df.empty:
            return pn.Spacer(width=380, height=260)

        disp = df.copy()
        disp["Peptide"] = disp["peptide_id"].astype(str)
        disp["Range log₂"] = pd.to_numeric(disp["range_log2"], errors="coerce")
        disp["q-value"] = (
            pd.to_numeric(disp["qval"], errors="coerce")
            .map(lambda x: f"{x:.3e}" if np.isfinite(x) else "nan")
        )

        styled = (
            disp[["Peptide", "Range log₂", "q-value", "peptide_id", "current"]]
            .style
            .apply(
                lambda row: ["background-color: rgba(255,235,59,0.35)"] * len(row)
                if bool(row["current"]) else [""] * len(row),
                axis=1,
            )
            .format({"Range log₂": "{:.3f}"})
        )

        row_h, header_h = 32, 30
        nrows = len(disp)
        visible = max(min(int(nrows), 3), 1)
        table_h = row_h * (3 if nrows > 3 else visible) + header_h

        tbl = pn.widgets.Tabulator(
            styled,
            formatters={
                "Range log₂": NumberFormatter(format="0.000"),
            },
            hidden_columns=["peptide_id", "current"],
            selectable=1,
            show_index=False,
            layout="fit_columns",
            disabled=True,
            height=table_h,
            width=780,
            widths={"Peptide": 190, "Range log₂": 90, "q-value": 90},
            configuration={
                "rowHeight": row_h,
                "columnHeaderVertAlign": "bottom",
                "movableColumns": False,
                "columnDefaults": {"editor": False, "headerSort": False},
            },
            margin=(8, 8, 8, 8),
        )

        def _on_select(event):
            sel = event.new
            if sel:
                picked = disp.iloc[sel[0]]["peptide_id"]
                search_input.value = str(picked)

        tbl.param.watch(_on_select, "selection")

        header = pn.pane.Markdown(
            "**Sister peptides**",
            styles={"font-size": "16px", "padding": "0", "line-height": "0px"},
        )
        return pn.Card(
            header,
            make_hr(),
            tbl,
            width=800,
            collapsible=False,
            hide_header=True,
            styles={
                "background": "#f9f9f9",
                "border-radius": "8px",
                "box-shadow": "3px 3px 5px #bcbcbc",
                "padding": "5px",
            },
        )

    def _render_local_stability_profile(peptide):
        if not peptide:
            return pn.Spacer(width=800, height=260)

        key = _normalize_search_token(peptide)
        fig = plot_pelsa_local_stability_profile(state, key, width=800, height=330)
        pane = pn.pane.Plotly(
            fig,
            width=800,
            height=330,
            margin=(0, 0, 0, 0),
            config={"responsive": True},
            styles={
                "border-radius": "8px",
                "box-shadow": "3px 3px 5px #bcbcbc",
            },
        )

        def _on_profile_click(event):
            click = event.new
            if not click or not click.get("points"):
                return

            pt = click["points"][0]
            cd = pt.get("customdata") or []
            if isinstance(cd, (list, tuple, np.ndarray)) and len(cd):
                picked = str(cd[0]).strip()
                if picked and picked.lower() not in {"nan", "none"}:
                    search_input.value = picked

        pane.param.watch(_on_profile_click, "click_data")
        return pane

    def _render_pep():
        peptide = search_input.value
        if not peptide:
            return pn.Spacer(width=800, height=320)

        #return pn.Row(
        #    _render_sister_peptide_table(peptide),
        #    pn.Spacer(width=20),
        #    _render_local_stability_profile(peptide),
        #    width=820,
        #    margin=(0, 0, 0, 0),
        #)
        return pn.Column(
            _render_sister_peptide_table(peptide),
            pn.Spacer(height=20),
            _render_local_stability_profile(peptide),
            width=820,
            margin=(0, 0, 0, 0),
        )

        return pn.Spacer(width=800, height=320)

    def _update_info(_=None):
        # No spinner here; it's cheap and we don't want a loader on empty states
        info_holder[:] = [_render_info()]

    def _update_bar(_=None):
        protein = search_input.value
        if not protein:
            bar_holder.loading = False
            bar_holder[:] = [pn.Spacer(width=800, height=500, margin=(-30, 0, 0, 0))]
            return

        # Only show a loader if there IS a protein selected (i.e., real work)
        bar_holder.loading = True
        try:
            bar_holder[:] = [_render_bar()]
        finally:
            bar_holder.loading = False

    def _update_pep(_=None):
        if not search_input.value:
            pep_holder.loading = False
            pep_holder[:] = [pn.Spacer(width=800, height=320)]
            return
        pep_holder.loading = True
        try:
            pep_holder[:] = [_render_pep()]
        finally:
            pep_holder.loading = False

    # Wire events:
    search_input.param.watch(lambda e: (_update_info(), _update_bar(), _update_pep()), "value")
    contrast_sel.param.watch(lambda e: (_update_info(), _update_bar(), _update_pep()), "value")
    layers_sel.param.watch(lambda e: (_update_info(), _update_bar()), "value")

    # Initial fill (after the page paints so we don’t see a flash)
    bokeh_doc.add_next_tick_callback(lambda: (_update_info(), _update_bar(), _update_pep()))

    # assemble into a layout, no legend‐based toggles
    volcano_and_detail = pn.Row(
        pn.Column(                 # left container that can stretch
            volcano_plot,
            #cohort_violin_view,
            sizing_mode="stretch_width",
            styles={
                "flex": "1",
            },
        ),
        pn.Spacer(width=30),
        detail_panel,               # fixed width on the right
        sizing_mode="stretch_width",
        styles={
            "align-items": "stretch",       # match heights nicely
        },
        margin=(20, 0, 0, 0),
    )

    volcano_pane = pn.Column(
        pn.pane.Markdown("##   Volcano plots", disable_anchors=True),
        pn.Row(
            color_by,
            pn.Spacer(width=20),
            pn.Column(
                hide_zero_q_sel,
                margin=(-30, 0, 0, 0),
            ),
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
            pn.Column(
                pn.Spacer(height=8),
                pn.Row(clear_all, margin=(0,0,0,0)),
                status_pane,
            ),
            pn.Spacer(width=20),
            make_vr(),
            pn.Spacer(width=20),
            search_input,
            pn.Row(clear_search, margin = (17,0,0,0)),
            pn.Spacer(width=0),
            pn.Row(layers_sel, margin = (-17,0,0,0)),
            pn.Spacer(width=20),
            download_selection,
            width=300,
            height=80,
        ),
        volcano_and_detail,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
        }
    )

    volcano_pane.height = pn.bind(lambda ids: 1460 if ids else 1310, group_ids_selected)

    # Tab layout
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

    #def _frame_styles(ids, protein):
    #    has_ids = bool(ids)
    #    has_protein = bool((protein or "").strip())
    #    if has_ids or has_protein:
    #        return FRAME_STYLES_TALL
    #    return FRAME_STYLES_SHORT

    #layout.styles = pn.bind(_frame_styles, group_ids_selected, search_input)

    return layout
