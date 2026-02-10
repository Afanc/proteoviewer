import os
import re
import panel as pn
import textwrap
from functools import lru_cache
from utils.session_state import SessionState
from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_volcanoes_wrapper,
    plot_intensity_by_protein,
    get_protein_info,
    plot_peptide_trends_centered,
    plot_group_violin_for_volcano,
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
def overview_tab(state: SessionState):
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
    input_layout  = preproc_cfg.get("input_layout", "")

    num_samples = len(adata.obs.index.unique())
    num_conditions = len(adata.obs["CONDITION"].unique())
    num_contrasts = int(num_conditions*(num_conditions-1)/2)

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

    # build a single Markdown string
    summary_md = textwrap.dedent(f"""

        **Analysis Type**: {analysis_type}

        {num_samples} Samples - {num_conditions} Conditions - {num_contrasts} Contrasts

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
        - **Differential expression**: eBayes via {ebayes_method}

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

    id_sort_toggle = make_id_sort_toggle(margin=(20, 0, 0, 20), width=170)

    barplot_title = "Protein IDs by Sample and Category"
    if peptidomics_mode:
        barplot_title = "Peptide IDs by Sample and Category"
    if phospho_mode:
        barplot_title = "Phosphosites by Sample and Category"

    hist_ID_dmap = pn.bind(
        plot_barplot_proteins_per_sample,
        adata=adata,
        sort_by=pn.bind(sort_arg, id_sort_toggle),
        title=barplot_title,
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
    cv_fig, rmad_fig = plot_violin_cv_rmad_per_condition(adata)
    metrics_pane = make_metrics_pane(cv_fig=cv_fig, rmad_fig=rmad_fig)

    clustering_pane = make_clustering_pane(
        adata=state.adata,
        plot_pca_2d=plot_pca_2d,
        plot_mds_2d=plot_mds_2d,
        plot_umap_2d=plot_umap_2d,
    )

    ## Volcanoes
    # Contrast selector
    contrasts = state.adata.uns["contrast_names"].tolist()

    contrast_sel = pn.widgets.Select(
        name="Contrast",
        options=contrasts,
        value=contrasts[0],
        width=250,
    )

    show_measured  = pn.widgets.Checkbox(name="Observed in Both", value=True)
    show_imp_cond1 = pn.widgets.Checkbox(name=f"", value=True)
    show_imp_cond2 = pn.widgets.Checkbox(name=f"", value=True)

    # Color selector
    color_options = ["Significance", "Avg Intensity"]
    if "ibaq" in state.adata.layers:
        color_options.append("Avg IBAQ")

    color_by = pn.widgets.Select(
        name="Color by",
        options=color_options,
        value=color_options[0],
        width=150,
    )

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

    # Min numb. precursors options
    max_prec_options = 6 if proteomics_mode else 4
    min_prec_title = "pep" if proteomics_mode else "prec"
    min_prec_sel, _min_prec_value = make_min_precursor_select(
        max_prec_options=max_prec_options,
        title_token=min_prec_title,
        width=80,
        default_label="≥0",
    )

    # search widget
    def _ensure_gene(token: str) -> str | None:
        """
        Return a gene symbol for `token`.
        - If `token` is already a gene in adata.var["GENE_NAMES"], return it.
        - Else, if `token` matches a UniProt ID (var_names), map it to its gene (first token before ';').
        - Else, return the original token (so downstream error handling can show 'No match').
        """
        if not token:
            return None
        ad = state.adata
        names = ad.var["GENE_NAMES"].astype(str)
        t = str(token)

        # fast path: exact gene match
        if t in set(names):
            return t

        # fallback: UniProt -> gene
        try:
            idx = ad.var_names.get_loc(t)
            gene = names.iloc[idx]
            return gene.split(";", 1)[0].strip() if isinstance(gene, str) else str(gene)
        except KeyError:
            return t

    search_input_name = "Search Protein"
    placeholder_txt="Gene name or Uniprot ID"
    options_list=list(state.adata.var["GENE_NAMES"]) + list(state.adata.var_names)
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
        plot_volcanoes_wrapper,
        state=state,
        contrast=contrast_sel,
        color_by=color_by,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        min_nonimp_per_cond=pn.bind(_min_meas_value, min_meas_sel),
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
            pt = click["points"][0]
            cd = pt.get("customdata") or []

            mode = str(state.adata.uns.get("preprocessing", {}).get("analysis_type", "")).lower()
            proteomics_mode = (mode in {"dia", "dda", "proteomics"})

            if proteomics_mode:
                search_input.value = str(pt.get("text", ""))
            else:
                search_input.value = str(cd[0] if isinstance(cd, (list, tuple)) and len(cd) else pt.get("text", ""))


    volcano_dmap_wrapped = bind_uirevision(volcano_dmap, contrast_sel, prefix="volcano")

    volcano_plot = pn.pane.Plotly(
        volcano_dmap_wrapped,
        height=900,
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
        file_text_widget=_file_text,
        clear_btn=clear_all,
        search_field_sel=search_field_sel,
    )

    # Cohort Violin View
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
            config={'responsive': True},
            styles={
                'border-radius':  '8px',
                'box-shadow':     '3px 3px 5px #bcbcbc',
            }
        )

    # Bind reactivity via pn.bind (don’t pass bind objects into @depends)
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

    # bind a detail‐plot function to the same contrast & search_input
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
    layers_sel.visible = bool(search_input.value)

    _intensity_slot = pn.Column(styles={'flex': '1'})

    @lru_cache(maxsize=4096)
    def _cached_string_link(uniprot_id: str) -> str:
        # Cache ONLY the URL string (safe). If the call fails, return "".
        try:
            return get_string_link(uniprot_id) or ""
        except Exception:
            return ""

    @pn.depends(protein=search_input, contrast=contrast_sel)
    def info_card(protein, contrast):
        if not protein:
            # Reserve the card’s area when nothing is selected
            return pn.Spacer(width=800, height=170)

        # pull values that do not depend on the "layer" toggle
        key = _ensure_gene(protein) if proteomics_mode else str(protein)
        protein_info = get_protein_info(state, contrast, key, layers_sel)

        uniprot_id   = protein_info['uniprot_id']
        idx = protein_info['index']

        def _first_token(x):
            if isinstance(x, str) and ";" in x:
                return x.split(";", 1)[0].strip()
            return x

        def _fmt_int(x):
            v = int(x)
            return f"{v:,}"

        def _fmt_ibaq(x):
            v = float(x)
            if v >= 1_000:
                return f"{v:,.0f}"
            elif v >= 1:
                return f"{v:.2f}"
            else:
                return f"{v:.2e}"

        def _safe_var(name, default=None):
            try:
                if name in adata.var.columns:
                    return adata.var[name].iloc[idx]
            except Exception:
                pass
            return default

        rec_val  = _safe_var("PRECURSORS_EXP", default=None)
        ibaq_avg = protein_info.get("avg_ibaq")
        ibaq_val = _fmt_ibaq(ibaq_avg) if ibaq_avg is not None else "n/a"

        re_count = _fmt_int(rec_val) if rec_val is not None else "n/a"

        Number = pn.indicators.Number

        # q-value and log2FC never depend on layer, keep these fixed in the card
        q_ind = Number(
            name="q-value",
            value=protein_info["qval"],
            format="{value:.3e}",
            default_color="red",
            font_size="12pt",
            styles={'flex': '1'}
        )
        lfc_ind = Number(
            name="log₂ FC",
            value=protein_info["logfc"],
            format="{value:.3f}",
            default_color="red",
            font_size="14pt",
            styles={'flex': '1'}
        )

        _intensity_slot[:] = [pn.Spacer(height=0)]

        # header 
        base_size = 18
        max_len   = 10
        length    = len(protein)
        top_padding = 6
        if length <= max_len:
            size = base_size
        else:
            size = max(10, int(base_size * (max_len / length)**0.5))
            top_padding = max(10, top_padding * (max_len / length)**0.5)

        item_styles = {
            "font-size": f"{size}px",
            "margin": "0px",
            "padding": f"{top_padding}px 0px 0px 0px",
            "line-height": "0px",
            "flex": "1",
            "min-width": "0",
        }
        sep_styles = {
            **item_styles,
            "flex": "0.1",
            "margin": "0px 0px",
        }
        mode = str(state.adata.uns.get("preprocessing", {}).get("analysis_type", "")).lower()
        peptido_mode = (mode in {"peptido", "peptidomics"})

        gene = _ensure_gene(protein)
        uid  = protein_info["uniprot_id"]
        if peptido_mode:
            # In peptido, "Uniprot ID" is the peptide sequence.
            peptide_md = pn.pane.Markdown(f"**Peptide**: {uniprot_id}",
                                          styles={"font-size": "16px", "padding": "0", "line-height": "0px"})

            header = pn.Row(
                peptide_md,
                sizing_mode="stretch_width",
                height=50,
                styles={
                    "display":         "flex",
                    "background":      "#f9f9f9",
                    "margin": "0px",
                    "padding": "0px",
                    "border-bottom":   "1px solid #ddd",
                }
            )
        else:
            gene_md = pn.pane.Markdown(f"**Gene(s)**: {gene}", styles=item_styles)
            sep1    = pn.pane.Markdown("|", styles=sep_styles)
            uid_md  = pn.pane.Markdown(f"**Uniprot ID**: {uniprot_id}", styles=item_styles)
            sep2    = pn.pane.Markdown("|", styles=sep_styles)
            idx_md  = pn.pane.Markdown(f"**Protein Index**: {protein_info['index']+1}",
                                       styles=item_styles)
            header = pn.Row(
                gene_md, sep1, uid_md, sep2, idx_md,
                sizing_mode="stretch_width",
                height=50,
                styles={
                    "display":         "flex",
                    "align-items":     "space-evenly",
                    "justify-content": "space-evenly",
                    "background":      "#f9f9f9",
                    "margin": "0px",
                    "padding": "0px",
                    "border-bottom":   "1px solid #ddd",
                }
            )

        # STRING link is layer-agnostic; fetch once & cache
        uid_for_link = _first_token(uniprot_id) or ""
        try:
            string_link = _cached_string_link(uid_for_link)
        except Exception:
            string_link = ""

        if peptido_mode:
            # Optional peptido metadata
            parent_uid = _safe_var("PARENT_PROTEIN", default=None)
            if parent_uid is None:
                parent_uid = _safe_var("UNIPROT", default=None)
            parent_uid = _first_token(str(parent_uid)) if parent_uid is not None else "n/a"

            pep_prec = _safe_var("PRECURSORS_USED", default=None)
            if pep_prec is None:
                pep_prec = _safe_var("PRECURSORS_EXP", default=None)
            pep_prec_txt = _fmt_int(pep_prec) if pep_prec is not None else "n/a"

            left_bits = [
                f"Gene: <b>{gene}</b>",
                f"UniProt: <b>{parent_uid}</b>",
                f"Precursors (peptide): <b>{pep_prec_txt}</b>",
                f"Peptide index: <b>{protein_info['index']+1}</b>",
            ]
        else:
            left_bits = []
            if rec_val is not None:
                left_bits.append(f"Precursors (global): <b>{re_count}</b>")
            if ibaq_avg is not None:
                left_bits.append(f"IBAQ (global mean): <b>{ibaq_val}</b>")
            # If neither is available, show a friendly placeholder
            if not left_bits:
                left_bits.append("No extra protein metrics available")

        footer_left = pn.pane.HTML(
            "<span style='font-size: 12px;'>" + " &nbsp;|&nbsp; ".join(left_bits) + "</span>"
        )

        footer_right = pn.pane.HTML("") if peptido_mode else pn.pane.HTML(
            f"<span style='font-size: 12px;'>"
            f"🔗 <a href='https://www.uniprot.org/uniprotkb/{uid_for_link}/entry' target='_blank' rel='noopener'>UniProt Entry</a>"
            f" &nbsp;|&nbsp; "
            f"<a href='{string_link}' target='_blank' rel='noopener'>STRING Entry</a>"
            f"</span>"
        )

        footer_links = pn.Row(
            footer_left, pn.Spacer(), footer_right,
            sizing_mode="stretch_width",
            styles={
                "justify-content": "space-between",   # left stats | right links
                "padding": "2px 8px 4px 0px",
                "margin-top": "-6px",
            }
        )

        hr = pn.Spacer(height=1, sizing_mode="stretch_width", styles={
            "background": "#ccc",
            "margin": "6px 0"
        })

        card_style = {
            'background':     '#f9f9f9',
            "align-items":     "center",
            'border-radius':  '8px',
            'text-align':     "center",
            'padding':        '5px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'justify-content':'space-evenly',
        }

        card = pn.Card(
            header,
            pn.Row(q_ind, lfc_ind, _intensity_slot, sizing_mode="stretch_width"),
            hr,
            footer_links,
            width=800,
            styles=card_style,
            collapsible=False,
            hide_header=True,
        )
        return card


    @pn.depends(protein=search_input, contrast=contrast_sel, layer=layers_sel)
    def bar_and_intensity(protein, contrast, layer):
        if not protein:
            # Reserve space for the bar plot when nothing is selected
            _intensity_slot[:] = [pn.Spacer(height=0)]
            return pn.Spacer(width=800, height=500, margin=(-30, 0, 0, 0))

        # bar plot
        key = _ensure_gene(protein) if proteomics_mode else str(protein)
        fig = plot_intensity_by_protein(state, contrast, key, layers_sel)
        protein_info = get_protein_info(state, contrast, key, layers_sel)

        barplot_pane = pn.pane.Plotly(
            fig,
            width=800,
            height=400,
            margin=(-30, 0, 0, 0),
            styles={
                'border-radius':  '8px',
                'box-shadow':     '3px 3px 5px #bcbcbc',
            }
        )

        # intensity Number
        intensity_scale = "Avg Log Intensity" if layer != "Raw" else "Avg Intensity"

        prot_avg_val = protein_info["avg_int"]
        prot_avg_val = f"{prot_avg_val:.3f}" if prot_avg_val <= 100 else f"{prot_avg_val:.0f}"

        Number = pn.indicators.Number
        int_ind = Number(
            name=intensity_scale,
            value=protein_info["avg_int"],
            format=prot_avg_val,
            default_color="darkorange",
            font_size="16pt",
            styles={'flex': '1'}
        )

        # Inject intensity Number into the card without rebuilding the card itself
        _intensity_slot[:] = [int_ind]

        return barplot_pane


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
        key = _ensure_gene(token) if proteomics_mode else str(token)
        info = get_protein_info(state, contrast_sel.value, key, layers_sel)
        return info["uniprot_id"]

    def _render_info():
        # pass current values explicitly (protein, contrast)
        return info_card(search_input.value, contrast_sel.value)

    def _render_bar():
        # pass (protein, contrast, layer) explicitly
        return bar_and_intensity(search_input.value, contrast_sel.value, layers_sel.value)

    def _render_pep():
        uid = _current_uniprot_id()
        if not uid:
            return pn.Spacer(width=800, height=320)
        fig = plot_peptide_trends_centered(state.adata, uid, contrast_sel.value)
        return pn.pane.Plotly(
            fig,
            height=285,
            width=800,
            margin=(0,0,0,0),
            styles={'border-radius': '8px', 'box-shadow': '3px 3px 5px #bcbcbc'}
        )

    def _update_info(_=None):
        # No spinner here; it's cheap and we don't want a loader on empty states
        info_holder[:] = [_render_info()]

    def _update_bar(_=None):
        protein = search_input.value
        if not protein:
            # No protein selected -> no spinner, show placeholder and clear intensity slot
            _intensity_slot[:] = [pn.Spacer(height=0)]
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
    layers_sel.param.watch(lambda e: _update_bar(), "value")

    # Initial fill (after the page paints so we don’t see a flash)
    bokeh_doc.add_next_tick_callback(lambda: (_update_info(), _update_bar(), _update_pep()))

    # assemble into a layout, no legend‐based toggles
    volcano_and_detail = pn.Row(
        pn.Column(                 # left container that can stretch
            volcano_plot,
            cohort_violin_view,
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
        pn.pane.Markdown("##   Volcano plots"),
        pn.Row(
            contrast_sel,
            pn.Spacer(width=20),
            color_by,
            pn.Spacer(width=20),
            make_vr(),
            pn.Spacer(width=20),
            pn.Column(
                show_measured,
                show_imp_cond1,
                show_imp_cond2,
                margin=(-5,0,0,0),
            ),
            pn.Spacer(width=10),
            pn.Column(
                min_meas_sel,
                min_prec_sel,
                margin=(-30,0,0,0),
            ),
            pn.Spacer(width=20),
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

    volcano_pane.height = pn.bind(lambda ids: 1200 if ids else 1060, group_ids_selected)

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

    def _frame_styles(ids, protein):
        has_ids = bool(ids)
        has_protein = bool((protein or "").strip())
        if has_ids or has_protein:
            return FRAME_STYLES_TALL
        return FRAME_STYLES_SHORT

    layout.styles = pn.bind(_frame_styles, group_ids_selected, search_input)

    return layout
