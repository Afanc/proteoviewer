import io
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
from components.structure_viewer import build_structure_viewer_pane, PeptideRegion
from components.selection_export import (
    make_volcano_selection_downloader,
    SelectionExportSpec,
    build_pelsa_selection_df,
    extract_feature_ids_from_selected_data,
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
    make_string_species_select,
    feature_ids_to_string_proteins,
    make_string_enrichment_card,
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

    string_species_sel = make_string_species_select(width=190)
    string_selected_feature_ids: list[str] = []

    def _selected_peptides_to_parent_proteins(feature_ids: list[str]) -> list[str]:
        return feature_ids_to_string_proteins(
            adata,
            feature_ids,
            prefer_parent=True,
            parent_col="PARENT_PROTEIN",
        )

    def _selected_peptide_table(feature_ids: list[str]):
        if not feature_ids:
            return pn.Spacer(width=820, height=0)

        required_var_cols = {"PARENT_PROTEIN", "GENE_NAMES"}
        missing_var_cols = sorted(required_var_cols - set(adata.var.columns))
        if missing_var_cols:
            raise KeyError(
                "Selected PELSA peptide table requires metadata columns in adata.var. "
                f"Missing={missing_var_cols}; available={list(adata.var.columns)!r}"
            )

        res = pd.DataFrame(pelsa_uns["curve_results"]).copy()
        required_res_cols = {
            "peptide_id",
            "curve_fold_change_log2",
            "curve_q_value",
            "pec50",
        }
        missing_res_cols = sorted(required_res_cols - set(res.columns))
        if missing_res_cols:
            raise KeyError(
                "Selected PELSA peptide table requires curve result columns. "
                f"Missing={missing_res_cols}; available={list(res.columns)!r}"
            )

        res["peptide_id"] = res["peptide_id"].astype(str)
        res = res.set_index("peptide_id", drop=False)

        ids = [str(x) for x in feature_ids]
        var_names = set(map(str, adata.var_names))
        missing_var = [x for x in ids if x not in var_names]
        if missing_var:
            raise ValueError(
                "Selected PELSA peptide table contains peptide ids missing from adata.var_names. "
                f"Examples={missing_var[:10]!r}"
            )

        missing_res = [x for x in ids if x not in set(res.index.astype(str))]
        if missing_res:
            raise ValueError(
                "Selected PELSA peptide table contains peptide ids missing from curve_results. "
                f"Examples={missing_res[:10]!r}"
            )

        var = adata.var.reindex(ids)
        sub = res.loc[ids]

        disp = pd.DataFrame({
            "Peptide": ids,
            "Range log₂": pd.to_numeric(
                sub["curve_fold_change_log2"],
                errors="raise",
            ).astype(float).values,
            "q-value": (
                pd.to_numeric(sub["curve_q_value"], errors="raise")
                .map(lambda x: f"{x:.3e}")
                .values
            ),
            "Gene": var["GENE_NAMES"].astype(str).values,
            "Parent protein": var["PARENT_PROTEIN"].astype(str).values,
            "pEC50": pd.to_numeric(sub["pec50"], errors="coerce").astype(float).values,
        })

        row_h, header_h = 30, 30
        visible_rows = min(max(len(disp), 1), 5)
        table_h = row_h * visible_rows + header_h

        tbl = pn.widgets.Tabulator(
            disp,
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=True,
            layout="fit_columns",
            height=table_h,
            pagination=None,
            sizing_mode="stretch_width",
            formatters={
                "Range log₂": NumberFormatter(format="0.000"),
                "pEC50": NumberFormatter(format="0.000"),
            },
            widths={
                "Peptide": 200,
                "Range log₂": 105,
                "q-value": 85,
                "Gene": 90,
                "Parent protein": 180,
                "pEC50": 80,
            },
            configuration={
                "rowHeight": row_h,
                "columnHeaderVertAlign": "bottom",
                "movableColumns": False,
                "columnDefaults": {"editor": False, "headerSort": True},
            },
            margin=(0,8,8,8),
        )

        return pn.Card(
            pn.pane.Markdown("**Selected peptides**", styles={"font-size": "15px", "padding": "0"}),
            tbl,
            collapsible=False,
            hide_header=True,
            sizing_mode="stretch_width",
            styles={"background": "#f9f9f9", "border-radius": "8px", "padding": "6px"},
        )

    ## Config Pane
    # Texts
    preproc_cfg = adata.uns["preprocessing"]
    analysis_cfg = adata.uns["analysis"]
    filtering      = preproc_cfg.get("filtering", {})
    normalization  = preproc_cfg.get("normalization", {})
    imputation     = preproc_cfg.get("imputation", {})
    analysis_type  = preproc_cfg.get("analysis_type", "")
    ebayes_method  = analysis_cfg.get("ebayes_method", "limma")
    batch_cols = analysis_cfg.get("batch_effect_columns", None)
    input_layout  = preproc_cfg.get("input_layout", "")

    num_samples = len(adata.obs.index.unique())
    num_replicates = len(adata.obs["REPLICATE"].astype(str).unique())
    num_concentrations = len(adata.obs[concentration_group_key].unique())

    quant_method = preproc_cfg.get("quantification", {}).get("peptide_rollup_method", "sum")
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

    barplot_title = "Peptide IDs by Sample and Category"

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

    #contrast_sel = pn.widgets.Select(
    #    name="Contrast",
    #    options=contrasts,
    #    value=contrasts[0],
    #    width=250,
    #    visible=False,
    #)

    show_measured  = pn.widgets.Checkbox(name="Observed in Both", value=True)
    show_imp_cond1 = pn.widgets.Checkbox(name=f"", value=True)
    show_imp_cond2 = pn.widgets.Checkbox(name=f"", value=True)

    # Color selector
    color_by = pn.widgets.Select(
        name="Color by",
        options=["Significance", "EC₅₀"],
        value="Significance",
        width=150,
    )

    structure_color_by = pn.widgets.Select(
        name="Structure color",
        options=["Significance", "None"],
        value="Significance",
        width=140,
    )

    structure_representation = pn.widgets.Select(
        name="Structure rep.",
        options=["Surface", "Cartoon"],
        value="Surface",
        width=140,
    )

    structure_controls = pn.Row(
        make_vr(),
        pn.Spacer(width=20),
        structure_color_by,
        pn.Spacer(width=20),
        structure_representation,
        pn.Spacer(width=20),
        visible=False,
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

    hide_zero_q_sel = pn.widgets.Checkbox(
        name="Hide Flat Curves (qval=1)",
        value=True,
    )

    search_input_name = "Search Peptide/Gene"
    placeholder_txt="Gene or peptide"
    options_list = _build_search_options(state.adata)

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
        color_by=color_by,
        sign_threshold=0.05,
        hide_zero_neglog10_q=hide_zero_q_sel,
        width=None,
        height=900,
    )

    def _on_volcano_click(event):
        click = event.new
        if click and click.get("points"):
            pt = click["points"][0]
            cd = pt.get("customdata") or []

            search_input.value = str(cd[0] if isinstance(cd, (list, tuple)) and len(cd) else pt.get("text", ""))

    def _with_pelsa_uirevision(fig):
        fig.update_layout(uirevision="pelsa-volcano")
        return fig

    volcano_dmap_wrapped = pn.bind(_with_pelsa_uirevision, volcano_dmap)

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
        contrast_getter= lambda: "pelsa_curve_fit",
        spec=SelectionExportSpec(
            filename="proteoflux_pelsa_selection.csv",
            label="Download selection",
            uniprot_var_col="PARENT_PROTEIN",
            id_col_name="PEPTIDE"
        ),
        selection_df_builder=build_pelsa_selection_df,
    )

    string_enrichment_holder = pn.Column(
        pn.Spacer(height=0),
        width=840,
        margin=(0, 0, 0, 0),
    )

    def _render_string_enrichment():
        proteins = _selected_peptides_to_parent_proteins(string_selected_feature_ids)
        return make_string_enrichment_card(
            selected_feature_ids=string_selected_feature_ids,
            proteins=proteins,
            species=string_species_sel.value,
            selected_table=_selected_peptide_table(string_selected_feature_ids),
            width=820,
         )

    def _update_string_enrichment(_=None) -> None:
        string_enrichment_holder.loading = True
        try:
            try:
                string_enrichment_holder[:] = [_render_string_enrichment()]
            except Exception as exc:
                string_enrichment_holder[:] = [pn.pane.Markdown(
                    f"**STRING enrichment failed**  \n`{exc}`",
                    styles={"background": "#fff3f3", "padding": "10px", "border-radius": "8px"},
                    sizing_mode="stretch_width",
                )]
        finally:
            string_enrichment_holder.loading = False
        _sync_detail_mode()

    def _on_pelsa_selected_data(event) -> None:
        selected_data = event.new
        _on_volcano_selected_data(selected_data)

        if selected_data is None or selected_data == {}:
            string_selected_feature_ids.clear()
            string_species_sel.visible = False
            _update_string_enrichment()
            _sync_detail_mode()
            return
        if "points" in selected_data and not selected_data["points"]:
            if selected_data.get("selector", "__missing__") is None:
                return
            string_selected_feature_ids.clear()
            string_species_sel.visible = False
            _update_string_enrichment()
            _sync_detail_mode()
            return

        string_selected_feature_ids.clear()
        string_selected_feature_ids.extend(extract_feature_ids_from_selected_data(selected_data))
        _update_string_enrichment()
        _sync_detail_mode()

    volcano_plot.param.watch(_on_pelsa_selected_data, "selected_data")

    volcano_plot.param.watch(lambda e: _on_volcano_click_data(e.new), "click_data")
    string_species_sel.param.watch(_update_string_enrichment, "value")

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
            return pn.Spacer(width=700, height=170)

        key = _normalize_search_token(protein)
        info = get_pelsa_info(state, key)

        def _first_token(x):
            s = str(x or "").strip()
            return s.split(";", 1)[0].strip() if ";" in s else s

        Number = pn.indicators.Number
        metrics_row_items = [
            Number(name="q-value", value=info["qval"], format="{value:.3e}", default_color="red", font_size="12pt", styles={"flex": "1"}),
            Number(name="Range log₂", value=info["range_log2"], format="{value:.3f}", default_color="red", font_size="14pt", styles={"flex": "1"}),
            Number(name="EC₅₀", value=info["ec50"], format="{value:.3f}", default_color="orange", font_size="14pt", styles={"flex": "1"}),
        ]

        header = pn.Row(
            pn.pane.HTML(
                f"""
                <style>
                .pelsa-peptide-scroll {{
                    scrollbar-width: none;      /* Firefox */
                    -ms-overflow-style: none;   /* old Edge/IE */
                }}
                .pelsa-peptide-scroll::-webkit-scrollbar {{
                    display: none;
                }}
                </style>
                <div class="pelsa-peptide-scroll" style="
                    font-size: 16px;
                    text-align: left;
                    white-space: nowrap;
                    overflow-x: auto;
                    overflow-y: hidden;
                    scrollbar-width: thin;
                    max-width: 100%;
                    width: 100%;
                    padding: 4px 6px 1px 6px;
                    box-sizing: border-box;
                    line-height: 18px;
                ">
                    <b>Peptide:</b>&nbsp;{info['peptide_id']}
                </div>
                """,
                sizing_mode="stretch_width",
                styles={
                    "min-width": "0",
                    "width": "100%",
                },
            ),
            sizing_mode="stretch_width",
            height=50,
            styles={
                "display": "flex",
                "align-items": "center",
                "background": "#f9f9f9",
                "padding": "0px",
                "border-bottom": "1px solid #ddd",
            },
        )

        parent_uniprot = _first_token(info.get("parent_protein", ""))
        if parent_uniprot.lower() in {"", "nan", "none"}:
            parent_uniprot = ""

        footer_left = pn.pane.HTML(
            "<span style='font-size: 12px;'>"
            f"Gene(s): <b>{info['gene_names']}</b>"
            f" &nbsp;|&nbsp; UniProt: <b>{parent_uniprot}</b>"
            f" &nbsp;|&nbsp; Peptide index: <b>{int(info['index']) + 1}</b>"
            "</span>"
        )

        string_link_for_footer = _cached_string_link(parent_uniprot) if parent_uniprot else ""
        footer_right = pn.pane.HTML(
            "" if not parent_uniprot else
            (
                f"<span style='font-size: 12px;'>"
                f"🔗 <a href='https://www.uniprot.org/uniprotkb/{parent_uniprot}/entry' "
                f"target='_blank' rel='noopener'>UniProt Entry</a>"
                f" &nbsp;|&nbsp; "
                f"<a href='https://www.ebi.ac.uk/interpro/protein/reviewed/{parent_uniprot}/' "
                f"target='_blank' rel='noopener'>InterPro Entry</a>"
                + (
                    f" &nbsp;|&nbsp; "
                    f"<a href='{string_link_for_footer}' target='_blank' rel='noopener'>STRING Entry</a>"
                    if string_link_for_footer else
                    ""
                )
                + "</span>"
            )
        )

        footer_links = pn.Row(
            footer_left,
            pn.Spacer(),
            footer_right,
            sizing_mode="stretch_width",
            styles={
                "justify-content": "space-between",
                "padding": "2px 8px 4px 0px",
                "margin-top": "-6px",
            },
        )

        hr = pn.Spacer(height=1, sizing_mode="stretch_width", styles={"background": "#ccc", "margin": "6px 0"})

        return pn.Card(
            header,
            pn.Row(*metrics_row_items, sizing_mode="stretch_width"),
            hr,
            footer_links,
            width=700,
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
            return pn.Spacer(width=1000, height=360, margin=(-30, 0, 0, 0))

        key = _normalize_search_token(protein)
        fig = plot_pelsa_curve(state, key, width=1300, height=360)

        return pn.pane.Plotly(
            fig,
            width=1300,
            height=360,
            margin=(0, 0, 10, 0),
            styles={
                "border-radius": "8px",
                "box-shadow": "3px 3px 5px #bcbcbc",
            },
        )

    info_holder = pn.Column()
    bar_holder  = pn.Column()
    pep_holder  = pn.Column()
    structure_holder = pn.Column()
    table_holder  = pn.Column()
    detail_mode_holder = pn.Column(
        pn.Spacer(width=840, height=320),
        width=840,
    )

    peptide_detail_panel = pn.Column(
        pn.Row(
            pn.Column(
                info_holder,
                pn.Spacer(height=20),
                table_holder,
                pn.Spacer(height=20),
                pep_holder,
                sizing_mode="stretch_width",
                styles={"align-items": "flex-start"},
            ),
            pn.Spacer(width=20),
            structure_holder,
            sizing_mode="stretch_width",
            styles={"align-items": "flex-start"},
        ),
        pn.Spacer(height=20),
        bar_holder,
        margin=(0, 20, 0, 0),
        sizing_mode="fixed",
        width=1300,
    )

    detail_panel = pn.Row(
        detail_mode_holder,
        margin=(0, 20, 0, 0),
        styles={
            "margin-left": "auto",
        }
    )

    bokeh_doc = pn.state.curdoc  # for next-tick scheduling

    def _sync_detail_mode() -> None:
        """
        Right-pane priority:
        1. clicked/searched peptide detail
        2. box/lasso STRING enrichment
        3. empty spacer
        """
        has_single_detail = bool(str(search_input.value or "").strip())
        structure_controls.visible = has_single_detail
        string_species_sel.visible = bool(string_selected_feature_ids)

        if has_single_detail:
            detail_mode_holder[:] = [peptide_detail_panel]
            detail_mode_holder.width = 1300
        elif string_selected_feature_ids:
            detail_mode_holder[:] = [string_enrichment_holder]
            detail_mode_holder.width = 840
        else:
            detail_mode_holder.width = 840
            detail_mode_holder[:] = [pn.Spacer(width=840, height=320)]

    def _render_info():
        # pass current values explicitly (protein, contrast)
        return info_card(search_input.value)

    def _structure_peptide_regions(peptide: str) -> list[PeptideRegion]:
        key = _normalize_search_token(peptide)
        try:
            sib = get_pelsa_sister_peptides(state, key)
        except Exception:
            return []

        if sib.empty or "peptide_id" not in sib.columns:
            return []

        out: list[PeptideRegion] = []
        for row in sib.itertuples(index=False):
            pid = str(getattr(row, "peptide_id", "")).strip()
            if not pid or pid not in adata.var.index:
                continue
            try:
                var_row = adata.var.loc[pid]
                out.append(PeptideRegion(
                    peptide_id=pid,
                    start=int(pd.to_numeric(var_row["PEPTIDE_START"], errors="raise")),
                    end=int(pd.to_numeric(var_row["PEPTIDE_END"], errors="raise")),
                    qval=float(getattr(row, "qval", np.nan)),
                    range_log2=float(getattr(row, "range_log2", np.nan)),
                    selected=(pid == key),
                ))
            except Exception:
                continue
        return out

    def _render_structure(peptide):
        if not peptide:
            return pn.Spacer(width=550, height=760)

        key = _normalize_search_token(peptide)
        try:
            info = get_pelsa_info(state, key)
        except Exception as exc:
            return pn.pane.Markdown(
                f"**Structure not available**  \n`{exc}`",
                width=400,
                height=120,
                styles={"background": "#fff3f3", "padding": "10px", "border-radius": "8px"},
            )

        parent = str(info.get("parent_protein", "") or "").strip()
        parent = parent.split(";", 1)[0].strip()

        return build_structure_viewer_pane(
            parent,
            peptides=_structure_peptide_regions(key),
            color_mode=structure_color_by.value,
            representation_mode=structure_representation.value,
            width=580,
            height=765,
        )

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
        # Peptide genomic/protein position metadata from adata.var.
        # Keep missing values as NA rather than failing the whole detail pane.
        starts = []
        for pid in disp["peptide_id"].astype(str):
            try:
                starts.append(int(pd.to_numeric(adata.var.loc[pid, "PEPTIDE_START"], errors="raise")))
            except Exception:
                starts.append(pd.NA)
        disp["Start Pos"] = pd.Series(starts, index=disp.index, dtype="Int64")
        disp["q-value"] = (
            pd.to_numeric(disp["qval"], errors="coerce")
            .map(lambda x: f"{x:.3e}" if np.isfinite(x) else "nan")
        )

        disp["EC₅₀"] = pd.to_numeric(disp["ec50"], errors="coerce")

        # Display adjacent peptides in protein-coordinate order.
        # Keep missing positions at the bottom, then stabilize by peptide id.
        disp = (
            disp.sort_values(["Start Pos", "peptide_id"], ascending=[True, True], na_position="last")
            .reset_index(drop=True)
        )

        sig = pd.to_numeric(disp["qval"], errors="coerce") < 0.05
        ranges = pd.to_numeric(disp["range_log2"], errors="coerce")
        colors = np.where(
            (ranges > 0) & sig,
            "red",
            np.where((ranges < 0) & sig, "blue", "gray"),
        )
        disp["Peptide_html"] = [
            f"<span style='color:{color}'>{peptide}</span>"
            for color, peptide in zip(colors, disp["Peptide"])
        ]

        styled = (
            disp[["Peptide_html", "Start Pos", "Range log₂", "EC₅₀", "q-value", "peptide_id", "current"]]
            .rename(columns={"Peptide_html": "Peptide"})
            .style
            .apply(
                lambda row: ["background-color: rgba(255,235,59,0.35)"] * len(row)
                if bool(row["current"]) else [""] * len(row),
                axis=1,
            )
            .format({"Range log₂": "{:.3f}", "EC₅₀": "{:.3f}", "Start Pos": "{:.0f}"})
        )

        sibling_ids = disp["peptide_id"].astype(str).tolist()

        row_h, header_h = 32, 30
        nrows = len(disp)
        visible = max(min(int(nrows), 4), 1)
        table_h = row_h * (4 if nrows > 4 else visible) + header_h

        tbl = pn.widgets.Tabulator(
            styled,
            formatters={
                "Peptide": {"type": "html"},
                "Range log₂": NumberFormatter(format="0.000"),
                "EC₅₀": NumberFormatter(format="0.000"),
            },
            hidden_columns=["peptide_id", "current"],
            selectable=1,
            show_index=False,
            layout="fit_columns",
            disabled=True,
            height=table_h,
            pagination=None,
            width=660,
            widths={"Peptide": 250, "Start Pos": 85, "Range log₂": 85, "EC₅₀": 65, "q-value": 90},
            configuration={
                "rowHeight": row_h,
                "columnHeaderVertAlign": "bottom",
                "movableColumns": False,
                "columnDefaults": {"editor": False, "headerSort": False},
            },
            margin=(-5, 8, 8, 8),
        )

        def _on_select(event):
            sel = event.new
            if sel:
                picked = disp.iloc[sel[0]]["peptide_id"]
                search_input.value = str(picked)

        tbl.param.watch(_on_select, "selection")

        def _adjacent_peptides_csv() -> bytes:
            if not sibling_ids:
                raise ValueError("No adjacent PELSA peptides to export.")
            df_export = build_pelsa_selection_df(
                state=state,
                contrast="pelsa_curve_fit",
                feature_ids=sibling_ids,
                uniprot_var_col="PARENT_PROTEIN",
                id_col_name="PEPTIDE",
            )
            return io.BytesIO(df_export.to_csv(index=False).encode("utf-8"))

        download_adjacent = pn.widgets.FileDownload(
            label="Download",
            callback=_adjacent_peptides_csv,
            filename="proteoflux_adjacent_peptides.csv",
            button_type="success",
            visible=True,
            margin=(5, 10, 0, 0),
        )

        header = pn.Row(
            pn.pane.Markdown(
                "**Adjacent peptides**",
                styles={"font-size": "16px", "padding": "0", "line-height": "0px"},
            ),
            pn.Spacer(sizing_mode="stretch_width"),
            download_adjacent,
            sizing_mode="stretch_width",
        )
        return pn.Card(
            header,
            make_hr(),
            tbl,
            width=700,
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
            return pn.Spacer(width=700, height=260)

        key = _normalize_search_token(peptide)
        fig = plot_pelsa_local_stability_profile(state, key, width=700, height=330)
        pane = pn.pane.Plotly(
            fig,
            width=700,
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

    def _render_table():
        peptide = search_input.value
        if not peptide:
            return pn.Spacer(width=380, height=320)

        return pn.Column(
            _render_sister_peptide_table(peptide),
            width=360,
            margin=(0, 0, 0, 0),
        )

    def _render_pep():
        peptide = search_input.value
        if not peptide:
            return pn.Spacer(width=380, height=320)

        return pn.Column(
            _render_local_stability_profile(peptide),
            width=320,
            margin=(0, 0, 0, 0),
        )

    def _update_info(_=None):
        # No spinner here; it's cheap and we don't want a loader on empty states
        info_holder[:] = [_render_info()]
        _sync_detail_mode()

    def _update_bar(_=None):
        protein = search_input.value
        if not protein:
            bar_holder.loading = False
            bar_holder[:] = [pn.Spacer(width=1000, height=500, margin=(-30, 0, 0, 0))]
            return

        # Only show a loader if there IS a protein selected (i.e., real work)
        bar_holder.loading = True
        try:
            bar_holder[:] = [_render_bar()]
        finally:
            bar_holder.loading = False
        _sync_detail_mode()

    def _update_table(_=None):
        if not search_input.value:
            table_holder.loading = False
            table_holder[:] = [pn.Spacer(width=400, height=320)]

            return
        table_holder.loading = True
        try:
            table_holder[:] = [_render_table()]
        finally:
            table_holder.loading = False
        _sync_detail_mode()

    def _update_pep(_=None):
        if not search_input.value:
            pep_holder.loading = False
            pep_holder[:] = [pn.Spacer(width=400, height=320)]

            return
        pep_holder.loading = True
        try:
            pep_holder[:] = [_render_pep()]
        finally:
            pep_holder.loading = False
        _sync_detail_mode()

    def _update_structure(_=None):
        if not search_input.value:
            structure_holder.loading = False
            structure_holder[:] = [pn.Spacer(width=600, height=800)]
            return

        structure_holder.loading = True
        try:
            structure_holder[:] = [_render_structure(search_input.value)]
        finally:
            structure_holder.loading = False
        _sync_detail_mode()

    def _update_detail(_=None):
        _update_info()
        _update_table()
        _update_pep()
        _update_bar()
        _update_structure()

    # Wire events:
    search_input.param.watch(_update_detail, "value")

    structure_color_by.param.watch(_update_structure, "value")
    structure_representation.param.watch(_update_structure, "value")

    # Initial fill (after the page paints so we don’t see a flash) - removed because fine ? check later

    # assemble into a layout, no legend‐based toggles
    volcano_and_detail = pn.Row(
        pn.Column(                 # left container that can stretch
            volcano_plot,
            sizing_mode="stretch_width",
            styles={
                "flex": "1.3",
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
                margin=(25, 0, 0, 0),
                width=175,
            ),
            pn.Spacer(width=20),
            make_vr(),
            pn.Spacer(width=30),
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
            pn.Spacer(width=20),
            download_selection,
            pn.Spacer(width=20),
            structure_controls,
            string_species_sel,
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

    return layout
