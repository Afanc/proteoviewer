import os
import panel as pn
from functools import lru_cache
from session_state import SessionState
from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_volcanoes_wrapper,
    plot_intensity_by_protein,
    get_protein_info,
    plot_peptide_trends_centered,
    resolve_pattern_to_uniprot_ids,
    plot_group_violin_for_volcano,
)
from components.plot_utils import plot_pca_2d, plot_umap_2d
from components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from components.string_links import get_string_link
from layout_utils import plotly_section, make_vr, make_hr, make_section, make_row, FRAME_STYLES
from utils import logger, log_time
import textwrap

def _fmt_files_list(files, max_items=6):
    """Return bullet lines for files, truncated to max_items with a '+N more' line."""
    if not files:
        return []
    short = [f"  - {os.path.basename(str(f))}" for f in files[:max_items]]
    rest = max(0, len(files) - max_items)
    if rest:
        short.append(f"  - … (+{rest} more)")
    return short

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
    analysis_type  = preproc_cfg.get("analysis_type", "DIA")
    ebayes_method  = analysis_cfg.get("ebayes_method", "limma")
    input_layout  = preproc_cfg.get("input_layout", "")

    num_samples = len(adata.obs.index.unique())
    num_conditions = len(adata.obs["CONDITION"].unique())
    num_contrasts = int(num_conditions*(num_conditions-1)/2)
    quant_method = preproc_cfg.get("quantification_method", "sum")
    flt_cfg = adata.uns.get("preprocessing", {}).get("filtering", [])
    n_cont   = flt_cfg.get("cont", {}).get("number_dropped", [])
    n_q   = flt_cfg.get("qvalue", {}).get("number_dropped", [])
    n_pep   = flt_cfg.get("pep", {}).get("number_dropped", [])
    n_re   = flt_cfg.get("rec", {}).get("number_dropped", [])
    thresh_q = flt_cfg.get("qvalue", {}).get("threshold", 0)
    thresh_pep = flt_cfg.get("pep", {}).get("threshold", 0)
    thresh_re = flt_cfg.get("rec", {}).get("threshold", 0)

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
    rec_step  = flt_cfg.get("rec", {})

    cont_txt, _          = _fmt_step(cont_step, "cont", "n/a")
    q_txt,    q_thr_txt  = _fmt_step(q_step,    "qvalue", "n/a")
    pep_txt,  pep_thr_txt= _fmt_step(pep_step,  "pep", "n/a")
    pep_op = "≥" if flt_cfg.get("pep").get("direction").startswith("greater") else "≤"
    rec_txt,  rec_thr_txt= _fmt_step(rec_step,  "rec", "n/a")

    contaminants_files = [os.path.basename(p) for p in flt_cfg.get('cont', {}).get('files', [])]

    # Norm condensation
    norm_methods = normalization.get("method", []).tolist()
    #if isinstance(norm_methods, list):
    #    norm_methods = "+".join(norm_methods)
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

    if "lc_conmed" in imp_method:
        lc_conmed_lod_k = preproc_cfg.get("imputation").get("lc_conmed_lod_k")
        imp_method += f", lod_k={lc_conmed_lod_k}"


    # build a single Markdown string
    summary_md = textwrap.dedent(f"""

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
        - **Normalization**: {norm_methods}
        - **Imputation**: {imp_method}
        - **Differential expression**: eBayes via {ebayes_method}
    """).strip()

    # intro_pane:
    summary_pane = pn.pane.Markdown(summary_md,
        sizing_mode="stretch_width",
        margin=(-10, 0, 0, 20),
        styles={
            "line-height":"1.4em",
            #"white-space": "pre-wrap",
            "word-break": "break-word",
            "overflow-wrap": "anywhere",
            "min-width": "0",
        }
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

    # translate widget → function arg
    def _sort_arg(mode: str) -> str:
        return "condition" if mode == "By condition" else "sample"

    hist_ID_dmap = pn.bind(
        plot_barplot_proteins_per_sample,
        adata=adata,
        sort_by=pn.bind(_sort_arg, id_sort_toggle),
    )

    hist_plot_pane = pn.pane.Plotly(hist_ID_dmap,
                       height=500,
                       margin=(-20, 20, 0, -190),
                       styles={"flex":"1",
                              }
    )

    intro_pane = pn.Row(
        pn.Column(
            pn.pane.Markdown("##   Summary"),
            summary_pane,
            styles={"flex":"0.32", "min-width": "0"}
        ),
        make_vr(),
        pn.Spacer(width=20),
        id_sort_toggle,
        hist_plot_pane,
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
        }
    )

    ## Metric Violins
    cv_fig, rmad_fig = plot_violin_cv_rmad_per_condition(adata)
    rmad_pane = pn.pane.Plotly(rmad_fig, height=500, sizing_mode="stretch_width",
                               styles={"flex":"1"}, config={'responsive':True},
                               margin=(0,0,0,-100))
    cv_pane = pn.pane.Plotly(cv_fig, height=500, sizing_mode="stretch_width",
                               styles={"flex":"1"}, config={'responsive':True})

    metrics_pane = pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex":"0.1", "z-index": "10"}),
        rmad_pane,
        pn.Spacer(width=25),
        make_vr(),
        pn.Spacer(width=25),
        cv_pane,
        pn.Spacer(width=50),
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
        },
    )
    ## UMAP and PCA
    pca_pane = pn.pane.Plotly(plot_pca_2d(state.adata),
                              height=500,
                              sizing_mode="stretch_width",
                              styles={"flex":"1"},
                              margin=(0,0,0,-100),
                              )
    umap_pane = pn.pane.Plotly(plot_umap_2d(state.adata),
                               height=500,
                               sizing_mode="stretch_width",
                               styles={"flex":"1"})
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
            styles={
                'border-radius':  '15px',
                'box-shadow':     '3px 3px 5px #bcbcbc',
                'width': '98vw',
            }
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
    color_options = ["Significance", "Avg Expression"]

    color_by = pn.widgets.Select(
        name="Color by",
        options=color_options,
        value=color_options[0],
        width=200,
    )
    # 3) Function to update toggle labels whenever contrast changes
    def _update_toggle_labels(event=None):
        grp1, grp2 = contrast_sel.value.split("_vs_")
        show_imp_cond1.name = f"▲ Fully Imputed in {grp1}"
        show_imp_cond2.name = f"▼ Fully Imputed in {grp2}"

    # initialize labels and watch for changes
    _update_toggle_labels()

    contrast_sel.param.watch(_update_toggle_labels, "value")

    # search widget
    search_input = pn.widgets.AutocompleteInput(
        name="Search Protein",
        options=list(state.adata.var["GENE_NAMES"]),
        placeholder="Type gene name…",
        width=200,
    )

    clear_search = pn.widgets.Button(name="Clear", width=80)
    clear_search.on_click(lambda event: setattr(search_input, "value", ""))

    search_field_sel = pn.widgets.Select(
        name="Field",
        options=["FASTA headers", "Gene names", "UniProt IDs"],
        value="FASTA headers",
        width=130,
        styles={"z-index": "10"},
    )

    search_input_group = pn.widgets.TextInput(
        name="Cohort", placeholder="e.g. *_ECOLI* or gene[0-9]+",
        width=200, styles={"z-index": "10"}
    )
    clear_group = pn.widgets.Button(name="Clear", width=80)
    clear_group.on_click(lambda event: setattr(search_input_group, "value", ""))

    # Turn (pattern, field) → sorted list of UniProt IDs using the shared helper
    def _group_ids(pattern, field):
        #pattern='ECOLI'
        try:
            ids = resolve_pattern_to_uniprot_ids(state.adata, field, pattern)
            return sorted(ids)
        except Exception:
            return []

    group_ids_dmap = pn.bind(_group_ids, search_input_group, search_field_sel)
    def _fmt_group_count(ids):
        if not ids:
            return ""
        return f"({len(ids)} match{'es' if len(ids)!=1 else ''})"

    group_count_text = pn.bind(_fmt_group_count, group_ids_dmap)
    group_count_md   = pn.pane.Markdown(group_count_text,
                                        styles={"min-width":"80px"},
                                        margin=(-10,0,0,120))
    volcano_dmap = pn.bind(
        plot_volcanoes_wrapper,
        state=state,
        contrast=contrast_sel,
        color_by=color_by,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        highlight=search_input,
        highlight_group=group_ids_dmap,
        sign_threshold=0.05,
        width=None,
        height=900,
    )

    def _on_volcano_click(event):
        click = event.new        # the new click_data dict
        if click and click.get("points"):
            gene = click["points"][0]["text"]
            search_input.value = gene

    def _with_uirevision(fig, contrast):
        fig.update_layout(uirevision=f"volcano-{contrast}")
        return fig

    volcano_dmap_wrapped = pn.bind(_with_uirevision, volcano_dmap, contrast_sel)

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

    volcano_pane_height = 1060
    # Cohort Violin View
    def _cohort_violin(ids, contrast, sm, s1, s2):
        if not ids:
            return pn.Spacer(height=0)  # collapses cleanly when no cohort
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
        volcano_pane_height = 1200
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
        group_ids_dmap,        # ids
        contrast_sel,          # contrast
        show_measured,         # sm
        show_imp_cond1,        # s1
        show_imp_cond2,        # s2
    )

    # bind a detail‐plot function to the same contrast & search_input
    layers = ["Processed", "Log (pre-norm)", "Raw"]

    layers_sel = pn.widgets.Select(
        name="Protein Data Layer",
        options=layers,
        value=layers[0],
        width=100,
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

        # --- pull values that do not depend on the "layer" toggle ---
        protein_info = get_protein_info(state, contrast, protein, layers_sel)  # OK: we only read layer-agnostic bits
        uniprot_id   = protein_info['uniprot_id']
        idx = protein_info['index']  # already used below for "Protein Index"

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
        ibaq_raw = _safe_var("IBAQ", default=None)

        re_count = _fmt_int(rec_val) if rec_val is not None else "n/a"
        ibaq_val = _fmt_ibaq(ibaq_raw) if ibaq_raw is not None else "n/a"

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

        # we will inject the layer-dependent intensity Number into _intensity_slot elsewhere
        _intensity_slot[:] = [pn.Spacer(height=0)]  # keeps layout tidy until bar function runs

        # --- header (unchanged) ---
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
        gene_md = pn.pane.Markdown(f"**Gene(s)**: {protein}", styles=item_styles)
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

        left_bits = []
        if rec_val is not None:
            left_bits.append(f"Precursors (global): <b>{re_count}</b>")
        if ibaq_raw is not None:
            left_bits.append(f"iBAQ: <b>{ibaq_val}</b>")
        # If neither is available, show a friendly placeholder
        if not left_bits:
            left_bits.append("No extra protein metrics available")

        footer_left = pn.pane.HTML(
            "<span style='font-size: 12px;'>" + " &nbsp;|&nbsp; ".join(left_bits) + "</span>"
        )

        footer_right = pn.pane.HTML(
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

        # IMPORTANT: keep the exact row with 3 slots (q, lfc, INTENSITY_SLOT)
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

        # --- bar plot (unchanged) ---
        fig = plot_intensity_by_protein(state, contrast, protein, layers_sel)
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

        # --- intensity Number (layer-dependent) ---
        protein_info   = get_protein_info(state, contrast, protein, layers_sel)
        intensity_scale = "Avg Log Intensity" if layer != "Raw" else "Avg Intensity"

        # NOTE: preserve your original formatting logic exactly
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
            sizing_mode="fixed",
            width=840,
        ),
        margin=(0, 0, 0, 0),
        sizing_mode="fixed",
        styles={
            "margin-left": "auto",     # ← push this block to the right
        }
    )

    bokeh_doc = pn.state.curdoc  # for next-tick scheduling if you want it

    def _current_uniprot_id():
        gene = search_input.value
        if not gene:
            return None
        # cheap call; we only need the ID
        info = get_protein_info(state, contrast_sel.value, gene, layers_sel)
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
            # No protein selected → no spinner, show placeholder and clear intensity slot
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

    # Initial fill (after the page paints so you don’t see a flash)
    bokeh_doc.add_next_tick_callback(lambda: (_update_info(), _update_bar(), _update_pep()))

    # 3) assemble into a layout, no legend‐based toggles
    volcano_and_detail = pn.Row(
        pn.Column(                 # left container that can stretch
            volcano_plot,
            cohort_violin_view,
            sizing_mode="stretch_both",
            styles={
                "flex": "1",               # ← soak up remaining horizontal space
                "min-width": "600px",      # ← keep a sane minimum for the plot
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
            pn.Spacer(width=30),
            pn.Column(
                show_measured,
                show_imp_cond1,
                show_imp_cond2,
                margin=(-5,0,0,0),
            ),
            pn.Spacer(width=20),
            make_vr(),
            pn.Spacer(width=10),
            search_input,
            pn.Row(clear_search, margin = (17,0,0,0)),
            pn.Spacer(width=20),
            make_vr(),
            pn.Spacer(width=20),
            search_field_sel,
            pn.Column(search_input_group, group_count_md),
            pn.Row(clear_group, margin = (17,0,0,0)),
            pn.Spacer(width=20),
            pn.Row(layers_sel, margin = (-17,0,0,0), ),
            sizing_mode="fixed",
            width=300,
            height=70,
        ),
        volcano_and_detail,
        height=1060,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
        }
    )
    volcano_pane.height = pn.bind(lambda ids: 1200 if ids else 1060, group_ids_dmap)

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
        sizing_mode="stretch_both",
        styles=FRAME_STYLES,
    )

    return layout
