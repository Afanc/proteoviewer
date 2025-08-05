import os
import panel as pn
from session_state import SessionState
from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_h_clustering_heatmap,
    plot_volcanoes_wrapper,
    plot_intensity_by_protein,
    get_protein_info
)
from components.plot_utils import plot_pca_2d, plot_umap_2d
from components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from components.string_links import get_string_link
from utils import logger, log_time
import textwrap

pn.extension("plotly")
pn.extension("indicator")

def make_vr(color="#ccc", margin="6px 0"):
    return pn.Spacer(
        width=1,
        sizing_mode="stretch_height",
        styles={"background": color, "margin": margin}
        )

def make_hr(color="#ccc", margin="6px 0"):
    return pn.Spacer(
        height=1,
        sizing_mode="stretch_width",
        styles={"background": color, "margin": margin}
        )


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
    ebayes_method  = analysis_cfg.get("ebayes_method", "limma")
    analysis_type  = analysis_cfg.get("analysis_type", "")


    # numbers of PSM removed (as you do in PDF)
    flt_cfg = adata.uns.get("preprocessing", {}).get("filtering", [])
    n_cont   = flt_cfg.get("cont", {}).get("number_dropped", [])
    n_q   = flt_cfg.get("qvalue", {}).get("number_dropped", [])
    n_pep   = flt_cfg.get("pep", {}).get("number_dropped", [])
    n_re   = flt_cfg.get("rec", {}).get("number_dropped", [])
    thresh_q = flt_cfg.get("qvalue", {}).get("threshold", 0)
    thresh_pep = flt_cfg.get("pep", {}).get("threshold", 0)
    thresh_re = flt_cfg.get("rec", {}).get("threshold", 0)

    contaminants_files = [os.path.basename(p) for p in flt_cfg.get('cont', {}).get('files', [])]

    # Norm condensation
    norm_methods = normalization.get("method", [])
    if isinstance(norm_methods, list):
        norm_methods = "+".join(norm_methods)
    if "loess" in norm_methods:
        loess_span = preproc.get("normalization").get("loess_span")
        norm_methods += f" (loess_span={loess_span})"

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
        rf_max_iter = preproc.get("imputation").get("rf_max_iter")
        imp_method += f", rf_max_iter={rf_max_iter}"


    # build a single Markdown string
    summary_md = textwrap.dedent(f"""
        **Analysis Type**: {analysis_type}

        **Pipeline steps**
        - **Filtering**:
            - Contaminants ({', '.join(contaminants_files)}): {n_cont:,} PSM removed
            - q-value ≤ {thresh_q}: {n_q:,} PSM removed
            - PEP ≤ {thresh_pep}: {n_pep:,} PSM removed
            - Min. run evidence count = {thresh_re}: {n_re:,} PSM removed
        - **Normalization**: {norm_methods}
        - **Imputation**: {imp_method}
        - **Differential expression**: eBayes via {ebayes_method}
    """).strip()

    # intro_pane:
    summary_pane = pn.pane.Markdown(summary_md,
        sizing_mode="stretch_width",
        margin=(-10, 0, 0, 20),
        styles={
            "line-height":"1.4em"
            #"white-space": "pre-wrap",
        }
    )

    hist_ID_fig = plot_barplot_proteins_per_sample(adata)
    vr = pn.Spacer(width=1, sizing_mode="stretch_height", styles={
            "background": "#ccc",
            "margin": "6px 0"
        })

    intro_pane = pn.Row(
        pn.Column(
            pn.pane.Markdown("##   Summary"),
            summary_pane,
            styles={"flex":"0.32"}
        ),
        vr,
        pn.Spacer(width=20),
        pn.pane.Plotly(hist_ID_fig,
                       height=500,
                       #sizing_mode="stretch_width",
                       #styles={"flex":"0.8"}
                       margin=(0, 20, 0, 0),
                       styles={"flex":"1",
                               #'border-radius':  '15px',
                               #'box-shadow':     '3px 3px 5px #bcbcbc',
                              }
        ),
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="fixed",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
        }
    )

    ## IDs barplot
    hist_ID_fig = plot_barplot_proteins_per_sample(adata)
    hist_pane = pn.Row(
            pn.pane.Markdown("##   Identifications",
                               styles={"flex":"0.1"}),
            hist_ID_fig,
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
                               styles={"flex":"1"}, config={'responsive':True})
    cv_pane = pn.pane.Plotly(cv_fig, height=500, sizing_mode="stretch_width",
                               styles={"flex":"1"}, config={'responsive':True})

    metrics_pane = pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex":"0.1"}),
        rmad_pane,
        pn.Spacer(width=25),
        vr,
        pn.Spacer(width=25),
        cv_pane,
        pn.Spacer(width=50),
        #width=1400,
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="fixed",
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
                              )
    umap_pane = pn.pane.Plotly(plot_umap_2d(state.adata),
                               height=500,
                               sizing_mode="stretch_width",
                               styles={"flex":"1"})
    #h_clustering_pane = pn.pane.Plotly(
    #    plot_h_clustering_heatmap(adata),
    #    height=400,
    #    sizing_mode="stretch_width",
    #)
    #h_clustering_pane = pn.pane.Matplotlib(plot_h_clustering_heatmap(adata),
    #                                   height=800,
    #                                   sizing_mode="stretch_width")
    clustering_pane = pn.Row(
            pn.pane.Markdown("##   Clustering", styles={"flex": "0.1"}),
            pca_pane,
            #pn.Spacer(width=25),
            vr,
            pn.Spacer(width=60),
            umap_pane,
            vr,
            #h_clustering_pane,
            height=530,
            margin=(0, 0, 0, 20),
            sizing_mode="stretch_width",
            styles={
                'border-radius':  '15px',
                'box-shadow':     '3px 3px 5px #bcbcbc',
                'width': '98vw',
            }
        )

    #clustering_pane = pn.Row(
    #        pn.pane.Markdown("##   Clustering", styles={"flex": "0.1"}),
    #        pca_pane,
    #        #pn.Spacer(width=25),
    #        vr,
    #        pn.Spacer(width=60),
    #        umap_pane,
    #        vr,
    #        h_clustering_pane,
    #        #width=2400,
    #        #height=530,
    #        height=530,
    #        margin=(0, 0, 0, 20),
    #        sizing_mode="stretch_width",
    #        styles={
    #            'border-radius':  '15px',
    #            'box-shadow':     '3px 3px 5px #bcbcbc',
    #            'width': '98vw',
    #        }
    #    )


    ## Volcanoes
    # Contrast selector
    contrasts = state.adata.uns["contrast_names"].tolist()

    contrast_sel = pn.widgets.Select(
        name="Contrast",
        options=contrasts,
        value=contrasts[0],
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
    #color_by.param.watch(_update_toggle_labels, "value")

    # search widget
    search_input = pn.widgets.AutocompleteInput(
        name="Search Protein",
        options=list(state.adata.var["GENE_NAMES"]),
        placeholder="Type gene name…",
    )

    clear_search = pn.widgets.Button(name="Clear Search", width=100)
    clear_search.on_click(lambda event: setattr(search_input, "value", ""))

    volcano_dmap = pn.bind(
        plot_volcanoes_wrapper,
        state=state,
        contrast=contrast_sel,
        color_by=color_by,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        highlight=search_input,
        sign_threshold=0.05,
        width=900,
        height=800,
    )

    def _on_volcano_click(event):
        click = event.new        # the new click_data dict
        if click and click.get("points"):
            gene = click["points"][0]["text"]
            search_input.value = gene

    volcano_plot = pn.pane.Plotly(
        volcano_dmap,
        width=900,
        height=800,
        margin=(-50, 0, 0, 20),
        sizing_mode="fixed",
        styles={
            'border-radius':  '8px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
        }
    )
    volcano_plot.param.watch(_on_volcano_click, "click_data")

    # bind a detail‐plot function to the same contrast & search_input
    layers = ["Processed", "Log-Normalized", "Raw"]

    layers_sel = pn.widgets.Select(
        name="Data Layer",
        options=layers,
        value=layers[0],
        width=100,
        margin=(20, 0, 0, 20),
    )
    def _toggle_layers_visibility(event):
        layers_sel.visible = bool(event.new)

    search_input.param.watch(_toggle_layers_visibility, "value")
    layers_sel.visible = bool(search_input.value)

    @pn.depends(protein=search_input, contrast=contrast_sel, layer=layers_sel)
    def detail_panel(protein, contrast, layer):
        if not protein:
            # an inert box *exactly* the size of your eventual plot
            return pn.Spacer(width=900, height=800)

        # otherwise build & return the real Plotly pane

        fig = plot_intensity_by_protein(state, contrast, protein, layers_sel)
        barplot_pane = pn.pane.Plotly(fig,
                                      width=800,
                                      height=500,
                                      margin=(-30, 0, 0, 0),
                                      styles={
                                                  'border-radius':  '8px',
                                                  'box-shadow':     '3px 3px 5px #bcbcbc',
                                              }
                                      )
        protein_info = get_protein_info(state, contrast, protein, layers_sel)
        uniprot_id = protein_info['uniprot_id']

        intensity_scale = "Avg Log Intensity"
        if layer == "Raw":
            intensity_scale = "Avg Intensity"

        Number = pn.indicators.Number

        # create one Number widget per metric
        q_ind = Number(
            name="q-value",
            value=protein_info["qval"],
            format="{value:.3e}",
            default_color="red",
            font_size="12pt",
            styles= {'flex': '1'}
            )
        lfc_ind = Number(
            name="log₂ FC",
            value=protein_info["logfc"],
            format="{value:.3f}",
            default_color="red",
            font_size= "14pt",
            styles= {'flex': '1'}
        )
        prot_avg_val = protein_info["avg_int"]
        prot_avg_val = f"{prot_avg_val:.3f}" if prot_avg_val <= 100 else f"{prot_avg_val:.0f}"
        int_ind = Number(
            name=intensity_scale,
            value=protein_info["avg_int"],
            format=prot_avg_val,
            default_color="darkorange",
            font_size= "16pt",
            styles= {'flex': '1'}
        )

        base_size = 18
        max_len   = 10
        length    = len(protein)
        top_padding = 6
        if length <= max_len:
            size = base_size
        else:
            # scale down linearly, but clamp at 10px minimum
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
        # header with gene name + UniProt
        gene_md = pn.pane.Markdown(f"**Gene(s)**: {protein}", styles=item_styles)
        sep1    = pn.pane.Markdown("|", styles=sep_styles)
        uid_md  = pn.pane.Markdown(f"**Uniprot ID**: {uniprot_id}", styles=item_styles)
        sep2    = pn.pane.Markdown("|", styles=sep_styles)
        idx_md  = pn.pane.Markdown(f"**Protein Index**: {protein_info['index']+1}",
                                   styles=item_styles) #correct for non-pythony users

        # 3) Flex container that centers everything
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

        try:
            string_link = get_string_link(uniprot_id)
        except:
            string_link = ""

        footer_links = pn.Row(
            pn.pane.HTML(
                f'<span style="font-size: 12px;">'
                f'🔗 <a href="https://www.uniprot.org/uniprotkb/{uniprot_id}/entry" target="_blank" rel="noopener">UniProt Entry</a>'
                f' &nbsp;|&nbsp; '
                f'<a href="{string_link}" target="_blank" rel="noopener">STRING Entry</a>'
                f'</span>'
            ),
            sizing_mode="stretch_width",
            styles={
                "justify-content": "flex-end",
                "padding": "2px 8px 4px 0px",
                "margin-top": "-6px",  # Optional: pull up slightly to hug bottom
            }
        )

        hr = pn.Spacer(height=1, sizing_mode="stretch_width", styles={
            "background": "#ccc",
            "margin": "6px 0"
        })

        card_style = {
            'background':     '#f9f9f9',       # light gray, like Plotly default
            "align-items":     "center",
            'border-radius':  '8px',
            'text-align': "center",
            'padding':        '5px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'justify-content': 'space-evenly',
        }

        # assemble them in a Card
        info_card = pn.Card(
            header,
            pn.Row(q_ind, lfc_ind, int_ind, sizing_mode="stretch_width"),
            hr,
            footer_links,
            width=800,
            styles=card_style,
            collapsible=False,
            hide_header=True,
        )

        #detailed_pane = pn.Row(
        detailed_pane = pn.Column(
            info_card,
            pn.Spacer(height=50),
            barplot_pane,
        )

        return detailed_pane

    # 3) assemble into a layout, no legend‐based toggles
    volcano_pane = pn.Column(
        pn.pane.Markdown("##   Volcano plots"),
        pn.Row(
            contrast_sel,
            pn.Spacer(width=40),
            color_by,
            pn.Spacer(width=40),
            pn.Row(
                show_measured,
                show_imp_cond1,
                show_imp_cond2,
                margin=(25,0,0,0),
            ),
            pn.Spacer(width=240),
            search_input,
            pn.Row(clear_search, margin = (17,0,0,0)),
            pn.Spacer(width=30),
            pn.Row(layers_sel, margin = (-17,0,0,0), ),
            sizing_mode="fixed",
            width=300,
            height=160,
        ),
        pn.Row(
            volcano_plot,
            pn.Spacer(width=50),
            detail_panel,
        ),
        #width=2400,
        height=1000,
        margin=(0, 0, 0, 20),
        sizing_mode="fixed",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
        }

    )

    # Tab layout

    layout = pn.Column(
        pn.Spacer(height=10),
        intro_pane,
        #pn.Spacer(height=30),
        #hist_pane,
        pn.Spacer(height=30),
        metrics_pane,
        pn.Spacer(height=30),
        clustering_pane,
        pn.Spacer(height=30),
        volcano_pane,

        sizing_mode="stretch_both",
    )

    return layout
