import panel as pn
from session_state import SessionState
from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_h_clustering_heatmap,
    plot_volcanoes_wrapper,
    plot_intensity_by_protein
)
from components.plot_utils import plot_pca_2d, plot_umap_2d
from components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from utils import logger, log_time

pn.extension("plotly")


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

    ## IDs barplot
    hist_ID_fig = plot_barplot_proteins_per_sample(adata)

    ## Metric Violins
    cv_fig, rmad_fig = plot_violin_cv_rmad_per_condition(adata)
    rmad_pane = pn.pane.Plotly(rmad_fig, height=500, sizing_mode="stretch_width")
    cv_pane   = pn.pane.Plotly(cv_fig,  height=500, sizing_mode="stretch_width")

    ## UMAP and PCA
    pca_pane = pn.pane.Plotly(plot_pca_2d(state.adata),
                              height=500,
                              sizing_mode="stretch_width")
    umap_pane = pn.pane.Plotly(plot_umap_2d(state.adata),
                               height=500,
                               sizing_mode="stretch_width")

    #h_clustering_pane = pn.pane.Plotly(plot_h_clustering_heatmap(adata),
    ##h_clustering_pane = pn.pane.Matplotlib(plot_h_clustering_heatmap(adata),
    #                                   height=800,
    #                                   sizing_mode="stretch_width")

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
    color_by.param.watch(_update_toggle_labels, "value")

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
            search_input.value = gene   # reuse your existing search box

    volcano_plot = pn.pane.Plotly(
        volcano_dmap,
        width=900, height=800,
        margin=(-50, 0, 0, 0),
        sizing_mode="fixed",
    )
    volcano_plot.param.watch(_on_volcano_click, "click_data")

    # bind a detail‐plot function to the same contrast & search_input
    @pn.depends(protein=search_input, contrast=contrast_sel)
    def detail_panel(protein, contrast):
        if not protein:
            # an inert box *exactly* the size of your eventual plot
            return pn.Spacer(width=400, height=350)
        # otherwise build & return the real Plotly pane
        fig = plot_intensity_by_protein(state, contrast, protein)
        return pn.pane.Plotly(fig, width=400, height=350)

    # 3) assemble into a layout, no legend‐based toggles
    volcano_pane = pn.Column(
        pn.Row(
            contrast_sel,
            pn.Spacer(width=50),
            color_by,
            pn.Spacer(width=50),
            pn.Row(
                show_measured,
                show_imp_cond1,
                show_imp_cond2,
                margin=(25,0,0,0),
            ),
            pn.Spacer(width=300),
            search_input,
            pn.Row(clear_search, margin = (17,0,0,0)),
            sizing_mode="fixed",
            width=300,
            height=160,
        ),
        pn.Row(
            volcano_plot,
            #pn.panel(volcano_dmap,
            #width=900,
            #height=800,
            #margin=(-50, 0, 0, 0),
            #sizing_mode="fixed",
            #max_states=20,
            #),
            #detail_pane,
            #detail_container,
            detail_panel,
        ),
        width=1200,
        height=900,
        sizing_mode="fixed",
    )

    # Texts
    intro_text = pn.pane.Markdown("some config text",
        width=1200,
        margin=(10, 0, 15, 0),
    )

    # Tab layout

    layout = pn.Column(
        pn.pane.Markdown("#   Summary of analysis"),
        intro_text,
        pn.pane.Markdown("##   Proteins Identified"),
        hist_ID_fig,
        pn.pane.Markdown("##   Metrics per Condition"),
        pn.Row(rmad_pane, cv_pane, sizing_mode="stretch_width"),
        pn.pane.Markdown("##   Clustering"),
        pn.Row(pca_pane, umap_pane, sizing_mode="stretch_width"),
        #h_clustering_pane,
        pn.pane.Markdown("##   Volcano plots"),
        volcano_pane,

        sizing_mode="stretch_both",
    )

    return layout
