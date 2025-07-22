import panel as pn
from session_state import SessionState
from components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_h_clustering_heatmap,
    plot_volcanoes_wrapper
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
    h_clustering_pane = pn.pane.Matplotlib(plot_h_clustering_heatmap(adata),
                                       height=800,
                                       sizing_mode="stretch_width")

    ## Volcanoes
    contrasts = state.adata.uns["contrast_names"].tolist()

    contrast_sel = pn.widgets.Select(
        name="Contrast",
        options=contrasts,
        value=contrasts[0],
    )

    show_measured  = pn.widgets.Checkbox(name="Observed in Both", value=True)
    show_imp_cond1 = pn.widgets.Checkbox(name=f"", value=True)
    show_imp_cond2 = pn.widgets.Checkbox(name=f"", value=True)

    # 3) Function to update toggle labels whenever contrast changes
    def _update_toggle_labels(event=None):
        grp1, grp2 = contrast_sel.value.split("_vs_")
        show_imp_cond1.name = f"▲ Fully Imputed in {grp1}"
        show_imp_cond2.name = f"▼ Fully Imputed in {grp2}"

    # initialize labels and watch for changes
    _update_toggle_labels()
    contrast_sel.param.watch(_update_toggle_labels, "value")

    search_input = pn.widgets.AutocompleteInput(
        name="Search Protein",
        options=list(state.adata.var["GENE_NAMES"]),
        placeholder="Type gene name…",
    )
    clear_search = pn.widgets.Button(name="Clear Search", width=100)
    # clicking it empties the search box
    clear_search.on_click(lambda event: setattr(search_input, "value", ""))

    volcano_dmap = pn.bind(
        plot_volcanoes_wrapper,
        state=state,
        contrast=contrast_sel,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        highlight=search_input,
        sign_threshold=0.05,
        width=900,
        height=800,
    )

    # 3) assemble into a layout, no legend‐based toggles
    volcano_pane = pn.Column(
        pn.Row(
            contrast_sel,
            pn.Spacer(width=140),
            pn.Row(
                show_measured,
                show_imp_cond1,
                show_imp_cond2,
                margin=(25,0,0,0),
            ),
            pn.Spacer(width=400),
            search_input,
            pn.Row(clear_search, margin = (17,0,0,0)),
            sizing_mode="fixed",
            width=300,
            height=160,
        ),
        pn.panel(volcano_dmap,
                 width=900,
                 height=800,
                 margin=(-50, 0, 0, 0),
                 sizing_mode="fixed",
                 max_states=20,
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
        h_clustering_pane,
        pn.pane.Markdown("##   Volcano plots"),
        volcano_pane,

        sizing_mode="stretch_both",
    )

    return layout
