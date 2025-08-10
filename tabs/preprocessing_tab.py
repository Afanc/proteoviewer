import panel as pn
#from panel.widgets import LoadingSpinner
from concurrent.futures import ThreadPoolExecutor
from session_state import SessionState
from components.preprocessing_plots import (
    plot_filter_histograms,
    plot_dynamic_range,
    plot_intensities_histogram,
    plot_violin_intensity_by_condition,
    plot_violin_intensity_by_sample,
    plot_cv_by_condition,
    plot_rmad_by_condition,
    plot_ma,
    plot_mv_barplots,
    plot_mv_heatmaps,
    plot_grouped_violin_imputation_by_condition,
    plot_grouped_violin_imputation_by_sample,
    plot_grouped_violin_imputation_metrics_by_condition,
)
from components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from utils import logger, log_time
from layout_utils import plotly_section, make_vr, make_hr, make_section, make_row, FRAME_STYLES
import textwrap

executor = ThreadPoolExecutor()

@log_time("Preparing Preprocessing Tab")
def preprocessing_tab(state: SessionState):
    """
    Panel layout for normalization overview:
     - Histogram of intensities
     - Violin by condition
     - Violin by sample
     - MA-plot selector
    """

    adata = state.adata

    ## filtering histograms
    hists = plot_filter_histograms(adata)

    q = plotly_section(hists['qvalue'], height=400, flex="0.32")
    p = plotly_section(hists['pep'], height=400, flex='0.32')
    r = plotly_section(hists['run_evidence_count'], height=400, flex='0.32')

    filtering_row = make_row(
        pn.pane.Markdown("##   Filtering", styles={"flex": "0.05"}),
        q, pn.Spacer(width=10), make_vr(), pn.Spacer(width=20),
        p, pn.Spacer(width=10), make_vr(), pn.Spacer(width=20),
        r,
        height=420,
        width='95vw',
    )

    psm_pane = make_section(
        header="Precursor-level",
        row=filtering_row,
        background="#E3F2FD",
        width="98vw",
        height=500
    )

    # Quantification
    dyn_fig = plotly_section(plot_dynamic_range(adata), height=400)

    dyn_fig_row = make_row(
        pn.pane.Markdown("##   Dynamic Range", styles={"flex": "0.05"}),
        dyn_fig,
        height=420,
        width='95vw',
    )

    quant_pane = make_section(
        header ="Quantification",
        row=dyn_fig_row,
        background="#E8F5E9",
        width="98vw",
        height=500
    )

    # pre/post log trasform
    hist_fig = plotly_section(plot_intensities_histogram(adata), height=430)

    loghistogram_row = make_row(
        pn.pane.Markdown("##   Distributions", styles={"flex": "0.05"}),
        hist_fig,
        height=450,
        width='92vw',
    )

    logtransform_pane = make_section(
        header="Log Transformation",
        row=loghistogram_row,
        height=530,
        background='#FFEFD6',
        width= '95vw',
    )

    norm_violin_by_condition = plotly_section(plot_violin_intensity_by_condition(adata),
                                            height=500,
                                            flex="0.5")

    norm_violin_by_sample = plotly_section(plot_violin_intensity_by_sample(adata),
                                         height=500,
                                         flex="1")
    norm_violins_row = make_row(
        pn.pane.Markdown("##   Distributions", styles={"flex":"0.1"}),
        norm_violin_by_condition, pn.Spacer(width=10), make_vr(), pn.Spacer(width=10),
        norm_violin_by_sample,
        height=530,
        width='92vw',
    )

    norm_violins_pane = pn.Row(
        pn.pane.Markdown("##   Distributions", styles={"flex":"0.1"}),
        norm_violin_by_condition,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=10),
        norm_violin_by_sample,
        height=530,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )

    rmad_pane = plotly_section(plot_rmad_by_condition(adata),
                                         height=500,
                                         flex="1")
    cv_pane = plotly_section(plot_cv_by_condition(adata),
                                         height=500,
                                         flex="1")

    norm_metrics_row = make_row(
        pn.pane.Markdown("##   Metrics", styles={"flex":"0.1"}),
        rmad_pane, make_vr(), pn.Spacer(width=60),
        cv_pane,
        height=520,
        width='92vw',
    )


    norm_metrics_pane = pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex":"0.1"}),
        rmad_pane,
        make_vr(),
        pn.Spacer(width=60),
        cv_pane,
        height=520,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )

    # MA plots
    sample_sel = pn.widgets.Select(
        name="Sample",
        options=list(state.adata.obs_names),
        value=state.adata.obs_names[0],
        width=150
    )

    status_text = pn.pane.Markdown("Computing…", visible=False)
    plot_area   = pn.Column(sizing_mode="stretch_width", height=500)

    def make_ma_row(before, after):
        return pn.Row(
            pn.pane.Plotly(before, sizing_mode="stretch_width"),
            make_vr(),
            pn.pane.Plotly(after,  sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    # Use one doc reference everywhere
    bokeh_doc = pn.state.curdoc

    # Generation token to drop stale finishes
    ma_generation = {"v": 0}

    def on_sample_change(event):
        sample = event.new
        ma_generation["v"] += 1
        this_gen = ma_generation["v"]

        # Show "Computing…"; keep old plot to avoid model teardown races
        status_text.visible = True
        try:
            # If Panel >=1.4, this overlays a spinner; safe no-op otherwise
            plot_area.loading = True
        except Exception:
            pass

        def compute():
            fig_before, fig_after = plot_ma(state.adata, sample)

            def finish():
                # Ignore stale results
                if this_gen != ma_generation["v"]:
                    return
                plot_area[:] = [make_ma_row(fig_before, fig_after)]
                status_text.visible = False
                try:
                    plot_area.loading = False
                except Exception:
                    pass

            bokeh_doc.add_next_tick_callback(finish)

        executor.submit(compute)

    sample_sel.param.watch(on_sample_change, "value")

    # Initial lazy compute AFTER first paint
    from types import SimpleNamespace
    def _initial_ma():
        on_sample_change(SimpleNamespace(new=sample_sel.value))
    bokeh_doc.add_next_tick_callback(_initial_ma)

    ma_pane = pn.Row(
        pn.Column(
            pn.pane.Markdown("##   MA Plots", styles={"flex":"0.1"}),
            sample_sel,
            status_text,
        ),
        plot_area,
        height=520,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius': '15px',
            'box-shadow':    '3px 3px 5px #bcbcbc',
            'background':    'white',
            'width':        '92vw',
        }
    )

    # Final Normalization pane
    norm_content = pn.Column(
        norm_violins_pane, pn.Spacer(height=30),
        norm_metrics_pane, pn.Spacer(height=30),
        ma_pane,
    )
    normalization_pane = make_section(
        header="Normalization",
        row = norm_content,
        height=1720,
        width='95vw',
        background= '#FFE0B2',
    )

    # Imputation
    mv_cond_fig, mv_sample_fig  = plot_mv_barplots(adata)

    mv_cond_pane = plotly_section(mv_cond_fig,
                                  height=500,
                                  flex="0.5")

    mv_sample_pane = plotly_section(mv_sample_fig,
                                    height=500,
                                    flex="1")

    # Placeholders that will get the real panes inserted later (keeps look/size)
    mv_corr_holder   = pn.Column(height=600, sizing_mode="stretch_width")
    mv_binary_holder = pn.Column(height=600, sizing_mode="stretch_width")

    # One-shot builder
    heatmaps_built = False
    bokeh_doc = pn.state.curdoc  # you already use this for MA plots

    def build_mv_heatmaps_once():
        nonlocal heatmaps_built
        if heatmaps_built:
            return
        mv_corr_holder.loading = True
        mv_binary_holder.loading = True

        def compute():
            # Compute both; binary is the heavy one. We insert both afterwards.
            binary_heatmap_fig, corr_heatmap_fig = plot_mv_heatmaps(adata)

            def finish():
                nonlocal heatmaps_built
                # Preserve your aesthetics by wrapping with plotly_section here
                mv_corr_holder[:] = [plotly_section(corr_heatmap_fig,  height=600, flex="1")]
                mv_binary_holder[:] = [plotly_section(binary_heatmap_fig, height=600, flex="1")]
                mv_corr_holder.loading = False
                mv_binary_holder.loading = False
                heatmaps_built = True

            bokeh_doc.add_next_tick_callback(finish)

        executor.submit(compute)

    # Kick it off right after the tab paints (shows spinner, then fills in)
    bokeh_doc.add_next_tick_callback(build_mv_heatmaps_once)


    mv_row = make_row(
        pn.pane.Markdown("##   Missing Values", styles={"flex":"0.05"}),
        pn.Column(
            pn.Row( #replace with helper ? why
                mv_cond_pane,
                pn.Spacer(width=10),
                make_vr(),
                pn.Spacer(width=10),
                mv_sample_pane,
                height=530,
                margin=(0, 0, 0, 0),
            ),
            make_hr(),
            pn.Spacer(width=20),
            pn.Row(
                #mv_corr_heatmap_pane,
                mv_corr_holder,
                pn.Spacer(width=10),
                make_vr(),
                #mv_binary_heatmap_pane,
                mv_binary_holder,
                height=530,
                margin=(0, 0, 0, 0),
            ),
            height=1260,
        ),
        height=1160,
        width="92vw",
    )

    # Metrics:

    imput_dist_cond_pane = plotly_section(plot_grouped_violin_imputation_by_condition(adata),
                                          height=500,
                                          flex="0.5")

    imput_dist_samp_pane = plotly_section(plot_grouped_violin_imputation_by_sample(adata),
                                          height=500,
                                          flex="1")

    imput_dist_pane = pn.Row(
        pn.pane.Markdown("##   Distributions", styles={"flex":"0.1"}),
        imput_dist_cond_pane,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=60),
        imput_dist_samp_pane,
        height=520,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )

    rmad_cond_fig, cv_cond_fig = plot_grouped_violin_imputation_metrics_by_condition(adata)
    imput_rmad_pane = plotly_section(rmad_cond_fig,
                                     height=500,
                                     flex="1")

    imput_cv_pane = plotly_section(cv_cond_fig,
                                   height=500,
                                   flex="1")

    imput_metrics_pane = pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex":"0.1"}),
        imput_rmad_pane,
        make_vr(),
        pn.Spacer(width=60),
        imput_cv_pane,
        height=520,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )

    dist_row = make_row(
        pn.pane.Markdown("##   Distributions", styles={"flex": "0.1"}),
        imput_dist_cond_pane,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=60),
        imput_dist_samp_pane,
        width="92vw",
        height=520,
    )

    metrics_row = make_row(
        pn.pane.Markdown("##   Metrics", styles={"flex": "0.1"}),
        imput_rmad_pane,
        make_vr(),
        pn.Spacer(width=60),
        imput_cv_pane,
        width="92vw",
        height=520,
    )

    imputation_pane = make_section(
        header="Imputation",
        row=pn.Column(
            #mv_pane,
            mv_row,
            pn.Spacer(height=30),
            dist_row,
            pn.Spacer(height=30),
            metrics_row,
            sizing_mode="stretch_width",
        ),
        background="#FFCC80",
        width="95vw",
        height=2340,
    )

    # final protein pane
    protein_pane = make_section(
        header="Protein-level",
        row=pn.Column(
            logtransform_pane,
            pn.Spacer(height=30),
            normalization_pane,
            pn.Spacer(height=30),
            imputation_pane,
            sizing_mode="stretch_width",
        ),
        background="#FFF8F0",
        width="98vw",
        height=4740,
    )

    return pn.Column(
        pn.Spacer(height=10),
        psm_pane,
        pn.Spacer(height=30),
        quant_pane,
        pn.Spacer(height=30),
        protein_pane,
        pn.Spacer(height=30),
        sizing_mode="stretch_both",
        styles=FRAME_STYLES,
    )
