import panel as pn
from concurrent.futures import ThreadPoolExecutor
from utils.session_state import SessionState
from components.preprocessing_plots import (
    plot_filter_histograms,
    plot_dynamic_range,
    plot_intensities_histogram,
    plot_violin_intensity_by_condition,
    plot_violin_intensity_by_sample,
    plot_line_density_by_sample,
    plot_cv_by_condition,
    plot_rmad_by_condition,
    plot_ma,
    plot_mv_barplots,
    plot_mv_heatmaps,
    plot_grouped_violin_imputation_by_condition,
    plot_grouped_violin_imputation_by_sample,
    plot_grouped_violin_imputation_metrics_by_condition,
    plot_grouped_violin_before_after_imputation_metrics_by_condition,
    plot_left_censoring_histogram,
    plot_prec_pep_distributions,
    _placeholder_plot,
)
from components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from utils.utils import logger, log_time
from utils.layout_utils import plotly_section, make_vr, make_hr, make_section, make_row, FRAME_STYLES
from types import SimpleNamespace

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
    try:
        hists = plot_filter_histograms(adata)
        q_fig = hists.get('qvalue')
        p_fig = hists.get('pep')
        prec_fig = hists.get('precursors')
    except Exception as e:
        # Make it crystal clear in logs but keep the UI alive
        logger.warning(f"plot_filter_histograms failed ({e}); showing placeholders.")
        q_fig = p_fig = r_fig = None

    # Graceful placeholders if any figure is missing
    q = plotly_section(q_fig if q_fig is not None else _placeholder_plot("q-value filtering"),
                       height=400, flex="0.32", margin=(0,0,0,-100))
    p = plotly_section(p_fig if p_fig is not None else _placeholder_plot("Probab. filtering"),
                       height=400, flex='0.32')
    prec = plotly_section(prec_fig if prec_fig is not None else _placeholder_plot("Precursors filtering"),
                       height=400, flex='0.32', margin=(0,10,0,0))

    filtering_row = make_row(
        pn.pane.Markdown("##   Filtering", styles={"flex": "0.05", "z-index": "10"}),
        q, pn.Spacer(width=10), make_vr(), pn.Spacer(width=20),
        p, pn.Spacer(width=10), make_vr(), pn.Spacer(width=20),
        prec,
        height=420,
        width='95vw',
    )

    censor_fig = plotly_section(plot_left_censoring_histogram(adata), height=400, flex="1", margin=(10,10,0,-150))
    censor_row = make_row(
        pn.pane.Markdown("##   Left-censoring", styles={"flex": "0.07", "z-index": "10"}),
        censor_fig,
        height=420,
        width='95vw',
    )

    psm_pane = make_section(
        header="Precursor-level",
        row=pn.Column(
            filtering_row,
            pn.Spacer(height=20),
            censor_row,
            sizing_mode="stretch_width",
        ),
        background="#E3F2FD",
        width="98vw",
        height=940,
    )

    # Quantification
    prec_pep_fig, pep_prot_fig, missed_cleav_raw_fig, missed_cleav_w_fig = plot_prec_pep_distributions(adata)

    prec_pep_pane = plotly_section(
        prec_pep_fig if prec_pep_fig is not None else _placeholder_plot("Precursors per peptide"),
        height=400,
        flex="0.2",
        margin=(10,0,0,-100),
    )

    pep_prot_pane = plotly_section(
        pep_prot_fig if pep_prot_fig is not None else _placeholder_plot("Peptides per protein"),
        height=400,
        flex="0.4",
    )

    mc_mode = pn.widgets.RadioButtonGroup(
        name="",
        options=["Raw", "Intensity-normalized"],
        value="Raw",
        width=160,
        button_type="default",
        styles={"z-index": "10"},
    )

    mc_plot_holder = pn.Column(sizing_mode="stretch_width")

    def _render_mc_plot(mode: str):
        fig = missed_cleav_raw_fig if mode == "Raw" else missed_cleav_w_fig
        if fig is None:
            fig = _placeholder_plot("Missed cleavages")
        mc_plot_holder[:] = [plotly_section(fig, height=400, flex="1")]

    _render_mc_plot(mc_mode.value)
    mc_mode.param.watch(lambda e: _render_mc_plot(e.new), "value")

    missed_cleav_pane = pn.Column(
        pn.Row(mc_mode, styles={"z-index": "10"}, margin=(10, 0, 0, 0)),
        pn.Row(mc_plot_holder, margin=(-30, 5, 0, 0)),
        sizing_mode="stretch_width",
        styles={"flex": "0.4"},
    )

    depth_row = make_row(
        pn.pane.Markdown("##   ID Depth", styles={"flex": "0.05", "z-index": "10"}),
        prec_pep_pane,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=10),
        pep_prot_pane,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=10),
        missed_cleav_pane,
        height=430,
        width='95vw',
    )

    dyn_fig = plotly_section(plot_dynamic_range(adata), height=400, margin=(10,10,0,-15))

    dyn_fig_row = make_row(
        pn.pane.Markdown("##   Dynamic Range", styles={"flex": "0.05"}),
        dyn_fig,
        height=420,
        width='95vw',
    )
    quant_content = pn.Column(
        depth_row,
        pn.Spacer(height=30),
        dyn_fig_row,
    )

    quant_pane = make_section(
        header ="Quantification",
        row=quant_content,
        background="#E8F5E9",
        width="98vw",
        height=950
    )

    # pre/post log trasform
    hist_fig = plotly_section(plot_intensities_histogram(adata), height=430, margin=(10,10,0,-100))

    loghistogram_row = make_row(
        pn.pane.Markdown("##   Distributions", styles={"flex": "0.05", "z-index": "10"}),
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
                                            flex="0.5",
                                            margin=(0, 0, 0, -100))

    # Right becomes: tiny selector + swappable plot
    dist_mode = pn.widgets.RadioButtonGroup(
        name="",
        options=["Violins", "Line charts"],
        value="Violins",
        width=100,
        button_type="default",
        styles={"z-index": "10"},
    )

    # a holder column for the plot pane
    sample_plot_holder = pn.Column(sizing_mode="stretch_width")

    def _render_sample_plot(mode: str):
        if mode == "Line charts":
            fig = plot_line_density_by_sample(adata)
        else:
            fig = plot_violin_intensity_by_sample(adata)
        sample_plot_holder[:] = [plotly_section(fig, height=500, flex="1")]

    # initial render
    _render_sample_plot(dist_mode.value)
    dist_mode.param.watch(lambda e: _render_sample_plot(e.new), "value")

    # assemble the row box
    norm_violins_pane = pn.Row(
        pn.pane.Markdown("##   Distributions", styles={"flex":"0.1", "z-index": "10"}),
        norm_violin_by_condition,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=10),
        pn.Column(
            pn.Row(
                pn.Row(dist_mode,
                       styles={"z-index": "10"},
                       margin=(20,10,0,0),
                       width=200,
                       sizing_mode="fixed"),
                pn.Row(sample_plot_holder, margin=(0,20,0,-200)),
                sizing_mode="stretch_width",
                styles={"flex": "1"},
            ),
        ),
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
                                         flex="1",
                                         margin=(20,0,0,-100))
    cv_pane = plotly_section(plot_cv_by_condition(adata),
                                         height=500,
                                         flex="1",
                                         margin=(20,0,0,-50))

    norm_metrics_pane = pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex":"0.1", "z-index": "10"}),
        rmad_pane,
        make_vr(),
        pn.Spacer(width=60),
        cv_pane,
        height=540,
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
        width=150,
        styles={"z-index": "10"},
        margin=(-10,0,0,20),
    )

    status_text = pn.pane.Markdown("Computing…", visible=False)
    plot_area   = pn.Column(sizing_mode="stretch_width", height=500)

    def make_ma_row(before, after):
        return pn.Row(
            pn.pane.Plotly(before, sizing_mode="stretch_width", margin=(0,0,0,-150)),
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
            """Compute MA plots in a worker thread; always schedule a UI update."""
            try:
                fig_before, fig_after = plot_ma(state.adata, sample)
            except Exception as e:
                # If compute fails, we MUST still clear the spinner and show something.
                logger.exception("MA plot computation failed")
                fig_before = _placeholder_plot("MA before normalization", subtitle=str(e))
                fig_after  = _placeholder_plot("MA after normalization",  subtitle=str(e))

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

            # pn.state.curdoc can be None in odd build contexts; fall back gracefully.
            if bokeh_doc is not None:
                bokeh_doc.add_next_tick_callback(finish)
            else:
                finish()

        executor.submit(compute)

    sample_sel.param.watch(on_sample_change, "value")

    # Initial lazy compute AFTER first paint
    def _initial_ma():
        on_sample_change(SimpleNamespace(new=sample_sel.value))
    if bokeh_doc is not None:
        bokeh_doc.add_next_tick_callback(_initial_ma)
    else:
        _initial_ma()

    ma_pane = pn.Row(
        pn.Column(
            pn.pane.Markdown("##   MA Plots", styles={"flex":"0.1", "z-index": "10"}),
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
        height=1740,
        width='95vw',
        background= '#FFE0B2',
    )

    # Imputation
    mv_cond_fig, mv_sample_fig  = plot_mv_barplots(adata)

    mv_cond_pane = plotly_section(mv_cond_fig,
                                  height=500,
                                  flex="0.5",
                                  margin=(20,0,0,-100))

    mv_sample_pane = plotly_section(mv_sample_fig,
                                    height=500,
                                    flex="1",
                                    margin=(20,0,0,0))

    # Placeholders that will get the real panes inserted later (keeps look/size)
    mv_corr_holder   = pn.Column(height=600, sizing_mode="stretch_width", styles={"flex": "0.6"})
    mv_binary_holder = pn.Column(height=600, sizing_mode="stretch_width", styles={"flex": "1"})

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
                mv_corr_holder[:] = [plotly_section(corr_heatmap_fig,
                                                    height=600,
                                                    flex="0.5",
                                                    margin=(0,-50,0,-50),
                                                    )]
                mv_binary_holder[:] = [plotly_section(binary_heatmap_fig,
                                                      height=600,
                                                      flex="1",
                                                      margin=(0,0,0,0))]
                mv_corr_holder.loading = False
                mv_binary_holder.loading = False
                heatmaps_built = True

            bokeh_doc.add_next_tick_callback(finish)

        executor.submit(compute)

    # Kick it off right after the tab paints (shows spinner, then fills in)
    bokeh_doc.add_next_tick_callback(build_mv_heatmaps_once)


    mv_row = make_row(
        pn.pane.Markdown("##   Missing Values", styles={"flex":"0.05", "z-index": "10"}),
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
                mv_corr_holder,
                pn.Spacer(width=10),
                make_vr(),
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
                                          flex="0.5",
                                          margin=(20,0,0,-100))

    imput_dist_samp_pane = plotly_section(plot_grouped_violin_imputation_by_sample(adata),
                                          height=500,
                                          flex="1",
                                          margin=(20,0,0,-50))

    fig_cv_ba, fig_rmad_ba = plot_grouped_violin_before_after_imputation_metrics_by_condition(adata)
    imput_rmad_ba_pane = plotly_section(fig_rmad_ba,
                                        height=500,
                                        flex="1",
                                        margin=(20,0,0,-100))


    imput_cv_ba_pane = plotly_section(fig_cv_ba,
                                      height=500,
                                      flex="1",
                                      margin=(20,0,0,-50))

    dist_row = make_row(
        pn.pane.Markdown("##   Distributions", styles={"flex": "0.1", "z-index": "10"}),
        imput_dist_cond_pane,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=60),
        imput_dist_samp_pane,
        width="92vw",
        height=540,
    )

    metrics_row = make_row(
        pn.pane.Markdown("##   Metrics", styles={"flex": "0.05", "z-index": "10"}),
        pn.Column(
            pn.Row(
                imput_rmad_ba_pane,
                make_vr(),
                pn.Spacer(width=60),
                imput_cv_ba_pane,
                height=540,
                margin=(0, 0, 0, 0),
            ),
            height=540,
        ),
        height=540,
        width="92vw",
    )

    imputation_pane = make_section(
        header="Imputation",
        row=pn.Column(
            mv_row,
            pn.Spacer(height=30),
            dist_row,
            pn.Spacer(height=30),
            metrics_row,
            sizing_mode="stretch_width",
        ),
        background="#FFCC80",
        width="95vw",
        height=2380,
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
        height=4780,
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
