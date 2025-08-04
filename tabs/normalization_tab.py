import panel as pn
import asyncio
from panel.widgets import LoadingSpinner
from concurrent.futures import ThreadPoolExecutor
from panel.pane import Placeholder
from session_state import SessionState
from components.normalization_plots import (
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
)
from components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from utils import logger, log_time
import textwrap

pn.extension("plotly")
executor = ThreadPoolExecutor()

@log_time("Preparing Normalization Tab")
def normalization_tab(state: SessionState):
    """
    Panel layout for normalization overview:
     - Histogram of intensities
     - Violin by condition
     - Violin by sample
     - MA-plot selector
    """

    adata = state.adata

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

    # filtering histograms
    hists = plot_filter_histograms(adata)
    qvalue_filter_pane = pn.pane.Plotly(
        hists['qvalue'],
        height=400,
        sizing_mode="stretch_width",
        styles={"flex":"0.32"},
        #margin=(-10,0,0,0),
        config={'responsive':True}
        )
    pep_filter_pane = pn.pane.Plotly(
        hists['pep'],
        height=400,
        sizing_mode="stretch_width",
        styles={"flex":"0.32"},
        #margin=(-10,0,0,0),
        config={'responsive':True}
        )
    re_filter_pane = pn.pane.Plotly(
        hists['run_evidence_count'],
        height=400,
        sizing_mode="stretch_width",
        styles={"flex":"0.32"},
        #margin=(-10,0,0,0),
        config={'responsive':True}
        )

    filtering_pane = pn.Row(
        pn.pane.Markdown("##   Filtering", styles={"flex": "0.05"}),
        qvalue_filter_pane,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=20),
        pep_filter_pane,
        pn.Spacer(width=10),
        make_vr(),
        pn.Spacer(width=20),
        re_filter_pane,
        height=420,
        margin=(0, 0, 0, 20),
        sizing_mode="fixed",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '95vw',
            'background': 'white',
        }
    )

    psm_pane = pn.Column(
        pn.pane.Markdown("##   Precursor-level", styles={"flex": "0.05"}),
        filtering_pane,
        height=500,
        margin=(0, 0, 0, 20),
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
            'background': '#E3F2FD',
        }

    )

    # Quantification
    dyn_fig = pn.pane.Plotly(
        plot_dynamic_range(state.adata),
        sizing_mode="stretch_width",
        height=400,
        config={"responsive": True},
        styles={"background":"white"}
    )

    dyn_fig_pane = pn.Row(
        pn.pane.Markdown("##   Dynamic Range", styles={"flex": "0.05"}),
        dyn_fig,
        height=420,
        sizing_mode="fixed",
        margin=(0, 0, 0, 20),
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '95vw',
            'background': 'white',
        }
    )

    quant_pane = pn.Column(
        pn.pane.Markdown("##   Quantification", styles={"flex": "0.05"}),
        dyn_fig_pane,
        height=500,
        margin=(0,0,0,20),
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
            'background': '#E8F5E9',
        }

    )

    # pre/post log trasform
    hist_fig = pn.pane.Plotly(
        plot_intensities_histogram(adata),
        height=430,
        sizing_mode="stretch_width",
        styles={
            'background': 'white',
        },
        config={'responsive':True},
    )

    loghistogram = pn.Row(
        pn.pane.Markdown("##   Distributions", styles={"flex": "0.05"}),
        hist_fig,
        height=450,
        sizing_mode="fixed",
        margin=(0, 0, 0, 20),
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )


    logtransform_pane = pn.Column(
        pn.pane.Markdown("##   Log Transformation", styles={"flex":"0.1"}),
        loghistogram,
        #pn.Spacer(width=10),
        height=520,
        margin=(0, 0, 0, 20),
        sizing_mode="fixed",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '95vw',
            #'background': 'white',
            'background': '#FFEFD6',
        }
    )
    # pre/post norm violins
    norm_viol_cond  = plot_violin_intensity_by_condition(adata)
    norm_viol_sample = plot_violin_intensity_by_sample(adata)
    norm_violin_by_condition = pn.pane.Plotly(
        norm_viol_cond,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex":"0.5"},
        config={'responsive':True}
        )

    norm_violin_by_sample = pn.pane.Plotly(
        norm_viol_sample,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex":"1"},
        config={'responsive':True}
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
        sizing_mode="fixed",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )

    # pre/post norm metrics violins
    rmad_fig = plot_rmad_by_condition(adata)
    rmad_pane = pn.pane.Plotly(
        rmad_fig,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex":"1"},
        config={'responsive':True}
        )


    cv_fig = plot_cv_by_condition(adata)
    cv_pane = pn.pane.Plotly(
        cv_fig,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex":"1"},
        config={'responsive':True}
        )

    norm_metrics_pane = pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex":"0.1"}),
        rmad_pane,
        #pn.Spacer(width=20),
        make_vr(),
        pn.Spacer(width=60),
        cv_pane,
        height=520,
        margin=(0, 0, 0, 20),
        sizing_mode="fixed",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )

    # MA plots
    # 1) sample selector as before
    sample_sel = pn.widgets.Select(
        name="Sample",
        options=list(state.adata.obs_names),
        value=state.adata.obs_names[0],
        width=150
    )

    executor = ThreadPoolExecutor()

    # Spinner + status text (off by default)
    #spinner     = pn.widgets.LoadingSpinner(visible=False, width=24, height=24)
    status_text = pn.pane.Markdown("Computing…",
                                   visible=False)

    # A Column to hold the MA plots (give it a fixed/min height so it never collapses)
    plot_area = pn.Column(sizing_mode="stretch_width", height=500)

    # Helper to build the row of two Plotly panes
    def make_ma_row(before, after):
        return pn.Row(
            pn.pane.Plotly(before, sizing_mode="stretch_width"),
            make_vr(),
            pn.pane.Plotly(after,  sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    # Imperative watcher: show spinner, fire off the compute in a thread,
    # then on the Bokeh IOLoop swap in the plots and hide the spinner.
    bokeh_doc = pn.state.curdoc

    def on_sample_change(event):
        sample = event.new

        # 1) show spinner + text, clear old plots
        #spinner.visible     = True
        #spinner.value     = True
        status_text.visible = True
        plot_area[:]        = []            # slice‐assign to preserve the Column’s height

        # 2) dispatch heavy work
        def compute():
            fig_before, fig_after = plot_ma(state.adata, sample)
            # schedule UI update back on main thread
            def finish():
                plot_area[:]        = [make_ma_row(fig_before, fig_after)]
                #spinner.visible     = False
                #spinner.value     = False
                status_text.visible = False

            bokeh_doc.add_next_tick_callback(finish)
            #pn.state.curdoc.add_next_tick_callback(finish)

        executor.submit(compute)

    # wire up and do the very first draw
    sample_sel.param.watch(on_sample_change, "value")
    on_sample_change(type("E", (), {"new": sample_sel.value})())

    ma_pane = pn.Row(
        pn.Column(
            pn.pane.Markdown("##   MA Plots", styles={"flex":"0.1"}),
            sample_sel,
            status_text,
            #pn.Row(
            #    spinner,
            #    status_text,
            #),
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
    normalization_pane = pn.Column(
        pn.pane.Markdown("##   Normalization"),
        norm_violins_pane,
        pn.Spacer(height=30),
        norm_metrics_pane,
        pn.Spacer(height=30),
        ma_pane,
        margin=(0, 0, 0, 20),
        height=1720,
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '95vw',
            'background': '#FFE0B2',
        }
    )

    # Imputation
    mv_cond_fig, mv_sample_fig  = plot_mv_barplots(adata)

    mv_cond_pane = pn.pane.Plotly(
        mv_cond_fig,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex":"0.5"},
        config={'responsive':True}
        )

    mv_sample_pane = pn.pane.Plotly(
        mv_sample_fig,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex":"1"},
        config={'responsive':True}
        )

    binary_heatmap_fig, corr_heatmap_fig = plot_mv_heatmaps(adata)

    mv_binary_heatmap_pane = pn.pane.Matplotlib(
        binary_heatmap_fig,
        tight=True,
        height=700,
        sizing_mode="stretch_width",
        styles={"flex":"1"},
        )

    mv_corr_heatmap_pane = pn.pane.Plotly(
        corr_heatmap_fig,
        height=700,
        sizing_mode="stretch_width",
        styles={"flex":"1"},
        config={'responsive':True}
        )

    mv_pane = pn.Row(
        pn.pane.Markdown("##   Missing Values", styles={"flex":"0.05"}),
        pn.Column(
            pn.Row(
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
                mv_corr_heatmap_pane,
                pn.Spacer(width=10),
                make_vr(),
                mv_binary_heatmap_pane,
                height=700,
                margin=(0, 0, 0, 0),
            ),
            height=1260,
        ),
        height=1260,
        margin=(0, 0, 0, 20),
        sizing_mode="fixed",
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '92vw',
            'background': 'white',
        }
    )


    imputation_pane = pn.Column(
        pn.pane.Markdown("##   Imputation"),
        mv_pane,
        margin=(0, 0, 0, 20),
        height=2500,
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '95vw',
            'background': '#FFCC80',
        }
    )

    # final protein pane
    protein_pane = pn.Column(
        pn.pane.Markdown("##   Protein-level", styles={"flex": "0.05"}),
        logtransform_pane,
        pn.Spacer(height=30),
        normalization_pane,
        pn.Spacer(height=30),
        imputation_pane,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_height",
        height=2500,
        styles={
            'border-radius':  '15px',
            'box-shadow':     '3px 3px 5px #bcbcbc',
            'width': '98vw',
            'background': '#FFF8F0',
        }
    )

    # Create the toggle button
    toggle = pn.widgets.Toggle(name="▼ Why a log2 transformation?", button_type="light")

    # Texts
    intro_text = pn.pane.Markdown(intro_preprocessing_text,
        width=1800,
        margin=(10, 0, 15, 0),
    )
    tooltip_text = pn.pane.Markdown(log_transform_text,
        visible=False,
        margin=(10, 0, 0, 25),
    )

    # Update function to toggle visibility and update button label
    def show_hide_text(event):
        tooltip_text.visible = toggle.value
        toggle.name = "▲ Why a log2 transformation?" if toggle.value else "▼ Why a log2 transformation?"

    toggle.param.watch(show_hide_text, "value")

    # Your layout

    return pn.Column(
        pn.Spacer(height=10),
        #filtering_pane,
        psm_pane,
        pn.Spacer(height=30),
        quant_pane,
        pn.Spacer(height=30),
        protein_pane,
        sizing_mode="stretch_both",
    )
