# tabs/analysis_tab.py
import panel as pn
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

from components.analysis_plots import (
    residual_variance_hist,
    log2fc_histogram,
    stat_histogram,
    stat_shrinkage_scatter,
    plot_h_clustering_heatmap,
)

from layout_utils import plotly_section, make_vr, make_hr, make_section, make_row, FRAME_STYLES

def analysis_tab(state):
    adata = state.adata
    bokeh_doc = pn.state.curdoc
    executor = ThreadPoolExecutor(max_workers=1)

    contrast_names = adata.uns.get("contrast_names")
    if contrast_names is None:
        n = adata.varm["log2fc"].shape[1]
        contrast_names = [f"C{i}" for i in range(n)]

    # ---------- widgets ----------
    contrast_sel_logfc = pn.widgets.Select(name="Contrast", options=list(contrast_names), value=contrast_names[0], width=180)

    # ---------- section: linear model (residual variance) ----------
    resvar_fig = plotly_section(
        residual_variance_hist(adata),
        height=430,
        flex='0.8')

    # ---------- section: log2FC distribution (per contrast) ----------
    @pn.depends(contrast=contrast_sel_logfc)
    def log2fc_pane(contrast):
        fig = log2fc_histogram(adata, contrast)
        return plotly_section(fig, height=430, flex='1')

    lin_reg_row = make_row(
        pn.pane.Markdown("##   Residuals", styles={"flex": "0.05"}),
        resvar_fig, pn.Spacer(width=10), make_vr(), pn.Spacer(width=20),
        pn.Column(
                pn.pane.Markdown("##   Log2FC"),#, styles={"flex": "0.05"}),
                pn.Spacer(width=100),
                contrast_sel_logfc,
                styles={"flex": "0.05"}
        ),
        log2fc_pane,
        height=440,
        width='95vw',
    )

    lin_reg_pane = make_section(
        header="Linear Regression",
        row=lin_reg_row,
        background="#E3F2FD",
        width="98vw",
        height=520
    )

    # ---------- section: stats distributions (p & q, overlay raw vs eBayes) ----------
    contrast_sel_stat = pn.widgets.Select(name="Contrast", options=list(contrast_names), value=contrast_names[0], width=180)

    @pn.depends(contrast=contrast_sel_stat)
    def stats_row(contrast):
        pfig = stat_histogram(adata, "p", contrast)
        qfig = stat_histogram(adata, "q", contrast)

        return pn.Row(
            plotly_section(pfig, height=420),
            pn.Spacer(width=10),
            make_vr(),
            plotly_section(qfig, height=420),
        )

    stats_row = make_row(
        pn.Column(
            pn.pane.Markdown("##  Distributions", styles={"flex": "0.05"}),
            contrast_sel_stat,
        ),
        stats_row,
        height=440,
        width='95vw',
    )

    # ---------- section: shrinkage scatter (p & q) ----------
    contrast_sel_shrink = pn.widgets.Select(name="Contrast", options=list(contrast_names), value=contrast_names[0], width=180)
    @pn.depends(contrast=contrast_sel_shrink)
    def shrink_row(contrast):
        pfig = stat_shrinkage_scatter(adata, "p", contrast)
        qfig = stat_shrinkage_scatter(adata, "q", contrast)
        return pn.Row(
            plotly_section(pfig, height=420),
            pn.Spacer(width=10),
            make_vr(),
            plotly_section(qfig, height=420),
        )

    shrink_row = make_row(
        pn.Column(
            pn.pane.Markdown("## Shrinkage", styles={"flex": "0.05"}),
            contrast_sel_shrink,
        ),
        shrink_row,
        height=440,
        width='95vw',
    )

    stats_pane = make_section(
        header ="Statistical Analysis",
        row=pn.Column(
            stats_row,
            pn.Spacer(height=30),
            shrink_row,
            pn.Spacer(height=30),
            sizing_mode="stretch_width",
        ),
        background="#E8F5E9",
        width="98vw",
        height=1000
    )

    # Clustering
    # ---------- section: hierarchical clustering heatmap (lazy-once) ----------
    heatmap_holder = pn.Column(height=800, sizing_mode="stretch_width")
    heatmap_built = {"v": False}

    def build_heatmap_once():
        if heatmap_built["v"]:
            return
        heatmap_holder.loading = True

        def compute():
            fig = plot_h_clustering_heatmap(adata)
            def finish():
                heatmap_holder[:] = [plotly_section(fig, height=800)]
                heatmap_holder.loading = False
                heatmap_built["v"] = True
            bokeh_doc.add_next_tick_callback(finish)

        executor.submit(compute)

    # run once after first paint
    bokeh_doc.add_next_tick_callback(build_heatmap_once)

    h_clustering_pane = make_row(
            pn.pane.Markdown("##   Clustering", styles={"flex": "0.05"}),
            heatmap_holder,
            height=830,
    )

    clustering_pane = make_section(
            header ="Expression",
            row=h_clustering_pane,
            height=920,
            width='98vw',
            background="#FFF8F0",
    )


    layout = pn.Column(
        pn.Spacer(height=10),
        lin_reg_pane,
        pn.Spacer(height=30),
        stats_pane,
        pn.Spacer(height=30),
        clustering_pane,
        pn.Spacer(height=30),
        sizing_mode="stretch_width",
        styles=FRAME_STYLES,
    )
    return layout

