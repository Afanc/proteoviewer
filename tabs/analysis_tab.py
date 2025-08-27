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
    contrast_sel_logfc = pn.widgets.Select(name="", options=list(contrast_names), value=contrast_names[0], width=160, styles={"z-index": "10"}, margin=(-10,0,0,0))

    # ---------- section: linear model (residual variance) ----------
    resvar_fig = plotly_section(
        residual_variance_hist(adata),
        height=430,
        flex='0.8',
        margin=(20,0,0,-50))

    # ---------- section: log2FC distribution (per contrast) ----------
    @pn.depends(contrast=contrast_sel_logfc)
    def log2fc_pane(contrast):
        fig = log2fc_histogram(adata, contrast)
        return plotly_section(fig, height=430, flex='1', margin=(-50,0,0,10))

    lin_reg_row = make_row(
        pn.pane.Markdown("##   Residuals", styles={"flex": "0.05", "z-index": "10"}),
        resvar_fig, pn.Spacer(width=10), make_vr(), pn.Spacer(width=20),
        pn.Column(
                pn.pane.Markdown("##   Log2FC", styles={"flex": "0.05", "z-index": "10"}),
                contrast_sel_logfc,
                log2fc_pane,
        ),
        height=460,
        width='95vw',
    )

    lin_reg_pane = make_section(
        header="Linear Regression",
        row=lin_reg_row,
        background="#E3F2FD",
        width="98vw",
        height=540
    )

    # ---------- section: stats distributions (p & q, overlay raw vs eBayes) ----------
    contrast_sel_stat = pn.widgets.Select(name="Contrast", options=list(contrast_names), value=contrast_names[0], width=180,
                                          margin=(-10,0,0,20), styles={"z-index": "10"})

    @pn.depends(contrast=contrast_sel_stat)
    def stats_row(contrast):
        pfig = stat_histogram(adata, "p", contrast)
        qfig = stat_histogram(adata, "q", contrast)

        return pn.Row(
            plotly_section(pfig, height=420, margin=(-50,0,0,0)),
            pn.Spacer(width=10),
            make_vr(),
            plotly_section(qfig, height=420, margin=(-50,0,0,0)),
        )

    stats_row = make_row(
        pn.Column(
            pn.pane.Markdown("##  Distributions", styles={"flex": "0.05", "z-index": "10"}),
            contrast_sel_stat,
            stats_row,
        ),
        #stats_row,
        height=470,
        width='95vw',
    )

    # ---------- section: shrinkage scatter (p & q) ----------
    contrast_sel_shrink = pn.widgets.Select(name="Contrast", options=list(contrast_names), value=contrast_names[0], width=180,
                                            margin=(-10,0,0,20), styles={"z-index": "10"})
    @pn.depends(contrast=contrast_sel_shrink)
    def shrink_row(contrast):
        pfig = stat_shrinkage_scatter(adata, "p", contrast)
        qfig = stat_shrinkage_scatter(adata, "q", contrast)
        return pn.Row(
            plotly_section(pfig, height=420, margin=(-80,0,0,0)),
            pn.Spacer(width=10),
            make_vr(),
            plotly_section(qfig, height=420, margin=(-80,0,0,0)),
        )

    shrink_row = make_row(
        pn.Column(
            pn.pane.Markdown("## Shrinkage", styles={"flex": "0.05", "z-index": "10"}),
            contrast_sel_shrink,
            shrink_row,
        ),
        #shrink_row,
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
        height=1020
    )

    # Clustering
    # --- Clustering (warning-free, stable panes) ---
    heatmap_toggle = pn.widgets.RadioButtonGroup(
        name="Matrix",
        options=["Deviations", "Intensities"],
        value="Deviations",
        button_type="default",
        width=170,
        styles={"z-index": "10"},
        margin=(10,0,0,0),
    )

    _modes = ("Deviations", "Intensities")

    # persistent Plotly panes; we never remove them, just toggle .visible and update .object
    _panes = {
        m: plotly_section(go.Figure(),
                          height=800,
                          margin=(0,0,0,-200),
                          flex='1')
        for m in _modes
    }
    for m, pane in _panes.items():
        pane.visible = False
        pane.min_width = 0

    # a Column that always contains both panes (stable children)
    heatmap_holder = pn.Column(*_panes.values(),
                               height=800,
                               sizing_mode="stretch_width",
                               margin=(0,0,0,-100),
                               styles={"flex": "1", "min-width": "0"})

    _heatmap_ready = set()
    _building       = set()
    _last_req_id    = {m: 0 for m in _modes}

    def _build_mode_async(mode: str):
        if mode in _heatmap_ready or mode in _building:
            return
        _building.add(mode)
        _last_req_id[mode] += 1
        req_id = _last_req_id[mode]

        heatmap_holder.loading = True

        def compute():
            fig = plot_h_clustering_heatmap(adata, mode=mode)

            def finish():
                if req_id != _last_req_id[mode]:
                    _building.discard(mode)
                    if not _building:
                        heatmap_holder.loading = False
                    return

                # update persistent pane
                _panes[mode].object = fig
                _heatmap_ready.add(mode)
                _building.discard(mode)

                if heatmap_toggle.value == mode:
                    _reveal_mode(mode)          # << no flicker here too

                if not _building:
                    heatmap_holder.loading = False

            bokeh_doc.add_next_tick_callback(finish)

        executor.submit(compute)

    def _reveal_mode(mode: str):
        # 1) show target
        if not _panes[mode].visible:
            _panes[mode].visible = True
        # 2) hide the rest
        for m, p in _panes.items():
            if m != mode and p.visible:
                p.visible = False

    def _on_heatmap_mode(event):
        mode = event.new
        if mode in _heatmap_ready:
            _reveal_mode(mode)             # << no flicker
        else:
            _build_mode_async(mode)


    def _build_initial():
        _build_mode_async(heatmap_toggle.value)

    heatmap_toggle.param.watch(_on_heatmap_mode, "value")
    bokeh_doc.add_next_tick_callback(_build_initial)

    h_clustering_pane = make_row(
        pn.pane.Markdown("##   Clustering", styles={"flex": "0.05", "z-index": "10"}),
        pn.Spacer(width=50),
        heatmap_toggle,
        heatmap_holder,  # stable container, children never removed
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

