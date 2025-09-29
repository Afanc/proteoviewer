# tabs/analysis_tab.py
import panel as pn
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from contextlib import contextmanager

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
    contrast_sel_logfc = pn.widgets.Select(name="Contrast", options=list(contrast_names), value=contrast_names[0], width=160, styles={"z-index": "10"}, margin=(-10,0,0,0))

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
    # --- Clustering (same pattern as volcano detail: holder + single pane + loader) ---

    heatmap_toggle = pn.widgets.RadioButtonGroup(
        name="Matrix",
        options=["Deviations", "Intensities"],
        value="Deviations",
        button_type="default",
        width=170,
        styles={"z-index": "10"},
        margin=(10, 0, 0, 0),
    )

    # One persistent pane for fast swaps
    _heatmap_pane = pn.pane.Plotly(
        go.Figure(),
        height=800,
        sizing_mode="stretch_width",
        config={"responsive": True},
        margin=(0,100,0,-100),
        styles={'width': '95vw'},
    )

    # Holder starts with a Spacer so the first-load overlay is visible
    heatmap_holder = pn.Column(
        pn.Spacer(height=800),
        sizing_mode="stretch_width",
        height=820,
        margin=(0,100,0,-100),
        styles={"flex": "1", "min-width": "0", 'width': '95vw'},
    )

    # Cache & worker (single worker; build only on demand)
    _heatmap_cache: dict[str, go.Figure] = {}
    _building: set[str] = set()
    _executor = ThreadPoolExecutor(max_workers=1)
    bokeh_doc = pn.state.curdoc

    def _render_heatmap(mode: str) -> go.Figure:
        return plot_h_clustering_heatmap(adata, mode=mode)

    def _normalize_plotly(fig: go.Figure) -> go.Figure:
        # Clear hard sizes so Panel can stretch it
        try:
            fig.update_layout(autosize=True)
            fig.layout.width = None
            # Let the pane control height (we set pane height=800)
            if getattr(fig.layout, "height", None) is not None:
                fig.layout.height = None
        except Exception:
            pass
        return fig

    def _mount_once():
        if _heatmap_pane not in heatmap_holder.objects:
            heatmap_holder[:] = [_heatmap_pane]

    def _apply_figure(fig: go.Figure):
        _heatmap_pane.object = fig
        _mount_once()

    def _build_async(mode: str, show_after_build: bool):
        if mode in _heatmap_cache or mode in _building:
            return
        _building.add(mode)
        if show_after_build:
            heatmap_holder.loading = True

        def worker():
            fig = _normalize_plotly(_render_heatmap(mode))

            def finish():
                _heatmap_cache[mode] = fig
                if show_after_build:
                    _apply_figure(fig)
                    # Clear the overlay regardless of other background work
                    heatmap_holder.loading = False
                _building.discard(mode)

            bokeh_doc.add_next_tick_callback(finish)

        _executor.submit(worker)

    def _show_mode(mode: str):
        if mode in _heatmap_cache:
            _apply_figure(_heatmap_cache[mode])  # instant swap
        else:
            _build_async(mode, show_after_build=True)

    def _on_heatmap_mode(event):
        if event.old == event.new:
            return
        _show_mode(event.new)

    heatmap_holder.loading = True
    bokeh_doc.add_next_tick_callback(lambda: _show_mode(heatmap_toggle.value))
    heatmap_toggle.param.watch(_on_heatmap_mode, "value")

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

