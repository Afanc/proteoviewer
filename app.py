import logging
import os
import sys
import socket

import panel as pn

from session_state import SessionState
from tabs.overview_tab import overview_tab
from tabs.preprocessing_tab import preprocessing_tab
from tabs.analysis_tab import analysis_tab

from utils import logger, log_time

logging.getLogger().setLevel(logging.INFO)

pn.extension('plotly', defer_load=True, loading_indicator=True)

MIN_PF_VERSION = os.environ.get("PF_MIN_PF_VERSION", "1.1.0")

DEV = False #change for env variable

def _parse_semver(v: str):
    try:
        core = v.split("+", 1)[0]
        parts = [int(p) for p in core.split(".")]
        return tuple(parts + [0] * (3 - len(parts)))[:3]
    except Exception:
        return (1, 0, 0)

def _check_pf_meta(adata):
    """Return (ok: bool, message: str, meta: dict)."""
    meta = adata.uns.get("proteoflux", {}) or {}
    pfv = meta.get("pf_version")
    created = meta.get("created_at", "unknown time")
    if not pfv:
        return (False, "No ProteoFlux version found in uns['proteoflux'].", meta)
    if _parse_semver(pfv) < _parse_semver(MIN_PF_VERSION):
        return (False, f" File written by ProteoFlux {pfv} (< {MIN_PF_VERSION}). Proceeding; some views may be limited.", meta)
    return (True, f"ProteoFlux {pfv} • {created}", meta)

def get_free_port(start=5006, end=5099):
    """Find an open port so users can run multiple instances."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available.")

def _lazy_tabs(state):
    """
    Build tabs lazily on first activation, then keep them alive.
    - Spinner shown while a tab is building (thanks to loading_indicator=True).
    - Revisits are instant (content persists; no dynamic teardown).
    """
    # List your tabs as (label, builder_fn)
    specs = [
        ("Overview",      lambda: overview_tab(state)),
        ("Preprocessing", lambda: preprocessing_tab(state)),
        ("Analysis", lambda: analysis_tab(state)),
    ]

    tabs = pn.Tabs(dynamic=False)  # keep content mounted once built
    holders   = []                 # one container per tab
    builders  = []                 # matching builders
    built     = []                 # flags per tab

    # Create placeholders
    for label, builder in specs:
        holder = pn.Column(
            sizing_mode="stretch_both",
            styles={"min-height": "240px"}
        )
        holders.append(holder)
        builders.append(builder)
        built.append(False)
        tabs.append((label, holder))

    def build_index(i: int):
        if built[i]:
            return
        holder = holders[i]
        holder.loading = True

        def do_build():
            try:
                content = builders[i]()  # heavy work
                holder[:] = [content]
                built[i] = True
            except Exception as e:
                holder[:] = [pn.pane.Markdown(f"**Error while building tab:** {e}")]
            finally:
                holder.loading = False

        # schedule on next tick so the loading overlay can render
        pn.state.curdoc.add_next_tick_callback(do_build)

    def _set_visibility(active: int):
        for i, h in enumerate(holders):
            h.visible = (i == active)

    # Build first tab: make it visible *then* build (spinner shows)
    _set_visibility(tabs.active)
    build_index(tabs.active)

    def _on_active(event):
        # Make target visible first so its spinner can render
        _set_visibility(event.new)
        build_index(event.new)

    tabs.param.watch(_on_active, "active", onlychanged=True)
    return tabs

@log_time("Building app")
def build_app():
    """
    - Content area below gets replaced with the Tabs once a file is loaded.
    """
    # widgets
    file_input = pn.widgets.FileInput(accept=".h5ad", name="Choose file…")
    exit_btn   = pn.widgets.Button(name="Exit", button_type="danger")
    status     = pn.pane.Markdown("### Please upload a .h5ad Proteoflux file.")
    # placeholder; replaced reactively below
    content    = pn.Column(sizing_mode="stretch_width")

    def _load(adata, fname):
        state = SessionState.initialize(adata)
        tabs  = _lazy_tabs(state)  # <-- use the lazy tabs
        content.clear()
        content.append(tabs)
        status.object      = f"**Loaded:** {fname}"

    # when user picks a file
    def _on_upload(event):

        import h5py
        from anndata import read_h5ad
        import io
        try:
            status.object = "Loading…"
            adata = read_h5ad(io.BytesIO(file_input.value))
            _load(adata, getattr(file_input, "filename", "uploaded file"))
        except Exception as e:
            status.object = f"Error: {e}"

    file_input.param.watch(_on_upload, "value")

    # Wire exit button
    def _on_exit(event):
        for w in (file_input, exit_btn, status):
            w.visible = False
        content[:] = [pn.pane.Markdown("## Thanks for using ProteoViewer!")]

        pn.state.curdoc.add_next_tick_callback(lambda: os._exit(0))

    exit_btn.on_click(_on_exit)

    # dev‐mode: load default without clicking
    if DEV:
        from anndata import read_h5ad
        adata = read_h5ad("proteoflux_results.h5ad")
        _load(adata, "proteoflux_results.h5ad")

    controls = pn.Row(file_input, exit_btn, status, sizing_mode="stretch_width")
    app = pn.Column("# ProteoViewer", controls, content, sizing_mode="stretch_width")
    return app


app = build_app()
app.servable()

if __name__ == "__main__":
    pn.serve(
        app,
        title="ProteoViewer",
        port=get_free_port(),
        autoreload=True,
        show=True,
    )
