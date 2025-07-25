import logging
import pandas as pd
from anndata import read_h5ad
import sys
import os
import io
import h5py
import socket

import panel as pn
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from session_state import SessionState
from tabs.overview_tab import overview_tab
#from tabs.normalization_tab import normalization_tab

# 1) Set up logging so we can see Bokeh/Panel messages in stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger().setLevel(logging.INFO)


# 2) Enable any Panel extensions you need
pn.extension('plotly')

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

def build_app():
    """
    Build the root Panel layout.
    - First shows a FileInput + status.
    - Once a .h5ad is uploaded, replaces itself with a Tabs layout.
    """
    file_input = pn.widgets.FileInput(accept='.h5ad')
    status     = pn.pane.Markdown("### Please upload a .h5ad Proteoflux file.")
    layout     = pn.Column("# ProteoViewer", file_input, status, sizing_mode="stretch_width")

    #def on_upload(event):
    #    try:
    #        status.object = "Loading…"
    #        adata = read_h5ad(io.BytesIO(file_input.value))

    #        state = SessionState.initialize(adata)
    #        tabs = pn.Tabs(
    #            ("Overview",      overview_tab(state)),
    #            #("Normalization", normalization_tab(state)),
    #            # add more tabs here as you implement them
    #        )
    #        layout[:] = [tabs]  # replace the Column contents with the tabs

    #    except Exception as e:
    #        import traceback
    #        traceback.print_exc()
    #        status.object = f"Error: {e}"


    #file_input.param.watch(on_upload, "value")

    # for dev-ing
    adata = read_h5ad("proteoflux_results.h5ad")
    state = SessionState.initialize(adata)
    tabs = pn.Tabs(
        ("Overview",      overview_tab(state)),
        #("Normalization", normalization_tab(state)),
        # add more tabs here as you implement them
    )
    layout[:] = [tabs]  # replace the Column contents with the tabs

    return layout


def _run_bokeh_server():
    port = get_free_port()
    server = Server({'/': app}, port=port)

    # When the browser tab closes, stop the IOLoop and exit the process
    def on_session_destroyed(session_context):
        IOLoop.current().stop()
        os._exit(0)

    server.on_session_destroyed = on_session_destroyed
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


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
