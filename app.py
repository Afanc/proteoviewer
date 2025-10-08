import os, sys
# --- Harden asyncio import order on Windows *before* importing Panel ---
if sys.platform.startswith("win"):
    import importlib, logging
    for _m in (
        'asyncio.base_events','asyncio.events','asyncio.format_helpers',
        'asyncio.futures','asyncio.protocols','asyncio.tasks','asyncio.transports',
        'asyncio.selector_events','asyncio.windows_events','asyncio.windows_utils',
    ):
        try:
            importlib.import_module(_m)
        except Exception as e:
            logging.getLogger(__name__).warning("Preload failed for %s: %s", _m, e)
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    from tornado.platform.asyncio import AsyncIOMainLoop
    AsyncIOMainLoop().install()
    # Optional tiny diagnostic
    try:
        import inspect
        logging.getLogger(__name__).info(
            "asyncio loaded from: %s",
            (inspect.getsourcefile(asyncio) or inspect.getfile(asyncio))
        )
        import asyncio.base_events as _be  # assert import works
    except Exception as e:
        logging.getLogger(__name__).exception("Asyncio diagnostic failed: %s", e)


from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtCore import Qt, QUrl, QStandardPaths

_QT_APP = QApplication.instance() or QApplication(sys.argv)

if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def pick_h5ad_path(title="Select .h5ad file") -> str | None:
    non_native = sys.platform.startswith("linux")

    # Build options correctly (no mixing with int 0)
    opts = QFileDialog.Options()
    opts |= QFileDialog.Option.ReadOnly
    if non_native:
        opts |= QFileDialog.Option.DontUseNativeDialog  # native on Win/mac, non-native on Linux

    dlg = QFileDialog(None, title)
    dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
    dlg.setNameFilters(["AnnData H5AD (*.h5ad)", "All files (*.*)"])
    dlg.setOptions(opts)

    # Optional sidebar (works in both modes)
    sidebar = []
    for loc in (QStandardPaths.DocumentsLocation,
                QStandardPaths.DesktopLocation,
                QStandardPaths.DownloadLocation):
        for p in QStandardPaths.standardLocations(loc):
            sidebar.append(QUrl.fromLocalFile(p))
    if sidebar:
        dlg.setSidebarUrls(sidebar)

    path = None
    if dlg.exec() == QFileDialog.DialogCode.Accepted:
        sel = dlg.selectedFiles()
        if sel:
            path = sel[0]

    _QT_APP.processEvents()
    return path or None

import logging
import socket

import panel as pn

from session_state import SessionState
from tabs.overview_tab import overview_tab
from tabs.preprocessing_tab import preprocessing_tab
from tabs.analysis_tab import analysis_tab

from utils import logger, log_time, logging

from logging.handlers import RotatingFileHandler
from pathlib import Path

# Verbose server-side logs while we’re iterating
logging.getLogger().setLevel(logging.INFO)

# Plotly support + loading overlay on slow renders
pn.extension('plotly', defer_load=True, loading_indicator=True)

MIN_PF_VERSION = os.environ.get("PF_MIN_PF_VERSION", "1.5.0")  # until we package

#DEV = True  # change for env variable
DEV = os.getenv("PV_DEV", "0") == "1"

def setup_logging(app_name="ProteoViewer", filename="proteoviewer.log"):
    # Prefer the folder containing the EXE (or the script during dev)
    if getattr(sys, "frozen", False):  # PyInstaller
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    log_path = base_dir / filename
    fallback_used = False

    # If we can't write next to the exe (e.g., Program Files), fall back
    try:
        base_dir.mkdir(parents=True, exist_ok=True)  # no-op if exists
        with open(log_path, "a", encoding="utf-8"):  # permission probe
            pass
    except Exception:
        fallback_used = True
        alt_base = (os.getenv("LOCALAPPDATA")
                    or os.getenv("APPDATA")
                    or str(Path.home()))
        base_dir = Path(alt_base) / app_name
        base_dir.mkdir(parents=True, exist_ok=True)
        log_path = base_dir / filename

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)

    # Also capture Bokeh/Panel/Tornado logs
    for name in ("bokeh", "panel", "tornado"):
        logging.getLogger(name).setLevel(logging.INFO)

    # Redirect print()/tracebacks to the log file
    class _StreamToLogger:
        def __init__(self, level): self.level = level
        def write(self, buf):
            for line in buf.rstrip().splitlines():
                logging.log(self.level, line)
        def flush(self): pass

    sys.stdout = _StreamToLogger(logging.INFO)
    sys.stderr = _StreamToLogger(logging.ERROR)

    def _excepthook(exc_type, exc, tb):
        logging.exception("Unhandled exception", exc_info=(exc_type, exc, tb))
        try:
            print(f"A fatal error occurred. See log at: {log_path}")
        except Exception:
            pass
    sys.excepthook = _excepthook

    if fallback_used:
        logging.warning(
            "Could not write log next to the executable; using fallback at: %s",
            log_path
        )

    return str(log_path)

LOGFILE = setup_logging()

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
    specs = [
        ("Overview",      lambda: overview_tab(state)),
        ("Preprocessing", lambda: preprocessing_tab(state)),
        ("Analysis",      lambda: analysis_tab(state)),
    ]

    tabs = pn.Tabs(dynamic=False)  # keep content mounted once built
    holders, builders, built = [], [], []

    for label, builder in specs:
        holder = pn.Column(sizing_mode="stretch_both", styles={"min-height": "240px"})
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
                import traceback
                tb = traceback.format_exc()
                holder[:] = [pn.pane.Markdown(f"**Error while building tab:** {e}")]
                print(f"[tabs] do_build({i}) EXCEPTION:", e, "\n", tb, flush=True)
            finally:
                holder.loading = False

        pn.state.curdoc.add_next_tick_callback(do_build)

    def _set_visibility(active: int):
        for j, h in enumerate(holders):
            h.visible = (j == active)

    _set_visibility(tabs.active)
    build_index(tabs.active)

    def _on_active(event):
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
    pick_btn   = pn.widgets.Button(name="Browse system files", button_type="primary")
    exit_btn   = pn.widgets.Button(name="Exit", button_type="danger")
    status     = pn.pane.Markdown("### Please upload a .h5ad ProteoFlux file.")
    content    = pn.Column(sizing_mode="stretch_width")  # replaced reactively below

    def _load(adata, fname):
        state = SessionState.initialize(adata)
        tabs = _lazy_tabs(state)
        content.clear()
        content.append(tabs)
        status.object = f"**Loaded:** {fname}"

    # Native system file dialog (local dev) → loads directly from path (no WS)
    def _on_pick_path(event):
        from anndata import read_h5ad
        try:
            status.object = "Waiting for system file dialog…"
            path = pick_h5ad_path()

            #import tkinter as tk
            #from tkinter import filedialog

            #root = tk.Tk()
            #root.withdraw()
            #try:
            #    root.attributes('-topmost', True)  # bring dialog to front (best effort)
            #except Exception:
            #    pass

            #path = filedialog.askopenfilename(
            #    title="Select .h5ad file",
            #    filetypes=[("AnnData H5AD", "*.h5ad"), ("All files", "*.*")],
            #)
            #root.destroy()

            if not path:
                status.object = "Selection cancelled."
                return

            status.object = "Loading…"
            adata = read_h5ad(path)  # disk read, no WebSocket
            _load(adata, os.path.basename(path))
        except Exception as e:
            import traceback
            status.object = f"**Error (system dialog):** {e}"
            print("[_on_pick_path] EXCEPTION:", e, "\n", traceback.format_exc(), flush=True)

    # Wire events
    pick_btn.on_click(_on_pick_path)

    # Exit button
    def _on_exit(event):
        for w in (pick_btn, exit_btn, status):
            w.visible = False
        content[:] = [pn.pane.Markdown("## Thanks for using ProteoViewer!")]
        pn.state.curdoc.add_next_tick_callback(lambda: os._exit(0))

    exit_btn.on_click(_on_exit)

    # Dev mode: load default without clicking
    if DEV:
        try:
            from anndata import read_h5ad
            adata = read_h5ad("proteoflux_results.h5ad")
            _load(adata, "proteoflux_results.h5ad")
            logging.info("DEV autoload successful.")
        except Exception:
            logging.exception("DEV autoload failed; starting with empty UI.")
    #if DEV:
    #    from anndata import read_h5ad
    #    adata = read_h5ad("proteoflux_results.h5ad")
    #    _load(adata, "proteoflux_results.h5ad")

    controls = pn.Column(
        pn.Row(pick_btn, exit_btn, sizing_mode="stretch_width"),
        pn.Row(status, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
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
        websocket_max_message_size=2000*1024*1024,            # big WS frames (other UI)
        http_server_kwargs={"max_buffer_size": 2000*1024*1024},  # generous HTTP buffer
    )

