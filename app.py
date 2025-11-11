import os, sys
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtCore import Qt, QUrl, QStandardPaths

import logging
import socket

import tempfile
import uuid
from pathlib import Path
import panel as pn
import time

from session_state import SessionState
from tabs.overview_tab import overview_tab
from tabs.overview_tab_phospho import overview_tab_phospho
from tabs.preprocessing_tab import preprocessing_tab
from tabs.analysis_tab import analysis_tab

from utils import logger, log_time, logging

from logging.handlers import RotatingFileHandler
from pathlib import Path

# Verbose server-side logs while we’re iterating
logging.getLogger().setLevel(logging.INFO)

# Plotly support + loading overlay on slow renders
pn.extension('plotly', defer_load=True, loading_indicator=True)
pn.extension('tabulator', defer_load=True, loading_indicator=True)

MIN_PF_VERSION = os.environ.get("PF_MIN_PF_VERSION", "1.7.0")  # until we package
UPLOAD_ROOT = Path(os.environ.get("PV_UPLOAD_DIR", "/mnt/DATA/proteoviewer_uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

#DEV = True  # change for env variable
DEV = os.getenv("PV_DEV", "0") == "1"

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
    meta = adata.uns.get("proteoflux", {}) or {}
    pfv = meta.get("pf_version")
    created = meta.get("created_at", "unknown time")

    if not pfv:
        return (False, "No ProteoFlux version found in uns['proteoflux'].", meta)

    if _parse_semver(pfv) < _parse_semver(MIN_PF_VERSION):
        return (False, f"File written by ProteoFlux {pfv} (Required >= {MIN_PF_VERSION}). Please re-export with a newer ProteoFlux.", meta)

    pilot_study_mode = adata.uns["pilot_study_mode"]

    if pilot_study_mode:
        return (False, "This experiment has at least 1 Condition with only 1 Replicate - Pilot Study Mode. Nothing to show in Proteoviewer.", meta)


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
    if state.adata.uns['analysis'].get('analysis_type'.lower(), "DIA") == "phospho":
        specs = [
            ("Overview",      lambda: overview_tab_phospho(state)),
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
    upload_in  = pn.widgets.FileInput(accept=".h5ad", multiple=False, width=220)
    status     = pn.pane.Markdown("### Please upload a .h5ad ProteoFlux file.")
    content    = pn.Column(sizing_mode="stretch_width")  # replaced reactively below

    def _load(adata, fname):
        ok, msg, _ = _check_pf_meta(adata)
        if not ok:
            # stop here — don't build tabs
            status.object = f"**Incompatible file** · {msg}"
            return

        state = SessionState.initialize(adata)
        tabs = _lazy_tabs(state)
        content.clear()
        content.append(tabs)
        status.object = f"**Loaded:** {fname}"

    # Minimal local upload (browser -> server temp -> read_h5ad(backed='r'))
    def _on_upload_change(event):
        """Write uploaded bytes to a unique temp path, then load backed='r'."""
        from anndata import read_h5ad
        data = event.new or b""
        if not data:
            return
        try:
            # Per-session-ish unique name, avoid collisions
            sid = (getattr(pn.state.curdoc, "session_context", None) and pn.state.curdoc.session_context.id) or uuid.uuid4().hex[:8]
            fname = upload_in.filename or "upload.h5ad"
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)
            sess_dir = UPLOAD_ROOT / sid
            sess_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = sess_dir / f"{int(time.time())}_{safe}"
            # Write to disk
            with open(tmp_path, "wb") as f:
                f.write(data)
            #status.object = f"Loading `{safe}` (backed='r')…"
            status.object = f"Loading `{safe}`..."
            adata = read_h5ad(str(tmp_path))#, backed="r")
            _load(adata, safe)
        except Exception as e:
            import traceback
            status.object = f"**Error (upload):** {e}"
            print("[upload] EXCEPTION:", e, "\n", traceback.format_exc(), flush=True)

    upload_in.param.watch(_on_upload_change, "value")

    # Dev mode: load default without clicking
    if DEV:
        try:
            from anndata import read_h5ad
            adata = read_h5ad("proteoflux_results_phospho.h5ad")
            _load(adata, "proteoflux_results_phospho.h5ad")
            logging.info("DEV autoload successful.")
        except Exception:
            logging.exception("DEV autoload failed; starting with empty UI.")

    # Minimal footer
    footer = pn.pane.Markdown(
        "<div style='font-size:12px;opacity:0.8;padding-top:6px'>"
        "<b>Proteomics Core Facility</b> - Biozentrum, University of Basel"
        "</div>", sizing_mode="stretch_width"
    )

    controls = pn.Column(
        pn.Row(upload_in, sizing_mode="stretch_width"),
        pn.Row(status, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    app = pn.Column(
        "# ProteoViewer",
        controls,
        content,
        sizing_mode="stretch_width")
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

