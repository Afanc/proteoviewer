import os, sys
import socket
import uuid

import panel as pn

from session_state import SessionState
from tabs.overview_tab import overview_tab
from tabs.overview_tab_phospho import overview_tab_phospho
from tabs.preprocessing_tab import preprocessing_tab
from tabs.analysis_tab import analysis_tab

from utils import logger, log_time, logging  # keep as-is; used elsewhere

# Console-only logging (no file handlers)
logging.getLogger().setLevel(logging.INFO)

# Panel extensions
pn.extension('plotly', 'tabulator', defer_load=True, loading_indicator=True)

MIN_PF_VERSION = os.environ.get("PF_MIN_PF_VERSION", "1.7.0")

# Single flag: dev (local run). If not dev => server mode.
DEV = os.getenv("PV_DEV", "0") == "1"

# Server upload root (server ONLY; dev never uses this)
UPROOT = os.environ.get("PV_UPLOAD_DIR", "/mnt/DATA/proteoviewer_uploads")

# Lazy Qt app for native dialogs (dev only)
_QT_APP = None


def pick_h5ad_path(title="Select .h5ad file") -> str | None:
    """Native system dialog for local dev; no-op on server."""
    if not DEV:
        return None
    from PySide6.QtWidgets import QApplication, QFileDialog
    from PySide6.QtCore import QUrl, QStandardPaths
    global _QT_APP
    _QT_APP = _QT_APP or (QApplication.instance() or QApplication(sys.argv))
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    opts = QFileDialog.Options()
    opts |= QFileDialog.Option.ReadOnly
    if sys.platform.startswith("linux"):
        opts |= QFileDialog.Option.DontUseNativeDialog

    dlg = QFileDialog(None, title)
    dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
    dlg.setNameFilters(["AnnData H5AD (*.h5ad)", "All files (*.*)"])
    dlg.setOptions(opts)

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
    """Find an open port for local dev (allows multiple instances)."""
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
    Spinner thanks to loading_indicator=True; revisits are instant.
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

    tabs = pn.Tabs(dynamic=False, sizing_mode="stretch_both")
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
                content = builders[i]()
                holder[:] = [content]
                built[i] = True
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                holder[:] = [pn.pane.Markdown(f"**Error while building tab:** {e}")]
                print(f"[tabs] do_build({i}) EXCEPTION:", e, "\n", tb, flush=True)
            finally:
                holder.loading = False

        do_build()

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


def _session_id():
    sc = getattr(pn.state.curdoc, "session_context", None)
    sid = getattr(sc, "id", None)
    return sid or str(uuid.uuid4())[:8]


@log_time("Building app")
def build_app():
    """
    - DEV: native file picker (no /mnt/data copy)
    - SERVER: Panel FileInput; copy to /mnt/data/proteoviewer_uploads/<session>/ then load
    """
    status   = pn.pane.Markdown("### Please load a .h5ad ProteoFlux file.")
    content  = pn.Column(pn.Spacer(height=1), sizing_mode="stretch_both", styles={"min-height": "300px"})

    # Shared loader
    def _load(adata, fname):
        ok, msg, _ = _check_pf_meta(adata)
        if not ok:
            status.object = f"**Incompatible file** · {msg}"
            return
        state = SessionState.initialize(adata)
        tabs = _lazy_tabs(state)
        content[:] = [tabs]
        try:
            tabs.active = 0
        except Exception:
            pass
        status.object = f"**Loaded:** {fname}"

    # ---- DEV UI ----
    if DEV:
        pick_btn = pn.widgets.Button(name="Browse system files", button_type="primary")

        # Native system file dialog (loads directly from path)
        def _on_pick_path(event):
            from anndata import read_h5ad
            try:
                status.object = "Waiting for system file dialog…"
                path = pick_h5ad_path()
                if not path:
                    status.object = "Selection cancelled."
                    return
                status.object = "Loading…"
                adata = read_h5ad(path)
                _load(adata, os.path.basename(path))
            except Exception as e:
                import traceback
                status.object = f"**Error (system dialog):** {e}"
                print("[_on_pick_path] EXCEPTION:", e, "\n", traceback.format_exc(), flush=True)

        pick_btn.on_click(_on_pick_path)

        controls = pn.Column(
            pn.Row(pick_btn, sizing_mode="stretch_width"),
            pn.Row(status, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

        # Optional autoload in dev (unchanged)
        try:
            from anndata import read_h5ad
            adata = read_h5ad("proteoflux_results_phospho.h5ad")
            _load(adata, "proteoflux_results_phospho.h5ad")
            logging.info("DEV autoload successful.")
        except Exception:
            logging.exception("DEV autoload failed; starting with empty UI.")

    # ---- SERVER UI ----
    else:
        # Use Panel's FileInput and copy to /mnt/data/<session>/ before loading
        file_in = pn.widgets.FileInput(accept='.h5ad', multiple=False)

        def _on_file_in(event):
            if not file_in.value:
                return
            from anndata import read_h5ad
            try:
                status.object = "Uploading…"
                sid = _session_id()
                dest_dir = os.path.join(UPROOT, sid)
                os.makedirs(dest_dir, exist_ok=True)
                fname = file_in.filename or "upload.h5ad"
                dest_path = os.path.join(dest_dir, fname)
                with open(dest_path, "wb") as f:
                    f.write(file_in.value)
                status.object = f"Saved to `{dest_path}`. Loading…"
                adata = read_h5ad(dest_path)
                _load(adata, fname)
            except Exception as e:
                import traceback
                status.object = f"**Upload error:** {e}"
                print("[server FileInput] EXCEPTION:", e, "\n", traceback.format_exc(), flush=True)

        file_in.param.watch(_on_file_in, 'value')

        controls = pn.Column(
            pn.Row(file_in, sizing_mode="stretch_width"),
            pn.Row(status, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    app = pn.Column("# ProteoViewer", controls, content, sizing_mode="stretch_width")
    return app


if os.environ.get("PV_PROGRAMMATIC", "0") != "1":
    # Normal 'panel serve app.py' or 'python app.py' usage
    app = build_app()
    app.servable()
    if __name__ == "__main__":
        pn.serve(
            app,
            title="ProteoViewer",
            port=(get_free_port() if DEV else 5007),
            autoreload=DEV,
            show=DEV,
            websocket_max_message_size=2000*1024*1024,
            http_server_kwargs={"max_buffer_size": 2000*1024*1024},
        )

