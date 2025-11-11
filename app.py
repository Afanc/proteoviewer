import os, sys
import socket
import uuid

import panel as pn

from session_state import SessionState
from tabs.overview_tab import overview_tab
from tabs.overview_tab_phospho import overview_tab_phospho
from tabs.preprocessing_tab import preprocessing_tab
from tabs.analysis_tab import analysis_tab

from layout_utils import make_vr, make_hr
from utils import logger, log_time, logging  # keep as-is; used elsewhere

# Console-only logging (no file handlers)
logging.getLogger().setLevel(logging.INFO)

# Panel extensions
pn.extension('plotly', 'tabulator', defer_load=True, loading_indicator=True, design="native")
pn.config.loading_spinner = 'petal'
pn.state.curdoc.title = "ProteoViewer"

MIN_PF_VERSION = os.environ.get("PF_MIN_PF_VERSION", "1.7.4")

# Single flag: dev (local run). If not dev => server mode.
DEV = os.getenv("PV_DEV", "0") == "1"

# Server upload root (server ONLY; dev never uses this)
UPROOT = os.environ.get("PV_UPLOAD_DIR", "/mnt/DATA/proteoviewer_uploads")

# Lazy Qt app for native dialogs (dev only)
_QT_APP = None

CSS = r"""
/* ---------- Toned-down, square look ---------- */

/* App header: calm indigo tint, square edges, no shadows */
.pv-header {
  background: #e9effa;          /* soft indigo-10-ish */
  color: #1f2a44;
  padding: 12px 16px;
  border-radius: 0;              /* square */
  box-shadow: none;
  border-bottom: 1px solid #d8e0f4;
}

/* Title + version badge: subtle */
.pv-title {
  font-weight: 700;
  font-size: 20px;
  letter-spacing: 0.2px;
  display: flex;
  align-items: center;
  gap: 10px;
  color: #182239;
}
.pv-badge {
  font-weight: 600;
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 999px;          /* tiny pill is okay */
  background: #dfe6fa;
  border: 1px solid #cdd8fb;
  color: #24304a;
}

/* Facility text on right */
.pv-facility {
  font-size: 12px;
  opacity: 0.9;
  white-space: normal;
  color: #2a3550;
  text-align: left;
}

/* Controls/status area: same color as header (no translucent box) */
.pv-subbar {
  background: #e9effa;          /* same as .pv-header */
  border: none;
  border-radius: 0;              /* square */
  padding: 8px 0 0 0;           /* small top spacing only */
}

/* Status text should remain readable on the tinted header */
.pv-status p, .pv-status h3, .pv-status h4 { color: #1f2a44 !important; }
/* Tabs header bar (slightly lighter than header) */
.bk-tabs-header {
  background: #eff3fd !important;   /* lighter than header (#e9effa) */
  border: 1px solid #e1e8fb;
  border-top: none;                  /* flush to header */
  border-radius: 0;                  /* square */
  padding: 0;
  box-shadow: none;
}

/* Default tab buttons: transparent so they blend into the header bar */
.bk-tab {
  background: transparent;           /* avoid inner “box” layer */
  border-radius: 0;                  /* square */
  margin: 0;
  padding: 8px 14px;
  color: #2a3550;
  font-weight: 500;
  border: none;
  outline: none;
  box-shadow: none;
  position: relative;
  border-bottom: 2px solid transparent; /* reserve space to avoid jump */
}

/* Remove any default pseudo-element highlight some themes add */
.bk-tab::before,
.bk-tab::after {
  display: none !important;
}

/* Hover: very subtle */
.bk-tab:hover {
  background: rgba(0,0,0,0.03);
}

/* Active tab: single, slightly darker background; darker text; no top bar */
.bk-tab.bk-active {
  background: #e7ecfb;               /* gentle step darker than bar */
  color: #162039;
  box-shadow: none;
  border: none;
  border-bottom: 2px solid #5a6ecf;   /* thin, clean indicator */
}

/* Focus states: no glow/outline */
.bk-tab:focus, .bk-tab.bk-active:focus {
  outline: none;
  box-shadow: none;
}

html, body, .bk-root {
  overflow-x: hidden !important;
}
* { box-sizing: border-box; }
"""
pn.config.raw_css = (pn.config.raw_css or []) + [CSS]

def _read_version_from_spec(spec_path: str = "proteoviewer.spec") -> str:
    """
    Read version like:
      version = "1.2.3"
      version = '1.2.3'
      Version: 1.2.3
    Fallback '0.0.0' if nothing found (never blank).
    """
    from pathlib import Path
    import re
    try:
        text = Path(spec_path).read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'(?i)\bversion\b\s*[:=]\s*[\'"]?(\d+\.\d+\.\d+(?:[-+.\w]*)?)', text)
        return m.group(1) if m else "0.0.0"
    except Exception:
        return "0.0.0"


def _about_modal_content(version: str) -> pn.viewable.Viewable:
    """Small modal with credits, companion note, and citation placeholder."""
    md = pn.pane.Markdown(
        f"""
### ProteoViewer {version}
**Developed at:** Proteomics Core Facility, Biozentrum – University of Basel
**Author / contact:** Your Name – <you@unibas.ch>

**Companion tool:** *ProteoViewer* is the visualization companion to the **ProteoFlux** analysis pipeline.

**How to cite (placeholder):**
If you use ProteoViewer/ProteoFlux, please cite:
*ProteoFlux (2025), Proteomics Core Facility, Biozentrum, University of Basel*
(DOI pending)
        """.strip(),
        sizing_mode="stretch_width"
    )
    # A compact card looks neat in the modal
    return pn.Card(md, title="About ProteoViewer", collapsible=False, styles={"max-width": "720px"})

def _make_about_card(version: str) -> pn.Card:
    # Collapsed by default; header is the toggle
    return pn.Card(
        _about_modal_content(version),   # reuse your existing content
        title="About",
        collapsible=True,                # header acts like a button
        collapsed=True,                  # start closed
        sizing_mode="fixed",
        width=720,                       # same width you liked
        styles={"max-width": "720px"},   # keep it from growing wider
    )

def _build_header(area_center, version: str, dev_flag: bool) -> pn.Column:
    """Header with left stack (title, browse, status) and right stack (logo, facility)."""
    pv_logo_path = "resources/pv_banner.png"
    pv_logo_pane = pn.pane.PNG(pv_logo_path, width=200, sizing_mode="fixed", margin=(-10,0,-40,0))

    ver_label = f"v{version}" + (" · DEV" if dev_flag else "")
    pv_ver = pn.pane.Markdown(
        f"<span class='pv-badge'>{ver_label}</span>",
        margin=(0,0,0,-20),
    )

    info = pn.widgets.TooltipIcon(
        value="""
    Proteomics Core Facility
    Biozentrum - University of Basel
    dariush.mollet@unibas.ch
    [DOI pending]
        """,
        margin=(0,0,0,-125),
    )

    # Left side: title on top, then the existing controls/status column underneath
    left_block = pn.Row(
                     pv_logo_pane,
                     pn.Spacer(width=10),
                     pn.Column(
                         pn.Spacer(height=25),
                         info,
                         pn.Spacer(height=5),
                         pv_ver,
                         width=100,
                     ),
                     area_center,
            sizing_mode="stretch_width",
            height=80,
    )


    # Right side: logo above facility — same width, left-aligned inside the block
    FACILITY_WIDTH = 140
    bz_logo_path = "resources/Biozentrum_Logo_2011.png"
    bz_logo_pane = pn.pane.PNG(bz_logo_path, width=FACILITY_WIDTH, sizing_mode="fixed", margin=(0, 50, 6, -15))
    facility = pn.pane.Markdown(
        "<div class='pv-facility'><b>Proteomics Core Facility</b></div>",
        width=FACILITY_WIDTH, sizing_mode="fixed",
    )
    right_top_row = pn.Row(bz_logo_pane,
                           pn.Spacer(width=100),
                           sizing_mode="fixed", align="center")
    right_block = pn.Column(right_top_row, width=FACILITY_WIDTH, sizing_mode="fixed")

    # One clean header row: left stack and right stack aligned at the top
    mainbar = pn.Row(
        left_block,
        pn.Spacer(),
        right_block,
        sizing_mode="stretch_width",
        align="center",                   # top-align both columns
        css_classes=["pv-subbar"],
    )

    return pn.Column(mainbar, sizing_mode="stretch_width", css_classes=["pv-header"])

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
            #pn.Spacer(height=10),
            pn.Row(pick_btn, sizing_mode="stretch_width"),
            pn.Row(status, sizing_mode="stretch_width", css_classes=["pv-status"]),
            sizing_mode="stretch_width",
        )

        # Optional autoload in dev (unchanged)
        try:
            from anndata import read_h5ad
            adata = read_h5ad("proteoflux_results.h5ad")
            _load(adata, "proteoflux_results.h5ad")
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
                status.object = f"Loading…"
                adata = read_h5ad(dest_path)
                _load(adata, fname)
            except Exception as e:
                import traceback
                status.object = f"**Upload error:** {e}"
                print("[server FileInput] EXCEPTION:", e, "\n", traceback.format_exc(), flush=True)

        file_in.param.watch(_on_file_in, 'value')

        controls = pn.Column(
            #pn.Spacer(height=10),
            pn.Row(file_in, sizing_mode="stretch_width"),
            pn.Row(status, sizing_mode="stretch_width", css_classes=["pv-status"]),
            sizing_mode="stretch_width",
        )

    #app = pn.Column("# ProteoViewer", controls, content, sizing_mode="stretch_width")
    # Build colored header with version + facility tag
    version = _read_version_from_spec("proteoviewer.spec")
    header  = _build_header(controls, version, DEV)

    # Final layout: header (colored) on top, then the tabs/content
    app = pn.Column(header, content, sizing_mode="stretch_width")

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

