import os, sys
import socket
import uuid

import panel as pn

import re
from utils.session_state import SessionState
from tabs.overview_tab import overview_tab
from tabs.overview_tab_phospho import overview_tab_phospho
from tabs.preprocessing_tab import preprocessing_tab
from tabs.analysis_tab import analysis_tab

from utils.layout_utils import make_vr, make_hr
from utils.utils import logger, log_time, logging

from pathlib import Path
import importlib.metadata as importlib_metadata

# Console-only logging (no file handlers)
logging.getLogger().setLevel(logging.INFO)

# Panel extensions
pn.extension('plotly', 'tabulator', defer_load=True, loading_indicator=True, design="native")
pn.config.loading_spinner = 'petal'

# with guard for the exe
doc = pn.state.curdoc
if doc is not None:
    doc.title = "ProteoViewer"

MIN_PF_VERSION = os.environ.get("PF_MIN_PF_VERSION", "1.7.7")

# flag: dev (local run). If not dev => server mode.
DEV = os.getenv("PV_DEV", "0") == "1"

FROZEN = bool(getattr(sys, "frozen", False))
DESKTOP = (sys.platform == "win32") and FROZEN

APP_VERSION_DESKTOP = "1.8.5"

def _resource_file(name: str) -> Path:
    """
    Resource location:
      - Windows frozen (PyInstaller onedir): <exe_dir>/_internal/resources/<name> (preferred)
                                            <exe_dir>/resources/<name> (fallback)
      - Everything else (Linux server/dev):  <source_dir>/resources/<name>  (unchanged)
    """
    if DESKTOP:
        exe_dir = Path(sys.executable).resolve().parent
        p = (exe_dir / "_internal" / "resources" / name)
        if p.is_file():
            return p
        return (exe_dir / "resources" / name)

    # Linux server/dev
    return (Path(__file__).resolve().parent / "resources" / name)


def _resource_bytes(name: str) -> bytes:
    p = _resource_file(name).resolve()
    if not p.is_file():
        raise RuntimeError(f"Missing resource file: {p}")
    return p.read_bytes()


def _read_version_from_spec(spec_path: str) -> str:

    text = Path(spec_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'(?i)\bversion\b\s*[:=]\s*[\'"]?(\d+\.\d+\.\d+(?:[-+.\w]*)?)', text)
    return m.group(1) if m else "0.0.0"


def _get_app_version() -> str:
    v = os.environ.get("PV_VERSION")
    if v:
        return v.strip()

    if DESKTOP:
        return APP_VERSION_DESKTOP

    # Linux dev/server: prefer package metadata if installed
    for dist_name in ("proteoviewer", "ProteoViewer"):
        try:
            return importlib_metadata.version(dist_name)
        except Exception:
            pass

    # Linux dev from source: fallback to local spec
    try:
        spec_path = Path(__file__).resolve().parent / "proteoviewer.spec"
        v = _read_version_from_spec(str(spec_path))
        return v
    except Exception:
        return "0.0.0"

def _get_upload_root() -> str:
    """
    Server-only upload directory.
    - If PV_UPLOAD_DIR is set, use it.
    - Otherwise, prefer the legacy Biozentrum path if it exists.
    - Else, fall back to a user cache directory.
    """
    v = os.environ.get("PV_UPLOAD_DIR")
    if v:
        return v
    legacy = Path("/mnt/DATA/proteoviewer_uploads")
    if legacy.exists():
        return str(legacy)
    fallback = Path.home() / ".cache" / "proteoviewer" / "uploads"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)

UPROOT = _get_upload_root()

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

html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow-x: hidden !important;
  overflow-y: hidden;
}

.bk-root {
  height: 100%;
}
* { box-sizing: border-box; }
"""
pn.config.raw_css = (pn.config.raw_css or []) + [CSS]


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
        _about_modal_content(version),
        title="About",
        collapsible=True,
        collapsed=True,
        styles={"max-width": "720px"},
    )

def _build_header(area_center, version: str, dev_flag: bool) -> pn.Column:
    """Header with left stack (title, browse, status) and right stack (logo, facility)."""
    pv_logo = _resource_bytes("pv_banner.png")
    pv_logo_pane = pn.pane.PNG(pv_logo,
                               width=200,
                               height=90,
                               sizing_mode="fixed",
                               margin=(-10,0,-40,0),
                               embed=True)


    ver_label = f"v{version}" + (" · DEV" if dev_flag else "")
    pv_ver = pn.pane.Markdown(
        f"<span class='pv-badge'>{ver_label}</span>",
        margin=(0,0,-20,-20),
    )

    info = pn.widgets.TooltipIcon(
        value="""
    Developed at the
    Proteomics Core Facility
    Biozentrum - University of Basel
    dariush.mollet@unibas.ch
    DOI: 10.5281/zenodo.18640999
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
                     pn.Spacer(width=30),
                     area_center,
            sizing_mode="stretch_width",
            height=40,
    )


    # Right side: logo above facility - same width, left-aligned inside the block
    FACILITY_WIDTH = 140

    facility_logo_path = os.environ.get("PV_FACILITY_LOGO_RESOURCE",
                                        "Biozentrum_Logo_2011.png")
    facility_logo = _resource_bytes(facility_logo_path)
    facility_logo_pane = pn.pane.PNG(facility_logo, width=FACILITY_WIDTH, height=90, margin=(0, 10, 6, -15))
    right_top_row = pn.Row(facility_logo_pane,
                           pn.Spacer(width=50),
                           height=90, align="center")
    right_block = pn.Column(right_top_row,
                            width=FACILITY_WIDTH, sizing_mode="fixed", height=90)

    # One header row: left stack and right stack aligned at the top
    mainbar = pn.Row(
        left_block,
        pn.Spacer(),
        right_block,
        sizing_mode="stretch_width",
        align="center",
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
            ("Preprocessing-PO4", lambda: preprocessing_tab(state)),
            ("Analysis-PO4", lambda: analysis_tab(state)),
        ]

    tabs = pn.Tabs(dynamic=True, sizing_mode="stretch_width")
    holders, builders, built = [], [], []

    for label, builder in specs:
        holder = pn.Column(sizing_mode="stretch_width",
                           styles={"min-height": "240px"})
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
    status   = pn.pane.Markdown("### Please load a .h5ad ProteoFlux file.", margin=(0,0,0,10))
    content  = pn.Column(pn.Spacer(height=1),
                         sizing_mode="stretch_width",
                         styles={"min-height": "300px"})

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
            pn.Spacer(height=10),
            pn.Row(pick_btn, sizing_mode="stretch_width"),
            pn.Row(status, sizing_mode="stretch_width", css_classes=["pv-status"]),
            sizing_mode="stretch_width",
        )

        # Optional autoload in dev
        try:
            from anndata import read_h5ad
            adata = read_h5ad("data/proteoflux_results_phospho.h5ad")
            #adata = read_h5ad("data/proteoflux_results.h5ad")
            _load(adata, "proteoflux_results.h5ad")
            logging.info("DEV autoload successful.")
        except Exception:
            logging.exception("DEV autoload failed; starting with empty UI.")

    # ---- SERVER UI ----
    else:
        # Use Panel's FileInput and copy to /path.../<session>/ before loading
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
            pn.Spacer(height=10),
            pn.Row(file_in, sizing_mode="stretch_width"),
            pn.Row(status, sizing_mode="stretch_width", css_classes=["pv-status"]),
            sizing_mode="stretch_width",
        )

    # Build colored header with version + facility tag
    version = _get_app_version()
    header  = _build_header(controls, version, DEV)

    # header (colored) on top, then the tabs/content
    app = pn.Column(header,
                    content,
                    sizing_mode="stretch_width")

    return app

if __name__ != "__main__":
    build_app().servable()

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        raise ValueError(f"{name} must be an integer, got: {v!r}")


def _env_csv(name: str) -> list[str]:
    v = os.environ.get(name, "")
    if not v.strip():
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def _default_allow_origins(port: int) -> list[str]:
    """
    Bokeh/Panel websocket origin allowlist.
    Defaults are safe/local + current hostnames. For public deployments,
    set PV_ALLOW_ORIGINS (comma-separated), e.g.:
      PV_ALLOW_ORIGINS="proteoviewer.example.org,proteoviewer.example.org:443"
    """
    origins: set[str] = set()
    origins.add(f"localhost:{port}")
    origins.add(f"127.0.0.1:{port}")
    try:
        origins.add(f"{socket.gethostname()}:{port}")
        origins.add(f"{socket.getfqdn()}:{port}")
    except Exception:
        pass
    for o in _env_csv("PV_ALLOW_ORIGINS"):
        origins.add(o)
        if ":" not in o:
            origins.add(f"{o}:{port}")
    return sorted(origins)

if __name__ == "__main__":
    target = build_app

    if DEV:
        # Local dev: first free port, autoreload, window popup
        pn.serve(
            target,
            title="ProteoViewer",
            address="localhost",
            port=get_free_port(),
            autoreload=True,
            show=True,
            websocket_max_message_size=2000 * 1024 * 1024,
            http_server_kwargs={"max_buffer_size": 2000 * 1024 * 1024},
        )
    elif DESKTOP:
        # Windows EXE: local single-process server (no num_procs), open browser
        port = get_free_port()
        pn.serve(
            target,
            title="ProteoViewer",
            address="localhost",
            port=port,
            autoreload=False,
            show=True,
            websocket_max_message_size=2_000 * 1024 * 1024,
            http_server_kwargs={"max_buffer_size": 2_000 * 1024 * 1024},
            allow_websocket_origin=[
                f"localhost:{port}",
                f"127.0.0.1:{port}",
            ],
        )
    else:
        # Server mode: fixed port by default, big buffers, proper origins, no GUI
        port = _env_int("PV_PORT", 5007)
        address = os.environ.get("PV_ADDRESS", "0.0.0.0")
        num_procs = _env_int("PV_NUM_PROCS", 8)
        pn.serve(
            target,
            title="ProteoViewer",
            address=address,
            port=port,
            autoreload=False,
            show=False,
            websocket_max_message_size=2_000 * 1024 * 1024,
            http_server_kwargs={"max_buffer_size": 2_000 * 1024 * 1024},
            allow_websocket_origin=_default_allow_origins(port),
            num_procs=num_procs,
         )
