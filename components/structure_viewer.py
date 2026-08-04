from __future__ import annotations

import html
import json
import time
import os
import re
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

import panel as pn
import requests


ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{version}.pdb"
ALPHAFOLD_MODEL_VERSIONS = (6, 5, 4, 3, 2, 1)

MISSING_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_TIMEOUT = (3.0, 10.0)

REQUEST_HEADERS = {
    "User-Agent": (
        "ProteoViewer/1.9 "
        "(Biozentrum Proteomics Core Facility; structure visualization module)"
    ),
    "Accept": "application/json,text/plain,*/*",
}


@dataclass(frozen=True)
class StructureFetchResult:
    ok: bool
    path: Path | None
    message: str = ""
    version: int | None = None
    source_url: str | None = None

@dataclass(frozen=True)
class PeptideRegion:
    peptide_id: str
    start: int
    end: int
    qval: float | None = None
    range_log2: float | None = None
    selected: bool = False

def _clean_uniprot_id(uniprot_id: str | None) -> str:
    """
    Normalize a UniProt accession for AlphaFold DB lookup.

    PELSA parent-protein values may contain grouped accessions separated by
    semicolons. The caller should normally pass the first accession, but this
    helper is intentionally defensive.
    """
    token = str(uniprot_id or "").strip()
    if not token or token.lower() in {"nan", "none"}:
        return ""

    token = token.split(";", 1)[0].strip()
    token = token.split(",", 1)[0].strip()

    # AlphaFold DB canonical URLs generally use the base accession.
    # This avoids Q03169-2 failing when Q03169 exists.
    token = token.split("-", 1)[0].strip()

    if not re.fullmatch(r"[A-Za-z0-9_]+", token):
        return ""
    return token


def _structure_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)

    base = Path(os.environ.get("PV_CACHE_DIR", "cache"))
    return Path(os.environ.get("PV_STRUCTURE_CACHE_DIR", str(base / "structures")))


def _missing_cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age < MISSING_CACHE_TTL_SECONDS


def _candidate_alphafold_urls(
    uniprot: str,
    *,
    timeout: tuple[float, float],
) -> list[str]:
    """
    Resolve candidate AlphaFold coordinate URLs.

    Prefer the AlphaFold API because it can tell us the currently available
    model version. Fall back to probing recent model-version URL patterns when
    the API is unavailable or restricted.
    """
    candidates: list[str] = []
    api_url = ALPHAFOLD_API_URL.format(uniprot=uniprot)

    try:
        response = requests.get(api_url, headers=REQUEST_HEADERS, timeout=timeout)
        if response.ok:
            payload = response.json()
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    pdb_url = item.get("pdbUrl") or item.get("pdb_url")
                    if isinstance(pdb_url, str) and pdb_url:
                        candidates.append(pdb_url)

                    latest_version = item.get("latestVersion") or item.get("modelVersion")
                    try:
                        latest_version = int(latest_version)
                    except Exception:
                        latest_version = None
                    if latest_version:
                        candidates.append(
                            ALPHAFOLD_PDB_URL.format(
                                uniprot=uniprot,
                                version=latest_version,
                            )
                        )
    except (requests.exceptions.RequestException, ValueError):
        # Fall back to deterministic model-version probing below.
        pass

    for version in ALPHAFOLD_MODEL_VERSIONS:
        candidates.append(ALPHAFOLD_PDB_URL.format(uniprot=uniprot, version=version))

    # Preserve order while deduplicating.
    return list(dict.fromkeys(candidates))


def _alphafold_version_from_url(url: str | None) -> int | None:
    m = re.search(r"model_v(\d+)\.pdb", str(url or ""))
    return int(m.group(1)) if m else None

def fetch_alphafold_structure(
    uniprot_id: str | None,
    *,
    cache_dir: str | Path | None = None,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> StructureFetchResult:
    """
    Fetch an AlphaFold DB PDB file with a small on-disk cache.

    Returns a result object instead of raising for normal availability failures,
    so the UI can render a clean "not available" panel.
    """
    uniprot = _clean_uniprot_id(uniprot_id)
    if not uniprot:
        return StructureFetchResult(
            ok=False,
            path=None,
            message="No UniProt accession available for structure lookup.",
        )

    root = _structure_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    pdb_path = root / f"{uniprot}.pdb"
    meta_path = root / f"{uniprot}.json"
    missing_path = root / f"{uniprot}.missing"

    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        version = None
        source_url = None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            version = int(meta["version"]) if meta.get("version") is not None else None
            source_url = str(meta.get("source_url") or "") or None
        except Exception:
            pass
        return StructureFetchResult(
            ok=True,
            path=pdb_path,
            version=version,
            source_url=source_url,
        )

    if _missing_cache_is_fresh(missing_path):
        return StructureFetchResult(
            ok=False,
            path=None,
            message=(
                f"AlphaFold structure not available for {uniprot}. "
                "This negative cache will be retried automatically later."
            ),
        )
    if missing_path.exists():
        try:
            missing_path.unlink()
        except OSError:
            pass

    urls = _candidate_alphafold_urls(uniprot, timeout=timeout)
    transient_errors: list[str] = []


    for url in urls:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            transient_errors.append(str(exc))
            continue

        if response.status_code == 404:
            continue

        if response.status_code in {403, 429, 500, 502, 503, 504}:
            transient_errors.append(f"HTTP {response.status_code} for {url}")
            continue

        if not response.ok:
            transient_errors.append(f"HTTP {response.status_code} for {url}")
            continue

        text = response.text
        content_type = response.headers.get("content-type", "")

        # EBI can return an HTML access-restriction page. Do not cache that as
        # "missing"; it is not biological absence.
        if "text/html" in content_type.lower() or "<html" in text[:500].lower():
            transient_errors.append(f"HTML response for {url}")
            continue

        if not text.strip() or ("ATOM" not in text and "MODEL" not in text):
            transient_errors.append(f"Invalid PDB payload for {url}")
            continue

        tmp_path = pdb_path.with_name(f"{pdb_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(pdb_path)

        version = _alphafold_version_from_url(url)
        meta_path.write_text(
            json.dumps({"uniprot": uniprot, "version": version, "source_url": url}, indent=2),
            encoding="utf-8",
        )

        return StructureFetchResult(
            ok=True,
            path=pdb_path,
            version=version,
            source_url=url,
        )

    # Only now negative-cache the biological "not found" case, and only for a
    # short TTL. This avoids permanently poisoning valid accessions.
    if not transient_errors:
        missing_path.write_text("404\n", encoding="utf-8")
        return StructureFetchResult(
            ok=False,
            path=None,
            message=f"AlphaFold structure not available for {uniprot}.",
        )

    return StructureFetchResult(
        ok=False,
        path=None,
        message=(
            f"AlphaFold structure lookup failed for {uniprot}. "
            f"Last error: {transient_errors[-1]}"
        ),
    )


def _message_pane(
    message: str,
    *,
    width: int,
    height: int,
    title: str = "Structure viewer",
) -> pn.Card:
    return pn.Card(
        pn.pane.Markdown(f"**{title}**", margin=(0, 0, 4, 0)),
        pn.pane.HTML(
            f"""
            <div style="
                height:{max(height - 60, 120)}px;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
                color:#555;
                font-size:13px;
                padding:12px;
                box-sizing:border-box;
            ">
                {html.escape(str(message))}
            </div>
            """,
            width=width - 20,
            height=max(height - 45, 120),
        ),
        width=width,
        height=height,
        collapsible=False,
        hide_header=True,
        styles={
            "background": "#f9f9f9",
            "border-radius": "8px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "padding": "8px",
        },
    )

def _safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _hex_to_int(color: str) -> int:
    color = str(color or "#bdbdbd").strip()
    if color.startswith("#"):
        color = color[1:]
    if len(color) != 6:
        color = "bdbdbd"
    return int(color, 16)


def _qvalue_color(qval: float | None, *, threshold: float = 0.05) -> str:
    """
    Compact viridis-like significance palette matching the local stability idea:
    grey below threshold, stronger color for lower q-values.
    """
    q = _safe_float(qval)
    if not math.isfinite(q) or q >= threshold:
        return "#bdbdbd"

    score = 8.0 if q <= 0 else max(0.0, -math.log10(q))
    low = -math.log10(threshold)
    t = min(1.0, max(0.0, (score - low) / max(6.0 - low, 1e-9)))

    palette = ["#73d055", "#2a788e", "#355f8d", "#440154"]
    pos = t * (len(palette) - 1)
    i = int(pos)
    j = min(i + 1, len(palette) - 1)
    frac = pos - i

    def _rgb(x: str) -> tuple[int, int, int]:
        x = x.lstrip("#")
        return int(x[0:2], 16), int(x[2:4], 16), int(x[4:6], 16)

    a = _rgb(palette[i])
    b = _rgb(palette[j])
    rgb = tuple(int(round(ai + (bi - ai) * frac)) for ai, bi in zip(a, b))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _structure_color_regions(peptides: list[PeptideRegion] | None, color_mode: str) -> list[dict]:
    mode = str(color_mode or "None")
    regions = []
    for pep in peptides or []:
        q = _safe_float(pep.qval)
        selected = bool(getattr(pep, "selected", False))
        effect = abs(_safe_float(pep.range_log2))
        effect_priority = effect if math.isfinite(effect) else float("-inf")
        q_priority = -q if math.isfinite(q) else float("-inf")
        priority = (
            int(selected),
            effect_priority,
            q_priority,
            str(pep.peptide_id),
        )

        if selected:
            regions.append({
                "start": int(pep.start),
                "end": int(pep.end),
                "color": "#d62728",
                "colorInt": _hex_to_int("#d62728"),
                "selected": True,
                "_priority": priority,
            })
        elif mode == "Significance" and math.isfinite(q) and q < 0.05:
            color = _qvalue_color(q)
            regions.append({
                "start": int(pep.start),
                "end": int(pep.end),
                "color": color,
                "colorInt": _hex_to_int(color),
                "selected": selected,
                "_priority": priority,
            })
    # Mol* uses the last matching color layer as the top layer.
    regions.sort(key=lambda region: region["_priority"])
    for region in regions:
        region.pop("_priority")
    return regions



def _molstar_srcdoc(
    *,
    pdb_text: str,
    uniprot_id: str,
    div_id: str,
    peptides: list[PeptideRegion] | None = None,
    color_mode: str = "None",
    representation_mode: str = "Surface",
) -> str:
    pdb_json = json.dumps(pdb_text)
    title_json = json.dumps(f"AlphaFold {uniprot_id}")
    regions_json = json.dumps(_structure_color_regions(peptides, color_mode))
    representation_json = json.dumps(str(representation_mode or "Surface"))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/molstar@4.11.0/build/viewer/molstar.css">
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #ffffff;
      font-family: Arial, sans-serif;
    }}
    #{div_id} {{
      position: absolute;
      inset: 0;
    }}
    .pv-structure-error {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #555;
      font-size: 13px;
      text-align: center;
      padding: 12px;
      box-sizing: border-box;
    }}
  </style>
</head>
<body>
  <div id="{div_id}"></div>
  <script src="https://cdn.jsdelivr.net/npm/molstar@4.11.0/build/viewer/molstar.js"></script>
  <script>
    const pdbData = {pdb_json};
    const structureTitle = {title_json};
    const colorRegions = {regions_json};
    const representationMode = {representation_json};

    function showError(message) {{
      const root = document.getElementById("{div_id}");
      root.innerHTML = `<div class="pv-structure-error">${{message}}</div>`;
    }}

    async function main() {{
      if (!window.molstar || !window.molstar.Viewer) {{
        showError("Mol* could not be loaded.");
        return;
      }}

      const viewer = await molstar.Viewer.create("{div_id}", {{
        layoutIsExpanded: false,
        layoutShowControls: false,
        layoutShowRemoteState: false,
        layoutShowSequence: false,
        layoutShowLog: false,
        layoutShowLeftPanel: false,
        viewportShowExpand: false,
        viewportShowSelectionMode: false,
        viewportShowAnimation: false,
        pdbProvider: "pdbe",
        emdbProvider: "pdbe",
      }});

      const mvs = molstar.PluginExtensions && molstar.PluginExtensions.mvs;
      if (!mvs || !mvs.MVSData || !mvs.loadMVS) {{
        console.warn("[ProteoViewer structure] MVS unavailable; falling back to default Mol* rendering.");
        await viewer.loadStructureFromData(pdbData, "pdb", false, {{
          label: structureTitle,
        }});
        return;
      }}

      const pdbUrl = URL.createObjectURL(
        new Blob([pdbData], {{ type: "text/plain" }})
      );

      const repType = representationMode === "Cartoon" ? "cartoon" : "surface";
      const builder = mvs.MVSData.createBuilder();

      const structure = builder
        .download({{ url: pdbUrl }})
        .parse({{ format: "pdb" }})
        .modelStructure({{}});

      // Base protein: always grey. This prevents the default AlphaFold/pLDDT
      // green/blue/purple theme from leaking into this compact statistical view.
      const proteinRepresentation = structure
        .component({{ selector: "polymer" }})
        .representation({{ type: repType }});

      proteinRepresentation.color({{ color: "#bdbdbd" }});
      // Significant peptide regions only. Non-significant or absent profile
      // regions remain grey through the base representation. Apply the colors
      // to the same representation instead of creating extra components; this
      // avoids surface striping and keeps cartoon/surface behavior consistent.
      for (const region of colorRegions || []) {{
        const start = Number(region.start);
        const end = Number(region.end);
        const isSelected = Boolean(region.selected);
        if (!Number.isFinite(start) || !Number.isFinite(end)) {{
          continue;
        }}

        proteinRepresentation.color({{
          selector: {{
            beg_auth_seq_id: start,
            end_auth_seq_id: end,
          }},
          color: isSelected ? "#d62728" : String(region.color || "#bdbdbd"),
        }});

      }}
      const mvsData = builder.getState();
      await mvs.loadMVS(viewer.plugin, mvsData, {{
        sourceUrl: undefined,
        sanityChecks: false,
        replaceExisting: true,
      }});

      setTimeout(() => URL.revokeObjectURL(pdbUrl), 5000);
    }}

    main().catch(err => {{
      console.error(err);
      showError("Structure rendering failed. See browser console for details.");
    }});
  </script>
</body>
</html>"""


def build_structure_viewer_pane(
    uniprot_id: str | None,
    *,
    peptides: list[PeptideRegion] | None = None,
    color_mode: str = "None",
    representation_mode: str = "Surface",
    cache_dir: str | Path | None = None,
    width: int = 400,
    height: int = 500,
) -> pn.viewable.Viewable:
    """
    Return a compact Mol* structure viewer pane for a UniProt accession.

    This first implementation is intentionally render-only. Peptide overlays,
    focus behavior, and coloring modes should be layered on after the basic
    AlphaFold/Mol* path is verified in the deployed Panel app.
    """
    uniprot = _clean_uniprot_id(uniprot_id)
    if not uniprot:
        return _message_pane(
            "No UniProt accession available for structure lookup.",
            width=width,
            height=height,
        )

    result = fetch_alphafold_structure(uniprot, cache_dir=cache_dir)
    if not result.ok or result.path is None:
        return _message_pane(
            result.message or f"Structure not available for {uniprot}.",
            width=width,
            height=height,
        )

    try:
        pdb_text = result.path.read_text(encoding="utf-8")
    except OSError as exc:
        return _message_pane(
            f"Cached structure could not be read for {uniprot}: {exc}",
            width=width,
            height=height,
        )

    div_id = f"pv-molstar-{uuid.uuid4().hex}"
    srcdoc = _molstar_srcdoc(
        pdb_text=pdb_text,
        uniprot_id=uniprot,
        div_id=div_id,
        peptides=peptides,
        color_mode=color_mode,
        representation_mode=representation_mode,
    )

    iframe = (
        f"<iframe "
        f"srcdoc='{html.escape(srcdoc, quote=True)}' "
        f"style='width:100%; height:{height - 58}px; border:0; border-radius:6px; background:white;' "
        f"allow='fullscreen'>"
        f"</iframe>"
    )

    version_txt = ""
    if result.version is not None:
        version_txt = f" model v{int(result.version)}"
    else:
        version_txt = " model version unknown"

    return pn.Card(
        pn.pane.Markdown(
            f"**Structure** &nbsp; "
            f"<span style='font-size:12px; color:#666;'>"
            f"AlphaFold: {html.escape(uniprot)}{html.escape(version_txt)}"
            f"</span>",
            margin=(0, 0, 4, 0),
        ),
        pn.pane.HTML(
            iframe,
            width=width - 20,
            height=height - 50,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        ),
        width=width,
        height=height,
        collapsible=False,
        hide_header=True,
        styles={
            "background": "#f9f9f9",
            "border-radius": "8px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "padding": "8px",
        },
    )
