from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd
import panel as pn

from session_state import SessionState


@dataclass(frozen=True)
class SelectionExportSpec:
    """
    Strict schema for volcano selection export.

    FEATURE_ID: adata.var_names (string)
    UNIPROT_ID: for proteomics -> FEATURE_ID; for others -> adata.var['UNIPROT']
    GENE_NAME : adata.var['GENE_NAMES']
    log2FC/p/q: contrast-specific from adata.varm
    RAW__/PROC__ per sample: from adata.layers['raw'] and adata.X
    AVG_IBAQ optional: from adata.layers['ibaq'] if present
    """
    filename: str = "proteoflux_selection.csv"
    label: str = "Download selection"
    uniprot_var_col: str = "UNIPROT"
    id_col_name: str = "UNIPROT_ID"


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)

def _contrast_tag(contrast: str) -> str:
    """
    Canonicalize a contrast name into an ALL_CAPS tag suitable for column names.
    Example: "WT vs KO" -> "WT_VS_KO"
    """
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(contrast)).strip("_").upper()
    if not tag:
        raise ValueError(f"Invalid contrast name for export: {contrast!r}")
    return tag

def extract_feature_ids_from_selected_data(selected_data: dict) -> list[str]:
    """
    Plotly selection payload -> list of feature IDs.

    Requires: each selected point has customdata[0] == feature_id.
    """
    if not selected_data:
        return []

    pts = selected_data.get("points")
    if not isinstance(pts, list):
        raise ValueError(
            "Volcano selection export: invalid selected_data payload. "
            "Expected dict with key 'points' as a list."
        )

    out: list[str] = []
    for pt in pts:
        if not isinstance(pt, dict):
            raise ValueError(
                "Volcano selection export: invalid point payload type. "
                f"Expected dict, got {type(pt).__name__}."
            )
        cd = pt.get("customdata")
        if not isinstance(cd, (list, tuple)) or len(cd) < 1:
            raise ValueError(
                "Volcano selection export requires plot points to carry customdata[0] = feature_id. "
                "Got missing/invalid customdata in a selected point."
            )
        out.append(str(cd[0]))

    # preserve order but drop duplicates
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped

def extract_feature_ids_from_click_data(click_data: dict) -> list[str]:
    """
    Plotly click payload -> list[str] of feature IDs (single-point).

    Requires: click_data['points'][0]['customdata'][0] == feature_id
    """
    if not click_data:
        return []

    pts = click_data.get("points")
    if not isinstance(pts, list) or not pts:
        raise ValueError(
            "Volcano click export: invalid click_data payload. "
            "Expected dict with key 'points' as a non-empty list."
        )

    pt = pts[0]
    if not isinstance(pt, dict):
        raise ValueError(
            "Volcano click export: invalid point payload type. "
            f"Expected dict, got {type(pt).__name__}."
        )

    cd = pt.get("customdata")
    if not isinstance(cd, (list, tuple)) or len(cd) < 1:
        raise ValueError(
            "Volcano click export requires plot points to carry customdata[0] = feature_id. "
            "Got missing/invalid customdata in clicked point."
        )

    return [str(cd[0])]


def build_volcano_selection_df(
    state: SessionState,
    contrast: str,
    feature_ids: list[str],
    *,
    uniprot_var_col: str = "UNIPROT",
    id_col_name: str = "UNIPROT_ID",
) -> pd.DataFrame:
    """
    Build the CSV export table for a volcano selection (non-phospho overview).

    Fail-fast:
    - requires adata.varm: log2fc, p_ebayes, q_ebayes
    - requires adata.layers['raw']
    - requires adata.var['GENE_NAMES']
    - requires adata.var['UNIPROT'] if not proteomics mode
    - requires all feature_ids to exist in adata.var_names
    """
    _require(bool(feature_ids), "No volcano datapoints selected.")

    adata = state.adata
    contrasts = list(map(str, adata.uns.get("contrast_names", [])))
    _require(
        contrast in set(contrasts),
        f"Unknown contrast {contrast!r}. Available={contrasts!r}",
    )

    required_varm = {"log2fc", "p_ebayes", "q_ebayes"}
    missing_varm = sorted(required_varm - set(adata.varm.keys()))
    _require(
        not missing_varm,
        "Cannot export volcano selection: required adata.varm entries are missing. "
        f"Missing={missing_varm}. Present={sorted(adata.varm.keys())}",
    )

    _require(
        "raw" in adata.layers,
        "Cannot export raw intensities: missing adata.layers['raw']. "
        f"Available layers={list(adata.layers.keys())!r}",
    )

    _require(
        "GENE_NAMES" in adata.var.columns,
        "Cannot export gene names: missing adata.var['GENE_NAMES']. "
        f"Available var columns={list(adata.var.columns)!r}",
    )

    mode = str(adata.uns.get("preprocessing", {}).get("analysis_type", "")).lower()
    proteomics_mode = (mode in {"dia", "dda", "proteomics"})
    peptido_mode = (mode in {"peptido", "peptidomics"})

    # Contrast-specific stats tables
    df_fc = pd.DataFrame(adata.varm["log2fc"], index=adata.var_names, columns=contrasts)
    df_p = pd.DataFrame(adata.varm["p_ebayes"], index=adata.var_names, columns=contrasts)
    df_q = pd.DataFrame(adata.varm["q_ebayes"], index=adata.var_names, columns=contrasts)

    gene_names = adata.var["GENE_NAMES"].astype(str)
    fasta_headers = adata.var["FASTA_HEADER"].astype(str) if "FASTA_HEADER" in adata.var.columns else None

    # Feature id is adata.var_names:
    # - proteomics: protein id (UniProt-like)
    # - peptido:    peptide sequence (possibly modified)
    feature_id = pd.Series(adata.var_names.astype(str), index=adata.var_names.astype(str))

    # Materialize matrices
    mat_proc = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    mat_raw = adata.layers["raw"].toarray() if hasattr(adata.layers["raw"], "toarray") else adata.layers["raw"]

    # Validate feature ids exist and map to indices
    var_index = {str(v): i for i, v in enumerate(map(str, adata.var_names))}
    idx: list[int] = []
    for fid in feature_ids:
        if str(fid) not in var_index:
            raise ValueError(
                "Cannot export volcano selection: selected feature id not found in adata.var_names. "
                f"Missing={fid!r}"
            )
        idx.append(var_index[str(fid)])

    # Assemble export
    df = pd.DataFrame(index=[str(fid) for fid in feature_ids])

    df[id_col_name] = feature_id.reindex(df.index).values

    if peptido_mode:
        _require(
            "UNIPROT" in adata.var.columns,
            "Cannot export peptido selection: missing adata.var['UNIPROT'] "
            "(required to populate UNIPROT_ID as parent protein). "
            f"Available var columns={list(adata.var.columns)!r}",
        )
        df.insert(
            0, "PEPTIDE", feature_id.reindex(df.index).values
        )
        df[id_col_name] = (
            adata.var["UNIPROT"].astype(str).reindex(df.index).values
        )
    else:
        df[id_col_name] = feature_id.reindex(df.index).values

    df["GENE_NAME"] = gene_names.reindex(df.index).values

    # Optional FASTA headers (present for proteomics; absent for peptido)
    if "FASTA_HEADERS" in adata.var.columns:
        df["FASTA_HEADERS"] = adata.var["FASTA_HEADERS"].astype(str).reindex(df.index).values

    # Contrast-specific stats (ALL CAPS + contrast name)
    ctag = str(contrast).upper()
    df[f"LOG2FC_{ctag}"] = df_fc.loc[df.index, contrast].astype(float).values
    df[f"PVALUE_{ctag}"] = df_p.loc[df.index, contrast].astype(float).values
    df[f"QVALUE_{ctag}"] = df_q.loc[df.index, contrast].astype(float).values

    # Intensities: all processed first, then all raw
    samples = list(map(str, adata.obs_names))
    for j, s in enumerate(samples):
        df[f"PROCESSED_INTENSITIES_{s}"] = mat_proc[j, idx]
    for j, s in enumerate(samples):
        df[f"RAW_INTENSITIES_{s}"] = mat_raw[j, idx]

    return df


def make_volcano_selection_downloader(
    *,
    state: SessionState,
    contrast_getter: Callable[[], str],
    spec: SelectionExportSpec = SelectionExportSpec(),
#) -> tuple[pn.widgets.FileDownload, Callable[[dict], None]]:
) -> tuple[
    pn.widgets.FileDownload,
    Callable[[dict], None],
    Callable[[dict], None],
    Callable[[list[str]], None],
]:
    """
    Returns:
      - FileDownload widget (initially hidden)
      - on_selected_data callback (lasso/box) -> wire to volcano_plot.selected_data
      - on_click_data callback (single datapoint) -> wire to volcano_plot.click_data
      - on_cohort_ids callback (pattern/file cohort) -> call when cohort changes

    Priority (non-negotiable):
      click datapoint > cohort > lasso/box selection
    """
    # Track sources independently so we can fall back deterministically.
    click_ids: list[str] = []
    cohort_ids: list[str] = []
    lasso_ids: list[str] = []
    effective_ids: list[str] = []

    def _set_loading_pulse() -> None:
        # show a brief loading pulse when the effective source changes.
        if not hasattr(download, "loading"):
            return
        download.loading = True
        doc = pn.state.curdoc
        if doc is None:
            download.loading = False
            return
        doc.add_next_tick_callback(lambda: setattr(download, "loading", False))


    def _csv_callback() -> bytes:
        df = build_volcano_selection_df(
            state=state,
            contrast=str(contrast_getter()),
            feature_ids=effective_ids,
            uniprot_var_col=spec.uniprot_var_col,
            id_col_name=spec.id_col_name,
        )

        # Panel FileDownload expects a file-like object or a filesystem path.
        data = df.to_csv(index=False).encode("utf-8")
        return io.BytesIO(data)

    download = pn.widgets.FileDownload(
        label=spec.label,
        callback=_csv_callback,
        filename=spec.filename,
        button_type="success",
        visible=False,
        margin=(20, 0, 0, 0),
    )

    def _recompute_effective_ids(*, pulse: bool = True) -> None:
        # Deterministic priority: click > cohort > lasso
        if click_ids:
            new_ids = list(click_ids)
        elif cohort_ids:
            new_ids = list(cohort_ids)
        else:
            new_ids = list(lasso_ids)

        changed = (new_ids != effective_ids)
        effective_ids.clear()
        effective_ids.extend(new_ids)
        download.visible = bool(effective_ids)

        if pulse and changed:
            _set_loading_pulse()

    def _on_selected_data(selected_data: dict) -> None:
        if selected_data is None:
            return
        if selected_data == {}:
            return
        if "points" in selected_data and not selected_data["points"]:
            # Panel/Plotly sometimes emits {"selector": None, "points": []} on re-render
            if selected_data.get("selector", "__missing__") is None:
                return

            lasso_ids.clear()
            _recompute_effective_ids()
            return

        lasso_ids.clear()
        lasso_ids.extend(extract_feature_ids_from_selected_data(selected_data))
        _recompute_effective_ids()

    def _on_click_data(click_data: dict) -> None:
        # single datapoint click (highest priority)
        click_ids.clear()
        click_ids.extend(extract_feature_ids_from_click_data(click_data or {}))
        _recompute_effective_ids()

    def _on_cohort_ids(ids: list[str]) -> None:
        # cohort (pattern/file). If empty -> it must relinquish control immediately.
        cohort_ids.clear()
        cohort_ids.extend([str(x) for x in (ids or [])])
        _recompute_effective_ids()

    return download, _on_selected_data, _on_click_data, _on_cohort_ids

def make_adjacent_sites_csv_callback(
    *,
    state,
    contrast_getter,
    siblings_getter,
):
    """
    Factory for Adjacent Sites CSV export.
    Explicitly captures all required state via closures.
    """

    def _adjacent_sites_csv() -> bytes:
        feature_ids = [str(x) for x in siblings_getter()]
        if not feature_ids:
            raise ValueError("No adjacent sites to export.")

        df = build_volcano_selection_df(
            state=state,
            contrast=str(contrast_getter()),
            feature_ids=feature_ids,
            uniprot_var_col="PARENT_PROTEIN",
            id_col_name="PHOSPHOSITE_ID",
        )
        return io.BytesIO(df.to_csv(index=False).encode("utf-8"))

    return _adjacent_sites_csv

