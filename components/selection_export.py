from __future__ import annotations

import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd
import panel as pn

from utils.session_state import SessionState


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
    phospho_mode = (mode == "phospho")

    # Contrast-specific stats tables
    df_fc_adj = pd.DataFrame(adata.varm["log2fc"], index=adata.var_names, columns=contrasts)
    df_p_adj = pd.DataFrame(adata.varm["p_ebayes"], index=adata.var_names, columns=contrasts)
    df_q_adj = pd.DataFrame(adata.varm["q_ebayes"], index=adata.var_names, columns=contrasts)
    df_nrsc = pd.DataFrame(adata.varm["nrsc"], index=adata.var_names, columns=contrasts) if "nrsc" in adata.varm else None
    df_nrsc_alignment = (
        pd.DataFrame(adata.varm["nrsc_misalignment"], index=adata.var_names, columns=contrasts)
        if "nrsc_misalignment" in adata.varm else None
    )

    # Optional: phospho raw statistics (falls back to adjusted for non-covariate runs)
    df_fc_raw = df_p_raw = df_q_raw = None
    if phospho_mode:
        if "raw_log2fc" in adata.varm:
            df_fc_raw = pd.DataFrame(adata.varm["raw_log2fc"], index=adata.var_names, columns=contrasts)
        else:
            df_fc_raw = df_fc_adj

        if "raw_p_ebayes" in adata.varm:
            df_p_raw = pd.DataFrame(adata.varm["raw_p_ebayes"], index=adata.var_names, columns=contrasts)
        else:
            df_p_raw = df_p_adj

        if "raw_q_ebayes" in adata.varm:
            df_q_raw = pd.DataFrame(adata.varm["raw_q_ebayes"], index=adata.var_names, columns=contrasts)
        else:
            df_q_raw = df_q_adj

    # Optional: phospho covariate-part + flowthrough stats
    df_cov_part = None
    if phospho_mode and ("cov_part" in adata.varm):
        df_cov_part = pd.DataFrame(adata.varm["cov_part"], index=adata.var_names, columns=contrasts)

    df_ft_fc = df_ft_p = df_ft_q = None
    if phospho_mode:
        if "ft_log2fc" in adata.varm:
            df_ft_fc = pd.DataFrame(adata.varm["ft_log2fc"], index=adata.var_names, columns=contrasts)
        if "ft_p_ebayes" in adata.varm:
            df_ft_p = pd.DataFrame(adata.varm["ft_p_ebayes"], index=adata.var_names, columns=contrasts)
        if "ft_q_ebayes" in adata.varm:
            df_ft_q = pd.DataFrame(adata.varm["ft_q_ebayes"], index=adata.var_names, columns=contrasts)

    gene_names = adata.var["GENE_NAMES"].astype(str)
    fasta_headers = adata.var["FASTA_HEADER"].astype(str) if "FASTA_HEADER" in adata.var.columns else None

    # Feature id is adata.var_names:
    # - proteomics: protein id (UniProt-like)
    # - peptido:    peptide sequence (possibly modified)
    feature_id = pd.Series(adata.var_names.astype(str), index=adata.var_names.astype(str))

    # Materialize matrices
    mat_proc = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    mat_raw = adata.layers["raw"].toarray() if hasattr(adata.layers["raw"], "toarray") else adata.layers["raw"]
    samples = list(map(str, adata.obs_names))

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
    if phospho_mode:
        _require(df_fc_raw is not None and df_p_raw is not None and df_q_raw is not None,
                 "Phospho selection export: internal error; raw stats tables not materialized.")

        df[f"RAW_LOG2FC_{ctag}"] = df_fc_raw.loc[df.index, contrast].astype(float).values
        df[f"RAW_PVALUE_{ctag}"] = df_p_raw.loc[df.index, contrast].astype(float).values
        df[f"RAW_QVALUE_{ctag}"] = df_q_raw.loc[df.index, contrast].astype(float).values

        df[f"ADJUSTED_LOG2FC_{ctag}"] = df_fc_adj.loc[df.index, contrast].astype(float).values
        df[f"ADJUSTED_PVALUE_{ctag}"] = df_p_adj.loc[df.index, contrast].astype(float).values
        df[f"ADJUSTED_QVALUE_{ctag}"] = df_q_adj.loc[df.index, contrast].astype(float).values

        if df_cov_part is not None:
            df[f"COVARIATE_PART_{ctag}"] = df_cov_part.loc[df.index, contrast].astype(float).values

        if (df_ft_fc is not None) and (df_ft_p is not None) and (df_ft_q is not None):
            df[f"FT_LOG2FC_{ctag}"] = df_ft_fc.loc[df.index, contrast].astype(float).values
            df[f"FT_PVALUE_{ctag}"] = df_ft_p.loc[df.index, contrast].astype(float).values
            df[f"FT_QVALUE_{ctag}"] = df_ft_q.loc[df.index, contrast].astype(float).values
    else:
        df[f"LOG2FC_{ctag}"] = df_fc_adj.loc[df.index, contrast].astype(float).values
        df[f"PVALUE_{ctag}"] = df_p_adj.loc[df.index, contrast].astype(float).values
        df[f"QVALUE_{ctag}"] = df_q_adj.loc[df.index, contrast].astype(float).values
        if df_nrsc is not None:
            df[f"NRSC_{ctag}"] = df_nrsc.loc[df.index, contrast].astype(float).values
        if df_nrsc_alignment is not None:
            df[f"NRSC_ALIGNMENT_{ctag}"] = df_nrsc_alignment.loc[df.index, contrast].astype(float).values

    # Optional: phospho flowthrough intensities (covariate matrices)
    if phospho_mode and ("processed_covariate" in adata.layers) and ("raw_covariate" in adata.layers):
        mat_ft_proc = (
            adata.layers["processed_covariate"].toarray()
            if hasattr(adata.layers["processed_covariate"], "toarray")
            else adata.layers["processed_covariate"]
        )
        mat_ft_raw = (
            adata.layers["raw_covariate"].toarray()
            if hasattr(adata.layers["raw_covariate"], "toarray")
            else adata.layers["raw_covariate"]
        )
        for j, s in enumerate(samples):
            df[f"FT_PROCESSED_INTENSITIES_{s}"] = mat_ft_proc[j, idx]
        for j, s in enumerate(samples):
            df[f"FT_RAW_INTENSITIES_{s}"] = mat_ft_raw[j, idx]

    # Intensities: all processed first, then all raw
    for j, s in enumerate(samples):
        df[f"PROCESSED_INTENSITIES_{s}"] = mat_proc[j, idx]
    for j, s in enumerate(samples):
        df[f"RAW_INTENSITIES_{s}"] = mat_raw[j, idx]

    return df


def build_kinase_selection_df(
    state: SessionState,
    contrast: str,
    feature_ids: list[str],
    *,
    sign_threshold: float = 0.05,
    uniprot_var_col: str = "KINASE_UNIPROT",
    id_col_name: str = "KINASE_ID",
) -> pd.DataFrame:
    """Build a contrast-local KSEA volcano selection export."""
    _require(bool(feature_ids), "No kinase volcano datapoints selected.")

    payload = state.adata.uns.get("kinase_activity")
    _require(
        isinstance(payload, Mapping),
        "Cannot export kinase selection: missing adata.uns['kinase_activity'].",
    )
    results = payload.get("results")
    _require(
        isinstance(results, pd.DataFrame),
        "Cannot export kinase selection: kinase activity results are missing.",
    )

    required = {
        "contrast",
        "kinase_id",
        "kinase",
        "kinase_gene",
        "kinase_uniprot",
        "n_substrates",
        "kinase_mean_log2fc",
        "global_mean_log2fc",
        "effect",
        "activity_score",
        "pvalue",
        "qvalue",
        "tested",
    }
    missing = sorted(required.difference(results.columns))
    _require(
        not missing,
        "Cannot export kinase selection: required result columns are missing. "
        f"Missing={missing!r}.",
    )

    sub = results.loc[
        results["contrast"].astype(str).eq(str(contrast))
    ].copy()
    _require(
        not sub.empty,
        f"Cannot export kinase selection: contrast {contrast!r} was not found.",
    )
    sub["kinase_id"] = sub["kinase_id"].astype(str).str.strip()
    _require(
        not sub["kinase_id"].duplicated().any(),
        "Cannot export kinase selection: kinase_id is not unique within "
        f"contrast {contrast!r}.",
    )
    sub = sub.set_index("kinase_id", drop=False)

    selected = list(dict.fromkeys(str(value) for value in feature_ids))
    missing_ids = [value for value in selected if value not in sub.index]
    _require(
        not missing_ids,
        "Cannot export kinase selection: selected kinase IDs were not found "
        f"in contrast {contrast!r}. Missing={missing_ids!r}.",
    )
    sub = sub.loc[selected].copy()

    if pd.api.types.is_bool_dtype(sub["tested"].dtype):
        tested = sub["tested"].fillna(False).astype(bool)
    else:
        tested = (
            sub["tested"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )

    qvalue = pd.to_numeric(sub["qvalue"], errors="coerce")
    activity_score = pd.to_numeric(sub["activity_score"], errors="coerce")
    invalid_identifiers = {"", "?", "nan", "none", "n/a", "na"}

    def _string_input(row: pd.Series) -> str:
        for column in (
            "kinase_uniprot",
            "kinase_gene",
            "kinase",
            "kinase_id",
        ):
            value = row.get(column, "")
            text = "" if pd.isna(value) else str(value).strip()
            if text.casefold() not in invalid_identifiers:
                return text
        return ""

    string_input = sub.apply(_string_input, axis=1)
    significant = (
        tested
        & np.isfinite(qvalue)
        & qvalue.lt(float(sign_threshold))
    )
    string_background = tested & string_input.ne("")
    string_query_activated = (
        significant & string_input.ne("") & activity_score.gt(0.0)
    )
    string_query_inhibited = (
        significant & string_input.ne("") & activity_score.lt(0.0)
    )
    string_query = string_query_activated | string_query_inhibited
    direction = pd.Series("not significant", index=sub.index, dtype=object)
    direction.loc[significant] = "no signed score"
    direction.loc[significant & activity_score.gt(0.0)] = "activated"
    direction.loc[significant & activity_score.lt(0.0)] = "inhibited"
    direction.loc[significant & activity_score.eq(0.0)] = "zero score"

    sources = pd.Series("", index=sub.index, dtype=object)
    relationships = payload.get("substrates")
    if isinstance(relationships, pd.DataFrame) and {
        "contrast",
        "kinase_id",
        "database_source",
    }.issubset(relationships.columns):
        rel = relationships.loc[
            relationships["contrast"].astype(str).eq(str(contrast))
        ].copy()
        rel["kinase_id"] = rel["kinase_id"].astype(str).str.strip()
        source_map = rel.groupby("kinase_id", sort=False)["database_source"].agg(
            lambda values: "; ".join(
                dict.fromkeys(
                    text
                    for value in values
                    if (text := str(value).strip())
                    and text.casefold() not in {"nan", "none"}
                )
            )
        )
        sources = source_map.reindex(sub.index).fillna("")

    export = pd.DataFrame(index=sub.index)
    export[id_col_name] = sub["kinase_id"].astype(str)
    export["KINASE"] = sub["kinase"].astype(str)
    export["KINASE_GENE"] = sub["kinase_gene"].astype(str)
    export[uniprot_var_col] = sub["kinase_uniprot"].astype(str)
    export["CONTRAST"] = str(contrast)
    export["TESTED"] = tested.to_numpy(dtype=bool)
    export["SIGNIFICANT"] = significant.to_numpy(dtype=bool)
    export["SIGN_THRESHOLD"] = float(sign_threshold)
    export["KSEA_DIRECTION"] = direction.astype(str)
    export["STRING_INPUT_IDENTIFIER"] = string_input.astype(str)
    export["STRING_QUERY_INPUT"] = string_query.to_numpy(dtype=bool)
    export["STRING_QUERY_ACTIVATED_INPUT"] = (
        string_query_activated.to_numpy(dtype=bool)
    )
    export["STRING_QUERY_INHIBITED_INPUT"] = (
        string_query_inhibited.to_numpy(dtype=bool)
    )
    export["STRING_BACKGROUND_INPUT"] = string_background.to_numpy(dtype=bool)
    export["N_SUBSTRATES"] = pd.to_numeric(
        sub["n_substrates"], errors="coerce"
    )
    export["MEAN_SUBSTRATE_LOG2FC"] = pd.to_numeric(
        sub["kinase_mean_log2fc"], errors="coerce"
    )
    export["GLOBAL_MEAN_LOG2FC"] = pd.to_numeric(
        sub["global_mean_log2fc"], errors="coerce"
    )
    export["MEAN_SHIFT"] = pd.to_numeric(sub["effect"], errors="coerce")
    export["ACTIVITY_Z_SCORE"] = activity_score
    export["PVALUE"] = pd.to_numeric(sub["pvalue"], errors="coerce")
    export["QVALUE"] = qvalue
    export["DATABASE_SOURCE"] = sources.astype(str)
    return export.reset_index(drop=True)


def build_pelsa_selection_df(
    state: SessionState,
    contrast: str,
    feature_ids: list[str],
    *,
    uniprot_var_col: str = "PARENT_PROTEIN",
    id_col_name: str = "PEPTIDE",
) -> pd.DataFrame:
    """
    Build the CSV export table for a PELSA pseudo-volcano selection.

    PELSA is peptide-centric and does not have contrast-specific limma varm
    statistics. Selection export therefore uses adata.uns['pelsa']['curve_results']
    plus peptide/protein metadata from adata.var.
    """
    _require(bool(feature_ids), "No PELSA datapoints selected.")

    adata = state.adata
    mode = str(adata.uns.get("preprocessing", {}).get("analysis_type", "")).lower()
    _require(
        mode == "pelsa",
        f"PELSA selection export requires analysis_type='pelsa'. Got {mode!r}."
    )

    pelsa = adata.uns.get("pelsa", {})
    _require(
        "curve_results" in pelsa,
        "Cannot export PELSA selection: missing adata.uns['pelsa']['curve_results']."
    )

    curve_results = pd.DataFrame(pelsa["curve_results"]).copy()
    required_curve_cols = {
        "peptide_id",
        "curve_fold_change_log2",
        "curve_p_value",
        "curve_q_value",
        "pec50",
    }
    missing_curve_cols = sorted(required_curve_cols - set(curve_results.columns))
    _require(
        not missing_curve_cols,
        "Cannot export PELSA selection: required curve result columns are missing. "
        f"Missing={missing_curve_cols}. Present={list(curve_results.columns)!r}"
    )

    required_var_cols = {uniprot_var_col, "GENE_NAMES"}
    missing_var_cols = sorted(required_var_cols - set(adata.var.columns))
    _require(
        not missing_var_cols,
        "Cannot export PELSA selection: required adata.var columns are missing. "
        f"Missing={missing_var_cols}. Present={list(adata.var.columns)!r}"
    )

    curve_results["peptide_id"] = curve_results["peptide_id"].astype(str)
    curve_results = curve_results.set_index("peptide_id", drop=False)

    var_index = set(map(str, adata.var_names))
    result_index = set(curve_results.index.astype(str))

    missing_var = [str(fid) for fid in feature_ids if str(fid) not in var_index]
    _require(
        not missing_var,
        "Cannot export PELSA selection: selected peptide id not found in adata.var_names. "
        f"Missing={missing_var[:10]!r}"
    )

    missing_results = [str(fid) for fid in feature_ids if str(fid) not in result_index]
    _require(
        not missing_results,
        "Cannot export PELSA selection: selected peptide id not found in curve_results. "
        f"Missing={missing_results[:10]!r}"
    )

    ids = [str(fid) for fid in feature_ids]
    res = curve_results.loc[ids]
    var = adata.var.reindex(ids)

    df = pd.DataFrame(index=ids)
    df[id_col_name] = ids
    df["UNIPROT_ID"] = var[uniprot_var_col].astype(str).values
    df["GENE_NAME"] = var["GENE_NAMES"].astype(str).values

    df["RANGE_LOG2"] = pd.to_numeric(res["curve_fold_change_log2"], errors="raise").astype(float).values
    df["PVALUE"] = pd.to_numeric(res["curve_p_value"], errors="raise").astype(float).values
    df["QVALUE"] = pd.to_numeric(res["curve_q_value"], errors="raise").astype(float).values
    pec50 = pd.to_numeric(res["pec50"], errors="raise").astype(float)
    df["LOG10_EC50"] = (-pec50).values

    return df.reset_index(drop=True)



def make_volcano_selection_downloader(
    *,
    state: SessionState,
    contrast_getter: Callable[[], str],
    spec: SelectionExportSpec = SelectionExportSpec(),
    selection_df_builder: Optional[Callable[..., pd.DataFrame]] = None,
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
        builder = selection_df_builder or build_volcano_selection_df
        df = builder(
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
        if selected_data is None or selected_data == {}:
            lasso_ids.clear()
            _recompute_effective_ids()
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
