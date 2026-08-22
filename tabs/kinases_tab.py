from __future__ import annotations

from collections.abc import Mapping
from html import escape
from itertools import combinations
from textwrap import wrap
import time

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import requests

import scipy.cluster.hierarchy as sch
from plotly.subplots import make_subplots

from components.selection_export import (
    SelectionExportSpec,
    build_kinase_selection_df,
    make_volcano_selection_downloader,
)
from components.string_links import (
    STRING_API_URL,
    STRING_CALLER_IDENTITY,
    get_string_link,
)
from tabs.overview_shared import STRING_SPECIES_OPTIONS, bind_uirevision
from utils.layout_utils import (
    FRAME_STYLES_TALL,
    make_vr,
    make_hr,
    )
from utils.session_state import SessionState
from components.plot_utils import get_color_map
from utils.utils import log_time


_REQUIRED_RESULT_COLUMNS = {
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

_REQUIRED_SUBSTRATE_COLUMNS = {
    "contrast",
    "kinase_id",
    "phosphosite_id",
    "matched_accession",
    "database_source",
}

_REQUIRED_CONDITION_KINASE_COLUMNS = {
    "condition",
    "kinase_id",
    "n_substrates",
}

_REQUIRED_DATABASE_SUMMARY_COLUMNS = {
    "filename",
    "new_relationships",
    "new_matched_relationships",
    "new_matched_phosphosites",
    "new_matched_kinases",
}

_UPSET_MAX_SUBSETS = 25

_ENRICHMENT_CATEGORY_OPTIONS = {
    "KEGG pathways": "KEGG",
    "Reactome pathways": "RCTM",
    "GO Biological Process": "Process",
}
_ENRICHMENT_METRIC_OPTIONS = {
    "Signal": "signal",
    "−log10(FDR)": "fdr",
    "Contributing kinases": "number_of_genes",
}
_ENRICHMENT_MAX_TERMS = 12
_ENRICHMENT_LABEL_WRAP_WIDTH = 44
_ENRICHMENT_MAX_CONTRASTS = 25


# False uses STRING's default whole-species background. Set to True to
# restrict enrichment to the kinases that were eligible/tested by KSEA.
# Still testing, restricting seems to always break all stat. power
_ENRICHMENT_USE_TESTED_BACKGROUND = False

def _kinase_activity(adata) -> Mapping:
    payload = adata.uns.get("kinase_activity")
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Kinase activity results are not available in "
            "adata.uns['kinase_activity']."
        )
    return payload


def _kinase_results(adata) -> pd.DataFrame:
    results = _kinase_activity(adata).get("results")
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "adata.uns['kinase_activity']['results'] must be a pandas DataFrame."
        )

    missing = sorted(_REQUIRED_RESULT_COLUMNS.difference(results.columns))
    if missing:
        raise ValueError(
            "Kinase activity results are missing required columns: "
            f"{missing!r}."
        )
    return results


def _kinase_substrates(adata) -> pd.DataFrame:
    substrates = _kinase_activity(adata).get("substrates")
    if not isinstance(substrates, pd.DataFrame):
        raise TypeError(
            "adata.uns['kinase_activity']['substrates'] must be a pandas DataFrame."
        )

    missing = sorted(_REQUIRED_SUBSTRATE_COLUMNS.difference(substrates.columns))
    if missing:
        raise ValueError(
            "Kinase substrate links are missing required columns: "
            f"{missing!r}."
        )
    return substrates


def _condition_kinases(adata) -> pd.DataFrame:
    condition_kinases = _kinase_activity(adata).get("condition_kinases")
    if condition_kinases is None:
        return pd.DataFrame(columns=sorted(_REQUIRED_CONDITION_KINASE_COLUMNS))
    if not isinstance(condition_kinases, pd.DataFrame):
        raise TypeError(
            "adata.uns['kinase_activity']['condition_kinases'] must be a "
            "pandas DataFrame."
        )

    missing = sorted(
        _REQUIRED_CONDITION_KINASE_COLUMNS.difference(
            condition_kinases.columns
        )
    )
    if missing:
        raise ValueError(
            "Condition-level kinase membership is missing required columns: "
            f"{missing!r}."
        )
    return condition_kinases


def _database_summary(adata) -> pd.DataFrame:
    summary = _kinase_activity(adata).get("database_summary")
    if summary is None:
        return pd.DataFrame(
            columns=sorted(_REQUIRED_DATABASE_SUMMARY_COLUMNS)
        )
    if not isinstance(summary, pd.DataFrame):
        raise TypeError(
            "adata.uns['kinase_activity']['database_summary'] must be a "
            "pandas DataFrame."
        )

    missing = sorted(
        _REQUIRED_DATABASE_SUMMARY_COLUMNS.difference(summary.columns)
    )
    if missing:
        raise ValueError(
            "KSEA database summary is missing required columns: "
            f"{missing!r}."
        )
    return summary


def _tested_mask(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
        .to_numpy(dtype=bool)
    )


def _text_series(values: pd.Series) -> pd.Series:
    """Normalize object, categorical, and H5AD-round-tripped text safely."""
    values = values.astype(object)
    return values.where(pd.notna(values), "").astype(str).str.strip()

def _text_value(value, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text or fallback


def _count_text(value, fallback: str = "Not recorded") -> str:
    try:
        if pd.isna(value):
            return fallback
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return fallback


def _qvalue_plot_values(qvalues: np.ndarray) -> np.ndarray:
    """Return finite -log10(q) values while retaining underflowed q=0 rows."""
    qvalues = np.asarray(qvalues, dtype=float)
    positive = qvalues[np.isfinite(qvalues) & (qvalues > 0.0)]
    if positive.size:
        max_score = float(np.nanmax(-np.log10(positive)))
        zero_score = min(300.0, max(1.0, np.ceil(max_score) + 1.0))
    else:
        zero_score = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = -np.log10(qvalues)
    scores[qvalues == 0.0] = zero_score
    return scores


def _significance_symbols(
    qvalues: np.ndarray,
    sign_threshold: float,
) -> np.ndarray:
    """Return q-value stars, limited by the configured significance cutoff."""
    qvalues = np.asarray(qvalues, dtype=float)
    symbols = np.full(qvalues.shape, "", dtype=object)
    significant = np.isfinite(qvalues) & (qvalues < float(sign_threshold))
    symbols[significant] = "*"
    symbols[significant & (qvalues < 0.01)] = "**"
    symbols[significant & (qvalues < 0.001)] = "***"
    return symbols


def _significance_labels(
    qvalues: np.ndarray,
    symbols: np.ndarray,
) -> np.ndarray:
    qvalues = np.asarray(qvalues, dtype=float)
    symbols = np.asarray(symbols, dtype=object)
    labels = np.full(qvalues.shape, "not tested", dtype=object)
    tested = np.isfinite(qvalues)
    labels[tested] = "not significant"
    labels[tested & (symbols != "")] = symbols[tested & (symbols != "")]
    return labels


def _string_identifier(row: pd.Series) -> str:
    invalid = {"", "?", "nan", "none", "n/a", "na"}
    for column in (
        "kinase_uniprot",
        "kinase_gene",
        "kinase",
        "kinase_id",
    ):
        value = _text_value(row.get(column, ""))
        if value.casefold() not in invalid:
            return value
    return ""


def _significant_kinases_by_contrast(
    results: pd.DataFrame,
    contrasts: list[str],
    sign_threshold: float,
) -> dict[str, dict[str, tuple[str, ...]]]:
    work = results.copy()
    work["contrast"] = _text_series(work["contrast"])
    work["qvalue"] = pd.to_numeric(work["qvalue"], errors="coerce")
    work["activity_score"] = pd.to_numeric(
        work["activity_score"], errors="coerce"
    )
    work["_tested"] = _tested_mask(work["tested"])

    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for contrast in contrasts:
        sub = work.loc[
            work["contrast"].eq(str(contrast))
            & work["_tested"]
            & work["qvalue"].lt(float(sign_threshold))
        ]
        output[str(contrast)] = {
            direction: tuple(
                sorted(
                    {
                        identifier
                        for _, row in sub.loc[mask].iterrows()
                        if (identifier := _string_identifier(row))
                    },
                    key=str.casefold,
                )
            )
            for direction, mask in {
                "activated": sub["activity_score"].gt(0.0),
                "inhibited": sub["activity_score"].lt(0.0),
            }.items()
        }
    return output


def _tested_kinases_by_contrast(
    results: pd.DataFrame,
    contrasts: list[str],
) -> dict[str, tuple[str, ...]]:
    work = results.copy()
    work["contrast"] = _text_series(work["contrast"])
    work["_tested"] = _tested_mask(work["tested"])

    output: dict[str, tuple[str, ...]] = {}
    for contrast in contrasts:
        sub = work.loc[
            work["contrast"].eq(str(contrast)) & work["_tested"]
        ]
        identifiers = {
            identifier
            for _, row in sub.iterrows()
            if (identifier := _string_identifier(row))
        }
        output[str(contrast)] = tuple(
            sorted(identifiers, key=str.casefold)
        )
    return output


def _string_post_json(method: str, parameters: dict) -> list[dict]:
    response = requests.post(
        f"{STRING_API_URL}/json/{method}",
        data=parameters,
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise TypeError(
            f"STRING {method!r} response must be a list of records."
        )
    return payload


def _map_to_string_ids(
    identifiers: tuple[str, ...],
    species: int,
) -> dict[str, str]:
    ordered = tuple(dict.fromkeys(str(value) for value in identifiers if value))
    if not ordered:
        return {}

    payload = _string_post_json(
        "get_string_ids",
        {
            "identifiers": "\r".join(ordered),
            "species": int(species),
            "echo_query": 1,
            "caller_identity": STRING_CALLER_IDENTITY,
        },
    )
    mapping: dict[str, str] = {}
    for record in payload:
        try:
            query_index = int(record["queryIndex"])
            string_id = str(record["stringId"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= query_index < len(ordered) and string_id:
            mapping[ordered[query_index]] = string_id
    return mapping


def _string_enrichment(
    query_string_ids: tuple[str, ...],
    background_string_ids: tuple[str, ...],
    species: int,
    use_tested_background: bool,
) -> list[dict]:
    if len(query_string_ids) < 2:
        return []
    parameters = {
        "identifiers": "\r".join(query_string_ids),
        "species": int(species),
        "caller_identity": STRING_CALLER_IDENTITY,
    }
    if use_tested_background:
        if not set(query_string_ids).issubset(background_string_ids):
            raise ValueError(
                "Mapped STRING enrichment query is not a subset of its "
                "background."
            )
        parameters["background_string_identifiers"] = "\r".join(
            background_string_ids
        )
    return _string_post_json("enrichment", parameters)


def _string_list(value) -> list[str]:
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        values = list(value)
    else:
        text = _text_value(value)
        values = text.replace(";", ",").split(",") if text else []
    return [
        text
        for item in values
        if (text := _text_value(item))
    ]


def _string_enrichment_frame(
    data: list[dict],
    category: str,
    query_sizes: dict[str, int],
) -> pd.DataFrame:
    columns = [
        "term",
        "description",
        "direction",
        "number_of_genes",
        "number_of_genes_in_background",
        "p_value",
        "fdr",
        "signal",
        "score",
        "query_fraction",
        "contributors",
    ]
    frame = pd.DataFrame(data)
    if frame.empty or "category" not in frame.columns:
        return pd.DataFrame(columns=columns)

    frame = frame.loc[frame["category"].astype(str).eq(str(category))].copy()
    required = {
        "term",
        "description",
        "number_of_genes",
        "number_of_genes_in_background",
        "p_value",
        "fdr",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "STRING enrichment result is missing required fields: "
            f"{missing!r}."
        )
    if frame.empty:
        return pd.DataFrame(columns=columns)

    if "_direction" not in frame.columns:
        return pd.DataFrame(columns=columns)
    frame["direction"] = frame["_direction"].astype(str)
    frame = frame.loc[
        frame["direction"].isin({"activated", "inhibited"})
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["p_value"] = pd.to_numeric(
        frame["p_value"], errors="coerce"
    )
    frame["fdr"] = pd.to_numeric(frame["fdr"], errors="coerce")
    frame["number_of_genes"] = pd.to_numeric(
        frame["number_of_genes"], errors="coerce"
    )
    frame["number_of_genes_in_background"] = pd.to_numeric(
        frame["number_of_genes_in_background"], errors="coerce"
    )
    frame = frame.loc[
        np.isfinite(frame["p_value"])
        & frame["p_value"].between(0.0, 1.0, inclusive="both")
        & np.isfinite(frame["fdr"])
        & frame["fdr"].between(0.0, 1.0, inclusive="both")
        & np.isfinite(frame["number_of_genes"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["score"] = _qvalue_plot_values(frame["fdr"].to_numpy(dtype=float))
    if "signal" in frame.columns:
        frame["signal"] = pd.to_numeric(frame["signal"], errors="coerce")
    else:
        frame["signal"] = np.nan
    missing_signal = ~np.isfinite(frame["signal"])
    if missing_signal.any():
        # STRING's table endpoint does not expose the graphical endpoint's
        # composite signal in every API version. Raw enrichment evidence is
        # the deterministic fallback; the exact raw p-value remains in hover.
        frame.loc[missing_signal, "signal"] = _qvalue_plot_values(
            frame.loc[missing_signal, "p_value"].to_numpy(dtype=float)
        )
    denominators = frame["direction"].map(query_sizes).fillna(0).clip(lower=1)
    frame["query_fraction"] = (
        frame["number_of_genes"] / denominators
    ).clip(0.0, 1.0)
    names_column = "preferredNames" if "preferredNames" in frame else "inputGenes"
    if names_column in frame:
        frame["contributors"] = frame[names_column].map(
            lambda value: ", ".join(_string_list(value))
        )
    else:
        frame["contributors"] = ""

    frame["term"] = frame["term"].astype(str)
    frame["description"] = frame["description"].astype(str)
    frame = (
        frame.sort_values(
            ["fdr", "number_of_genes", "description"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates(["direction", "term"], keep="first")
    )
    return frame[columns].reset_index(drop=True)


def _marker_sizes(n_substrates: np.ndarray) -> np.ndarray:
    counts = np.asarray(n_substrates, dtype=float)
    if counts.size == 0:
        return counts
    roots = np.sqrt(np.clip(counts, 1.0, None))
    lo = float(np.nanmin(roots))
    hi = float(np.nanmax(roots))
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return np.full(counts.shape, 9.0, dtype=float)
    return 7.0 + 10.0 * (roots - lo) / (hi - lo)


def _matching_kinase_mask(results: pd.DataFrame, token: str | None) -> np.ndarray:
    token = str(token or "").strip().casefold()
    if not token:
        return np.zeros(len(results), dtype=bool)

    matched = np.zeros(len(results), dtype=bool)
    for column in ("kinase_id", "kinase", "kinase_gene", "kinase_uniprot"):
        matched |= (
            _text_series(results[column])
            .str.casefold()
            .eq(token)
            .to_numpy(dtype=bool)
        )
    return matched


def plot_kinase_volcano(
    state: SessionState,
    contrast: str,
    sign_threshold: float = 0.05,
    highlight: str | None = None,
    width: int | None = None,
    height: int = 1150,
) -> go.Figure:
    """Plot contrast-local KSEA effects against kinase-level q-values."""
    results = _kinase_results(state.adata)
    sub = results.loc[results["contrast"].astype(str).eq(str(contrast))].copy()

    tested = _tested_mask(sub["tested"])
    x = pd.to_numeric(sub["effect"], errors="coerce").to_numpy(dtype=float)
    qvalue = pd.to_numeric(sub["qvalue"], errors="coerce").to_numpy(dtype=float)
    y = _qvalue_plot_values(qvalue)
    n_substrates = pd.to_numeric(
        sub["n_substrates"], errors="coerce"
    ).to_numpy(dtype=float)

    visible = (
        tested
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(qvalue)
        & (qvalue >= 0.0)
        & (qvalue <= 1.0)
        & np.isfinite(n_substrates)
    )
    sub = sub.loc[visible].copy()
    x = x[visible]
    y = y[visible]
    qvalue = qvalue[visible]
    n_substrates = n_substrates[visible]

    significant = qvalue < float(sign_threshold)
    colors = np.where(
        significant & (x > 0.0),
        "red",
        np.where(significant & (x < 0.0), "blue", "gray"),
    )

    selected = _matching_kinase_mask(sub, highlight)
    opacity = np.where(selected, 1.0, 0.75)
    if selected.any():
        opacity = np.where(selected, 1.0, 0.10)

    sizes = _marker_sizes(n_substrates)
    sizes = np.where(selected, sizes + 3.0, sizes)
    line_width = np.where(selected, 1.5, 0.0)

    kinase_ids = _text_series(sub["kinase_id"]).to_numpy()
    kinase_names = _text_series(sub["kinase"]).to_numpy()
    kinase_genes = _text_series(sub["kinase_gene"]).to_numpy()
    kinase_uniprot = _text_series(sub["kinase_uniprot"]).to_numpy()
    activity = pd.to_numeric(
        sub["activity_score"], errors="coerce"
    ).to_numpy(dtype=float)
    pvalue = pd.to_numeric(sub["pvalue"], errors="coerce").to_numpy(dtype=float)
    kinase_mean = pd.to_numeric(
        sub["kinase_mean_log2fc"], errors="coerce"
    ).to_numpy(dtype=float)
    global_mean = pd.to_numeric(
        sub["global_mean_log2fc"], errors="coerce"
    ).to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker={
                "size": sizes,
                "color": colors,
                "opacity": opacity,
                "line": {"color": "black", "width": line_width},
            },
            text=kinase_ids,
            customdata=np.column_stack(
                [
                    kinase_ids,
                    kinase_names,
                    kinase_genes,
                    kinase_uniprot,
                    activity,
                    pvalue,
                    qvalue,
                    n_substrates,
                    kinase_mean,
                    global_mean,
                ]
            ),
            hovertemplate=(
                "Kinase: %{customdata[1]}<br>"
                "Gene: %{customdata[2]}<br>"
                "UniProt: %{customdata[3]}<br>"
                "Mean shift: %{x:.3f}<br>"
                "Z-score: %{customdata[4]:.3f}<br>"
                "q-value: %{customdata[6]:.3e}<br>"
                "Substrates: %{customdata[7]:.0f}<br>"
                "<extra></extra>"
            ),
            name="",
        )
    )

    threshold_y = -np.log10(float(sign_threshold))
    if x.size:
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        ymax = max(float(np.nanmax(y)), threshold_y)
    else:
        xmin, xmax, ymax = -1.0, 1.0, max(1.0, threshold_y)
    xpad = max((xmax - xmin) * 0.05, 0.1)

    increased = int(np.sum(significant & (x > 0.0)))
    decreased = int(np.sum(significant & (x < 0.0)))
    unchanged = int(len(x) - increased - decreased)
    annotations = [
        {
            "x": 0.02,
            "y": 0.98,
            "xref": "paper",
            "yref": "paper",
            "opacity": 0.7,
            "text": f"<b>{decreased}</b>",
            "bgcolor": "blue",
            "font": {"color": "white"},
            "showarrow": False,
        },
        {
            "x": 0.5,
            "y": 0.98,
            "xref": "paper",
            "yref": "paper",
            "text": f"<b>{unchanged}</b>",
            "bgcolor": "lightgrey",
            "font": {"color": "black"},
            "showarrow": False,
        },
        {
            "x": 0.98,
            "y": 0.98,
            "xref": "paper",
            "yref": "paper",
            "opacity": 0.7,
            "text": f"<b>{increased}</b>",
            "bgcolor": "red",
            "font": {"color": "white"},
            "showarrow": False,
        },
    ]

    selected_indices = np.flatnonzero(selected)
    if selected_indices.size:
        index = int(selected_indices[0])
        direction = 1 if x[index] >= 0.0 else -1
        label = (
            kinase_genes[index]
            or kinase_names[index]
            or kinase_ids[index]
        )
        annotations.append(
            {
                "x": float(x[index]) + direction * 0.05,
                "y": float(y[index]) + 0.05,
                "ax": float(x[index]) + direction * 0.5,
                "ay": float(y[index]) + 0.5,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "text": f"{escape(str(label))}",
                "showarrow": True,
                "arrowhead": 0,
            }
        )

    fig.update_layout(
        title={"text": "Kinase Volcano Plot", "x": 0.5},
        annotations=annotations,
        width=width,
        height=height,
        margin={"l": 70, "r": 50, "t": 60, "b": 60, "autoexpand": False},
        showlegend=False,
        shapes=[
            {
                "type": "line",
                "x0": xmin - xpad,
                "x1": xmax + xpad,
                "y0": threshold_y,
                "y1": threshold_y,
                "line": {"color": "black", "dash": "dash"},
            },
            {
                "type": "line",
                "x0": 0,
                "x1": 0,
                "y0": 0,
                "y1": ymax,
                "line": {"color": "black", "dash": "dash"},
            },
        ],
        xaxis={"title": "Mean substrate shift vs background (log2FC)"},
        yaxis={"title": "-log10(kinase q-value)"},
    )
    return fig


def _search_options(results: pd.DataFrame) -> list[str]:
    values: set[str] = set()
    for column in ("kinase", "kinase_gene", "kinase_uniprot", "kinase_id"):
        values.update(
            value
            for value in _text_series(results[column])
            if value
        )
    return sorted(values, key=str.casefold)


def _selected_kinase_row(
    results: pd.DataFrame,
    contrast: str,
    token: str | None,
) -> pd.Series | None:
    sub = results.loc[results["contrast"].astype(str).eq(str(contrast))]
    matched = _matching_kinase_mask(sub, token)
    if not matched.any():
        return None
    return sub.iloc[int(np.flatnonzero(matched)[0])]


def _dendrogram_lines(
    fig: go.Figure,
    linkage: np.ndarray,
    *,
    orientation: str,
    row: int,
    col: int,
) -> None:
    linkage = np.asarray(linkage, dtype=float)
    if linkage.ndim != 2 or linkage.shape[0] == 0:
        # Keep the subplot axis alive so singleton labels remain visible.
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers",
                marker={"opacity": 0.0},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        return

    dendrogram = sch.dendrogram(linkage, no_plot=True)
    line_x: list[float | None] = []
    line_y: list[float | None] = []

    for icoord, dcoord in zip(
        dendrogram["icoord"],
        dendrogram["dcoord"],
    ):
        leaf_coord = (np.asarray(icoord, dtype=float) - 5.0) / 10.0
        distance = np.asarray(dcoord, dtype=float)

        if orientation == "top":
            x, y = leaf_coord, distance
        else:
            x, y = distance, leaf_coord
        line_x.extend([*x.tolist(), None])
        line_y.extend([*y.tolist(), None])

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": "#555", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _kinase_activity_heatmap(
    adata,
    results: pd.DataFrame,
    profile_name: str = "significant",
) -> pn.viewable.Viewable:
    clustering = _kinase_activity(adata).get("clustering")
    if not isinstance(clustering, Mapping):
        return pn.pane.Alert(
            "Kinase activity clustering is not available in this result.",
            alert_type="light",
            sizing_mode="stretch_width",
        )
    profiles = clustering.get("profiles")
    if isinstance(profiles, Mapping):
        profile = profiles.get(str(profile_name))
    elif profile_name == "significant":
        # Compatibility with results created before the toggle existed.
        profile = clustering
    else:
        profile = None

    if not isinstance(profile, Mapping):
        return pn.pane.Alert(
            "The all-tested clustering is not available in this result. "
            "Rerun it with the current ProteoFlux version.",
            alert_type="light",
            sizing_mode="stretch_width",
        )

    kinase_ids = [
        str(value) for value in profile.get("kinase_ids", [])
    ]
    contrast_names = [
        str(value) for value in profile.get("contrast_names", [])
    ]
    kinase_order = [
        str(value)
        for value in profile.get("kinase_order", kinase_ids)
    ]
    contrast_order = [
        str(value)
        for value in profile.get("contrast_order", contrast_names)
    ]

    if not kinase_ids or not contrast_names:
        if profile_name == "all_tested":
            message = "No kinase had a valid KSEA test in any contrast."
        else:
            threshold = float(clustering.get("sign_threshold", 0.05))
            message = (
                f"No kinase was significant at q < {threshold:g} "
                "in any contrast."
            )
        return pn.pane.Alert(
            message,
            alert_type="light",
            sizing_mode="stretch_width",
        )

    work = results.copy()
    work["kinase_id"] = _text_series(work["kinase_id"])
    work["kinase_gene"] = _text_series(work["kinase_gene"])
    work["contrast"] = _text_series(work["contrast"])
    for column in (
        "activity_score",
        "effect",
        "qvalue",
        "n_substrates",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    # Untested cells must stay blank, even if an old result contains
    # placeholder numeric values.
    untested = ~_tested_mask(work["tested"])
    work.loc[
        untested,
        ["activity_score", "effect", "qvalue"],
    ] = np.nan

    def pivot(column: str) -> pd.DataFrame:
        return (
            work.pivot(
                index="kinase_id",
                columns="contrast",
                values=column,
            )
            .reindex(index=kinase_order, columns=contrast_order)
        )

    activity = pivot("activity_score")
    effects = pivot("effect")
    qvalues = pivot("qvalue")
    substrate_counts = pivot("n_substrates")

    kinase_lookup = (
        work.drop_duplicates("kinase_id", keep="first")
        .set_index("kinase_id")
    )

    kinase_labels = []
    kinase_genes = []
    kinase_uniprots = []
    for kinase_id in kinase_order:
        row = kinase_lookup.loc[kinase_id]
        kinase = _text_value(row.get("kinase", ""))
        gene = _text_value(row.get("kinase_gene", ""))
        uniprot = _text_value(row.get("kinase_uniprot", ""))

        kinase_labels.append(kinase or gene or kinase_id)
        kinase_genes.append(gene)
        kinase_uniprots.append(uniprot)

    z = activity.to_numpy(dtype=float)
    finite = np.abs(z[np.isfinite(z)])
    color_limit = float(np.max(finite)) if finite.size else 1.0
    if color_limit <= 0.0:
        color_limit = 1.0

    qvalue_matrix = qvalues.to_numpy(dtype=float)
    sign_threshold = float(clustering.get("sign_threshold", 0.05))
    significance_symbols = _significance_symbols(
        qvalue_matrix,
        sign_threshold,
    )
    significance_labels = _significance_labels(
        qvalue_matrix,
        significance_symbols,
    )

    customdata = np.empty(
        (len(kinase_order), len(contrast_order), 8),
        dtype=object,
    )
    customdata[:, :, 0] = np.asarray(
        kinase_labels, dtype=object
    )[:, None]
    customdata[:, :, 1] = np.asarray(
        kinase_genes, dtype=object
    )[:, None]
    customdata[:, :, 2] = np.asarray(
        kinase_uniprots, dtype=object
    )[:, None]
    customdata[:, :, 3] = effects.to_numpy(dtype=float)
    customdata[:, :, 4] = qvalue_matrix
    customdata[:, :, 5] = substrate_counts.to_numpy(dtype=float)
    customdata[:, :, 6] = np.asarray(
        [value.replace("_vs_", "_v_") for value in contrast_order],
        dtype=object,
    )[None, :]
    customdata[:, :, 7] = significance_labels

    contrast_labels = [
        value.replace("_vs_", "_v_")
        for value in contrast_order
    ]
    contrast_range = [-0.5, len(contrast_order) - 0.5]
    kinase_range = [len(kinase_order) - 0.5, -0.5]

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.18, 0.82],
        column_widths=[0.12, 0.88],
        specs=[[None, {}], [{}, {}]],
        shared_xaxes="columns",
        shared_yaxes="rows",
        horizontal_spacing=0.06,
        vertical_spacing=0.01,
    )

    _dendrogram_lines(
        fig,
        profile.get(
            "contrast_linkage",
            np.empty((0, 4)),
        ),
        orientation="top",
        row=1,
        col=2,
    )
    _dendrogram_lines(
        fig,
        profile.get(
            "kinase_linkage",
            np.empty((0, 4)),
        ),
        orientation="left",
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=np.arange(len(contrast_order)),
            y=np.arange(len(kinase_order)),
            customdata=customdata,
            colorscale="RdBu_r",
            zmin=-color_limit,
            zmax=color_limit,
            zmid=0.0,
            xgap=1,
            ygap=1,
            colorbar={
                "title": "Z-score",
                "x": 1.12,
            },
            hovertemplate=(
                "Kinase: %{customdata[0]}<br>"
                "Gene: %{customdata[1]}<br>"
                "UniProt: %{customdata[2]}<br>"
                "Contrast: %{customdata[6]}<br>"
                "Z-score: %{z:.3f}<br>"
                "Mean shift: %{customdata[3]:.3f}<br>"
                "q-value: %{customdata[4]:.3e}<br>"
                "Significance: %{customdata[7]}<br>"
                "Substrates: %{customdata[5]:.0f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )

    star_rows, star_cols = np.nonzero(significance_symbols != "")
    if star_rows.size:
        star_values = z[star_rows, star_cols]
        dark_cell = np.abs(star_values) >= 0.55 * color_limit
        for use_dark_cells, text_color in (
            (False, "#111"),
            (True, "white"),
        ):
            keep = dark_cell == use_dark_cells
            if not keep.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=star_cols[keep],
                    y=star_rows[keep],
                    mode="text",
                    text=significance_symbols[
                        star_rows[keep],
                        star_cols[keep],
                    ],
                    textfont={"color": text_color, "size": 13},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    heatmap_height = max(
        520,
        min(1050, 260 + 16 * len(kinase_order)),
    )
    fig.update_layout(
        title={
            "text": "Kinase activity across contrasts",
            "x": 0.5,
        },
        height=heatmap_height,
        autosize=True,
        showlegend=False,
        margin={
            "l": 25,
            "r": 150,
            "t": 65,
            "b": 100,
        },
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=contrast_range,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        autorange="reversed",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(kinase_order))),
        ticktext=kinase_labels,
        showticklabels=True,
        side="right",
        ticklabelstandoff=40,
        automargin=True,
        showgrid=False,
        zeroline=False,
        range=kinase_range,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(contrast_order))),
        ticktext=contrast_labels,
        showticklabels=True,
        tickangle=-45,
        title="Contrasts",
        range=contrast_range,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        showticklabels=False,
        range=kinase_range,
        row=2,
        col=2,
    )

    return pn.pane.Plotly(
        fig,
        height=heatmap_height,
        sizing_mode="stretch_width",
        config={"responsive": True},
        styles={"overflow": "hidden"},
    )


def _contrast_sample_indices(adata, contrast: str) -> np.ndarray:
    if "_vs_" not in str(contrast):
        return np.array([], dtype=int)

    condition_a, condition_b = str(contrast).split("_vs_", 1)
    conditions = _text_series(adata.obs["CONDITION"]).to_numpy()

    def condition_indices(condition: str) -> np.ndarray:
        indices = np.flatnonzero(conditions == condition)
        sample_names = (
            adata.obs_names[indices]
            .astype(str)
            .to_numpy()
        )
        return indices[np.argsort(sample_names, kind="stable")]

    return np.concatenate(
        [
            condition_indices(condition_a),
            condition_indices(condition_b),
        ]
    )


def _kinase_substrate_heatmap(
    adata,
    results: pd.DataFrame,
    substrates: pd.DataFrame,
    contrast: str,
    kinase_token: str | None,
) -> pn.viewable.Viewable:
    row = _selected_kinase_row(results, contrast, kinase_token)
    if row is None:
        return pn.Spacer(width=800, height=1)

    kinase_id = str(row["kinase_id"])
    kinase = str(row.get("kinase", "") or kinase_id)
    link_mask = (
        _text_series(substrates["contrast"]).eq(str(contrast))
        & _text_series(substrates["kinase_id"]).eq(kinase_id)
    )
    site_rows = (
        substrates.loc[
            link_mask,
            ["phosphosite_id", "database_source"],
        ]
        .drop_duplicates("phosphosite_id")
    )
    site_ids = _text_series(site_rows["phosphosite_id"]).tolist()
    site_sources = _text_series(site_rows["database_source"]).tolist()

    feature_indices = adata.var_names.astype(str).get_indexer(site_ids)
    present = feature_indices >= 0
    site_ids = [site for site, keep in zip(site_ids, present) if keep]
    site_sources = [
        source for source, keep in zip(site_sources, present) if keep
    ]
    feature_indices = feature_indices[present]
    if "GENE_NAMES" in adata.var.columns:
        site_genes = (
            _text_series(adata.var.iloc[feature_indices]["GENE_NAMES"])
            .replace("", "n/a")
            .tolist()
        )
    else:
        site_genes = ["n/a"] * len(site_ids)
    sample_indices = _contrast_sample_indices(adata, contrast)

    if not site_ids or sample_indices.size == 0:
        return pn.Card(
            pn.pane.Markdown(
                "**Contributing Phosphosites**",
                styles={
                    "font-size": "16px",
                    "padding": "0",
                    "line-height": "0px",
                },
            ),
            make_hr(),
            pn.pane.Markdown(
                "No contributing phosphosite profiles are available for this contrast."
            ),
            width=410,
            collapsible=False,
            hide_header=True,
            styles={
                "background": "#f9f9f9",
                "border-radius": "8px",
                "box-shadow": "3px 3px 5px #bcbcbc",
                "padding": "8px",
            },
        )

    matrix = adata.X[sample_indices, :][:, feature_indices]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    absolute = np.asarray(matrix, dtype=float).T

    with np.errstate(invalid="ignore"):
        centered = absolute - np.nanmean(absolute, axis=1, keepdims=True)

    site_log2fc = np.full(len(site_ids), np.nan, dtype=float)
    contrast_names = [str(value) for value in adata.uns.get("contrast_names", [])]
    if contrast in contrast_names and "log2fc" in adata.varm:
        contrast_index = contrast_names.index(contrast)
        site_log2fc = np.asarray(
            adata.varm["log2fc"][feature_indices, contrast_index],
            dtype=float,
        ).ravel()
        sort_values = np.where(
            np.isfinite(site_log2fc),
            site_log2fc,
            -np.inf,
        )
        order = np.argsort(-sort_values, kind="stable")
        site_ids = [site_ids[index] for index in order]
        site_sources = [site_sources[index] for index in order]
        site_genes = [site_genes[index] for index in order]
        centered = centered[order]
        absolute = absolute[order]
        site_log2fc = site_log2fc[order]

    sample_names = adata.obs_names[sample_indices].astype(str).tolist()
    sample_conditions = _text_series(
        adata.obs.iloc[sample_indices]["CONDITION"]
    ).to_numpy(dtype=object)
    finite = np.abs(centered[np.isfinite(centered)])
    color_limit = float(np.max(finite)) if finite.size else 1.0
    if color_limit <= 0.0:
        color_limit = 1.0

    customdata = np.empty(
        (len(site_ids), len(sample_names), 5),
        dtype=object,
    )
    customdata[:, :, 0] = absolute
    customdata[:, :, 1] = sample_conditions[None, :]
    customdata[:, :, 2] = site_log2fc[:, None]
    customdata[:, :, 3] = np.asarray(
        site_sources,
        dtype=object,
    )[:, None]
    customdata[:, :, 4] = np.asarray(
        site_genes,
        dtype=object,
    )[:, None]

    fig = go.Figure(
        go.Heatmap(
            z=centered,
            x=sample_names,
            y=site_ids,
            customdata=customdata,
            colorscale="RdBu_r",
            zmin=-color_limit,
            zmax=color_limit,
            zmid=0.0,
            colorbar={"title": "Deviation"},
            hovertemplate=(
                "Phosphosite: %{y}<br>"
                "Substrate gene: %{customdata[4]}<br>"
                "Database source: %{customdata[3]}<br>"
                "Sample: %{x}<br>"
                "Condition: %{customdata[1]}<br>"
                "Deviation from site mean: %{z:.3f}<br>"
                "Final intensity: %{customdata[0]:.3f}<br>"
                "Contrast log2FC: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    heatmap_height = max(
        330,
        min(650, 150 + 18 * len(site_ids)),
    )
    fig.update_layout(
        title={"text": f"Substrate profiles — {kinase}", "x": 0.5},
        height=heatmap_height,
        autosize=True,
        margin={"l": 110, "r": 20, "t": 55, "b": 90},
        xaxis={"title": "Samples", "tickangle": -45},
        yaxis={"title": "Phosphosites", "autorange": "reversed", "showticklabels":False},
    )

    header=pn.Row(
        pn.pane.Markdown(
            "**Contributing Phosphosites**",
            styles={"font-size": "16px", "padding": "0", "line-height": "0px"},
        )
    )
    return pn.Card(
        header,
        make_hr(),
        pn.pane.Plotly(
            fig,
            height=heatmap_height,
            sizing_mode="stretch_width",
            config={"responsive": True},
            styles={"overflow": "hidden"},
        ),
        width=800,
        height=heatmap_height + 65,
        collapsible=False,
        hide_header=True,
        styles={
            "background": "#f9f9f9",
            "border-radius": "8px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "padding": "8px",
            "overflow": "hidden",
        },
    )


def _kinase_detail_card(
    results: pd.DataFrame,
    contrast: str,
    kinase_token: str | None,
) -> pn.viewable.Viewable:
    row = _selected_kinase_row(results, contrast, kinase_token)
    if row is None:
        return pn.Spacer(width=800, height=1)

    kinase = _text_value(
        row.get("kinase", ""),
        str(row["kinase_id"]),
    )
    gene = _text_value(row.get("kinase_gene", ""), "n/a")
    uniprot = _text_value(row.get("kinase_uniprot", ""))
    string_link = get_string_link(uniprot) if uniprot else ""


    Number = pn.indicators.Number
    effect = Number(
        name="Mean shift",
        value=float(row["effect"]),
        format="{value:.3f}",
        default_color="red",
        font_size="12pt",
        styles={"flex": "1"},
    )
    activity = Number(
        name="Z-score",
        value=float(row["activity_score"]),
        format="{value:.3f}",
        default_color="purple",
        font_size="12pt",
        styles={"flex": "1"},
    )
    qvalue = Number(
        name="Kinase q-value",
        value=float(row["qvalue"]),
        format="{value:.3e}",
        default_color="red",
        font_size="12pt",
        styles={"flex": "1"},
    )
    substrates = Number(
        name="Substrates",
        value=int(row["n_substrates"]),
        format="{value:.0f}",
        default_color="gray",
        font_size="12pt",
        styles={"flex": "1"},
    )

    header = pn.pane.Markdown(
        f"**Kinase**: {kinase}",
        styles={"font-size": "16px", "padding": "0", "line-height": "0px"},
    )

    footer_left = pn.pane.HTML(
        "<span style='font-size: 12px;'>"
        f"p-value: <b>{float(row['pvalue']):.3e}</b>"
        f" &nbsp;|&nbsp; Gene: <b>{gene}</b>"
        f" &nbsp;|&nbsp; UniProt: <b>{uniprot or 'n/a'}</b>"
        "</span>"
    )

    footer_right = pn.pane.HTML("")
    if uniprot:
        string_html = (
            " &nbsp;|&nbsp; "
            f"<a href='{escape(string_link, quote=True)}' "
            "target='_blank' rel='noopener'>STRING Entry</a>"
            if string_link
            else ""
        )
        footer_right = pn.pane.HTML(
            "<span style='font-size: 12px;'>"
            "🔗 "
            f"<a href='https://www.uniprot.org/uniprotkb/{uniprot}/entry' "
            "target='_blank' rel='noopener'>UniProt Entry</a>"
            f"{string_html}"
            "</span>",
            styles={"text-align": "right"},
        )

    footer = pn.Row(
        footer_left,
        footer_right,
        sizing_mode="stretch_width",
        styles={
            "justify-content": "space-between",
            "padding": "2px 8px 4px 0px",
            "margin-top": "-6px",
        },
    )

    card_style = {
        "background": "#f9f9f9",
        "align-items": "center",
        "border-radius": "8px",
        "text-align": "center",
        "padding": "5px",
        "box-shadow": "3px 3px 5px #bcbcbc",
        "justify-content": "space-evenly",
    }

    return pn.Card(
        header,
        make_hr(),
        pn.Row(effect, activity, sizing_mode="stretch_width"),
        make_hr(),
        pn.Row(qvalue, substrates, sizing_mode="stretch_width"),
        make_hr(),
        footer,
        width=800,
        collapsible=False,
        hide_header=True,
        styles=card_style,
    )


def _coordinated_enrichment_figure(
    *,
    contrasts: list[str],
    frames: dict[str, pd.DataFrame],
    errors_by_contrast: dict[str, dict[str, str]],
    query_sizes: dict[str, dict[str, int]],
    background_labels: dict[str, str],
    contrast: str,
    active_contrast: str,
    metric: str,
    show_term_labels: bool,
    max_terms: int = _ENRICHMENT_MAX_TERMS,
) -> go.Figure:
    metric_columns = {
        "signal": "signal",
        "fdr": "score",
        "number_of_genes": "number_of_genes",
    }
    metric_titles = {
        "signal": "Signal",
        "fdr": "−log10(FDR)",
        "number_of_genes": "Contributing kinases",
    }
    if metric not in metric_columns:
        raise ValueError(f"Unsupported enrichment metric: {metric!r}.")
    metric_column = metric_columns[metric]

    active = frames.get(active_contrast, pd.DataFrame())
    if active.empty:
        selected = active
    else:
        selected = (
            active.sort_values(
                [metric_column, "fdr", "number_of_genes", "description"],
                ascending=[False, True, False, True],
                kind="stable",
            )
            .drop_duplicates("term", keep="first")
            .head(int(max_terms))
        )

    term_order = selected["term"].astype(str).tolist()
    raw_term_labels = (
        selected.drop_duplicates("term").set_index("term")["description"].to_dict()
    )
    wrapped_lines = {
        term: wrap(
            str(label),
            width=_ENRICHMENT_LABEL_WRAP_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for term, label in raw_term_labels.items()
    }
    term_labels = {
        term: "<br>".join(escape(line) for line in lines)
        for term, lines in wrapped_lines.items()
    }
    max_label_lines = max(
        (len(lines) for lines in wrapped_lines.values()),
        default=1,
    )
    row_height = max(30, 15 * max_label_lines + 6)
    term_positions = {
        term: position for position, term in enumerate(term_order)
    }

    plotted_frames: list[pd.DataFrame] = []
    for contrast_name in contrasts:
        frame = frames[contrast_name]
        if term_order and not frame.empty:
            visible = frame.loc[frame["term"].isin(term_order)].copy()
            if not visible.empty:
                plotted_frames.append(visible)
    max_count = max(
        (
            float(frame["number_of_genes"].max())
            for frame in plotted_frames
            if not frame.empty
        ),
        default=1.0,
    )
    max_metric = max(
        (
            float(frame[metric_column].max())
            for frame in plotted_frames
            if not frame.empty
        ),
        default=1.0,
    )
    x_max = max(1.0, max_metric * 1.12)

    frame = frames.get(contrast, pd.DataFrame())
    visible = (
        frame.loc[frame["term"].isin(term_order)].copy()
        if term_order and not frame.empty
        else pd.DataFrame(columns=frame.columns)
    )
    fig = go.Figure()
    if not visible.empty:
        visible["_position"] = visible["term"].map(term_positions)
        visible["_metric_value"] = visible[metric_column]
        visible = visible.sort_values(
            ["_position", "direction"], kind="stable"
        )
        counts = visible["number_of_genes"].to_numpy(dtype=float)
        marker_sizes = 8.0 + 18.0 * np.sqrt(
            np.clip(counts / max(max_count, 1.0), 0.0, 1.0)
        )
        direction_labels = visible["direction"].map(
            {
                "activated": "Activated (KSEA Z > 0)",
                "inhibited": "Inhibited (KSEA Z < 0)",
            }
        )
        direction_query_sizes = np.asarray(
            [
                query_sizes.get(contrast, {}).get(direction, 0)
                for direction in visible["direction"]
            ],
            dtype=int,
        )
        direction_colors = np.where(
            visible["direction"].eq("activated"),
            "#d62728",
            "#1f77b4",
        )

        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for value, position in zip(
            visible["_metric_value"], visible["_position"]
        ):
            line_x.extend([0.0, float(value), None])
            line_y.extend([float(position), float(position), None])
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": "#a9a9a9", "width": 1.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )

        customdata = np.empty((len(visible), 12), dtype=object)
        customdata[:, 0] = visible["description"].astype(str).to_numpy()
        customdata[:, 1] = visible["term"].astype(str).to_numpy()
        customdata[:, 2] = visible["fdr"].to_numpy(dtype=float)
        customdata[:, 3] = counts
        customdata[:, 4] = visible[
            "number_of_genes_in_background"
        ].to_numpy(dtype=float)
        customdata[:, 5] = direction_query_sizes
        customdata[:, 6] = background_labels.get(contrast, "Unknown")
        customdata[:, 7] = visible["contributors"].astype(str).to_numpy()
        customdata[:, 8] = direction_labels.astype(str).to_numpy()
        customdata[:, 9] = visible["p_value"].to_numpy(dtype=float)
        customdata[:, 10] = visible["signal"].to_numpy(dtype=float)
        customdata[:, 11] = visible["score"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=visible["_metric_value"],
                y=visible["_position"],
                mode="markers",
                marker={
                    "size": marker_sizes,
                    "color": direction_colors,
                    "symbol": "circle",
                    "line": {"color": "white", "width": 0.8},
                },
                customdata=customdata,
                hovertemplate=(
                    "ID: %{customdata[1]}<br>"
                    "Signal: %{customdata[10]:.3f}<br>"
                    "−log10(FDR): %{customdata[11]:.3f}<br>"
                    "Raw p-value: %{customdata[9]:.3e}<br>"
                    "Contributing kinases: %{customdata[3]:.0f} / "
                    "%{customdata[5]:.0f}<br>"
                    "%{customdata[7]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    message = ""
    if frame.empty:
        if errors_by_contrast.get(contrast):
            message = "STRING query failed"
        elif not any(
            size >= 2 for size in query_sizes.get(contrast, {}).values()
        ):
            message = "No data available"
        else:
            message = "No data available"
    elif not term_order:
        message = "No match"
    elif visible.empty:
        message = "No match"
    if message:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=message,
            showarrow=False,
            align="center",
            font={"color": "#777", "size": 12},
        )

    is_active = contrast == active_contrast
    border_color = "#4c78a8" if is_active else "#c7c7c7"
    fig.update_xaxes(
        title=metric_titles[metric],
        range=[0.0, x_max],
        showgrid=True,
        gridcolor="#dedede",
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor=border_color,
        linewidth=2 if is_active else 1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(term_order))),
        ticktext=[term_labels[term] for term in term_order],
        showticklabels=bool(show_term_labels),
        ticklabelstandoff=15 if show_term_labels else 0,
        range=[len(term_order) - 0.5, -0.5] if term_order else [-0.5, 0.5],
        automargin=bool(show_term_labels),
        showgrid=True,
        gridcolor="#e4e4e4",
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor=border_color,
        linewidth=2 if is_active else 1,
    )

    fig.update_layout(
        height=max(460, 145 + row_height * max(len(term_order), 8)),
        autosize=True,
        showlegend=False,
        hovermode="closest",
        margin={
            "l": 300 if show_term_labels else 14,
            "r": 14,
            "t": 14,
            "b": 65,
        },
        plot_bgcolor="#edf4fb" if is_active else "#f4f4f4",
        paper_bgcolor="white",
    )
    return fig


def _maximal_uniform_subsets(
    present_conditions: tuple[str, ...],
    qualifying_pairs: set[frozenset[str]],
) -> list[tuple[str, ...]]:
    """Return inclusion-maximal subsets whose every pair qualifies."""
    order = {
        condition: index
        for index, condition in enumerate(present_conditions)
    }
    present = set(present_conditions)
    neighbours = {
        condition: set()
        for condition in present_conditions
    }

    for pair in qualifying_pairs:
        members = [
            condition
            for condition in pair
            if condition in present
        ]
        if len(members) != 2:
            continue
        left, right = members
        neighbours[left].add(right)
        neighbours[right].add(left)

    maximal: list[tuple[str, ...]] = []

    def visit(
        clique: set[str],
        candidates: set[str],
        excluded: set[str],
    ) -> None:
        if not candidates and not excluded:
            if len(clique) >= 2:
                maximal.append(
                    tuple(sorted(clique, key=order.__getitem__))
                )
            return

        pivot_pool = candidates | excluded
        pivot = max(
            pivot_pool,
            key=lambda condition: len(
                candidates & neighbours[condition]
            ),
            default=None,
        )
        extensions = candidates - (
            neighbours[pivot] if pivot is not None else set()
        )

        for condition in sorted(
            extensions,
            key=order.__getitem__,
        ):
            visit(
                clique | {condition},
                candidates & neighbours[condition],
                excluded & neighbours[condition],
            )
            candidates.remove(condition)
            excluded.add(condition)

    visit(set(), set(present_conditions), set())
    return sorted(
        maximal,
        key=lambda subset: (
            -len(subset),
            tuple(order[condition] for condition in subset),
        ),
    )


def _kinase_upset_figure(
    adata,
    results: pd.DataFrame,
    condition_kinases: pd.DataFrame,
    contrasts: list[str],
    sign_threshold: float,
) -> go.Figure:
    conditions = sorted(
        set(_text_series(adata.obs["CONDITION"]).tolist()),
        key=str.casefold,
    )
    condition_color_map = get_color_map(
        sorted(conditions, key=str.casefold),
        palette=px.colors.qualitative.Plotly,
        anchor=None,
    )
    condition_positions = {
        condition: index for index, condition in enumerate(conditions)
    }

    if condition_kinases.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=(
                "Condition-level kinase membership is unavailable.<br>"
                "Rerun ProteoFlux to generate the condition subset plot."
            ),
            showarrow=False,
            align="center",
            font={"color": "#777", "size": 12},
        )
        fig.update_layout(
            title={"text": "Kinase condition subsets", "x": 0.5},
            height=440,
            autosize=True,
            margin={"l": 25, "r": 25, "t": 55, "b": 25},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    membership = condition_kinases.copy()
    membership["condition"] = _text_series(membership["condition"])
    membership["kinase_id"] = _text_series(membership["kinase_id"])
    membership = membership.loc[
        membership["condition"].isin(conditions)
        & membership["kinase_id"].ne("")
    ].drop_duplicates(["condition", "kinase_id"], keep="first")

    condition_sets = {
        condition: set(
            membership.loc[
                membership["condition"].eq(condition), "kinase_id"
            ]
        )
        for condition in conditions
    }
    presence_by_kinase: dict[str, tuple[str, ...]] = {}
    for kinase_id in sorted(
        set().union(*condition_sets.values()),
        key=str.casefold,
    ):
        present = tuple(
            condition
            for condition in conditions
            if kinase_id in condition_sets[condition]
        )
        if present:
            presence_by_kinase[kinase_id] = present

    work = results.copy()
    work["contrast"] = _text_series(work["contrast"])
    work["kinase_id"] = _text_series(work["kinase_id"])
    work["qvalue"] = pd.to_numeric(work["qvalue"], errors="coerce")
    work["_tested"] = _tested_mask(work["tested"])
    work["_significant"] = (
        work["_tested"]
        & np.isfinite(work["qvalue"])
        & work["qvalue"].lt(float(sign_threshold))
    )

    kinase_gene_by_id: dict[str, str] = {}
    for kinase_id, kinase_gene in work[
        ["kinase_id", "kinase_gene"]
    ].itertuples(index=False, name=None):
        if kinase_gene and kinase_id not in kinase_gene_by_id:
            kinase_gene_by_id[kinase_id] = kinase_gene

    def _hover_gene_preview(kinase_ids: set[str], limit: int = 4) -> str:
        names = sorted(
            {
                kinase_gene_by_id.get(kinase_id, kinase_id)
                for kinase_id in kinase_ids
            },
            key=str.casefold,
        )
        preview = ", ".join(names[:limit])
        if len(names) > limit:
            preview += f" (+{len(names) - limit} more)"
        return preview or "None"

    contrast_pairs: dict[str, frozenset[str]] = {}
    for contrast in contrasts:
        if "_vs_" not in str(contrast):
            continue
        condition_a, condition_b = str(contrast).split("_vs_", 1)
        if (
            condition_a in condition_positions
            and condition_b in condition_positions
            and condition_a != condition_b
        ):
            contrast_pairs[str(contrast)] = frozenset(
                (condition_a, condition_b)
            )

    tested_pairs_by_kinase: dict[str, set[frozenset[str]]] = {}
    significant_pairs_by_kinase: dict[str, set[frozenset[str]]] = {}
    tested_rows = work.loc[
        work["_tested"],
        ["contrast", "kinase_id", "_significant"],
    ]
    for contrast, kinase_id, is_significant in tested_rows.itertuples(
        index=False,
        name=None,
    ):
        pair = contrast_pairs.get(str(contrast))
        kinase_id = str(kinase_id)
        if pair is not None and kinase_id in presence_by_kinase:
            tested_pairs_by_kinase.setdefault(kinase_id, set()).add(pair)
            if bool(is_significant):
                significant_pairs_by_kinase.setdefault(
                    kinase_id, set()
                ).add(pair)

    groups: dict[tuple[str, ...], dict[str, set[str]]] = {}
    for kinase_id, present in presence_by_kinase.items():
        tested_pairs = tested_pairs_by_kinase.get(kinase_id, set())
        significant_pairs = significant_pairs_by_kinase.get(
            kinase_id, set()
        )
        nonsignificant_pairs = tested_pairs.difference(significant_pairs)

        significant_subsets = _maximal_uniform_subsets(
            present,
            significant_pairs,
        )
        nonsignificant_subsets = _maximal_uniform_subsets(
            present,
            nonsignificant_pairs,
        )

        for subset in significant_subsets:
            groups.setdefault(
                subset,
                {"significant": set(), "other": set()},
            )["significant"].add(kinase_id)
        for subset in nonsignificant_subsets:
            groups.setdefault(
                subset,
                {"significant": set(), "other": set()},
            )["other"].add(kinase_id)

        if (
            len(present) >= 2
            and not significant_subsets
            and not nonsignificant_subsets
        ):
            groups.setdefault(
                present,
                {"significant": set(), "other": set()},
            )["other"].add(kinase_id)

    all_ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]["significant"]) - len(item[1]["other"]),
            -len(item[0]),
            tuple(condition_positions[condition] for condition in item[0]),
        ),
    )
    total_group_count = len(all_ordered_groups)
    ordered_groups = all_ordered_groups[:_UPSET_MAX_SUBSETS]
    subset_title = "Kinase condition subsets"
    if total_group_count > len(ordered_groups):
        subset_title += (
            f" — top {len(ordered_groups)} of {total_group_count}"
        )
    subsets = [subset for subset, _ in ordered_groups]
    significant_counts = [
        len(counts["significant"]) for _, counts in ordered_groups
    ]
    nonsignificant_counts = [
        len(counts["other"]) for _, counts in ordered_groups
    ]
    totals = np.asarray(significant_counts) + np.asarray(
        nonsignificant_counts
    )
    x = np.arange(len(subsets), dtype=float)
    subset_labels = [" + ".join(subset) for subset in subsets]
    significant_names = [
        _hover_gene_preview(counts["significant"])
        for _, counts in ordered_groups
    ]
    nonsignificant_names = [
        _hover_gene_preview(counts["other"])
        for _, counts in ordered_groups
    ]
    set_sizes = [len(condition_sets[condition]) for condition in conditions]

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.16, 0.84],
        row_heights=[0.68, 0.32],
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=nonsignificant_counts,
            name="Non-significant in every contrast",
            marker_color="#bdbdbd",
            customdata=np.column_stack(
                [subset_labels, nonsignificant_names]
            ),
            hovertemplate=(
                "Condition subset: %{customdata[0]}<br>"
                "Non-significant in every contrast: %{y:.0f}<br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=significant_counts,
            name=f"Significant in every contrast",
            marker_color="#8064a2",
            customdata=np.column_stack(
                [subset_labels, significant_names]
            ),
            hovertemplate=(
                "Condition subset: %{customdata[0]}<br>"
                "Significant in every contrast: %{y:.0f}<br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=totals,
            mode="text",
            text=[str(int(value)) for value in totals],
            textposition="top center",
            textfont={"color": "#555", "size": 10},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    condition_y = np.arange(len(conditions), dtype=float)
    fig.add_trace(
        go.Bar(
            x=set_sizes,
            y=condition_y,
            orientation="h",
            marker_color=[
                condition_color_map[condition] for condition in conditions
            ],
            customdata=np.asarray(conditions, dtype=object),
            hovertemplate=(
                "Condition: %{customdata}<br>"
                "Eligible kinases: %{x:.0f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if conditions and subsets:
        fig.add_trace(
            go.Scatter(
                x=np.repeat(x, len(conditions)),
                y=np.tile(condition_y, len(subsets)),
                mode="markers",
                marker={"size": 8, "color": "#dedede"},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        line_x: list[float | None] = []
        line_y: list[float | None] = []
        active_x: list[float] = []
        active_y: list[float] = []
        active_labels: list[str] = []
        for position, subset, label in zip(x, subsets, subset_labels):
            subset_positions = [
                float(condition_positions[condition]) for condition in subset
            ]
            line_x.extend([float(position), float(position), None])
            line_y.extend(
                [min(subset_positions), max(subset_positions), None]
            )
            active_x.extend([float(position)] * len(subset_positions))
            active_y.extend(subset_positions)
            active_labels.extend([label] * len(subset_positions))

        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": "#555", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=active_x,
                y=active_y,
                mode="markers",
                marker={"size": 10, "color": "#555"},
                customdata=np.asarray(active_labels, dtype=object),
                hovertemplate=(
                    "Condition subset: %{customdata}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Explicit annotations are more reliable than subplot tick labels here.
    matrix_x0 = float(fig.layout.xaxis4.domain[0])
    for condition, position in zip(conditions, condition_y):
        fig.add_annotation(
            x=matrix_x0 - 0.008,
            y=float(position),
            xref="paper",
            yref="y4",
            text=f"<b>{escape(condition)}</b>",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font={"size": 13, "color": "#444"},
        )

    figure_height = max(460, 340 + 26 * len(conditions))
    fig.update_layout(
        title={"text": subset_title, "x": 0.61},
        height=figure_height,
        autosize=True,
        barmode="stack",
        bargap=0.28,
        hovermode="closest",
        legend={
            "orientation": "v",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.85,
            "yanchor": "bottom",
        },
        margin={"l": 100, "r": 25, "t": 70, "b": 35},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[-0.5, len(subsets) - 0.5],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title="Kinases",
        rangemode="tozero",
        gridcolor="#e7e7e7",
        zeroline=False,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title={
            "text": "Eligible kinases",
            "font": {"size": 10, "color": "#666"},
            "standoff": 4,
        },
        autorange="reversed",
        showline=True,
        linecolor="#aaa",
        linewidth=1,
        ticks="outside",
        ticklen=3,
        tickcolor="#aaa",
        tickfont={"size": 9, "color": "#666"},
        tickformat="d",
        nticks=4,
        showgrid=True,
        gridcolor="#ececec",
        gridwidth=1,
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[len(conditions) - 0.5, -0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[-0.5, len(subsets) - 0.5],
        row=2,
        col=2,
    )
    fig.update_yaxes(
        range=[len(conditions) - 0.5, -0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=2,
        col=2,
    )
    return fig


def _contrast_options(adata, results: pd.DataFrame) -> list[str]:
    available = set(results["contrast"].astype(str))
    configured = [str(value) for value in adata.uns.get("contrast_names", [])]
    ordered = [contrast for contrast in configured if contrast in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def _enrichment_contrast_page(
    contrasts: list[str],
    reference_contrast: str,
    page_size: int = _ENRICHMENT_MAX_CONTRASTS,
) -> list[str]:
    """Return the fixed-size contrast page containing the reference."""
    if page_size < 1:
        raise ValueError("Enrichment contrast page size must be >= 1.")
    if len(contrasts) <= page_size:
        return list(contrasts)

    try:
        reference_index = contrasts.index(str(reference_contrast))
    except ValueError:
        reference_index = 0

    page_start = (reference_index // page_size) * page_size
    return list(contrasts[page_start : page_start + page_size])


@log_time("Preparing Kinases Tab")
def kinases_tab(state: SessionState):
    adata = state.adata
    activity_payload = _kinase_activity(adata)
    results = _kinase_results(adata)
    substrates = _kinase_substrates(adata)
    condition_kinases = _condition_kinases(adata)
    database_summary = _database_summary(adata)
    contrasts = _contrast_options(adata, results)
    if not contrasts:
        return pn.pane.Markdown("No kinase activity contrasts are available.")

    analysis = adata.uns.get("analysis", {}) or {}
    clustering = activity_payload.get("clustering", {}) or {}
    sign_threshold_value = clustering.get(
        "sign_threshold",
        analysis.get("sign_threshold", 0.05),
    )
    sign_threshold = float(
        0.05
        if sign_threshold_value is None
        else sign_threshold_value
    )
    method = _text_value(activity_payload.get("method", "ksea"), "ksea").upper()
    min_substrates = activity_payload.get("min_substrates", "n/a")
    database_metadata = activity_payload.get("database", {}) or {}
    database_filename = (
        _text_value(database_metadata.get("filename"), "Not recorded")
        if isinstance(database_metadata, Mapping)
        else _text_value(database_metadata, "Not recorded")
    )
    if isinstance(database_metadata, Mapping):
        database_relationships = database_metadata.get("relationships")
        matched_relationships = database_metadata.get(
            "matched_relationships"
        )
        matched_phosphosites = database_metadata.get(
            "matched_phosphosites"
        )
        matched_kinases = database_metadata.get("matched_kinases")
    else:
        database_relationships = None
        matched_relationships = None
        matched_phosphosites = None
        matched_kinases = None

    # Older H5AD files do not contain the global matched counts. Derive the
    # closest equivalent from their compact per-contrast substrate links.
    if matched_relationships is None:
        matched_relationships = len(
            substrates.drop_duplicates(["kinase_id", "phosphosite_id"])
        )
    if matched_phosphosites is None:
        matched_phosphosites = substrates["phosphosite_id"].nunique()

    if matched_kinases is None:
        matched_kinases = substrates["kinase_id"].nunique()

    relationship_summary = (
        f"{_count_text(database_relationships)} unique relationships; "
        f"{_count_text(matched_relationships)} kinase-site matches.\n\n"
        f"**In this analysis:** {_count_text(matched_phosphosites)} phosphosites, "
        f"{_count_text(matched_kinases)} kinases across all tested contrasts."
    )
    if database_summary.empty:
        database_summary_md = f"**Database:** `{database_filename}`\n\n"
    else:
        database_lines = []
        for _, database_row in database_summary.iterrows():
            database_lines.append(
                f"- `{_text_value(database_row['filename'], 'Unknown')}` --- "
                f"+{_count_text(database_row['new_relationships'], '0')} "
                "relationships; "
                f"+{_count_text(database_row['new_matched_relationships'], '0')} "
                "kinase-site matches \n\n"
                f"\t+{_count_text(database_row['new_matched_phosphosites'], '0')} "
                "sites, "
                f"+{_count_text(database_row['new_matched_kinases'], '0')} "
                "kinases across all tested contrasts."
            )
        database_summary_md = (
            "**Databases - incremental additions in configured order:**\n\n"
            + "\n".join(database_lines)
            + "\n\n"
        )
    conditions = sorted(
        set(_text_series(adata.obs["CONDITION"]).tolist()),
        key=str.casefold,
    )
    enrichment_background_label = (
        "KSEA-tested kinases"
        if _ENRICHMENT_USE_TESTED_BACKGROUND
        else "STRING species proteome"
    )
    summary_md = (
        f"{len(conditions)} Conditions - {len(contrasts)} Contrasts\n\n"
        f"**Method:** {method} (minimum {min_substrates} substrates)\n\n"
        f"{database_summary_md}"
        f"**Total:** {relationship_summary}\n\n"
        f"**Background:** {enrichment_background_label}\n\n"
    )
    summary_pane = pn.pane.Markdown(
        summary_md,
        sizing_mode="stretch_width",
        margin=(-10, 0, 0, 20),
        styles={
            "line-height": "1.4em",
            "word-break": "break-word",
            "overflow-wrap": "anywhere",
            "min-width": "0",
        },
    )
    upset_figure = _kinase_upset_figure(
        adata,
        results,
        condition_kinases,
        contrasts,
        sign_threshold,
    )
    upset_height = int(upset_figure.layout.height or 460)
    upset_plot = pn.pane.Plotly(
        upset_figure,
        height=upset_height,
        margin=(0, 20, 0, 0),
        sizing_mode="stretch_width",
        config={"responsive": True},
        styles={"flex": "1", "overflow": "hidden"},
    )
    kinase_summary_pane = pn.Row(
        pn.Column(
            pn.pane.Markdown("##   Summary", disable_anchors=True),
            summary_pane,
            styles={"flex": "0.32", "min-width": "0"},
        ),
        make_vr(),
        pn.Spacer(width=20),
        upset_plot,
        height=max(500, upset_height + 20),
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": "98vw",
        },
    )
    enrichment_queries = _significant_kinases_by_contrast(
        results,
        contrasts,
        sign_threshold,
    )
    enrichment_backgrounds = _tested_kinases_by_contrast(
        results,
        contrasts,
    )

    contrast_sel = pn.widgets.Select(
        name="Contrast",
        options=contrasts,
        value=contrasts[0],
        width=180,
    )
    search_input = pn.widgets.AutocompleteInput(
        name="Search Kinase",
        options=_search_options(results),
        case_sensitive=False,
        width=190,
    )
    clear_search = pn.widgets.Button(name="Clear", width=80)
    clear_search.on_click(lambda _event: setattr(search_input, "value", ""))

    volcano_dmap = pn.bind(
        plot_kinase_volcano,
        state=state,
        contrast=contrast_sel,
        sign_threshold=sign_threshold,
        highlight=search_input,
        width=None,
        height=1150,
    )
    volcano_dmap = bind_uirevision(
        volcano_dmap,
        contrast_sel,
        prefix="kinase-volcano",
    )
    volcano_plot = pn.pane.Plotly(
        volcano_dmap,
        #height=1150,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        config={"responsive": True},
        styles={
            "border-radius": "8px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "flex": "1",
        },
    )

    def _on_volcano_click(event) -> None:
        click = event.new or {}
        points = click.get("points") or []
        if points:
            kinase_id = str(points[0].get("text") or "").strip()
            if kinase_id:
                search_input.value = kinase_id

    volcano_plot.param.watch(_on_volcano_click, "click_data")

    def _kinase_selection_builder(**kwargs) -> pd.DataFrame:
        return build_kinase_selection_df(
            **kwargs,
            sign_threshold=sign_threshold,
        )

    (
        download_selection,
        _on_volcano_selected_data,
        _on_volcano_click_data,
        _on_cohort_ids,
    ) = make_volcano_selection_downloader(
        state=state,
        contrast_getter=lambda: str(contrast_sel.value),
        spec=SelectionExportSpec(
            filename="proteoflux_kinase_selection.csv",
            label="Download selection",
            uniprot_var_col="KINASE_UNIPROT",
            id_col_name="KINASE_ID",
        ),
        selection_df_builder=_kinase_selection_builder,
    )
    volcano_plot.param.watch(
        lambda event: _on_volcano_selected_data(event.new),
        "selected_data",
    )
    volcano_plot.param.watch(
        lambda event: _on_volcano_click_data(event.new),
        "click_data",
    )

    # Plotly click_data is sticky. Clearing the search must explicitly
    # relinquish the click source so export priority falls back to
    # cohort, then lasso, exactly as in the Overview tab.
    def _on_search_cleared(event) -> None:
        if not _text_value(event.new):
            _on_volcano_click_data({})

    search_input.param.watch(_on_search_cleared, "value")

    detail = pn.bind(
        _kinase_detail_card,
        results=results,
        contrast=contrast_sel,
        kinase_token=search_input,
    )

    substrate_heatmap = pn.bind(
        _kinase_substrate_heatmap,
        adata=adata,
        results=results,
        substrates=substrates,
        contrast=contrast_sel,
        kinase_token=search_input,
    )

    activity_filter = pn.widgets.RadioButtonGroup(
        name="Kinases",
        options={
            "Significant in ≥1 contrast": "significant",
            "All tested": "all_tested",
        },
        value="significant",
        button_type="default",
        margin=(12,0,0,10),
    )
    activity_heatmap_holder = pn.Column(
        _kinase_activity_heatmap(
            adata,
            results,
            profile_name=str(activity_filter.value),
        ),
        sizing_mode="stretch_width",
    )

    def _update_activity_heatmap(event) -> None:
        activity_heatmap_holder.loading = True
        try:
            activity_heatmap_holder[:] = [
                _kinase_activity_heatmap(
                    adata,
                    results,
                    profile_name=str(event.new),
                )
            ]
        finally:
            activity_heatmap_holder.loading = False

    activity_filter.param.watch(_update_activity_heatmap, "value")
    activity_info = pn.widgets.TooltipIcon(
        value=f"""
        Cells show signed Z-scores; untested kinase–contrast pairs are blank.
        Significance symbols are * q < {sign_threshold:g}, ** q < 0.01,
        and *** q < 0.001.
        Kinases use average-linkage clustering with 1 − Spearman correlation.
        Contrasts use 1 − |Spearman correlation|, so profiles that differ only
        because the contrast direction is reversed can cluster together.
        Missing values are median-filled only for linkage calculation.
        """,
        margin=0,
        styles={"z-index": "10"},
        stylesheets=[
            """
            :host {
                width: 18px !important;
                min-width: 18px !important;
                max-width: 18px !important;
                height: 18px !important;
                min-height: 18px !important;
                max-height: 18px !important;
                margin: 0 !important;
                padding: 0 !important;
                position: static !important;
            }
            """
        ],
    )
    activity_info_box = pn.Column(
        activity_info,
        width=18,
        height=18,
        min_width=18,
        min_height=18,
        margin=(10, 0, 0, 0),
        styles={
            "flex": "0 0 18px",
            "padding": "0",
            "margin": "0",
            "overflow": "visible",
            "align-self": "flex-start",
            "justify-content": "flex-start",
            "position": "relative",
            "top": "18px",
        },
    )

    species_sel = pn.widgets.Select(
        name="Species",
        options=STRING_SPECIES_OPTIONS,
        value=None,
        width=190,
    )
    category_sel = pn.widgets.Select(
        name="Category",
        options=_ENRICHMENT_CATEGORY_OPTIONS,
        value="KEGG",
        width=190,
    )
    enrichment_metric_sel = pn.widgets.Select(
        name="X axis",
        options=_ENRICHMENT_METRIC_OPTIONS,
        value="signal",
        width=190,
    )

    enrichment_default = next(
        (
            contrast
            for contrast in contrasts
            if any(
                len(identifiers) >= 2
                for identifiers in enrichment_queries.get(
                    contrast, {}
                ).values()
            )
        ),
        contrasts[0],
    )
    reference_mode = len(contrasts) > _ENRICHMENT_MAX_CONTRASTS
    active_enrichment_contrast = pn.widgets.Select(
        name="Reference contrast",
        options=contrasts,
        value=enrichment_default,
        visible=reference_mode,
        width=220,
    )
    enrichment_mapping_cache: dict[
        tuple[int, tuple[str, ...]],
        tuple[dict[str, str], str],
    ] = {}
    enrichment_cache: dict[
        tuple[int, tuple[str, ...], tuple[str, ...], bool],
        tuple[list[dict], str],
    ] = {}
    last_string_request = {"time": 0.0}

    def _rate_limited_string_call(function, *args):
        elapsed = time.monotonic() - last_string_request["time"]
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        try:
            return function(*args)
        finally:
            last_string_request["time"] = time.monotonic()

    def _fetch_enrichment(
        species: int,
        requested_contrasts: list[str],
    ) -> tuple[
        dict[str, list[dict]],
        dict[str, dict[str, str]],
        dict[str, dict[str, int]],
        dict[str, str],
    ]:
        data_by_contrast: dict[str, list[dict]] = {}
        errors_by_contrast: dict[str, dict[str, str]] = {}
        query_sizes: dict[str, dict[str, int]] = {}
        background_labels: dict[str, str] = {}

        all_tested = tuple(
            sorted(
                {
                    identifier
                    for contrast in requested_contrasts
                    for identifier in enrichment_backgrounds.get(
                        contrast, ()
                    )
                },
                key=str.casefold,
            )
        )
        mapping_key = (int(species), all_tested)
        if mapping_key not in enrichment_mapping_cache:
            try:
                mapping = _rate_limited_string_call(
                    _map_to_string_ids,
                    all_tested,
                    int(species),
                )
                enrichment_mapping_cache[mapping_key] = (mapping, "")
            except Exception as exc:
                enrichment_mapping_cache[mapping_key] = ({}, str(exc))

        mapping, mapping_error = enrichment_mapping_cache[mapping_key]
        for contrast in requested_contrasts:
            data_by_contrast[contrast] = []
            errors_by_contrast[contrast] = {}
            query_sizes[contrast] = {}
            background_ids = tuple(
                sorted(
                    {
                        mapping[identifier]
                        for identifier in enrichment_backgrounds.get(
                            contrast, ()
                        )
                        if identifier in mapping
                    }
                )
            )
            background_labels[contrast] = (
                f"{len(background_ids)} KSEA-tested kinases"
                if _ENRICHMENT_USE_TESTED_BACKGROUND
                else "STRING species proteome"
            )

            if mapping_error:
                errors_by_contrast[contrast]["mapping"] = (
                    f"STRING identifier mapping failed: {mapping_error}"
                )
                continue

            for direction in ("activated", "inhibited"):
                identifiers = enrichment_queries.get(contrast, {}).get(
                    direction, ()
                )
                query_ids = tuple(
                    sorted(
                        {
                            mapping[identifier]
                            for identifier in identifiers
                            if identifier in mapping
                        }
                    )
                )
                query_sizes[contrast][direction] = len(query_ids)
                if len(query_ids) < 2:
                    continue

                cache_key = (
                    int(species),
                    query_ids,
                    (
                        background_ids
                        if _ENRICHMENT_USE_TESTED_BACKGROUND
                        else ()
                    ),
                    _ENRICHMENT_USE_TESTED_BACKGROUND,
                )
                if cache_key not in enrichment_cache:
                    try:
                        response = _rate_limited_string_call(
                            _string_enrichment,
                            query_ids,
                            background_ids,
                            int(species),
                            _ENRICHMENT_USE_TESTED_BACKGROUND,
                        )
                        enrichment_cache[cache_key] = (
                            list(response or []),
                            "",
                        )
                    except Exception as exc:
                        enrichment_cache[cache_key] = ([], str(exc))

                data, error = enrichment_cache[cache_key]
                data_by_contrast[contrast].extend(
                    {**record, "_direction": direction}
                    for record in data
                )
                if error:
                    errors_by_contrast[contrast][direction] = error
        return (
            data_by_contrast,
            errors_by_contrast,
            query_sizes,
            background_labels,
        )

    def _activate_enrichment_contrast(_event, contrast: str) -> None:
        active_enrichment_contrast.value = contrast

    def _enrichment_view(
        species,
        category: str,
        active_contrast: str,
        metric: str,
    ) -> pn.viewable.Viewable:
        if species is None:
            return pn.pane.Alert(
                "Select a species to run STRING enrichment across contrasts.",
                alert_type="light",
                height=150,
                sizing_mode="stretch_width",
            )
        displayed_contrasts = _enrichment_contrast_page(
            contrasts,
            str(active_contrast),
        )

        (
            data_by_contrast,
            errors_by_contrast,
            query_sizes,
            background_labels,
        ) = _fetch_enrichment(
            int(species),
            displayed_contrasts,
        )

        frames = {
            contrast: _string_enrichment_frame(
                data_by_contrast.get(contrast, []),
                str(category),
                query_sizes.get(contrast, {}),
            )
            for contrast in displayed_contrasts
        }
        plot_column_width = 240
        label_margin = 300
        regular_margin = 14
        first_column_width = (
            plot_column_width + label_margin - regular_margin
        )
        first_button_offset = label_margin - regular_margin
        contrast_columns: list[pn.Column] = []
        for index, contrast in enumerate(displayed_contrasts):
            button = pn.widgets.Button(
                name=contrast.replace("_vs_", "_v_"),
                button_type=(
                    "primary" if contrast == active_contrast else "default"
                ),
                height=34,
                margin=0,
                sizing_mode="stretch_width",
            )
            button.on_click(
                lambda event, contrast=contrast: (
                    _activate_enrichment_contrast(event, contrast)
                )
            )
            button_header = pn.Row(
                *(
                    [pn.Spacer(width=first_button_offset, margin=0)]
                    if index == 0
                    else []
                ),
                button,
                height=34,
                margin=(0, 0, 8, 0),
                sizing_mode="stretch_width",
            )
            figure = _coordinated_enrichment_figure(
                contrasts=displayed_contrasts,
                frames=frames,
                errors_by_contrast=errors_by_contrast,
                query_sizes=query_sizes,
                background_labels=background_labels,
                contrast=contrast,
                active_contrast=str(active_contrast),
                metric=str(metric),
                show_term_labels=(index == 0),
            )
            plot = pn.pane.Plotly(
                figure,
                sizing_mode="stretch_width",
                config={"responsive": True},
                styles={"overflow": "hidden", "width": "100%"},
            )
            contrast_columns.append(
                pn.Column(
                    button_header,
                    plot,
                    sizing_mode="stretch_width",
                    styles={
                        "flex": (
                            f"0 0 {first_column_width}px"
                            if index == 0
                            else f"0 0 {plot_column_width}px"
                        ),
                        "min-width": (
                            f"{first_column_width}px"
                            if index == 0
                            else f"{plot_column_width}px"
                        ),
                        "max-width": (
                            f"{first_column_width}px"
                            if index == 0
                            else f"{plot_column_width}px"
                        ),
                    },
                )
            )
        return pn.Row(
            *contrast_columns,
            sizing_mode="stretch_width",
            styles={
                "align-items": "stretch",
                "gap": "8px",
                "overflow-x": "auto",
            },
        )

    enrichment_plot = pn.param.ParamFunction(
        pn.bind(
            _enrichment_view,
            species=species_sel,
            category=category_sel,
            active_contrast=active_enrichment_contrast,
            metric=enrichment_metric_sel,
        ),
        inplace=True,
        sizing_mode="stretch_width",
    )

    enrichment_background_label = (
        "KSEA-tested kinases"
        if _ENRICHMENT_USE_TESTED_BACKGROUND
        else "STRING species proteome"
    )
    enrichment_background_explanation = (
        "the contrast-specific set of all KSEA-tested kinases"
        if _ENRICHMENT_USE_TESTED_BACKGROUND
        else "STRING's default whole-species proteome"
    )
    enrichment_info = pn.widgets.TooltipIcon(
        value=f"""
        KSEA-significant kinases (q < {sign_threshold:g}) are split by their
        signed activity Z-score. STRING enrichment is run separately for
        activated kinases (Z > 0) and inhibited kinases (Z < 0), using
        {enrichment_background_explanation} as the enrichment background. All
        identifiers are first mapped to unique STRING IDs; directional queries
        with fewer than two mapped kinases are not run. The reference contrast
        defines the shared top {_ENRICHMENT_MAX_TERMS} terms, ranked from best
        to least by the selected X-axis metric. For more than
        {_ENRICHMENT_MAX_CONTRASTS} contrasts, only the page containing the
        selected reference is queried and displayed.
        """,
        margin=0,
        styles={"z-index": "10"},
    )
    enrichment_legend = pn.pane.HTML(
        """
        <div style="display:flex;align-items:center;gap:22px;padding:0 4px;"
             aria-label="Enrichment plot legend">
          <span><span style="color:#d62728;font-size:20px;vertical-align:-1px;">●</span>
            Activated (KSEA Z &gt; 0)</span>
          <span><span style="color:#1f77b4;font-size:20px;vertical-align:-1px;">●</span>
            Inhibited (KSEA Z &lt; 0)</span>
          <span style="color:#666;position:relative;top:2px;">Dot size: contributing kinases</span>
        </div>
        """,
        sizing_mode="stretch_width",
        margin=(0, 0, 6, 0),
    )

    controls = pn.Row(
        contrast_sel,
        pn.Spacer(width=15),
        make_vr(),
        pn.Spacer(width=15),
        search_input,
        pn.Row(clear_search, margin=(15, 0, 0, 0)),
        pn.Spacer(width=20),
        download_selection,
        height=70,
    )

    volcano_pane = pn.Column(
        pn.pane.Markdown("##   Kinase activity", disable_anchors=True),
        controls,
        pn.Row(
            volcano_plot,
            pn.Spacer(width=30),
            pn.Column(
                detail,
                pn.Spacer(height=15),
                substrate_heatmap,
                width=840,
                margin=(20, 20, 0, 0),
            ),
            sizing_mode="stretch_width",
            styles={"align-items": "stretch"},
        ),
        height=1310,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": "98vw",
        },
    )

    clustering_pane = pn.Column(
        pn.Row(
            pn.pane.Markdown(
                "##   Activity overview",
                disable_anchors=True,
            ),
            activity_info_box,
            pn.Spacer(width=10),
            activity_filter,
            margin=(0, 20, 0, 0),
            sizing_mode="stretch_width",
        ),
        activity_heatmap_holder,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": "98vw",
            "overflow": "hidden",
        },
    )

    enrichment_pane = pn.Column(
        pn.Row(
            pn.pane.Markdown(
                "##   Kinase pathway context",
                disable_anchors=True,
            ),
            pn.Column(
                enrichment_info,
                width=18,
                height=18,
                margin=(18, 0, 0, 0),
                styles={"overflow": "visible"},
            ),
        ),
        pn.Row(
            pn.Spacer(width=15),
            species_sel,
            *(
                [pn.Spacer(width=12), active_enrichment_contrast]
                if reference_mode
                else []
            ),
            pn.Spacer(width=12),
            category_sel,
            pn.Spacer(width=12),
            enrichment_metric_sel,
            margin=(0, 20, 0, 0),
            sizing_mode="stretch_width",
        ),
        pn.Spacer(height=10),
        enrichment_legend,
        enrichment_plot,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": "98vw",
            "overflow": "hidden",
        },
    )

    return pn.Column(
        pn.Spacer(height=10),
        kinase_summary_pane,
        pn.Spacer(height=30),
        clustering_pane,
        pn.Spacer(height=30),
        volcano_pane,
        pn.Spacer(height=30),
        enrichment_pane,
        pn.Spacer(height=30),
        sizing_mode="stretch_width",
        styles=FRAME_STYLES_TALL,
    )
