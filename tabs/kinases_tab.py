from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

import scipy.cluster.hierarchy as sch
from plotly.subplots import make_subplots

from tabs.overview_shared import bind_uirevision
from utils.layout_utils import (
    FRAME_STYLES_TALL,
    make_vr,
    make_hr,
    )
from utils.session_state import SessionState
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
                "Kinase: %{customdata[0]}<br>"
                "Gene: %{customdata[1]}<br>"
                "UniProt: %{customdata[2]}<br>"
                "Mean shift: %{x:.3f}<br>"
                "Z-score: %{customdata[3]:.3f}<br>"
                "p-value: %{customdata[4]:.3e}<br>"
                "q-value: %{customdata[5]:.3e}<br>"
                "Substrates: %{customdata[6]:.0f}<br>"
                "Mean substrate log2FC: %{customdata[7]:.3f}<br>"
                "Global mean log2FC: %{customdata[8]:.3f}"
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
                "Select a kinase to show its contributing phosphosite profiles."
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

    kinase_id = str(row["kinase_id"])
    kinase = str(row.get("kinase", "") or kinase_id)
    link_mask = (
        _text_series(substrates["contrast"]).eq(str(contrast))
        & _text_series(substrates["kinase_id"]).eq(kinase_id)
    )
    site_ids = (
        _text_series(substrates.loc[link_mask, "phosphosite_id"])
        .drop_duplicates()
        .tolist()
    )

    feature_indices = adata.var_names.astype(str).get_indexer(site_ids)
    present = feature_indices >= 0
    site_ids = [site for site, keep in zip(site_ids, present) if keep]
    feature_indices = feature_indices[present]
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
        (len(site_ids), len(sample_names), 3),
        dtype=object,
    )
    customdata[:, :, 0] = absolute
    customdata[:, :, 1] = sample_conditions[None, :]
    customdata[:, :, 2] = site_log2fc[:, None]

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
        return pn.Card(
            pn.pane.Markdown(
                "**Kinase details**",
                styles={
                    "font-size": "16px",
                    "padding": "0",
                    "line-height": "0px",
                },
            ),
            make_hr(),
            pn.pane.Markdown(
                "Click a kinase in the volcano or use **Search Kinase** to inspect it."
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

    kinase = _text_value(
        row.get("kinase", ""),
        str(row["kinase_id"]),
    )
    gene = _text_value(row.get("kinase_gene", ""), "n/a")
    uniprot = _text_value(row.get("kinase_uniprot", ""))

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
        footer_right = pn.pane.HTML(
            "<span style='font-size: 12px;'>"
            "🔗 "
            f"<a href='https://www.uniprot.org/uniprotkb/{uniprot}/entry' "
            "target='_blank' rel='noopener'>UniProt Entry</a>"
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


def _contrast_options(adata, results: pd.DataFrame) -> list[str]:
    available = set(results["contrast"].astype(str))
    configured = [str(value) for value in adata.uns.get("contrast_names", [])]
    ordered = [contrast for contrast in configured if contrast in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


@log_time("Preparing Kinases Tab")
def kinases_tab(state: SessionState):
    adata = state.adata
    results = _kinase_results(adata)
    substrates = _kinase_substrates(adata)
    contrasts = _contrast_options(adata, results)
    if not contrasts:
        return pn.pane.Markdown("No kinase activity contrasts are available.")

    analysis = adata.uns.get("analysis", {}) or {}
    clustering = _kinase_activity(adata).get("clustering", {}) or {}
    sign_threshold_value = clustering.get(
        "sign_threshold",
        analysis.get("sign_threshold", 0.05),
    )
    sign_threshold = float(
        0.05
        if sign_threshold_value is None
        else sign_threshold_value
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

    method = str(_kinase_activity(adata).get("method", "ksea")).upper()
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
    activity_heatmap = pn.bind(
        _kinase_activity_heatmap,
        adata=adata,
        results=results,
        profile_name=activity_filter,
    )
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
    controls = pn.Row(
        contrast_sel,
        pn.Spacer(width=15),
        make_vr(),
        pn.Spacer(width=15),
        search_input,
        pn.Row(clear_search, margin=(15, 0, 0, 0)),
        pn.Spacer(width=20),
        pn.pane.Markdown(
            f"**Method:** {method}  \n**q-value threshold:** {sign_threshold:g}",
            margin=(0, 0, 0, 0),
        ),
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
        activity_heatmap,
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
        clustering_pane,
        pn.Spacer(height=30),
        volcano_pane,
        pn.Spacer(height=30),
        sizing_mode="stretch_width",
        styles=FRAME_STYLES_TALL,
    )
