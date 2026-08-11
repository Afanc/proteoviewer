from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from tabs.overview_shared import bind_uirevision
from utils.layout_utils import FRAME_STYLES_TALL, make_vr
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
    height: int = 1350,
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
                "KSEA z-score: %{customdata[3]:.3f}<br>"
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


def _contrast_sample_indices(adata, contrast: str) -> np.ndarray:
    if "_vs_" not in str(contrast):
        return np.array([], dtype=int)

    condition_a, condition_b = str(contrast).split("_vs_", 1)
    conditions = _text_series(adata.obs["CONDITION"]).to_numpy()
    return np.concatenate(
        [
            np.flatnonzero(conditions == condition_a),
            np.flatnonzero(conditions == condition_b),
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
                "Select a kinase to show its contributing phosphosite profiles."
            ),
            title="Contributing phosphosites",
            width=410,
            collapsible=False,
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
                "No contributing phosphosite profiles are available for this contrast."
            ),
            title="Contributing phosphosites",
            width=410,
            collapsible=False,
        )

    matrix = adata.X[sample_indices, :][:, feature_indices]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    absolute = np.asarray(matrix, dtype=float).T

    with np.errstate(invalid="ignore"):
        centered = absolute - np.nanmean(absolute, axis=1, keepdims=True)

    contrast_names = [str(value) for value in adata.uns.get("contrast_names", [])]
    if contrast in contrast_names and "log2fc" in adata.varm:
        contrast_index = contrast_names.index(contrast)
        log2fc = np.asarray(
            adata.varm["log2fc"][feature_indices, contrast_index],
            dtype=float,
        ).ravel()
        sort_values = np.where(np.isfinite(log2fc), log2fc, -np.inf)
        order = np.argsort(-sort_values, kind="stable")
        site_ids = [site_ids[index] for index in order]
        centered = centered[order]
        absolute = absolute[order]

    sample_names = adata.obs_names[sample_indices].astype(str).tolist()
    finite = np.abs(centered[np.isfinite(centered)])
    color_limit = float(np.max(finite)) if finite.size else 1.0
    if color_limit <= 0.0:
        color_limit = 1.0

    fig = go.Figure(
        go.Heatmap(
            z=centered,
            x=sample_names,
            y=site_ids,
            customdata=absolute,
            colorscale="RdBu_r",
            zmin=-color_limit,
            zmax=color_limit,
            zmid=0.0,
            colorbar={"title": "Deviation"},
            hovertemplate=(
                "Phosphosite: %{y}<br>"
                "Sample: %{x}<br>"
                "Deviation from site mean: %{z:.3f}<br>"
                "Final intensity: %{customdata:.3f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={"text": f"Substrate profiles — {kinase}", "x": 0.5},
        height=max(330, min(650, 150 + 18 * len(site_ids))),
        margin={"l": 110, "r": 20, "t": 55, "b": 90},
        xaxis={"title": "Samples", "tickangle": -45},
        yaxis={"title": "Phosphosites", "autorange": "reversed", "showticklabels":False},
    )

    return pn.Card(
        pn.pane.Plotly(
            fig,
            sizing_mode="stretch_width",
            config={"responsive": True},
        ),
        title="Contributing phosphosites",
        width=800,
        collapsible=False,
        styles={
            "background": "#f9f9f9",
            "border-radius": "8px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "padding": "8px",
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
                "Click a kinase in the volcano or use **Search Kinase** to inspect it."
            ),
            title="Kinase details",
            width=410,
            collapsible=False,
            styles={
                "background": "#f9f9f9",
                "border-radius": "8px",
                "box-shadow": "3px 3px 5px #bcbcbc",
                "padding": "8px",
            },
        )

    kinase = str(row.get("kinase", "") or row["kinase_id"])
    gene = str(row.get("kinase_gene", "") or "n/a")
    uniprot = str(row.get("kinase_uniprot", "") or "")
    tested = bool(_tested_mask(pd.Series([row["tested"]]))[0])
    reason = str(row.get("reason", "") or "").replace("_", " ")
    test_status = (
        "calculated"
        if tested
        else f"not calculated ({reason or 'ineligible'})"
    )

    Number = pn.indicators.Number
    effect = Number(
        name="Mean shift",
        value=float(row["effect"]),
        format="{value:.3f}",
        default_color="purple",
        font_size="12pt",
        styles={"flex": "1"},
    )
    activity = Number(
        name="KSEA z-score",
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
        f"### {kinase}\n**Gene:** {gene}",
        margin=(0, 0, 5, 0),
    )
    stats = pn.pane.Markdown(
        "\n".join(
            [
                f"- Mean substrate log2FC: **{float(row['kinase_mean_log2fc']):.3f}**",
                f"- Global mean log2FC: **{float(row['global_mean_log2fc']):.3f}**",
                f"- p-value: **{float(row['pvalue']):.3e}**",
                f"- KSEA test: **{test_status}**",
            ]
        ),
        margin=(5, 0, 0, 0),
    )

    footer = pn.pane.HTML("")
    if uniprot:
        footer = pn.pane.HTML(
            "<span style='font-size: 12px;'>"
            f"<a href='https://www.uniprot.org/uniprotkb/{uniprot}/entry' "
            "target='_blank' rel='noopener'>UniProt Entry</a>"
            "</span>"
        )

    return pn.Card(
        header,
        pn.Row(effect, activity, sizing_mode="stretch_width"),
        pn.Row(qvalue, substrates, sizing_mode="stretch_width"),
        stats,
        footer,
        title="Kinase details",
        width=800,
        collapsible=False,
        styles={
            "background": "#f9f9f9",
            "border-radius": "8px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "padding": "8px",
        },
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
    sign_threshold = float(analysis.get("sign_threshold", 0.05))

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
        height=1350,
    )
    volcano_dmap = bind_uirevision(
        volcano_dmap,
        contrast_sel,
        prefix="kinase-volcano",
    )
    volcano_plot = pn.pane.Plotly(
        volcano_dmap,
        height=1350,
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
                width=1200,
                margin=(20, 20, 0, 0),
            ),
            sizing_mode="stretch_width",
            styles={"align-items": "stretch"},
        ),
        height=1500,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": "98vw",
        },
    )

    return pn.Column(
        pn.Spacer(height=10),
        volcano_pane,
        pn.Spacer(height=30),
        sizing_mode="stretch_width",
        styles=FRAME_STYLES_TALL,
    )
