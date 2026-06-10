from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models.widgets.tables import NumberFormatter

from components.string_links import get_string_functional_enrichment
from components.overview_plots import (
    resolve_pattern_to_uniprot_ids,
    resolve_exact_list_to_uniprot_ids,
)
from components.plot_utils import (
    get_volcano_classification_masks,
    get_nrsc_alignment_mask,
)
from utils.layout_utils import make_vr

STRING_SPECIES_OPTIONS = {
    "Select species": None,
    "Homo sapiens": 9606,
    "Mus musculus": 10090,
    "Arabidopsis thaliana": 3702,
    "Saccharomyces cerevisiae": 4932,
    "Drosophila melanogaster": 7227,
    "Escherichia coli K-12": 511145,
    "Pseudomonas aeruginosa PAO1": 208964,
}


def metadata_list(value) -> list:
    """
    Normalize AnnData/HDF5 metadata values to a plain Python list.

    AnnData may round-trip empty YAML lists as empty NumPy arrays. Those
    cannot be used directly in truth-value checks, e.g. ``if value``.
    """
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, np.ndarray):
        return [x for x in value.tolist() if str(x).strip()]

    if isinstance(value, (list, tuple, set, pd.Index, pd.Series)):
        return [x for x in list(value) if str(x).strip()]

    return [value]


def make_id_sort_toggle(*, margin=(20, 0, 0, 20), width=170) -> pn.widgets.RadioButtonGroup:
    return pn.widgets.RadioButtonGroup(
        name="Order",
        options=["By condition", "By sample"],
        value="By condition",
        button_type="default",
        width=width,
        margin=margin,
        styles={"z-index": "10"},
    )


def sort_arg(mode: str) -> str:
    return "condition" if mode == "By condition" else "sample"


def fmt_files_list(paths: Iterable[str], max_items: int = 6) -> list[str]:
    paths = list(paths or [])
    if not paths:
        return []
    head = [f"  - {os.path.basename(str(p))}" for p in paths[:max_items]]
    rest = len(paths) - max_items
    if rest > 0:
        head.append(f"  - … (+{rest} more)")
    return head


def make_intro_pane(
    *,
    summary_pane: pn.viewable.Viewable,
    id_sort_toggle: pn.widgets.RadioButtonGroup,
    hist_plot_pane: pn.viewable.Viewable,
    hist_plot_margin=(0, 20, 0, -190),
    height=530,
) -> pn.Row:
    # Keep layout parity with both implementations; caller controls margins.
    if hasattr(hist_plot_pane, "margin"):
        hist_plot_pane.margin = hist_plot_margin

    return pn.Row(
        pn.Column(
            pn.pane.Markdown("##   Summary", disable_anchors=True),
            summary_pane,
            styles={"flex": "0.32", "min-width": "0"},
        ),
        make_vr(),
        pn.Spacer(width=20),
        id_sort_toggle,
        hist_plot_pane,
        height=height,
        margin=(0, 0, 0, 20),
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": "98vw",
        },
    )


def make_min_meas_select(
    *,
    adata,
    contrast_sel: pn.widgets.Select,
    allow_zero: bool,
    name: str,
    width: int,
    default_value_label: Optional[str] = None,
):
    def _min_max_reps_for_contrast(contrast: str) -> tuple[int, int]:
        grp1, grp2 = contrast.split("_vs_")
        n1 = int((adata.obs["CONDITION"] == grp1).sum())
        n2 = int((adata.obs["CONDITION"] == grp2).sum())
        return min(n1, n2), max(n1, n2)

    def _mk_opts(mx: int) -> dict[str, int]:
        start = 0 if allow_zero else 1
        return {f"≥{i}": i for i in range(start, mx + 1)}

    _init_min, _init_max = _min_max_reps_for_contrast(contrast_sel.value)
    opts = _mk_opts(_init_max)

    # Preserve the original defaulting behavior by letting the caller specify a label.
    if default_value_label is None:
        default_value_label = f"≥{_init_min}" if not allow_zero else f"≥{_init_max}"

    sel = pn.widgets.Select(
        name=name,
        options=list(opts.keys()),
        value=default_value_label,
        width=width,
    )

    def value_fn(label: str) -> int:
        return opts.get(label, 0 if allow_zero else 1)

    def _refresh(event):
        nonlocal opts
        _mn, mx = _min_max_reps_for_contrast(event.new)
        opts = _mk_opts(mx)
        sel.options = list(opts.keys())

        # Clamp / preserve where possible.
        cur = sel.value or ("≥0" if allow_zero else "≥1")
        if cur not in opts:
            try:
                cur_i = int(str(cur).lstrip("≥") or ("0" if allow_zero else "1"))
            except Exception:
                cur_i = 0 if allow_zero else 1
            cur_i = min(mx, max(0 if allow_zero else 1, cur_i))
            sel.value = f"≥{cur_i}"

    contrast_sel.param.watch(_refresh, "value")
    return sel, value_fn


def make_toggle_label_updater(
    *,
    contrast_sel: pn.widgets.Select,
    show_imp_cond1: pn.widgets.Checkbox,
    show_imp_cond2: pn.widgets.Checkbox,
) -> Callable[[], None]:
    def _update(_event=None):
        grp1, grp2 = contrast_sel.value.split("_vs_")
        show_imp_cond1.name = f"▲ Fully Imputed in {grp1}"
        show_imp_cond2.name = f"▼ Fully Imputed in {grp2}"

    _update()
    contrast_sel.param.watch(_update, "value")
    return _update


def make_cohort_inspector_widgets(
    *,
    adata,
    search_field_options: list[str],
    search_field_default: str,
    pattern_placeholder: str,
    status_margin=(-10, 0, 0, 0),
    clear_btn_width=90,
    file_btn_width=200,
    pattern_width=200,
    field_width=130,
    field_margin=(2, 0, 0, 0),
    pattern_margin=None,
):
    """
    Shared “Cohort Inspector” widgets:
    - Search field selector
    - Pattern input
    - FileInput with hot-swap reset
    - Either-or semantics (file overrides pattern; typing pattern clears file)
    - Status pane (“**N matches**”) with identical show/hide behavior
    - Returns group_ids_selected pn.bind and a push hook-friendly _file_text
    """
    search_field_sel = pn.widgets.Select(
        name="Search Field",
        options=search_field_options,
        value=search_field_default,
        width=field_width,
        styles={"z-index": "10"},
        margin=field_margin,
    )

    search_input_group = pn.widgets.TextInput(
        name="Pattern or File",
        placeholder=pattern_placeholder,
        width=pattern_width,
        styles={"z-index": "10"},
    )
    if pattern_margin is not None:
        search_input_group.margin = pattern_margin

    def _group_ids(pattern, field):
        try:
            return sorted(resolve_pattern_to_uniprot_ids(adata, field, pattern))
        except Exception:
            return []

    group_ids_dmap = pn.bind(_group_ids, search_input_group, search_field_sel)

    _file_text = pn.widgets.TextAreaInput(visible=False)
    cohort_filename = pn.widgets.StaticText(name="", value="")
    file_holder = pn.Column()

    def _new_file_input():
        fi = pn.widgets.FileInput(accept=".txt,.csv,.tsv", multiple=False, width=file_btn_width)

        def _on_change(event):
            b = event.new or b""
            try:
                txt = b.decode("utf-8", errors="ignore")
            except Exception:
                txt = ""
            _file_text.value = txt
            cohort_filename.value = fi.filename or ""
            if txt.strip():
                try:
                    search_input_group.value = ""
                    if hasattr(search_input_group, "value_input"):
                        search_input_group.value_input = ""
                except Exception:
                    pass

        fi.param.watch(_on_change, "value")
        return fi

    file_upload = _new_file_input()
    file_holder.objects = [file_upload]

    def _parse_file_text(text: str) -> list[str]:
        if not text:
            return []
        tokens = re.split(r"[,\t;\r\n\s]+", text)
        return sorted({t.strip() for t in tokens if t and not t.isspace()})

    def _file_ids(file_text: str, field: str):
        items = _parse_file_text(file_text or "")
        try:
            return sorted(resolve_exact_list_to_uniprot_ids(adata, field, items))
        except Exception:
            return []

    group_file_ids_dmap = pn.bind(_file_ids, _file_text, search_field_sel)

    def _either(ids_pat, ids_file, pat_text, file_text):
        use_file = bool((file_text or "").strip())
        return ids_file if use_file else ids_pat

    group_ids_selected = pn.bind(_either, group_ids_dmap, group_file_ids_dmap, search_input_group, _file_text)

    def _fmt_status(ids_pat, ids_file, fname, pat_text, file_text):
        if not (pat_text and str(pat_text).strip()) and not (file_text and str(file_text).strip()):
            return ""
        active_from_file = bool((file_text or "").strip())
        n = len(ids_file or []) if active_from_file else len(ids_pat or [])
        label = "match" if n == 1 else "matches"
        _ = os.path.basename(str(fname)) if (active_from_file and fname) else ""
        return f"**{n} {label}**"

    status_md = pn.bind(_fmt_status, group_ids_dmap, group_file_ids_dmap, cohort_filename, search_input_group, _file_text)
    status_pane = pn.pane.Markdown(status_md, margin=status_margin, align="center")

    def _has_query(pat_text, file_text):
        return bool((pat_text or "").strip()) or bool((file_text or "").strip())

    status_pane.visible = pn.bind(_has_query, search_input_group, _file_text)

    clear_all = pn.widgets.Button(name="Clear", width=clear_btn_width)

    def _on_clear_all(_event=None):
        search_input_group.value = ""
        _file_text.value = ""
        cohort_filename.value = ""
        new_file = _new_file_input()
        file_holder.objects = [new_file]
        nonlocal file_upload
        file_upload = new_file

    clear_all.on_click(_on_clear_all)

    def _on_pattern_change(event):
        val = event.new or ""
        if str(val).strip():
            _file_text.value = ""
            cohort_filename.value = ""
            new_file = _new_file_input()
            file_holder.objects = [new_file]
            nonlocal file_upload
            file_upload = new_file

    search_input_group.param.watch(_on_pattern_change, "value")

    return (
        search_field_sel,
        search_input_group,
        file_holder,
        clear_all,
        status_pane,
        group_ids_selected,
        _file_text,
        cohort_filename,
    )


def filter_feature_ids_to_visible_volcano(
    *,
    adata,
    contrast: str,
    min_nonimp_per_cond: int,
    min_consistent_peptides: int,
    min_nrsc_alignment: float,
    show_measured: bool,
    show_imp_cond1: bool,
    show_imp_cond2: bool,
    feature_ids: list[str],
) -> list[str]:
    """
    Restrict an exported cohort/ID list to the subset currently visible in the volcano plot.

    Visibility is defined by:
      - post-test missingness classification (Observed/Imputed grp1/Imputed grp2)
      - min_nonimp_per_cond threshold
      - consistent pep/prec threshold
      - the three visibility toggles (show_measured/show_imp_cond1/show_imp_cond2)

    This uses the same classification logic as plot_volcanoes() via get_volcano_classification_masks().
    """
    if not feature_ids:
        return []

    measured, imp1, imp2 = get_volcano_classification_masks(
        adata=adata,
        contrast=str(contrast),
        min_nonimp_per_cond=int(min_nonimp_per_cond or 0),
        min_consistent_peptides=int(min_consistent_peptides or 0),
    )

    align_keep = get_nrsc_alignment_mask(
        adata=adata,
        contrast=str(contrast),
        min_nrsc_alignment=float(min_nrsc_alignment or 0.0),
    )
    measured &= align_keep
    imp1 &= align_keep
    imp2 &= align_keep

    visible = (measured & bool(show_measured)) | (imp1 & bool(show_imp_cond1)) | (imp2 & bool(show_imp_cond2))
    visible_ids = set(adata.var_names[visible].astype(str).tolist())

    # Preserve caller order
    return [str(x) for x in feature_ids if str(x) in visible_ids]


def make_metrics_pane(
    *,
    cv_fig,
    rmad_fig,
    height: int = 530,
    margin=(0, 0, 0, 20),
    width_style: str = "98vw",
) -> pn.Row:
    """
    Shared Metrics pane (RMAD + CV violins) with identical layout/styling.
    Caller provides Plotly figures.
    """
    rmad_pane = pn.pane.Plotly(
        rmad_fig,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex": "1"},
        config={"responsive": True},
        margin=(0, 0, 0, -100),
    )
    cv_pane = pn.pane.Plotly(
        cv_fig,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex": "1"},
        config={"responsive": True},
    )

    return pn.Row(
        pn.pane.Markdown("##   Metrics", styles={"flex": "0.1", "z-index": "10"}, disable_anchors=True),
        rmad_pane,
        pn.Spacer(width=25),
        make_vr(),
        pn.Spacer(width=25),
        cv_pane,
        pn.Spacer(width=50),
        height=height,
        margin=margin,
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": width_style,
        },
    )


def make_clustering_pane(
    *,
    adata,
    plot_pca_2d,
    plot_mds_2d,
    plot_umap_2d,
    height: int = 570,
    plot_height: int = 500,
    margin=(0, 0, 0, 20),
    width_style: str = "98vw",
    tooltip_margin=(-475, 0, 0, -80),
    color_key: str = "CONDITION",
) -> pn.Row:
    """
    Shared Clustering pane (PCA + MDS/UMAP) with identical layout/styling.
    MDS is used when 'X_mds' is present in adata.obsm; otherwise UMAP.
    """
    pca_pc_sel = pn.widgets.Select(
        name="PCA axes",
        options={
            "PC1 vs PC2": (1, 2),
            "PC1 vs PC3": (1, 3),
            "PC2 vs PC3": (2, 3),
        },
        value=(1, 2),
        width=110,
        margin=(0, 0, 0, 10),
        styles={"z-index": "10"},
    )
    show_pca_ellipses = pn.widgets.Checkbox(
        name="PCA 95% CI",
        value=True,
        margin=(0, 0, 0, 10),
        styles={"z-index": "10"},
    )
    show_mds_ellipses = pn.widgets.Checkbox(
        name="MDS 95% CI",
        value=True,
        margin=(0, 0, 0, 10),
        visible=("X_mds" in adata.obsm),
        styles={"z-index": "10"},
    )

    def _pca_fig(pc, show_ellipses):
        return plot_pca_2d(
            adata=adata,
            pc=tuple(pc),
            color_key=color_key,
            show_ellipses=bool(show_ellipses),
            width=None,
            height=plot_height,
        )

    pca_fig = pn.bind(_pca_fig, pca_pc_sel, show_pca_ellipses)

    pca_pane = pn.pane.Plotly(
        pca_fig,
        sizing_mode="stretch_width",
        styles={"flex": "1"},
        margin=(0, 0, 0, 0),
        config={"responsive": True},
    )

    def _emb_fig(show_ellipses):
        if "X_mds" in adata.obsm:
            return plot_mds_2d(
                adata=adata,
                color_key=color_key,
                title="MDS",
                show_ellipses=bool(show_ellipses),
                width=None,
                height=plot_height,
            )
        else:
            return plot_umap_2d(
                adata=adata,
                color_key=color_key,
                title="UMAP",
                width=None,
                height=plot_height,
            )

    emb_fig = pn.bind(_emb_fig, show_mds_ellipses)

    emb_pane = pn.pane.Plotly(
        emb_fig,
        sizing_mode="stretch_width",
        margin=(0,0,0,0),
        config={"responsive": True},
        styles={"flex": "1"},
    )

    cluster_info = pn.widgets.TooltipIcon(
        value="""
        Using left-censored QC data.
        Results may differ from
        analysis of processed data.
        Multidimensional Scaling uses
        correlation distances.
        Ellipses show the 95% confidence
        of each displayed group in the embedding.
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
    cluster_info_box = pn.Column(
        cluster_info,
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

    return pn.Column(
        pn.Row(
            pn.pane.Markdown("##   Clustering", styles={"flex": "1", "z-index": "10"}, disable_anchors=True),
            cluster_info_box,
        ),
        pn.Spacer(height=10),
        pn.Row(
            pca_pc_sel,
            pn.Spacer(width=30),
            pn.Column(
                show_pca_ellipses,
                pn.Spacer(height=10),
                show_mds_ellipses,
                margin=(10,0,0,0),
            ),
            margin=(0,0,0,10),
        ),
        pn.Row(
            pca_pane,
            pn.Spacer(width=30),
            make_vr(),
            pn.Spacer(width=30),
            emb_pane,
            #margin=(0,0,0,0),
            margin=(-60, 0, 0, 20),
            sizing_mode="stretch_width",
            styles={"flex": "1", "align-items": "stretch"},
        ),
        height=height,
        margin=margin,
        sizing_mode="stretch_width",
        styles={
            "border-radius": "15px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "width": width_style,
        },
    )

def bind_uirevision(fig_dmap, contrast_sel, *, prefix: str = "volcano"):
    """
    Attach a stable uirevision key so Plotly preserves zoom/selection per-contrast.
    Identical behavior to the per-tab inline helper.
    """
    def _with_uirevision(fig, contrast):
        fig.update_layout(uirevision=f"{prefix}-{contrast}")
        return fig
    return pn.bind(_with_uirevision, fig_dmap, contrast_sel)


def make_min_precursor_select(
    *,
    max_prec_options: int,
    title_token: str,
    width: int = 80,
    default_label: str = "≥0",
):
    """
    Shared 'Consistent pep/prec' widget.
    Returns (SelectWidget, value_fn(label)->int).
    Keeps the exact label→int mapping from both tabs.
    """
    min_prec_options = {f"≥{i}": i for i in range(0, max_prec_options)}
    sel = pn.widgets.Select(
        name=f"Consistent {title_token}",
        options=list(min_prec_options.keys()),
        value=default_label,
        width=width,
    )
    def value_fn(label: str) -> int:
        return min_prec_options[label]
    return sel, value_fn

def wire_cohort_export_updates(
    *,
    group_ids_selected,
    on_cohort_ids: Callable[[list[str]], None],
    search_input_group: pn.widgets.TextInput,
    file_text_widget: pn.widgets.TextAreaInput,
    clear_btn: pn.widgets.Button,
    search_field_sel: Optional[pn.widgets.Select] = None,
    transform_ids: Optional[Callable[[list[str]], list[str]]] = None,
    refresh_controls: Optional[Sequence[tuple[object, str]]] = None,
) -> Callable[[], None]:
    """
    Ensure cohort changes update selection exporter state immediately.
    Watches:
      - pattern input (search_input_group.value)
      - file text holder (file_text_widget.value)
      - optional search field selector (search_field_sel.value)
      - clear button click
      - optional additional controls affecting visible export rows
    Returns the push function for manual calls if needed.
    """
    def _push(_=None) -> None:
        #on_cohort_ids(list(group_ids_selected() or []))
        ids = list(group_ids_selected() or [])
        if transform_ids is not None:
            ids = list(transform_ids(ids) or [])
        on_cohort_ids(ids)

    search_input_group.param.watch(_push, "value")
    file_text_widget.param.watch(_push, "value")
    if search_field_sel is not None:
        search_field_sel.param.watch(_push, "value")
    clear_btn.on_click(lambda _e: _push())

    for obj, param_name in (refresh_controls or []):
        obj.param.watch(_push, param_name)

    return _push

def make_string_species_select(width: int = 190) -> pn.widgets.Select:
    sel = pn.widgets.Select(
        name="Species (for STRING)",
        options=STRING_SPECIES_OPTIONS,
        value=None,
        width=width,
    )
    sel.visible = False
    return sel


def _first_token(value) -> str:
    s = str(value or "").strip()
    return s.split(";", 1)[0].strip() if ";" in s else s


def feature_ids_to_string_proteins(
    adata,
    feature_ids: list[str],
    *,
    prefer_parent: bool = False,
    parent_col: str = "PARENT_PROTEIN",
) -> list[str]:
    """
    Convert selected volcano feature ids to protein identifiers suitable for STRING.

    For protein-level workflows, feature ids are usually already protein ids.
    For peptide, phospho, and PELSA workflows, parent_col is preferred when available.
    """
    if not feature_ids:
        return []

    ids = [str(x) for x in feature_ids]
    var_names = set(map(str, adata.var_names))
    missing = [x for x in ids if x not in var_names]
    if missing:
        raise ValueError(
            "STRING enrichment selection contains feature ids not found in adata.var_names. "
            f"Examples={missing[:10]!r}"
        )

    out: list[str] = []
    seen: set[str] = set()

    use_parent = bool(prefer_parent and parent_col in adata.var.columns)
    values = adata.var.reindex(ids)[parent_col].astype(str) if use_parent else pd.Series(ids, index=ids)

    for value in values.astype(str):
        for token in str(value).split(";"):
            protein = token.strip()
            if not protein or protein.lower() in {"nan", "none"}:
                continue
            if protein not in seen:
                seen.add(protein)
                out.append(protein)

    return out


def make_string_category_table(data: list[dict], category: str, title: str):
    df = pd.DataFrame(data)
    if df.empty or "category" not in df.columns:
        return pn.pane.Markdown(
            f"**{title}**  \nNo enriched terms.",
            margin=(0, 0, 0, 0),
        )

    sub = df[df["category"].astype(str) == category].copy()
    if sub.empty:
        return pn.pane.Markdown(
            f"**{title}**  \nNo enriched terms.",
            margin=(0, 0, 0, 0),
        )

    required = {
        "term",
        "description",
        "number_of_genes",
        "number_of_genes_in_background",
        "p_value",
        "fdr",
    }
    missing = sorted(required - set(sub.columns))
    if missing:
        raise ValueError(
            "STRING enrichment result is missing required fields. "
            f"Missing={missing}; present={list(sub.columns)!r}"
        )

    sub["fdr_num"] = pd.to_numeric(sub["fdr"], errors="raise")
    sub = (
        sub[sub["fdr_num"] <= 0.05]
        .sort_values("fdr_num", ascending=True, kind="mergesort")
        .copy()
    )

    if sub.empty:
        return pn.pane.Markdown(
            f"**{title}**  \nNo significant terms at FDR ≤ 0.05.",
            margin=(0, 0, 0, 0),
        )

    disp = pd.DataFrame({
        "Term": sub["term"].astype(str).values,
        "Description": sub["description"].astype(str).values,
        "Count in network": (
            pd.to_numeric(sub["number_of_genes"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .values
        ),
        "Count in background": (
            pd.to_numeric(sub["number_of_genes_in_background"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .values
        ),
        "FDR": sub["fdr_num"].map(lambda x: f"{x:.3e}").values,
    })

    tbl = pn.widgets.Tabulator(
        disp,
        show_index=False,
        disabled=True,
        layout="fit_columns",
        height=190,
        sizing_mode="stretch_width",
        pagination=None,
        selectable=True,
        sortable=True,
        widths={
            "Term": 110,
            "Description": 350,
            "Count in network": 100,
            "Count in background": 100,
            "FDR": 90,
        },
        configuration={
            "rowHeight": 30,
            "columnDefaults": {"editor": False, "headerSort": False},
        },
        margin=(-5, 8, 8, 8),
    )

    safe_title = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(title).strip()).strip("_")
    download_btn = pn.widgets.Button(
        name="Download",
        button_type="success",
        width=90,
        margin=(0, 0, 0, 0),
    )

    def _download_string_table(_event):
        tbl.download(filename=f"proteoflux_string_{safe_title or 'table'}.csv")

    download_btn.on_click(_download_string_table)

    header = pn.Row(
        pn.pane.Markdown(f"**{title}**", styles={"font-size": "15px", "padding": "0"}),
        pn.Spacer(sizing_mode="stretch_width"),
        download_btn,
        sizing_mode="stretch_width",
    )

    return pn.Card(
        header,
        tbl,
        collapsible=False,
        hide_header=True,
        sizing_mode="stretch_width",
        styles={"background": "#f9f9f9", "border-radius": "8px", "padding": "6px"},
    )


def make_string_selected_feature_table(
    adata,
    feature_ids: list[str],
    *,
    title: str = "Selected features",
    parent_col: str = "PARENT_PROTEIN",
    contrast: str | None = None,
) -> pn.viewable.Viewable:
    if not feature_ids:
        return pn.Spacer(width=820, height=0)

    ids = [str(x) for x in feature_ids]
    var = adata.var.reindex(ids)

    def _contrast_index() -> int | None:
        if contrast is None:
            return None
        try:
            contrast_names = list(map(str, adata.uns["contrast_names"]))
            return contrast_names.index(str(contrast))
        except Exception:
            return None

    def _varm_vector(candidates: tuple[str, ...]) -> pd.Series:
        j = _contrast_index()
        if j is None:
            return pd.Series(np.nan, index=ids, dtype=float)

        pos = adata.var.index.get_indexer(ids)
        valid = pos >= 0

        for key in candidates:
            if key not in adata.varm:
                continue
            arr = np.asarray(adata.varm[key])
            if arr.ndim != 2 or arr.shape[1] <= j:
                continue

            out = np.full(len(ids), np.nan, dtype=float)
            out[valid] = arr[pos[valid], j]
            return pd.Series(out, index=ids, dtype=float)

        return pd.Series(np.nan, index=ids, dtype=float)

    disp = pd.DataFrame({"Feature": ids})
    disp["log₂ FC"] = _varm_vector(("log2fc", "log2_fc", "logFC", "logfc")).values
    disp["q-value"] = _varm_vector(("qvalue", "qval", "q", "q_ebayes")).map(
        lambda x: f"{x:.3e}" if np.isfinite(x) else "nan"
    ).values


    if "GENE_NAMES" in adata.var.columns:
        disp["Gene"] = var["GENE_NAMES"].astype(str).values
    if parent_col in adata.var.columns:
        disp["Parent protein"] = var[parent_col].astype(str).values

    tbl = pn.widgets.Tabulator(
        disp,
        show_index=False,
        disabled=True,
        selectable=False,
        sortable=True,
        layout="fit_columns",
        height=min(max(len(disp), 1), 5) * 30 + 30,
        pagination=None,
        sizing_mode="stretch_width",
        formatters={
            "log₂ FC": NumberFormatter(format="0.000"),
        },
        configuration={
            "rowHeight": 30,
            "columnDefaults": {"editor": False, "headerSort": True},
        },
        margin=(0, 8, 8, 8),
    )

    return pn.Card(
        pn.pane.Markdown(f"**{title}**", styles={"font-size": "15px", "padding": "0"}),
        tbl,
        collapsible=False,
        hide_header=True,
        sizing_mode="stretch_width",
        styles={"background": "#f9f9f9", "border-radius": "8px", "padding": "6px"},
    )


def make_string_enrichment_card(
    *,
    selected_feature_ids: list[str],
    proteins: list[str],
    species: int | None,
    selected_table,
    width: int = 820,
):
    if not selected_feature_ids:
        return pn.Spacer(width=width, height=320)

    if species is None:
        return pn.pane.Markdown(
            "**STRING enrichment**  \nSelect a species to run enrichment on the current box/lasso selection.",
            styles={"background": "#f9f9f9", "padding": "10px", "border-radius": "8px"},
            width=width,
            height=120,
            margin=(0, 0, 0, 0),
        )

    if len(proteins) < 2:
        return pn.pane.Markdown(
            "**STRING enrichment**  \nSelect features from at least two parent proteins. "
            "STRING expands single-protein queries, so single-protein enrichment is not shown here.",
            styles={"background": "#f9f9f9", "padding": "10px", "border-radius": "8px"},
            sizing_mode="stretch_width",
            width=width,
            height=120,
            margin=(0, 0, 0, 0),
        )

    data = get_string_functional_enrichment(tuple(sorted(proteins)), int(species))
    if not data:
        return pn.pane.Markdown(
            f"**STRING enrichment**  \nNo enriched terms returned for {len(proteins)} parent proteins.",
            styles={"background": "#f9f9f9", "padding": "10px", "border-radius": "8px"},
            sizing_mode="stretch_width",
            width=width,
            height=120,
            margin=(0, 0, 0, 0),
        )

    return pn.Card(
        pn.pane.Markdown(
            f"### STRING enrichment  | {len(selected_feature_ids)} features → {len(proteins)} parent proteins",
            margin=(0, 0, 5, 0),
        ),
        selected_table,
        pn.Spacer(height=10),
        make_string_category_table(data, "Process", "GO Biological Process"),
        pn.Spacer(height=10),
        make_string_category_table(data, "Function", "GO Molecular Function"),
        pn.Spacer(height=10),
        make_string_category_table(data, "Component", "GO Cellular Component"),
        collapsible=False,
        hide_header=True,
        width=width,
        styles={
            "background": "#f9f9f9",
            "border-radius": "8px",
            "box-shadow": "3px 3px 5px #bcbcbc",
            "padding": "10px",
        },
    )
