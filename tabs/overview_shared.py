from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Optional

import panel as pn

from components.overview_plots import (
    resolve_pattern_to_uniprot_ids,
    resolve_exact_list_to_uniprot_ids,
)
from utils.layout_utils import make_vr


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
            pn.pane.Markdown("##   Summary"),
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
        pn.pane.Markdown("##   Metrics", styles={"flex": "0.1", "z-index": "10"}),
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
    height: int = 530,
    margin=(0, 0, 0, 20),
    width_style: str = "98vw",
    tooltip_margin=(-475, 0, 0, -80),
) -> pn.Row:
    """
    Shared Clustering pane (PCA + MDS/UMAP) with identical layout/styling.
    MDS is used when 'X_mds' is present in adata.obsm; otherwise UMAP.
    """
    pca_pane = pn.pane.Plotly(
        plot_pca_2d(adata),
        height=500,
        sizing_mode="stretch_width",
        styles={"flex": "1"},
        margin=(0, 0, 0, -100),
    )

    if "X_mds" in adata.obsm:
        emb_fig = plot_mds_2d(adata, title="MDS")
    else:
        emb_fig = plot_umap_2d(adata, title="UMAP")

    umap_pane = pn.pane.Plotly(
        emb_fig,
        height=500,
        sizing_mode="stretch_width",
        styles={"flex": "1"},
    )

    cluster_info = pn.widgets.TooltipIcon(
        value="""
        Using left-censored QC data.
        Results may differ from
        analysis of processed data.
        Multidimensional Scaling uses
        correlation distances.
        """,
        margin=tooltip_margin,
        styles={"z-index": "10"},
    )

    return pn.Row(
        pn.pane.Markdown("##   Clustering", styles={"flex": "0.15", "z-index": "10"}),
        cluster_info,
        pca_pane,
        make_vr(),
        pn.Spacer(width=60),
        umap_pane,
        make_vr(),
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
) -> Callable[[], None]:
    """
    Ensure cohort changes update selection exporter state immediately.
    Watches:
      - pattern input (search_input_group.value)
      - file text holder (file_text_widget.value)
      - optional search field selector (search_field_sel.value)
      - clear button click
    Returns the push function for manual calls if needed.
    """
    def _push(_=None) -> None:
        on_cohort_ids(list(group_ids_selected() or []))

    search_input_group.param.watch(_push, "value")
    file_text_widget.param.watch(_push, "value")
    if search_field_sel is not None:
        search_field_sel.param.watch(_push, "value")
    clear_btn.on_click(lambda _e: _push())
    return _push
