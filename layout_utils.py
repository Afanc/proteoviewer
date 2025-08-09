import panel as pn

COMMON_PLOTLY_KW = dict(
    sizing_mode="stretch_width",
    config={'responsive': True},
)

SECTION_STYLES = {
    'border-radius': '15px',
    'box-shadow':    '3px 3px 5px #bcbcbc',
}

DEFAULT_MARGIN = (0, 0, 0, 20)
DEFAULT_SIZING = "stretch_width"

FRAME_STYLES = {
    "max-height": "calc(100vh - 160px)",  # tweak the 160px to your header/controls height
    "overflow": "auto",
}

def make_row(*components,
             width: str = "95vw",
             height: int = None,
             margin: tuple = DEFAULT_MARGIN):
    """
    A styled Row for one “sub‐section” (no header), with configurable width.
    """
    return pn.Row(
        *components,
        height=height,
        margin=margin,
        sizing_mode=DEFAULT_SIZING,
        styles={
            **SECTION_STYLES,
            'background': 'white',
            'width': width
        }
    )

def make_section(header: str, row: pn.Row,
                 background: str = 'white',
                 width: str = '95vw',
                 height: int = None):
    """
    Wraps a header + one Row into a Column “section”.
    """
    hdr = pn.pane.Markdown(f"##   {header}", styles={"flex": "0.05"})
    return pn.Column(
        hdr,
        row,
        height=height,
        margin=DEFAULT_MARGIN,
        sizing_mode=DEFAULT_SIZING,
        styles={**SECTION_STYLES, 'background': background, 'width': width}
    )

def plotly_section(fig, height, flex=None, background="white"):
    styles = {
        'background': background,
        **({'flex': flex} if flex else {}),
    }
    return pn.pane.Plotly(fig, height=height, styles=styles, **COMMON_PLOTLY_KW)


def make_vr(color="#ccc", margin="6px 0"):
    return pn.Spacer(
        width=1,
        sizing_mode="stretch_height",
        styles={"background": "#ccc", "margin": "6px 0"}
    )

def make_hr(color="#ccc", margin="6px 0"):
    return pn.Spacer(
        height=1,
        sizing_mode="stretch_width",
        styles={"background": color, "margin": margin}
    )

