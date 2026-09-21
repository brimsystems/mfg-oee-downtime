"""
brand.py
Shared BRIM styling kit for the ML HTML deliverables. Mirrors the Case 01 report
layout (TOC sidebar, section blocks, KPI cards, data tables, model card, status
block, callouts) so the two cases read as one portfolio, restyled to the current
brand palette.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # noqa: F401 (re-exported for report use)

# ── Palette (brand style guide) ──────────────────────────────────────────────
DARK_GREY  = "#322B4B"   # document chrome: header bars, titles, dividers
BG_GREY    = "#F3F5F7"   # box / card backgrounds
DARK_BLUE  = "#381FA1"   # chart primary
LIGHT_BLUE = "#54C0E8"   # chart secondary
ACCENT_RED = "#CC0000"   # chart accent; conditional-formatting "bad"
MUTED_RED  = "#FFA3A3"   # chart secondary red
GREEN      = "#00A84C"   # conditional-formatting "good"
AMBER      = "#FFBA3F"   # conditional-formatting "medium"
MED_GREY   = "#8093A4"   # chart neutral
LIGHT_GREY = "#D5DCE1"   # chart neutral (gridlines)
TEXT       = "#000000"   # body font

CHART_W, CHART_H, CHART_H_T, CHART_DPI = 8.2, 3.8, 4.5, 150

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": LIGHT_GREY, "font.family": "sans-serif",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "text.color": TEXT, "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "figure.dpi": CHART_DPI,
})


def chart_style(ax):
    ax.yaxis.grid(True, color=LIGHT_GREY, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)


def make_fig(h=None):
    return plt.subplots(figsize=(CHART_W, h or CHART_H))


def b64(figure) -> str:
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode()
    plt.close(figure)
    return out


# ── HTML component helpers ───────────────────────────────────────────────────
def chart(title, image_b64, caption=""):
    # Captions under charts are intentionally never rendered: the chart title
    # carries the "what" and the lead-in paragraph carries the "so what". The
    # caption argument is accepted but ignored so existing call sites still work.
    if image_b64 is None:
        return ('<div class="chart-wrap"><p style="color:#999;text-align:center;'
                'padding:20px;">Chart not available.</p></div>')
    t = f'<div class="chart-title">{title}</div>' if title else ""
    return (f'<div class="chart-wrap">{t}<img src="data:image/png;base64,{image_b64}" '
            f'style="width:100%;height:auto;display:block;"></div>')


def section(id_, label, title):
    # Subsection labels carry a dotted number (e.g. "Section 2.1"); render them
    # with a lighter treatment so they read as a rung below the main sections.
    cls = "section-title-block sub" if "." in label else "section-title-block"
    return (f'<div class="{cls}" id="{id_}">'
            f'<div class="section-label">{label}</div>'
            f'<h2 class="section-title">{title}</h2></div>')


def kpi_card(value, label, sub="", color=None):
    color = color or DARK_GREY
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">{value}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>')


def kpi_row(*cards):
    return f'<div class="kpi-row">{"".join(cards)}</div>'


def data_table(headers, rows, right=None):
    right = set(right or [])
    th = "".join(f'<th style="text-align:{"right" if i in right else "left"};">{h}</th>'
                 for i, h in enumerate(headers))
    body = ""
    for r in rows:
        tds = "".join(c if str(c).startswith("<td") else
                      f'<td style="text-align:{"right" if i in right else "left"};">{c}</td>'
                      for i, c in enumerate(r))
        body += f"<tr>{tds}</tr>"
    return f'<table class="data-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def badge(text, color):
    return f'<span class="tier-badge" style="background:{color};">{text}</span>'


def callout(html):
    return f'<div class="callout">{html}</div>'


def _css():
    return f"""
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:#fff; color:{TEXT}; font-size:16px; line-height:1.7; }}
  .page-header {{ background:{DARK_GREY}; color:#fff; padding:14px 40px; }}
  .page-header h1 {{ font-size:21px; font-weight:700; letter-spacing:-0.3px; }}
  .page-header .sub {{ font-size:13px; color:{LIGHT_GREY}; margin-top:3px; }}
  .layout {{ display:flex; max-width:1200px; margin:0 auto; padding:0 40px; }}
  .toc {{ width:210px; flex-shrink:0; padding:36px 20px 40px 0; position:sticky; top:0;
    height:100vh; overflow-y:auto; border-right:1px solid {LIGHT_GREY}; }}
  .toc-title {{ font-size:10px; letter-spacing:2px; text-transform:uppercase; color:{MED_GREY};
    margin-bottom:14px; font-weight:700; }}
  .toc a {{ display:block; font-size:13px; color:{MED_GREY}; text-decoration:none;
    padding:4px 0 4px 10px; border-left:2px solid transparent; line-height:1.4; }}
  .toc a:hover {{ color:{DARK_GREY}; border-left-color:{DARK_GREY}; }}
  .toc a.sub {{ font-size:12px; padding-left:20px; }}
  .toc hr {{ border:none; border-top:1px solid {LIGHT_GREY}; margin:8px 0; }}
  .content {{ flex:1; padding:36px 0 80px 52px; max-width:880px; }}
  .section-title-block {{ margin:46px 0 22px; padding-bottom:12px; border-bottom:2px solid {DARK_GREY}; }}
  .content > .section-title-block:first-child {{ margin-top:8px; }}
  .section-label {{ font-size:10px; letter-spacing:2px; text-transform:uppercase; color:{DARK_GREY};
    font-weight:700; margin-bottom:4px; }}
  .section-title {{ font-size:22px; font-weight:700; color:{DARK_GREY}; }}
  .section-title-block.sub {{ margin:34px 0 14px; padding-bottom:0; border-bottom:none;
    border-left:3px solid {LIGHT_BLUE}; padding-left:12px; }}
  .section-title-block.sub .section-label {{ color:{MED_GREY}; margin-bottom:2px; }}
  .section-title-block.sub .section-title {{ font-size:16px; font-weight:600; letter-spacing:.2px; }}
  p {{ margin-bottom:16px; }}
  code {{ background:{BG_GREY}; padding:1px 5px; border-radius:3px; font-size:14px; }}
  .kpi-row {{ display:flex; gap:16px; margin:22px 0; flex-wrap:wrap; }}
  .kpi-card {{ flex:1; min-width:150px; background:{BG_GREY}; border-radius:8px;
    padding:18px 22px 14px; border-bottom:4px solid {DARK_GREY}; }}
  .kpi-value {{ font-size:30px; font-weight:700; line-height:1; margin-bottom:6px; }}
  .kpi-label {{ font-size:12px; color:{MED_GREY}; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }}
  .kpi-sub {{ font-size:12px; color:{MED_GREY}; margin-top:4px; }}
  .chart-title {{ font-size:15px; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
    color:{DARK_GREY}; text-align:center; margin-bottom:8px; }}
  .chart-wrap {{ margin:18px 0; border:1px solid {LIGHT_GREY}; border-radius:4px; padding:12px; }}
  .chart-caption {{ font-size:12px; color:{MED_GREY}; margin-top:8px; text-align:center; font-style:italic; }}
  .chart-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:18px 0; }}
  .chart-pair .chart-wrap {{ margin:0; }}
  .data-table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:14px; color:{TEXT}; }}
  .data-table th {{ background:{BG_GREY}; padding:10px 12px; text-align:left; font-size:12px;
    font-weight:700; text-transform:uppercase; letter-spacing:.5px; color:{DARK_GREY}; border-bottom:2px solid {LIGHT_GREY}; }}
  .data-table td {{ padding:9px 12px; border-bottom:1px solid {LIGHT_GREY}; color:{TEXT}; }}
  .data-table tr:hover td {{ background:{BG_GREY}; }}
  .tier-badge {{ display:inline-block; padding:2px 8px; border-radius:3px; color:#fff; font-size:11px;
    font-weight:700; text-transform:uppercase; letter-spacing:.5px; }}
  .callout {{ background:{BG_GREY}; border-left:4px solid {DARK_GREY}; padding:14px 20px; margin:18px 0;
    font-size:15px; }}
  .callout strong {{ color:{DARK_GREY}; }}
  .model-card {{ background:{BG_GREY}; border-top:3px solid {DARK_GREY}; padding:22px 26px; margin-bottom:22px; }}
  .model-card-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px 32px; font-size:14px; }}
  .mc-label {{ color:{MED_GREY}; font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
  .mc-value {{ color:{TEXT}; font-weight:500; margin-top:2px; }}
  .limitation-list {{ margin:8px 0 20px 20px; }}
  .limitation-list li {{ margin-bottom:8px; font-size:15px; line-height:1.6; }}
  .status-block {{ border:2px solid; border-radius:6px; overflow:hidden; margin:20px 0; }}
  .status-header {{ display:flex; align-items:center; gap:12px; padding:14px 20px; color:#fff; }}
  .status-icon {{ font-size:20px; font-weight:700; }}
  .status-label {{ font-size:17px; font-weight:700; letter-spacing:.3px; }}
  .status-body {{ padding:20px; display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  .status-meta {{ display:flex; flex-direction:column; gap:8px; }}
  .status-meta > div {{ display:flex; gap:10px; font-size:14px; }}
  .meta-label {{ color:{MED_GREY}; font-weight:700; font-size:12px; text-transform:uppercase;
    letter-spacing:.5px; width:150px; flex-shrink:0; padding-top:1px; }}
  .meta-val {{ font-weight:500; }}
  .trigger-list {{ margin:6px 0 0 18px; }}
  .trigger-list li {{ font-size:14px; margin-bottom:5px; line-height:1.5; }}
"""


def page(title, subtitle, toc, body):
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title><style>{_css()}</style></head>
<body>
<div class="page-header"><h1>{title}</h1>{sub_html}</div>
<div class="layout">
  <nav class="toc"><div class="toc-title">Contents</div>{toc}</nav>
  <main class="content">{body}</main>
</div>
</body></html>"""
