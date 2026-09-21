from pathlib import Path
import base64
import io

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH   = REPO_ROOT / "data_source" / "oee_predmaint.duckdb"
OUTPUT    = REPO_ROOT / "docs" / "reports" / "dashboard.html"

# ── Palette ──────────────────────────────────────────────────────────────────
DARK_GREY  = "#322B4B"   # document chrome: header bars, chart titles, dividers
BG_GREY    = "#F3F5F7"   # box / card backgrounds
DARK_BLUE  = "#381FA1"   # chart primary
LIGHT_BLUE = "#54C0E8"   # chart secondary (used heavily)
ACCENT_RED = "#CC0000"   # chart accent; conditional-formatting "bad" (low range)
MUTED_RED  = "#FFA3A3"   # chart, secondary red
GREEN      = "#00A84C"   # conditional-formatting "good" (high range)
AMBER      = "#FFBA3F"   # conditional-formatting "medium" (mid range)
MED_GREY   = "#8093A4"   # chart neutral
LIGHT_GREY = "#D5DCE1"   # chart neutral (gridlines, gauge track)
TEXT       = "#000000"   # body font (black)

# Chart text is sized in points so that, rendered at ~130 dpi and displayed at
# roughly natural width, it reads at about the same size as the 15px table text.
CHART_FS = 8.3

# Sequential colormap for the downtime heatmaps, kept within the brand reds.
HEAT_CMAP = LinearSegmentedColormap.from_list("brim_heat", [BG_GREY, MUTED_RED, ACCENT_RED])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": LIGHT_GREY, "font.family": "sans-serif",
    "font.size": CHART_FS, "axes.titlesize": CHART_FS, "axes.titleweight": "bold",
    "axes.labelsize": CHART_FS, "xtick.labelsize": CHART_FS, "ytick.labelsize": CHART_FS,
    "legend.fontsize": CHART_FS,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "figure.dpi": 130,
})


def chart_style(ax):
    ax.yaxis.grid(True, color=LIGHT_GREY, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)


def fig_to_b64(fig, pad=0.08):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, pad_inches=pad)
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b


# ── Load marts (OEE mart is daily: machine x shift x day) ────────────────────
con = duckdb.connect(str(DB_PATH), read_only=True)
mp = con.execute("select * from mart_oee__machine_performance").df()
da = con.execute("select * from mart_oee__downtime_analysis").df()
pm = con.execute("select * from mart_oee__pm_compliance").df()
con.close()

mp["period_date"]  = pd.to_datetime(mp["period_date"])
mp["period_month"] = pd.to_datetime(mp["period_month"])
QUALITY = float(mp["quality_rate"].iloc[0])


def plant_by(df, key):
    """Time-weighted plant-level A / P / Q / OEE aggregated to `key`."""
    g = df.groupby(key)
    a = g["run_minutes"].sum() / g["planned_production_minutes"].sum()
    p = g.apply(lambda x: (x["performance"] * x["run_minutes"]).sum() / x["run_minutes"].sum(),
                include_groups=False)
    out = pd.DataFrame({"availability": a, "performance": p})
    out["quality"] = QUALITY
    out["oee"] = out["availability"] * out["performance"] * QUALITY
    return out.sort_index()


daily_plant   = plant_by(mp, "period_date")
monthly_plant = plant_by(mp, "period_month")
cur_date = daily_plant.index.max()
cur = daily_plant.loc[cur_date]
WINDOW = f"{mp['period_date'].min():%b %Y} to {mp['period_date'].max():%b %Y}"

GAUGE_METRICS = [("OEE", "oee"), ("Availability", "availability"),
                 ("Performance", "performance"), ("Quality", "quality")]


def gauge_color(key, v):
    """Threshold band for a gauge. OEE and its components use different cuts."""
    if key == "oee":
        return GREEN if v >= 0.85 else AMBER if v >= 0.60 else ACCENT_RED
    return GREEN if v >= 0.90 else AMBER if v >= 0.80 else ACCENT_RED


def ap_band(v):
    """Availability / Performance band: green >= 90%, amber 80-90%, else red."""
    return GREEN if v >= 0.90 else AMBER if v >= 0.80 else ACCENT_RED


# ══════════════════════════════════════════════════════════════════════════════
# TOP-OF-PAGE: speedometer gauges + KPI history
# ══════════════════════════════════════════════════════════════════════════════

def gauge(value, color):
    """Three-quarter speedometer gauge: 270-degree arc, open at the bottom."""
    frac = max(0.0, min(1.0, float(value)))
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    r, w = 1.0, 0.42
    ax.add_patch(Wedge((0, 0), r, -45, 225, width=w, facecolor=LIGHT_GREY))
    ax.add_patch(Wedge((0, 0), r, 225 - frac * 270, 225, width=w, facecolor=color))
    ax.text(0, -0.05, f"{frac * 100:.1f}%", ha="center", va="center",
            fontsize=15, fontweight="bold", color=color)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.92, 1.12)
    ax.set_aspect("equal"); ax.axis("off")
    return fig_to_b64(fig, pad=0.04)


def kpi_history():
    cutoff = cur_date - pd.Timedelta(days=31)
    d = daily_plant[daily_plant.index >= cutoff]
    fig, ax = plt.subplots(figsize=(5.3, 1.9))
    ax.plot(d.index, d["oee"] * 100, color=DARK_BLUE, lw=1.8, label="OEE")
    ax.plot(d.index, d["availability"] * 100, color=LIGHT_BLUE, lw=1.5, label="Availability")
    ax.plot(d.index, d["performance"] * 100, color=MED_GREY, lw=1.5, label="Performance")
    ax.plot(d.index, d["quality"] * 100, color=LIGHT_GREY, lw=1.5, label="Quality")
    ax.set_ylim(50, 102)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=False)
    chart_style(ax)
    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PER-MACHINE BARS + HEATMAPS + BELOW-THE-FOLD CHARTS
# ══════════════════════════════════════════════════════════════════════════════

_cur_df  = mp[mp["period_date"] == cur_date]
_week_df = mp[mp["period_date"] >= cur_date - pd.Timedelta(days=6)]
_all_machines = sorted(mp["machine_id"].unique())


def _mavail(dsub):
    g = dsub.groupby("machine_id")
    return g["run_minutes"].sum() / g["planned_production_minutes"].sum()


def _mperf(dsub):
    g = dsub.groupby("machine_id")
    return g.apply(lambda x: (x["performance"] * x["run_minutes"]).sum() / x["run_minutes"].sum(),
                   include_groups=False)


def bar_by_machine(series):
    s = series.reindex(_all_machines).fillna(0.0)
    vals = (s * 100).values
    fig, ax = plt.subplots(figsize=(5.3, 2.3))
    ax.bar(range(len(s)), vals, color=[ap_band(v) for v in s.values], width=0.72)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=CHART_FS, color=TEXT)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([m.replace("MCH-", "") for m in s.index])
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_downtime_heatmap_monthly():
    st = da[da["source_system"] == "MACHINE_STATE"].copy()
    st["event_month"] = pd.to_datetime(st["event_date"]).dt.to_period("M").dt.to_timestamp()
    recent = sorted(st["event_month"].unique())[-12:]
    st = st[st["event_month"].isin(recent)]
    piv = (st.pivot_table(index="machine_id", columns="event_month",
                          values="downtime_hours", aggfunc="sum", fill_value=0).sort_index())
    return _heatmap(piv, [pd.Timestamp(c).strftime("%b'%y") for c in piv.columns])


def chart_downtime_heatmap_daily():
    st = da[da["source_system"] == "MACHINE_STATE"].copy()
    st["d"] = pd.to_datetime(st["event_date"])
    recent = sorted(st["d"].unique())[-10:]
    st = st[st["d"].isin(recent)]
    piv = (st.pivot_table(index="machine_id", columns="d",
                          values="downtime_hours", aggfunc="sum", fill_value=0).sort_index())
    return _heatmap(piv, [pd.Timestamp(c).strftime("%m/%d") for c in piv.columns])


def _heatmap(piv, xlabels):
    fig, ax = plt.subplots(figsize=(5.3, 3.6))
    im = ax.imshow(piv.values, aspect="auto", cmap=HEAT_CMAP)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Downtime (hrs)")
    cb.ax.tick_params(labelsize=CHART_FS)
    plt.tight_layout()
    return fig_to_b64(fig)


# Reliability by machine. MTBF and MTTR are drawn as separate charts so each one
# carries the dashboard's standard HTML chart title rather than a title baked
# into the image.
_rel_cmms   = da[da["source_system"] == "CMMS_REPAIR"]
_rel_runhrs = mp.groupby("machine_id")["run_minutes"].sum() / 60.0
_rel_nfail  = _rel_cmms.groupby("machine_id")["downtime_key"].count()
mtbf_by_machine = (_rel_runhrs / _rel_nfail).sort_values(ascending=False)
mttr_by_machine = _rel_cmms.groupby("machine_id")["downtime_hours"].mean().reindex(mtbf_by_machine.index)
_rel_aging  = pm.set_index("machine_id")["is_aging_asset"]


def _reliability_bars(series, fmt, xlabel):
    colors = [ACCENT_RED if _rel_aging.get(m, False) else DARK_BLUE for m in series.index]
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    ax.barh(series.index, series.values, color=colors, height=0.65)
    ax.set_xlim(0, series.max() * 1.18)
    ax.invert_yaxis()
    for yy, v in enumerate(series.values):
        ax.text(v + series.max() * 0.015, yy, fmt.format(v), va="center", color=TEXT)
    ax.set_xlabel(xlabel)
    chart_style(ax); ax.xaxis.grid(True, color=LIGHT_GREY); ax.yaxis.grid(False)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_mtbf():
    return _reliability_bars(mtbf_by_machine, "{:,.0f}", "Running hours between unplanned failures")


def chart_mttr():
    return _reliability_bars(mttr_by_machine, "{:.1f}", "Downtime hours per repair")


# ══════════════════════════════════════════════════════════════════════════════
# DOWNTIME ANALYSIS + STATUS: data prep and new charts
# ══════════════════════════════════════════════════════════════════════════════

CUR_MONTH = mp["period_month"].max()
da["event_month"] = pd.to_datetime(da["event_date"]).dt.to_period("M").dt.to_timestamp()

# Unplanned downtime for the current month, read from the CMMS repair record.
# The machine-state log is the other candidate source, but it only sees a stoppage
# during scheduled production (Mon to Sat, 06:00 to 22:00), whereas a repair's
# booked hours run wall-clock and spill past the shift window, so the two sources
# disagree by a few percent. The CMMS is used here so this card, the failure-code
# breakdown, and the top-events table all tie to one number. The mart already
# prices each downtime hour at the machine type's contribution margin (CNC Lathe
# $95, Vertical Mill $125, Horizontal Mill $145).
TOP_EVENTS_N = 8
_cmms_cur = da[(da["source_system"] == "CMMS_REPAIR") & (da["event_month"] == CUR_MONTH)]
top_events = _cmms_cur.nlargest(TOP_EVENTS_N, "downtime_hours")

mtd_hours  = float(_cmms_cur["downtime_hours"].sum())
mtd_cost   = float(_cmms_cur["downtime_cost"].sum())
mtd_events = int(len(_cmms_cur))
mtd_avg    = mtd_hours / mtd_events if mtd_events else 0.0
mtd_machines = int(_cmms_cur["machine_id"].nunique())

# Month-over-month direction on the downtime hours.
_months     = sorted(mp["period_month"].unique())
PRIOR_MONTH = _months[-2] if len(_months) > 1 else CUR_MONTH
_cmms_prev  = da[(da["source_system"] == "CMMS_REPAIR") & (da["event_month"] == PRIOR_MONTH)]
_prev_hours = float(_cmms_prev["downtime_hours"].sum())
mtd_delta   = ((mtd_hours - _prev_hours) / _prev_hours) if _prev_hours else 0.0

# Failure-code Pareto for the current month.
pareto = _cmms_cur.groupby("failure_code")["downtime_hours"].sum().sort_values(ascending=False)

# Plant time allocation by month, as a share of scheduled time, trailing 12
# months. The state log records one state at a time, so an alarm here means the
# machine faulted and stopped rather than raised a warning while still cutting.
# That time is unplanned stoppage, so it sits inside Unplanned Downtime rather
# than standing as its own operating condition. Alarm frequency is still tracked
# in its own right through the alarm-rate analysis.
_alloc = (mp.groupby("period_month")
          .agg(sched=("scheduled_minutes", "sum"), run=("run_minutes", "sum"),
               setup=("setup_minutes", "sum"), idle=("idle_minutes", "sum"),
               unpl=("unplanned_down_minutes", "sum"), pland=("planned_down_minutes", "sum"))
          .sort_index().tail(12))
_alloc["alarm"] = (_alloc["sched"]
                   - _alloc[["run", "setup", "idle", "unpl", "pland"]].sum(axis=1)).clip(lower=0)
alloc_monthly = (pd.DataFrame({
    "Running":            _alloc["run"],
    "Setup":              _alloc["setup"],
    "Idle":               _alloc["idle"],
    "Unplanned Downtime": _alloc["unpl"] + _alloc["alarm"],
    "Planned Downtime":   _alloc["pland"],
}).div(_alloc["sched"], axis=0) * 100)
ALLOC_COLORS = {"Running": DARK_BLUE, "Setup": LIGHT_BLUE, "Idle": MED_GREY,
                "Unplanned Downtime": ACCENT_RED, "Planned Downtime": MUTED_RED}
ALLOC_LABEL_FG = {"Running": "#fff", "Setup": "#000", "Idle": "#fff",
                  "Unplanned Downtime": "#fff", "Planned Downtime": "#000"}

# Unplanned downtime hours by failure code and month, trailing 12 months.
FAILURE_CODE_ORDER = ["TOOLING", "MECHANICAL", "ELECTRICAL", "OPERATOR_INDUCED", "ENVIRONMENTAL"]
FAILURE_CODE_COLORS = {"TOOLING": DARK_BLUE, "MECHANICAL": LIGHT_BLUE, "ELECTRICAL": MED_GREY,
                       "OPERATOR_INDUCED": AMBER, "ENVIRONMENTAL": MUTED_RED}
_fc = (_cmms_all := da[da["source_system"] == "CMMS_REPAIR"])
fc_monthly = (_fc.pivot_table(index="event_month", columns="failure_code",
                              values="downtime_hours", aggfunc="sum", fill_value=0)
              .sort_index().tail(12))
fc_monthly = fc_monthly[[c for c in FAILURE_CODE_ORDER if c in fc_monthly.columns]]

# Current machine state and time in that state, from the machine-state log.
PE_CSV = REPO_ROOT / "data_source" / "raw" / "machinemetrics" / "production_events.csv"
_pe = pd.read_csv(PE_CSV, usecols=["machine_id", "event_timestamp", "machine_state",
                                   "state_duration_minutes"], parse_dates=["event_timestamp"])
machine_status = {}
for _m, _d in _pe.groupby("machine_id"):
    _d = _d.sort_values("event_timestamp")
    _states = _d["machine_state"].values
    _mins   = _d["state_duration_minutes"].values
    _last   = _states[-1]
    _dur    = 0.0
    for _i in range(len(_states) - 1, -1, -1):
        if _states[_i] == _last:
            _dur += _mins[_i]
        else:
            break
    machine_status[_m] = (_last, _dur / 60.0)

STATE_STYLE = {
    "RUNNING":        ("RUNNING", GREEN,      "#fff"),
    "IDLE":           ("IDLE",    MED_GREY,   "#fff"),
    "SETUP":          ("SETUP",   DARK_BLUE,  "#fff"),
    "UNPLANNED_DOWN": ("DOWN",    ACCENT_RED, "#fff"),
    "ALARM":          ("ALARM",   AMBER,      "#000"),
    "PLANNED_DOWN":   ("PLANNED", DARK_GREY,  "#fff"),
}


def chart_time_allocation():
    """Monthly split of scheduled time, trailing 12 months. Sized for a half-page
    panel, so only segments with room carry a label; the rest read off the axis."""
    d = alloc_monthly
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    bottom = np.zeros(len(d))
    for col in d.columns:
        vals = d[col].values
        ax.bar(x, vals, bottom=bottom, width=0.68, color=ALLOC_COLORS[col], label=col)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 4:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=CHART_FS - 2.3, fontweight="bold", color=ALLOC_LABEL_FG[col])
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(m).strftime("%b'%y") for m in d.index], rotation=45, ha="right")
    ax.set_ylabel("% of scheduled time")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False,
              fontsize=CHART_FS - 1)
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_failure_code_monthly():
    """Monthly unplanned downtime stacked by failure code, trailing 12 months.
    Bar height is hours; each segment is labelled with its share of that month."""
    d = fc_monthly
    totals = d.sum(axis=1)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    bottom = np.zeros(len(d))
    for code in d.columns:
        vals = d[code].values
        ax.bar(x, vals, bottom=bottom, width=0.68, color=FAILURE_CODE_COLORS[code],
               label=code.replace("_", " ").title())
        for xi, (v, b, t) in enumerate(zip(vals, bottom, totals.values)):
            share = v / t * 100 if t else 0
            if share >= 8:
                ax.text(xi, b + v / 2, f"{share:.0f}%", ha="center", va="center",
                        fontsize=CHART_FS - 1.3, fontweight="bold",
                        color="#000" if code in ("MECHANICAL", "OPERATOR_INDUCED",
                                                 "ENVIRONMENTAL") else "#fff")
        bottom = bottom + vals
    # Month totals above each column.
    for xi, t in enumerate(totals.values):
        ax.text(xi, t + totals.max() * 0.02, f"{t:,.0f}", ha="center", va="bottom",
                fontsize=CHART_FS - 1, fontweight="bold", color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(m).strftime("%b'%y") for m in d.index], rotation=45, ha="right")
    ax.set_ylabel("Downtime (hrs)")
    ax.set_ylim(0, totals.max() * 1.16)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False,
              fontsize=CHART_FS - 1)
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


charts = {
    "kpi_history":     kpi_history(),
    "heatmap":         chart_downtime_heatmap_monthly(),
    "heatmap_daily":   chart_downtime_heatmap_daily(),
    "mtbf":            chart_mtbf(),
    "mttr":            chart_mttr(),
    "avail_cur":       bar_by_machine(_mavail(_cur_df)),
    "perf_cur":        bar_by_machine(_mperf(_cur_df)),
    "time_alloc":      chart_time_allocation(),
    "failure_code":    chart_failure_code_monthly(),
}
gauges = {k: gauge(float(cur[k]), gauge_color(k, float(cur[k]))) for _, k in GAUGE_METRICS}


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════

def delta_cell(d):
    if d > 0.0005:
        return f'<span style="color:{GREEN};">&#9650;&nbsp;{d * 100:.1f}%</span>'
    if d < -0.0005:
        return f'<span style="color:{ACCENT_RED};">&#9660;&nbsp;{abs(d) * 100:.1f}%</span>'
    return '<span>&#8594;&nbsp;0.0%</span>'


def img(k):
    return f'<img src="data:image/png;base64,{charts[k]}" style="width:100%;height:auto;display:block;">'


# Daily OEE summary: daily plant KPIs, newest first. The table carries a
# fortnight of history but opens on the most recent DAILY_ROWS_VISIBLE days; the
# rest is a scroll away. The longer trend lives in the Monthly KPI History chart.
DAILY_ROWS_TOTAL   = 14
DAILY_ROWS_VISIBLE = 6
recent_days = daily_plant.tail(DAILY_ROWS_TOTAL).iloc[::-1]
workshift_rows = ""
for day, r in recent_days.iterrows():
    workshift_rows += (
        f'<tr><td style="font-weight:600;">{pd.Timestamp(day):%m/%d/%y}</td>'
        f'<td style="text-align:right;font-weight:700;">{r["oee"]:.0%}</td>'
        f'<td style="text-align:right;">{r["availability"]:.0%}</td>'
        f'<td style="text-align:right;">{r["performance"]:.0%}</td>'
        f'<td style="text-align:right;">{r["quality"]:.0%}</td></tr>'
    )

# Monthly OEE by machine (current vs prior month), OEE column first, black text.
mv = list(monthly_plant.index)
cur_month, prior_month = mv[-1], mv[-2]


def machine_metrics(month):
    d = mp[mp["period_month"] == month].groupby("machine_id")
    a = d["run_minutes"].sum() / d["planned_production_minutes"].sum()
    p = d.apply(lambda x: (x["performance"] * x["run_minutes"]).sum() / x["run_minutes"].sum(),
                include_groups=False)
    out = pd.DataFrame({"availability": a, "performance": p})
    out["oee"] = out["availability"] * out["performance"] * QUALITY
    return out


cur_m = machine_metrics(cur_month)
prev_m = machine_metrics(prior_month)
mtype = mp.groupby("machine_id")["machine_type"].first()
mtable = cur_m.add_suffix("_c").join(prev_m.add_suffix("_p")).join(mtype.rename("machine_type")).sort_index()


def machine_cells(oee, oee_d, av, av_d, pf, pf_d):
    return (
        f'<td style="text-align:right;font-weight:700;">{oee:.0%}</td>'
        f'<td style="text-align:right;">{delta_cell(oee_d)}</td>'
        f'<td style="text-align:right;">{av:.0%}</td><td style="text-align:right;">{delta_cell(av_d)}</td>'
        f'<td style="text-align:right;">{pf:.0%}</td><td style="text-align:right;">{delta_cell(pf_d)}</td>'
        f'<td style="text-align:right;">{QUALITY:.0%}</td><td style="text-align:right;">{delta_cell(0.0)}</td>'
    )


machine_rows = ""
for mid, r in mtable.iterrows():
    machine_rows += (
        f'<tr><td style="font-weight:600;">{mid}</td><td>{r["machine_type"]}</td>'
        + machine_cells(r["oee_c"], r["oee_c"] - r["oee_p"], r["availability_c"], r["availability_c"] - r["availability_p"],
                        r["performance_c"], r["performance_c"] - r["performance_p"])
        + "</tr>"
    )

pa_c, pp_c, po_c = monthly_plant.loc[cur_month, ["availability", "performance", "oee"]]
pa_p, pp_p, po_p = monthly_plant.loc[prior_month, ["availability", "performance", "oee"]]
total_row = (
    '<tr class="total-row"><td>TOTAL PLANT</td><td></td>'
    + machine_cells(po_c, po_c - po_p, pa_c, pa_c - pa_p, pp_c, pp_c - pp_p)
    + "</tr>"
)

def gauge_cell(label, key):
    return (
        f'<div class="gcell"><div class="chart-title">{label}</div>'
        f'<img class="gauge" src="data:image/png;base64,{gauges[key]}"></div>'
    )


def cell(title, key):
    return f'<div class="chart-cell"><div class="chart-title">{title}</div>{img(key)}</div>'


def machine_status_strip():
    tiles = ""
    for m in _all_machines:
        st, hrs = machine_status.get(m, ("IDLE", 0.0))
        label, bg, fg = STATE_STYLE.get(st, (st, MED_GREY, "#fff"))
        tiles += (f'<div class="mtile" style="background:{bg};color:{fg};">'
                  f'<span class="mt-id">{m}</span>'
                  f'<span class="mt-state">{label}</span>'
                  f'<span class="mt-time">{hrs:.1f} hrs</span></div>')
    return f'<div class="mstrip">{tiles}</div>'


def downtime_cost_card():
    # More downtime rises, so an increase reads red and a decrease green.
    arrow = "&#9650;" if mtd_delta > 0.0005 else "&#9660;" if mtd_delta < -0.0005 else "&#8594;"
    dcol  = ACCENT_RED if mtd_delta > 0.0005 else GREEN if mtd_delta < -0.0005 else MED_GREY
    # Volume first, then the cost it translates into, so the card closes on the
    # dollar figure.
    rows = [
        (f"{mtd_hours:,.0f} hrs", "total unplanned downtime", DARK_GREY,
         f'<span class="dcard-delta" style="color:{dcol};">{arrow} {abs(mtd_delta):.0%} '
         f'vs {pd.Timestamp(PRIOR_MONTH):%b}</span>'),
        # How many machines the events span is a reading in its own right, so the
        # count carries emphasis inside the caption.
        (f"{mtd_events}", f"unplanned repair events across <strong>{mtd_machines}</strong> machines",
         DARK_GREY, ""),
        (f"{mtd_avg:.1f} hrs", "average duration per repair", DARK_GREY, ""),
        (f"${mtd_cost:,.0f}", "estimated contribution-margin impact", ACCENT_RED, ""),
    ]
    body = "".join(
        f'<div class="dcard-metric">'
        f'<div class="dcard-v" style="color:{colour};">{value}{extra}</div>'
        f'<div class="dcard-s">{label}</div></div>'
        for value, label, colour, extra in rows)
    return ('<div class="dcard">'
            f'<div class="dcard-label">Unplanned Downtime: {pd.Timestamp(CUR_MONTH):%b %Y}</div>'
            f'<div class="dcard-row">{body}</div></div>')


def top_events_table():
    rows = ""
    for _, r in top_events.iterrows():
        rows += (f'<tr><td style="font-weight:600;">{r["machine_id"]}</td>'
                 f'<td>{pd.Timestamp(r["event_date"]):%m/%d/%y}</td>'
                 f'<td>{str(r["failure_code"]).replace("_", " ").title()}</td>'
                 f'<td style="text-align:right;font-weight:700;">{r["downtime_hours"]:.1f}</td></tr>')
    return ('<table><thead><tr><th>Machine</th><th>Date</th><th>Failure Code</th>'
            '<th style="text-align:right;">Duration (hrs)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def threshold_legend():
    b = lambda c: f'<span class="box" style="background:{c};"></span>'
    return (
        '<div class="legend">'
        f'<strong>OEE:</strong> {b(GREEN)}Good &ge;85%&nbsp; {b(AMBER)}Fair 60-85%&nbsp; '
        f'{b(ACCENT_RED)}Poor &lt;60%&nbsp;&nbsp;&middot;&nbsp;&nbsp;'
        f'<strong>Availability / Performance / Quality:</strong> {b(GREEN)}Good &ge;90%&nbsp; '
        f'{b(AMBER)}Fair 80-90%&nbsp; {b(ACCENT_RED)}Poor &lt;80%</div>')


SYNC_SCRIPT = (
    "<script>(function(){"
    "function sync(){"
    "var s=document.querySelector('.scroll');if(!s)return;"
    "var th=s.querySelector('thead');var rows=s.querySelectorAll('tbody tr');"
    "if(!th||!rows.length)return;"
    "var h=th.getBoundingClientRect().height;"
    f"var n=Math.min({DAILY_ROWS_VISIBLE},rows.length);"
    "for(var i=0;i<n;i++){h+=rows[i].getBoundingClientRect().height;}"
    "h=Math.ceil(h)+2;"
    "s.style.height=h+'px';"
    "}"
    "window.addEventListener('load',sync);window.addEventListener('resize',sync);"
    "requestAnimationFrame(sync);"
    "var b=document.querySelector('.two-col');"
    "if(window.ResizeObserver&&b){new ResizeObserver(sync).observe(b);}"
    "})();</script>"
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analytics Dashboard: OEE</title>
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#fff; color:{TEXT}; font-size:15px; }}
  .header {{ background:{DARK_GREY}; color:#fff; padding:14px 0; }}
  .header .wrap {{ max-width:1500px; margin:0 auto; padding:0 48px; }}
  .header h1 {{ font-size:21px; font-weight:700; letter-spacing:-0.3px; }}
  .screen {{ max-width:1500px; margin:0 auto; min-height:calc(100vh - 50px); display:flex; flex-direction:column;
    padding:16px 48px 16px; gap:10px; }}
  .gauges {{ display:grid; grid-template-columns:repeat(4,1fr); gap:24px; }}
  .gcell {{ display:flex; flex-direction:column; align-items:center; text-align:center; }}
  .gcell .chart-title {{ margin:0 0 2px; text-decoration:underline; text-underline-offset:4px; }}
  img.gauge {{ width:78%; max-width:250px; }}
  .scroll {{ overflow-y:auto; min-height:0; border:1px solid {LIGHT_GREY}; border-radius:6px; }}
  .kpi-wrap {{ display:flex; }}
  .kpi-img {{ width:100%; height:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:15px; color:#000; }}
  thead th {{ position:sticky; top:0; background:{BG_GREY}; padding:9px 12px; text-align:left; font-size:15px;
    font-weight:700; color:#000; border-bottom:2px solid {LIGHT_GREY}; }}
  td {{ padding:9px 12px; border-bottom:1px solid {LIGHT_GREY}; color:#000; }}
  tbody tr:hover td {{ background:{BG_GREY}; }}
  .scorecard {{ table-layout:fixed; }}
  .scorecard th:first-child, .scorecard td:first-child {{ white-space:nowrap; }}
  .total-row td {{ background:{BG_GREY}; font-weight:700; border-top:2px solid {DARK_GREY}; border-bottom:2px solid {DARK_GREY}; }}
  .total-row:hover td {{ background:{BG_GREY}; }}
  .below {{ max-width:1500px; margin:0 auto; padding:8px 48px 48px; }}
  .chart-title {{ font-size:17px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:{DARK_GREY}; margin:22px 0 8px; }}
  .chart-grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:8px 32px; margin-bottom:6px; }}
  .chart-cell img {{ width:100%; height:auto; display:block; }}
  .chart-cell .chart-title {{ margin-top:14px; }}
  .footnote {{ font-size:13px; color:#000; margin-top:6px; font-style:italic; }}
  .legend {{ font-size:13px; color:#000; margin-top:8px; }}
  .legend.center {{ text-align:center; margin-top:4px; }}
  /* The events block lines up with the left edge of the chart above it. */
  .events-col {{ text-align:left; }}
  .events-col .inner {{ display:inline-block; text-align:left; }}
  /* The events table sizes to its content rather than stretching the column,
     which would leave a gap between the failure code and its duration. */
  .tight-wrap table {{ width:auto; }}
  .tight-wrap th, .tight-wrap td {{ white-space:nowrap; padding-right:22px; }}
  .tight-wrap th:last-child, .tight-wrap td:last-child {{ padding-right:12px; }}
  .legend .box {{ display:inline-block; width:12px; height:12px; margin-right:5px; vertical-align:middle; border-radius:2px; }}
  /* The time-allocation bar sits tight to its own title and legend. */
  .chart-title.tight {{ margin-bottom:2px; }}
  .doc {{ max-width:1500px; margin:0 auto; padding:0 48px 48px; }}
  .section-band {{ font-size:17px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
    color:{TEXT}; border-bottom:2px solid {DARK_GREY}; padding:12px 0 6px; margin:28px 0 14px; }}
  .section-band.first {{ margin-top:10px; padding-top:0; }}
  /* Section bands read black; every chart, table and card heading under them
     reads dark grey. */
  .subttl {{ font-size:17px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:{DARK_GREY}; margin:16px 0 8px; }}
  .two-col {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:32px; align-items:start; }}
  .mstrip {{ display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:6px; margin:6px 0 4px; }}
  .mtile {{ border-radius:6px; padding:8px 3px; text-align:center; display:flex; flex-direction:column; gap:1px; }}
  .mt-id {{ font-size:12px; font-weight:700; }}
  .mt-state {{ font-size:12px; font-weight:700; letter-spacing:0.3px; }}
  .mt-time {{ font-size:11px; opacity:0.92; }}
  .dcard {{ background:{BG_GREY}; border-top:3px solid {DARK_GREY}; border-radius:4px;
    padding:22px 26px 26px; margin:0 auto 18px; max-width:400px; text-align:center; }}
  /* The card starts at the top of its column, level with the table's title
     beside it. The subtitle carries a top margin, so the card takes a matching
     one to sit on the same line. */
  .two-col.spaced .dcard {{ margin-top:16px; }}
  .two-col.spaced {{ margin-bottom:10px; }}
  /* Sized, spaced and coloured to match .chart-title so the card heads like
     every other panel on the dashboard. */
  .dcard-label {{ font-size:17px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;
    color:{DARK_GREY}; margin-bottom:20px; }}
  .dcard-row {{ display:flex; flex-direction:column; gap:22px; }}
  .dcard-v {{ font-size:28px; font-weight:700; color:{DARK_GREY}; line-height:1.15; }}
  .dcard-s {{ font-size:15px; color:{MED_GREY}; margin-top:4px; }}
  /* A figure called out inside a caption reads as a takeaway: same dark grey as
     the headline values, but held at the caption's size. */
  .dcard-s strong {{ color:{DARK_GREY}; }}
  .dcard-delta {{ font-size:15px; font-weight:700; margin-left:8px; white-space:nowrap; }}
  @page {{ size:landscape; margin:8mm; }}
  @media print {{
    .header .wrap, .doc {{ max-width:none; }}
    .doc {{ padding:8px 24px 24px; }}
    .scroll {{ overflow:visible !important; max-height:none !important; min-height:0; border:none; }}
    thead th {{ position:static; }}
    tr, .chart-cell, .gcell, .mtile, .dcard, .two-col {{ break-inside:avoid; }}
    .section-band {{ break-before:page; }}
    .section-band.first {{ break-before:auto; }}
  }}
</style>
</head>
<body>
<div class="header"><div class="wrap"><h1>Analytics Dashboard: OEE</h1></div></div>
<div class="doc">

  <div class="section-band first">Plant OEE</div>
  <div class="subttl">Current Workshift: {pd.Timestamp(cur_date):%m/%d/%y}</div>
  <div class="gauges">
    {gauge_cell("OEE", "oee")}
    {gauge_cell("Availability", "availability")}
    {gauge_cell("Performance", "performance")}
    {gauge_cell("Quality (Est.)", "quality")}
  </div>
  {threshold_legend()}
  <div class="two-col">
    <div>
      <div class="subttl">OEE Summary: Daily</div>
      <div class="scroll">
        <table>
          <thead><tr><th>Day</th><th style="text-align:right;">OEE</th>
            <th style="text-align:right;">Availability</th><th style="text-align:right;">Performance</th>
            <th style="text-align:right;">Quality (Est.)</th></tr></thead>
          <tbody>{workshift_rows}</tbody>
        </table>
      </div>
    </div>
    <div>
      <div class="subttl">OEE History: Past Month</div>
      <div class="kpi-wrap"><img class="kpi-img" src="data:image/png;base64,{charts['kpi_history']}"></div>
    </div>
  </div>
  <div class="section-band">Machine-Level Breakdown</div>
  <div class="subttl">Current Machine Status</div>
  {machine_status_strip()}
  <div class="chart-grid">
    {cell("Availability by Machine: Current Workshift", "avail_cur")}
    {cell("Performance by Machine: Current Workshift", "perf_cur")}
  </div>
  <div class="chart-title">OEE by Machine: {pd.Timestamp(CUR_MONTH):%b %Y}</div>
  <table class="scorecard">
    <thead><tr>
      <th style="width:11%;">Machine</th><th style="width:11%;">Type</th>
      <th style="width:9.75%;text-align:right;">OEE</th><th style="width:9.75%;text-align:right;">&Delta;%</th>
      <th style="width:9.75%;text-align:right;">Availability</th><th style="width:9.75%;text-align:right;">&Delta;%</th>
      <th style="width:9.75%;text-align:right;">Performance</th><th style="width:9.75%;text-align:right;">&Delta;%</th>
      <th style="width:9.75%;text-align:right;">Quality (Est.)</th><th style="width:9.75%;text-align:right;">&Delta;%</th>
    </tr></thead>
    <tbody>{machine_rows}{total_row}</tbody>
  </table>
  <div class="footnote">&Delta;% compares the current month to the prior month.</div>

  <div class="section-band">Downtime</div>
  <div class="two-col spaced">
    <div>{downtime_cost_card()}</div>
    <div class="events-col">
      <div class="inner">
        <div class="subttl">Top Unplanned Downtime Events: {pd.Timestamp(CUR_MONTH):%b %Y}</div>
        <div class="tight-wrap">{top_events_table()}</div>
      </div>
    </div>
  </div>
  <div class="chart-grid">
    {cell("Plant Time Allocation: Monthly", "time_alloc")}
    {cell("Unplanned Downtime by Failure Code: Monthly", "failure_code")}
  </div>
  <div class="chart-grid">
    {cell("Unplanned Downtime Heatmap: Daily", "heatmap_daily")}
    {cell("Unplanned Downtime Heatmap: Monthly", "heatmap")}
  </div>

  <div class="section-band">Reliability</div>
  <div class="chart-grid">
    {cell("Mean Time Between Failures (MTBF) by Machine", "mtbf")}
    {cell("Mean Time To Repair (MTTR) by Machine", "mttr")}
  </div>
  <div class="footnote">MTBF is the average running hours between unplanned failures (higher is better);
    MTTR is the average downtime hours per repair (lower is better). Both are measured in hours.</div>
  <div class="legend"><span class="box" style="background:{ACCENT_RED};"></span>Aging machine (&gt;9 years).
    Computed over the full observation window ({WINDOW}).</div>

</div>
{SYNC_SCRIPT}
</body>
</html>"""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(html, encoding="utf-8")
print(f"KPI dashboard written to {OUTPUT}")
print(f"  Latest day {pd.Timestamp(cur_date):%m/%d/%y}: OEE {cur['oee']:.1%}  "
      f"A {cur['availability']:.1%}  P {cur['performance']:.1%}  Q {cur['quality']:.0%}")
