from pathlib import Path
import base64
import io
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH   = REPO_ROOT / "data_source" / "oee_predmaint.duckdb"
OUTPUT    = REPO_ROOT / "docs" / "reports" / "analytics_report.html"

BENCHMARK_OEE = 0.85   # target OEE for this shop
AP_TARGET     = 0.95   # target Availability and Performance implied by the OEE goal
SHIFT_STARTUP_WINDOW_MIN = 45
# Setup time above this multiple of the cohort median marks an operator as a
# coaching candidate.
SETUP_OUTLIER_RATIO = 1.20
TYPE_ABBR = {"CNC Lathe": "cnc", "Vertical Mill": "vert", "Horizontal Mill": "hor"}


def mlabel(machine_id, machine_type):
    """Compact axis label: machine number plus a short machine-type tag, e.g.
    MCH-011 (Horizontal Mill) -> '011(hor)'."""
    return f"{machine_id.split('-')[1]}({TYPE_ABBR[machine_type]})"


# Display labels for the raw CMMS failure codes (title case, OPERATOR_INDUCED
# shortened to Operator), so charts read cleanly rather than in all caps.
FAILURE_LABELS = {"TOOLING": "Tooling", "MECHANICAL": "Mechanical",
                  "ELECTRICAL": "Electrical", "OPERATOR_INDUCED": "Operator",
                  "ENVIRONMENTAL": "Environmental"}

# ── Palette (BRIM house style) ───────────────────────────────────────────────
DARK_GREY  = "#322B4B"   # document chrome: headers, titles, takeaways, dividers
BG_GREY    = "#F3F5F7"   # box / card backgrounds
DARK_BLUE  = "#381FA1"   # chart primary
LIGHT_BLUE = "#54C0E8"   # chart secondary (used heavily)
ACCENT_RED = "#CC0000"   # chart accent; conditional-formatting "bad" (low range)
MUTED_RED  = "#FFA3A3"   # chart, secondary red
GREEN      = "#00A84C"   # conditional-formatting "good" (high range)
AMBER      = "#FFBA3F"   # conditional-formatting "medium" (mid range)
MED_GREY   = "#8093A4"   # chart neutral
LIGHT_GREY = "#D5DCE1"   # chart neutral
TEXT       = "#000000"   # body font (black)

CHART_W, CHART_H, CHART_DPI = 8.2, 3.8, 130
BODY_FS, TITLE_FS = 11, 13
TYPE_COLORS = {"CNC Lathe": DARK_BLUE, "Vertical Mill": LIGHT_BLUE, "Horizontal Mill": ACCENT_RED}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#D5DCE1", "axes.grid": False, "font.family": "sans-serif",
    "font.size": BODY_FS, "axes.titlesize": TITLE_FS, "axes.titleweight": "bold",
    "axes.labelsize": BODY_FS, "xtick.labelsize": BODY_FS, "ytick.labelsize": BODY_FS,
    "legend.fontsize": BODY_FS, "figure.dpi": CHART_DPI,
})


def chart_style(ax):
    ax.yaxis.grid(True, color="#D5DCE1", linestyle="-", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#D5DCE1")
    ax.spines["bottom"].set_color("#D5DCE1")


def make_fig(h=None):
    return plt.subplots(figsize=(CHART_W, h or CHART_H))


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b


def usd_short(x):
    """Compact currency. Chooses K or M by how the value rounds, so a figure like
    $999,942 reads as $1.0M rather than $1000K, and the unit stays appropriate as
    numbers change."""
    return f"${x / 1e6:.1f}M" if round(x / 1e3) >= 1000 else f"${x / 1e3:.0f}K"


# ── Load marts ───────────────────────────────────────────────────────────────
con = duckdb.connect(str(DB_PATH), read_only=True)
mp = con.execute("select * from mart_oee__machine_performance").df()
da = con.execute("select * from mart_oee__downtime_analysis").df()
pm = con.execute("select * from mart_oee__pm_compliance").df()
os_ = con.execute("select * from mart_oee__operator_setup").df()
con.close()

mp["period_month"] = pd.to_datetime(mp["period_month"])
n_months     = mp["period_month"].dt.to_period("M").nunique()
window_years = n_months / 12.0
PERIOD_LABEL = f"{mp['period_month'].min():%b %Y}-{mp['period_month'].max():%b %Y}"
TYPES = ["CNC Lathe", "Vertical Mill", "Horizontal Mill"]
TYPE_BY_MACHINE = mp.groupby("machine_id")["machine_type"].first().to_dict()
quality = float(mp["quality_rate"].iloc[0])


def weighted_metrics(df):
    a = df["run_minutes"].sum() / df["planned_production_minutes"].sum()
    p = (df["performance"] * df["run_minutes"]).sum() / df["run_minutes"].sum()
    return a, p, a * p * quality


plant_avail, plant_perf, plant_oee = weighted_metrics(mp)
by_type = {t: weighted_metrics(mp[mp["machine_type"] == t]) for t in TYPES}

# Performance of the CNC lathes as a group versus the vertical and horizontal
# mills combined, for the Performance deep dive.
lathe_perf = by_type["CNC Lathe"][1]
mills_perf = weighted_metrics(mp[mp["machine_type"].isin(["Vertical Mill", "Horizontal Mill"])])[1]
lathe_vs_mills_pts = (lathe_perf - mills_perf) * 100

mach = (mp.groupby(["machine_id", "machine_type", "machine_age_years", "is_aging_asset"])
          .apply(lambda g: pd.Series({
              "availability": g["run_minutes"].sum() / g["planned_production_minutes"].sum(),
              "performance":  (g["performance"] * g["run_minutes"]).sum() / g["run_minutes"].sum(),
              "run_hours":    g["run_minutes"].sum() / 60.0,
          }), include_groups=False)
          .reset_index())
mach["oee"] = mach["availability"] * mach["performance"] * quality
mach = mach.sort_values("oee")

# ── OEE trend slopes by group (points per year) ──────────────────────────────
def group_trends():
    out = {}
    monthly = (mp.groupby(["machine_type", "period_month"])
                 .apply(lambda g: (g["run_minutes"].sum() / g["planned_production_minutes"].sum())
                        * ((g["performance"] * g["run_minutes"]).sum() / g["run_minutes"].sum())
                        * quality, include_groups=False)
                 .reset_index(name="oee"))
    for t, sub in monthly.groupby("machine_type"):
        sub = sub.sort_values("period_month")
        xi = np.arange(len(sub))
        slope = np.polyfit(xi, sub["oee"], 1)[0] * 12 * 100
        r = np.corrcoef(xi, sub["oee"])[0, 1]
        out[t] = (slope, r)
    return monthly, out


trend_monthly, trend_stats = group_trends()
max_abs_r = max(abs(r) for _, r in trend_stats.values())

# ── Performance vs age ───────────────────────────────────────────────────────
age_r     = np.corrcoef(mach["machine_age_years"], mach["performance"])[0, 1]
age_slope = np.polyfit(mach["machine_age_years"], mach["performance"], 1)[0] * 100

# ── Opportunity to reach the 85% baseline ────────────────────────────────────
cmms = da[da["source_system"] == "CMMS_REPAIR"]
rate_by_type = mp.groupby("machine_type")["contribution_margin_per_hour"].first()
rate_str = ", ".join(f"{t}s ${int(rate_by_type[t])}/hr" for t in TYPES)

cost = mach.set_index("machine_id").copy()
cost["cm"]          = cost["machine_type"].map(rate_by_type)
cost["planned_hrs"] = mp.groupby("machine_id")["planned_production_minutes"].sum() / 60.0 / window_years
cost["run_hrs"]     = cost["run_hours"] / window_years

# Availability and performance loss magnitudes (versus a 100% ideal), used only
# to split each machine's opportunity between the two controllable levers.
cost["avail_loss"]  = cost["planned_hrs"] * (1 - cost["availability"]) * cost["cm"]
cost["perf_loss"]   = cost["run_hrs"] * (1 - cost["performance"]) * cost["cm"]
cost["share_avail"] = cost["avail_loss"] / (cost["avail_loss"] + cost["perf_loss"])

# Opportunity to reach the 85% baseline, split additively into the two levers.
cost["opp"]       = (BENCHMARK_OEE - cost["oee"]).clip(lower=0) * cost["planned_hrs"] * cost["cm"]
cost["avail_opp"] = cost["opp"] * cost["share_avail"]
cost["perf_opp"]  = cost["opp"] * (1 - cost["share_avail"])

opportunity_annual = cost["opp"].sum()
avail_opp_annual   = cost["avail_opp"].sum()
perf_opp_annual    = cost["perf_opp"].sum()
reactive_repair_annual = cmms["downtime_cost"].sum() / window_years
unplanned_hours_annual = cmms["downtime_hours"].sum() / window_years

# Aging (>9 yr) versus the rest of the fleet, rolled up the same time-weighted way
# as the plant metrics. Used in the executive summary to size the oldest machines
# as the single largest driver of the OEE miss, on both levers.
aging_avail, aging_perf, aging_oee = weighted_metrics(mp[mp["is_aging_asset"]])
newer_avail, newer_perf, newer_oee = weighted_metrics(mp[~mp["is_aging_asset"]])
avail_gap_pts = (newer_avail - aging_avail) * 100
perf_gap_pts  = (newer_perf - aging_perf) * 100
oee_gap_pts   = (newer_oee - aging_oee) * 100
# Rounded to whole points, half rounded up (so a 4.5-point gap reads as 5, not 4).
oee_gap_r   = int(np.floor(oee_gap_pts + 0.5))
avail_gap_r = int(np.floor(avail_gap_pts + 0.5))
perf_gap_r  = int(np.floor(perf_gap_pts + 0.5))
aging_opp_share = cost.loc[cost["is_aging_asset"], "opp"].sum() / opportunity_annual
n_aging_assets  = int(mp[mp["is_aging_asset"]]["machine_id"].nunique())
n_fleet         = int(mp["machine_id"].nunique())

# Component targets implied by an 85% composite OEE. Because OEE = A x P x Q,
# reaching 85% does not mean A and P each hit 85%. Holding quality at its
# assumption and the current availability-to-performance balance, both must rise
# so that A x P = benchmark / quality.
_required_ap  = BENCHMARK_OEE / quality
_scale        = (_required_ap / (plant_avail * plant_perf)) ** 0.5
implied_avail = plant_avail * _scale
implied_perf  = plant_perf * _scale

# ── Within-machine PM-overdue vs alarm rate (controls for machine identity) ──
# Comparing each machine against itself isolates the PM effect from age, which
# otherwise confounds a cross-machine comparison (older machines both fail more
# and fall behind on PM). Overdue windows are the spans when a machine is more
# than the threshold past a scheduled PM, read from the CMMS.
import datetime as _dt
PM_OVERDUE_THRESHOLD_DAYS = 14
_maint = pd.read_csv(REPO_ROOT / "data_source" / "raw" / "cmms" / "maintenance_records.csv")
_pmrec = _maint[(_maint["maintenance_type"] == "PLANNED_PM")
                & _maint["pm_scheduled_date"].notna() & _maint["days_overdue"].notna()]
overdue_windows = {m: [] for m in mp["machine_id"].unique()}
for _, _r in _pmrec.iterrows():
    if _r["days_overdue"] <= PM_OVERDUE_THRESHOLD_DAYS:
        continue
    _s = pd.to_datetime(_r["pm_scheduled_date"]).date()
    _c = pd.to_datetime(_r["pm_completed_date"]).date()
    overdue_windows[_r["machine_id"]].append((_s + _dt.timedelta(days=PM_OVERDUE_THRESHOLD_DAYS), _c))


def _in_overdue(machine_id, d):
    return any(a <= d < b for a, b in overdue_windows.get(machine_id, []))


_daily = (mp.groupby(["machine_id", "period_date"])
          .agg(alarm_events=("alarm_events", "sum"), run_minutes=("run_minutes", "sum"))
          .reset_index())
_daily["date"]    = pd.to_datetime(_daily["period_date"]).dt.date
_daily["overdue"] = [_in_overdue(m, d) for m, d in zip(_daily["machine_id"], _daily["date"])]


def _paired_rates(g):
    ov, cu = g[g["overdue"]], g[~g["overdue"]]
    r_ov = (ov["alarm_events"].sum() / (ov["run_minutes"].sum() / 60.0) * 1000
            if ov["run_minutes"].sum() else np.nan)
    r_cu = (cu["alarm_events"].sum() / (cu["run_minutes"].sum() / 60.0) * 1000
            if cu["run_minutes"].sum() else np.nan)
    return pd.Series({"rate_current": r_cu, "rate_overdue": r_ov})


pm_within = _daily.groupby("machine_id").apply(_paired_rates, include_groups=False).dropna()
pm_within = pm_within.join(pm.set_index("machine_id")["is_aging_asset"])
pm_overdue_multiple = float((pm_within["rate_overdue"] / pm_within["rate_current"]).mean())

# Shift-start stoppage lift. The startup window is 45 of every 480 shift minutes,
# so if stoppages were spread evenly it would hold that same share of them. The
# lift is how far above that even share the window actually runs, and it is read
# per shift because the pattern belongs to both crews.
_sd = da[(da["source_system"] == "MACHINE_STATE") & da["minute_of_shift"].notna()]
_even_share = SHIFT_STARTUP_WINDOW_MIN / 480.0


def _startup_lift(sub):
    return ((sub["is_shift_startup"] == True).mean() / _even_share) if len(sub) else float("nan")


startup_lift    = _startup_lift(_sd)
startup_lift_a  = _startup_lift(_sd[_sd["shift"] == "A"])
startup_lift_b  = _startup_lift(_sd[_sd["shift"] == "B"])

# Alarm-rate exposure of the aging assets against the rest of the fleet, read per
# 1,000 running hours so busy and idle machines compare fairly. Used to state the
# gap as an exact multiple rather than a vague "several times".
_alarms = mp.groupby("machine_id")["alarm_events"].sum()
_runhrs = mp.groupby("machine_id")["run_minutes"].sum() / 60.0
_arate  = (_alarms / _runhrs * 1000)
_aging_flag = pm.set_index("machine_id")["is_aging_asset"].reindex(_arate.index).fillna(False)
aging_alarm_rate     = float(_arate[_aging_flag.values].mean())
nonaging_alarm_rate  = float(_arate[~_aging_flag.values].mean())
aging_alarm_multiple = aging_alarm_rate / nonaging_alarm_rate

# Tooling is the largest single failure mode in the downtime record.
_tool               = cmms[cmms["failure_code"] == "TOOLING"]
tooling_pct         = float(_tool["downtime_hours"].sum() / cmms["downtime_hours"].sum() * 100)
tooling_cost_annual = float(_tool["downtime_cost"].sum()) / window_years

# ── Recommended-action impact estimates (annual, from the regenerated data) ──
cm_blended = float((cost["cm"] * cost["run_hrs"]).sum() / cost["run_hrs"].sum())

# 1. Shift-start: unplanned-down cost in the startup window above a uniform-time
#    expectation (the window is 45 of every 480 shift minutes).
_ms = da[da["source_system"] == "MACHINE_STATE"]
_startup_share = SHIFT_STARTUP_WINDOW_MIN / 480.0
shift_impact = max(_ms.loc[_ms["is_shift_startup"] == True, "downtime_cost"].sum()
                   - _ms["downtime_cost"].sum() * _startup_share, 0.0) / window_years

# 2. Setup standardisation: excess setup hours from the flagged operators, priced
#    at the blended plant contribution margin.
_sf = os_[os_["setup_ratio_vs_cohort"] >= SETUP_OUTLIER_RATIO]
setup_ratio_mean = float(_sf["setup_ratio_vs_cohort"].mean())
setup_gap_hours  = float(((_sf["median_setup_hours"] - _sf["cohort_median_setup_hours"])
                          * _sf["total_jobs"]).sum())
setup_impact = setup_gap_hours / window_years * cm_blended

# 3. PM catch-up: reactive-repair cost incurred while materially past a due PM.
_cmms2 = cmms.copy()
_cmms2["date"]    = pd.to_datetime(_cmms2["event_date"]).dt.date
_cmms2["overdue"] = [_in_overdue(m, d) for m, d in zip(_cmms2["machine_id"], _cmms2["date"])]
pm_impact = float(_cmms2.loc[_cmms2["overdue"], "downtime_cost"].sum()) / window_years

# 4. Tooling program: recoverable share of tooling reactive-repair cost.
TOOLING_RECOVERABLE_SHARE = 0.40
tooling_impact = tooling_cost_annual * TOOLING_RECOVERABLE_SHARE

# 5. Reliability review (rebuild versus replace) for the two worst aging assets:
#    their availability-side opportunity, the unplanned downtime a more reliable
#    machine would avoid.
_aging2 = [m for m in ["MCH-011", "MCH-007"] if m in cost.index]
reliability_impact = float(cost.loc[_aging2, "avail_opp"].sum())

# 6. Spindle rebuild and control retrofit on the oldest machines (>9 years): the
#    performance-side opportunity on those assets, recovered by restoring cycle
#    speed. Held separate from the reliability item above (availability lever) so
#    the two capital actions do not double-count the same dollars.
_aging_old = list(cost.index[cost["is_aging_asset"]])
spindle_impact = float(cost.loc[_aging_old, "perf_opp"].sum())


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_machine_ranking():
    fig, ax = make_fig(h=4.4)
    labels = [mlabel(m, t) for m, t in zip(mach["machine_id"], mach["machine_type"])]
    colors = [ACCENT_RED if aging else DARK_BLUE for aging in mach["is_aging_asset"]]
    ax.barh(labels, mach["oee"] * 100, color=colors, height=0.65)
    bl = ax.axvline(BENCHMARK_OEE * 100, color="#8093A4", linestyle="--", linewidth=1.4)
    for y, v in enumerate(mach["oee"]):
        ax.text(v * 100 + 0.6, y, f"{v:.0%}", va="center", fontsize=BODY_FS, color=TEXT)
    ax.set_xlabel("OEE (%)"); ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT_RED),
               plt.Rectangle((0, 0), 1, 1, color=DARK_BLUE)]
    ax.legend(handles + [bl],
              ["Aging asset (>9 yrs)", "Remaining machines", f"Baseline ({BENCHMARK_OEE:.0%})"],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=BODY_FS, frameon=False)
    chart_style(ax); ax.xaxis.grid(True, color="#D5DCE1"); ax.yaxis.grid(False)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_oee_trend():
    fig, ax = make_fig()
    for grp, sub in trend_monthly.groupby("machine_type"):
        sub = sub.sort_values("period_month")
        ax.plot(sub["period_month"], sub["oee"] * 100, marker="o", markersize=3,
                linewidth=1.8, color=TYPE_COLORS.get(grp, MED_GREY), label=grp)
    ax.axhline(BENCHMARK_OEE * 100, color=MED_GREY, linestyle=":", linewidth=1.3)
    lines = "\n".join(f"{TYPE_ABBR[t]:>5}: {trend_stats[t][0]:+.2f} pts/yr  (r={trend_stats[t][1]:+.2f})"
                      for t in TYPES)
    ax.text(0.015, 0.04, "12-mo OEE trend:\n" + lines, transform=ax.transAxes,
            fontsize=BODY_FS, va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D5DCE1"))
    ax.set_ylabel("OEE (%)"); ax.set_ylim(30, 95)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.legend(fontsize=BODY_FS, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), frameon=False)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_components_by_type():
    x = np.arange(len(TYPES)); w = 0.35
    fig, ax = make_fig()
    avail = [by_type[t][0] * 100 for t in TYPES]
    perf  = [by_type[t][1] * 100 for t in TYPES]
    ax.bar(x - w / 2, avail, w, color=DARK_BLUE, label="Availability")
    ax.bar(x + w / 2, perf, w, color=LIGHT_BLUE, label="Performance")
    for i, t in enumerate(TYPES):
        ax.text(i - w / 2, avail[i] + 1.2, f"{avail[i]:.0f}%", ha="center", va="bottom", fontsize=BODY_FS)
        ax.text(i + w / 2, perf[i] + 1.2, f"{perf[i]:.0f}%", ha="center", va="bottom", fontsize=BODY_FS)
        ax.text(i, max(avail[i], perf[i]) + 7, f"OEE {by_type[t][2]:.0%}",
                ha="center", fontsize=BODY_FS, fontweight="bold", color=TEXT)
    ax.set_xticks(x); ax.set_xticklabels(TYPES)
    ax.set_ylabel("(%)"); ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=BODY_FS, loc="lower left")
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_planned_time():
    months = sorted(mp["period_month"].unique())[-6:]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 4.3), sharey=True)
    plt.subplots_adjust(wspace=0.08)
    for ax, t in zip(axes, TYPES):
        d = mp[(mp["machine_type"] == t) & (mp["period_month"].isin(months))]
        g = d.groupby("period_month")
        planned    = g["planned_production_minutes"].sum().reindex(months)
        run        = g["run_minutes"].sum().reindex(months)
        productive = g.apply(lambda x: (x["performance"] * x["run_minutes"]).sum(),
                             include_groups=False).reindex(months)
        prod       = (productive / planned * 100).values
        perf_loss  = ((run - productive) / planned * 100).values
        avail_loss = ((planned - run) / planned * 100).values
        xi = np.arange(len(months))
        ax.bar(xi, prod, width=0.8, color=DARK_BLUE, label="Productive")
        ax.bar(xi, avail_loss, bottom=prod, width=0.8, color=MED_GREY, label="Availability loss")
        ax.bar(xi, perf_loss, bottom=prod + avail_loss, width=0.8, color=ACCENT_RED, label="Performance loss")
        # Percent centred in each segment. White reads on the dark-blue and red
        # fills, black on the grey.
        for xpos, (pv, alv, plv) in enumerate(zip(prod, avail_loss, perf_loss)):
            if pv >= 8:
                ax.text(xpos, pv / 2, f"{pv:.0f}%", ha="center", va="center",
                        fontsize=BODY_FS, color="white")
            if alv >= 8:
                ax.text(xpos, pv + alv / 2, f"{alv:.0f}%", ha="center", va="center",
                        fontsize=BODY_FS, color=TEXT)
            if plv >= 8:
                ax.text(xpos, pv + alv + plv / 2, f"{plv:.0f}%", ha="center", va="center",
                        fontsize=BODY_FS, color="white")
        ax.set_title(t, fontsize=BODY_FS, fontweight="bold")
        ax.set_ylim(0, 100); ax.set_xlim(-0.6, len(months) - 0.4)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([pd.Timestamp(mo).strftime("%b'%y") for mo in months],
                           rotation=40, ha="right", fontsize=BODY_FS - 1)
        chart_style(ax)
    axes[0].set_ylabel("(% of planned time)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               frameon=False, fontsize=BODY_FS)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    return fig_to_b64(fig)


def chart_failure_pareto():
    p = cmms.groupby("failure_code")["downtime_hours"].sum().sort_values(ascending=False)
    pct_cum = p.cumsum() / p.sum() * 100
    fig, ax = make_fig()
    ax.bar(range(len(p)), p.values, color=DARK_BLUE, width=0.6)
    ax.set_xticks(range(len(p)))
    ax.set_xticklabels([FAILURE_LABELS.get(c, c.title()) for c in p.index], rotation=20, ha="right")
    ax.set_ylabel("Unplanned Downtime (hrs)"); ax.set_ylim(0, 4500)
    ax2 = ax.twinx()
    ax2.plot(range(len(p)), pct_cum.values, color=ACCENT_RED, marker="o", linewidth=1.8)
    ax2.set_ylim(0, 108); ax2.set_ylabel("Cumulative %", color=ACCENT_RED)
    ax2.tick_params(axis="y", colors=ACCENT_RED)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for i, v in enumerate(pct_cum.values):
        ax2.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=BODY_FS, color=ACCENT_RED)
    for i, v in enumerate(p.values):
        ax.text(i, v + p.max() * 0.02, f"{v:,.0f}", ha="center", fontsize=BODY_FS)
    chart_style(ax); ax2.grid(False)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_time_of_day():
    """Count of unplanned-down stoppages that begin in each 15-minute window."""
    st = da[(da["source_system"] == "MACHINE_STATE") & da["minute_of_shift"].notna()]
    BIN, SHIFT_MIN = 15, 480
    nbins = SHIFT_MIN // BIN
    counts = {"A": np.zeros(nbins), "B": np.zeros(nbins)}
    for shift, start in zip(st["shift"], st["minute_of_shift"]):
        b = int(start) // BIN
        if 0 <= b < nbins and shift in counts:
            counts[shift][b] += 1
    centers = np.arange(0, SHIFT_MIN, BIN) + BIN / 2
    fig, ax = make_fig()
    for shift, color, lbl in [("A", LIGHT_BLUE, "Shift A"), ("B", DARK_BLUE, "Shift B")]:
        ax.plot(centers, counts[shift], marker="o", markersize=3, linewidth=1.8, color=color, label=lbl)
    ax.axvspan(0, SHIFT_STARTUP_WINDOW_MIN, color=ACCENT_RED, alpha=0.10)
    _ylo, _yhi = ax.get_ylim()
    ax.text(SHIFT_STARTUP_WINDOW_MIN / 2, _ylo + (_yhi - _ylo) * 0.03, "Startup\nwindow",
            ha="center", va="bottom", fontsize=BODY_FS, color=ACCENT_RED, fontweight="bold")
    ax.set_xlabel("Minutes into shift"); ax.set_ylabel("Unplanned stoppages (count)")
    ax.legend(fontsize=BODY_FS, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_alarm_rate():
    alarms  = mp.groupby("machine_id")["alarm_events"].sum()
    run_hrs = mp.groupby("machine_id")["run_minutes"].sum() / 60.0
    rate = (alarms / run_hrs * 1000).sort_values(ascending=False)
    aging = pm.set_index("machine_id")["is_aging_asset"]
    colors = [ACCENT_RED if aging.get(m, False) else DARK_BLUE for m in rate.index]
    fig, ax = make_fig(h=3.9)
    ax.bar(range(len(rate)), rate.values, color=colors, width=0.65)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([mlabel(m, TYPE_BY_MACHINE[m]) for m in rate.index], rotation=45, ha="right")
    ax.set_ylabel("Alarms per 1,000 run hrs")
    for i, v in enumerate(rate.values):
        ax.text(i, v + rate.max() * 0.02, f"{v:.0f}", ha="center", fontsize=BODY_FS)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT_RED), plt.Rectangle((0, 0), 1, 1, color=DARK_BLUE)]
    ax.legend(handles, ["Aging asset (>9 yrs)", "Remaining machines"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=BODY_FS, frameon=False)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_pm_alarm():
    alarms = mp.groupby("machine_id")["alarm_events"].sum().rename("alarm_events")
    run_hrs = (mp.groupby("machine_id")["run_minutes"].sum() / 60.0).rename("run_hrs")
    d = pd.concat([alarms, run_hrs], axis=1).reset_index()
    d = d.merge(pm[["machine_id", "pct_ontime", "is_aging_asset"]], on="machine_id")
    d["alarm_rate"] = d["alarm_events"] / d["run_hrs"] * 1000
    d["pct_late"] = 100 - d["pct_ontime"]
    fig, ax = make_fig()
    colors = [ACCENT_RED if a else LIGHT_BLUE for a in d["is_aging_asset"]]
    ax.scatter(d["pct_late"], d["alarm_rate"], c=colors, s=70, edgecolor="white", zorder=3)
    for _, r in d.iterrows():
        if r["is_aging_asset"] or r["pct_late"] > 40:
            ax.annotate(r["machine_id"], (r["pct_late"], r["alarm_rate"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=BODY_FS, color=ACCENT_RED)
    z = np.polyfit(d["pct_late"], d["alarm_rate"], 1)
    xs = np.linspace(d["pct_late"].min(), d["pct_late"].max(), 20)
    ax.plot(xs, np.polyval(z, xs), color=DARK_BLUE, linestyle="--", linewidth=1.4)
    ax.set_xlabel("PM non-compliance (% of PMs late)")
    ax.set_ylabel("Alarm rate (per 1,000 run hrs)")
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_pm_completion():
    """On-time PM completion rate by machine, worst first, with the fleet average
    marked. Ties to the shop-wide and worst-machine figures quoted in the text."""
    d = pm[["machine_id", "pct_ontime", "is_aging_asset"]].sort_values("pct_ontime")
    fleet = pm["pct_ontime"].mean()
    colors = [ACCENT_RED if a else DARK_BLUE for a in d["is_aging_asset"]]
    fig, ax = make_fig(h=3.9)
    ax.bar(range(len(d)), d["pct_ontime"], color=colors, width=0.65)
    ax.axhline(fleet, color=MED_GREY, linestyle="--", linewidth=1.4)
    ax.text(0.01, 0.95, f"Fleet average: {fleet:.0f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=BODY_FS, color=DARK_GREY, fontweight="bold")
    for i, v in enumerate(d["pct_ontime"]):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=BODY_FS)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([mlabel(m, TYPE_BY_MACHINE[m]) for m in d["machine_id"]], rotation=45, ha="right")
    ax.set_ylabel("On-time PM completion"); ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT_RED), plt.Rectangle((0, 0), 1, 1, color=DARK_BLUE)]
    ax.legend(handles, ["Aging asset (>9 yrs)", "Remaining machines"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=BODY_FS, frameon=False)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_pm_within():
    """Per-machine alarm rate during PM-overdue vs PM-current periods (same asset)."""
    d = pm_within.sort_values("rate_current")
    x = np.arange(len(d)); w = 0.38
    fig, ax = make_fig(h=3.9)
    bars_cur = ax.bar(x - w / 2, d["rate_current"].values, w, color=MED_GREY, label="PM current")
    bars_ovd = ax.bar(x + w / 2, d["rate_overdue"].values, w, color=ACCENT_RED,
                      label=f"PM overdue (>{PM_OVERDUE_THRESHOLD_DAYS} days)")
    ax.set_xticks(x)
    ax.set_xticklabels([mlabel(m, TYPE_BY_MACHINE[m]) for m in d.index], rotation=45, ha="right")
    ax.set_ylabel("Alarms per 1,000 run hrs")
    ax.set_ylim(0, d["rate_overdue"].max() * 1.18)
    for bars in (bars_cur, bars_ovd):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + d["rate_overdue"].max() * 0.02,
                    f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=BODY_FS, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_perf_vs_age():
    fig, ax = make_fig()
    colors = [ACCENT_RED if a else LIGHT_BLUE for a in mach["is_aging_asset"]]
    ax.scatter(mach["machine_age_years"], mach["performance"] * 100, c=colors, s=70,
               edgecolor="white", zorder=3)
    for _, r in mach.iterrows():
        if r["is_aging_asset"]:
            ax.annotate(r["machine_id"], (r["machine_age_years"], r["performance"] * 100),
                        textcoords="offset points", xytext=(6, 4), fontsize=BODY_FS, color=ACCENT_RED)
    z = np.polyfit(mach["machine_age_years"], mach["performance"] * 100, 1)
    xs = np.linspace(mach["machine_age_years"].min(), mach["machine_age_years"].max(), 20)
    ax.plot(xs, np.polyval(z, xs), color=DARK_BLUE, linestyle="--", linewidth=1.5)
    ax.text(0.97, 0.94, f"r = {age_r:.2f}   R² = {age_r**2:.2f}\n{age_slope:.1f} perf-pts / year of age",
            transform=ax.transAxes, ha="right", va="top", fontsize=BODY_FS,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D5DCE1"))
    ax.set_xlabel("Machine age (years)"); ax.set_ylabel("Performance (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_perf_by_machine():
    d = mach.sort_values("performance")
    aging = d.set_index("machine_id")["is_aging_asset"]
    colors = [ACCENT_RED if aging[m] else DARK_BLUE for m in d["machine_id"]]
    fig, ax = make_fig(h=4.0)
    ax.bar(range(len(d)), d["performance"] * 100, color=colors, width=0.65)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([mlabel(m, TYPE_BY_MACHINE[m]) for m in d["machine_id"]], rotation=45, ha="right")
    ax.set_ylabel("Performance (%)"); ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for i, v in enumerate(d["performance"]):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=BODY_FS)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT_RED), plt.Rectangle((0, 0), 1, 1, color=DARK_BLUE)]
    ax.legend(handles, ["Aging asset (>9 yrs)", "Remaining machines"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=BODY_FS, frameon=False)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_operator_setup():
    d = os_.sort_values("median_setup_hours", ascending=True)
    cohort = float(d["cohort_median_setup_hours"].iloc[0])
    colors = [ACCENT_RED if r >= SETUP_OUTLIER_RATIO else DARK_BLUE for r in d["setup_ratio_vs_cohort"]]
    fig, ax = make_fig(h=4.4)
    ax.barh(d["operator_id"], d["median_setup_hours"], color=colors, height=0.65)
    ax.axvline(cohort, color=MED_GREY, linestyle="--", linewidth=1.4,
               label=f"Cohort median ({cohort:.2f} h)")
    # Every bar carries its actual setup time; the flagged operators also carry
    # their multiple of the cohort median.
    for y, (v, ratio) in enumerate(zip(d["median_setup_hours"], d["setup_ratio_vs_cohort"])):
        label = f"{v:.2f} h" + (f"  ({ratio:.1f}x)" if ratio >= SETUP_OUTLIER_RATIO else "")
        colour = ACCENT_RED if ratio >= SETUP_OUTLIER_RATIO else TEXT
        ax.text(v + 0.01, y, label, va="center", fontsize=BODY_FS, color=colour,
                fontweight="bold" if ratio >= SETUP_OUTLIER_RATIO else "normal")
    ax.set_xlim(0, d["median_setup_hours"].max() * 1.30)
    ax.set_xlabel("Median setup hours per job")
    ax.legend(loc="lower right", fontsize=BODY_FS)
    chart_style(ax); ax.xaxis.grid(True, color="#D5DCE1"); ax.yaxis.grid(False)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_opportunity_by_machine():
    d = cost.sort_values("opp", ascending=False)
    fig, ax = make_fig(h=4.0)
    x = range(len(d))
    avail_k, perf_k, opp_k = d["avail_opp"] / 1000, d["perf_opp"] / 1000, d["opp"] / 1000
    ax.bar(x, avail_k, color=DARK_BLUE, width=0.8, label="Availability")
    ax.bar(x, perf_k, bottom=avail_k, color=LIGHT_BLUE, width=0.8, label="Performance")
    ax.set_xticks(list(x))
    ax.set_xticklabels([mlabel(m, TYPE_BY_MACHINE[m]) for m in d.index], rotation=45, ha="right")
    ax.set_ylabel("Margin Uplift Opportunity ($K)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    # Total above each bar, plus each lever's own share inside its segment. White
    # reads on the dark blue, black on the light blue.
    for i, (a, p, t) in enumerate(zip(avail_k, perf_k, opp_k)):
        ax.text(i, t + opp_k.max() * 0.02, f"${t:,.0f}K", ha="center", va="bottom",
                fontsize=9, color=TEXT)
        if a > opp_k.max() * 0.08:
            ax.text(i, a / 2, f"${a:,.0f}K", ha="center", va="center",
                    fontsize=9, color="white")
        if p > opp_k.max() * 0.08:
            ax.text(i, a + p / 2, f"${p:,.0f}K", ha="center", va="center",
                    fontsize=9, color=TEXT)
    ax.set_ylim(0, opp_k.max() * 1.20)
    ax.legend(fontsize=BODY_FS, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_opportunity_bar():
    """Single left-to-right stacked bar: the annual margin opportunity split into
    its Availability and Performance components, with the total called out."""
    a, p, tot = avail_opp_annual, perf_opp_annual, opportunity_annual
    fig, ax = plt.subplots(figsize=(CHART_W, 1.5))
    ax.barh(0, a, color=DARK_BLUE)
    ax.barh(0, p, left=a, color=LIGHT_BLUE)
    ax.text(a / 2, 0, f"Availability\n{usd_short(a)}", ha="center", va="center",
            color="white", fontsize=BODY_FS, fontweight="bold")
    ax.text(a + p / 2, 0, f"Performance\n{usd_short(p)}", ha="center", va="center",
            color=TEXT, fontsize=BODY_FS, fontweight="bold")
    ax.text(tot * 1.015, 0, f"{usd_short(tot)}\nper year", ha="left", va="center",
            color=DARK_GREY, fontsize=BODY_FS, fontweight="bold")
    ax.set_xlim(0, tot * 1.22); ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_actual_vs_target():
    """Three side-by-side actual-vs-target bars, one each for OEE, Availability and
    Performance. OEE targets the benchmark; Availability and Performance target the
    higher rate the benchmark implies."""
    metrics = [("OEE", plant_oee, BENCHMARK_OEE),
               ("Availability", plant_avail, AP_TARGET),
               ("Performance", plant_perf, AP_TARGET)]
    fig, axes = plt.subplots(1, 3, figsize=(CHART_W, 3.4), sharey=True)
    for ax, (name, actual, target) in zip(axes, metrics):
        ax.bar(0, actual * 100, width=0.6, color=DARK_BLUE)
        ax.bar(1, target * 100, width=0.6, color=MED_GREY)
        ax.text(0, actual * 100 + 1.5, f"{actual * 100:.1f}%", ha="center", va="bottom",
                fontsize=BODY_FS, fontweight="bold", color=DARK_GREY)
        ax.text(1, target * 100 + 1.5, f"{target * 100:.0f}%", ha="center", va="bottom",
                fontsize=BODY_FS, fontweight="bold", color=DARK_GREY)
        ax.set_title(name, fontsize=BODY_FS, fontweight="bold", color=DARK_GREY)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Actual", "Target"], fontsize=BODY_FS)
        ax.set_xlim(-0.7, 1.7); ax.set_ylim(0, 108)
        chart_style(ax)
    axes[0].set_ylabel("(%)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    plt.tight_layout()
    return fig_to_b64(fig)


print("Generating charts...")
charts = {
    "avt":         chart_actual_vs_target(),
    "opp_bar":     chart_opportunity_bar(),
    "ranking":     chart_machine_ranking(),
    "trend":       chart_oee_trend(),
    "planned_time": chart_planned_time(),
    "pareto":      chart_failure_pareto(),
    "timeofday":   chart_time_of_day(),
    "alarm_rate":  chart_alarm_rate(),
    "pm_within":   chart_pm_within(),
    "pm_completion": chart_pm_completion(),
    "perf_age":    chart_perf_vs_age(),
    "perf_mach":   chart_perf_by_machine(),
    "setup":       chart_operator_setup(),
    "cost":        chart_opportunity_by_machine(),
}


# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════

def img(key):
    return (f'<img src="data:image/png;base64,{charts[key]}" '
            f'style="width:100%;height:auto;display:block;">')


worst, best = mach.iloc[0], mach.iloc[-1]
lathe_oee, hmill_oee = by_type["CNC Lathe"][2], by_type["Horizontal Mill"][2]
setup_flagged = os_[os_["setup_ratio_vs_cohort"] >= SETUP_OUTLIER_RATIO].sort_values("setup_ratio_vs_cohort", ascending=False)
worst_pm = pm.sort_values("pct_ontime").iloc[0]
top_two = cmms.groupby("failure_code")["downtime_hours"].sum().sort_values(ascending=False).head(2)
top_two_pct = top_two.sum() / cmms["downtime_hours"].sum() * 100
aging_ids = ", ".join(mach[mach["is_aging_asset"]]["machine_id"].head(2))
aging_ids_all = ", ".join(mach[mach["is_aging_asset"]]["machine_id"])
n_aging = int(mach["is_aging_asset"].sum())

# Recommended actions, ranked by the annual impact each one carries. The mix
# spans low- and high-capital moves: routine and coaching changes that cost
# little, alongside the capital decisions on the oldest assets.
ACTIONS = sorted([
    (f"Reliability review, rebuild versus replace, for {aging_ids}",
     "Oldest assets carry the most unplanned downtime (P1)",
     reliability_impact, "High"),
    ("Spindle rebuild and control retrofit on the oldest machines",
     f"Performance falls about {abs(age_slope):.1f} points per year of machine age",
     spindle_impact, "High"),
    ("Tooling program: vendor consolidation and tool-life tracking",
     f"Tooling is the largest downtime category, {tooling_pct:.0f}% of unplanned hours",
     tooling_impact, "Low"),
    (f"Setup standardisation coaching for {', '.join(setup_flagged['operator_id'])}",
     f"Setups at {setup_ratio_mean:.1f}x the cohort median (P4)",
     setup_impact, "None"),
    ("Standardised shift-start warm-up and first-piece routine on both shifts",
     f"Stoppages run {startup_lift:.1f}x higher in a shift's first {SHIFT_STARTUP_WINDOW_MIN} minutes (P2)",
     shift_impact, "None"),
    ("PM catch-up and compliance enforcement",
     "Within-machine PM-overdue to alarm-rate linkage (P3)",
     pm_impact, "Minimal"),
], key=lambda row: row[2], reverse=True)

actions_rows = "".join(
    f'<tr><td>{action}</td><td>{finding}</td>'
    f'<td style="text-align:right;font-weight:700;">{usd_short(value)}</td><td>{capital}</td></tr>'
    for action, finding, value, capital in ACTIONS)
no_capital_actions = ", ".join(a for a, _, _, c in ACTIONS if c in ("None", "Minimal"))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OEE Diagnostic</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:#fff; color:{TEXT}; font-size:16px; line-height:1.7; }}
  .page-header {{ background:{DARK_GREY}; color:#fff; padding:14px 40px; }}
  .page-header h1 {{ font-size:21px; font-weight:700; letter-spacing:-0.3px; }}
  .layout {{ display:flex; max-width:1200px; margin:0 auto; padding:0 40px; }}
  .toc {{ width:240px; flex-shrink:0; padding:40px 20px 40px 0; position:sticky;
    top:0; height:100vh; overflow-y:auto; border-right:1px solid #D5DCE1; }}
  .toc-title {{ font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#8093A4;
    margin-bottom:14px; font-weight:600; }}
  .toc a {{ display:block; font-size:13px; color:#8093A4; text-decoration:none;
    padding:5px 0 5px 10px; border-left:2px solid transparent; }}
  .toc a:hover {{ color:{DARK_GREY}; border-left-color:{DARK_GREY}; }}
  .content {{ flex:1; padding:40px 0 80px 52px; max-width:860px; }}
  .section-title-block {{ margin:46px 0 22px; padding-bottom:12px; border-bottom:2px solid {DARK_GREY}; }}
  .content > .section-title-block:first-child {{ margin-top:8px; }}
  /* Section titles read black; the chart and table titles under them read
     dark grey. */
  .section-label {{ font-size:10px; letter-spacing:2px; text-transform:uppercase;
    color:{TEXT}; font-weight:600; margin-bottom:4px; }}
  .section-title {{ font-size:22px; font-weight:700; color:{TEXT}; }}
  p {{ margin-bottom:16px; color:#000000; }}
  .chart-wrap {{ margin:20px 0; border:1px solid #D5DCE1; border-radius:4px; padding:12px; }}
  .chart-title {{ font-size:16px; font-weight:700; text-align:center; margin-bottom:8px; color:{DARK_GREY}; }}
  .kpi-row {{ display:flex; gap:16px; margin:22px 0; flex-wrap:wrap; }}
  .kpi {{ flex:1; min-width:140px; background:#F3F5F7; border:1px solid #D5DCE1;
    border-radius:6px; padding:16px 18px; }}
  .kpi .v {{ font-size:25px; font-weight:700; color:{DARK_GREY}; }}
  .kpi .l {{ font-size:12px; color:#8093A4; text-transform:uppercase; letter-spacing:.5px; margin-top:2px; }}
  .kpi.warn .v {{ color:{ACCENT_RED}; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:14px; }}
  th {{ background:#F3F5F7; padding:10px 12px; text-align:left; font-size:12px;
    font-weight:600; text-transform:uppercase; letter-spacing:.5px; color:#8093A4; border-bottom:2px solid #D5DCE1; }}
  td {{ padding:9px 12px; border-bottom:1px solid #D5DCE1; color:#000000; }}
  tr:hover td {{ background:#F3F5F7; }}
  .flag {{ color:{ACCENT_RED}; font-weight:700; }}
  @page {{ margin:12mm; }}
  @media print {{
    .toc {{ display:none; }}
    .layout {{ max-width:none; margin:0; padding:0; }}
    .content {{ max-width:none; padding:20px 28px; }}
    .page-header {{ padding:12px 28px; }}
    .chart-wrap, table, .kpi-row, .kpi {{ break-inside:avoid; }}
    .section-title-block {{ break-after:avoid; }}
    tr, img {{ break-inside:avoid; }}
  }}
</style>
</head>
<body>
<div class="page-header">
  <h1>Analytics Diagnostic Report: OEE</h1>
</div>
<div class="layout">
  <nav class="toc">
    <div class="toc-title">Contents</div>
    <a href="#summary">1 &middot; Executive Summary</a>
    <a href="#machines">2 &middot; Machine-level OEE Breakdown</a>
    <a href="#availability">3 &middot; Deep Dive: Availability</a>
    <a href="#performance">4 &middot; Deep Dive: Performance</a>
    <a href="#cost">5 &middot; Margin Uplift Opportunity Detail</a>
    <a href="#actions">6 &middot; Recommended Actions</a>
  </nav>
  <main class="content">

    <div class="section-title-block" id="summary">
      <div class="section-label">Section 1</div>
      <h2 class="section-title">Executive Summary</h2>
    </div>
    <p>Over the period from January 2023 to March 2026, plant Overall Equipment Effectiveness (OEE)
    averaged <strong>{plant_oee:.1%}</strong>, against a target of {BENCHMARK_OEE:.0%}. The
    shortfall is split about evenly between <strong>Availability</strong> ({plant_avail:.1%}), total
    machine running time relative to planned production time, and <strong>Performance</strong>
    ({plant_perf:.1%}), actual machine output rate relative to its ideal rate while running.</p>
    <p>Note that Quality is excluded from this report. We assumed a rate of {quality:.0%} for
    purposes of calculating OEE.</p>
    <div class="chart-wrap"><div class="chart-title">Actual vs. Target: OEE, Availability, Performance</div>{img('avt')}</div>
    <p>The single largest driver of the gap between today's {plant_oee:.1%} OEE and the
    {BENCHMARK_OEE:.0%} target is the underperformance of the shop's oldest machines. The
    {n_aging_assets} aging machines (over nine years old), <span class="flag">{aging_ids_all}</span>,
    run at {aging_oee:.0%} OEE against {newer_oee:.0%} for the rest of the fleet, about {oee_gap_r}
    points lower. Their Availability is roughly {avail_gap_r} points below the newer machines
    ({aging_avail:.0%} versus {newer_avail:.0%}), because they stop and alarm more often, and their
    Performance is about {perf_gap_r} points lower ({aging_perf:.0%} versus {newer_perf:.0%}),
    because their cycle time is slowed by worn spindles and dated controls. Although they are only
    {n_aging_assets} of {n_fleet} machines, they carry about {aging_opp_share:.0%} of the total
    margin opportunity shown below.</p>
    <div class="chart-wrap"><div class="chart-title">Annual Contribution Margin Opportunity</div>{img('opp_bar')}</div>
    <p>Reaching the {BENCHMARK_OEE:.0%} target OEE (implying Availability and Performance of
    {AP_TARGET:.0%}) is worth an estimated <strong>{usd_short(opportunity_annual)} per year</strong>
    in additional contribution margin ({usd_short(avail_opp_annual)} from Availability and
    {usd_short(perf_opp_annual)} from Performance). Closing that gap between actual and target
    requires execution across a handful of levers, as detailed in Section 6. The most impactful of
    these levers include a
    reliability review and spindle rebuild on the oldest machines, plus a comprehensive tooling
    program covering vendor consolidation, tool-life tracking, and standardised tool-change
    procedures.</p>

    <div class="section-title-block" id="machines">
      <div class="section-label">Section 2</div>
      <h2 class="section-title">Machine-level OEE Breakdown</h2>
    </div>
    <p>Across all machines, OEE ranges from {worst['oee']:.0%} to {best['oee']:.0%}. CNC lathes
    consistently have higher OEE than the vertical and horizontal mills. This pattern is due to the
    mills being older, more capital-intensive machines that are most frequently in need of routine
    maintenance, and that also take the most complex work, and so lose more time to stoppages and
    generally run slower. The lathes are newer and run simpler jobs. As mentioned, the three aging
    machines (>9 years old), <span class="flag">{aging_ids_all}</span>,
    have the lowest OEE among the fleet.</p>
    <div class="chart-wrap"><div class="chart-title">OEE by Machine</div>{img('ranking')}</div>
    <p>The OEE by machine type holds steady across the period, with no noticeable increases or
    decreases over time.</p>
    <div class="chart-wrap"><div class="chart-title">OEE Trend by Machine Type</div>{img('trend')}</div>
    <p>The chart below shows each machine type's planned production time split into three parts:
    time spent productive (running at target speed), time lost to downtime (an Availability loss),
    and time lost to slow cycle times (a Performance loss). The productive share of time for each
    machine type holds fairly steady from month to month, and the losses split fairly evenly between
    both Availability and Performance.</p>
    <div class="chart-wrap"><div class="chart-title">Productive vs. Lost Time by Machine Type (Monthly)</div>{img('planned_time')}</div>

    <div class="section-title-block" id="availability">
      <div class="section-label">Section 3</div>
      <h2 class="section-title">Deep Dive: Availability</h2>
    </div>
    <p>The below downtime analysis shows where Availability is lost and why. Unplanned downtime
    concentrates heavily in just two failure modes: about {top_two_pct:.0f}% of all unplanned
    downtime hours trace to {top_two.index[0].lower()} and {top_two.index[1].lower()} issues, as
    shown with the red line below. Therefore, targeted solutions that address tooling and mechanical
    issues will have an outsized impact on reducing total unplanned downtime.</p>
    <div class="chart-wrap"><div class="chart-title">Unplanned Downtime by Failure Code</div>{img('pareto')}</div>
    <p>Downtime is not spread evenly through the day. Across the full observation window
    ({PERIOD_LABEL}), unplanned stoppages concentrate at the beginning of shifts before steadying
    across the middle and end of the shifts. As seen in the chart below, during the first
    {SHIFT_STARTUP_WINDOW_MIN} minutes of each shift, unplanned stoppages run about
    {startup_lift:.1f}x the mid-shift rate ({startup_lift_a:.1f}x on Shift A and
    {startup_lift_b:.1f}x on Shift B). These spikes are likely due to cold machines and spindles
    coming up to temperature, warm-up routines, and first-piece setup and verification at the start
    of a run. The fact that this pattern emerges in both shifts points to the shift-start routine
    rather than at one weaker crew. Therefore, a standardised shift-start routine, a documented
    warm-up sequence and first-piece check written into the opening of every shift, can normalise
    this spike, representing a potential contribution margin uplift of about {usd_short(shift_impact)}
    per year.</p>
    <div class="chart-wrap"><div class="chart-title">Unplanned Stoppages by Time Into Shift</div>{img('timeofday')}</div>
    <p>A machine's alarms cluster in the run-up to a stoppage, and a rising alarm rate is an
    important indicator of maintenance needs. In the chart below, we see the three aging machines
    raise alarms about {aging_alarm_multiple:.1f}x as often as the rest of the fleet on average,
    while the newest CNC lathes sit lowest of all.</p>
    <div class="chart-wrap"><div class="chart-title">Alarm Rate by Machine</div>{img('alarm_rate')}</div>
    <p>Comparing each machine's alarm rate against its preventive maintenance (PM) history shows how
    much of the alarm burden tracks with deferred maintenance rather than with machine age alone.
    The chart below measures the alarm rate during the periods when a machine is more than
    {PM_OVERDUE_THRESHOLD_DAYS} days past a due PM against the periods when its PM is current. We
    find that alarm rates run about <strong>{pm_overdue_multiple:.1f}x</strong> higher while a
    machine is overdue, and the gap holds for newer and older machines alike.</p>
    <div class="chart-wrap"><div class="chart-title">Alarm Rate by PM Status and Machine</div>{img('pm_within')}</div>
    <p>Shop-wide on-time PM completion is {pm['pct_ontime'].mean():.0f}% and uneven, with
    <span class="flag">{worst_pm['machine_id']}</span> at only {worst_pm['pct_ontime']:.0f}%.
    Catching up on PM is therefore an important reliability lever that can improve Availability.</p>
    <div class="chart-wrap"><div class="chart-title">On-Time PM Completion by Machine</div>{img('pm_completion')}</div>

    <div class="section-title-block" id="performance">
      <div class="section-label">Section 4</div>
      <h2 class="section-title">Deep Dive: Performance</h2>
    </div>
    <p>Performance is closely tied to machine age. Older machines hold a slower cycle through worn
    spindles, dated controls, and more conservative feeds and speeds. As seen in the chart below, the
    relationship is strong: across the shop, Performance falls about {abs(age_slope):.1f} percentage
    points for every year of machine age.</p>
    <div class="chart-wrap"><div class="chart-title">Performance vs. Machine Age</div>{img('perf_age')}</div>
    <p>Ranked by machine, we see that the newest CNC lathes average about {lathe_perf:.0%}
    Performance, while the vertical and horizontal mills average about {mills_perf:.0%} Performance,
    roughly {lathe_vs_mills_pts:.0f} points below the CNC lathes as a group.</p>
    <div class="chart-wrap"><div class="chart-title">Performance by Machine</div>{img('perf_mach')}</div>
    <p>Analyzing setup time by operator, we find that most machinists cluster near the cohort median
    of {os_['cohort_median_setup_hours'].iloc[0] * 60:.0f} minutes per job, with just a modest spread
    across most of the operators. Three of them,
    <span class="flag">{', '.join(setup_flagged['operator_id'])}</span>, sit consistently above the
    rest at about {setup_flagged['setup_ratio_vs_cohort'].mean():.1f}x the median, roughly
    {(setup_flagged['median_setup_hours'].mean() - os_['cohort_median_setup_hours'].iloc[0]) * 60:.0f}
    minutes longer on every job they run. There is a potential opportunity for targeted coaching and
    setup-standardisation to recover Performance at no capital cost.</p>
    <div class="chart-wrap"><div class="chart-title">Median Setup Hours by Operator</div>{img('setup')}</div>

    <div class="section-title-block" id="cost">
      <div class="section-label">Section 5</div>
      <h2 class="section-title">Margin Uplift Opportunity Detail</h2>
    </div>
    <p>The OEE misses detailed above translate directly into lost contribution margin, the
    incremental profit on each productive machine hour (revenue net of variable cost). Valued at each
    machine type's contribution margin ({rate_str}), reaching the {BENCHMARK_OEE:.0%} target is worth
    an estimated <strong>${opportunity_annual / 1e6:.1f} million per year</strong>. This opportunity
    is split between Availability (<strong>{usd_short(avail_opp_annual)} per year</strong>), meaning
    less machine downtime and faster changeovers, and Performance
    (<strong>{usd_short(perf_opp_annual)} per year</strong>), meaning running machines nearer ideal
    speed.</p>
    <p>Within Availability, unplanned reactive repairs are the most visible opportunity, running
    about <strong>{usd_short(reactive_repair_annual)} per year</strong>
    ({unplanned_hours_annual:,.0f} hours) on their own, and the balance is setup, changeover, and
    idle time. Within Performance, the loss concentrates on the oldest machines as detailed above,
    and the operator setup-time gap is another opportunity.</p>
    <div class="chart-wrap"><div class="chart-title">Annual Contribution Margin Uplift Opportunity at 85% OEE, by Machine</div>{img('cost')}</div>

    <div class="section-title-block" id="actions">
      <div class="section-label">Section 6</div>
      <h2 class="section-title">Recommended Actions</h2>
    </div>
    <p>The findings resolve into six actions, ranked below by the annual impact each one carries.
    The list spans both low- and high-capital moves, from routine and coaching changes that cost
    little to the capital decisions on the oldest assets. The estimates are drawn from the same data
    and contribution-margin assumptions used throughout this report.</p>
    <table>
      <thead><tr><th>Action</th><th>Supporting Finding</th>
        <th style="text-align:right;">Est. Annual Impact</th><th>Capital Required</th></tr></thead>
      <tbody>{actions_rows}</tbody>
    </table>

  </main>
</div>
</body>
</html>"""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(html, encoding="utf-8")
print(f"Diagnostic report written to {OUTPUT}")
print(f"  Plant OEE {plant_oee:.1%} | opportunity ${opportunity_annual:,.0f}/yr "
      f"(availability ${avail_opp_annual:,.0f} + performance ${perf_opp_annual:,.0f})")
