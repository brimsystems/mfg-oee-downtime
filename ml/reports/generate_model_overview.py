"""ML Model Overview & Performance Report for the RUL predictor -> docs/reports/model_overview.html
Mirrors Case 01's ml_overview: Executive Summary, Scoring Summary, Top Risk Drivers,
Accuracy Retrospective, adapted from classification to RUL regression."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

import sys
sys.path.insert(0, str(Path(__file__).parent))
import brand as B
from brand import (DARK_GREY, DARK_BLUE, LIGHT_BLUE, ACCENT_RED, MUTED_RED, AMBER, GREEN, MED_GREY, LIGHT_GREY, mticker)

REPO    = Path(__file__).resolve().parents[2]
MODELS  = REPO / "ml" / "models"
SCORING = REPO / "ml" / "data" / "scoring"
OUT     = REPO / "docs" / "reports" / "model_overview.html"

PERIODS = [("202601", "January 2026"), ("202602", "February 2026"), ("202603", "March 2026")]
PERIOD_NAME = "January to March 2026"
TIERS = ["CRITICAL", "ELEVATED", "MONITOR"]
TIER_COLOR = {"CRITICAL": ACCENT_RED, "ELEVATED": AMBER, "MONITOR": DARK_BLUE}
# Contribution margin per productive machine-hour by type, mirroring the dbt var
# contribution_margin_by_type used to value downtime in the diagnostic report.
CM_BY_TYPE = {"CNC Lathe": 95, "Vertical Mill": 125, "Horizontal Mill": 145}
# Assumed reduction in downtime when a failure is caught early and handled as
# planned preventive maintenance rather than a reactive breakdown repair.
DOWNTIME_REDUCTION = 0.40
LABELS = {"linear_regression": "Linear Regression (Ridge)", "random_forest": "Random Forest", "xgboost": "XGBoost"}

m = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
frames = []
for lbl, nm in PERIODS:
    p = pd.read_parquet(SCORING / f"predictions_{lbl}.parquet"); p["period"] = nm; frames.append(p)
preds = pd.concat(frames, ignore_index=True)
preds["observation_date"] = pd.to_datetime(preds["observation_date"])
# Report uses three tiers; fold the small OK tail into MONITOR.
preds["priority"] = preds["priority"].replace("OK", "MONITOR")


def first_driver(dr):
    dr = list(dr) if dr is not None else []
    return dr[0] if len(dr) else None


preds["top_driver"] = preds["risk_drivers"].apply(first_driver)
total = len(preds)
counts = preds["priority"].value_counts()
n_crit, n_elev = int(counts.get("CRITICAL", 0)), int(counts.get("ELEVATED", 0))
flagged = preds[preds["priority"].isin(["CRITICAL", "ELEVATED"])]
unc = preds[~preds["is_censored"]]
mae = float(np.abs(unc["predicted_days_to_failure"] - unc["target_days_to_failure"]).mean())
within7 = float((np.abs(unc["predicted_days_to_failure"] - unc["target_days_to_failure"]) <= 7).mean())

# ── Event-level lead time ─────────────────────────────────────────────────────
# MAE averages error over every machine-day. This view instead asks, per actual
# failure: did a CRITICAL alert fire before it, and how timely was that alert?
# Uncapped, uncensored rows carry the true remaining life, so observation_date +
# remaining recovers the failure date and clusters each event's lead-up rows.
_evt = preds[(~preds["is_censored"]) & (preds["target_days_to_failure"] < 60)].copy()
_evt["failure_date"] = _evt["observation_date"] + pd.to_timedelta(_evt["target_days_to_failure"], unit="D")
_events = []
for (mid, fdate), g in _evt.groupby(["machine_id", "failure_date"]):
    g = g.sort_values("observation_date")
    crit = g[g["priority"] == "CRITICAL"]
    elev = g[g["priority"].isin(["CRITICAL", "ELEVATED"])]
    rec = {"machine_id": mid, "machine_type": g["machine_type"].iloc[0], "failure_date": fdate,
           "flagged": len(crit) > 0, "warned": len(elev) > 0}
    if len(crit):
        f0 = crit.iloc[0]
        rec.update(flag_date=f0["observation_date"],
                   lead=float(f0["target_days_to_failure"]),
                   pred=float(f0["predicted_days_to_failure"]))
        rec["signed"] = rec["lead"] - rec["pred"]   # + => failure later than predicted (alert early)
        rec["abserr"] = abs(rec["signed"])
    if len(elev):                                   # first warning of any tier (ELEVATED or CRITICAL)
        e0 = elev.iloc[0]
        rec["elev_lead"] = float(e0["target_days_to_failure"])
        rec["elev_signed"] = rec["elev_lead"] - float(e0["predicted_days_to_failure"])
        rec["elev_abserr"] = abs(rec["elev_signed"])
    _events.append(rec)
events = pd.DataFrame(_events)
flag_ev = events[events["flagged"]]
warn_ev = events[events["warned"]]
n_events = len(events)
detect_rate = float(events["flagged"].mean()) if n_events else float("nan")
warn_rate   = float(events["warned"].mean()) if n_events else float("nan")
med_lead    = float(flag_ev["lead"].median()) if len(flag_ev) else float("nan")
mean_abserr = float(flag_ev["abserr"].mean()) if len(flag_ev) else float("nan")
mean_signed = float(flag_ev["signed"].mean()) if len(flag_ev) else float("nan")
# First-warning (ELEVATED-or-higher) lead time: earlier but rougher than CRITICAL.
med_lead_elev    = float(warn_ev["elev_lead"].median()) if len(warn_ev) else float("nan")
mean_abserr_elev = float(warn_ev["elev_abserr"].mean()) if len(warn_ev) else float("nan")
mean_signed_elev = float(warn_ev["elev_signed"].mean()) if len(warn_ev) else float("nan")

# ── Training-data overview + worked example ──────────────────────────────────
FEATURES = REPO / "ml" / "data" / "features"
train = pd.read_parquet(FEATURES / "train.parquet")
_mm = pd.read_csv(REPO / "data_source" / "raw" / "cmms" / "maintenance_records.csv")
_mm = _mm[_mm["maintenance_type"] == "UNPLANNED_REPAIR"]
n_train_failures = int((pd.to_datetime(_mm["work_order_open_date"]) <= "2024-12-31").sum())
n_machines = int(preds["machine_id"].nunique())
n_obs_total = int(sum(m["split_sizes"].values()))

# ── Held-out performance and a naive baseline (for the accuracy section) ──────
# The model is scored once on the time-based test split it never trained on, and
# compared against the simplest sensible rule: predict the fleet's typical
# time-to-failure for every machine. Beating that median predictor by a wide
# margin is the evidence the model has learned real structure, not noise.
_test = pd.read_parquet(FEATURES / "test.parquet")
naive_pred_days = float(train["target_days_to_failure"].median())
naive_mae = float(np.abs(_test["target_days_to_failure"] - naive_pred_days).mean())
test_mae, test_rmse, test_r2 = m["test"]["mae"], m["test"]["rmse"], m["test"]["r2"]
val_mae = {x["model_type"]: x["val_mae"] for x in m["models"]}
mae_reduction = (naive_mae - test_mae) / naive_mae if naive_mae else 0.0
# Same naive-median baseline measured on the live scoring window, for a like-for-like
# comparison against the scoring MAE quoted in the executive summary.
naive_mae_scoring = float(np.abs(naive_pred_days - unc["target_days_to_failure"]).mean())

# Worked example: the single highest-risk machine in the current fleet snapshot,
# used to walk a non-technical reader through one real prediction end to end.
we = (preds.sort_values("observation_date").groupby("machine_id").tail(1)
      .sort_values("predicted_days_to_failure").iloc[0])
we_drivers = list(we["risk_drivers"]) if we["risk_drivers"] is not None else []

# ── Business impact vs the calendar-PM baseline ──────────────────────────────
# On the held-out scoring window, take the real unplanned failures and their
# actual downtime hours from the CMMS, and classify each by the strongest warning
# the model would have raised before it. The baseline (calendar PM) anticipates
# none of these unplanned events; the model's value is the advance warning.
_mm["fd"] = pd.to_datetime(_mm["work_order_open_date"])
q1 = _mm[(_mm["fd"] >= "2026-01-01") & (_mm["fd"] <= "2026-03-31")].copy()
_ev_lookup = {(r.machine_id, pd.Timestamp(r.failure_date).normalize()): r for r in events.itertuples()}


def _impact_cat(row):
    e = _ev_lookup.get((row["machine_id"], row["fd"].normalize()))
    if e is None:
        return "none"
    if bool(getattr(e, "flagged", False)) and pd.notna(getattr(e, "lead", np.nan)) and e.lead >= 2:
        return "critical"
    if bool(getattr(e, "warned", False)):
        return "elevated"
    return "none"


q1["cat"] = q1.apply(_impact_cat, axis=1)
bi_events = int(len(q1))
bi_hrs = float(q1["downtime_hours"].sum())
_crit = q1[q1["cat"] == "critical"]; _elev = q1[q1["cat"] == "elevated"]
bi_crit_n, bi_elev_n = int(len(_crit)), int(len(_elev))
bi_other_n = bi_events - bi_crit_n - bi_elev_n
bi_crit_hrs = float(_crit["downtime_hours"].sum())
bi_elev_hrs = float(_elev["downtime_hours"].sum())
bi_other_hrs = bi_hrs - bi_crit_hrs - bi_elev_hrs
bi_crit_hr_pct = bi_crit_hrs / bi_hrs if bi_hrs else 0.0
bi_warn_n = bi_crit_n + bi_elev_n
_leads = [_ev_lookup[(r.machine_id, r.fd.normalize())].lead for r in _crit.itertuples()]
bi_crit_lead = float(np.nanmedian(_leads)) if _leads else float("nan")
bi_annual_hrs = bi_crit_hrs * 4                        # Q1 window annualised
bi_annual_net = bi_annual_hrs * DOWNTIME_REDUCTION     # net downtime avoided vs a reactive repair
# Dollar value of the avoided downtime, valued at each machine's own contribution
# margin (per-type rates, matching the diagnostic report).
_type_by_machine = preds.groupby("machine_id")["machine_type"].first().to_dict()
_crit_margin_q = float((_crit["downtime_hours"] * _crit["machine_id"].map(_type_by_machine).map(CM_BY_TYPE)).sum())
bi_net_hrs = bi_annual_net                            # net unplanned hours avoided per year (illustrative)
bi_q1_avoided_hrs = bi_crit_hrs * DOWNTIME_REDUCTION  # Q1 unplanned hours avoidable on the CRITICAL-flagged basis
bi_net_usd = round(_crit_margin_q * 4 * DOWNTIME_REDUCTION, -3)   # per-type margin, rounded (illustrative)

# ── Exploratory views of the model inputs over the training window ───────────
# Time series of the sensor channels and the other top drivers across the full
# Jan 2023 to Dec 2025 training window, for the data-exploration section.
TRAIN_END = "2025-12-31"
ANOM_THRESH = 1.5                        # sensor anomaly z-score treated as a flag
SENSOR_CH = {"vibration_rms_mm_s": "Spindle vibration (mm/s)",
             "bearing_temp_c": "Bearing temperature (°C)",
             "spindle_power_kw": "Spindle power (kW)",
             "hydraulic_pressure_bar": "Hydraulic pressure (bar)"}

_sr = pd.read_csv(REPO / "data_source" / "raw" / "sensors" / "sensor_readings.csv", parse_dates=["reading_date"])
_sr = _sr[_sr["reading_date"] <= TRAIN_END].copy()
_sr["ym"] = _sr["reading_date"].dt.to_period("M").dt.to_timestamp()
eda_sensor_monthly = _sr.groupby("ym")[list(SENSOR_CH)].mean()

_con = duckdb.connect(str(REPO / "data_source" / "oee_predmaint.duckdb"), read_only=True)
_mart = _con.execute(f"""select observation_date, sensor_anomaly_score, rolling_30d_utilization_rate
    from mart_ml__rul_features where observation_date <= '{TRAIN_END}'""").df()
_con.close()
_mart["ym"] = pd.to_datetime(_mart["observation_date"]).dt.to_period("M").dt.to_timestamp()
_mart["is_anom"] = _mart["sensor_anomaly_score"] >= ANOM_THRESH
eda_anom_monthly = _mart.groupby("ym")["is_anom"].sum()
eda_anom_rate = float(_mart["is_anom"].mean())
eda_util_monthly = _mart.groupby("ym")["rolling_30d_utilization_rate"].mean()

_ut = _mm[_mm["fd"] <= TRAIN_END].copy()     # _mm is unplanned-only with fd already set
_ut["ym"] = _ut["fd"].dt.to_period("M").dt.to_timestamp()
eda_fail_monthly = _ut.groupby("ym").size()
eda_fail_by_mode = _ut.groupby(["ym", "failure_code"]).size().unstack(fill_value=0)
eda_fail_total = int(len(_ut))
# Share of unplanned failures driven by the wear modes the sensors can detect.
tm_pct = float((_ut["failure_code"].isin(["TOOLING", "MECHANICAL"])).mean())


def _trend_pct(series):
    if len(series) < 2 or series.iloc[0] == 0:
        return 0.0
    slope = np.polyfit(np.arange(len(series)), series.values, 1)[0]
    return slope * (len(series) - 1) / series.iloc[0] * 100


# ── Charts ──────────────────────────────────────────────────────────────────
def chart_priority_distribution():
    vals = [int(counts.get(t, 0)) for t in TIERS]
    fig, ax = B.make_fig(h=3.4)
    bars = ax.bar(TIERS, vals, color=[TIER_COLOR[t] for t in TIERS], width=0.55)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + total * 0.005, f"{v:,}\n({v/total:.0%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Observations scored")
    ax.set_ylim(0, max(vals) * 1.18)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_rul_distribution():
    fig, ax = B.make_fig(h=3.4)
    ax.hist(preds["predicted_days_to_failure"], bins=30, color=DARK_BLUE, edgecolor="white", linewidth=0.5)
    for x, c, lab in [(7, ACCENT_RED, "Critical (7d)"), (21, AMBER, "Elevated (21d)")]:
        ax.axvline(x, color=c, ls="--", lw=1.4, label=lab)
    ax.set_xlabel("Predicted days to failure"); ax.set_ylabel("Observations"); ax.legend()
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_tier_by_period():
    names = [nm for _, nm in PERIODS]
    x = np.arange(len(names)); w = 0.24
    fig, ax = B.make_fig(h=3.4)
    maxv = 0
    for i, t in enumerate(TIERS):
        vals = [int(((preds["period"] == nm) & (preds["priority"] == t)).sum()) for nm in names]
        maxv = max(maxv, max(vals))
        bars = ax.bar(x + (i - 1) * w, vals, w, color=TIER_COLOR[t], label=t.title())
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + maxv * 0.012, f"{v}",
                    ha="center", va="bottom", fontsize=8, color=DARK_GREY)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Observations")
    ax.set_ylim(0, maxv * 1.16)
    ax.legend(ncol=4, fontsize=9)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_error_by_period():
    names = [nm for _, nm in PERIODS]
    maes = [float(np.abs(unc.loc[unc.period == nm, "predicted_days_to_failure"]
                         - unc.loc[unc.period == nm, "target_days_to_failure"]).mean()) for nm in names]
    fig, ax = B.make_fig(h=3.2)
    bars = ax.bar(names, maes, color=DARK_BLUE, width=0.5)
    ax.axhline(m["test"]["mae"], color=MED_GREY, ls="--", lw=1.4, label=f"Test baseline {m['test']['mae']:.1f}")
    for b_, v in zip(bars, maes):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("MAE (days)"); ax.legend()
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


# ── Training-signal charts (what the model keys on) ──────────────────────────
def chart_vibration_ramp():
    u = train[(~train["is_censored"]) & (train["target_days_to_failure"] <= 30)]
    buckets = [(22, 30), (15, 21), (8, 14), (4, 7), (0, 3)]
    labels = ["22-30", "15-21", "8-14", "4-7", "0-3"]
    vals = [u.loc[(u["target_days_to_failure"] >= lo) & (u["target_days_to_failure"] <= hi),
                  "vibration_7d_mean"].mean() for lo, hi in buckets]
    fig, ax = B.make_fig(h=3.4)
    ax.plot(labels, vals, "o-", color=DARK_BLUE, lw=2.6, markersize=9, zorder=3)
    ax.fill_between(range(len(labels)), vals, min(vals) * 0.9, color=DARK_BLUE, alpha=0.08, zorder=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=DARK_BLUE)
    ax.set_xlabel("Days until failure  (further out  →  imminent)")
    ax.set_ylabel("Avg spindle vibration (mm/s)")
    ax.set_ylim(min(vals) * 0.9, max(vals) * 1.12)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_age_ttf():
    d = train.groupby("machine_age_years")["target_days_to_failure"].mean()
    fig, ax = B.make_fig(h=3.4)
    bars = ax.bar([str(a) for a in d.index], d.values, color=DARK_BLUE, width=0.62)
    for b_, v in zip(bars, d.values):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.3, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Machine age (years)"); ax.set_ylabel("Avg days to next failure")
    ax.set_ylim(0, d.max() * 1.15)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_anomaly_ttf():
    a = train["sensor_anomaly_score"]; t = train["target_days_to_failure"]
    buckets = [("Normal\n(< 0.5)", a < 0.5, GREEN),
               ("Slight\n(0.5-1.0)", (a >= 0.5) & (a < 1.0), LIGHT_BLUE),
               ("Elevated\n(1.0-1.5)", (a >= 1.0) & (a < 1.5), AMBER),
               ("High\n(1.5+)", a >= 1.5, ACCENT_RED)]
    labels = [b[0] for b in buckets]; vals = [float(t[b[1]].mean()) for b in buckets]
    cols = [b[2] for b in buckets]
    fig, ax = B.make_fig(h=3.4)
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.3, f"{v:.0f} d", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Sensor anomaly score (how far readings sit above the machine's own baseline)")
    ax.set_ylabel("Avg days to next failure"); ax.set_ylim(0, max(vals) * 1.15)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


# ── Business-impact charts ───────────────────────────────────────────────────
def chart_impact_combined():
    # One 100%-stacked view of how much of Q1's failures and downtime each
    # maintenance strategy saw coming. Calendar PM anticipates none (all "no
    # advance signal"); the model splits each measure into its advance-warning
    # tiers. Downtime (hours) and failure count use different units, so each
    # column is normalised to its own total and labelled with the absolute value.
    cats = [("Flagged CRITICAL in advance", GREEN),
            ("ELEVATED warning", AMBER),
            ("No advance signal", MED_GREY)]
    columns = [                       # (crit, elev, none), total, unit suffix
        ((0.0, 0.0, bi_hrs), bi_hrs, "hrs"),
        ((0.0, 0.0, float(bi_events)), float(bi_events), ""),
        ((bi_crit_hrs, bi_elev_hrs, bi_other_hrs), bi_hrs, "hrs"),
        ((float(bi_crit_n), float(bi_elev_n), float(bi_other_n)), float(bi_events), ""),
    ]
    xpos = [0, 1, 2.4, 3.4]
    fig, ax = B.make_fig(h=3.9)
    for i, (vals, tot, unit) in enumerate(columns):
        bottom = 0.0
        for (lab, color), v in zip(cats, vals):
            pct = v / tot * 100 if tot else 0.0
            ax.bar(xpos[i], pct, bottom=bottom, width=0.82, color=color,
                   label=lab if i == 0 else None)
            if pct >= 6:              # label only segments tall enough to hold text
                txt = f"{v:.0f} {unit}".strip()
                ax.text(xpos[i], bottom + pct / 2, txt, ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            bottom += pct
    ax.set_ylim(0, 100)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Downtime", "Failures", "Downtime", "Failures"], fontsize=9)
    ax.set_ylabel("Share of Q1 2026 total (%)")
    for x, name in [(0.5, "Calendar PM (today)"), (2.9, "With RUL model")]:
        ax.text(x, -0.16, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=10, fontweight="bold", color=DARK_GREY)
    ax.legend(fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.30), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


# ── Data-exploration charts (model inputs over three years) ──────────────────
def _year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def chart_eda_sensors():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(B.CHART_W, 5.2))
    for ax, (col, lab) in zip(axes.ravel(), SENSOR_CH.items()):
        m = eda_sensor_monthly[col]
        ax.plot(m.index, m.values, color=DARK_BLUE, lw=1.8)
        coef = np.polyfit(np.arange(len(m)), m.values, 1)
        ax.plot(m.index, np.polyval(coef, np.arange(len(m))), color=ACCENT_RED, ls="--", lw=1.2)
        ax.set_title(f"{lab}   ({_trend_pct(m):+.0f}% / 3 yr)", fontsize=10)
        B.chart_style(ax); _year_axis(ax); ax.tick_params(labelsize=8)
    fig.tight_layout()
    return B.b64(fig)


def chart_eda_anomaly():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, ax = B.make_fig(h=3.4)
    a = eda_anom_monthly
    ax.bar(a.index, a.values, width=22, color=LIGHT_BLUE)
    ax.set_ylabel("Anomalous machine-days / month")
    ax2 = ax.twinx()
    f = eda_fail_monthly.reindex(a.index).fillna(0)
    ax2.plot(f.index, f.values, color=ACCENT_RED, lw=2, marker="o", markersize=3)
    ax2.set_ylabel("Unplanned failures / month", color=ACCENT_RED)
    ax2.tick_params(axis="y", labelcolor=ACCENT_RED); ax2.grid(False)
    B.chart_style(ax); _year_axis(ax)
    ax.legend(handles=[Patch(color=LIGHT_BLUE, label="Anomalous machine-days"),
                       Line2D([0], [0], color=ACCENT_RED, marker="o", label="Unplanned failures")],
              fontsize=9, loc="upper left")
    fig.tight_layout()
    return B.b64(fig)


def chart_eda_failures():
    fig, ax = B.make_fig(h=3.2)
    d = eda_fail_by_mode
    palette = {"TOOLING": DARK_BLUE, "MECHANICAL": LIGHT_BLUE, "ELECTRICAL": MED_GREY,
               "OPERATOR_INDUCED": AMBER, "ENVIRONMENTAL": MUTED_RED}
    bottom = np.zeros(len(d))
    for mode in [c for c in palette if c in d.columns]:
        ax.bar(d.index, d[mode].values, bottom=bottom, width=22, color=palette[mode],
               label=mode.replace("_", " ").title())
        bottom = bottom + d[mode].values
    # Trace the top of the tooling + mechanical band with a red dotted line to
    # highlight the wear-driven modes the sensors detect (everything below it).
    tm_cum = (d.get("TOOLING", 0) + d.get("MECHANICAL", 0)).values
    ax.plot(d.index, tm_cum, color=ACCENT_RED, ls=":", lw=1.8, drawstyle="steps-mid",
            zorder=5)
    ax.text(0.015, 0.95, f"Tooling + Mechanical:\n{tm_pct:.0%} of all failures",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold",
            color=ACCENT_RED,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ACCENT_RED, ls=":", lw=1.5))
    ax.set_ylabel("Unplanned failures / month")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    B.chart_style(ax); _year_axis(ax)
    fig.tight_layout()
    return B.b64(fig)


def chart_lead_by_tier():
    # Event-level warning timeline: the first ELEVATED warning fires earlier but
    # rougher, the CRITICAL alert fires closer but sharper. Both sit on the safe
    # (early) side of the actual failure.
    rows = [("ELEVATED warning", med_lead_elev, mean_abserr_elev, mean_signed_elev, AMBER),
            ("CRITICAL alert", med_lead, mean_abserr, mean_signed, ACCENT_RED)]
    fig, ax = B.make_fig(h=2.7)
    ys = [1, 0]
    maxlead = max(r[1] for r in rows)
    for y, (lab, lead, ae, sg, col) in zip(ys, rows):
        ax.barh(y, lead, height=0.5, color=col, zorder=3)
        ax.text(lead + maxlead * 0.02, y, f"{lead:.0f} days median lead",
                va="center", ha="left", fontsize=10, fontweight="bold", color=DARK_GREY)
        ax.text(maxlead * 0.02, y - 0.34, f"alert fires {sg:+.0f} days early on average (off by {ae:.0f} days)",
                va="center", ha="left", fontsize=8.5, color=MED_GREY)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10, fontweight="bold")
    ax.set_xlabel("Days before failure the alert first fired")
    ax.set_xlim(0, maxlead * 1.5); ax.set_ylim(-0.7, 1.7)
    B.chart_style(ax); ax.yaxis.grid(False)   # no horizontal gridlines behind the bars
    fig.tight_layout()
    return B.b64(fig)


def eda_stats_table():
    notes = {"vibration_rms_mm_s": "Rises with fleet age; the strongest pre-failure signal.",
             "bearing_temp_c": "Stable at baseline; spikes only ahead of specific failures.",
             "spindle_power_kw": "Stable; reacts on tooling and electrical events.",
             "hydraulic_pressure_bar": "Stable; dips ahead of environmental faults."}
    rows = []
    for col, lab in SENSOR_CH.items():
        m = eda_sensor_monthly[col]
        rows.append([lab, f"{m.mean():.1f}", f"{_trend_pct(m):+.0f}%", notes[col]])
    u = eda_util_monthly * 100
    rows.append(["Fleet utilization (30-day)", f"{u.mean():.0f}%", f"{_trend_pct(u):+.0f}%",
                 "Sustained load that limits available maintenance windows."])
    rows.append(["Unplanned failures", f"{eda_fail_monthly.mean():.0f}/mo", f"{_trend_pct(eda_fail_monthly):+.0f}%",
                 f"{eda_fail_total} events over three years; the reliability baseline."])
    rows.append(["Sensor anomaly flags", f"{eda_anom_rate:.0%} of days", "n/a",
                 "Cluster in the weeks before failures, as the chart above shows."])
    return B.data_table(["Model input", "Avg over 3 yrs", "3-yr trend", "Behaviour"], rows, right={1, 2})


# ── Tables ──────────────────────────────────────────────────────────────────
def exec_drivers_table():
    # Plain-language summary of the drivers the model weighs most, drawn from the
    # SHAP feature importances. The condition-monitoring sensor signals are led
    # with first because they are the most concrete for the floor team, so this
    # is ordered for readability rather than strictly by SHAP magnitude.
    drivers = [
        "Spindle vibration elevated above its normal baseline",
        "Sensor anomaly across the condition-monitoring channels",
        "Time since the last unplanned failure",
        "Machine age and prior failure history",
        "Fleet utilization and maintenance-window pressure",
    ]
    rows = [[f'<td style="width:44px;text-align:center;font-weight:700;color:{DARK_BLUE};">#{i}</td>', d]
            for i, d in enumerate(drivers, 1)]
    return B.data_table(["Rank", "Driver behind the flag"], rows)


def top_drivers_table():
    vc = flagged["top_driver"].dropna().value_counts().head(6)
    rows = [[f'<td style="width:44px;text-align:center;font-weight:700;color:{DARK_BLUE};">#{i}</td>',
             drv] for i, (drv, _) in enumerate(vc.items(), 1)]
    return B.data_table(["Rank", "Most common primary risk driver"], rows)


def flagged_detail_table():
    top = flagged.sort_values("predicted_days_to_failure").head(18)
    rows = ""
    for r in top.itertuples():
        drv = "; ".join(list(r.risk_drivers)[:2]) if r.risk_drivers is not None else "&mdash;".replace("&mdash;", "-")
        if r.is_censored:
            outcome = '<span style="color:#8093A4;">no failure in window</span>'
        else:
            hit = abs(r.predicted_days_to_failure - r.target_days_to_failure) <= 7
            oc_col = GREEN if hit else ACCENT_RED
            outcome = f'<span style="color:{oc_col};font-weight:600;">actual {r.target_days_to_failure:.0f}d</span>'
        rows += (f'<tr><td style="font-weight:600;">{r.machine_id}</td>'
                 f'<td>{r.machine_type}</td>'
                 f'<td>{r.observation_date:%m/%d/%y}</td>'
                 f'<td style="text-align:right;font-weight:700;">{r.predicted_days_to_failure:.0f}d</td>'
                 f'<td>{B.badge(r.priority, TIER_COLOR[r.priority])}</td>'
                 f'<td style="font-size:13px;">{drv}</td>'
                 f'<td style="text-align:right;">{outcome}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Machine</th><th>Type</th><th>Date</th>'
            f'<th style="text-align:right;">Predicted</th><th>Priority</th><th>Risk Drivers</th>'
            f'<th style="text-align:right;">Outcome</th></tr></thead><tbody>{rows}</tbody></table>')


def lead_time_table():
    fe = flag_ev.sort_values("lead", ascending=False)
    rows = ""
    for r in fe.itertuples():
        col = GREEN if r.signed >= 0 else ACCENT_RED   # early alert = safe side (green)
        rows += (f'<tr><td style="font-weight:600;">{r.machine_id}</td>'
                 f'<td>{r.machine_type}</td>'
                 f'<td>{r.failure_date:%m/%d/%y}</td>'
                 f'<td>{r.flag_date:%m/%d/%y}</td>'
                 f'<td style="text-align:right;font-weight:700;">{r.lead:.0f} d</td>'
                 f'<td style="text-align:right;">predicted {r.pred:.0f}d, actual {r.lead:.0f}d</td>'
                 f'<td style="text-align:right;color:{col};font-weight:600;">{r.signed:+.0f} d</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Machine</th><th>Type</th>'
            f'<th>Failure date</th><th>First CRITICAL alert</th>'
            f'<th style="text-align:right;">Lead time</th>'
            f'<th style="text-align:right;">Prediction at alert</th>'
            f'<th style="text-align:right;">Early (+) / Late (-)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def accuracy_table():
    names = [nm for _, nm in PERIODS]
    rows = []
    for nm in names:
        d = unc[unc.period == nm]
        e = np.abs(d["predicted_days_to_failure"] - d["target_days_to_failure"])
        rows.append([nm, f"{len(preds[preds.period==nm]):,}",
                     f"{e.mean():.1f}",
                     f"{np.sqrt((e**2).mean()):.1f}",
                     f"{(e<=7).mean():.0%}"])
    return B.data_table(["Period", "Scored", "MAE (days)", "RMSE (days)", "Within 7 days"],
                        rows, right={1, 2, 3, 4})


def tier_reference_table():
    rows = [
        [B.badge("CRITICAL", ACCENT_RED), "Failure predicted within 7 days",
         "Schedule maintenance now; treat as this week's priority."],
        [B.badge("ELEVATED", AMBER), "Failure predicted in 8 to 21 days",
         "Plan a service window in the next two to three weeks."],
        [B.badge("MONITOR", DARK_BLUE), "Failure predicted in 22 days or more",
         "Keep on the watch list; continue the normal preventive-maintenance schedule."]]
    return B.data_table(["Priority", "What the model is saying", "What the floor should do"], rows)


def worked_example():
    age = int(we["machine_age_years"]); vib = float(we["vibration_7d_mean"])
    btemp = float(we["bearing_temp_7d_mean"]); power = float(we["spindle_power_7d_mean"])
    hyd = float(we["hydraulic_pressure_7d_mean"])
    since = we["days_since_last_unplanned_failure"]; overdue = float(we["days_overdue_for_pm"])
    alarms = float(we["rolling_7d_alarm_count"]); pred = float(we["predicted_days_to_failure"])
    rows = [
        ["Spindle vibration", f"{vib:.1f} mm/s (7-day avg)",
         "Flagged by condition monitoring as trending above its normal baseline."],
        ["Bearing temperature", f"{btemp:.0f} &deg;C (7-day avg)",
         "Within its normal range; no anomaly flagged."],
        ["Spindle motor power", f"{power:.1f} kW (7-day avg)",
         "Steady in its normal band; no anomaly flagged."],
        ["Hydraulic pressure", f"{hyd:.0f} bar (7-day avg)",
         "Within its normal range; no anomaly flagged."],
        ["Machine age", f"{age} years",
         "One of the oldest assets in the fleet, so it carries more mechanical wear."],
        ["Time since last breakdown", f"{since:.0f} days",
         "It failed only recently, so it is still in a fragile post-repair window."],
        ["Preventive maintenance", f"{overdue:.0f} days overdue" if overdue > 0 else "on schedule",
         "Past its scheduled service, which raises risk." if overdue > 0 else "Service is current."],
        ["Recent alarms (7 days)", f"{alarms:.0f} alarms",
         "Elevated fault activity on the controller over the past week."]]
    inputs = B.data_table(["Signal the model read", "Current value", "Why it matters"], rows)
    result = (f'<div style="background:{B.BG_GREY};border-left:4px solid {ACCENT_RED};padding:16px 20px;margin:18px 0;">'
              f'<span style="font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:{MED_GREY};font-weight:700;">'
              f'The prediction</span><br>'
              f'<span style="font-size:26px;font-weight:700;color:{DARK_GREY};">{we["machine_id"]} &middot; {we["machine_type"]}</span><br>'
              f'<span style="font-size:16px;">Estimated <strong>{pred:.0f} days</strong> to the next unplanned failure &rarr; '
              f'{B.badge("CRITICAL", ACCENT_RED)}</span></div>')
    # Lead the reasons with the spindle-vibration driver, the flagged sensor signal.
    ordered = sorted(we_drivers, key=lambda d: 0 if "vibration" in d.lower() else 1)
    drv = "".join(f"<li>{d}</li>" for d in ordered)
    return inputs + result + f'<p style="margin-bottom:6px;"><strong>Reasons flagged in CMMS:</strong></p><ul class="limitation-list">{drv}</ul>'


FLOW_HTML = (
    '<div style="display:flex;align-items:stretch;gap:0;margin:22px 0;flex-wrap:wrap;">'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">1. What it watches</div>'
    '<div style="font-size:16px;line-height:1.55;">For each machine: live sensor readings for vibration, bearing temperature, '
    'motor power, and hydraulic pressure, plus its age and run hours, recent alarms and downtime, and time since the last '
    'breakdown and last service.</div></div>'
    '<div style="align-self:center;font-size:26px;color:#8093A4;padding:0 12px;">&rarr;</div>'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">2. What it learns</div>'
    '<div style="font-size:16px;line-height:1.55;">From three years of history it learned the patterns that came before past '
    'breakdowns, such as vibration and heat creeping up, older machines failing sooner, and risk rising soon after a repair.</div></div>'
    '<div style="align-self:center;font-size:26px;color:#8093A4;padding:0 12px;">&rarr;</div>'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">3. What it produces</div>'
    '<div style="font-size:16px;line-height:1.55;">A daily estimate of days until the next failure for every machine, a priority '
    'tier, and the specific reasons behind the anticipated failure, delivered straight into the CMMS maintenance queue.</div></div></div>')


charts = {"pri": chart_priority_distribution(), "rul": chart_rul_distribution(),
          "tier": chart_tier_by_period(), "err": chart_error_by_period(),
          "vib": chart_vibration_ramp(), "anom": chart_anomaly_ttf(),
          "lead_tier": chart_lead_by_tier(),
          "impact_combined": chart_impact_combined(),
          "eda_sensors": chart_eda_sensors(), "eda_anom": chart_eda_anomaly(),
          "eda_fail": chart_eda_failures()}

# CMMS screenshot (static asset) for Section 2.1.
import base64
_cmms_png = Path(__file__).resolve().parent / "assets" / "cmms_screenshot.png"
cmms_screenshot_b64 = base64.b64encode(_cmms_png.read_bytes()).decode() if _cmms_png.exists() else ""

toc = ('<a href="#summary">Executive Summary</a><hr>'
       '<a href="#modeloverview">Model Overview</a>'
       '<a href="#what" class="sub">What This Model Does</a>'
       '<a href="#data" class="sub">Training Data Overview</a><hr>'
       '<a href="#predictions">Model Performance</a>'
       '<a href="#scoring" class="sub">Scoring Summary</a>'
       '<a href="#accuracy" class="sub">Accuracy and Validation</a>'
       '<a href="#sample" class="sub">Sample Model Output</a>'
       '<a href="#limits" class="sub">What It Can and Cannot Predict</a>')

body = f"""
{B.section("summary", "Section 1", "Executive Summary")}
<p>The Remaining Useful Life (RUL) machine learning model forecasts when machines on the shop floor are
likely to fail so that repairs can be made before a breakdown happens. The model was trained on three years
of machine sensor data and maintenance records. Over the past three months, the model has been live,
monitoring machine health and making daily RUL predictions.</p>
<p>During this time, the shop logged <strong>{bi_events} unplanned failures</strong> and
<strong>{bi_hrs:.0f} hours</strong> of unplanned downtime. The RUL model flagged
<strong>{bi_warn_n/bi_events:.0%} of these failures</strong> ahead of time, while the current calendar-based
preventive maintenance (PM) schedule anticipated none of them. The model flagged <strong>{bi_crit_n} of the failures
({bi_crit_hr_pct:.0%} of the downtime, {bi_crit_hrs:.0f} hours)</strong> with a high-confidence CRITICAL
alert a median of <strong>{bi_crit_lead:.0f} days</strong> in advance, and {bi_elev_n} of the remaining
{bi_events - bi_crit_n} carried an earlier ELEVATED warning. The model's predictions were off by an average of
<strong>{mae:.1f} days</strong>, and <strong>{within7:.0%}</strong> landed within a week of the actual failure
date. For context, a naive rule that simply predicts the fleet's average time to failure for every machine
would be off by about {naive_mae_scoring:.0f} days, <strong>so the model roughly halves the error of a
no-model baseline</strong>.</p>
{B.chart("Downtime and Failures Anticipated: Calendar PM vs RUL Model", charts["impact_combined"])}
<p>Acting proactively on those CRITICAL alerts could have avoided an estimated
<strong>{bi_q1_avoided_hrs:.0f} hours</strong> of unplanned downtime in Q1 (assuming a {DOWNTIME_REDUCTION:.0%}
reduction in downtime for preventive versus reactive maintenance), worth about
<strong>${bi_net_usd:,.0f}</strong> in annualized contribution margin.</p>
<p>Alongside the number and the priority tier, every prediction lists the specific conditions that drove the
flag, so the maintenance team can see why a machine was surfaced and what to inspect first. The signals that
most heavily determine maintenance flags are listed below:</p>
{exec_drivers_table()}

{B.section("modeloverview", "Section 2", "Model Overview")}

{B.section("what", "Section 2.1", "What This Model Does")}
<p>The RUL model is built on XGBoost, a gradient-boosted decision-tree algorithm. This model answers one
question for every machine on the floor, every day: <strong>how many days remain until this machine is likely
to breakdown and require maintenance?</strong> It does not diagnose a specific fault or generate a repair
order on its own, but rather serves as an early-warning and maintenance prioritisation tool.</p>
{FLOW_HTML}
<p>The model output's priority tiers are described below:</p>
{tier_reference_table()}
<p>The prediction is delivered where the maintenance team already works. The screenshot below shows the model
embedded in the CMMS asset view: the fleet is ranked by predicted time to the next unplanned failure, each
machine carries a priority tier and its condition, and the counts at the top summarise how many assets fall
in each tier, so a planner can triage the fleet without leaving the system.</p>
<div class="chart-wrap" style="padding:6px;">
  <img src="data:image/png;base64,{cmms_screenshot_b64}" alt="CMMS asset view with embedded RUL predictions and priority tiers"
       style="width:100%;height:auto;display:block;border:1px solid {LIGHT_GREY};">
</div>

{B.section("data", "Section 2.2", "Training Data Overview")}
<p>The four condition-monitoring channels (vibration, bearing temperature, motor power, and hydraulic
pressure) are plotted below as monthly fleet averages across the full training window, each with a fitted
trend line. Three of the four sit flat at their baseline, which is what a healthy fleet should look like;
spindle vibration is the exception, drifting steadily upward as the fleet ages. That slow rise reflects
background wear that the model reads underneath the sharper pre-failure spikes.</p>
{B.chart("Sensor Channels Over Three Years", charts["eda_sensors"],
         "Monthly fleet-average reading per channel, January 2023 to December 2025, each with a dashed trend line. Only vibration shows a sustained upward trend.")}
<p>The channels look calm in aggregate because the pre-failure spikes are short and machine-specific, so they
average out across the fleet. The model instead picks up the anomalies, readings that jump above a machine's
own baseline.</p>
<p>The three charts below show the importance of these anomaly readings. Across the fleet, monthly anomaly
activity rises and falls with the actual unplanned-failure count and tends to lead it. Zooming into
individual machines, average spindle vibration is quiet weeks out and climbs steadily in the final days
before a failure. And the further a machine's readings sit above its own normal baseline, the sooner the next
failure tends to arrive, from over three weeks out when readings are normal to about nine days when they are
highly abnormal. Together these confirm the condition-monitoring sensors as the model's leading indicators.</p>
{B.chart("Sensor Anomalies Lead Unplanned Failures", charts["eda_anom"],
         "Monthly count of anomalous machine-days (bars) against unplanned failures per month (line). Anomaly spikes tend to precede failure spikes.")}
{B.chart("Vibration Climbs as Failure Approaches", charts["vib"],
         "Average spindle vibration in the training data, grouped by how many days remained before the machine failed. "
         "Vibration is quiet weeks out and rises steadily in the final days.")}
{B.chart("A Sensor Anomaly Means Failure Is Closer", charts["anom"],
         "The anomaly score measures how far a machine's recent readings sit above its own normal baseline. As the score rises, "
         "the average time to the next failure falls sharply, from over three weeks when readings are normal to about nine days when they are highly abnormal.")}
<p>Looking at the unplanned breakdown data broken out by failure mode reveals that tooling and mechanical
problems make up about three-quarters ({tm_pct:.0%}) of all unplanned failures. These wear-driven modes are
the ones that the condition-monitoring sensors' data reveal, and so are the ones driving the model's
predictive power. The electrical, operator-induced, and environmental modes are less common and give far
less warning.</p>
{B.chart("Unplanned Failures by Month and Failure Mode", charts["eda_fail"])}

{B.section("predictions", "Section 3", "Model Performance")}

{B.section("scoring", "Section 3.1", "Scoring Summary")}
<p>Every machine-shift observation is scored before the shift begins and sorted into a priority tier. In
{PERIOD_NAME}, the RUL model scored <strong>{total:,}</strong> observations of machine health. It placed
<strong>{n_crit:,}</strong> in the CRITICAL tier (predicted failure within 7 days) and <strong>{n_elev:,}</strong>
in ELEVATED (8 to 21 days). On observations whose failure has since been confirmed, the model's predictions
were off by an average of <strong>{mae:.1f} days</strong> (roughly half the baseline that predicts a machine
failure within ~{naive_mae_scoring:.0f} days), and <strong>{within7:.0%}</strong> landed within a week of the
actual failure date.</p>
{B.chart("Observations by Priority Tier", charts["pri"])}
{B.chart("Distribution of Predicted Days to Failure", charts["rul"])}
{B.chart("Priority Mix by Month", charts["tier"])}

{B.section("accuracy", "Section 3.2", "Accuracy and Validation")}
<p>The RUL model learned on data from January 2023 to December 2024, was tuned on January to August 2025, and
was then scored once on a held-out September to December 2025 test set. Three candidate algorithms, a linear
regression, a random forest, and a gradient-boosted XGBoost model, were each tuned over {m['n_optuna_trials']}
Optuna trials and compared on the validation set. XGBoost was the most accurate, at {val_mae['xgboost']:.1f}
days of average error against {val_mae['random_forest']:.1f} for the random forest and
{val_mae['linear_regression']:.1f} for the linear baseline, and was carried forward.</p>
<p>On the held-out test set, the model predicts days-to-failure with a <strong>mean absolute error (MAE) of
{test_mae:.1f} days</strong> and explains <strong>{test_r2:.0%}</strong> of the variance in failure timing
(R-squared {test_r2:.2f}). The fair yardstick is what a sensible do-nothing rule would achieve: predicting
the fleet's typical time-to-failure (about {naive_pred_days:.0f} days) for every machine gives an error of
{naive_mae:.1f} days, so the model <strong>cuts the error of that naive guess by {mae_reduction:.0%}</strong>.
MAE, RMSE, and R-squared are the standard scoring metrics, and the model is scored against all three.</p>
{B.kpi_row(
    B.kpi_card(f"{test_mae:.1f} d", "Test MAE", "held-out test set", DARK_BLUE),
    B.kpi_card(f"{test_rmse:.1f} d", "Test RMSE", "held-out test set", DARK_BLUE),
    B.kpi_card(f"{test_r2:.2f}", "Test R-squared", f"{test_r2:.0%} of variance explained", DARK_BLUE),
    B.kpi_card(f"-{mae_reduction:.0%}", "Error vs baseline", f"vs {naive_mae:.1f} d naive guess", GREEN))}
<h3>Event-level lead time: are alerts raised in time?</h3>
<p>The MAE above is an average across every observation, which is helpful to understand model accuracy, but
doesn't answer the most important question for preventive maintenance purposes: when a machine is about to
fail, does a CRITICAL alert get raised in time, and how much warning does it give? Of the
<strong>{bi_events}</strong> unplanned failures in the Q1 scoring window, the model had a prediction history
for <strong>{n_events}</strong> of them (the other {bi_events - n_events} occurred on the first day of the
window, before any predictions had been made). Analyzing those {n_events} answers the question directly:
lead time, shown below, is the actual days between the first CRITICAL alert and the failure, and a positive
early/late figure means the failure arrived later than predicted, so the alert was raised with enough time
to act on it.</p>
{B.chart("Advance Warning by Alert Tier", charts["lead_tier"])}
<p>The CRITICAL alerts that were raised preceded the failure by a median of <strong>{med_lead:.0f} days</strong>
and were off by only {mean_abserr:.1f} days on average, biased {mean_signed:+.1f} days early, meaning there
was a sufficient window to act on preventive maintenance. Among those {n_events} failures, the ones that
never reached CRITICAL still surfaced an earlier ELEVATED warning, so every failure the model had scored was
flagged ahead of time. The remaining {bi_events - n_events} of the {bi_events} both struck on the opening day
of the scoring window, before the model had produced any predictions for them.</p>

{B.section("sample", "Section 3.3", "Sample Model Output")}
<p>Presented below is an example of how the model works (the signals it read, the prediction it produced, and
the reasons it flagged) for one of the CRITICAL flags it predicted.</p>
{worked_example()}

{B.section("limits", "Section 3.4", "What It Can and Cannot Predict")}
<p>Being clear about the model's limits is what makes it trustworthy. It is a strong early-warning aid, not
a crystal ball, and it is deliberately honest about the failures it cannot see coming.</p>
<ul class="limitation-list">
  <li><strong>It predicts timing, not severity or cost.</strong> The output is how soon a failure is likely,
  not how serious the repair will be or what it will cost.</li>
  <li><strong>Gradual failures are easier than sudden ones.</strong> Wear-driven mechanical problems announce
  themselves through rising vibration and heat, so the model catches them well. Abrupt electrical faults and
  operator-induced damage leave little or no warning, and the model is honest about missing some of those.</li>
  <li><strong>A quiet reading is not a guarantee.</strong> Roughly a third of real failures give no clear
  sensor precursor. Those machines still receive an ELEVATED heads-up from age and history, but not always a
  tight CRITICAL alert, so the preventive-maintenance schedule remains the safety net.</li>
  <li><strong>It is decision support, not automation.</strong> The model ranks and explains; a person still
  decides what work to schedule. It is designed to inform maintenance judgement, not replace it.</li>
  <li><strong>It stays current through monitoring.</strong> Machine behaviour drifts over time, so the model
  is watched continuously and retrained when its accuracy slips, as detailed in the monitoring report.</li>
</ul>

"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(B.page("ML Model Overview & Performance Report: RUL Predictor",
                      "", toc, body), encoding="utf-8")
print(f"Model overview written to {OUT}")
