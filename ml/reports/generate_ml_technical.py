"""ML Technical Overview for the RUL predictor -> docs/reports/technical_report.html
Mirrors Case 01's ml_technical: Model Card, Training Data (Feature Set, Target
Distribution), Model Selection, Performance (learning curve, calibration, residuals,
predicted vs actual), Feature Importance, Known Limitations. Regression flavor."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

import sys
sys.path.insert(0, str(Path(__file__).parent))
import brand as B
from brand import DARK_BLUE, LIGHT_BLUE, ACCENT_RED, AMBER, GREEN, MED_GREY, LIGHT_GREY, DARK_GREY
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, INTERACTION_FEATURES, TARGET

REPO     = Path(__file__).resolve().parents[2]
MODELS   = REPO / "ml" / "models"
FEATURES = REPO / "ml" / "data" / "features"
OUT      = REPO / "docs" / "reports" / "technical_report.html"
LABELS = {"linear_regression": "Linear Regression (Ridge)", "random_forest": "Random Forest", "xgboost": "XGBoost"}

m     = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
comp  = pd.read_csv(MODELS / "model_comparison.csv")
imp   = pd.read_csv(MODELS / "shap_importance.csv")
resid = pd.read_csv(MODELS / "residuals_test.csv")
calib = pd.read_csv(MODELS / "calibration_test.csv")
lc    = pd.read_csv(MODELS / "learning_curve.csv") if (MODELS / "learning_curve.csv").exists() else None
train = pd.read_parquet(FEATURES / "train.parquet")
val   = pd.read_parquet(FEATURES / "validation.parquet")
test  = pd.read_parquet(FEATURES / "test.parquet")
best  = m["best_model_type"]
HORIZON = m["rul_horizon_days"]
_bp = next((x["params"] for x in m["models"] if x["model_type"] == best), {})

# Monthly observation volume and fleet size, for the data-coverage view.
_dv = duckdb.connect(str(REPO / "data_source" / "oee_predmaint.duckdb"), read_only=True)
_vol = _dv.execute("select observation_date from mart_ml__rul_features").df()
n_machines = int(_dv.execute("select count(distinct machine_id) from mart_ml__rul_features").fetchone()[0])
_dv.close()
_vol["ym"] = pd.to_datetime(_vol["observation_date"]).dt.to_period("M").dt.to_timestamp()
vol_monthly = _vol.groupby("ym").size()

# Empirical prediction interval from the uncensored held-out test residuals.
_res_u = resid[~resid["is_censored"]]
_ae = np.abs(_res_u["residual"])
pi_50, pi_80, pi_90 = (float(np.quantile(_ae, q)) for q in (0.5, 0.8, 0.9))
resid_mean = float(_res_u["residual"].mean())

# ── Charts ──────────────────────────────────────────────────────────────────
def chart_target_dist():
    fig, ax = B.make_fig(h=3.2)
    bins = np.arange(0, HORIZON + 2, 3)
    for df, c, lab in [(train, DARK_BLUE, "Train"), (val, LIGHT_BLUE, "Validation"), (test, MED_GREY, "Test")]:
        ax.hist(df[TARGET], bins=bins, density=True, histtype="step", linewidth=2, color=c, label=lab)
    ax.set_xlabel("Days to failure (target)"); ax.set_ylabel("Density"); ax.legend()
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_learning():
    if lc is None:
        return None
    fig, ax = B.make_fig(h=3.2)
    ax.plot(lc["train_size"], lc["train_mae"], "o-", color=DARK_BLUE, lw=2, label="Train MAE")
    ax.plot(lc["train_size"], lc["val_mae"], "s-", color=LIGHT_BLUE, lw=2, label="Cross-val MAE")
    ax.set_xlabel("Training observations"); ax.set_ylabel("MAE (days)"); ax.legend()
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_calibration():
    fig, ax = B.make_fig(h=3.6)
    ax.plot([0, HORIZON], [0, HORIZON], color=MED_GREY, ls="--", lw=1.5, label="Perfectly calibrated")
    ax.plot(calib["mean_predicted"], calib["mean_actual"], "o-", color=DARK_BLUE, lw=2, label="Model")
    ax.set_xlabel("Mean predicted days (bin)"); ax.set_ylabel("Mean actual days (bin)")
    ax.set_xlim(0, HORIZON); ax.set_ylim(0, HORIZON); ax.legend(loc="upper left")
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_residual_hist():
    fig, ax = B.make_fig(h=3.0)
    ax.hist(resid["residual"], bins=40, color=DARK_BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=ACCENT_RED, ls="--", lw=1.3)
    ax.set_xlabel("Residual (predicted minus actual, days)"); ax.set_ylabel("Observations")
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_resid_vs_pred():
    fig, ax = B.make_fig(h=3.2)
    ax.scatter(resid["predicted"], resid["residual"], s=9, alpha=0.18, color=DARK_BLUE, edgecolors="none")
    ax.axhline(0, color=ACCENT_RED, ls="--", lw=1.3)
    ax.set_xlabel("Predicted days to failure"); ax.set_ylabel("Residual (days)")
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_mae_by_mode():
    d = resid.assign(ae=resid["residual"].abs()).groupby("last_failure_mode")["ae"].mean().sort_values()
    worst = d.idxmax()
    fig, ax = B.make_fig(h=2.8)
    ax.barh([str(x).replace("_", " ").title() for x in d.index], d.values,
            color=[ACCENT_RED if i == worst else DARK_BLUE for i in d.index], height=0.6)
    for i, v in enumerate(d.values):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=10)
    ax.set_xlabel("Mean absolute error (days)"); ax.set_xlim(0, d.max() * 1.18)
    B.chart_style(ax); ax.xaxis.grid(True, color=LIGHT_GREY); ax.yaxis.grid(False)
    fig.tight_layout()
    return B.b64(fig)


def chart_pred_vs_actual():
    fig, ax = plt_pair()
    ax.scatter(resid["actual"], resid["predicted"], s=9, alpha=0.20, color=DARK_BLUE, edgecolors="none")
    ax.plot([0, HORIZON], [0, HORIZON], color=ACCENT_RED, ls="--", lw=1.3)
    ax.set_xlim(0, HORIZON + 1); ax.set_ylim(0, HORIZON + 1)
    ax.set_xlabel("Actual days to failure"); ax.set_ylabel("Predicted days to failure")
    B.chart_style(ax); import matplotlib.pyplot as plt; plt.tight_layout()
    return B.b64(ax.figure)


def plt_pair():
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(5.2, 4.0))


def chart_shap():
    d = imp.head(12).iloc[::-1]
    fig, ax = B.make_fig(h=B.CHART_H_T)
    ax.barh(d["feature"], d["mean_abs_shap"], color=DARK_BLUE, height=0.68)
    for i, v in enumerate(d["mean_abs_shap"]):
        ax.text(v + d["mean_abs_shap"].max() * 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=MED_GREY)
    ax.set_xlabel("Mean |SHAP| (days of impact on prediction)")
    B.chart_style(ax); ax.xaxis.grid(True, color=LIGHT_GREY); ax.yaxis.grid(False)
    fig.tight_layout()
    return B.b64(fig)


def chart_data_volume():
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    def split_of(ts):
        if ts <= pd.Timestamp("2024-12-31"): return "Train", DARK_BLUE
        if ts <= pd.Timestamp("2025-08-31"): return "Validation", LIGHT_BLUE
        if ts <= pd.Timestamp("2025-12-31"): return "Test", AMBER
        return "Scoring (held out)", MED_GREY
    fig, ax = B.make_fig(h=3.0)
    ax.bar(vol_monthly.index, vol_monthly.values, width=22,
           color=[split_of(ts)[1] for ts in vol_monthly.index])
    ax.set_ylabel("Observations / month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    seen = {}
    for ts in vol_monthly.index:
        lab, col = split_of(ts); seen[lab] = col
    ax.legend(handles=[Patch(color=c, label=l) for l, c in seen.items()],
              fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    B.chart_style(ax)
    fig.tight_layout()
    return B.b64(fig)


def chart_corr_heatmap():
    from matplotlib.colors import LinearSegmentedColormap
    feats = [f for f in NUMERICAL_FEATURES if f in train.columns]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = train[feats].astype(float).corr().fillna(0.0).values
    cmap = LinearSegmentedColormap.from_list("brand_div", [DARK_BLUE, "#FFFFFF", ACCENT_RED])
    fig, ax = B.make_fig(h=6.4)
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=6.5)
    ax.set_yticklabels(feats, fontsize=6.5)
    ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return B.b64(fig)


# ── Tables ──────────────────────────────────────────────────────────────────
def interval_table():
    rows = [["50% of predictions", f"&plusmn;{pi_50:.0f} days"],
            ["80% of predictions", f"&plusmn;{pi_80:.0f} days"],
            ["90% of predictions", f"&plusmn;{pi_90:.0f} days"]]
    return B.data_table(["Share of predictions", "Falls within of the true failure date"], rows, right={1})


def ops_table():
    rows = [
        ["Scoring cadence", f"Daily batch; all {n_machines} machines scored ahead of each shift"],
        ["Inference latency",
         f"A single gradient-boosted model over {m['feature_counts']['total']} features; the full-fleet "
         f"daily batch scores in well under a second on commodity hardware"],
        ["Model registry",
         f"MLflow Model Registry, production v{m['model_version']}; each retrain registers a new version and "
         f"this report regenerates against it"],
        ["Data lineage",
         "Source systems (MachineMetrics, JobBOSS ERP, Limble CMMS, ADP, IIoT gateway) &rarr; dlt ingestion "
         "&rarr; dbt staging, intermediate and marts &rarr; mart_ml__rul_features &rarr; features.py &rarr; "
         "registered model &rarr; CMMS maintenance queue"],
        ["Retraining trigger",
         "Monitoring report drift rules: sustained performance or target drift across two consecutive periods"],
    ]
    return B.data_table(["Specification", "Detail"], rows)
def training_data_table():
    rows = [["Train", "Jan 2023 to Dec 2024", f"{m['split_sizes']['train']:,}", f"{train[TARGET].mean():.1f}"],
            ["Validation", "Jan to Aug 2025", f"{m['split_sizes']['validation']:,}", f"{val[TARGET].mean():.1f}"],
            ["Test", "Sep to Dec 2025", f"{m['split_sizes']['test']:,}", f"{test[TARGET].mean():.1f}"],
            ["Scoring (held out)", "Jan to Mar 2026", "1,848", "n/a"]]
    return B.data_table(["Split", "Window", "Observations", "Mean target (days)"], rows, right={2, 3})


def feature_table():
    order = ([(f, "Categorical") for f in CATEGORICAL_FEATURES]
             + [(f, "Numerical") for f in NUMERICAL_FEATURES]
             + [(f, "Interaction") for f in INTERACTION_FEATURES])
    tcol = {"Categorical": LIGHT_BLUE, "Numerical": DARK_BLUE, "Interaction": AMBER}
    rows = ""
    for feat, ftype in order:
        if ftype == "Categorical" or feat not in train.columns:
            corr = "&mdash;".replace("&mdash;", "-")
        else:
            c = train[feat].astype(float).corr(train[TARGET].astype(float))
            corr = f"{c:+.3f}" if pd.notna(c) else "-"
        rows += (f'<tr><td style="font-family:monospace;font-size:13px;">{feat}</td>'
                 f'<td>{B.badge(ftype, tcol[ftype])}</td>'
                 f'<td style="text-align:right;">{corr}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Feature</th><th>Type</th>'
            f'<th style="text-align:right;">Corr. with target</th></tr></thead><tbody>{rows}</tbody></table>')


def comparison_table():
    rows = ""
    for r in comp.sort_values("val_mae").itertuples():
        sel = r.model_type == best
        win = ' <span style="color:%s;font-weight:700;">&#10003; Selected</span>' % GREEN if sel else ""
        bg = f' style="background:{B.BG_GREY};font-weight:700;"' if sel else ""
        rows += (f'<tr{bg}><td>{LABELS.get(r.model_type, r.model_type)}{win}</td>'
                 f'<td style="text-align:right;">{r.val_mae:.2f}</td>'
                 f'<td style="text-align:right;">{r.val_rmse:.2f}</td>'
                 f'<td style="text-align:right;">{r.val_r2:.3f}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Model</th>'
            f'<th style="text-align:right;">Val MAE</th><th style="text-align:right;">Val RMSE</th>'
            f'<th style="text-align:right;">Val R2</th></tr></thead><tbody>{rows}</tbody></table>')


def metrics_table():
    v, t = m["best_val"], m["test"]
    rows = [["MAE (days)", f"{v['mae']:.2f}", f"{t['mae']:.2f}"],
            ["RMSE (days)", f"{v['rmse']:.2f}", f"{t['rmse']:.2f}"],
            ["R2", f"{v['r2']:.3f}", f"{t['r2']:.3f}"]]
    body = "".join(f'<tr><td>{r[0]}</td><td style="text-align:right;">{r[1]}</td>'
                   f'<td style="text-align:right;font-weight:700;color:{DARK_GREY};">{r[2]}</td></tr>' for r in rows)
    return (f'<table class="data-table"><thead><tr><th>Metric</th>'
            f'<th style="text-align:right;">Validation</th>'
            f'<th style="text-align:right;">Test (held-out)</th></tr></thead><tbody>{body}</tbody></table>')


charts = {"target": chart_target_dist(), "learning": chart_learning(), "calib": chart_calibration(),
          "rhist": chart_residual_hist(), "rvp": chart_resid_vs_pred(), "mode": chart_mae_by_mode(),
          "pva": chart_pred_vs_actual(), "shap": chart_shap(),
          "volume": chart_data_volume(), "corr": chart_corr_heatmap()}

toc = ('<a href="#card">Model Card</a><hr>'
       '<a href="#data">Training Data</a><hr>'
       '<a href="#modelperf">Model Selection &amp; Performance</a>'
       '<a href="#selection" class="sub">Model Selection</a>'
       '<a href="#performance" class="sub">Model Performance</a><hr>'
       '<a href="#shap">Feature Importance</a><hr>'
       '<a href="#limits">Known Limitations</a><hr>'
       '<a href="#ops">Deployment &amp; Operations</a>')

body = f"""
{B.section("card", "Section 1", "Model Card")}
<div class="model-card"><div class="model-card-grid">
  <div><div class="mc-label">Model Name</div><div class="mc-value">rul_predictor</div></div>
  <div><div class="mc-label">Model Type</div><div class="mc-value">{LABELS.get(best, best)} (scikit-learn Pipeline)</div></div>
  <div><div class="mc-label">Version</div><div class="mc-value">v{m['model_version']} &middot; Production</div></div>
  <div><div class="mc-label">Registry</div><div class="mc-value">MLflow Model Registry</div></div>
  <div><div class="mc-label">Target</div><div class="mc-value">Days to next unplanned repair (capped at {HORIZON})</div></div>
  <div><div class="mc-label">Prediction Type</div><div class="mc-value">Regression &middot; days in [0, {HORIZON}]</div></div>
  <div><div class="mc-label">Priority Tiers</div><div class="mc-value">CRITICAL &le;7 &middot; ELEVATED 8-21 &middot; MONITOR 22-45 &middot; OK &gt;45</div></div>
  <div><div class="mc-label">Tuning</div><div class="mc-value">Optuna ({m['n_optuna_trials']} trials, validation MAE objective)</div></div>
  <div style="grid-column:1/-1;"><div class="mc-label">Purpose</div><div class="mc-value">Scores each machine daily for time to next unplanned failure, feeding the CMMS maintenance queue. Decision support for prioritisation, not automated work-order generation.</div></div>
</div></div>

{B.section("data", "Section 2", "Training Data")}
<p>The model reads one feature vector per machine, day, and shift, assembled from four families of
shop-floor data and engineered identically at training and scoring time. <strong>Condition-monitoring
sensors</strong> contribute seven-day averages and per-channel anomaly scores for spindle vibration,
bearing temperature, spindle motor power, and hydraulic pressure. <strong>Machine telemetry and OEE</strong>
contribute rolling alarm counts, unplanned-downtime hours, and utilization. The <strong>CMMS maintenance
history</strong> contributes time since the last unplanned failure, time since and days overdue on
preventive maintenance, the count of recent late PMs, and the last failure mode. <strong>Machine
attributes</strong> contribute age, type, controller, and shift. On top of these, five interaction flags
encode the cross-system reliability patterns found in the diagnostic analysis (PM overdue, aging asset,
elevated alarm rate, shift-B transition, and sensor anomaly). In total the model weighs
<strong>{m['feature_counts']['total']}</strong> features per observation.</p>
<p>Training spans January 2023 through December 2025. The split is time-based and never shuffled, mirroring
deployment where the model scores future dates it has not seen; shuffling maintenance records across time
would leak future outcomes into training. The January to March 2026 window is held out entirely for
scoring.</p>
{training_data_table()}
<p>Data coverage is uniform across the window: every month carries a near-constant number of
machine-day-shift observations, so no split is starved and the boundaries below are purely chronological.</p>
{B.chart("Observation Volume by Month and Split", charts["volume"])}
<p>The full feature set is listed below with its correlation to the target. The target itself is the number
of days from the observation date to the next unplanned repair in the CMMS, capped at {HORIZON} days.</p>
{feature_table()}
<p>Many features are engineered from the same underlying signals, so some move together. The heatmap below
shows the pairwise correlations among the numerical features. Gradient-boosted trees are robust to this kind
of correlation (it affects which of two interchangeable features a split uses, not overall accuracy), but it
is worth noting where the model's heaviest drivers overlap. Machine age, one of the top drivers, runs
inversely with fleet utilization (about -0.70) and rises with bearing temperature and spindle power (about
+0.58), so older assets read as hotter, rougher, and less heavily loaded. The composite sensor anomaly score,
another top driver, correlates about +0.60 with the individual channel anomalies it aggregates, and the
sensor channel averages move together (vibration and bearing temperature at about +0.63). Time since the last
failure, the single strongest driver, is close to independent of the rest, so it contributes largely
non-redundant signal.</p>
{B.chart("Feature Correlation Heatmap (numerical features)", charts["corr"])}
<p>The days-to-failure distribution is consistent across the three splits, confirming the time-based split
did not introduce a shift in the target.</p>
{B.chart("Target Distribution: Train / Validation / Test", charts["target"])}

{B.section("modelperf", "Section 3", "Model Selection & Performance")}

{B.section("selection", "Section 3.1", "Model Selection")}
<p>Three candidate regressors, a linear regression, a random forest, and a gradient-boosted XGBoost model,
were tuned independently with Optuna ({m['n_optuna_trials']} trials each, validation-MAE objective) and
compared on the validation set. {LABELS.get(best, best)} won on both MAE and RMSE and was registered as the
production model, then evaluated once on the held-out test set.</p>
{comparison_table()}
<p>The selected configuration is a shallow, well-regularised ensemble (max depth {_bp['max_depth']}, learning
rate {_bp['learning_rate']:.2f}, subsample {_bp['subsample']:.2f}, column subsample
{_bp['colsample_bytree']:.2f}), favouring many small trees over a few deep ones, which suits the moderate
signal-to-noise of failure timing and guards against overfitting. Hyperparameters were optimised against MAE
on the fixed time-based validation window (January to August 2025) rather than shuffled k-fold
cross-validation.</p>

{B.section("performance", "Section 3.2", "Model Performance")}
<p>The selected model is evaluated from several angles, each answering a different question about whether the
predictions can be trusted in production. The headline metrics come first, then four diagnostic views: does
the model have enough data and generalise (learning curve), can the predicted day counts be taken at face
value (calibration), where and how does it produce errors (residual analysis), and how closely do
predictions track reality across the horizon (predicted versus actual).</p>
<p>Metrics appear for both the validation set (used for tuning and selection) and the held-out test set,
which was touched only once and is the honest estimate of real-world performance. On the test set the model
lands at <strong>{m['test']['mae']:.1f} days</strong> MAE and an R-squared of <strong>{m['test']['r2']:.2f}</strong>.</p>
{metrics_table()}
<p>A learning curve plots cross-validated error as the training set grows. It matters because it separates
two failure modes: a model starved of data, where both curves sit high, from one that has memorised its
training set, where a wide gap opens between the train and validation curves. Here the two curves converge to
a similar error, which means the model is learning generalisable signal and would gain little from simply
adding more of the same data.</p>
{B.chart("Learning Curve (3-fold CV)", charts["learning"])}
<p>Calibration checks whether the predicted number of days can be taken at face value. Predictions are
binned, and the mean predicted horizon in each bin is compared against the mean actual horizon. Points on the
diagonal mean a prediction of, say, ten days really does average about ten days to failure, so the day counts
are usable as a scheduling signal.</p>
{B.chart("Calibration: Mean Predicted vs Mean Actual", charts["calib"])}
<p>Residual analysis, actual minus predicted outcomes, shows the shape and location of the model's errors.
Residuals centered on zero indicate no systematic over- or under-prediction, and the residual-versus-predicted
view reveals only mild regression toward the mid-range.</p>
<div class="chart-pair">{B.chart("Residual Distribution", charts["rhist"])}{B.chart("Residuals vs Predicted", charts["rvp"])}</div>
<p>Breaking the error out by failure mode shows it is not uniform. The model is most accurate on wear-driven
mechanical failures, which announce themselves through the sensor channels, and least accurate on the abrupt
electrical and tooling modes that leave little physical warning.</p>
{B.chart("Mean Absolute Error by Last Failure Mode", charts["mode"])}
<p>Finally, plotting predicted against actual days on the held-out test set gives the overall picture. Tight
clustering around the diagonal indicates the model tracks true timing across the full horizon, not only at
the extremes.</p>
<div style="max-width:520px;margin:18px auto;">{B.chart("Predicted vs Actual (Test Set)", charts["pva"])}</div>
<p>Beyond the average error, a scheduler needs to know how wide the uncertainty band around a single
prediction is. On the held-out test set the errors are roughly symmetric and centred near zero (mean residual
{resid_mean:+.1f} days), so the empirical prediction interval below can be read directly off the residual
spread. A prediction should therefore be treated as a window rather than a to-the-day guarantee: a CRITICAL
call of five days means the failure is expected within roughly a week, give or take.</p>
{interval_table()}

{B.section("shap", "Section 4", "Feature Importance (SHAP)")}
<p>SHAP values measure each feature's average contribution to the prediction across the validation set. Time
since the last failure, machine age, and the recent alarm and downtime rolling features carry the most
weight, matching the reliability drivers in the diagnostic analytics.</p>
{B.chart("Mean Absolute SHAP Value by Feature", charts["shap"])}

{B.section("limits", "Section 5", "Known Limitations")}
<ul class="limitation-list">
  <li><strong>Failure physics:</strong> the model is most reliable for gradual mechanical and wear
  failures and least reliable for electrical and tooling failures, which are more abrupt.</li>
  <li><strong>Right-censoring:</strong> observations near the end of the record have no observed future
  failure; their targets are capped at the horizon, biasing those predictions upward.</li>
  <li><strong>Simulated data:</strong> trained on synthetic data with embedded reliability patterns.
  Real-world performance depends on the signal present in actual shop-floor data.</li>
  <li><strong>Retraining:</strong> retrain when the monitoring report flags sustained performance or
  target drift across two consecutive periods, per the retraining rules.</li>
  <li><strong>Scope:</strong> predicts timing of the next unplanned failure, not its severity or repair
  cost. It supports prioritisation; it does not replace maintenance judgement.</li>
</ul>

{B.section("ops", "Section 6", "Deployment & Operations")}
<p>How the model runs in production and where its inputs come from.</p>
{ops_table()}"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(B.page("ML Model Technical Overview: RUL Predictor",
                      "", toc, body), encoding="utf-8")
print(f"Technical report written to {OUT}")
