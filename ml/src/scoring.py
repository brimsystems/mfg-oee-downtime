"""
scoring.py
Batch scoring for the Remaining Useful Life (RUL) model.

Loads the registered production model, scores each month of the forward window
(Jan-Mar 2026) independently, assigns a maintenance priority tier, and derives
the top plain-language risk drivers per observation from SHAP. Writes one file
per period for the monitoring layer, plus a one-row-per-machine fleet snapshot
that feeds the CMMS maintenance-queue deliverable.

These steps are plain functions; in production they would be wrapped as a
scheduled orchestration flow. Run:  python ml/src/scoring.py
"""
import logging
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.sklearn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, ALL_FEATURES, TARGET, ID_COL

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
REPO_ROOT   = Path(__file__).resolve().parents[2]
DB_PATH     = REPO_ROOT / "data_source" / "oee_predmaint.duckdb"
MODELS_DIR  = REPO_ROOT / "ml" / "models"
SCORING_DIR = REPO_ROOT / "ml" / "data" / "scoring"
MLRUNS_DIR  = REPO_ROOT / "ml" / "mlruns"

MLFLOW_TRACKING = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "c02_rul_predictor"
MODEL_NAME      = "rul_predictor"
PROD_ALIAS      = "production"

RUL_HORIZON  = 60
N_DRIVERS    = 3
PERIODS = [
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-31"),
]

# Maintenance priority tiers on predicted days-to-failure.
def priority_tier(days: float) -> str:
    if days <= 7:   return "CRITICAL"
    if days <= 21:  return "ELEVATED"
    if days <= 45:  return "MONITOR"
    return "OK"


# Plain-language risk drivers, keyed by model feature. Each returns a sentence
# built from the machine's own value, with no feature names exposed. Only shown
# when the feature is actually pushing the prediction toward a sooner failure.
def driver_phrase(feature: str, value) -> str | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = value
    if feature in ("vibration_anomaly", "vibration_7d_mean"):
        return "Spindle vibration trending above its normal baseline"
    if feature in ("bearing_temp_anomaly", "bearing_temp_7d_mean"):
        return "Bearing temperature running above normal"
    if feature in ("spindle_power_anomaly", "spindle_power_7d_mean"):
        return "Spindle motor power draw elevated"
    if feature in ("hydraulic_pressure_anomaly", "hydraulic_pressure_7d_mean"):
        return "Hydraulic pressure deviating from its normal range"
    if feature in ("sensor_anomaly_score", "is_sensor_anomaly"):
        return "Condition-monitoring sensors flag an anomaly"
    if feature in ("days_overdue_for_pm", "is_pm_overdue") and isinstance(v, float) and v > 0:
        return f"Preventive maintenance overdue by {int(v)} days" if feature == "days_overdue_for_pm" \
               else "Preventive maintenance interval exceeded"
    if feature in ("rolling_7d_alarm_count", "is_high_alarm_rate", "rolling_30d_alarm_count"):
        return "Alarm frequency elevated over the recent window"
    if feature in ("machine_age_years", "is_aging_machine"):
        return "Machine age and failure history indicate elevated mechanical wear risk"
    if feature == "rolling_7d_unplanned_downtime_hours" and isinstance(v, float) and v > 0:
        return f"Recent unplanned downtime elevated ({v:.1f} hrs in the past week)"
    if feature == "days_since_last_unplanned_failure":
        return "Short interval since the last unplanned failure"
    if feature == "count_late_pms_last_6m" and isinstance(v, float) and v > 0:
        return "Repeated late preventive maintenance over the past six months"
    if feature == "rolling_30d_utilization_rate":
        return "Sustained high utilization limiting maintenance windows"
    if feature in ("is_shift_b_transition", "shift"):
        return "Shift-transition downtime pattern detected on this asset"
    if feature == "last_failure_mode":
        return "Recent failure history on this asset"
    return None


# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    try:
        pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{PROD_ALIAS}")
        log.info(f"Loaded '{MODEL_NAME}@{PROD_ALIAS}' from MLflow registry")
    except Exception as e:
        log.info(f"Registry load failed ({e}); falling back to local pipeline")
        pipe = joblib.load(MODELS_DIR / "rul_pipeline.pkl")
    return pipe


def load_scoring_data(start: str, end: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        select * from mart_ml__rul_features
        where observation_date >= '{start}' and observation_date <= '{end}'
        order by observation_date
    """).df()
    con.close()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    return engineer_features(df)


def transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    prep = pipeline.named_steps["prep"]
    Xt = np.asarray(prep.transform(X), dtype="float64")
    try:
        names = [n.split("__")[-1] for n in prep.get_feature_names_out()]
        if len(names) != Xt.shape[1]:
            names = [f"f{i}" for i in range(Xt.shape[1])]
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    return pd.DataFrame(Xt, columns=names, index=X.index)


def score_period(pipeline, start: str, end: str) -> pd.DataFrame:
    df = load_scoring_data(start, end)
    X = df[ALL_FEATURES]
    pred = np.clip(pipeline.predict(X), 0, RUL_HORIZON)
    df = df.copy()
    df["predicted_days_to_failure"] = np.round(pred, 2)
    df["priority"] = [priority_tier(p) for p in pred]

    # SHAP drivers: features pushing the prediction toward a sooner failure
    # (negative contribution to predicted days) are the risk drivers.
    model = pipeline.named_steps["model"]
    Xt = transformed_frame(pipeline, X)
    try:
        if model.__class__.__name__ in ("XGBRegressor", "RandomForestRegressor"):
            sv = shap.TreeExplainer(model).shap_values(Xt)
        else:
            sv = shap.LinearExplainer(model, Xt).shap_values(Xt)
    except Exception as e:
        log.info(f"  SHAP unavailable ({e}); drivers left blank")
        sv = np.zeros((len(df), Xt.shape[1]))

    names = list(Xt.columns)
    drivers = []
    for i in range(len(df)):
        row = sv[i]
        order = np.argsort(row)            # most negative first = strongest risk
        phrases = []
        for idx in order:
            if row[idx] >= 0:
                break
            raw_val = df.iloc[i].get(names[idx], Xt.iloc[i, idx])
            ph = driver_phrase(names[idx], raw_val)
            if ph and ph not in phrases:
                phrases.append(ph)
            if len(phrases) >= N_DRIVERS:
                break
        drivers.append(phrases)
    df["risk_drivers"] = drivers
    return df


def run():
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)
    pipeline = load_model()

    keep = list(dict.fromkeys(
        [ID_COL, "machine_id", "observation_date", "shift", "machine_type"] + ALL_FEATURES + [
            "predicted_days_to_failure", "priority", "risk_drivers", TARGET, "is_censored"]))
    all_scored = []

    for start, end in PERIODS:
        label = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m")
        scored = score_period(pipeline, start, end)
        out = scored[[c for c in keep if c in scored.columns]]
        out.to_parquet(SCORING_DIR / f"predictions_{label}.parquet", index=False)

        unc = scored[~scored["is_censored"]]
        mae = float(np.abs(unc["predicted_days_to_failure"] - unc[TARGET]).mean()) if len(unc) else float("nan")
        tiers = scored["priority"].value_counts()
        log.info(f"Period {label}: {len(scored):,} scored | uncensored MAE {mae:.2f} | "
                 f"CRITICAL {tiers.get('CRITICAL',0)} ELEVATED {tiers.get('ELEVATED',0)} "
                 f"MONITOR {tiers.get('MONITOR',0)} OK {tiers.get('OK',0)}")

        with mlflow.start_run(run_name=f"scoring_{label}"):
            mlflow.set_tags({"run_type": "scoring", "period": label,
                             "scoring_start": start, "scoring_end": end})
            mlflow.log_metrics({"rows_scored": len(scored), "uncensored_mae": mae,
                                "critical_count": int(tiers.get("CRITICAL", 0)),
                                "elevated_count": int(tiers.get("ELEVATED", 0))})
        all_scored.append(scored)

    # Fleet snapshot: latest observation per machine across the window, one row
    # per machine, for the CMMS maintenance queue.
    combined = pd.concat(all_scored, ignore_index=True)
    latest = (combined.sort_values("observation_date")
              .groupby("machine_id", as_index=False).tail(1)
              .sort_values("predicted_days_to_failure").reset_index(drop=True))
    snap_cols = ["machine_id", "machine_type", "observation_date", "shift",
                 "predicted_days_to_failure", "priority", "risk_drivers",
                 "days_overdue_for_pm", "rolling_7d_alarm_count", "machine_age_years"]
    latest[snap_cols].to_parquet(SCORING_DIR / "fleet_snapshot.parquet", index=False)
    log.info(f"Fleet snapshot: {len(latest)} machines -> fleet_snapshot.parquet")
    log.info("Scoring complete.")


if __name__ == "__main__":
    run()
