"""
features.py
Shared feature engineering for the Remaining Useful Life (RUL) model.
Manually maintained: update here when feature definitions change.
Imported by src/training.py, src/scoring.py, and src/monitoring.py only.

The backward-looking rolling features and the target are built upstream in
mart_ml__rul_features. This module adds the domain interaction features,
imputes the small number of early-window nulls with fixed constants (so the
transform is identical on train, validation, test, and scoring data), and
declares the feature contract the pipeline trains on.
"""

import pandas as pd

CATEGORICAL_FEATURES = [
    "machine_type",
    "controller_type",
    "last_failure_mode",
    "shift",
]

NUMERICAL_FEATURES = [
    "machine_age_years",
    "rolling_7d_unplanned_downtime_hours",
    "rolling_7d_alarm_count",
    "rolling_30d_alarm_count",
    "rolling_30d_utilization_rate",
    "days_since_last_unplanned_failure",
    "days_since_last_pm",
    "days_overdue_for_pm",
    "count_late_pms_last_6m",
    # Condition-monitoring sensor layer (rolling means + anomaly z-scores).
    "vibration_7d_mean",
    "bearing_temp_7d_mean",
    "spindle_power_7d_mean",
    "hydraulic_pressure_7d_mean",
    "vibration_anomaly",
    "bearing_temp_anomaly",
    "spindle_power_anomaly",
    "hydraulic_pressure_anomaly",
    "sensor_anomaly_score",
]

INTERACTION_FEATURES = [
    "is_pm_overdue",
    "is_aging_machine",
    "is_high_alarm_rate",
    "is_shift_b_transition",
    "is_sensor_anomaly",
]

# Sensor anomaly z-score above which a reading is treated as anomalous.
SENSOR_ANOMALY_THRESHOLD = 2.0

SENSOR_FEATURES = [
    "vibration_7d_mean", "bearing_temp_7d_mean", "spindle_power_7d_mean",
    "hydraulic_pressure_7d_mean", "vibration_anomaly", "bearing_temp_anomaly",
    "spindle_power_anomaly", "hydraulic_pressure_anomaly", "sensor_anomaly_score",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + INTERACTION_FEATURES

TARGET  = "target_days_to_failure"
ID_COL  = "rul_key"

# ── Domain constants ─────────────────────────────────────────────────────────
# Machines older than this are treated as aged assets (matches the reliability
# framing used across the analytics deliverables).
AGING_AGE_THRESHOLD = 9

# Preventive-maintenance cadence; a machine is "overdue" once it exceeds this.
PM_INTERVAL_DAYS = 42

# 75th percentile of the 7-day alarm count on the training window, used as the
# fixed high-alarm threshold. Computed once from training data (leakage-free);
# revisit if the fleet or window changes.
ALARM_RATE_P75 = 15

# Fixed imputation values for the early-window nulls (observations before the
# first PM or failure was recorded). Fixed rather than data-derived so the
# transform is identical across every split.
FILL_DAYS_SINCE_FAILURE = 365.0   # no failure on record: encoded as long-healthy
FILL_DAYS_SINCE_PM      = float(PM_INTERVAL_DAYS)  # neutral: about one interval
FILL_UTILIZATION        = 0.80    # neutral utilisation when no 30-day history yet


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute early-window nulls with fixed constants and derive the domain
    interaction features. Called identically on train, validation, test, and
    scoring data. All features are backward-looking as of the observation date.
    """
    df = df.copy()

    # ── Impute early-window nulls (fixed, leakage-free) ────────────────────
    df["last_failure_mode"] = df["last_failure_mode"].fillna("NONE")
    df["days_since_last_unplanned_failure"] = (
        df["days_since_last_unplanned_failure"].fillna(FILL_DAYS_SINCE_FAILURE))
    df["days_since_last_pm"]  = df["days_since_last_pm"].fillna(FILL_DAYS_SINCE_PM)
    df["days_overdue_for_pm"] = df["days_overdue_for_pm"].fillna(0.0)
    df["rolling_30d_utilization_rate"] = (
        df["rolling_30d_utilization_rate"].fillna(FILL_UTILIZATION))

    # Sensor features: readings exist from day one, so nulls occur only in the
    # first couple of days (no baseline yet). Fill with 0 (no deviation).
    for c in SENSOR_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ── Interaction features ───────────────────────────────────────────────
    # PM overdue: past the scheduled interval since the last completed PM.
    df["is_pm_overdue"] = (df["days_overdue_for_pm"] > 0).astype(int)

    # Aging asset: older machines carry more mechanical wear risk.
    df["is_aging_machine"] = (df["machine_age_years"] > AGING_AGE_THRESHOLD).astype(int)

    # Elevated recent alarm activity relative to the fleet.
    df["is_high_alarm_rate"] = (df["rolling_7d_alarm_count"] > ALARM_RATE_P75).astype(int)

    # Shift B carries the crew-handover startup risk documented in the analytics.
    df["is_shift_b_transition"] = (df["shift"] == "B").astype(int)

    # Condition-monitoring sensors flagging an anomaly vs the machine's baseline.
    df["is_sensor_anomaly"] = (df["sensor_anomaly_score"] > SENSOR_ANOMALY_THRESHOLD).astype(int)

    return df
