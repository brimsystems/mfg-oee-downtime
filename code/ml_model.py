"""Remaining-useful-life model: feature engineering, model training and selection, batch scoring, and monitoring. Reads the modeled marts and writes the registered model and scored outputs."""

import duckdb
import joblib
import json
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import shap
import sys
import warnings
from datetime import datetime
from datetime import datetime, timezone
from evidently import Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift
from mlflow import MlflowClient
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import learning_curve as sk_learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


# ==========================================================================
# Feature engineering
# ==========================================================================

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


# ==========================================================================
# Model training and selection
# ==========================================================================

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
REPO_ROOT     = Path(__file__).resolve().parents[2]
DB_PATH       = REPO_ROOT / "data_source" / "oee_predmaint.duckdb"
FEATURES_DIR  = REPO_ROOT / "ml" / "data" / "features"
MODELS_DIR    = REPO_ROOT / "ml" / "models"
MLRUNS_DIR    = REPO_ROOT / "ml" / "mlruns"
# Time-based split (never shuffled). Train Jan 2023-Dec 2024, Validate
# Jan-Aug 2025, Test Sep-Dec 2025. The Jan-Mar 2026 scoring window is held out
# entirely (handled by scoring.py) and must not leak into any split here.
TRAIN_END = "2024-12-31"
VAL_END   = "2025-08-31"
TEST_END  = "2025-12-31"
MLFLOW_TRACKING = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "rul_predictor"
MODEL_NAME      = "rul_predictor"
PROD_ALIAS      = "production"
N_TRIALS        = 100
RANDOM_SEED     = 42
RUL_HORIZON     = 60   # target cap in days
# DATA
def prepare_features() -> None:
    log.info("Preparing feature splits from mart...")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("select * from mart_ml__rul_features").df()
    con.close()

    df["observation_date"] = pd.to_datetime(df["observation_date"])
    log.info(f"  Loaded {len(df):,} observations from mart")

    df = engineer_features(df)

    train_mask = df["observation_date"] <= TRAIN_END
    val_mask   = (df["observation_date"] > TRAIN_END) & (df["observation_date"] <= VAL_END)
    test_mask  = (df["observation_date"] > VAL_END) & (df["observation_date"] <= TEST_END)

    export_cols = [ID_COL] + ALL_FEATURES + [TARGET, "is_censored"]
    export_cols = [c for c in export_cols if c in df.columns]

    for label, mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
        split = df[mask][export_cols]
        path  = FEATURES_DIR / f"{label}.parquet"
        split.to_parquet(path, index=False)
        log.info(f"  {label:<12} {len(split):>7,} rows  mean target: "
                 f"{split[TARGET].mean():.1f} days  -> {path.name}")
def load_splits() -> tuple:
    for split in ["train", "validation", "test"]:
        if not (FEATURES_DIR / f"{split}.parquet").exists():
            prepare_features()
            break

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val   = pd.read_parquet(FEATURES_DIR / "validation.parquet")
    test  = pd.read_parquet(FEATURES_DIR / "test.parquet")

    X_train, y_train = train[ALL_FEATURES], train[TARGET].astype(float)
    X_val,   y_val   = val[ALL_FEATURES],   val[TARGET].astype(float)
    X_test,  y_test  = test[ALL_FEATURES],  test[TARGET].astype(float)
    return X_train, y_train, X_val, y_val, X_test, y_test, test
def build_preprocessors(X: pd.DataFrame) -> tuple:
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    tree_prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ], remainder="drop")

    linear_prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ], remainder="drop")

    return tree_prep, linear_prep
def reg_metrics(y_true, y_pred, prefix="") -> dict:
    return {
        f"{prefix}mae":  float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}r2":   float(r2_score(y_true, y_pred)),
    }
# HYPERPARAMETER SEARCH (objective = validation MAE, minimized)
def _study():
    return optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
def tune_linear(X_train, y_train, X_val, y_val, linear_prep):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-2, 100.0, log=True)
        pipe = Pipeline([("prep", clone(linear_prep)),
                         ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED))])
        pipe.fit(X_train, y_train)
        return mean_absolute_error(y_val, pipe.predict(X_val))

    study = _study(); study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    pipe = Pipeline([("prep", clone(linear_prep)),
                     ("model", Ridge(**best, random_state=RANDOM_SEED))]).fit(X_train, y_train)
    log.info(f"  linear_regression best {best}  val_mae={study.best_value:.3f}")
    return pipe, best
def tune_random_forest(X_train, y_train, X_val, y_val, tree_prep):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "max_depth":        trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
        pipe = Pipeline([("prep", clone(tree_prep)),
                         ("model", RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1))])
        pipe.fit(X_train, y_train)
        return mean_absolute_error(y_val, pipe.predict(X_val))

    study = _study(); study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    pipe = Pipeline([("prep", clone(tree_prep)),
                     ("model", RandomForestRegressor(**best, random_state=RANDOM_SEED, n_jobs=-1))]).fit(X_train, y_train)
    log.info(f"  random_forest best {best}  val_mae={study.best_value:.3f}")
    return pipe, best
def tune_xgboost(X_train, y_train, X_val, y_val, tree_prep):
    prep = clone(tree_prep).fit(X_train)
    X_tr, X_va = prep.transform(X_train), prep.transform(X_val)

    def objective(trial):
        params = {
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        }
        model = XGBRegressor(**params, n_estimators=1000, objective="reg:squarederror",
                             eval_metric="mae", early_stopping_rounds=30,
                             random_state=RANDOM_SEED, verbosity=0)
        model.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
        return mean_absolute_error(y_val, model.predict(X_va))

    study = _study(); study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    log.info(f"  xgboost best {best}  val_mae={study.best_value:.3f}")

    model = XGBRegressor(**best, n_estimators=1000, objective="reg:squarederror",
                         eval_metric="mae", early_stopping_rounds=30,
                         random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
    pipe = Pipeline([("prep", prep), ("model", model)])
    return pipe, best
# ARTIFACTS (data only; the report generators draw the brand-styled charts)
def shap_importance(pipeline, X_sample: pd.DataFrame) -> pd.DataFrame:
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = np.asarray(prep.transform(X_sample), dtype="float64")
    try:
        names = [n.split("__")[-1] for n in prep.get_feature_names_out()]
        if len(names) != Xt.shape[1]:
            names = [f"f{i}" for i in range(Xt.shape[1])]
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    Xt_df = pd.DataFrame(Xt, columns=names)

    if isinstance(model, (XGBRegressor, RandomForestRegressor)):
        vals = shap.TreeExplainer(model).shap_values(Xt_df)
    else:
        vals = shap.LinearExplainer(model, Xt_df).shap_values(Xt_df)

    return (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(vals).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
def learning_curve_data(pipeline, X_train, y_train) -> pd.DataFrame:
    lc_pipe = clone(pipeline)
    step = lc_pipe.named_steps["model"]
    if isinstance(step, XGBRegressor):
        step.set_params(early_stopping_rounds=None, n_estimators=300)
    sizes, train_sc, val_sc = sk_learning_curve(
        lc_pipe, X_train, y_train, cv=3, scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.15, 1.0, 6), n_jobs=-1, random_state=RANDOM_SEED)
    return pd.DataFrame({
        "train_size": sizes,
        "train_mae":  -train_sc.mean(axis=1),
        "val_mae":    -val_sc.mean(axis=1),
    })
# MAIN
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    prepare_features()
    X_train, y_train, X_val, y_val, X_test, y_test, test_df = load_splits()
    log.info(f"Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,}")

    tree_prep, linear_prep = build_preprocessors(X_train)
    run_name = f"training_{datetime.now():%Y%m%d_%H%M}"

    tuners = [
        ("linear_regression", lambda: tune_linear(X_train, y_train, X_val, y_val, linear_prep)),
        ("random_forest",     lambda: tune_random_forest(X_train, y_train, X_val, y_val, tree_prep)),
        ("xgboost",           lambda: tune_xgboost(X_train, y_train, X_val, y_val, tree_prep)),
    ]

    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags({"target": TARGET, "train_rows": len(X_train), "val_rows": len(X_val),
                         "test_rows": len(X_test), "n_optuna_trials": N_TRIALS, "random_seed": RANDOM_SEED})
        results = []
        for name, tuner in tuners:
            log.info(f"Training {name}...")
            pipe, params = tuner()
            val_m = reg_metrics(y_val, pipe.predict(X_val), prefix="val_")
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.log_params({f"{name}__{k}": v for k, v in params.items()})
                mlflow.log_metrics(val_m)
                mlflow.set_tag("model_type", name)
                mlflow.sklearn.log_model(pipe, name=f"{name}_pipeline",
                                         serialization_format="cloudpickle")
                results.append({"model_type": name, "pipeline": pipe, "params": params,
                                "child_run_id": child.info.run_id, **val_m})
            log.info(f"  {name}: val_mae={val_m['val_mae']:.2f}  val_rmse={val_m['val_rmse']:.2f}  val_r2={val_m['val_r2']:.3f}")

        comparison = pd.DataFrame([{
            "model_type": r["model_type"], "val_mae": round(r["val_mae"], 3),
            "val_rmse": round(r["val_rmse"], 3), "val_r2": round(r["val_r2"], 3),
        } for r in results]).sort_values("val_mae").reset_index(drop=True)
        log.info("\nModel comparison (validation, lower MAE is better):\n" + comparison.to_string(index=False))

        best = min(results, key=lambda r: r["val_mae"])
        best_pipe, best_type = best["pipeline"], best["model_type"]
        mlflow.set_tag("best_model_type", best_type)
        log.info(f"\nBest model: {best_type} (val_mae={best['val_mae']:.2f})")

        # Validation predictions: prediction-drift reference for monitoring.
        val_pred = best_pipe.predict(X_val)
        pd.DataFrame({"predicted_days_to_failure": np.round(val_pred, 3),
                      TARGET: y_val.values}).to_parquet(
            FEATURES_DIR / "validation_predictions.parquet", index=False)

        # Held-out test evaluation (touched once).
        test_pred = best_pipe.predict(X_test)
        test_m = reg_metrics(y_test, test_pred, prefix="test_")
        mlflow.log_metrics(test_m)
        log.info(f"  TEST  mae={test_m['test_mae']:.2f}  rmse={test_m['test_rmse']:.2f}  r2={test_m['test_r2']:.3f}")

        # Test metrics excluding right-censored observations.
        mask = ~test_df["is_censored"].values
        test_m_unc = reg_metrics(y_test[mask], test_pred[mask], prefix="test_uncensored_")

        # ── Chart data for the reports ──────────────────────────────────────
        imp = shap_importance(best_pipe, X_val)
        imp.to_csv(MODELS_DIR / "shap_importance.csv", index=False)

        resid = pd.DataFrame({
            "predicted": np.round(test_pred, 3),
            "actual": y_test.values,
            "residual": np.round(test_pred - y_test.values, 3),
            "machine_type": test_df["machine_type"].values,
            "last_failure_mode": test_df["last_failure_mode"].values,
            "is_censored": test_df["is_censored"].values,
        })
        resid.to_csv(MODELS_DIR / "residuals_test.csv", index=False)

        # Calibration: mean actual RUL within predicted-RUL bins.
        bins = np.arange(0, RUL_HORIZON + 1, 10)
        resid["pred_bin"] = pd.cut(resid["predicted"].clip(0, RUL_HORIZON), bins=bins, include_lowest=True)
        calib = (resid.groupby("pred_bin", observed=True)
                 .agg(mean_predicted=("predicted", "mean"),
                      mean_actual=("actual", "mean"), count=("actual", "size"))
                 .reset_index(drop=True))
        calib.to_csv(MODELS_DIR / "calibration_test.csv", index=False)

        try:
            learning_curve_data(best_pipe, X_train, y_train).to_csv(
                MODELS_DIR / "learning_curve.csv", index=False)
        except Exception as e:
            log.info(f"  learning curve skipped: {e}")

        comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
        mlflow.log_artifact(str(MODELS_DIR / "model_comparison.csv"))

        # ── Register best model + production alias ──────────────────────────
        model_uri = f"runs:/{best['child_run_id']}/{best_type}_pipeline"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        client = MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, mv.version)
        client.update_model_version(MODEL_NAME, mv.version,
            description=(f"{best_type}. val MAE {best['val_mae']:.2f}, "
                         f"test MAE {test_m['test_mae']:.2f}, test R2 {test_m['test_r2']:.3f}."))
        mlflow.set_tag("model_version", mv.version)

        # ── Persist model + metrics summary for scoring and reports ─────────
        joblib.dump(best_pipe, MODELS_DIR / "rul_pipeline.pkl")
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "best_model_type": best_type,
            "model_version": mv.version,
            "n_optuna_trials": N_TRIALS,
            "rul_horizon_days": RUL_HORIZON,
            "target": TARGET,
            "feature_counts": {"categorical": len(CATEGORICAL_FEATURES),
                               "numerical": len(NUMERICAL_FEATURES),
                               "interaction": len(INTERACTION_FEATURES),
                               "total": len(ALL_FEATURES)},
            "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
            "models": [{"model_type": r["model_type"], "params": r["params"],
                        "val_mae": r["val_mae"], "val_rmse": r["val_rmse"], "val_r2": r["val_r2"]}
                       for r in results],
            "best_val": {"mae": best["val_mae"], "rmse": best["val_rmse"], "r2": best["val_r2"]},
            "test": {"mae": test_m["test_mae"], "rmse": test_m["test_rmse"], "r2": test_m["test_r2"]},
            "test_uncensored": {"mae": test_m_unc["test_uncensored_mae"],
                                "rmse": test_m_unc["test_uncensored_rmse"],
                                "r2": test_m_unc["test_uncensored_r2"]},
        }
        (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(f"Saved model + metrics to {MODELS_DIR}")
        log.info(f"Registered '{MODEL_NAME}' v{mv.version} @{PROD_ALIAS}")
if __name__ == "__main__":
    main()


# ==========================================================================
# Batch scoring
# ==========================================================================

SCORING_DIR = REPO_ROOT / "ml" / "data" / "scoring"
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


# ==========================================================================
# Monitoring and drift
# ==========================================================================

MONITORING_DIR = REPO_ROOT / "ml" / "data" / "monitoring"
DRIFT_THRESHOLD    = 0.10   # Jensen-Shannon distance flag
PERF_TOLERANCE_DAYS = 3.0   # period MAE above (baseline + this) is degraded
TARGET_DRIFT_COL   = "target"
MAX_DRIFTED_FEATURES = 3    # feature-drift count that warrants investigation
def js_drift(ref: pd.DataFrame, cur: pd.DataFrame, cols: list, cat_cols: list):
    """Per-column Jensen-Shannon drift distances (ref vs cur). Returns a table
    and the Evidently result (for the HTML artifact)."""
    num_cols = [c for c in cols if c not in cat_cols]
    r, c = ref[cols].copy(), cur[cols].copy()
    for col in cat_cols:
        r[col] = r[col].astype(str); c[col] = c[col].astype(str)
    for col in num_cols:
        r[col] = r[col].astype(float); c[col] = c[col].astype(float)
    dd = DataDefinition(numerical_columns=num_cols, categorical_columns=cat_cols)
    rds = Dataset.from_pandas(r, data_definition=dd)
    cds = Dataset.from_pandas(c, data_definition=dd)
    metrics = [ValueDrift(column=col, method="jensenshannon") for col in cols]
    res = Report(metrics).run(reference_data=rds, current_data=cds)
    vals = [float(m.get("value")) for m in res.dict()["metrics"]]
    table = pd.DataFrame({
        "feature": cols,
        "drift_score": [round(v, 4) for v in vals],
        "drift_detected": [v >= DRIFT_THRESHOLD for v in vals],
    }).sort_values("drift_score", ascending=False).reset_index(drop=True)
    return table, res
def one_col_drift(ref_vals: pd.Series, cur_vals: pd.Series, name: str) -> float:
    ref = pd.DataFrame({name: ref_vals.astype(float).values})
    cur = pd.DataFrame({name: cur_vals.astype(float).values})
    dd = DataDefinition(numerical_columns=[name], categorical_columns=[])
    res = Report([ValueDrift(column=name, method="jensenshannon")]).run(
        reference_data=Dataset.from_pandas(ref, data_definition=dd),
        current_data=Dataset.from_pandas(cur, data_definition=dd))
    return float(res.dict()["metrics"][0]["value"])
