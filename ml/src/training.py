"""
training.py
Trains the Remaining Useful Life (RUL) regressor.

Runs a three-way bake-off (regularized linear regression, random forest,
XGBoost), tunes each with Optuna against validation MAE, logs every run to
MLflow, selects the winner by validation MAE/RMSE, registers it in the MLflow
registry under the "production" alias, and evaluates it once on the held-out
test set. Model, metrics, and chart data are written to ml/models/ for the
report generators to consume.
"""
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import shap
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import learning_curve as sk_learning_curve
from sklearn.base import clone
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    engineer_features,
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    INTERACTION_FEATURES, TARGET, ID_COL,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
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
EXPERIMENT_NAME = "c02_rul_predictor"
MODEL_NAME      = "rul_predictor"
PROD_ALIAS      = "production"
N_TRIALS        = 100
RANDOM_SEED     = 42
RUL_HORIZON     = 60   # target cap in days


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH (objective = validation MAE, minimized)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACTS (data only; the report generators draw the brand-styled charts)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
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
