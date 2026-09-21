"""
monitoring.py
Batch monitoring for the RUL model, run per calendar month (Jan-Mar 2026),
each month compared against the training/validation references. Four layers:

  1. Performance      - MAE / RMSE vs the held-out test baseline (uses actuals,
                        available here because failures land within the window)
  2. Target drift     - actual days-to-failure distribution vs training baseline
  3. Prediction drift - predicted-RUL distribution vs the validation reference
  4. Feature drift    - input feature distributions vs training

Drift metric: Evidently ValueDrift with the Jensen-Shannon distance (0-1);
drift is flagged when distance >= threshold (a distance test, not a p-value).

Retraining logic: performance or target drift are primary triggers (RETRAIN);
prediction or feature drift are secondary (INVESTIGATE). A trigger must persist
across two consecutive periods before it is recommended.

Run:  python ml/src/monitoring.py
"""
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from evidently import Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import (CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
                      INTERACTION_FEATURES, ALL_FEATURES, TARGET)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
REPO_ROOT      = Path(__file__).resolve().parents[2]
FEATURES_DIR   = REPO_ROOT / "ml" / "data" / "features"
SCORING_DIR    = REPO_ROOT / "ml" / "data" / "scoring"
MONITORING_DIR = REPO_ROOT / "ml" / "data" / "monitoring"
MODELS_DIR     = REPO_ROOT / "ml" / "models"
MLRUNS_DIR     = REPO_ROOT / "ml" / "mlruns"

MLFLOW_TRACKING = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "c02_rul_predictor"

PERIODS = [
    ("202601", "January 2026"),
    ("202602", "February 2026"),
    ("202603", "March 2026"),
]

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


def main():
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val_ref = pd.read_parquet(FEATURES_DIR / "validation_predictions.parquet")
    baseline_mae = json.loads((MODELS_DIR / "metrics.json").read_text())["test"]["mae"]
    train_target = train[~train["is_censored"]][TARGET] if "is_censored" in train else train[TARGET]

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in ALL_FEATURES]
    log.info(f"References: train {len(train):,} rows | baseline test MAE {baseline_mae:.2f}")

    rows = []
    for label, name in PERIODS:
        preds = pd.read_parquet(SCORING_DIR / f"predictions_{label}.parquet")
        unc = preds[~preds["is_censored"]]

        # 1. Performance
        mae  = float(mean_absolute_error(unc[TARGET], unc["predicted_days_to_failure"]))
        rmse = float(root_mean_squared_error(unc[TARGET], unc["predicted_days_to_failure"]))
        perf_degraded = mae > baseline_mae + PERF_TOLERANCE_DAYS

        # 2. Target drift (actual RUL distribution vs training)
        target_score = one_col_drift(train_target, unc[TARGET], "rul_target")
        target_drift = target_score >= DRIFT_THRESHOLD

        # 3. Prediction drift (predicted RUL vs validation reference)
        pred_score = one_col_drift(val_ref["predicted_days_to_failure"],
                                   preds["predicted_days_to_failure"], "rul_pred")
        pred_drift = pred_score >= DRIFT_THRESHOLD

        # 4. Feature drift
        ftable, fres = js_drift(train, preds, ALL_FEATURES, cat_cols)
        ftable.to_csv(MONITORING_DIR / f"feature_drift_{label}.csv", index=False)
        try:
            fres.save_html(str(MONITORING_DIR / f"drift_report_{label}.html"))
        except Exception as e:
            log.info(f"  [{label}] HTML save skipped: {e}")
        n_drifted = int(ftable["drift_detected"].sum())

        primary   = perf_degraded or target_drift
        secondary = pred_drift or (n_drifted > MAX_DRIFTED_FEATURES)
        status = "RETRAIN" if primary else "INVESTIGATE" if secondary else "HEALTHY"

        log.info(f"[{label}] MAE {mae:.2f} (base {baseline_mae:.2f}) | "
                 f"target JS {target_score:.3f} | pred JS {pred_score:.3f} | "
                 f"feat drifted {n_drifted}/{len(ALL_FEATURES)} | {status}")

        row = {"period_label": label, "period_name": name, "n_scored": len(preds),
               "mae": round(mae, 3), "rmse": round(rmse, 3), "baseline_mae": round(baseline_mae, 3),
               "perf_degraded": perf_degraded,
               "target_drift_score": round(target_score, 4), "target_drift": target_drift,
               "prediction_drift_score": round(pred_score, 4), "prediction_drift": pred_drift,
               "n_features_drifted": n_drifted, "n_features": len(ALL_FEATURES),
               "primary_trigger": primary, "secondary_trigger": secondary, "status": status}
        rows.append(row)

        with mlflow.start_run(run_name=f"monitoring_{label}"):
            mlflow.set_tags({"run_type": "monitoring", "period": label, "status": status})
            mlflow.log_metrics({k: float(v) for k, v in row.items()
                                if isinstance(v, (int, float, bool))})

    out = pd.DataFrame(rows)
    out.to_csv(MONITORING_DIR / "period_monitoring.csv", index=False)

    # Two-consecutive-period rule for the standing recommendation.
    def consecutive(flag):
        f = out[flag].tolist()
        return any(f[i] and f[i + 1] for i in range(len(f) - 1))

    if consecutive("primary_trigger"):
        recommendation = "RETRAIN"
    elif consecutive("secondary_trigger"):
        recommendation = "INVESTIGATE"
    else:
        recommendation = "HEALTHY"
    summary = {"recommendation": recommendation,
               "periods": out["status"].tolist(),
               "latest_status": rows[-1]["status"]}
    (MONITORING_DIR / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("\n" + out[["period_label", "mae", "target_drift_score",
                         "prediction_drift_score", "n_features_drifted", "status"]].to_string(index=False))
    log.info(f"Standing recommendation (two-consecutive rule): {recommendation}")


if __name__ == "__main__":
    main()
