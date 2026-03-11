# Project Structure for Electricity Price Spike ML Pipeline

#Project: Electricity Price Spike Prediction (DE_LU Zone)
#Purpose: Build, validate, and serve a production-ready ML model for price spike detection.
#Features: 
#- Data ingestion, preprocessing, and feature engineering
#- Schema & semantic validation
#- Price spike classification with temporal ML + Optuna tuning
#- Model calibration and evaluation
#- Production inference, drift monitoring, logging
#- Shadow deployment for A/B comparison

#ml-open-power-project/
#├── README.md                # Project overview, architecture, setup, and usage instructions
#├── requirements.txt         # Python dependencies for training, inference, and monitoring
#│
#├── data/
#│   ├── raw/                 # Immutable raw input datasets (never modified)
#│   │   └── opsd/
#│   │       ├── time_series.csv   # Historical load, price, solar, wind time series
#│   │       └── weather_data.csv  # Historical weather features aligned to timestamps
#│   ├── processed/           # Versioned feature tables and reference datasets for drift monitoring
#│   └── features/            # Schema snapshots and feature metadata for validation and CI checks
#│ 
#├── scripts/
#│   ├── download_data.py     # Used so the datasets can be reproduced locally in GitHub
#│
#├── models/
#│   └── logs/                # Production inference logs, drift metrics, and shadow deployment telemetry
#│
#├── src/
#│   ├── config.py            # Centralized paths, thresholds, and operational constants
#│   ├── ingest_features.py   # Raw data ingestion, feature engineering, and snapshot versioning
#│   ├── validate_features.py # Schema enforcement, semantic validation, and OOD checks
#│   ├── train_model.py       # Target construction, temporal splitting, Optuna tuning, model training
#│   ├── serve_inference.py   # Production batch inference with drift monitoring and logging
#│   └── shadow_deploy.py     # Parallel shadow inference for A/B testing and safe model iteration
#│   └── feature_parity.py    # Validates train + inference feature parity (schema, order, dtypes) to prevent CI/CD deployment mismatches.
#│
#└── notebook/
#    └── full_pipeline.ipynb    # Original end-to-end notebook prior to modularization


#Installation of all required packages

# Upgrade pip using Python executable 
!python -m pip install --upgrade pip

# Data handling
!pip install pandas numpy pyarrow

# Modeling
!pip install lightgbm xgboost scikit-learn optuna

# Model serving
!pip install fastapi uvicorn pydantic joblib

# Data visualization (optional for EDA)
!pip install matplotlib seaborn

# Utilities
!pip install tqdm


# Load Raw Data + Build Features (Production-Grade)
# Includes feature snapshot versioning & reference dataset for drift monitoring


import pandas as pd
import numpy as np
from pathlib import Path
import json


# File paths

RAW_DIR = Path(r"C:\Users\JoshuaMcCourt\Documents\Programming Work\Python\ML Open Power Project\data\raw\opsd")
TIME_SERIES_FILE = RAW_DIR / "time_series.csv"
WEATHER_FILE     = RAW_DIR / "weather_data.csv"


# Load raw data

ts_df = pd.read_csv(TIME_SERIES_FILE)
weather_df = pd.read_csv(WEATHER_FILE)

ts_df["utc_timestamp"] = pd.to_datetime(ts_df["utc_timestamp"], utc=True)
weather_df["utc_timestamp"] = pd.to_datetime(weather_df["utc_timestamp"], utc=True)


# Zone-specific columns

ZONE = "DE_LU"
load_col  = f"{ZONE}_load_actual_entsoe_transparency"
price_col = f"{ZONE}_price_day_ahead"
solar_col = f"{ZONE}_solar_generation_actual"
wind_col  = f"{ZONE}_wind_onshore_generation_actual"
temp_col  = "DE_temperature"


# Merge datasets

df = ts_df[["utc_timestamp", load_col, price_col, solar_col, wind_col]].merge(
    weather_df[["utc_timestamp", temp_col]],
    on="utc_timestamp",
    how="inner"
).rename(columns={
    load_col: "load",
    price_col: "price",
    solar_col: "solar",
    wind_col: "wind",
    temp_col: "temperature"
})

# Keep only valid load/price & drop missing solar
df = df[df["load"].notna() & df["price"].notna()].dropna(subset=["solar"]).reset_index(drop=True)


# Temporal and lag features

df = df.sort_values("utc_timestamp").reset_index(drop=True)

df["load_lag_1h"] = df["load"].shift(1)
df["load_roll_24h_mean"] = df["load"].rolling(24, min_periods=6).mean()
df["price_roll_24h_std"] = df["price"].rolling(24, min_periods=6).std().fillna(0)
df["solar_lag_1h"] = df["solar"].shift(1)
df["wind_lag_1h"] = df["wind"].shift(1)
df["renewables_share"] = (df["solar"] + df["wind"]) / df["load"]
df["hour"] = df["utc_timestamp"].dt.hour
df["dayofweek"] = df["utc_timestamp"].dt.dayofweek


# Drop rows with missing lag/rolling values

feature_df = df.dropna(subset=[
    "load_lag_1h",
    "load_roll_24h_mean",
    "solar_lag_1h",
    "wind_lag_1h"
]).reset_index(drop=True)


# Feature contract + variance checks

FEATURES = [
    "load", "price", "solar", "wind", "temperature",
    "load_lag_1h", "load_roll_24h_mean",
    "price_roll_24h_std",
    "solar_lag_1h", "wind_lag_1h", "renewables_share",
    "hour", "dayofweek"
]

missing = set(FEATURES) - set(feature_df.columns)
if missing:
    raise ValueError(f"Missing expected features: {missing}")

low_var = feature_df[FEATURES].std() < 1e-6
if low_var.any():
    bad = low_var[low_var].index.tolist()
    raise RuntimeError(f"Low-variance features detected: {bad}")


# Feature statistics

feature_stats = feature_df[FEATURES].agg(["mean","std"]).T
print("\nFeature statistics (mean/std):")
print(feature_stats)


# Feature snapshot versioning

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Create versioned filename with date
version = "v1"  # increment manually or automate later
date_str = pd.Timestamp.now().strftime("%Y%m%d")
feature_file = processed_dir / f"feature_table_{version}_{date_str}.csv"
feature_df.to_csv(feature_file, index=False)
print(f"\nVersioned feature snapshot saved: {feature_file.resolve()}")


# Reference dataset for drift monitoring

reference_file = processed_dir / f"reference_features_{version}_{date_str}.csv"
feature_df[FEATURES].to_csv(reference_file, index=False)
print(f"Reference dataset saved for drift monitoring: {reference_file.resolve()}")


# Summary

print("\nFinal Feature DF shape:", feature_df.shape)
display(feature_df.head())
print("\nFeature columns included:")
print(feature_df.columns.tolist())


# Schema + Semantic Validation + Training Safety Checks
# Includes automated anomaly / OOD monitoring for alerts

import pandas as pd
import numpy as np


# Required schema (columns + expected dtypes)

REQUIRED_COLS = {
    "utc_timestamp": "datetime64[ns, UTC]",
    "load": "float",
    "price": "float",
    "solar": "float",
    "wind": "float",
    "temperature": "float",
    "load_lag_1h": "float",
    "load_roll_24h_mean": "float",
    "price_roll_24h_std": "float",
    "solar_lag_1h": "float",
    "wind_lag_1h": "float",
    "renewables_share": "float",
    "hour": "int",
    "dayofweek": "int"
}


# Column existence check

missing_cols = set(REQUIRED_COLS) - set(feature_df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in feature_df: {missing_cols}")
print("All required columns present")


# NaN validation (hard fail)

nan_rates = feature_df[list(REQUIRED_COLS)].isna().mean()
if nan_rates.max() > 0:
    bad_cols = nan_rates[nan_rates > 0].index.tolist()
    raise ValueError(f"NaNs detected in required feature columns: {bad_cols}")
print("No NaNs in required columns")


# Semantic validation (domain constraints)

if (feature_df["load"] < 0).any():
    raise ValueError("Negative load values detected")
if (feature_df["solar"] < 0).any():
    raise ValueError("Negative solar generation detected")
if (feature_df["wind"] < 0).any():
    raise ValueError("Negative wind generation detected")
if not feature_df["renewables_share"].between(0, 5).all():
    raise ValueError("renewables_share outside expected bounds")
if not feature_df["temperature"].between(-40, 50).all():
    raise ValueError("Temperature outside realistic bounds")
if not feature_df["hour"].between(0, 23).all():
    raise ValueError("Invalid hour values")
if not feature_df["dayofweek"].between(0, 6).all():
    raise ValueError("Invalid dayofweek values")
print("Semantic value ranges valid")


# Temporal validation

if not feature_df["utc_timestamp"].is_monotonic_increasing:
    raise ValueError("Timestamps not sorted — temporal leakage risk")
time_deltas = feature_df["utc_timestamp"].diff().dropna().dt.total_seconds() / 3600
hourly_ratio = (time_deltas == 1).mean()
if hourly_ratio < 0.95:
    print("Warning: Non-hourly gaps detected in data")


# Feature logic / anti-leakage

lag_equal_ratio = (feature_df["load"] == feature_df["load_lag_1h"]).mean()
if lag_equal_ratio > 0.10:
    print("Warning: load_lag_1h frequently equals current load")

rolling_variance_check = feature_df["load_roll_24h_mean"].std() >= feature_df["load"].std()
if rolling_variance_check:
    print("Warning: load_roll_24h_mean variance >= raw load variance")

# Verify renewables_share is consistent with solar + wind / load
renewables_recalc = (feature_df["solar"] + feature_df["wind"]) / feature_df["load"]
max_diff = (feature_df["renewables_share"] - renewables_recalc).abs().max()
if max_diff > 1e-6:
    raise ValueError("renewables_share does not match solar + wind / load")
print("Feature logic consistency checks passed")


# Training-time numeric safety / variance checks

numeric_features = [c for c in REQUIRED_COLS if feature_df[c].dtype in [np.float64, np.int64]]
low_var = feature_df[numeric_features].std() < 1e-6
if low_var.any():
    bad_feats = low_var[low_var].index.tolist()
    raise RuntimeError(f"Low-variance features detected: {bad_feats}")


# Automated anomaly / Out-of-Distribution (OOD) monitoring
# Using Mean ± 4*(Standard deviation) from historical stats as a simple alert system

anomaly_warnings = []

for col in numeric_features:
    col_mean = feature_df[col].mean()
    col_std  = feature_df[col].std()
    upper_bound = col_mean + 4*col_std
    lower_bound = col_mean - 4*col_std
    ood_rows = feature_df[(feature_df[col] > upper_bound) | (feature_df[col] < lower_bound)]
    if not ood_rows.empty:
        anomaly_warnings.append((col, len(ood_rows)))
        print(f"Anomaly detected in '{col}': {len(ood_rows)} rows outside ±4σ")

if not anomaly_warnings:
    print("No out-of-distribution anomalies detected")

# Summary statistics
print("\nFeature stats summary:")
display(feature_df[list(REQUIRED_COLS)].describe())

# Prepare training matrices
X_train = feature_df[numeric_features].copy()
y_train = (X_train["price"].pct_change().shift(-1) > 0.05).astype(int)  # Example target for spike detection

# Post-feature-engineering statistics
feature_stats = X_train.agg(["mean", "std"]).T
print("\nPost-feature-engineering statistics:")
display(feature_stats)

print("\nSchema + semantic validation + training safety + OOD monitoring checks passed")


# Persist Feature Store (Processed Data Layer)

from pathlib import Path

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

feature_path = processed_dir / "feature_table.csv"
feature_df.to_csv(feature_path, index=False)

print(f"Feature table saved to: {feature_path.resolve()}")


# Schema Snapshot

import json
from pathlib import Path
import pandas as pd

schema_dir = Path("data/features")
schema_dir.mkdir(parents=True, exist_ok=True)

schema_path = schema_dir / f"schema_snapshot_v1_{pd.Timestamp.now().strftime('%Y%m%d')}.json"

schema = {
    "columns": list(feature_df.columns),
    "dtypes": feature_df.dtypes.astype(str).to_dict(),
    "feature_stats": feature_df.describe().to_dict()  # optional for monitoring
}

with open(schema_path, "w") as f:
    json.dump(schema, f, indent=2)

print(f"Schema snapshot saved: {schema_path.resolve()}")


# Price Spike Classification + Temporal ML Training
# Includes Optuna automated hyperparameter tuning

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime, timezone

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import optuna


# Define spike threshold (domain-driven)

SPIKE_QUANTILE = 0.95
price_threshold = feature_df["price"].quantile(SPIKE_QUANTILE)

feature_df["price_spike"] = (feature_df["price"] >= price_threshold).astype(int)

print(f"Price spike threshold (p{int(SPIKE_QUANTILE*100)}): {price_threshold:.2f}")
print("Spike class balance:")
print(feature_df["price_spike"].value_counts(normalize=True))


# Causal lag features / anti-leakage

feature_df["price_lag_1h"] = feature_df["price"].shift(1)
feature_df["price_roll_24h_std_lag"] = feature_df["price"].shift(1).rolling(24, min_periods=6).std()

feature_df["net_load"] = feature_df["load"] - (feature_df["solar"] + feature_df["wind"])
feature_df["net_load_lag_1h"] = feature_df["net_load"].shift(1)

feature_df["load_ramp_1h"] = feature_df["load"] - feature_df["load_lag_1h"]
feature_df["wind_ramp_1h"] = feature_df["wind"] - feature_df["wind_lag_1h"]
feature_df["solar_ramp_1h"] = feature_df["solar"] - feature_df["solar_lag_1h"]

# Drop rows with NaNs introduced by lag/rolling
feature_df = feature_df.dropna().reset_index(drop=True)

TARGET = "price_spike"

FEATURES = [
    "load",
    "solar",
    "wind",
    "temperature",
    "load_lag_1h",
    "load_roll_24h_mean",
    "price_lag_1h",
    "price_roll_24h_std_lag",
    "solar_lag_1h",
    "wind_lag_1h",
    "renewables_share",
    "net_load_lag_1h",
    "load_ramp_1h",
    "wind_ramp_1h",
    "solar_ramp_1h",
    "hour",
    "dayofweek",
]

X = feature_df[FEATURES]
y = feature_df[TARGET]


# Temporal split (walk-forward)

SPLIT_DATE = feature_df["utc_timestamp"].quantile(0.80)

train_mask = feature_df["utc_timestamp"] < SPLIT_DATE
X_train, X_val = X[train_mask], X[~train_mask]
y_train, y_val = y[train_mask], y[~train_mask]

print(f"Train period: {feature_df[train_mask]['utc_timestamp'].min()} → {feature_df[train_mask]['utc_timestamp'].max()}")
print(f"Val period:   {feature_df[~train_mask]['utc_timestamp'].min()} → {feature_df[~train_mask]['utc_timestamp'].max()}")
print("Train size:", X_train.shape, "Val size:", X_val.shape)


# Optuna hyperparameter tuning

def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 32, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "class_weight": {0: 1, 1: 20},
        "random_state": 42,
        "n_jobs": -1
    }
    model = LGBMClassifier(**param)
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_probs)
    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params
print(f"\nBest Optuna parameters: {best_params}")


# Train final tuned model

final_model = LGBMClassifier(
    **best_params,
    class_weight={0:1, 1:20},
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train, y_train)


# Probability calibration

calibrated_model = CalibratedClassifierCV(estimator=final_model, method="isotonic", cv=3)
calibrated_model.fit(X_train, y_train)
val_probs = calibrated_model.predict_proba(X_val)[:, 1]


# Evaluation metrics

roc_auc = roc_auc_score(y_val, val_probs)
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
pr_auc = auc(recall, precision)

print(f"\nValidation ROC-AUC: {roc_auc:.4f}")
print(f"Validation PR-AUC:  {pr_auc:.4f}")


# Cost-optimal threshold + ops guardrails

FN_COST, FP_COST = 5.0, 1.0
costs = FN_COST * (1 - recall[:-1]) + FP_COST * precision[:-1]
best_idx = np.argmin(costs)
optimal_threshold = thresholds[best_idx]

MIN_ALERT_RATE = 0.005
min_alert_threshold = np.quantile(val_probs, 1 - MIN_ALERT_RATE)
final_threshold = min(optimal_threshold, min_alert_threshold)

y_val_final = (val_probs >= final_threshold).astype(int)

print(f"\nRaw optimal threshold: {optimal_threshold:.4f}")
print(f"Min-alert threshold:   {min_alert_threshold:.4f}")
print(f"Final serving threshold: {final_threshold:.4f}")

print("\nClassification report @ production threshold:")
print(classification_report(y_val, y_val_final, digits=4, zero_division=0))


# Diagnostics & calibration sanity checks

pred_rate = y_val_final.mean()
true_rate = y_val.mean()
mean_proba, max_proba = float(val_probs.mean()), float(val_probs.max())

print("\nDiagnostics:")
print(f"Predicted spike rate: {pred_rate:.4%}")
print(f"True spike rate:      {true_rate:.4%}")
print(f"Mean predicted prob:  {mean_proba:.6f}")
print(f"Max predicted prob:   {max_proba:.6f}")

if pred_rate < 0.001 or pred_rate > 0.5:
    print("Warning: Abnormal alert rate detected — check threshold / calibration")


# Feature importance

importances = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance (top 10):")
display(importances.head(10))


# Persist model + schema + metadata

ARTIFACT_DIR = Path("models")
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "price_spike_lgbm_calibrated_optuna.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "price_spike_schema_optuna.json"

joblib.dump(calibrated_model, MODEL_PATH)

schema = {
    "features": FEATURES,
    "target": TARGET,
    "price_spike_quantile": float(SPIKE_QUANTILE),
    "price_spike_threshold": float(price_threshold),
    "cost_optimal_threshold": float(optimal_threshold),
    "final_serving_threshold": float(final_threshold),
    "model_type": "LightGBMClassifier + Optuna + IsotonicCalibration",
    "trained_on_rows": int(len(feature_df)),
    "roc_auc_val": float(roc_auc),
    "pr_auc_val": float(pr_auc),
    "alert_rate_val": float(pred_rate),
    "optuna_trials": len(study.trials),
    "optuna_best_params": best_params,
    "saved_at": datetime.now(timezone.utc).isoformat()
}

with open(SCHEMA_PATH, "w") as f:
    json.dump(schema, f, indent=2)

print(f"\nCalibrated & tuned model saved to: {MODEL_PATH}")
print(f"Schema + metadata saved to: {SCHEMA_PATH}")


# CI/CD Train + Inference Feature Parity Validation

import json
import pandas as pd

def validate_inference_schema(X: pd.DataFrame, schema_path: Path):
    """
    Validates that the inference features match the training schema.
    Ensures:
      - All required columns exist
      - Column order is preserved
      - Dtypes match saved training schema
    """
    with open(schema_path, "r") as f:
        saved_schema = json.load(f)
    
    schema_cols = saved_schema["features"]
    
    # Column existence check
    missing_cols = set(schema_cols) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in inference data: {missing_cols}")
    
    # Column order check
    if list(X.columns) != schema_cols:
        raise ValueError("Column order mismatch with training schema")
    
    # Dtype check (fallback to inferred if dtypes not saved)
    saved_dtypes = saved_schema.get("dtypes", {})
    for col in schema_cols:
        expected_dtype = saved_dtypes.get(col, str(X[col].dtype))
        if str(X[col].dtype) != expected_dtype:
            raise TypeError(f"Dtype mismatch for '{col}': expected {expected_dtype}, got {X[col].dtype}")
    
    print("Train + inference feature parity validated")


# Run validation for current features

validate_inference_schema(feature_df[FEATURES], SCHEMA_PATH)


# Production Inference + Drift Monitoring + Logging
# Includes real-time metrics hooks (Prometheus/Grafana)

import joblib
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import hashlib

# Metrics integration placeholder
try:
    from prometheus_client import Gauge, start_http_server
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    warnings.warn("Prometheus client not installed. Metrics will not be exported.")


# Paths & constants

ARTIFACT_DIR = Path("models")
MODEL_PATH = ARTIFACT_DIR / "price_spike_lgbm_calibrated.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "price_spike_schema.json"
REFERENCE_PATH = ARTIFACT_DIR / "training_features.csv"
LOG_DIR = ARTIFACT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DRIFT_ALERT_THRESHOLD = 0.20
PSI_ALERT_THRESHOLD = 0.25
ALERT_RATE_MAX = 0.50
ALERT_RATE_MIN = 0.001
MIN_PROBA_STD = 1e-4
MIN_FEATURE_STD = 1e-6


# Load model + schema

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH) as f:
    schema = json.load(f)

FEATURES = schema["features"]
FEATURE_VERSION = schema.get("feature_version", "v1")
raw_threshold = float(schema.get("final_serving_threshold", 0.5))
THRESHOLD = float(np.clip(raw_threshold, 0.05, 0.95))
if raw_threshold < 0.05:
    warnings.warn("Serving threshold was extremely low and auto-clipped to 0.05")

MODEL_HASH = hashlib.md5(MODEL_PATH.read_bytes()).hexdigest()


# Reference dataset for drift monitoring

REFERENCE_DF = None
if REFERENCE_PATH.exists():
    REFERENCE_DF = (
        pd.read_csv(REFERENCE_PATH)
        .loc[:, FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )


# Feature validation & health

def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("NaNs detected in features.")
    return X

def assert_feature_health(X: pd.DataFrame):
    low_var = X.std() < MIN_FEATURE_STD
    if low_var.any():
        raise RuntimeError(f"Near-constant features: {list(low_var[low_var].index)}")


# PSI / drift monitoring

def population_stability_index(ref: pd.Series, cur: pd.Series, bins=10) -> float:
    ref_perc = np.histogram(ref, bins=bins)[0] / len(ref)
    cur_perc = np.histogram(cur, bins=bins)[0] / len(cur)
    return np.sum((cur_perc - ref_perc) * np.log((cur_perc + 1e-6) / (ref_perc + 1e-6)))

def detect_feature_drift(batch_X: pd.DataFrame, reference_X: pd.DataFrame) -> pd.DataFrame:
    drift = {}
    for col in FEATURES:
        mean_shift = (batch_X[col].mean() - reference_X[col].mean()) / (abs(reference_X[col].mean()) + 1e-8)
        psi = population_stability_index(reference_X[col], batch_X[col])
        drift[col] = {"mean_shift": mean_shift, "psi": psi}
    return pd.DataFrame(drift).T.sort_values("psi", ascending=False)


# Logging

def log_predictions(df: pd.DataFrame, log_file: Path):
    df.to_csv(log_file, mode="a" if log_file.exists() else "w",
              header=not log_file.exists(), index=False)


# Real-time metrics setup

if METRICS_ENABLED:
    start_http_server(8000)
    METRIC_ALERT_RATE = Gauge("price_spike_alert_rate", "Batch alert rate")
    METRIC_PROBA_STD = Gauge("price_spike_proba_std", "Batch probability std")
    METRIC_PSI_MAX = Gauge("price_spike_max_psi", "Maximum PSI in batch vs reference")


# Batch inference (resilient)

def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    X = validate_features(df)
    assert_feature_health(X)
    
    # Predict probabilities & classes
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    
    alert_rate = preds.mean()
    proba_std = probs.std()
    
    print(f"Batch alert rate: {alert_rate:.2%}")
    print(f"Probability std: {proba_std:.6f}")
    print(f"Serving threshold: {THRESHOLD:.4f}")
    
    if alert_rate > ALERT_RATE_MAX or alert_rate < ALERT_RATE_MIN:
        warnings.warn(f"Alert rate {alert_rate:.2%} outside expected bounds. Logging but not blocking.")
    if proba_std < MIN_PROBA_STD:
        warnings.warn(f"Probability std {proba_std:.6f} very low. Logging but not blocking.")
    
    result = pd.DataFrame({
        "utc_timestamp": df["utc_timestamp"].values,
        "price_spike_proba": probs,
        "price_spike_pred": preds,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_hash": MODEL_HASH,
        "feature_version": FEATURE_VERSION,
        "serving_threshold": THRESHOLD
    })
    
    # Drift monitoring
    if REFERENCE_DF is not None:
        drift_df = detect_feature_drift(X, REFERENCE_DF)
        high_drift = drift_df[(drift_df["psi"] > PSI_ALERT_THRESHOLD) |
                              (drift_df["mean_shift"].abs() > DRIFT_ALERT_THRESHOLD)]
        if not high_drift.empty:
            print("Drift detected in batch:")
            display(high_drift.head(10))
        
        # Export metrics to Prometheus/Grafana if enabled
        if METRICS_ENABLED:
            METRIC_PSI_MAX.set(drift_df["psi"].max())
            METRIC_ALERT_RATE.set(alert_rate)
            METRIC_PROBA_STD.set(proba_std)
    
    # Logging predictions
    log_file = LOG_DIR / f"batch_predictions_{datetime.now(timezone.utc).date()}.csv"
    log_predictions(result, log_file)
    
    return result


# Run production inference

prod_preds = score_batch(feature_df)
display(prod_preds.head())
print("Production inference completed successfully")


# Shadow Deployment (Parallel / A/B Inference)

# This shadow setup runs a "trial" model in parallel to production.
# It allows us to compare predictions, track drift, and measure new model performance
# without impacting the live serving system.

import copy

# For example: create a slightly different calibration for shadow evaluation
shadow_model = copy.deepcopy(model)

# Simulate hyperparameter variation or threshold experimentation
shadow_threshold = np.clip(THRESHOLD * 0.9, 0.05, 0.95)

def score_shadow_batch(df: pd.DataFrame) -> pd.DataFrame:
    X = validate_features(df)
    assert_feature_health(X)
    
    shadow_probs = shadow_model.predict_proba(X)[:, 1]
    shadow_preds = (shadow_probs >= shadow_threshold).astype(int)
    
    result = pd.DataFrame({
        "utc_timestamp": df["utc_timestamp"].values,
        "shadow_proba": shadow_probs,
        "shadow_pred": shadow_preds,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_hash": MODEL_HASH,
        "feature_version": FEATURE_VERSION,
        "shadow_threshold": shadow_threshold
    })
    
    # Compare with production predictions
    result["prod_pred"] = prod_preds["price_spike_pred"].values
    result["disagreement"] = (result["shadow_pred"] != result["prod_pred"]).astype(int)
    disagreement_rate = result["disagreement"].mean()
    
    print(f"Shadow disagreement rate vs prod: {disagreement_rate:.2%}")
    
    log_file = LOG_DIR / f"shadow_predictions_{datetime.now(timezone.utc).date()}.csv"
    log_predictions(result, log_file)
    
    return result


# Run shadow inference

shadow_preds = score_shadow_batch(feature_df)
display(shadow_preds.head())
print("Shadow deployment inference completed successfully")