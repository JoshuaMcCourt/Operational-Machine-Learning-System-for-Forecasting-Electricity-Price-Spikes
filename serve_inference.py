import joblib
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from src.feature_parity import validate_inference_schema
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

    # CI/CD feature contract enforcement
    validate_inference_schema(X, SCHEMA_PATH)
    
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