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

feature_df = pd.read_csv("data/processed/feature_table.csv", parse_dates=["utc_timestamp"])

SPIKE_QUANTILE = 0.95
price_threshold = feature_df["price"].quantile(SPIKE_QUANTILE)
feature_df["price_spike"] = (feature_df["price"] >= price_threshold).astype(int)

feature_df["price_lag_1h"] = feature_df["price"].shift(1)
feature_df["price_roll_24h_std_lag"] = feature_df["price"].shift(1).rolling(24, min_periods=6).std()
feature_df["net_load"] = feature_df["load"] - (feature_df["solar"] + feature_df["wind"])
feature_df["net_load_lag_1h"] = feature_df["net_load"].shift(1)
feature_df["load_ramp_1h"] = feature_df["load"] - feature_df["load_lag_1h"]
feature_df["wind_ramp_1h"] = feature_df["wind"] - feature_df["wind_lag_1h"]
feature_df["solar_ramp_1h"] = feature_df["solar"] - feature_df["solar_lag_1h"]

feature_df = feature_df.dropna().reset_index(drop=True)

FEATURES = [
    "load","solar","wind","temperature","load_lag_1h","load_roll_24h_mean",
    "price_lag_1h","price_roll_24h_std_lag","solar_lag_1h","wind_lag_1h",
    "renewables_share","net_load_lag_1h","load_ramp_1h","wind_ramp_1h",
    "solar_ramp_1h","hour","dayofweek"
]

X = feature_df[FEATURES]
y = feature_df["price_spike"]

SPLIT_DATE = feature_df["utc_timestamp"].quantile(0.80)
train_mask = feature_df["utc_timestamp"] < SPLIT_DATE
X_train, X_val = X[train_mask], X[~train_mask]
y_train, y_val = y[train_mask], y[~train_mask]

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
    return roc_auc_score(y_val, val_probs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

final_model = LGBMClassifier(**study.best_params, class_weight={0:1,1:20}, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(final_model, method="isotonic", cv=3)
calibrated_model.fit(X_train, y_train)

joblib.dump(calibrated_model, "models/price_spike_lgbm_calibrated_optuna.joblib")
print("Model saved")