import pandas as pd
import numpy as np

def validate_schema(feature_df):
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

    missing_cols = set(REQUIRED_COLS) - set(feature_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if feature_df[list(REQUIRED_COLS)].isna().any().any():
        raise ValueError("NaNs detected")

    if (feature_df["load"] < 0).any(): raise ValueError("Negative load")
    if (feature_df["solar"] < 0).any(): raise ValueError("Negative solar")
    if (feature_df["wind"] < 0).any(): raise ValueError("Negative wind")
    if not feature_df["renewables_share"].between(0, 5).all(): raise ValueError("Bad renewables_share")
    if not feature_df["temperature"].between(-40, 50).all(): raise ValueError("Bad temperature")
    if not feature_df["hour"].between(0, 23).all(): raise ValueError("Bad hour")
    if not feature_df["dayofweek"].between(0, 6).all(): raise ValueError("Bad dayofweek")

    if not feature_df["utc_timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps not monotonic")

    low_var = feature_df.std(numeric_only=True) < 1e-6
    if low_var.any():
        raise RuntimeError(f"Low variance features: {list(low_var[low_var].index)}")

    print("Schema + semantic + variance checks passed")