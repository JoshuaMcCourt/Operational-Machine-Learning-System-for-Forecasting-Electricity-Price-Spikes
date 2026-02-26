import pandas as pd
import numpy as np
from pathlib import Path
import json
from config import *

ts_df = pd.read_csv(TIME_SERIES_FILE)
weather_df = pd.read_csv(WEATHER_FILE)

ts_df["utc_timestamp"] = pd.to_datetime(ts_df["utc_timestamp"], utc=True)
weather_df["utc_timestamp"] = pd.to_datetime(weather_df["utc_timestamp"], utc=True)

load_col  = f"{ZONE}_load_actual_entsoe_transparency"
price_col = f"{ZONE}_price_day_ahead"
solar_col = f"{ZONE}_solar_generation_actual"
wind_col  = f"{ZONE}_wind_onshore_generation_actual"
temp_col  = "DE_temperature"

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

df = df[df["load"].notna() & df["price"].notna()].dropna(subset=["solar"]).reset_index(drop=True)
df = df.sort_values("utc_timestamp").reset_index(drop=True)

df["load_lag_1h"] = df["load"].shift(1)
df["load_roll_24h_mean"] = df["load"].rolling(24, min_periods=6).mean()
df["price_roll_24h_std"] = df["price"].rolling(24, min_periods=6).std().fillna(0)
df["solar_lag_1h"] = df["solar"].shift(1)
df["wind_lag_1h"] = df["wind"].shift(1)
df["renewables_share"] = (df["solar"] + df["wind"]) / df["load"]
df["hour"] = df["utc_timestamp"].dt.hour
df["dayofweek"] = df["utc_timestamp"].dt.dayofweek

feature_df = df.dropna().reset_index(drop=True)

version = FEATURE_VERSION
date_str = pd.Timestamp.now().strftime("%Y%m%d")
feature_file = PROCESSED_DIR / f"feature_table_{version}_{date_str}.csv"
feature_df.to_csv(feature_file, index=False)

reference_file = PROCESSED_DIR / f"reference_features_{version}_{date_str}.csv"
feature_df.to_csv(reference_file, index=False)

print(f"Feature snapshot: {feature_file.resolve()}")
print(f"Reference snapshot: {reference_file.resolve()}")