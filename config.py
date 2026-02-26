from pathlib import Path

RAW_DIR = Path(r"C:\Users\JoshuaMcCourt\Documents\Programming Work\Python\ML Open Power Project\data\raw\opsd")
TIME_SERIES_FILE = RAW_DIR / "time_series.csv"
WEATHER_FILE     = RAW_DIR / "weather_data.csv"

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
MODEL_DIR = Path("models")
LOG_DIR = MODEL_DIR / "logs"

ZONE = "DE_LU"

FEATURE_VERSION = "v1"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)