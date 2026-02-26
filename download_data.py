import requests
from pathlib import Path

OPSd_BASE_URL = "https://data.open-power-system-data.org/time_series/latest"
FILES = {
    "time_series.csv": f"{OPSd_BASE_URL}/time_series_60min_singleindex.csv",
    "weather_data.csv": f"{OPSd_BASE_URL}/weather_data.csv",
}

RAW_DIR = Path("data/raw/opsd")
RAW_DIR.mkdir(parents=True, exist_ok=True)

for fname, url in FILES.items():
    out_path = RAW_DIR / fname
    if not out_path.exists():
        print(f"Downloading {fname}...")
        r = requests.get(url)
        r.raise_for_status()
        out_path.write_bytes(r.content)
    else:
        print(f"{fname} already exists")

print("Raw data download complete.")