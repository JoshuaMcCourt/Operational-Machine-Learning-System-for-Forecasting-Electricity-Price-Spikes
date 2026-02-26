# Operational-Machine-Learning-System-for-Forecasting-Electricity-Price-Spikes
End-to-end ML pipeline for detecting high-risk periods in day-ahead electricity prices using load, renewables, and weather data. Includes feature versioning, schema validation, drift detection, calibrated predictions, threshold tuning, and monitored production inference.

## Data Sources
Raw data is not committed to this repository due to size constraints and best practices.
This project uses data from Open Power System Data (OPSD).

To download the raw datasets:
```bash
scripts/download_data.py
