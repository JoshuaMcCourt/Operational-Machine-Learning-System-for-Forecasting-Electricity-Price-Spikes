# ML Open Power Project
### Production-Oriented ML Pipeline for Electricity Price Spike Risk

## Summary

This repository implements a modular, production-style machine learning pipeline for detecting elevated risk of electricity price spikes using historical demand, renewable generation, and weather data. The project emphasizes system design, reliability, and operational safeguards rather than just model performance.

The codebase mirrors how ML systems are typically structured in production: reproducible data ingestion, versioned feature generation, schema enforcement, train–serve parity checks, model training with tuning and calibration, and monitored batch inference with shadow deployments.

## What This Demonstrates

#### Machine Learning Engineering
    -Reproducible feature engineering from raw time-series inputs
    -Temporal feature construction (lags, rolling statistics)
    -Time-aware train/validation splits
    -Hyperparameter tuning with Optuna
    -Probability calibration for operational decision thresholds

#### Production Readiness
    -Schema snapshots and strict feature validation
    -Train + inference feature parity checks
    -Drift monitoring using PSI
    -Structured inference logging
    -Shadow deployments for safe model iteration

#### Applied Analytics
    -Rare-event classification (price spike risk)
    -Threshold tuning to manage alert rates
    -Monitoring prediction stability over time
    -Model lifecycle management across retrains

## Repository Structure - Key Modules

- `src/ingest_features.py` – Raw data ingestion and feature engineering  
- `src/validate_features.py` – Schema and semantic validation  
- `src/train_model.py` – Training, tuning, and calibration  
- `src/serve_inference.py` – Batch inference and monitoring  
- `src/feature_parity.py` – Train ↔ serve feature parity checks  

ml-open-power-project/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── opsd/
│   │       ├── time_series.csv
│   │       └── weather_data.csv
│   ├── processed/
│   └── features/
│
├── scripts/
│   └── download_data.py
│
├── models/
│   └── logs/
│
├── src/
│   ├── config.py
│   ├── ingest_features.py
│   ├── validate_features.py
│   ├── train_model.py
│   ├── serve_inference.py
│   ├── shadow_deploy.py
│   └── feature_parity.py
│
└── notebook/
    └── full_pipeline.ipynb

## Typical Workflow

#### Data Acquisition
Use scripts/download_data.py to fetch and reproduce the raw datasets locally.

#### Feature Ingestion & Versioning
src/ingest_features.py loads raw time-series data, constructs features, and writes versioned feature tables and schema snapshots.

#### Schema & Feature Validation
src/validate_features.py enforces column presence, constraints, and basic semantic checks.
src/feature_parity.py guarantees train–serve feature parity (schema, column order, dtypes).

#### Model Training
src/train_model.py performs target construction, temporal splits, Optuna tuning, and probability calibration. Trained artifacts and metadata are versioned.

#### Batch Inference & Monitoring
src/serve_inference.py runs production-style batch inference, logs predictions, and computes drift metrics.
src/shadow_deploy.py runs parallel inference for safe model comparisons and iteration.

## Engineering Focus

This project is designed to reflect production ML workflows:

    -Clear separation between ingestion, validation, training, and serving
    -Defensive checks to prevent silent train/serve mismatches
    -Feature schema versioning for reproducibility
    -Observability through drift metrics and structured logs
    -Shadow deployment patterns used in real ML systems

The emphasis is on building an ML system that is robust to data changes and operational failure modes.

## Use Case

Although framed around electricity price spike risk, the architecture generalizes to any time-series ML problem where schema stability, feature drift, and safe deployment practices are critical (e.g., demand forecasting, anomaly detection, fraud detection).

## Disclaimer

This repository is for demonstration and portfolio purposes only and is not intended for use in live trading or operational decision systems.







