# ML Open Power Project
### Production-Oriented ML Pipeline for Electricity Price Spike Risk

## Summary

This repository implements a modular, production-style machine learning pipeline for detecting elevated risk of electricity price spikes using historical demand, renewable generation, and weather data. The project emphasizes system design, reliability, and operational safeguards rather than just model performance.

The codebase mirrors how ML systems are typically structured in production: reproducible data ingestion, versioned feature generation, schema enforcement, train–serve parity checks, model training with tuning and calibration, and monitored batch inference with shadow deployments.

## Business Context & Value

Electricity price spikes can create significant risk in energy markets, affecting operational stability, cost efficiency, and planning decisions.

This system demonstrates how machine learning can support:

- Early identification of high-risk pricing periods
- Operational risk monitoring in energy systems
- Data-driven decision support for market volatility
- Improved awareness of system stress conditions using external signals (demand, renewables, weather)

## System Architecture

The pipeline is structured as a production-grade ML system with clear separation of concerns:

Raw Data 

   ↓

Feature Engineering

   ↓

Schema Validation Layer

   ↓

Model Training & Calibration

   ↓

Model Artifacts / Versioning

   ↓

Batch Inference

   ↓

Monitoring & Drift Detection

### Design Principles
- Strict `train/serve feature parity enforcement` to prevent inference skew
- Versioned feature generation for reproducibility and auditability
- Modular pipeline design enabling independent retraining and deployment
- Built-in observability via structured logging and drift monitoring
- Shadow deployment support for safe model iteration

## What This Demonstrates

#### Machine Learning Engineering
- Reproducible feature engineering from raw time-series inputs
- Temporal feature construction (lags, rolling statistics, trend features)
- Time-aware train/validation splitting to prevent leakage
- Hyperparameter optimisation using Optuna
- Probability calibration for decision-threshold stability

#### Production Readiness
- Schema snapshots and strict feature validation
- Train–serve feature parity enforcement
- Drift monitoring using Population Stability Index (PSI)
- Structured inference logging for observability
- Shadow deployment for safe model comparison

#### Applied Analytics
- Rare-event classification (electricity price spike risk)
- Threshold tuning for operational alert control
- Monitoring prediction stability over time
- End-to-end model lifecycle management across retraining cycles

## Model Performance

The model was evaluated using time-aware validation to simulate real-world forecasting conditions and avoid temporal leakage.

#### Key metrics:
- ROC-AUC: 0.9966  
- Spike class prevalence: ~5.0% (highly imbalanced classification problem)  
- Price spike threshold (p95): 68.09  

The model demonstrates strong discriminative performance in identifying rare high-price events under severe class imbalance.

Threshold selection is explicitly optimised to balance:
- Sensitivity to rare spike events  
- Operational alert fatigue  
- Stability of predictions over time

Note: Precision, recall, and false positive rate are dependent on the chosen decision threshold and are monitored during inference for operational tuning.

## Data Sources

This project uses publicly available data from the `Open Power System Data (OPSD)` initiative.

Raw datasets are not included in the repository due to size and reproducibility best practices.

To reproduce:

`scripts/download_data.py`

## Repository Structure - Key Modules

- `src/config.py` - Centralized paths, thresholds, and operational constants
- `src/ingest_features.py` – Raw data ingestion and feature engineering  
- `src/validate_features.py` – Schema and semantic validation  
- `src/train_model.py` – Training, tuning, and calibration  
- `src/serve_inference.py` – Batch inference and monitoring  
- `src/feature_parity.py` – Train ↔ serve feature parity checks  

## Workflow
#### 1. Data Acquisition

Raw datasets are downloaded and prepared locally using:
`scripts/download_data.py`

#### 2. Feature Engineering & Versioning
- Time-series transformation (lags, rolling statistics, temporal signals)
- Versioned feature outputs for reproducibility
- Schema snapshotting for auditability

#### 3. Validation Layer
- Column-level schema enforcement
- Data type and constraint validation
- Semantic consistency checks across pipeline stages
- Feature parity enforcement between training and inference

#### 4. Model Training
- Time-aware train/test splitting
- Hyperparameter tuning using Optuna
- Class imbalance handling for rare spike events
- Probability calibration for operational stability
- Versioned model artifact storage

#### 5. Batch Inference & Monitoring
- Production-style batch scoring pipeline
- Structured prediction logging
- Drift detection using PSI
- Prediction stability monitoring over time
- Shadow deployment support for safe model comparison

## Engineering Focus

This project prioritises production machine learning system design over isolated model development.

Key engineering features include:
- Strict separation between ingestion, training, and serving layers
- Defensive design to prevent silent train/serve mismatches
- Feature schema versioning for reproducibility
- Monitoring layer for drift and prediction stability
- Shadow deployment patterns for safe experimentation

The system is designed to remain robust under data drift, schema changes, and operational constraints commonly encountered in real-world ML systems.

## Why This Project Matters

This project demonstrates capability beyond model development by focusing on `end-to-end machine learning system design`, including:

- Production reliability considerations
- Monitoring and lifecycle management
- Safe deployment strategies
- Data integrity enforcement
- Operational decision constraints rather than offline accuracy alone

It reflects how ML systems are designed and maintained in real production environments.

## Potential Extensions
- Deployment via FastAPI for real-time inference
- Airflow orchestration for scheduled pipelines
- Cloud deployment (AWS / GCP / Azure)
- MLflow integration for experiment tracking and model registry
- Streaming-based feature pipelines using Kafka
- Advanced anomaly detection for drift monitoring

## Disclaimer

This project is for portfolio and educational purposes only and is not intended for live trading or operational deployment in production environments.
