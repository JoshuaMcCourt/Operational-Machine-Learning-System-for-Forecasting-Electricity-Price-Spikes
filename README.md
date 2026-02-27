# ML Open Power Project
### End-to-End ML Pipeline for Electricity Price Spike Risk Detection

## Overview

This repository contains a complete machine learning workflow for identifying periods of elevated risk in day-ahead electricity prices using historical system load, renewable generation, and weather data. The focus of the project is on building a robust and maintainable system rather than a one-off modeling exercise.

In addition to model development, the pipeline includes data validation, feature versioning, drift monitoring, probability calibration, operational thresholding, and shadow inference. The structure and safeguards mirror the kinds of practices used in applied energy analytics and risk monitoring environments, where data quality and operational reliability are as important as raw predictive performance.

This project is intended as a practical demonstration of applied ML engineering and MLOps principles in an energy market setting.

Key Capabilities

Reproducible data ingestion and feature construction

Temporal features with lagged and rolling-window context

Schema enforcement and feature parity checks

Supervised classification for rare-event detection

Hyperparameter optimization with Optuna

Probability calibration for operational decision thresholds

Cost-aware alert threshold selection

Batch inference with alert-rate monitoring

Feature drift detection using PSI

Shadow deployment for safe model iteration

Structured logging for auditability and debugging

Repository Layout
ml-open-power-project/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── opsd/
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
Data Handling
Raw Inputs

Raw datasets are not committed to the repository due to size constraints. The pipeline expects the following inputs:

Time series data for load, price, and renewable generation

Weather data aligned to the same timestamps

To reproduce the datasets locally, run:

python scripts/download_data.py

All transformations and derived features are written to the processed data layer to preserve the original inputs.

Environment Setup
Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate     # Windows
Install dependencies
pip install -r requirements.txt
Running the Pipeline

The pipeline can be executed end-to-end or stage-by-stage.

python main.py

This performs:

Data ingestion

Feature engineering and validation

Feature snapshotting

Target construction

Temporal train/validation splitting

Hyperparameter tuning

Final model training and calibration

Threshold selection

Model persistence

Batch inference

Drift monitoring

Shadow deployment evaluation

Feature Parity and Deployment Safety

To avoid silent failures during deployment, the project enforces strict feature parity between training and inference. Incoming inference data must match the training schema in:

Column presence

Column order

Data types

This check is implemented in src/feature_parity.py and is designed to be run as part of a CI pipeline before deploying new code or models.

Drift Monitoring

Feature drift is monitored using Population Stability Index (PSI). Incoming batches are compared against a stored reference snapshot from training-time features. Alerts are raised when drift exceeds configured thresholds, helping surface upstream data issues or regime changes.

Shadow Inference

Shadow inference allows a candidate model or configuration to run alongside the production model without affecting live decisions. Outputs are logged and compared to quantify disagreement and assess whether a new model is suitable for promotion.

Model Artifacts and Logs

Trained models, schemas, thresholds, and performance metadata are stored in the models/ directory. Inference logs and drift metrics are written to models/logs/ to support monitoring, debugging, and retrospective analysis.

Design Goals

This repository is structured around the following principles:

Reproducibility over ad-hoc experimentation

Clear separation between data ingestion, validation, training, and serving

Operational safeguards to catch data and model failures early

Emphasis on maintainability and inspection rather than quick demos

Disclaimer

This project is for educational and demonstration purposes only. It is not intended for use in live trading or operational decision-making environments.

If you want, I can also help you:

tighten the README for recruiters,

write a short project summary for LinkedIn,

or produce architecture diagrams for your repo.

Yes, please write out a tightened README that I can include for recruiters

Here’s a tightened, recruiter-facing README you can drop straight into your repo. It’s concise, technically credible, and framed to highlight engineering maturity and applied ML skills rather than just modeling:

ML Open Power Project

Production-Oriented ML Pipeline for Electricity Price Spike Risk

Summary

This project implements a production-style machine learning pipeline for identifying elevated risk in day-ahead electricity prices using historical load, renewable generation, and weather data. The emphasis is on building a reliable, testable, and monitorable ML system rather than a standalone model.

The repository demonstrates applied MLOps practices, including feature validation, schema enforcement, feature parity checks, drift monitoring, and shadow deployment. The design mirrors how ML systems are typically built and maintained in real-world energy, forecasting, and risk analytics environments.

What This Demonstrates

ML Engineering

Reproducible data ingestion and feature engineering

Temporal feature construction with rolling windows and lags

Proper train/validation splitting for time series

Hyperparameter tuning with Optuna

Probability calibration for operational thresholds

Production Readiness

Schema validation and feature parity enforcement

Drift monitoring using PSI

Structured logging for inference and monitoring

Shadow deployment for safe iteration

Config-driven pipeline design

Applied Analytics

Rare-event classification (price spike risk)

Cost-aware thresholding

Monitoring alert rates over time

Model performance tracking across retrains

Repository Structure (High Level)
ml-open-power-project/
├── data/         # Raw, processed, and feature snapshot layers
├── src/          # Ingestion, validation, training, inference, monitoring
├── scripts/      # Reproducible data download utilities
├── models/       # Trained models and inference logs
└── notebook/     # Original exploratory notebook
How to Run (Local)
pip install -r requirements.txt
python main.py

The pipeline executes ingestion, feature construction, validation, training, calibration, inference, and monitoring in sequence.

Engineering Focus

This project prioritizes:

Clear separation of concerns (ingestion vs training vs serving)

Defensive checks to prevent silent deployment failures

Reproducibility and traceability of features and models

Observability through logging and drift metrics

The structure is intentionally closer to how production ML systems are built than to a typical notebook-based project.

Use Case

This system is framed around energy market risk detection, but the architecture generalizes to any time-series classification or forecasting problem where data quality, schema stability, and monitoring are operationally critical (e.g., demand forecasting, anomaly detection, fraud, churn).

Disclaimer

This repository is for demonstration and portfolio purposes only and is not intended for use in live trading or operational decision systems.

If you want, I can also:

help you write a one-paragraph LinkedIn project description,

draft resume bullet points from this repo, or

review your repo structure like a hiring manager would.
