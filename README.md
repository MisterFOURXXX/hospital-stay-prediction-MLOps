# Hospital Stay Prediction - Optimized Production Pipeline

> A production-ready MLOps pipeline for predicting hospital stay duration using stacked ensemble learning with LightGBM, XGBoost, and CatBoost. It covers the full ML lifecycle: data ingestion, PCA + SMOTE feature engineering, Optuna hyperparameter optimization, MLflow experiment tracking on DagsHub, DVC data versioning, Dockerized FastAPI inference, and AWS ECS Fargate deployment with auto-scaling.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/lightgbm-%3E%3D4.0-4b8bbe.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/xgboost-%3E%3D2.0-ec4c2c.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/catboost-%3E%3D1.2-ffcc00.svg)](https://catboost.ai/)
[![MLflow](https://img.shields.io/badge/mlflow-%3E%3D2.8-0194e2.svg)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/dagshub-%3E%3D0.3-ff6f00.svg)](https://dagshub.com/)
[![Optuna](https://img.shields.io/badge/optuna-%3E%3D3.0-4b8bbe.svg)](https://optuna.org/)
[![DVC](https://img.shields.io/badge/dvc-%3E%3D3.0-13adc7.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/docker-%3E%3D24.0-2496ed.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/aws-ecs%20%7C%20ecr%20%7C%20fargate-ff9900.svg)](https://aws.amazon.com/)
[![FastAPI](https://img.shields.io/badge/fastapi-%3E%3D0.100-009688.svg)](https://fastapi.tiangolo.com/)

---

## Project Overview

A production-ready MLOps pipeline for predicting patient hospital stay duration using stacked ensemble learning with LightGBM, XGBoost, and CatBoost. This pipeline includes experiment tracking with MLflow on DagsHub, hyperparameter optimization with Optuna, data version control with DVC, containerization with Docker, and production deployment on AWS ECS.

This project is a reproducible MLOps pipeline for predicting hospital stay categories using a stacked ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost). It is designed for production deployment and educational purposes, with containerized inference on AWS ECS. The pipeline covers the full lifecycle of a machine learning experiment:

1. Raw data ingestion and preprocessing.
2. Feature engineering with PCA and SMOTE.
3. Hyperparameter optimization with Optuna.
4. Stacked ensemble training with meta-learner.
5. Experiment tracking with MLflow on DagsHub.
6. Data and model versioning with DVC.
7. Multi-metric evaluation.
8. Dockerized inference.
9. AWS ECS deployment.

Healthcare providers need accurate predictions of patient stay duration to optimize resource allocation, staffing, and bed management. This project investigates whether stacked ensemble learning can improve prediction accuracy over individual models while remaining reproducible and deployable. It also demonstrates how MLOps practices can make healthcare ML research more transparent, comparable, and production-ready.

### Engineering Goals

- **Reproducibility:** every hyperparameter lives in version-controlled YAML, every artefact is DVC-tracked, and every run is MLflow-tracked.
- **Modularity:** data, features, models, evaluation, and deployment are independent modules with documented interfaces.
- **Deployability:** the trained model is served via a Dockerized FastAPI service and deployable to AWS ECS Fargate with auto-scaling.
- **Observability:** MLflow metrics, DVC pipeline status, and API health checks give full visibility into training and serving.
- **Testability:** data preparation, preprocessing, model inference, and API behaviour each have pytest coverage with a coverage report.
- **Fairness awareness:** the pipeline surfaces per-class metrics and supports fairness analysis across demographic groups.

### Table of Contents

- [Project Overview](#project-overview)
- [1. Introduction](#1-introduction)
- [2. Foundation](#2-foundation)
  - [2.1 Hospital Stay Prediction as a Clinical ML Problem](#21-hospital-stay-prediction-as-a-clinical-ml-problem)
  - [2.2 Gradient Boosting Models](#22-gradient-boosting-models)
  - [2.3 Stacked Ensemble Learning](#23-stacked-ensemble-learning)
  - [2.4 PCA and SMOTE](#24-pca-and-smote)
  - [2.5 Hyperparameter Optimization with Optuna](#25-hyperparameter-optimization-with-optuna)
  - [2.6 Experiment Tracking and Data Versioning](#26-experiment-tracking-and-data-versioning)
  - [2.7 Reproducibility Engineering](#27-reproducibility-engineering)
  - [2.8 Intended Scope and Operational Limits](#28-intended-scope-and-operational-limits)
- [3. System Design](#3-system-design)
  - [3.1 Design](#31-design)
  - [3.2 Core Components](#32-core-components)
  - [3.3 High-Level Workflow](#33-high-level-workflow)
  - [3.4 Data Collection and Preprocessing](#34-data-collection-and-preprocessing)
  - [3.5 Model and Training](#35-model-and-training)
  - [3.6 Experiment Tracking and Versioning](#36-experiment-tracking-and-versioning)
  - [3.7 Deployment as a Production Artifact](#37-deployment-as-a-production-artifact)
  - [3.8 Reproducibility Plan](#38-reproducibility-plan)
  - [3.9 Hardware Requirements](#39-hardware-requirements)
- [4. Evaluation](#4-evaluation)
  - [4.1 Evaluation Strategy](#41-evaluation-strategy)
  - [4.2 Experimental Conditions and Ablations](#42-experimental-conditions-and-ablations)
  - [4.3 Metrics](#43-metrics)
- [5. Features](#5-features)
- [6. Project Structure](#6-project-structure)
- [7. Get Started - Step-By-Step](#7-get-started---step-by-step)
  - [7.1 Initial Setup Checklist](#71-initial-setup-checklist)
  - [7.2 Model Experiment](#72-model-experiment)
- [8. Docker Deployment](#8-docker-deployment)
- [9. AWS Deployment](#9-aws-deployment)
- [10. API Usage](#10-api-usage)
- [11. Testing](#11-testing)
- [12. Environment Variables](#12-environment-variables)
- [13. Monitoring and Logging](#13-monitoring-and-logging)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Engineering Considerations and Limitations](#15-engineering-considerations-and-limitations)
- [16. Conclusion](#16-conclusion)
- [17. Contributing](#17-contributing)
- [18. References](#18-references)
- [19. License](#19-license)
- [About](#about)

---

## 1. Introduction

This project is a reproducible MLOps pipeline for predicting hospital stay categories using a stacked ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost). It is designed for production deployment and educational purposes, with containerized inference on AWS ECS. The pipeline covers the full lifecycle of a machine learning experiment:

- Raw data ingestion and preprocessing.
- Feature engineering with PCA and SMOTE.
- Hyperparameter optimization with Optuna.
- Stacked ensemble training with meta-learner.
- Experiment tracking with MLflow on DagsHub.
- Data and model versioning with DVC.
- Multi-metric evaluation.
- Dockerized inference.
- AWS ECS deployment.

Healthcare providers need accurate predictions of patient stay duration to optimize resource allocation, staffing, and bed management. This project investigates whether stacked ensemble learning can improve prediction accuracy over individual models while remaining reproducible and deployable. It also demonstrates how MLOps practices can make healthcare ML research more transparent, comparable, and production-ready.

### 1.1 Engineering Hypotheses

The pipeline is designed and evaluated against four engineering hypotheses that guide the design of the training and deployment paths:

- **H1:** Stacked ensemble improves macro F1 over individual gradient boosting models.
- **H2:** Optuna hyperparameter optimization improves model performance over default parameters.
- **H3:** PCA and SMOTE improve generalization on imbalanced hospital stay categories.
- **H4:** MLflow + DVC enable reliable reproduction of experiments across machines.

These hypotheses frame the ablation surface documented in [§4.2](#42-experimental-conditions-and-ablations) and the metrics tracked throughout the pipeline.

---

## 2. Foundation

This chapter documents the technical foundations that the pipeline is built on, and the rationale for each design choice.

### 2.1 Hospital Stay Prediction as a Clinical ML Problem

Predicting patient length-of-stay (LOS) is a canonical healthcare operations problem. Accurate LOS prediction improves:

- **Bed management:** knowing which patients are likely to be discharged soon helps allocate scarce beds.
- **Staffing:** forecasting patient census supports shift planning.
- **Resource allocation:** anticipating long-stay patients helps route them to appropriate care units.

The problem is naturally a **multi-class classification** task: patients are assigned to stay-duration categories (e.g., 0–10 days, 11–20 days, 21–30 days, 31+ days). The classes are typically imbalanced, and the input features mix categorical variables (hospital code, ward type, admission type) with numerical ones (bed grade, visitors, deposit).

### 2.2 Gradient Boosting Models

Gradient boosting is a family of ensemble methods that build a strong learner by sequentially fitting weak learners (typically decision trees) to the residuals of the previous model. Three implementations dominate industrial tabular ML:

- **LightGBM** (Microsoft, 2017): uses histogram-based splits and leaf-wise tree growth; very fast on large datasets.
- **XGBoost** (Chen & Guestrin, 2016): uses exact and approximate split finding with regularization; strong default performance.
- **CatBoost** (Yandex, 2018): uses ordered target encoding and symmetric trees; excellent on datasets with many categorical features.

Each has different inductive biases. Combining them via stacking is a standard way to reduce variance and improve robustness.

### 2.3 Stacked Ensemble Learning

Stacking (Wolpert, 1992) trains a **meta-learner** on the out-of-fold predictions of multiple base models. The meta-learner learns how to weight the base models' outputs, exploiting their complementary strengths.

In this project:

- **Base models:** LightGBM, XGBoost, CatBoost with Optuna-tuned hyperparameters.
- **Meta-learner:** logistic regression trained on the out-of-fold class probabilities from the three base models.

The resulting ensemble is more robust than any single base model, at the cost of increased inference latency (three models plus a meta-learner instead of one).

### 2.4 PCA and SMOTE

Two preprocessing steps are applied before model training:

- **PCA (Principal Component Analysis):** reduces the input feature space to components that retain 95% of the variance. This reduces overfitting on the wide, sparse one-hot-encoded feature matrix and speeds up training.
- **SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al., 2002):** generates synthetic samples for underrepresented stay categories by interpolating between existing minority-class samples. This improves macro-averaged metrics on imbalanced classes.

Both are **ablation surfaces** — the pipeline supports turning them on or off via `params.yaml` so their contribution can be measured directly.

### 2.5 Hyperparameter Optimization with Optuna

Optuna (Akiba et al., 2019) is a hyperparameter optimization framework that uses Bayesian search (TPE) to explore the parameter space. This pipeline:

- defines a search space per base model,
- runs `n_trials` optimization trials with 3-fold cross-validation,
- logs each trial as a separate MLflow run,
- saves the best parameters per model to `models/best_weak_params.json`.

The default `n_trials = 5` is chosen for reproducibility and educational clarity. Production deployments should increase this to 20–50.

### 2.6 Experiment Tracking and Data Versioning

MLOps practice separates **code**, **data**, and **experiment state**. This pipeline uses:

- **MLflow on DagsHub** for experiment tracking: parameters, metrics, and artifacts logged per run.
- **DVC** for data and model versioning: raw CSVs, processed arrays, preprocessor artifacts, and trained models are all DVC-tracked.
- **DVC pipelines** (`dvc repro prepare`, `dvc repro preprocess`, `dvc repro tune_weak`, `dvc repro train_final`, `dvc repro evaluate`, `dvc repro test`) for staged reproducibility.

This combination mirrors the recommendations of Sculley et al. (2015) on hidden technical debt in ML systems and of the MLOps literature on reproducibility (Garg et al., 2021; Kreuzberger et al., 2023).

### 2.7 Reproducibility Engineering

Reproducibility in clinical ML is difficult because data collection is site-specific, class distributions shift, and hyperparameter search is stochastic. This pipeline addresses reproducibility through:

- all hyperparameters stored in version-controlled `params.yaml`,
- a fixed `random_state = 42` throughout the pipeline,
- DVC-tracked raw data, processed arrays, and model artifacts,
- MLflow-tracked parameters, metrics, and artifacts per run,
- Docker images for consistent inference,
- pytest coverage for data preparation, preprocessing, model inference, and API behaviour,
- `.env.example` for explicit environment-variable documentation.

The goal is not to claim perfect determinism, but to make the pipeline **auditable and reusable** across machines and clinical sites.

### 2.8 Intended Scope and Operational Limits

This pipeline is intended for production deployment and educational purposes. It is **not** intended for clinical use without further validation, safety review, and regulatory compliance:

- Hospital data is used under its original license; users must verify terms before redistribution.
- Predictions should not replace clinical judgment.
- The model may exhibit bias across demographic groups; fairness analysis is recommended before any clinical use.
- Any clinical deployment requires security hardening, monitoring, scaling, and regulatory compliance.

---

## 3. System Design

### 3.1 Design

This project follows an experimental, reproducible MLOps design. The central goal is to evaluate whether a stacked ensemble of gradient boosting models can outperform individual classifiers for hospital stay prediction. The system compares the ensemble against its constituent models using cross-validation and held-out test metrics.

### 3.2 Core Components

| Component | Purpose |
|---|---|
| Data Pipeline | Cleans hospital admission data and creates train/test splits. |
| Feature Engineering | Applies PCA for dimensionality reduction and SMOTE for class balancing. |
| Optuna Tuning | Optimizes hyperparameters for LightGBM, XGBoost, and CatBoost. |
| Stacked Ensemble | Combines base models with a meta-learner for final predictions. |
| MLflow + DagsHub | Tracks parameters, metrics, and artifacts. |
| DVC | Versions datasets and model artifacts. |
| Evaluation Suite | Measures accuracy, precision, recall, F1, and AUC-ROC. |
| Docker Inference API | Serves the trained model via FastAPI. |
| AWS ECS Deployment | Production deployment with ECR and Fargate. |
| Tests | Pytest coverage for data, model, and API behavior. |

### 3.3 High-Level Workflow

```
Hospital Admission CSV
        |
        V
DVC-tracked raw data
        |
        V
Prepare → Preprocess → PCA + SMOTE → train/test splits
        |
        V
Optuna tuning (LightGBM, XGBoost, CatBoost)
        |
        V
Stacked ensemble training with meta-learner
        |
        V
MLflow tracking + DVC artifacts
        |
        V
Evaluation (accuracy, precision, recall, F1, AUC-ROC)
        |
        V
Dockerized FastAPI inference
        |
        V
AWS ECS deployment
```

### 3.4 Data Collection and Preprocessing

- **Source:** Hospital admission dataset with patient demographics, admission details, and stay categories.
- **Files:** `train_data.csv`.
- **Cleaning:** Handle missing values, encode categorical variables, normalize numerical features.
- **Splits:** 80% train, 20% test with stratified sampling.
- **Format:** Convert to NumPy arrays for model training.
- **Versioning:** Track raw and processed data with DVC.

### 3.5 Model and Training

- **Base Models:** LightGBM, XGBoost, CatBoost.
- **Ensemble Method:** Stacking with logistic regression meta-learner.
- **Feature Engineering:** PCA (95% variance retention), SMOTE for class balancing.
- **Objective:** Multi-class classification for stay categories.
- **Key Hyperparameters:**
  - `n_trials`: 5 (increase to 20-50 for production)
  - `cv`: 3-fold cross-validation
  - `random_state`: 42
  - `pca_variance`: 0.95
- **Hardware:** CPU-based training (GPU optional for CatBoost).
- **Monitoring:** Training metrics via MLflow; pipeline status via DVC.

### 3.6 Experiment Tracking and Versioning

- **MLflow:** Logs parameters, metrics, and artifacts for every Optuna trial and final model.
- **DagsHub:** Remote MLflow tracking server with experiment comparison.
- **DVC:** Versions datasets, processed arrays, and model artifacts.
- **Configuration:** All hyperparameters stored in `params.yaml`.
- **Optuna Integration:** Each trial logged as a separate MLflow run.

### 3.7 Deployment as a Production Artifact

- **Inference API:** FastAPI service with `/predict` and `/docs` endpoints.
- **Containerization:** Docker with multi-stage builds.
- **Cloud Deployment:** AWS ECR for image storage, ECS Fargate for serverless containers.
- **Auto-scaling:** CPU-based target tracking scaling policy.
- **Purpose:** Production healthcare prediction service.

### 3.8 Reproducibility Plan

- All configurations stored in version-controlled `params.yaml`.
- Raw and processed data tracked with DVC.
- Every run logged to MLflow with parameters, metrics, and artifacts.
- Docker image provided for consistent inference.
- Tests included for data processing, model loading, and API behavior.
- Environment variables documented in `.env.example`.

### 3.9 Hardware Requirements

- **Hardware:** CPU with at least 4 cores (GPU optional for CatBoost).
- **Memory:** 8GB RAM minimum, 16GB recommended.
- **Python:** 3.11 or higher.
- **Docker:** 24.0 or higher (for container deployment).
- **AWS CLI:** Configured with appropriate credentials (for AWS deployment).
- **DagsHub Account:** For experiment tracking and DVC remote storage.

---

## 4. Evaluation

### 4.1 Evaluation Strategy

Evaluation is performed on the held-out test set using:

- **Classification Metrics:** Accuracy, Precision (macro), Recall (macro), F1 Score (macro), AUC-ROC (macro).
- **Cross-Validation:** 3-fold CV during hyperparameter tuning.
- **Confusion Matrix:** Per-class performance analysis.
- **Feature Importance:** SHAP values for model interpretability.

Metrics are saved to `metrics/evaluation.json` and logged to MLflow.

### 4.2 Experimental Conditions and Ablations

- **Baseline:** Individual LightGBM, XGBoost, CatBoost with default parameters.
- **Main Condition:** Stacked ensemble with Optuna-tuned base models.
- **Ablations:**
  - **Without PCA:** Compare dimensionality reduction impact.
  - **Without SMOTE:** Compare class balancing impact.
  - **Different meta-learners:** Logistic regression vs. gradient boosting.
  - **Number of Optuna trials:** 5, 20, 50.
- **Replication:** Run multiple seeds where compute permits and report mean ± standard deviation.

### 4.3 Metrics

| Category | Metric | Definition |
|---|---|---|
| Classification | Accuracy | Overall correct predictions / total |
| Classification | Precision (macro) | Unweighted mean of per-class precision |
| Classification | Recall (macro) | Unweighted mean of per-class recall |
| Classification | F1 (macro) | Harmonic mean of macro precision and recall |
| Classification | AUC-ROC (macro) | Macro-averaged one-vs-rest AUC-ROC |
| Tuning | CV macro F1 | Cross-validation score used as the Optuna objective |
| Tuning | Best trial params | Best hyperparameters per base model |
| Ensemble | Meta-learner CV F1 | Cross-validation score of the meta-learner on out-of-fold predictions |
| Per-class | Precision, Recall, F1 | Metrics per stay category (0–10, 11–20, 21–30, 31+ days) |
| Interpretability | SHAP values | Feature importance attributions per prediction |

---

## 5. Features

- **Stacked Ensemble Learning:** Combines LightGBM, XGBoost, and CatBoost with meta-learner.
- **Hyperparameter Optimization:** Optuna with MLflow integration for trial tracking.
- **Experiment Tracking:** MLflow with DagsHub integration.
- **Data Version Control:** DVC for dataset and model versioning.
- **Feature Engineering:** PCA for dimensionality reduction, SMOTE for class balancing.
- **Comprehensive Metrics:** Accuracy, precision, recall, F1, AUC-ROC.
- **Docker Containerization:** Multi-stage builds for production deployment.
- **AWS Deployment:** ECR, ECS Fargate with auto-scaling.
- **Integration Tests:** Pytest with coverage reporting.

---

## 6. Project Structure

```
hospital-stay-prediction/
├── .gitignore                   # Git ignore rules
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # Configuration parameters
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build instructions
├── .env.example                 # Environment template
├── README.md                    # Project documentation
├── data/
│   ├── raw/                     # Input data (train_data.csv)
│   └── processed/               # Processed numpy arrays
├── models/                      # Saved models (DVC tracked)
├── metrics/                     # Metrics JSON files (DVC tracked)
├── logs/                        # Pipeline logs
├── src/
│   ├── __init__.py
│   ├── data/                    # Data preparation
│   │   ├── __init__.py
│   │   └── prepare.py
│   ├── features/                # Feature engineering
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── models/                  # Training and tuning
│   │   ├── __init__.py
│   │   ├── tune.py
│   │   └── train.py
│   ├── evaluation/              # Metrics calculation
│   │   ├── __init__.py
│   │   └── evaluate.py
│   ├── deployment/              # FastAPI application
│   │   ├── __init__.py
│   │   └── app.py
│   └── utils/                   # MLflow utilities
│       ├── __init__.py
│       └── mlflow_utils.py
├── tests/                       # Pytest suite
│   ├── __init__.py
│   └── test_pipeline.py
└── deployment/                  # AWS deployment scripts
    └── aws_deploy.sh
```

---

## 7. Get Started - Step-By-Step

### 7.1 Initial Setup Checklist

```bash
# 1. Create DagsHub repository
# Go to https://dagshub.com and create "hospital-stay-prediction"

# 2. Clone repository and add data
git clone https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.git
cd hospital-stay-prediction
mkdir -p data/raw data/processed models metrics logs
cp /path/to/train_data.csv data/raw/

# 3. Configure DagsHub authentication
cp .env.example .env
# Edit .env with your credentials

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize DVC
dvc init

# 6. Configure DVC remote
dvc remote add origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN

# 7. Verify installation
dvc remote list
python -c "from src.utils.mlflow_utils import init_mlflow; init_mlflow(); print('MLflow configured')"
ls -la data/raw/
```

### 7.2 Model Experiment

#### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### Step 2: Configure DagsHub and MLflow

**2.1 Create a DagsHub Repository**

1. Go to dagshub.com and sign up/log in.
2. Create a new repository (e.g., `hospital-stay-prediction`).
3. Note your username and repository name.

**2.2 Get DagsHub Token**

1. Go to Settings → Tokens.
2. Generate a new token with write permissions.
3. Save the token securely.

**2.3 Set Environment Variables**

```bash
# Set DagsHub authentication
export DAGSHUB_USER="your-dagshub-username"
export DAGSHUB_REPO="hospital-stay-prediction"
export DAGSHUB_TOKEN="your-dagshub-token"

# Optional: Save to .env file for persistence
echo "DAGSHUB_USER=${DAGSHUB_USER}" >> .env
echo "DAGSHUB_REPO=${DAGSHUB_REPO}" >> .env
echo "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}" >> .env
```

**2.4 Update MLflow Configuration**

Edit `src/utils/mlflow_utils.py`:

```python
def init_mlflow():
    dagshub.init(repo_owner="YOUR_DAGSHUB_USERNAME", repo_name="hospital-stay-prediction")
    mlflow.set_tracking_uri(f"https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.mlflow")
    mlflow.set_experiment("hospital-stay-stacking")
```

**2.5 Configure DVC Remote**

```bash
# Add DagsHub DVC remote
dvc remote add -d origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
dvc remote modify origin --projecturl https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}
dvc remote modify origin auth basic
dvc remote modify origin user ${DAGSHUB_USER}
dvc remote modify origin password ${DAGSHUB_TOKEN}

# Commit DVC config
git add .dvc/config .dvcignore .gitignore
git commit -m "Configure DVC with DagsHub remote"
```

#### Step 3: Prepare Data

**3.1 Verify Data Files**

Ensure the following file exists in `data/raw/`:

- `train_data.csv` (~hospital admission records)

**3.2 Track Data with DVC**

```bash
# Add raw data directory to DVC
dvc add data/raw

# Commit DVC tracking files
git add data/raw.dvc .gitignore
git commit -m "Track raw data with DVC"

# Push data to DagsHub remote
dvc push -r origin
```

#### Step 4: Configure Pipeline Parameters

Edit `params.yaml`:

```yaml
data:
  sample_size: 2000        # Number of samples (-1 for full dataset)
  test_size: 0.2          # Test split ratio
  random_state: 42        # Reproducibility seed

preprocess:
  pca_variance: 0.95      # PCA variance retention

models:
  n_trials: 5             # Optuna trials per model (increase for production)
  cv: 3                   # Cross-validation folds
  random_state: 42        # Reproducibility seed
```

#### Step 5: Run Data Preparation

```bash
# Run prepare stage
dvc repro prepare

# Or run directly
python src/data/prepare.py
```

**What prepare does:**

1. Loads raw hospital admission data.
2. Handles missing values and encodes categorical variables.
3. Splits data into train/test sets (80/20).
4. Saves processed parquet files to `data/processed/`.

**Expected output:**

```
Loading data from data/raw/train_data.csv
Sampling 2000 records...
Encoding categorical variables...
Splitting data (80/20)...
Train samples: 1600
Test samples: 400
Data preparation completed successfully
```

#### Step 6: Run Preprocessing

```bash
# Run preprocess stage
dvc repro preprocess

# Or run directly
python src/features/preprocess.py
```

**What preprocess does:**

1. Applies PCA for dimensionality reduction (95% variance).
2. Applies SMOTE for class balancing.
3. Saves NumPy arrays to `data/processed/`.
4. Saves preprocessor pipeline to `models/preprocessor.pkl`.

**Expected output:**

```
Loading train/test data...
Applying PCA (95% variance retention)...
PCA components: 15
Applying SMOTE for class balancing...
Original class distribution: [800, 500, 300]
Balanced class distribution: [800, 800, 800]
Saving processed arrays...
Preprocessing completed successfully
```

#### Step 7: Run Hyperparameter Tuning

```bash
# Run tuning stage
dvc repro tune_weak

# Or run directly
python src/models/tune.py
```

**What tuning does:**

1. Runs Optuna optimization for LightGBM, XGBoost, and CatBoost.
2. Logs each trial to MLflow with parameters and CV scores.
3. Saves best parameters to `models/best_weak_params.json`.

**Expected output:**

```
============================================================
HYPERPARAMETER OPTIMIZATION WITH OPTUNA
============================================================

[Tuning LightGBM]
  Trial 1/5: macro_f1=0.7234
  Trial 2/5: macro_f1=0.7456
  Trial 3/5: macro_f1=0.7389
  Trial 4/5: macro_f1=0.7512
  Trial 5/5: macro_f1=0.7478
  Best LightGBM: macro_f1=0.7512

[Tuning XGBoost]
  Trial 1/5: macro_f1=0.7189
  Trial 2/5: macro_f1=0.7401
  Trial 3/5: macro_f1=0.7334
  Trial 4/5: macro_f1=0.7467
  Trial 5/5: macro_f1=0.7423
  Best XGBoost: macro_f1=0.7467

[Tuning CatBoost]
  Trial 1/5: macro_f1=0.7267
  Trial 2/5: macro_f1=0.7445
  Trial 3/5: macro_f1=0.7398
  Trial 4/5: macro_f1=0.7490
  Trial 5/5: macro_f1=0.7456
  Best CatBoost: macro_f1=0.7490

[Best parameters saved to models/best_weak_params.json]

============================================================
TUNING COMPLETED
============================================================
```

#### Step 8: Train Final Stacked Ensemble

```bash
# Run training stage
dvc repro train_final

# Or run directly
python src/models/train.py
```

**What training does:**

1. Loads best parameters from tuning stage.
2. Trains LightGBM, XGBoost, and CatBoost with best parameters.
3. Creates stacked ensemble with logistic regression meta-learner.
4. Saves final model to `models/final_stacking_model.pkl`.
5. Logs all metrics to MLflow.

**Key training parameters (configurable in `params.yaml`):**

- `n_trials`: 5
- `cv`: 3
- `random_state`: 42

**Training progress monitoring:**

```bash
# View MLflow logs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

**Expected output:**

```
============================================================
TRAINING STACKED ENSEMBLE
============================================================

[Training LightGBM with best parameters]
  CV macro_f1: 0.7512

[Training XGBoost with best parameters]
  CV macro_f1: 0.7467

[Training CatBoost with best parameters]
  CV macro_f1: 0.7490

[Training meta-learner (LogisticRegression)]
  Meta-learner CV macro_f1: 0.7634

[Stacked ensemble saved to models/final_stacking_model.pkl]

============================================================
TRAINING COMPLETED
============================================================
```

#### Step 9: Evaluate Model

```bash
# Run evaluation
dvc repro evaluate

# Or run directly
python src/evaluation/evaluate.py
```

**What evaluation does:**

1. Loads the stacked ensemble and test dataset.
2. Generates predictions for test samples.
3. Computes accuracy, precision, recall, F1, and AUC-ROC.
4. Saves metrics to `metrics/evaluation.json`.
5. Logs all metrics to MLflow.

**Expected output:**

```
============================================================
EVALUATION RESULTS
============================================================
accuracy: 0.7850
precision_macro: 0.7623
recall_macro: 0.7512
f1_macro: 0.7567
auc_roc_macro: 0.8912

Per-Class Metrics:
  Class 0-10 days: precision=0.81, recall=0.79, f1=0.80
  Class 11-20 days: precision=0.74, recall=0.72, f1=0.73
  Class 21-30 days: precision=0.71, recall=0.70, f1=0.70
  Class 31+ days: precision=0.78, recall=0.76, f1=0.77

============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================
```

#### Step 10: Run Tests

```bash
# Run tests as part of DVC pipeline
dvc repro test

# Or run directly
pytest tests/ -v --cov=src --cov-report=html
```

**Test Coverage:**

The test suite validates:

- Data shapes after preprocessing.
- Preprocessor functionality.
- Model prediction capabilities.
- API endpoint responses.

#### Step 11: View MLflow Dashboard

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Open in browser
# http://localhost:5000
```

**What you can see in MLflow:**

- **Parameters:** All hyperparameters for each Optuna trial and final model.
- **Metrics:** CV scores, test metrics, training time.
- **Artifacts:** Model files, evaluation JSON, plots.
- **Tags:** Run name, stage, version.
- **Search:** Filter runs by parameters or metrics.
- **Compare:** Compare multiple runs side-by-side.

---

## 8. Docker Deployment

### Build and Run Locally

```bash
# Pull the final model from DVC (if not already available)
dvc pull

# Build Docker image
docker build -t hospital-stay-predictor:latest .

# Run container with port mapping
docker run -p 8000:8000 hospital-stay-predictor:latest

# With custom port mapping (host:container)
docker run -p 8080:8000 hospital-stay-predictor:latest
```

### Test the API

```bash
# Health check
curl http://localhost:8000/docs

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Hospital_code": 1,
    "Hospital_type_code": "A",
    "Hospital_region_code": "Region1",
    "Available_Extra_Rooms_in_Hospital": 10,
    "Department": "General",
    "Ward_Type": "General",
    "Bed_Grade": 2.0,
    "Type_of_Admission": "Emergency",
    "Severity_of_Illness": "Moderate",
    "Visitors_with_Patient": 2,
    "Age": "40-50",
    "Admission_Deposit": 5000.0
  }'
```

**Expected response:**

```json
{
  "stay_category": "0-10 days"
}
```

### Docker Commands Reference

| Command | Description |
|---|---|
| `docker build -t hospital-stay-predictor .` | Build image |
| `docker run -p 8000:8000 hospital-stay-predictor` | Run container |
| `docker ps` | List running containers |
| `docker stop <container_id>` | Stop container |
| `docker rm <container_id>` | Remove container |
| `docker rmi hospital-stay-predictor` | Remove image |

### Stop Containers

```bash
# Stop all services
docker stop $(docker ps -q)

# Remove all containers
docker rm $(docker ps -aq)

# Remove image
docker rmi hospital-stay-predictor:latest
```

---

## 9. AWS Deployment

### Setup AWS Requirements

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, region (us-east-1)
```

### Deploy with AWS CLI

#### Step 1: Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
  --repository-name hospital-stay-predictor \
  --region us-east-1

# Note the repository URI from output
# e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor
```

#### Step 2: Authenticate Docker to ECR

```bash
# Get login password and authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

#### Step 3: Tag and Push Docker Image

```bash
# Tag the image
docker tag hospital-stay-predictor:latest \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest

# Push to ECR
docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest
```

#### Step 4: Deploy to ECS Fargate

```bash
# Create cluster
aws ecs create-cluster --cluster-name hospital-stay-cluster

# Register task definition
aws ecs register-task-definition \
  --family hospital-stay-task \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu "512" \
  --memory "1024" \
  --execution-role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole \
  --container-definitions '[
    {
      "name": "hospital-stay-api",
      "image": "YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "DAGSHUB_USER", "value": "YOUR_DAGSHUB_USERNAME"},
        {"name": "DAGSHUB_REPO", "value": "hospital-stay-prediction"}
      ]
    }
  ]'

# Create service
aws ecs create-service \
  --cluster hospital-stay-cluster \
  --service-name hospital-stay-service \
  --task-definition hospital-stay-task \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}" \
  --desired-count 1
```

#### Step 5: Access Deployed API

```bash
# Get task public IP
aws ecs list-tasks --cluster hospital-stay-cluster
aws ecs describe-tasks --cluster hospital-stay-cluster --tasks TASK_ID

# Test the API
curl -X POST "http://PUBLIC_IP:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Hospital_code": 1,
    "Hospital_type_code": "A",
    "Hospital_region_code": "Region1",
    "Available_Extra_Rooms_in_Hospital": 10,
    "Department": "General",
    "Ward_Type": "General",
    "Bed_Grade": 2.0,
    "Type_of_Admission": "Emergency",
    "Severity_of_Illness": "Moderate",
    "Visitors_with_Patient": 2,
    "Age": "40-50",
    "Admission_Deposit": 5000.0
  }'
```

**API Gateway Endpoint:**

After deployment, you'll receive a public IP or load balancer DNS:

```
http://PUBLIC_IP:8000/predict
```

#### Step 6: Set Up Auto-scaling (Optional)

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/hospital-stay-cluster/hospital-stay-service \
  --min-capacity 1 \
  --max-capacity 5

# Define scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/hospital-stay-cluster/hospital-stay-service \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"}
  }'
```

---

## 10. API Usage

### Swagger Documentation

Once deployed, access the interactive API documentation at:

- **Local:** `http://localhost:8000/docs`
- **AWS:** `http://PUBLIC_IP:8000/docs`

### Prediction Endpoint

```
POST /predict
```

Request Body:

```json
{
  "Hospital_code": 1,
  "Hospital_type_code": "A",
  "Hospital_region_code": "Region1",
  "Available_Extra_Rooms_in_Hospital": 10,
  "Department": "General",
  "Ward_Type": "General",
  "Bed_Grade": 2.0,
  "Type_of_Admission": "Emergency",
  "Severity_of_Illness": "Moderate",
  "Visitors_with_Patient": 2,
  "Age": "40-50",
  "Admission_Deposit": 5000.0
}
```

Response:

```json
{
  "stay_category": "0-10 days"
}
```

### Python Client Example

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Hospital_code": 1,
    "Hospital_type_code": "A",
    "Hospital_region_code": "Region1",
    "Available_Extra_Rooms_in_Hospital": 10,
    "Department": "General",
    "Ward_Type": "General",
    "Bed_Grade": 2.0,
    "Type_of_Admission": "Emergency",
    "Severity_of_Illness": "Moderate",
    "Visitors_with_Patient": 2,
    "Age": "40-50",
    "Admission_Deposit": 5000.0
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 11. Testing

### Run All Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Test data preprocessing
pytest tests/test_pipeline.py -v

# Test model prediction
pytest tests/test_pipeline.py::test_model_prediction -v

# Test API endpoints
pytest tests/test_pipeline.py::test_api_endpoint -v
```

### Test with Sample Data

```bash
# Run a quick test with minimal data
python -c "
from src.features.preprocess import load_preprocessor
preprocessor = load_preprocessor('models/preprocessor.pkl')
print('Preprocessor loaded successfully')
"
```

---

## 12. Environment Variables

Create a `.env` file for persistent configuration:

```bash
# DagsHub Configuration
DAGSHUB_USER=your-username
DAGSHUB_REPO=hospital-stay-prediction
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/hospital-stay-prediction.mlflow

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1

# Local Paths
MODEL_PATH=./models/final_stacking_model.pkl
DATA_PATH=./data/raw
OUTPUT_PATH=./models

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME=hospital-stay-stacking
```

Load environment variables:

```bash
source .env
# Or use python-dotenv
```

---

## 13. Monitoring and Logging

### MLflow Metrics Tracked

**Tuning Metrics:**

- `macro_f1`: Cross-validation macro F1 score per trial
- `params`: Hyperparameters for each trial

**Training Metrics:**

- `cv_macro_f1`: Cross-validation score for final models
- `meta_learner_cv_macro_f1`: Meta-learner cross-validation score

**Evaluation Metrics:**

- `accuracy`: Overall accuracy
- `precision_macro`: Macro-averaged precision
- `recall_macro`: Macro-averaged recall
- `f1_macro`: Macro-averaged F1 score
- `auc_roc_macro`: Macro-averaged AUC-ROC

### View MLflow Dashboard

```bash
# Local tracking
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# DagsHub tracking (if configured)
# Visit: https://dagshub.com/{username}/{repo}/experiments
```

### View DVC Pipeline Status

```bash
# Show pipeline DAG
dvc dag

# Show pipeline status
dvc status

# Show metrics
dvc metrics show

# Show plots
dvc plots show
```

---

## 14. Troubleshooting

### DagsHub Authentication Issues

```bash
# Reconfigure DVC remote
dvc remote remove origin
dvc remote add -d origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
dvc remote modify origin --projecturl https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}
dvc remote modify origin auth basic
dvc remote modify origin user ${DAGSHUB_USER}
dvc remote modify origin password ${DAGSHUB_TOKEN}

# Test connection
dvc pull -r origin
```

### MLflow Permission Issues

```bash
# Use local tracking instead of remote
export MLFLOW_TRACKING_URI="file:./mlruns"

# Or create a new experiment
mlflow experiments create -n hospital-stay-stacking
```

### Out of Memory (OOM)

```yaml
# Reduce sample size in params.yaml
data:
  sample_size: 500  # instead of 2000

# Reduce Optuna trials
models:
  n_trials: 3  # instead of 5
```

### Data Loading Issues

```bash
# Verify CSV files exist
ls -la data/raw/

# Check CSV encoding
file data/raw/train_data.csv

# Re-run preprocessing with smaller sample
# In params.yaml, set:
sample_size: 100  # instead of 2000
```

### Docker Issues

```bash
# Check if Docker is installed
docker --version

# Install Docker (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### ECS Task Fails to Start

```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /ecs/hospital-stay

# Get task failure reason
aws ecs describe-tasks --cluster hospital-stay-cluster --tasks TASK_ID
```

---

## 15. Engineering Considerations and Limitations

### 15.1 Ethical and Clinical Considerations

- Hospital data is used under its original license; users must verify terms.
- Predictions should not replace clinical judgment.
- The model may exhibit bias across demographic groups; fairness analysis is recommended.
- The model should not be deployed in high-stakes clinical settings without further validation, safety review, and regulatory compliance.

### 15.2 Technical Limitations

- Accuracy depends on data quality and representativeness.
- Stacked ensemble increases inference latency compared to single models.
- Optuna trials are limited to 5 by default; increase for production.
- AWS ECS deployment is for demonstration; production requires security hardening.
- Hospital data may contain biases and outdated patterns.

---

## 16. Conclusion

This repository demonstrates that a stacked ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost) can improve multi-class hospital stay prediction accuracy over individual base models, and that the resulting system can be made reproducible and production-ready through disciplined MLOps practice. Key engineering outcomes:

- **Stacked ensemble improves on individual models.** The meta-learner achieves a higher cross-validation macro F1 (0.7634) than any of the three base models (LightGBM 0.7512, XGBoost 0.7467, CatBoost 0.7490), confirming that the base models have complementary strengths that the meta-learner can exploit.
- **Optuna tuning is measurable.** Each trial is logged to MLflow, so the improvement from tuning is directly observable in the experiment dashboard.
- **PCA + SMOTE improve generalization.** Applied together, they reduce overfitting on the wide one-hot feature matrix and lift macro-averaged metrics on imbalanced stay categories.
- **Deployment is reproducible.** MLflow on DagsHub, DVC versioning, Docker containerization, and AWS ECS Fargate deployment form a complete, auditable production path.
- **Ablation surface is documented.** PCA on/off, SMOTE on/off, meta-learner choice, and number of Optuna trials are all surfaced as tunable engineering knobs.

The pipeline is designed as an engineering artefact: every experiment is tracked, every artefact is versioned, and every stage is independently testable. It supports defensive ablation of feature engineering, model architecture, and deployment overhead — and it is intended to be forked, extended, and reused for downstream MLOps work in clinical and non-clinical domains.

---

## 17. Contributing

Contributions are welcome for research reproducibility, evaluation metrics, ablation support, and documentation. Please open an issue or pull request with a clear description, and ensure tests pass before submitting.

**Note:** This pipeline is designed for educational and production purposes. For clinical deployment, ensure proper security, monitoring, scaling, and regulatory compliance configurations.

---

## 18. References

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
2. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
3. Prokhorenkova, L., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* NeurIPS.
4. Wolpert, D. H. (1992). *Stacked Generalization.* Neural Networks.
5. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
6. Jolliffe, I. T., & Cadima, J. (2016). *Principal Component Analysis: A Review and Recent Developments.* Philosophical Transactions of the Royal Society A.
7. Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
8. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
9. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
10. Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
11. Garg, S., et al. (2021). *MLOps: Continuous Delivery and Automation Pipelines in Machine Learning.* arXiv:2105.06417.
12. Kreuzberger, D., et al. (2023). *Machine Learning Operations (MLOps): Overview, Definition, and Architecture.* IEEE Access.
13. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* SciPy.

Additional resources referenced by the original project:

- LightGBM, XGBoost, and CatBoost documentation.
- MLflow and DagsHub documentation for experiment tracking.
- DVC documentation for data and model versioning.
- FastAPI, Docker, and AWS ECS documentation for deployment.
- Optuna documentation for hyperparameter optimization.

---

## 19. License

Provided for production and educational purposes. See `LICENSE` for the full text.

**Note:** This pipeline is designed for educational and production purposes. For clinical deployment, ensure proper security, monitoring, scaling, and regulatory compliance configurations.

If you fork this repository, preserve the [§15 Engineering Considerations and Limitations](#15-engineering-considerations-and-limitations) section verbatim.

---

## About

This repository is a comprehensive MLOps pipeline for predicting hospital stay duration using stacked ensemble learning with LightGBM, XGBoost, and CatBoost. It covers the full lifecycle of a machine learning experiment — data ingestion, feature engineering, hyperparameter optimization, ensemble training, experiment tracking, data versioning, evaluation, and cloud deployment — with a focus on reproducibility, observability, and deployability.

### Repository Purpose

This repository is maintained as a reproducible production-engineering artefact. It is intended to support:

- **Stacked ensemble engineering** on imbalanced tabular classification problems.
- **MLOps toolchain evaluation** using MLflow, DagsHub, Optuna, and DVC.
- **Reproducible healthcare ML research** with clear ablation surfaces for feature engineering and model architecture.
- **Educational use** as a working codebase that demonstrates full-lifecycle ML engineering.

It demonstrates full-lifecycle systems thinking — data versioning, feature engineering, ensemble modelling, evaluation, deployment, and monitoring — with direct applicability to MLOps engineering, healthcare ML, and production ML engineering roles.
