# MLOps Iris Classification Project 🌸

This project demonstrates a complete MLOps lifecycle for the Iris dataset, featuring automated pipelines, versioning, and containerized deployment.

## 📁 Project Structure

```text
mlops-iris-project/
├── .dvc/                    # DVC (Data Version Control) configuration
├── .gitlab-ci.yml           # GitLab CI/CD Pipeline configuration
├── configs/                 # Hyperparameters and model configurations
├── data/                    # Data storage (versioned by DVC)
│   ├── raw/                 # Original, immutable data (e.g., iris.csv)
│   └── processed/           # Cleaned/transformed data for training
├── docker/                  # Dockerfiles and deployment configurations
├── docs/                    # Documentation and screenshots
├── mlartifacts/             # MLflow artifacts (models, plots)
├── mlruns/                  # MLflow experiment tracking data
├── models/                  # Versioned model artifacts (.pkl files)
├── monitoring/              # Grafana dashboards for performance tracking
├── reports/                 # Generated reports (evaluation, analysis)
├── scripts/                 # Utility scripts (setup, smoke tests, demos)
├── src/                     # Source Code
│   ├── data/                # Data loading and preprocessing logic
│   ├── models/              # Model architecture and training logic
│   ├── optimization/        # Hyperparameter tuning (Optuna)
│   ├── pipelines/           # ZenML training pipelines orchestration
│   └── serving/             # FastAPI implementation for model serving
├── tests/                   # Unit, integration, and performance tests
├── Dockerfile               # Production container definition
├── docker-compose.yml       # Orchestration for local multi-container setup
├── requirements.txt         # Project dependencies
└── setup_project.sh         # Script to initialize the project structure
```

## � Data Management

### Raw Data (`data/raw/`)
The `data/raw/` directory contains the **immutable source of truth** for the project.
- **Source**: `iris.csv` (Standard Fisher's Iris dataset).
- **Format**: CSV with 150 samples and 5 columns (sepal length, sepal width, petal length, petal width, target/species).
- **Rule**: Never modify files in this directory. Any cleaning or transformation must result in new files in `data/processed/`.

### Data Versioning (DVC)
All data files are tracked using **DVC** to avoid bloating the Git repository with large binary files.
- The `.dvc` files in the `data/` directory track the versions of the actual datasets.
- Use `dvc pull` to retrieve the data after cloning the repository.

## �🛠 Prerequisites

- Python 3.9+
- Docker & Docker Compose
- DVC
- ZenML
- MLflow

## 🚀 Execution Guide

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Post-setup
bash setup_project.sh

### 2. Data Preparation & DVC
```bash
# Download Iris data from sklearn
python src/data/load_data.py


# تثبيت DVC
pip install dvc
# Initialize DVC
dvc init
dvc remote add -d mystorage C:\Users\user\dvc-storage
mkdir C:\Users\user\dvc-storage

# Track data with DVC
dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc data/raw/.gitignore
git commit -m "Add iris dataset with DVC tracking"
dvc push
```
```
الخطوة 2.5: التحقق من DVC
powershell# التحقق من حالة DVC
dvc status

# عرض معلومات DVC
dvc remote list 

### 2. Training Pipeline (ZenML)
The project uses ZenML to orchestrate the ML pipeline.
```bash
# Initialize ZenML
zenml init

# Run the training pipeline
python src/pipelines/training_pipeline.py --test-size 0.2 --C 1.0
```

### 3. Experiment Tracking (MLflow)
Track runs, metrics, and parameters locally or on a server.
```bash
mlflow ui
```

### 4. Data Versioning (DVC)
```bash
dvc pull  # Download data
dvc push  # Upload changes
```

### 5. Section 3.9 : Déploiement (Serving) 🚀
This project implements a professional serving layer as per academic requirements:

- **Stable Inference API**: Built with **FastAPI**, providing independent endpoints for:
  - `/predict`: Model inference.
  - `/health`: Service monitoring.
- **Docker Compose Orchestration**: Manages the full stack locally:
  - `api-v1`: Stable baseline service.
  - `api-v2` / `optuna_best`: Optimized services.
  - `mlflow`: Tracking backend.
- **v1 → v2 + Rollback Simulation**: 
  - Verified via the `demo_deployment_v1_v2_rollback.py` script.
  - **Proof**: Terminal output logs success for upgrade and emergency rollback to v1.

```bash
# Run the Rollback simulation
python demo_deployment_v1_v2_rollback.py
```

### 6. Hyperparameter Optimization (Optuna)
Fine-tune model parameters using Optuna.
```bash
python src/optimization/optuna_tuning.py
```

### 7. Monitoring (Grafana)
The project includes predefined dashboards in `monitoring/grafana-dashboards/`.
- Import these dashboards into your Grafana instance to monitor model health and performance metrics.

## 🔄 CI/CD Automation
The project supports dual CI/CD platforms for maximum flexibility:
1. **GitHub Actions**: Workflows are located in `.github/workflows/`. This enables the **"Actions"** tab on GitHub.
2. **GitLab CI**: Configuration is in `.gitlab-ci.yml`.

### Pipeline Stages:
1. **Test**: Linting (flake8) and Unit Tests (pytest).
2. **Build**: Automated Docker image building.
3. **Smoke Test**: 
    - **Continuous Training (CT)**: Automated quick training with a subset of data (30 samples, 10 iterations).
    - **API Health**: Verification of endpoints.

---
Developed as part of the MLOps training program.
