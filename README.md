<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=00C9A7&center=true&vCenter=true&width=750&lines=Chest+Cancer+Classification;End-to-End+MLOps+Pipeline;VGG16+%2B+TensorFlow+%2B+AWS" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![VGG16](https://img.shields.io/badge/VGG16-Transfer%20Learning-blueviolet?style=for-the-badge)](https://keras.io/api/applications/vgg/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECR-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **A production-grade MLOps pipeline for medical imaging** — classifying chest CT scans as cancerous or normal using a fine-tuned VGG16 model, with full experiment tracking, DVC-versioned pipeline, MLflow Model Registry, and automated AWS deployment via GitHub Actions.

<br/>

[🏗️ Architecture](#️-system-architecture) · [⚡ Quick Start](#-getting-started) · [📖 Documentation](#-table-of-contents) · [📊 Results](#-results)

---

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#️-system-architecture)
- [ML Pipeline](#-ml-pipeline-dvc)
- [CI/CD Pipeline](#️-cicd-pipeline)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [MLflow Model Registry](#-mlflow-model-registry)
- [API Reference](#-api-reference)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [MLOps Skills Demonstrated](#-mlops-skills-demonstrated)
- [Author](#-author)

---

## 🔍 Overview

**End-to-End Chest Cancer Classification** is a full-stack medical imaging MLOps system. It takes chest CT scan images as input and classifies them as **Adenocarcinoma**, **Large Cell Carcinoma**, **Squamous Cell Carcinoma**, or **Normal** — using a fine-tuned **VGG16** convolutional neural network.

The project is built with production engineering at its core: a reproducible DVC pipeline, MLflow experiment tracking with a Model Registry, a hardened Flask inference API, Docker containerization, and automated cloud deployment to AWS — all orchestrated via GitHub Actions.

### ✨ Key Highlights

| Feature | Description |
|---|---|
| 🧠 **VGG16 Transfer Learning** | Fine-tuned deep CNN on chest CT scan imagery for cancer detection |
| 🔬 **Medical Imaging Pipeline** | End-to-end from raw CT data to production-ready inference |
| 📊 **MLflow Model Registry** | Environment-gated model promotion: Staging → Production |
| 🔁 **DVC Pipeline** | Reproducible 4-stage pipeline tracked with `dvc.yaml` |
| 🧪 **Test Suite** | pytest-based unit and integration testing for pipeline reliability |
| 🌐 **Flask REST API** | Hardened inference endpoint with image upload support |
| 🐳 **Dockerized** | Consistent dev-to-prod containerization |
| ☁️ **AWS Deployment** | Auto-deployed to EC2 via ECR on every push to `main` |
| ⚙️ **GitHub Actions CI/CD** | Fully automated build → test → push → deploy workflow |

---

## 🏗️ System Architecture

The system is organized into five integrated layers: data versioning, model training, experiment tracking with registry, API serving, and automated cloud deployment.

```mermaid
flowchart TB
    subgraph DATA["📦  Data Layer"]
        RAW["🗂️ Raw CT Scan Data\nChest Cancer Dataset"]
        DVC_STORE["🔄 DVC Remote\nData Versioning"]
        RAW --> DVC_STORE
    end

    subgraph PIPELINE["🔁  ML Pipeline  •  DVC Orchestrated"]
        direction LR
        INGEST["📥 Data\nIngestion"]
        BASE["🏗️ Prepare\nBase Model\nVGG16 Frozen"]
        TRAIN["🏋️ Model\nTraining\nFine-tuning"]
        EVAL["📊 Model\nEvaluation\nMetrics + Scores"]

        INGEST --> BASE --> TRAIN --> EVAL
    end

    subgraph TRACKING["📈  Experiment Tracking  •  MLflow + DagsHub"]
        MLFLOW["MLflow\nRuns + Metrics"]
        REGISTRY["📋 Model Registry\nStaging → Production"]
        DAGSHUB["DagsHub\nRemote Backend"]
        PARAMS["params.yaml\nHyperparameters"]

        PARAMS --> TRAIN
        TRAIN --> MLFLOW
        MLFLOW --> REGISTRY
        MLFLOW <--> DAGSHUB
    end

    subgraph SERVING["⚡  Serving Layer"]
        FLASK["🌐 Flask REST API\nPOST /predict"]
        MODEL_LOAD["📦 Load Production\nModel from Registry"]
        DOCKER["🐳 Docker Container"]

        REGISTRY --> MODEL_LOAD --> FLASK --> DOCKER
    end

    subgraph CICD["⚙️  CI/CD  •  GitHub Actions"]
        direction LR
        PUSH["git push\nmain"]
        TEST["✅ pytest\nTest Suite"]
        BUILD["🔨 Docker\nBuild"]
        ECR["📤 Push to\nAWS ECR"]
        DEPLOY["🚀 Deploy to\nAWS EC2"]

        PUSH --> TEST --> BUILD --> ECR --> DEPLOY
    end

    subgraph CLOUD["☁️  Cloud Infrastructure  •  AWS"]
        EC2["🖥️ EC2\nRuntime Host"]
        ECREG["🗄️ ECR\nImage Registry"]
        DEPLOY --> EC2
        ECR --> ECREG --> EC2
    end

    DATA --> PIPELINE
    DOCKER --> EC2

    style DATA fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style PIPELINE fill:#1b263b,stroke:#415a77,color:#fff
    style TRACKING fill:#415a77,stroke:#778da9,color:#fff
    style SERVING fill:#1b263b,stroke:#00c9a7,color:#fff
    style CICD fill:#0d1b2a,stroke:#415a77,color:#fff
    style CLOUD fill:#1b263b,stroke:#00c9a7,color:#fff
```

---

## 🔄 ML Pipeline (DVC)

The full pipeline is defined in `dvc.yaml` with four sequential stages. DVC caches intermediate outputs and only re-runs stages whose inputs have changed — enabling fast, reproducible iteration.

```mermaid
graph LR
    A["📥 data_ingestion\nDownload & store\nchest CT scan dataset"] -->
    B["🏗️ prepare_base_model\nLoad VGG16\nFreeze base layers\nAttach custom head"] -->
    C["🏋️ training\nFine-tune on CT data\nApply augmentation\nTrack with MLflow"] -->
    D["📊 evaluation\nCompute accuracy & loss\nLog to MLflow Registry\nSave scores.json"]

    style A fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style B fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style C fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style D fill:#0d1b2a,stroke:#00c9a7,color:#fff
```

```bash
# Reproduce the full pipeline (only changed stages re-run)
dvc repro

# View the DAG
dvc dag

# Run an experiment with different hyperparameters
dvc exp run --set-param training.EPOCHS=20 --set-param training.LEARNING_RATE=0.0001
dvc exp show

# Sync artifacts with DVC remote
dvc push   # Upload to remote storage
dvc pull   # Download tracked artifacts
```

---

## ⚙️ CI/CD Pipeline

Every push to `main` triggers the full deployment pipeline automatically — zero manual intervention.

```mermaid
sequenceDiagram
    participant DEV as 👨‍💻 Developer
    participant GH as 🐙 GitHub
    participant GA as ⚙️ GitHub Actions
    participant ECR as 🗄️ AWS ECR
    participant EC2 as 🖥️ AWS EC2

    DEV->>GH: git push main
    GH->>GA: Trigger workflow
    GA->>GA: ✅ Run pytest test suite
    GA->>GA: 🔨 docker build -t chest-cancer-api .
    GA->>ECR: 📤 Push tagged Docker image
    GA->>EC2: 🔐 SSH into instance
    EC2->>ECR: 📥 docker pull latest
    EC2->>EC2: 🚀 docker run -p 8080:8080
    EC2-->>DEV: ✅ Live on AWS
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow / Keras, VGG16 (ImageNet pretrained) |
| **Transfer Learning** | Fine-tuned VGG16 with custom classification head |
| **Experiment Tracking** | MLflow, DagsHub |
| **Model Registry** | MLflow Model Registry (Staging → Production) |
| **Data & Pipeline Versioning** | DVC (`dvc.yaml`, `dvc.lock`) |
| **Configuration Management** | `config/config.yaml`, `params.yaml` |
| **Testing** | pytest (unit + integration test suite) |
| **API Serving** | Flask, Jinja2 Templates |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS EC2 (compute), AWS ECR (image registry) |
| **Language** | Python 3.10+ |

</div>

---

## 📁 Project Structure

```
End-to-end-Chest-Cancer-Classification/
│
├── .github/
│   └── workflows/                  # GitHub Actions CI/CD pipeline
│
├── .dvc/                           # DVC configuration & cache
├── dvc.yaml                        # Pipeline stage definitions
├── params.yaml                     # Model hyperparameters
│
├── config/
│   └── config.yaml                 # Path & artifact configuration
│
├── src/cnnClassifier/
│   ├── components/                 # Pipeline stage implementations
│   │   ├── data_ingestion.py       # Dataset download & extraction
│   │   ├── prepare_base_model.py   # VGG16 model construction
│   │   ├── model_trainer.py        # Training loop + augmentation
│   │   └── model_evaluation.py     # Metrics logging + MLflow registry
│   │
│   ├── pipeline/                   # Stage orchestration scripts
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   └── stage_04_model_evaluation.py
│   │
│   ├── entity/                     # Dataclass configs for each stage
│   ├── config/                     # ConfigurationManager
│   └── utils/                      # Shared utilities
│
├── model/                          # Saved model artifacts
│   └── model.h5                    # Trained VGG16 model weights
│
├── research/                       # Jupyter notebooks for EDA & prototyping
│
├── templates/                      # Jinja2 HTML templates for web UI
│
├── app.py                          # Flask REST API entry point
├── main.py                         # Full pipeline runner
├── scores.json                     # Latest evaluation metrics
├── Dockerfile
├── requirements.txt
├── setup.py
└── template.py                     # Project scaffolding script
```

---

## 🧠 Model Details

### Architecture: Fine-Tuned VGG16

```mermaid
graph TB
    A["🖼️ Input\nChest CT Scan\n224 × 224 × 3"] -->
    B["🔒 VGG16 Base\nPretrained on ImageNet\nFrozen Convolutional Layers\n13 Conv + 3 Pooling"] -->
    C["🔓 Custom Head\nFlatten → Dense(256, ReLU)\nDropout(0.5) → Dense(4, Softmax)"] -->
    D["📊 Output\nAdenocarcinoma\nLarge Cell Carcinoma\nSquamous Cell Carcinoma\nNormal"]

    style A fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style B fill:#415a77,stroke:#778da9,color:#fff
    style C fill:#1b263b,stroke:#00c9a7,color:#fff
    style D fill:#0d1b2a,stroke:#00c9a7,color:#fff
```

### Training Configuration (`params.yaml`)

```yaml
training:
  EPOCHS: 10
  BATCH_SIZE: 16
  IS_AUGMENTATION: True
  IMAGE_SIZE: [224, 224, 3]
  LEARNING_RATE: 0.01

prepare_base_model:
  IMAGE_SIZE: [224, 224, 3]
  INCLUDE_TOP: False
  WEIGHTS: imagenet
  CLASSES: 4
```

---

## 📋 MLflow Model Registry

Models are automatically promoted through environments based on evaluation thresholds. The evaluation stage logs the trained model to MLflow and registers it in the Model Registry.

```mermaid
stateDiagram-v2
    [*] --> Training : dvc repro
    Training --> Evaluation : Model trained
    Evaluation --> Staging : Log to MLflow\nRegister version
    Staging --> Production : Passes threshold\nManual / auto promotion
    Production --> Serving : Flask API loads\nproduction model
    Staging --> Archived : Below threshold
```

**DagsHub Tracking:** [View Experiments →](https://dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow/#/experiments)

---

## 🌐 API Reference

**Base URL:** `http://<your-ec2-host>:8080`

### `POST /predict`

Upload a chest CT scan image and receive a cancer classification.

**Request:** `multipart/form-data`

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@chest_scan.jpg"
```

**Response:**
```json
{
  "prediction": "Adenocarcinoma",
  "confidence": 0.91,
  "all_scores": {
    "Adenocarcinoma": 0.91,
    "Large Cell Carcinoma": 0.04,
    "Squamous Cell Carcinoma": 0.03,
    "Normal": 0.02
  }
}
```

### `GET /`

Returns the web UI for manual image upload and prediction.

### `GET /train`

Triggers a full pipeline re-run (`dvc repro`) on the server.

---

## 📊 Results

<div align="center">

| Metric | Score |
|---|---|
| **Accuracy** | *See `scores.json` / MLflow run* |
| **Loss** | *See `scores.json` / MLflow run* |
| **Val Accuracy** | *See MLflow experiment dashboard* |
| **Val Loss** | *See MLflow experiment dashboard* |

</div>

> 📈 Full experiment history, metric curves, and model version comparisons are tracked in **MLflow on DagsHub**:
> [dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow](https://dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow/#/experiments)

---

## 🚀 Getting Started

### Prerequisites

```
Python 3.10+  |  Docker  |  DVC  |  AWS CLI
```

```bash
pip install dvc mlflow tensorflow
```

### 1. Clone the Repository

```bash
git clone https://github.com/omarhatem44/End-to-end-Chest-Cancer-Classification-.git
cd End-to-end-Chest-Cancer-Classification-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Data & Model Artifacts

```bash
dvc pull
```

### 4. Reproduce the ML Pipeline

```bash
dvc repro
```

### 5. Run Locally with Docker

```bash
docker build -t chest-cancer-api .
docker run -p 8080:8080 chest-cancer-api
```

### 6. Access the Web UI

Open your browser at `http://localhost:8080` and upload a chest CT scan image.

---

## 🔧 Development Workflow

Follow this workflow when modifying any pipeline stage:

```
1. Update config/config.yaml       → Add paths or artifact locations
2. Update params.yaml              → Adjust hyperparameters
3. Update entity/                  → Define or update dataclasses
4. Update config/configuration.py  → Update ConfigurationManager
5. Update components/              → Implement stage logic
6. Update pipeline/                → Wire stage into pipeline
7. Update main.py                  → Register stage in full runner
8. Update dvc.yaml                 → Define stage deps/outputs
9. Run: dvc repro                  → Execute updated pipeline
10. Run: pytest                    → Verify test suite passes
```

---

## 🧠 MLOps Skills Demonstrated

<div align="center">

| MLOps Pillar | Implementation |
|---|---|
| **Transfer Learning** | VGG16 pretrained on ImageNet, fine-tuned for 4-class medical imaging |
| **Data Versioning** | DVC tracks raw CT data and processed artifacts |
| **Pipeline Reproducibility** | `dvc repro` re-runs only changed stages from `dvc.yaml` |
| **Experiment Tracking** | MLflow logs all runs: loss, accuracy, hyperparameters |
| **Model Registry** | MLflow Model Registry with Staging → Production promotion |
| **Configuration Management** | Centralized `config.yaml` + `params.yaml` with entity dataclasses |
| **Test Suite** | pytest covering pipeline components and API endpoints |
| **Model Serving** | Flask API with image upload, prediction, and web UI |
| **Containerization** | Docker for consistent, reproducible deployment |
| **CI/CD Automation** | GitHub Actions: test → build → push ECR → deploy EC2 |
| **Cloud Deployment** | Live inference on AWS EC2 with image from AWS ECR |

</div>

---

## 👤 Author

<div align="center">

**Omar Hatem**

🎓 Computer Science Student — Modern Academy for Computer Science, Cairo, Egypt
💼 ML Engineer · MLOps Enthusiast · Medical AI Builder

[![GitHub](https://img.shields.io/badge/GitHub-omarhatem44-181717?style=for-the-badge&logo=github)](https://github.com/omarhatem44)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/omar-hatem-44)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-F5C518?style=for-the-badge)](https://dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow)

</div>

---

<div align="center">

*Built end-to-end with production MLOps practices — medical imaging, transfer learning, and automated cloud deployment* 🩺🚀

⭐ **Star this repo** if you found it useful!

</div>
