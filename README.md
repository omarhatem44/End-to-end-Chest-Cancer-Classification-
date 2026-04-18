<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=00C9A7&center=true&vCenter=true&width=750&lines=Chest+Cancer+Classification;End-to-End+MLOps+Pipeline;VGG16+%2B+TensorFlow+%2B+Kubernetes" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![VGG16](https://img.shields.io/badge/VGG16-Transfer%20Learning-blueviolet?style=for-the-badge)](https://keras.io/api/applications/vgg/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://minikube.sigs.k8s.io)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECR-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-AWS%20EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](http://13.221.125.15:8080/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **A production-grade MLOps pipeline for medical imaging** — classifying chest CT scans as cancerous or normal using a fine-tuned VGG16 model, with full experiment tracking, DVC-versioned pipeline, MLflow Model Registry, Docker containerization, and Kubernetes orchestration via Minikube.

<br/>

[🏗️ Architecture](#️-system-architecture) · [⚡ Quick Start](#-getting-started) · [📖 Documentation](#-table-of-contents) · [📊 Results](#-results) · [🌐 Live Demo](http://13.221.125.15:8080/)

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
- [Backend API Reference](#-backend-api-reference)
- [Frontend UI](#️-frontend-ui)
- [Containerization](#-containerization-docker)
- [Kubernetes Deployment](#️-kubernetes-deployment-minikube)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Challenges Faced](#️-challenges-faced)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [MLOps Skills Demonstrated](#-mlops-skills-demonstrated)
- [Author](#-author)

---

## 🔍 Overview

**End-to-End Chest Cancer Classification** is a full-stack medical imaging MLOps system. It takes chest CT scan images as input and classifies them as **Adenocarcinoma**, **Large Cell Carcinoma**, **Squamous Cell Carcinoma**, or **Normal** — using a fine-tuned **VGG16** convolutional neural network.

The focus of this project is not only model performance but also **production-level deployment using MLOps practices**: a reproducible DVC pipeline, MLflow experiment tracking with a Model Registry, a hardened Flask inference API, Docker containerization, and Kubernetes orchestration via Minikube.

### ✨ Key Highlights

| Feature | Description |
|---|---|
| 🧠 **VGG16 Transfer Learning** | Fine-tuned deep CNN on chest CT scan imagery for cancer detection |
| 🔬 **Medical Imaging Pipeline** | End-to-end from raw CT data to production-ready inference |
| 📊 **MLflow Model Registry** | Environment-gated model promotion: Staging → Production |
| 🔁 **DVC Pipeline** | Reproducible 4-stage pipeline tracked with `dvc.yaml` |
| 🧪 **Test Suite** | pytest-based unit and integration testing for pipeline reliability |
| 🌐 **Flask REST API** | Hardened inference endpoint with health check and CORS support |
| 🐳 **Dockerized** | Gunicorn-served container pushed to Docker Hub |
| ☸️ **Kubernetes Deployment** | Deployed on Minikube with Deployment + NodePort Service YAMLs |
| ⚙️ **GitHub Actions CI/CD** | Fully automated build → test → push → deploy workflow |
| 🚀 **Live Demo** | [Try it live on AWS EC2 →](http://13.221.125.15:8080/) |

---

## 🏗️ System Architecture

The system is organized into six integrated layers: data versioning, model training, experiment tracking, API serving, containerization, and Kubernetes orchestration.

```
User → Web UI → Flask API → ML Model → Prediction → Response
                          ↓
                     Docker Container
                          ↓
                     Kubernetes (Minikube)
```

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
        FLASK["🌐 Flask REST API\nPOST /predict\nGET /health"]
        MODEL_LOAD["📦 Load Production\nModel from Registry"]
        DOCKER["🐳 Docker Container\nGunicorn WSGI"]

        REGISTRY --> MODEL_LOAD --> FLASK --> DOCKER
    end

    subgraph K8S["☸️  Kubernetes  •  Minikube"]
        DEPLOY_K8S["📄 deployment.yaml\nReplicas + Port 8080"]
        SVC["🌐 service.yaml\nNodePort Exposure"]
        MINIKUBE["⚙️ Minikube Cluster\nLocal Orchestration"]

        DOCKER --> DEPLOY_K8S --> MINIKUBE
        SVC --> MINIKUBE
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

    DATA --> PIPELINE
    MINIKUBE --> CICD

    style DATA fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style PIPELINE fill:#1b263b,stroke:#415a77,color:#fff
    style TRACKING fill:#415a77,stroke:#778da9,color:#fff
    style SERVING fill:#1b263b,stroke:#00c9a7,color:#fff
    style K8S fill:#0d1b2a,stroke:#326CE5,color:#fff
    style CICD fill:#0d1b2a,stroke:#415a77,color:#fff
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
| **API Serving** | Flask + Gunicorn, CORS enabled |
| **Containerization** | Docker (pushed to Docker Hub) |
| **Orchestration** | Kubernetes (Minikube) — Deployment + NodePort Service |
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
├── k8s/
│   ├── deployment.yaml             # Kubernetes Deployment manifest
│   └── service.yaml                # Kubernetes NodePort Service manifest
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
├── model/
│   └── model.h5                    # Trained VGG16 model weights
│
├── research/                       # Jupyter notebooks for EDA & prototyping
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
    A["🖼️ Input\nChest CT Scan\n224 × 224 × 3\nBase64 Encoded"] -->
    B["🔒 VGG16 Base\nPretrained on ImageNet\nFrozen Convolutional Layers\n13 Conv + 3 Pooling"] -->
    C["🔓 Custom Head\nFlatten → Dense(256, ReLU)\nDropout(0.5) → Dense(4, Softmax)"] -->
    D["📊 Output\nAdenocarcinoma\nLarge Cell Carcinoma\nSquamous Cell Carcinoma\nNormal"]

    style A fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style B fill:#415a77,stroke:#778da9,color:#fff
    style C fill:#1b263b,stroke:#00c9a7,color:#fff
    style D fill:#0d1b2a,stroke:#00c9a7,color:#fff
```

### Model Integration

| Property | Detail |
|---|---|
| **Format** | `.h5` (Keras SavedModel) |
| **Loaded via** | `PredictionPipeline(self.filename)` |
| **Input** | Base64-encoded CT scan image |
| **Output** | Classification label |

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

## 🌐 Backend API Reference

**Base URL:** [http://13.221.125.15:8080](http://13.221.125.15:8080/)

The Flask backend has CORS enabled for frontend communication and uses JSON-based messaging throughout.

### `GET /`

Returns the web UI for manual image upload and prediction.

---

### `POST /predict`

Accepts a base64-encoded CT scan image and returns a cancer classification.

**Request:**
```json
{
  "image": "base64_string"
}
```

**Response:**
```json
{
  "prediction": "Cancer / Normal"
}
```

**cURL Example:**
```bash
curl -X POST http://13.221.125.15:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'
```

---

### `GET /health`

Used for monitoring and Kubernetes readiness/liveness probes.

**Response:**
```json
{
  "status": "healthy"
}
```

---

## 🖥️ Frontend UI

A lightweight HTML interface served directly by Flask:

- Upload a chest CT scan image
- Trigger the `/predict` endpoint
- Display the classification result in real time

---

## 🐳 Containerization (Docker)

### Dockerfile Overview

```dockerfile
FROM python:3.10-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
```

The app is served via **Gunicorn** (4 workers) for production-grade WSGI performance.

### Build & Push to Docker Hub

```bash
# Build image locally
docker build -t omarhatemmohamed/chest-cancer-app .

# Push to Docker Hub
docker push omarhatemmohamed/chest-cancer-app
```

> **Final image size:** ≈ 2.6 GB

---

## ☸️ Kubernetes Deployment (Minikube)

### Why Minikube?

Due to AWS EC2 quota limitations (fleet request limits and NodeGroup creation failures), deployment was completed locally using **Minikube** as a production-equivalent Kubernetes environment.

### Manifests

**`deployment.yaml`** — defines the container image, exposed port (8080), and number of replicas.

**`service.yaml`** — exposes the application externally via a `NodePort` service.

### Deployment Steps

```bash
# Start the local cluster
minikube start

# Apply manifests
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Access the Application

```bash
# Auto-open in browser
minikube service <service-name>
```

Or access directly via:

```
http://<minikube-ip>:<nodeport>
```

### Kubernetes Flow

```mermaid
graph LR
    A["🐳 Docker Hub\nomarhatemmohamed/chest-cancer-app"] -->
    B["📄 deployment.yaml\nReplicas + Port 8080"] -->
    C["⚙️ Minikube Cluster\nkubectl apply"] -->
    D["🌐 service.yaml\nNodePort Exposure"] -->
    E["🌍 External Access\nhttp://minikube-ip:nodeport"]

    style A fill:#0d1b2a,stroke:#00c9a7,color:#fff
    style B fill:#0d1b2a,stroke:#326CE5,color:#fff
    style C fill:#1b263b,stroke:#326CE5,color:#fff
    style D fill:#0d1b2a,stroke:#326CE5,color:#fff
    style E fill:#0d1b2a,stroke:#00c9a7,color:#fff
```

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

### Testing & Validation

| Scenario | Status |
|---|---|
| UI image upload | ✅ |
| API JSON response | ✅ |
| Model inference | ✅ |
| Docker container execution | ✅ |
| Kubernetes service exposure | ✅ |

---

## 🚀 Getting Started

### Prerequisites

```
Python 3.10+  |  Docker  |  DVC  |  kubectl  |  Minikube
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

### 6. Deploy on Kubernetes (Minikube)

```bash
minikube start
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
minikube service <service-name>
```

---

### 🌐 Live Demo

The API is deployed and running live on AWS EC2:

👉 **[http://13.221.125.15:8080/](http://13.221.125.15:8080/)**

Upload a chest CT scan image and get a real-time cancer classification — no setup required.

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

## ⚠️ Challenges Faced

### 1. Flask Route Issues
- `/health` endpoint was initially missing → caused 404 errors in Kubernetes readiness probes
- **Fix:** Added dedicated health check endpoint returning `{"status": "healthy"}`

### 2. HTTP Method Errors
- `/predict` requires `POST` → frontend was sending an incorrect request method
- **Fix:** Corrected the frontend request configuration

### 3. Docker Issues
- Docker daemon not running during initial build attempts
- Missing `Dockerfile` discovered mid-deployment
- Large final image size (~2.6 GB) due to TensorFlow dependencies

### 4. AWS Deployment Failure
- EC2 instance quota limits exceeded
- Fleet request limits blocked instance creation
- EKS NodeGroup creation failed entirely
- **Resolution:** Pivoted to Minikube as a production-equivalent local Kubernetes environment

---

## 🧠 Key Learnings

- Difference between local vs cloud production deployment environments
- Docker image size optimization challenges with heavy ML dependencies
- Kubernetes resource management and NodePort service exposure patterns
- Critical importance of health check endpoints in container orchestration
- How to handle cloud infrastructure quota limits gracefully

---

## 🚀 Future Improvements

| Improvement | Description |
|---|---|
| **Multi-stage Docker build** | Reduce image size significantly below ~2.6 GB |
| **Model on S3** | Store `.h5` in S3 instead of baking it into the container |
| **Full CI/CD to K8s** | Extend GitHub Actions to auto-deploy to Kubernetes |
| **AWS EKS / ECS** | Migrate from Minikube to cloud Kubernetes after quota increase |
| **Monitoring** | Add Prometheus + Grafana dashboards for inference metrics |

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
| **Model Serving** | Flask + Gunicorn API with base64 input, CORS, and health endpoint |
| **Containerization** | Docker image built and pushed to Docker Hub (~2.6 GB) |
| **Kubernetes Orchestration** | Minikube Deployment + NodePort Service with readiness probes |
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

*Built end-to-end with production MLOps practices — medical imaging, transfer learning, Docker, Kubernetes, and automated cloud deployment* 🩺🚀

⭐ **Star this repo** if you found it useful!

</div>
