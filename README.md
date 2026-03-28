<div align="center">

<img src="https://img.shields.io/badge/TensorFlow-2.12.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/VGG16-Transfer%20Learning-0072C6?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-2.2.2-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/AWS-EC2%20Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
<img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white"/>

<br/><br/>

# 🫁 End-to-End Chest Cancer Classification

### A Production-Grade MLOps Pipeline for CT Scan Classification

**Live Demo → [http://ec2-3-219-222-157.compute-1.amazonaws.com:8080](http://ec2-3-219-222-157.compute-1.amazonaws.com:8080)**

---

</div>

## 📌 Overview

This project is a **fully productionized, end-to-end MLOps system** for classifying chest CT scan images to detect cancer. It is not just a model — it is a complete machine learning platform, covering every stage from raw data ingestion to live prediction through a REST API deployed on AWS EC2.

The system is designed with **MLOps best practices** at its core: reproducible pipelines managed by DVC, automated CI/CD via GitHub Actions, containerized deployment with Docker, and full experiment tracking via MLflow integrated with DagsHub — making it suitable for real-world clinical and production environments.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                               │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │     Data     │───▶│  Prepare Base    │───▶│    Training     │   │
│  │  Ingestion   │    │  Model (VGG16)   │    │  (Fine-Tuning)  │   │
│  └──────────────┘    └──────────────────┘    └────────┬────────┘   │
│                                                        │            │
│                              ┌─────────────────────────▼──────┐    │
│                              │       Model Evaluation          │    │
│                              │  (MLflow + DagsHub Tracking)    │    │
│                              └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Deployment Layer                              │
│                                                                      │
│   Flask REST API  ──▶  Docker Container  ──▶  AWS EC2 (Port 8080)   │
│                                ▲                                     │
│                     GitHub Actions CI/CD                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ MLOps Pipeline Stages

### Stage 1 — Data Ingestion
Automatically downloads and organizes the **Chest CT Scan dataset** into structured artifacts.

- **Input:** `config/config.yaml`
- **Output:** `artifacts/data_ingestion/Chest-CT-Scan-data/`
- **Versioned by:** DVC (full reproducibility guaranteed)

---

### Stage 2 — Prepare Base Model
Loads and configures **VGG16** (pre-trained on ImageNet) as the backbone, with custom classification head for binary cancer detection.

- **Architecture:** VGG16 (Transfer Learning, `include_top=False`)
- **Input size:** `224 × 224 × 3`
- **Classes:** 2 (Normal / Adenocarcinoma)
- **Output:** `artifacts/prepare_base_model/`

---

### Stage 3 — Model Training
Fine-tunes the model on the CT scan data with configurable augmentation and hyperparameters — all tracked as DVC params.

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 224 × 224 × 3 |
| Batch Size | 16 |
| Epochs | 2 |
| Learning Rate | 0.02 |
| Augmentation | ✅ Enabled |
| Pre-trained Weights | ImageNet |

- **Output:** `artifacts/training/model.h5`

---

### Stage 4 — Model Evaluation
Evaluates the trained model and logs all metrics to **MLflow on DagsHub** for full experiment traceability.

- **Metrics output:** `scores.json` (DVC-tracked, cache disabled for live comparison)
- **Tracking dashboard:** [DagsHub MLflow](https://dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow/#/experiments)

---

## 🔄 DVC DAG (Directed Acyclic Graph)

```
+----------------+   +--------------------+
| data_ingestion |   | prepare_base_model |
+----------------+** +--------------------+
          *    ******       *
           *         ****  *
            **           **
            +----------+
            | training |
            +----------+
                  *
                  *
           +------------+
           | evaluation |
           +------------+
```

Run the full pipeline with a single command:
```bash
dvc repro
```

---

## 🚀 REST API Endpoints

The Flask application exposes three endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI for image upload and prediction |
| `GET/POST` | `/train` | Triggers full retraining pipeline |
| `POST` | `/predict` | Accepts base64-encoded image, returns classification result |

**Prediction Request Example:**
```json
POST /predict
{
  "image": "<base64_encoded_ct_scan_image>"
}
```

**Response Example:**
```json
[{"image": "Normal"}]
// or
[{"image": "Adenocarcinoma Cancer"}]
```

---

## 📊 Experiment Tracking

All experiments are logged automatically to **MLflow hosted on DagsHub**:

- **DagsHub Repo:** [omarhatem44/End-to-end-Chest-Cancer-Classification](https://dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow/#/experiments)
- **MLflow Experiment Name:** `Chest Cancer Pipeline`
- **Tracked Metrics:** Loss, Accuracy per run
- **Tracked Params:** All hyperparameters from `params.yaml`
- **Artifacts:** Trained model versions

---

## 🐳 Containerization & Deployment

### Docker
```dockerfile
FROM python:3.10-slim-bookworm
WORKDIR /app
# Install dependencies, copy source, run API
CMD ["python3", "app.py"]   # Serves on port 8080
```

Build and run locally:
```bash
docker build -t chest-cancer-classifier .
docker run -p 8080:8080 chest-cancer-classifier
```

### CI/CD — GitHub Actions
Every push to `main` automatically:
1. Builds the Docker image
2. Pushes to **AWS ECR**
3. Pulls and redeploys on **AWS EC2**

Zero-downtime deployment — the live endpoint is always up-to-date.

---

## 📁 Project Structure

```
End-to-end-Chest-Cancer-Classification/
│
├── .github/workflows/          # CI/CD GitHub Actions pipeline
├── .dvc/                       # DVC configuration
│
├── src/cnnClassifier/
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_trainer_model.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py
│   ├── components/             # Core ML logic components
│   ├── config/                 # Configuration manager
│   ├── entity/                 # Data classes / config schemas
│   └── utils/                  # Helper utilities
│
├── config/
│   └── config.yaml             # Paths & pipeline configuration
├── research/                   # Experimental Jupyter notebooks
├── model/                      # Saved model artifacts
├── templates/
│   └── index.html              # Web UI
│
├── app.py                      # Flask REST API
├── main.py                     # Full pipeline runner (MLflow + DagsHub)
├── dvc.yaml                    # DVC pipeline stages
├── params.yaml                 # Hyperparameters (DVC-tracked)
├── scores.json                 # Evaluation metrics output
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | TensorFlow 2.12.0, Keras, VGG16 |
| **Data Versioning** | DVC |
| **Experiment Tracking** | MLflow 2.2.2, DagsHub |
| **API** | Flask, Flask-CORS |
| **Containerization** | Docker |
| **Cloud** | AWS EC2, AWS ECR |
| **CI/CD** | GitHub Actions |
| **Config Management** | YAML, python-box |
| **Data Utilities** | gdown, NumPy, Pandas, Matplotlib |

---

## ⚡ Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/omarhatem44/End-to-end-Chest-Cancer-Classification-.git
cd End-to-end-Chest-Cancer-Classification-
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Run all 4 stages (data ingestion → evaluation)
python main.py

# Or use DVC for cached, reproducible execution
dvc repro
```

### 3. Start the API Server

```bash
python app.py
# → Serving on http://localhost:8080
```

### 4. Retrain via API

```bash
curl -X POST http://localhost:8080/train
```

---

## 🌐 Live Deployment

The application is live and running on **AWS EC2**:

**→ [http://ec2-3-219-222-157.compute-1.amazonaws.com:8080](http://ec2-3-219-222-157.compute-1.amazonaws.com:8080)**

Upload a chest CT scan image through the web interface to get an instant classification result.

---

## 📈 What Makes This Production-Ready

- ✅ **Reproducible pipelines** — Every stage is versioned and cached via DVC; results are identical across environments
- ✅ **Full experiment traceability** — Every training run is logged to MLflow with params, metrics, and model artifacts
- ✅ **Containerized** — Docker ensures consistent runtime across development, staging, and production
- ✅ **Automated deployment** — GitHub Actions pushes and redeploys on every merge to main
- ✅ **YAML-driven configuration** — No hardcoded values; everything is configurable via `config.yaml` and `params.yaml`
- ✅ **Modular codebase** — Clear separation between pipeline stages, components, and API layer
- ✅ **REST API** — Model inference is accessible programmatically, ready for integration into any frontend or clinical system

---

## 👤 Author

**Omar Hatem**
ML/AI Engineer | MLOps Practitioner

[![GitHub](https://img.shields.io/badge/GitHub-omarhatem44-181717?style=flat&logo=github)](https://github.com/omarhatem44)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-945DD6?style=flat&logo=dvc)](https://dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow/#/experiments)

---

<div align="center">

**⭐ If this project helped you or impressed you, give it a star!**

</div>
