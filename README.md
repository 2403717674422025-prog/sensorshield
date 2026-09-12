# 🛡️ SensorShield

**AI-Powered Industrial Sensor Reliability & Predictive Maintenance Platform**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)

---

## 🔍 Overview

SensorShield is a production-grade industrial IoT monitoring system that uses machine learning to detect sensor anomalies, score reliability, and predict equipment failures in real time.

The system monitors **4 industrial machines** with **23 sensors** and provides:
- Real-time anomaly detection using an LSTM Autoencoder
- Sensor health scoring with explainable diagnostics
- Failure probability prediction using XGBoost + LSTM ensemble
- Signal reconstruction for degraded/missing sensors
- Live WebSocket dashboard with Power BI analytics integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Dashboard                       │
│         (TypeScript + Recharts + WebSocket)              │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP/WebSocket
┌───────────────────────▼─────────────────────────────────┐
│                  FastAPI Backend                         │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  REST API   │  │  WebSocket   │  │  MQTT Sub     │  │
│  │  /api/*     │  │  /ws/sensors │  │  (paho-mqtt)  │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬────────┘  │
│         └────────────────┼──────────────────┘           │
│                          │                              │
│  ┌───────────────────────▼──────────────────────────┐   │
│  │              ML Pipeline                         │   │
│  │  1. Preprocessing  → Normalize window            │   │
│  │  2. LSTM Autoencoder → Anomaly score             │   │
│  │  3. Reliability Engine → Health score (0-100)    │   │
│  │  4. Signal Reconstructor → Repair missing data   │   │
│  │  5. XGBoost + LSTM → Failure probability         │   │
│  │  6. Alert Engine → Auto-generate alerts          │   │
│  └───────────────────────┬──────────────────────────┘   │
└───────────────────────────┼─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                    PostgreSQL 16                         │
│  sensors │ sensor_health │ predictions │ alerts          │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 ML Pipeline

| Stage | Model | Purpose |
|-------|-------|---------|
| Anomaly Detection | LSTM Autoencoder (PyTorch) | Reconstruction error → anomaly score |
| Health Scoring | Rule-based + statistical | Drift, noise, missing data detection |
| Signal Reconstruction | Bi-LSTM + Spatial Regressors | Restore corrupted/missing sensor values |
| Failure Prediction | XGBoost + LSTM with MC-Dropout | Failure probability + uncertainty |
| Trust Scoring | Derived from health score | TRUSTED / CAUTION / UNTRUSTED |

---

## 🖥️ Dashboard Features

- **Real-time WebSocket streaming** — live health updates every 3 seconds
- **Machine panels** — 4 collapsible panels with per-machine color coding
- **Sensor cards** — animated SVG icons, sparkline charts, health bars
- **Alert center** — auto-generated alerts with impact & recommended actions
- **Failure prediction** — live failure probability with trust status
- **Power BI integration** — historical analytics dashboard

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 20+
- PostgreSQL 16
- Docker (optional)

### Option 1: Docker (Recommended)

```bash
docker-compose up --build
```

Open http://localhost in your browser.

### Option 2: Manual Setup

**1. Clone the repo**
```bash
git clone https://github.com/2403717674422025-prog/sensorshield
cd sensorshield
```

**2. Set up the database**
```bash
# Create PostgreSQL user and database
psql -U postgres
CREATE USER sensorshield WITH PASSWORD 'sensorshield';
CREATE DATABASE sensorshield OWNER sensorshield;
\q
```

**3. Backend setup**
```bash
cd backend
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**4. Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

**5. Run migrations**
```bash
cd backend
alembic upgrade head
```

**6. Train ML models**
```bash
cd ..
python scripts/run_preprocessing.py
python scripts/train_phase5_lstm_autoencoder.py
python scripts/train_prediction_models.py
python scripts/train_phase7_signal_reconstruction.py
```

**7. Seed the database**
```bash
python scripts/seed_database.py
```

**8. Start the backend**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**9. Start the frontend**
```bash
cd frontend
npm install
npm run dev
```

**10. Run the simulator**
```bash
python scripts/simulate_live_readings.py
```

Open http://localhost:5173

---

## 📁 Project Structure

```
SensorShield/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # FastAPI endpoints
│   │   ├── core/                # Config, DB, logging
│   │   ├── ml/
│   │   │   ├── anomaly_detection/   # LSTM Autoencoder
│   │   │   ├── prediction/          # XGBoost + LSTM predictor
│   │   │   ├── preprocessing/       # Data pipeline
│   │   │   ├── reconstruction/      # Signal reconstructor
│   │   │   └── reliability/         # Health scoring engine
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic schemas
│   │   └── services/            # Pipeline, alerts, WebSocket, email
│   └── alembic/                 # DB migrations
├── frontend/
│   └── src/
│       ├── components/          # React components
│       └── hooks/               # WebSocket, polling hooks
├── scripts/                     # Training & simulation scripts
├── models/                      # Trained model checkpoints
├── data/                        # Raw & processed datasets
└── docker-compose.yml
```

---

## 🔥 Databricks Integration

SensorShield uses Apache Spark on Databricks for large-scale sensor data processing and ML experiment tracking.

### What's in the Databricks notebook (`databricks/SensorShield_Analytics.py`):

| Step | Description |
|------|-------------|
| Data Ingestion | Load 52,000 sensor readings from Unity Catalog Volume using PySpark |
| Spark SQL Analytics | Statistical analysis — mean, stddev, min, max per sensor |
| Anomaly Detection | Rule-based fault labelling using Spark SQL CASE expressions |
| Delta Lake | 3 Delta tables: `sensorshield_readings`, `sensorshield_anomalies`, `sensorshield_sensor_summary` |
| MLflow Tracking | Experiment tracking with parameters and metrics logged to `/SensorShield_Analytics` |

### Run the notebook:
1. Upload `data/raw/sensor_data.csv` to Databricks Unity Catalog Volume
2. Open `databricks/SensorShield_Analytics.py` in Databricks
3. Run all cells
4. View MLflow experiments at **Experiments → SensorShield_Analytics**

### Delta tables (connect from Power BI):
- `sensorshield_readings` — raw sensor time series
- `sensorshield_anomalies` — fault-labelled dataset
- `sensorshield_sensor_summary` — per-sensor statistics for dashboards

---

## 📊 Power BI Dashboard

Connect Power BI Desktop to PostgreSQL:
- Server: `localhost`
- Database: `sensorshield`
- Username: `sensorshield`
- Password: `sensorshield`

Tables: `sensor_health`, `sensors`, `alerts`, `predictions`, `machines`

---

## 📧 Email Alerts

Configure in `.env`:
```env
EMAIL_ENABLED=true
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_TO=recipient@gmail.com
```

---

## 🛠️ Tech Stack

**Backend:** Python, FastAPI, SQLAlchemy, Alembic, asyncpg, paho-mqtt  
**ML:** PyTorch, XGBoost, scikit-learn, NumPy, Pandas  
**Big Data:** Apache Spark (PySpark), Databricks, Delta Lake, MLflow  
**Frontend:** React 18, TypeScript, Recharts, WebSocket  
**Database:** PostgreSQL 16  
**Infrastructure:** Docker, Nginx, Uvicorn  
**Analytics:** Power BI, Databricks SQL  

---

## 📝 License

MIT License — feel free to use this project for learning and portfolio purposes.
