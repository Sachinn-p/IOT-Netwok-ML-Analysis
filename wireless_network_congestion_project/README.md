# Wireless Network Congestion Project

FastAPI-based web application for IoT network congestion analysis and prediction, with an Oracle-style frontend (HTML/CSS/JS).

## Features

- Dataset understanding (shape, columns, dtypes, important network features)
- Data cleaning (missing values + duplicate handling)
- Feature engineering:
  - `packet_rate`
  - `flow_duration`
  - `bytes_per_second`
  - `network_load`
  - target label: `congestion_level` (`Low`, `Medium`, `High`)
- EDA visualizations:
  - Correlation heatmap
  - Bandwidth vs packet rate
  - Congestion distribution
  - Congestion trend
- Model training and evaluation:
  - Logistic Regression
  - Random Forest
  - Decision Tree
- Best model selection and saving
- Prediction API + browser dashboard

## Project Structure

```text
wireless_network_congestion_project/
├── main.py
├── dataset_loader.py
├── data_analysis.py
├── train_model.py
├── predict.py
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── artifacts/
│   ├── best_model.joblib
│   └── model_metrics.json
└── plots/
    ├── correlation_heatmap.png
    ├── bandwidth_vs_packet_rate.png
    ├── congestion_distribution.png
    └── congestion_trend.png
```

## Requirements

- Python 3.11
- `uv`
- Dataset file already downloaded in:
  - `.kagglehub-cache/datasets/programmer3/iot-network-traffic-dataset/versions/1/Iot_Network_data.csv`

## Setup

From repository root (`/home/sachinn-p/Codes/IOT DATA Prediction`):

```bash
uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python \
  pandas numpy scikit-learn seaborn matplotlib joblib fastapi uvicorn jinja2
```

If you see a NumPy import error about an "older version of numpy", reinstall with:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python --reinstall numpy==2.3.3
```

## Run Web App

```bash
.venv/bin/python -m uvicorn wireless_network_congestion_project.main:app --host 0.0.0.0 --port 8000
```

Open in browser:

```text
http://localhost:8000
```

Development mode (auto-reload):

```bash
UVICORN_RELOAD=1 .venv/bin/python wireless_network_congestion_project/main.py
```

Note: training writes files under `plots/` and `artifacts/`. If the server reloads during `/api/train`, browsers can show `NetworkError when attempting to fetch resource`.

## API Endpoints

### `GET /api/summary`

Returns dataset info, best model name, saved model metrics, and plot URLs.

### `POST /api/train`

Runs full ML pipeline:

1. load dataset
2. clean data
3. engineer features
4. generate EDA plots
5. train/evaluate/select model
6. save artifacts

Response includes `best_model`, `cleaning_report`, `congestion_distribution`, and per-model metrics.

### `POST /api/predict`

Predict congestion level.

Request body:

```json
{
  "packet_rate": 0.5,
  "bandwidth": 40.0,
  "latency": 30.0,
  "packet_loss": 0.1
}
```

Response:

```json
{
  "congestion_level": "Medium",
  "network_load": 20.0
}
```

## Frontend Flow

- Dashboard loads `/api/summary`
- **Run Full Pipeline** button calls `/api/train`
- Prediction form calls `/api/predict`
- Metrics table and plot images update automatically

## Notes

- First training run may take 1-2 minutes.
- If `best_model.joblib` does not exist, run `/api/train` before prediction.
- Generated outputs are stored in:
  - `wireless_network_congestion_project/artifacts`
  - `wireless_network_congestion_project/plots`
