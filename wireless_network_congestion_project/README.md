# Wireless Network Congestion Project

FastAPI-based web application for IoT network congestion analysis, model training, visualization, and live prediction.

This project reads the IoT traffic dataset, cleans it, engineers network features, creates a congestion label, trains multiple machine learning models, compares their performance, saves the best model, and exposes everything through a browser dashboard and REST API.

## What This Project Does

The application has 4 main parts:

1. Data preparation
   Loads the dataset, fills missing values, removes duplicates, and creates derived network features.

2. Congestion modeling
   Builds a composite congestion target, trains multiple classifiers, compares them, and saves the best one.

3. Visualization
   Generates plots for correlation, congestion distribution, congestion trends, and bandwidth vs packet rate.

4. Prediction
   Accepts manual network inputs or samples the current system, then predicts whether congestion is `Low`, `Medium`, or `High`.

## Main Features

- Dataset understanding summary
- Data cleaning for missing values and duplicate rows
- Feature engineering:
  - `packet_rate`
  - `flow_duration`
  - `bytes_per_second`
  - `network_load`
  - `congestion_score`
  - `congestion_level`
- EDA plots:
  - Correlation heatmap
  - Bandwidth vs packet rate
  - Congestion distribution
  - Congestion trend
- Model training:
  - Logistic Regression
  - Random Forest
  - Decision Tree
- Model evaluation:
  - Accuracy
  - Accuracy (%)
  - Confusion matrix
  - Classification report
  - Train accuracy
  - Cross-validation mean/std
  - Generalization gap
- Best model export with `joblib`
- Browser dashboard
- Prediction API
- Current system data sampling button for Linux and Windows

## Project Structure

```text
wireless_network_congestion_project/
├── __init__.py
├── main.py
├── dataset_loader.py
├── data_analysis.py
├── train_model.py
├── predict.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
├── artifacts/
│   ├── best_model.joblib
│   └── model_metrics.json
└── plots/
    ├── correlation_heatmap.png
    ├── bandwidth_vs_packet_rate.png
    ├── congestion_distribution.png
    └── congestion_trend.png
```

## Dataset

Expected dataset location:

```text
.kagglehub-cache/datasets/programmer3/iot-network-traffic-dataset/versions/1/Iot_Network_data.csv
```

The loader will:

- use the explicit dataset path if provided
- otherwise use the default cached Kaggle dataset path
- otherwise search the repo for `Iot_Network_data.csv`

## How Congestion Is Created

The project does not use a raw congestion label from the dataset. Instead, it builds one from multiple network stress signals.

The target is based on a weighted `congestion_score` using:

- `packet_rate`
- `latency`
- `packet_loss_rate`
- `jitter`
- bandwidth pressure
- `energy_usage`
- protocol risk
- device-type risk

Then the score is split into 3 classes:

- `Low`
- `Medium`
- `High`

This is important because an earlier version used a label too close to `network_load`, which caused leakage and unrealistically perfect scores. The current version reduces that problem and reports train/test/CV metrics so overfitting is easier to detect.

## Model Inputs

The final models are trained on these features:

- `packet_rate`
- `allocated_bandwidth`
- `latency`
- `packet_loss_rate`

These are the same fields used by the prediction form.

## Requirements

- Python 3.11
- `uv` recommended for environment setup

Python dependencies:

- `fastapi`
- `uvicorn`
- `jinja2`
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `joblib`
- `kagglehub[pandas-datasets]`

## Setup

Run all commands from the repository root:

```text
/home/sachinn-p/Codes/IOT DATA Prediction
```

### Option 1: Setup with `uv` on Linux/macOS

```bash
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install -r wireless_network_congestion_project/requirements.txt
```

### Option 2: Setup with `uv` on Windows PowerShell

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1
uv pip install -r wireless_network_congestion_project\requirements.txt
```

### Option 3: Setup with plain `pip`

Linux/macOS:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r wireless_network_congestion_project/requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r wireless_network_congestion_project\requirements.txt
```

### If NumPy needs reinstalling

If you hit a NumPy binary/version mismatch:

Linux/macOS:

```bash
uv pip install --reinstall numpy==2.3.3
```

Windows PowerShell:

```powershell
uv pip install --reinstall numpy==2.3.3
```

## Running the App

### Recommended run command

After activating the virtual environment:

```bash
python -m uvicorn wireless_network_congestion_project.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

### Development mode with reload

Use this only while editing UI/backend code:

```bash
python -m uvicorn wireless_network_congestion_project.main:app --host 0.0.0.0 --port 8000 --reload
```

Note:

- `--reload` is convenient for development
- but during `/api/train`, file writes under `plots/` and `artifacts/` can trigger reloads
- for stable training runs, use the non-reload command

### Alternate run mode

You can also use:

```bash
UVICORN_RELOAD=1 python wireless_network_congestion_project/main.py
```

On Windows PowerShell:

```powershell
$env:UVICORN_RELOAD="1"
python wireless_network_congestion_project/main.py
```

## How To Use the Application

### 1. Open the dashboard

When the page loads, it fetches:

- dataset summary
- best saved model name
- saved metrics
- generated plot URLs

### 2. Run the full pipeline

Click `Run Full Pipeline`.

This performs:

1. dataset loading
2. cleaning
3. feature engineering
4. plot generation
5. model training
6. model evaluation
7. best model selection
8. artifact saving

### 3. Predict congestion manually

Enter:

- Packet Rate
- Bandwidth
- Latency
- Packet Loss

Then click `Predict Congestion`.

### 4. Predict using current machine data

Click `Use Current System Data`.

The app samples the host machine for about 1 second and fills:

- Packet Rate
- Bandwidth
- Latency
- Packet Loss

Platform behavior:

- Linux: reads network counters from `/proc/net/dev`
- Windows: tries PowerShell adapter statistics first, then falls back to `netstat -e`

Notes:

- if the machine is idle, values may be `0.00`
- latency can also be `0.00` if probing is blocked or unavailable

## API Endpoints

### `GET /`

Returns the HTML dashboard.

### `GET /api/summary`

Returns:

- generation timestamp
- dataset info
- best model name
- saved metrics
- plot URLs

Example response shape:

```json
{
  "generated_at": "2026-03-31T12:00:00+00:00",
  "dataset": {
    "dataset_path": "...",
    "shape": { "rows": 100000, "columns": 11 },
    "columns": ["packet_size", "transmission_time"],
    "dtypes": { "packet_size": "int64" },
    "important_features": {}
  },
  "best_model": "Logistic Regression",
  "model_metrics": {},
  "plots": {
    "correlation_heatmap": "/plots/correlation_heatmap.png"
  }
}
```

### `GET /api/system-data`

Returns current system telemetry used to fill the prediction form.

Example response:

```json
{
  "packet_rate": 12.48,
  "bandwidth": 0.31,
  "latency": 18.72,
  "packet_loss": 0.00,
  "sample_seconds": 1.0,
  "sampled_at": "2026-03-31T12:00:00+00:00",
  "measurement_note": "Packet rate and bandwidth come from a short local network sample. Packet loss is estimated from interface drop/error counters."
}
```

### `POST /api/train`

Runs the entire ML pipeline and returns:

- best model name
- model artifact path
- cleaning report
- normalized shape
- congestion distribution
- model metrics
- plot URLs

### `POST /api/predict`

Predicts congestion from manual or system-filled inputs.

Request body:

```json
{
  "packet_rate": 0.5,
  "bandwidth": 40.0,
  "latency": 30.0,
  "packet_loss": 0.1
}
```

Response body:

```json
{
  "congestion_level": "Medium",
  "network_load": 20.0
}
```

## Training Flow Explained

### Data cleaning

The cleaning step:

- fills missing numeric values with the median
- fills missing categorical values with the mode
- removes duplicate rows

### Feature engineering

The app creates:

- `flow_duration`
- `packet_rate`
- `bytes_per_second`
- `network_load`
- `congestion_score`
- `congestion_level`

### Normalization

Numeric columns are normalized for analysis output. The model training path uses the selected training features, and Logistic Regression additionally uses `StandardScaler` in its pipeline.

### Model training

Three models are trained:

- Logistic Regression
- Random Forest
- Decision Tree

Regularization / constraints used:

- Logistic Regression uses scaling and `C=0.5`
- Random Forest uses limited depth and minimum sample thresholds
- Decision Tree uses limited depth and minimum sample thresholds

### Model evaluation

Each model reports:

- `accuracy`
- `train_accuracy`
- `cv_accuracy_mean`
- `cv_accuracy_std`
- `generalization_gap`
- `confusion_matrix`
- `classification_report`

### Best model selection

The best model is selected by:

1. higher cross-validation mean accuracy
2. lower generalization gap when CV scores tie

### Saved artifacts

After training:

- best model bundle is saved to `artifacts/best_model.joblib`
- metrics are saved to `artifacts/model_metrics.json`
- plots are saved to `plots/`

## Frontend Explanation

The dashboard is built with:

- Jinja2 template
- static CSS
- static JavaScript

The frontend:

- loads summary data on page load
- triggers training from a button
- refreshes plots after training
- fills prediction form values
- predicts congestion
- fetches current system telemetry for auto-fill
- renders a model metrics table with:
  - Accuracy
  - Accuracy (%)
  - Confusion Matrix

## Important Files

- [main.py](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/main.py)
  FastAPI app, routes, system-data sampling, and dashboard serving

- [dataset_loader.py](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/dataset_loader.py)
  Dataset discovery and loading

- [data_analysis.py](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/data_analysis.py)
  Cleaning, feature engineering, normalization, and EDA plots

- [train_model.py](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/train_model.py)
  Train/test split, model building, evaluation, selection, and artifact saving

- [predict.py](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/predict.py)
  Single-record prediction helper

- [templates/index.html](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/templates/index.html)
  Dashboard layout

- [static/js/app.js](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/static/js/app.js)
  Frontend behavior

- [static/css/styles.css](/home/sachinn-p/Codes/IOT DATA Prediction/wireless_network_congestion_project/static/css/styles.css)
  Dashboard styling

## Troubleshooting

### `/api/system-data` returns `404 Not Found`

Your server is running an older version of the app.

Fix:

1. stop the current server
2. restart with:

```bash
python -m uvicorn wireless_network_congestion_project.main:app --host 0.0.0.0 --port 8000
```

3. hard refresh the browser

### `Use Current System Data` fills zeros

This can be normal if:

- the machine is idle
- there is no measurable traffic during the sample window
- latency probing is blocked

### Prediction fails with model-not-found

Run the full pipeline first so `artifacts/best_model.joblib` is created.

### Training restarts unexpectedly in development mode

Avoid `--reload` while running `/api/train`.

### Browser still shows old UI

Restart the server and hard refresh the browser. Static files are versioned, but an old process can still serve stale routes/templates.

### Windows system-data collection fails

Make sure at least one of these is available:

- PowerShell with adapter statistics support
- `netstat`

If both are blocked by environment policy, the system-data endpoint may fail.

## Outputs Generated by the App

### Model artifacts

```text
wireless_network_congestion_project/artifacts/best_model.joblib
wireless_network_congestion_project/artifacts/model_metrics.json
```

### Plot outputs

```text
wireless_network_congestion_project/plots/correlation_heatmap.png
wireless_network_congestion_project/plots/bandwidth_vs_packet_rate.png
wireless_network_congestion_project/plots/congestion_distribution.png
wireless_network_congestion_project/plots/congestion_trend.png
```

## Recommended Workflow

1. Create and activate `.venv`
2. Install requirements
3. Start FastAPI
4. Open the dashboard
5. Click `Run Full Pipeline`
6. Review metrics and plots
7. Test manual prediction
8. Test `Use Current System Data`

## Summary

This project is a complete mini-ML system:

- dataset ingestion
- cleaning
- feature engineering
- label creation
- model training
- overfitting-aware evaluation
- visualization
- REST API
- interactive frontend

It is designed for demonstration, analysis, and experimentation with network congestion prediction using IoT traffic data.
