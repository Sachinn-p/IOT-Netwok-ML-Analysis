# IoT Data Prediction

This project predicts wireless network congestion from IoT traffic data using a FastAPI web application and a small machine learning pipeline.

It includes:

- Dataset loading from KaggleHub
- Data cleaning and feature engineering
- Exploratory plots for network behavior
- Model training and best-model selection
- A browser UI and prediction API

## About

The application analyzes IoT network traffic data and classifies congestion into three levels:

- `Low`
- `Medium`
- `High`

The training pipeline creates derived features such as:

- `packet_rate`
- `flow_duration`
- `bytes_per_second`
- `network_load`

The app then trains multiple classifiers, compares their accuracy, saves the best model, and uses it for live predictions.

## Project Structure

```text
.
├── load_iot_dataset.py
├── wireless_network_congestion_project/
│   ├── main.py
│   ├── dataset_loader.py
│   ├── data_analysis.py
│   ├── train_model.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── templates/
│   ├── static/
│   ├── plots/
│   └── artifacts/
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r wireless_network_congestion_project/requirements.txt
```

### 3. Download or locate the dataset

The app expects the IoT dataset CSV file named `Iot_Network_data.csv`.

You can download it with:

```bash
python load_iot_dataset.py
```

By default, the project looks for the dataset in:

```text
.kagglehub-cache/datasets/programmer3/iot-network-traffic-dataset/versions/1/Iot_Network_data.csv
```

If the file already exists somewhere else in the project, the loader can still discover it automatically.

## Running the Project

Start the FastAPI server from the repository root:

```bash
python -m uvicorn wireless_network_congestion_project.main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

## Development Mode

If you want auto-reload while editing:

```bash
UVICORN_RELOAD=1 python wireless_network_congestion_project/main.py
```

## How to Use

1. Open the web app in your browser.
2. Run the training pipeline from the dashboard or call the training API.
3. Wait for the model, metrics, and plots to be generated.
4. Enter prediction values for packet rate, bandwidth, latency, and packet loss.
5. View the predicted congestion level.

## API Endpoints

- `GET /api/summary` returns dataset details, saved metrics, and plot paths
- `POST /api/train` runs the full training pipeline
- `POST /api/predict` returns a congestion prediction from input values

## Output Files

After training, generated files are stored in:

- `wireless_network_congestion_project/artifacts/`
- `wireless_network_congestion_project/plots/`

These include:

- `best_model.joblib`
- `model_metrics.json`
- plot images for EDA and congestion trends

## Notes

- Run training at least once before using prediction if no saved model exists.
- The first training run can take some time because plots and model artifacts are generated.
- Python 3.11 or newer is recommended for best compatibility with the pinned dependencies.
