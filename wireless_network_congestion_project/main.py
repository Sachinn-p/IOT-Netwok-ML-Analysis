import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

try:
    from .data_analysis import (
        clean_data,
        engineer_features,
        normalize_numeric_features,
        run_eda,
    )
    from .dataset_loader import (
        identify_important_features,
        load_dataset,
        resolve_dataset_path,
    )
    from .predict import predict_congestion
    from .train_model import train_and_select_model
except ImportError:
    from data_analysis import (
        clean_data,
        engineer_features,
        normalize_numeric_features,
        run_eda,
    )
    from dataset_loader import (
        identify_important_features,
        load_dataset,
        resolve_dataset_path,
    )
    from predict import predict_congestion
    from train_model import train_and_select_model


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
PLOT_DIR = APP_DIR / "plots"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

for directory in [ARTIFACT_DIR, PLOT_DIR, STATIC_DIR, TEMPLATES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Wireless Network Congestion Predictor",
    description="FastAPI app for congestion analysis and prediction using IoT traffic data.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/plots", StaticFiles(directory=PLOT_DIR), name="plots")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class PredictionRequest(BaseModel):
    packet_rate: float = Field(..., gt=0)
    bandwidth: float = Field(..., gt=0)
    latency: float = Field(..., ge=0)
    packet_loss: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    congestion_level: str
    network_load: float


def _read_existing_metrics() -> Dict[str, Any]:
    metrics_path = ARTIFACT_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _read_best_model_name() -> str:
    model_path = ARTIFACT_DIR / "best_model.joblib"
    if not model_path.exists():
        return "Not trained yet"
    model_bundle = joblib.load(model_path)
    return str(model_bundle.get("model_name", "Unknown"))


def _plot_urls() -> Dict[str, str]:
    return {
        "correlation_heatmap": "/plots/correlation_heatmap.png",
        "bandwidth_vs_packet_rate": "/plots/bandwidth_vs_packet_rate.png",
        "congestion_distribution": "/plots/congestion_distribution.png",
        "congestion_trend": "/plots/congestion_trend.png",
    }


def _dataset_summary() -> Dict[str, Any]:
    dataset_path = resolve_dataset_path()
    df = load_dataset(str(dataset_path))

    return {
        "dataset_path": str(dataset_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "important_features": identify_important_features(df),
    }


def _serializable_metrics(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        model_name: {
            "accuracy": float(metrics["accuracy"]),
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
        }
        for model_name, metrics in results.items()
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/summary")
async def summary() -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _dataset_summary(),
        "best_model": _read_best_model_name(),
        "model_metrics": _read_existing_metrics(),
        "plots": _plot_urls(),
    }


@app.post("/api/train")
async def train_pipeline() -> Dict[str, Any]:
    raw_df = load_dataset(str(resolve_dataset_path()))
    cleaned_df, cleaning_report = clean_data(raw_df)
    featured_df = engineer_features(cleaned_df)
    normalized_df, _ = normalize_numeric_features(
        featured_df, exclude_columns=["congestion_level"]
    )

    run_eda(featured_df, PLOT_DIR)
    results, best_model_name, model_path = train_and_select_model(
        featured_df, ARTIFACT_DIR
    )

    return {
        "message": "Pipeline completed successfully.",
        "best_model": best_model_name,
        "model_path": str(model_path),
        "cleaning_report": cleaning_report,
        "normalized_shape": {
            "rows": int(normalized_df.shape[0]),
            "columns": int(normalized_df.shape[1]),
        },
        "congestion_distribution": {
            str(key): int(value)
            for key, value in featured_df["congestion_level"].value_counts().items()
        },
        "model_metrics": _serializable_metrics(results),
        "plots": _plot_urls(),
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        label = predict_congestion(
            packet_rate=payload.packet_rate,
            bandwidth=payload.bandwidth,
            latency=payload.latency,
            packet_loss=payload.packet_loss,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    return PredictionResponse(
        congestion_level=label,
        network_load=payload.packet_rate * payload.bandwidth,
    )


if __name__ == "__main__":
    import uvicorn

    reload_enabled = os.getenv("UVICORN_RELOAD", "0") == "1"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=reload_enabled,
        # Prevent training output writes from triggering reload restarts.
        reload_excludes=[
            "wireless_network_congestion_project/plots/*",
            "wireless_network_congestion_project/artifacts/*",
            "*.joblib",
            "*.png",
            "*.json",
        ],
    )
