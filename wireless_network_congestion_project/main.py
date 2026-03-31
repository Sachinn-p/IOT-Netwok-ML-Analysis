import asyncio
import json
import os
import platform
import re
import subprocess
import socket
import time
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
SYSTEM_SAMPLE_SECONDS = 1.0
LATENCY_PROBE_TARGETS = [("1.1.1.1", 53), ("8.8.8.8", 53)]


class PredictionRequest(BaseModel):
    packet_rate: float = Field(..., ge=0)
    bandwidth: float = Field(..., ge=0)
    latency: float = Field(..., ge=0)
    packet_loss: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    congestion_level: str
    network_load: float


class SystemTelemetryResponse(BaseModel):
    packet_rate: float
    bandwidth: float
    latency: float
    packet_loss: float
    sample_seconds: float
    sampled_at: str
    measurement_note: str


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


def _asset_version() -> int:
    asset_paths = [STATIC_DIR / "js" / "app.js", STATIC_DIR / "css" / "styles.css"]
    mtimes = [int(path.stat().st_mtime) for path in asset_paths if path.exists()]
    return max(mtimes, default=int(time.time()))


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


def _empty_network_counters() -> Dict[str, int]:
    return {
        "bytes_recv": 0,
        "packets_recv": 0,
        "errors_recv": 0,
        "drops_recv": 0,
        "bytes_sent": 0,
        "packets_sent": 0,
        "errors_sent": 0,
        "drops_sent": 0,
    }


def _read_linux_network_counters() -> Dict[str, int]:
    counters = _empty_network_counters()
    proc_net_dev = Path("/proc/net/dev")
    if not proc_net_dev.exists():
        raise FileNotFoundError("Live system metrics require /proc/net/dev.")

    with proc_net_dev.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()[2:]

    for line in lines:
        if ":" not in line:
            continue

        interface, values = line.split(":", maxsplit=1)
        interface = interface.strip()
        if interface == "lo":
            continue

        fields = values.split()
        if len(fields) < 12:
            continue

        counters["bytes_recv"] += int(fields[0])
        counters["packets_recv"] += int(fields[1])
        counters["errors_recv"] += int(fields[2])
        counters["drops_recv"] += int(fields[3])
        counters["bytes_sent"] += int(fields[8])
        counters["packets_sent"] += int(fields[9])
        counters["errors_sent"] += int(fields[10])
        counters["drops_sent"] += int(fields[11])

    return counters


def _read_windows_powershell_counters() -> Dict[str, int]:
    command = (
        "$adapters = Get-NetAdapterStatistics; "
        "if (-not $adapters) { throw 'No Windows adapters found.' }; "
        "$summary = [pscustomobject]@{"
        "ReceivedBytes = (($adapters | Measure-Object ReceivedBytes -Sum).Sum); "
        "SentBytes = (($adapters | Measure-Object SentBytes -Sum).Sum); "
        "ReceivedPackets = ((($adapters | Measure-Object ReceivedUnicastPackets -Sum).Sum) + (($adapters | Measure-Object ReceivedNonUnicastPackets -Sum).Sum)); "
        "SentPackets = ((($adapters | Measure-Object SentUnicastPackets -Sum).Sum) + (($adapters | Measure-Object SentNonUnicastPackets -Sum).Sum)); "
        "ReceivedDiscards = (($adapters | Measure-Object ReceivedDiscardedPackets -Sum).Sum); "
        "SentDiscards = (($adapters | Measure-Object OutboundDiscardedPackets -Sum).Sum); "
        "ReceivedErrors = (($adapters | Measure-Object ReceivedPacketErrors -Sum).Sum); "
        "SentErrors = (($adapters | Measure-Object OutboundPacketErrors -Sum).Sum)"
        "}; "
        "$summary | ConvertTo-Json -Compress"
    )

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=8,
    )
    payload = json.loads(result.stdout.strip())

    return {
        "bytes_recv": int(payload.get("ReceivedBytes", 0) or 0),
        "packets_recv": int(payload.get("ReceivedPackets", 0) or 0),
        "errors_recv": int(payload.get("ReceivedErrors", 0) or 0),
        "drops_recv": int(payload.get("ReceivedDiscards", 0) or 0),
        "bytes_sent": int(payload.get("SentBytes", 0) or 0),
        "packets_sent": int(payload.get("SentPackets", 0) or 0),
        "errors_sent": int(payload.get("SentErrors", 0) or 0),
        "drops_sent": int(payload.get("SentDiscards", 0) or 0),
    }


def _parse_windows_netstat_output(output: str) -> Dict[str, int]:
    counters = _empty_network_counters()
    metric_map = {
        "bytes": ("bytes_recv", "bytes_sent"),
        "unicast packets": ("packets_recv", "packets_sent"),
        "non-unicast packets": ("packets_recv", "packets_sent"),
        "discards": ("drops_recv", "drops_sent"),
        "errors": ("errors_recv", "errors_sent"),
    }

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized = " ".join(line.lower().split())
        for label, (recv_key, sent_key) in metric_map.items():
            if not normalized.startswith(label):
                continue

            values = re.findall(r"\d[\d,]*", line)
            if len(values) < 2:
                break

            counters[recv_key] += int(values[0].replace(",", ""))
            counters[sent_key] += int(values[1].replace(",", ""))
            break

    if counters["bytes_recv"] == 0 and counters["bytes_sent"] == 0:
        raise ValueError("Unable to parse network counters from 'netstat -e' output.")

    return counters


def _read_windows_network_counters() -> Dict[str, int]:
    try:
        return _read_windows_powershell_counters()
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass

    try:
        result = subprocess.run(
            ["netstat", "-e"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Live system metrics require the 'netstat' command on Windows."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Unable to read Windows network counters: {exc.stderr.strip() or exc}"
        ) from exc

    return _parse_windows_netstat_output(result.stdout)


def _read_network_counters() -> Dict[str, int]:
    system_name = platform.system().lower()
    if system_name == "windows":
        return _read_windows_network_counters()
    if system_name == "linux":
        return _read_linux_network_counters()

    proc_net_dev = Path("/proc/net/dev")
    if proc_net_dev.exists():
        return _read_linux_network_counters()

    raise RuntimeError(
        f"Unsupported operating system for live system metrics: {platform.system()}."
    )


def _measure_latency_ms(timeout: float = 1.0) -> float:
    for host, port in LATENCY_PROBE_TARGETS:
        start_time = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return max((time.perf_counter() - start_time) * 1000.0, 0.0)
        except OSError:
            continue

    return 0.0


def _collect_system_telemetry(
    sample_seconds: float = SYSTEM_SAMPLE_SECONDS,
) -> Dict[str, Any]:
    start_counters = _read_network_counters()
    time.sleep(sample_seconds)
    end_counters = _read_network_counters()

    packet_delta = max(
        (end_counters["packets_recv"] - start_counters["packets_recv"])
        + (end_counters["packets_sent"] - start_counters["packets_sent"]),
        0,
    )
    byte_delta = max(
        (end_counters["bytes_recv"] - start_counters["bytes_recv"])
        + (end_counters["bytes_sent"] - start_counters["bytes_sent"]),
        0,
    )
    loss_delta = max(
        (end_counters["errors_recv"] - start_counters["errors_recv"])
        + (end_counters["drops_recv"] - start_counters["drops_recv"])
        + (end_counters["errors_sent"] - start_counters["errors_sent"])
        + (end_counters["drops_sent"] - start_counters["drops_sent"]),
        0,
    )

    total_packets = max(packet_delta, 1)

    return {
        "packet_rate": round(packet_delta / sample_seconds, 4),
        "bandwidth": round((byte_delta * 8.0) / sample_seconds / 1_000_000.0, 4),
        "latency": round(_measure_latency_ms(), 4),
        "packet_loss": round((loss_delta / total_packets) * 100.0, 4),
        "sample_seconds": sample_seconds,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "measurement_note": (
            "Packet rate and bandwidth come from a short local network sample. "
            "Packet loss is estimated from interface drop/error counters."
        ),
    }


def _serializable_metrics(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        model_name: {
            "accuracy": float(metrics["accuracy"]),
            "train_accuracy": float(metrics["train_accuracy"]),
            "cv_accuracy_mean": float(metrics["cv_accuracy_mean"]),
            "cv_accuracy_std": float(metrics["cv_accuracy_std"]),
            "generalization_gap": float(metrics["generalization_gap"]),
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
        }
        for model_name, metrics in results.items()
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "asset_version": _asset_version()},
    )


@app.get("/api/summary")
async def summary() -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _dataset_summary(),
        "best_model": _read_best_model_name(),
        "model_metrics": _read_existing_metrics(),
        "plots": _plot_urls(),
    }


@app.get("/api/system-data", response_model=SystemTelemetryResponse)
async def current_system_data() -> SystemTelemetryResponse:
    try:
        telemetry = await asyncio.to_thread(_collect_system_telemetry)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"System data collection failed: {exc}"
        ) from exc

    return SystemTelemetryResponse(**telemetry)


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
