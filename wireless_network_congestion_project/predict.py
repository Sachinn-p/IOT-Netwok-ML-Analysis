from pathlib import Path
from typing import Optional

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "best_model.joblib"


def _build_input_frame(
    packet_rate: float,
    bandwidth: float,
    latency: float,
    packet_loss: float,
) -> pd.DataFrame:
    network_load = float(packet_rate) * float(bandwidth)
    return pd.DataFrame(
        [
            {
                "packet_rate": float(packet_rate),
                "allocated_bandwidth": float(bandwidth),
                "latency": float(latency),
                "packet_loss_rate": float(packet_loss),
                "network_load": network_load,
            }
        ]
    )


def predict_congestion(
    packet_rate: float,
    bandwidth: float,
    latency: float,
    packet_loss: float,
    model_path: Optional[str] = None,
) -> str:
    resolved_model_path = (
        Path(model_path).resolve() if model_path else DEFAULT_MODEL_PATH
    )
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {resolved_model_path}. Run train_model.py or main.py first."
        )

    model_bundle = joblib.load(resolved_model_path)
    model = model_bundle["model"]
    features = model_bundle["features"]

    input_df = _build_input_frame(packet_rate, bandwidth, latency, packet_loss)
    predicted_label = model.predict(input_df[features])[0]
    return str(predicted_label)
